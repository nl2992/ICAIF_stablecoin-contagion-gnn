"""
Hub ranking artifact — the export that drives Repo 2 (ABM).

=======================================================================
ON PROPAGATOR-LABEL CIRCULARITY
=======================================================================

The propagator_label[j] is computed from RAW PRICE DATA only:
    propagator_label[j] = 1  iff
        (a) j ≠ origin_node  (not the trigger)
        (b) j was NOT stressed before shock onset  (pre-existing mask)
        (c) j entered stress within 24h of the origin's onset
            (using the pre-registered threshold, not the model's output)

It does NOT use:
  - GNNExplainer mask output
  - Model predicted probabilities
  - Betweenness centrality

This means the hub score is NOT circular: the propagator label is an
empirical observation, and the GNN learns to predict it.  The hub
score then combines that predictive importance with structural centrality.

To make this explicit and testable, we provide TWO hub variants:

  1. hub_score_structural  = betweenness_centrality only (no model output)
  2. hub_score_full        = α·norm_gnn + (1−α)·norm_bc×prop  (full composite)

If hub_score_structural ≈ hub_score_full, the GNN adds nothing beyond
graph topology — report this honestly.

=======================================================================

Score formula (full variant):
    hub_score = α × norm(gnn_mask_sum) + (1−α) × norm(betweenness × propagator_label)
    α = 0.5 (tested in sensitivity; see exports/schema_v1.json)

Output: versioned CSV + JSON sidecar in exports/
Schema: exports/schema_v1.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.features.labels import _stress_indicator

SCHEMA_VERSION = "1"
_ALPHA = 0.5


# ------------------------------------------------------------------ propagator label


def compute_propagator_labels(
    peg_deviations: Dict[str, pd.Series],
    origin_node: str,
    thresholds_bps: Dict[str, float],
    sustained_min: int = 10,
    propagation_window_min: int = 1440,   # 24h
) -> Dict[str, int]:
    """
    Empirical propagator label — computed from raw price data only.

    Returns {node_str: 0|1} where 1 = node entered new stress
    within `propagation_window_min` of the origin's shock onset.

    Non-circular: purely observational, not derived from model output.
    """
    if origin_node not in peg_deviations:
        return {n: 0 for n in peg_deviations}

    origin_thr = thresholds_bps.get(origin_node, 25.0)
    origin_stress = _stress_indicator(peg_deviations[origin_node], origin_thr, sustained_min)
    onset_times = origin_stress[origin_stress == 1].index
    if len(onset_times) == 0:
        return {n: 0 for n in peg_deviations}

    origin_onset = onset_times[0]
    window_end = origin_onset + pd.Timedelta(minutes=propagation_window_min)

    labels: Dict[str, int] = {}
    for node, dev in peg_deviations.items():
        if node == origin_node:
            labels[node] = 0   # origin excluded
            continue

        thr = thresholds_bps.get(node, 25.0)
        # Pre-existing stress mask: was node stressed BEFORE origin onset?
        before_onset = dev[dev.index < origin_onset]
        pre_stressed = bool(_stress_indicator(before_onset, thr, sustained_min).sum() > 0)
        if pre_stressed:
            labels[node] = 0   # pre-existing stress → not a propagation event
            continue

        # Did node enter stress in (origin_onset, origin_onset + 24h]?
        after_onset = dev[(dev.index > origin_onset) & (dev.index <= window_end)]
        if after_onset.empty:
            labels[node] = 0
            continue
        stress_after = _stress_indicator(after_onset, thr, sustained_min)
        labels[node] = int(stress_after.sum() > 0)

    return labels


# ------------------------------------------------------------------ normalization


def _normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(np.zeros(len(s)), index=s.index)


# ------------------------------------------------------------------ hub score computation


def compute_hub_scores(
    registry: NodeRegistry,
    gnn_mask_sums: Dict[str, float],
    G: nx.DiGraph,
    propagator_labels: Dict[str, int],
    alpha: float = _ALPHA,
) -> pd.DataFrame:
    """
    Compute BOTH hub variants for every node:
      hub_score_structural : betweenness only  (no model output)
      hub_score_full       : α·norm_gnn + (1−α)·norm_bc×prop

    Use hub_score_structural vs hub_score_full comparison to assess
    whether the GNN adds anything beyond raw topology.
    """
    UG = G.to_undirected()
    betweenness = nx.betweenness_centrality(UG, weight="weight")

    rows = []
    for node in registry:
        ns = str(node)
        mask = gnn_mask_sums.get(ns, 0.0)
        bc = betweenness.get(ns, 0.0)
        prop = propagator_labels.get(ns, 0)
        rows.append({
            "node": ns,
            "asset": node.asset,
            "venue": node.venue,
            "fee_tier": node.fee_tier,
            "gnn_mask_sum": mask,
            "betweenness": bc,
            "propagator_label": prop,
            "betweenness_x_prop": bc * (prop if prop else 0.1),
        })

    df = pd.DataFrame(rows)
    df["norm_gnn"] = _normalize(df["gnn_mask_sum"])
    df["norm_bc"] = _normalize(df["betweenness"])
    df["norm_bc_x_prop"] = _normalize(df["betweenness_x_prop"])

    # Structural variant (centrality only — no model output)
    df["hub_score_structural"] = df["norm_bc"]

    # Full composite variant
    df["hub_score_full"] = alpha * df["norm_gnn"] + (1 - alpha) * df["norm_bc_x_prop"]

    # Primary export column (full) for backward compatibility
    df["hub_score"] = df["hub_score_full"]

    df = df.sort_values("hub_score", ascending=False).reset_index(drop=True)
    df["rank_full"] = df.index + 1
    df["rank_structural"] = df["hub_score_structural"].rank(ascending=False).astype(int)

    # Report whether GNN adds beyond topology
    rho, pval = spearmanr(df["hub_score_structural"], df["hub_score_full"])
    df.attrs["structural_vs_full_rho"] = float(rho)
    df.attrs["structural_vs_full_pval"] = float(pval)

    return df


def gnn_adds_beyond_topology(df: pd.DataFrame) -> bool:
    """
    True if GNN-full hub ranking differs meaningfully from structural-only.
    Returns True if Spearman ρ(structural, full) < 0.9.
    """
    rho = df.attrs.get("structural_vs_full_rho", 1.0)
    return rho < 0.9


# ------------------------------------------------------------------ confidence intervals


def add_confidence_intervals(
    df: pd.DataFrame,
    hub_scores_per_seed: List[Dict[str, float]],
) -> pd.DataFrame:
    """Attach per-node 95% CI from multi-seed hub scores."""
    all_nodes = df["node"].tolist()
    ci_lo, ci_hi, ci_std = [], [], []
    for node in all_nodes:
        seed_vals = np.array([hs.get(node, 0.0) for hs in hub_scores_per_seed])
        ci_lo.append(float(np.percentile(seed_vals, 2.5)))
        ci_hi.append(float(np.percentile(seed_vals, 97.5)))
        ci_std.append(float(seed_vals.std()))
    df = df.copy()
    df["ci_lo"] = ci_lo
    df["ci_hi"] = ci_hi
    df["ci_std"] = ci_std
    return df


# ------------------------------------------------------------------ save


def save_hub_ranking(
    df: pd.DataFrame,
    episode_tag: str,
    out_dir: Path = Path("exports"),
    version: str = SCHEMA_VERSION,
    stability: Optional[dict] = None,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = episode_tag.replace(" ", "_").replace("/", "-")
    csv_path = out_dir / f"hub_ranking_v{version}_{slug}.csv"
    json_path = out_dir / f"hub_ranking_v{version}_{slug}.json"

    df.to_csv(csv_path, index=False)

    sidecar = {
        "schema_version": version,
        "episode_tag": episode_tag,
        "n_nodes": len(df),
        "alpha": _ALPHA,
        "columns": list(df.columns),
        "structural_vs_full_rho": df.attrs.get("structural_vs_full_rho", None),
        "gnn_adds_beyond_topology": gnn_adds_beyond_topology(df),
        "top5_hubs_full": df.nsmallest(5, "rank_full")[["node", "hub_score_full", "rank_full"]].to_dict(orient="records"),
        "top5_hubs_structural": df.nsmallest(5, "rank_structural")[["node", "hub_score_structural", "rank_structural"]].to_dict(orient="records"),
    }
    if stability:
        sidecar["hub_stability"] = stability

    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Hub ranking saved: {csv_path}  (structural_vs_full ρ={sidecar['structural_vs_full_rho']:.3f})")
    return csv_path, json_path


def build_all_rankings(
    registry: NodeRegistry,
    episode_results: Dict[str, dict],
    out_dir: Path = Path("exports"),
) -> Dict[str, pd.DataFrame]:
    from scgnn.train.ensemble import compute_hub_stability
    all_rankings = {}
    for episode_name, data in episode_results.items():
        df = compute_hub_scores(
            registry=registry,
            gnn_mask_sums=data["gnn_mask_sums"],
            G=data["G"],
            propagator_labels=data["propagator_labels"],
        )
        if "hub_scores_per_seed" in data:
            df = add_confidence_intervals(df, data["hub_scores_per_seed"])
            stability = compute_hub_stability(data["hub_scores_per_seed"])
        else:
            stability = None
        is_real = data.get("is_real", True)
        episode_tag = episode_name if is_real else f"synthetic_{episode_name}"
        df["episode_tag"] = episode_tag
        df["is_real"] = is_real
        save_hub_ranking(df, episode_tag, out_dir, stability=stability)
        all_rankings[episode_name] = df
    return all_rankings
