"""
Hub ranking artifact — the export that drives Repo 2 (ABM).

Score for each node:
    hub_score = α × norm(gnn_mask_sum) + (1−α) × norm(betweenness × propagator_label)

where norm() maps to [0, 1] within the node set.
α = 0.5 by default (equal weight; tested in sensitivity).

Output is a versioned CSV + JSON sidecar:
    exports/hub_ranking_v{VERSION}_{episode_tag}.csv
    exports/hub_ranking_v{VERSION}_{episode_tag}.json  (schema)

Schema is defined in src/scgnn/export/schema.py and versioned independently.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.train.ensemble import compute_hub_stability

SCHEMA_VERSION = "1"
_ALPHA = 0.5    # weight between GNN-mask and betweenness


def _normalize(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else pd.Series(np.zeros(len(s)), index=s.index)


def compute_hub_scores(
    registry: NodeRegistry,
    gnn_mask_sums: Dict[str, float],          # {node_str: total GNNExplainer mask weight}
    G: nx.DiGraph,                             # snapshot graph for centrality
    propagator_labels: Dict[str, int],         # {node_str: 0|1}
    alpha: float = _ALPHA,
) -> pd.DataFrame:
    """
    Compute the composite hub score for every node in the registry.

    Returns DataFrame with columns:
        node, asset, venue, fee_tier,
        gnn_mask_sum, betweenness, propagator_label,
        norm_gnn, norm_betweenness_x_prop,
        hub_score
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
            "betweenness_x_prop": bc * (prop if prop else 0.1),  # 0.1 floor avoids zero-out
        })

    df = pd.DataFrame(rows)
    df["norm_gnn"] = _normalize(df["gnn_mask_sum"])
    df["norm_bc_x_prop"] = _normalize(df["betweenness_x_prop"])
    df["hub_score"] = alpha * df["norm_gnn"] + (1 - alpha) * df["norm_bc_x_prop"]
    df = df.sort_values("hub_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def add_confidence_intervals(
    df: pd.DataFrame,
    hub_scores_per_seed: List[Dict[str, float]],
) -> pd.DataFrame:
    """
    Attach per-node 95% CI from multi-seed hub scores to the ranking DataFrame.
    """
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


def split_real_vs_synthetic(
    df: pd.DataFrame,
    episode_tag: str,
) -> pd.DataFrame:
    """Tag every row with its episode source."""
    df = df.copy()
    df["episode_tag"] = episode_tag
    df["is_real"] = not episode_tag.startswith("synthetic")
    return df


def save_hub_ranking(
    df: pd.DataFrame,
    episode_tag: str,
    out_dir: Path = Path("exports"),
    version: str = SCHEMA_VERSION,
    stability: Optional[dict] = None,
) -> Tuple[Path, Path]:
    """
    Write versioned CSV + JSON sidecar.
    Returns (csv_path, json_path).
    """
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
        "top5_hubs": df.head(5)[["node", "hub_score", "rank"]].to_dict(orient="records"),
    }
    if stability:
        sidecar["hub_stability"] = stability

    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Hub ranking saved: {csv_path}")
    return csv_path, json_path


def build_all_rankings(
    registry: NodeRegistry,
    episode_results: Dict[str, dict],    # {episode_name: {gnn_mask_sums, G, propagator_labels, hub_scores_per_seed}}
    out_dir: Path = Path("exports"),
) -> Dict[str, pd.DataFrame]:
    """
    Build separate hub rankings for each episode (real) and one aggregated
    synthetic ranking.  Returns {episode_name: DataFrame}.
    """
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
        df = split_real_vs_synthetic(df, episode_tag)
        save_hub_ranking(df, episode_tag, out_dir, stability=stability)
        all_rankings[episode_name] = df

    return all_rankings
