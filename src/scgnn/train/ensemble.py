"""
Multi-seed ensemble training for GNNs (and any other PyTorch model).

Trains ≥5 independent runs with different seeds, then aggregates:
  - mean and 95% CI for every metric
  - per-seed prediction arrays (stored for hub-stability analysis)
  - Spearman rank-correlation of hub scores across seeds

Why 5 seeds?
  The GNN optimisation landscape is non-convex; a single run may hit a local
  optimum.  Five seeds is the minimum that makes a bootstrap CI meaningful
  while staying tractable.  We report the *interval*, not just the mean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from scgnn.utils.seeds import set_all_seeds


ENSEMBLE_SEEDS = [42, 137, 271, 503, 789]


def _ci_95(values: np.ndarray) -> tuple[float, float]:
    """Bootstrap 95% CI from a small array of metric values."""
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    return lo, hi


def run_ensemble(
    train_fn: Callable[[int], dict],     # seed → {metric: value, probs: np.ndarray, hub_scores: dict}
    seeds: List[int] = ENSEMBLE_SEEDS,
    out_dir: Optional[Path] = None,
    model_name: str = "graphsage",
    horizon_min: int = 60,
) -> dict:
    """
    Run train_fn independently for each seed, aggregate results.

    train_fn must return a dict with at minimum:
      {
        "pr_auc": float,
        "roc_auc": float,
        "weighted_f1": float,
        "probs": np.ndarray,          # per-sample predicted probabilities
        "hub_scores": dict[str, float],  # {node_str: importance_score}
      }

    Returns aggregated dict with mean, CI, and per-seed raw results.
    """
    per_seed: List[dict] = []
    for seed in seeds:
        set_all_seeds(seed)
        result = train_fn(seed)
        result["seed"] = seed
        per_seed.append(result)
        print(f"  seed={seed}: pr_auc={result.get('pr_auc', float('nan')):.4f}")

    # Aggregate metrics
    metrics = ["pr_auc", "roc_auc", "weighted_f1", "precision", "recall"]
    agg: dict = {"model": model_name, "horizon_min": horizon_min, "n_seeds": len(seeds)}
    for m in metrics:
        vals = np.array([r.get(m, float("nan")) for r in per_seed])
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        lo, hi = _ci_95(valid)
        agg[m] = float(valid.mean())
        agg[f"{m}_ci_lo"] = lo
        agg[f"{m}_ci_hi"] = hi
        agg[f"{m}_std"] = float(valid.std())

    # Ensemble probability (mean across seeds)
    if all("probs" in r for r in per_seed):
        probs_stack = np.stack([r["probs"] for r in per_seed], axis=0)
        agg["probs_mean"] = probs_stack.mean(axis=0)
        agg["probs_std"] = probs_stack.std(axis=0)

    # Hub stability: Spearman rank-correlation across seed pairs
    if all("hub_scores" in r for r in per_seed):
        hub_stability = compute_hub_stability([r["hub_scores"] for r in per_seed])
        agg["hub_rank_corr_mean"] = hub_stability["mean_rho"]
        agg["hub_rank_corr_min"] = hub_stability["min_rho"]

    agg["per_seed"] = per_seed

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Strip large arrays before JSON serialisation
        json_safe = {k: v for k, v in agg.items()
                     if k not in ("probs_mean", "probs_std", "per_seed")}
        with open(out_dir / f"ensemble_{model_name}_h{horizon_min}.json", "w") as f:
            json.dump(json_safe, f, indent=2)
        if "probs_mean" in agg:
            np.save(out_dir / f"probs_{model_name}_h{horizon_min}.npy", agg["probs_mean"])

    return agg


def compute_hub_stability(hub_scores_per_seed: List[Dict[str, float]]) -> dict:
    """
    Compute mean and min Spearman rank-correlation across all pairs of seeds.

    A hub that reshuffles every seed has low stability — not a reliable hub.
    Returns {"mean_rho": float, "min_rho": float, "pairwise": List[float]}.
    """
    from scipy.stats import spearmanr

    all_nodes = sorted(set().union(*[set(hs.keys()) for hs in hub_scores_per_seed]))
    if len(all_nodes) < 2 or len(hub_scores_per_seed) < 2:
        return {"mean_rho": float("nan"), "min_rho": float("nan"), "pairwise": []}

    # Build matrix: (n_seeds, n_nodes)
    matrix = np.array([
        [hs.get(node, 0.0) for node in all_nodes]
        for hs in hub_scores_per_seed
    ])

    pairwise = []
    n = len(hub_scores_per_seed)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(matrix[i], matrix[j])
            pairwise.append(float(rho))

    return {
        "mean_rho": float(np.mean(pairwise)) if pairwise else float("nan"),
        "min_rho": float(np.min(pairwise)) if pairwise else float("nan"),
        "pairwise": pairwise,
        "nodes": all_nodes,
    }


def format_metric_with_ci(mean: float, lo: float, hi: float, decimals: int = 3) -> str:
    """Format as '0.812 [0.798–0.826]' for paper tables."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} [{fmt.format(lo)}–{fmt.format(hi)}]"
