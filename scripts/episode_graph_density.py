"""Graph density vs GAT margin analysis.

For each LOCO episode, compute mean absolute pairwise Pearson correlation of
price deviation series across all node pairs (graph density proxy), then
correlate with the GAT-vs-XGBoost PR-AUC margin from LOCO results.

The hypothesis: GAT's advantage concentrates in structurally connected episodes
where cross-coin correlation is high; in idiosyncratic collapses it adds noise.

Outputs:
  results/eval/episode_density_vs_margin.csv
  results/eval/graph_density_summary.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parents[1]
GRAPH_DIR = ROOT / "data/processed/graphs"
LOCO_CSV = ROOT / "results/eval/loco_stability_comparison_h1440.csv"
OUT_DIR = ROOT / "results/eval"

# PKL filename → LOCO episode name mapping
PKL_TO_LOCO = {
    "UST_Terra":    "Terra_2022",
    "USDC_SVB":     "SVB_2023",
    "USDT_May2022": "USDT_2018",   # closest match
    "BUSD_winddown":"BUSD_2023",
    "DAI_FTX":      "FTX_2022",
    "FRAX_SVB":     None,           # not in LOCO
}

# Crisis window: steps around the event peak to use for correlation
# dev_bps_1m is 1-minute data; use ±48h = 2880 steps centred on middle of series
WINDOW_HALF = 2880


def graph_density(dev_dict: dict) -> float:
    """Mean absolute Pearson correlation across all node pairs in crisis window."""
    nodes = [n for n, v in dev_dict.items() if v is not None and len(v) > 0]
    if len(nodes) < 2:
        return float("nan")

    series = []
    for n in nodes:
        arr = np.asarray(dev_dict[n], dtype=float)
        mid = len(arr) // 2
        lo = max(0, mid - WINDOW_HALF)
        hi = min(len(arr), mid + WINDOW_HALF)
        series.append(arr[lo:hi])

    # Align lengths to shortest
    min_len = min(len(s) for s in series)
    mat = np.stack([s[:min_len] for s in series])  # (n_nodes, T)

    if min_len < 10:
        return float("nan")

    # Normalise each series to avoid vecdot overflow in large bps values
    mat = mat - mat.mean(axis=1, keepdims=True)
    std = mat.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    mat = mat / std

    # Pairwise absolute Pearson correlations
    abs_corrs = []
    for i, j in combinations(range(len(nodes)), 2):
        r, _ = stats.pearsonr(mat[i], mat[j])
        if np.isfinite(r):
            abs_corrs.append(abs(r))

    return float(np.mean(abs_corrs)) if abs_corrs else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load LOCO results
    loco = pd.read_csv(LOCO_CSV)
    # Extract: episode name → all_7 margin
    loco_map = {}
    for _, row in loco.iterrows():
        ep = str(row.get("fold", ""))
        margin = row.get("all_7_margin_vs_xgb", float("nan"))
        if ep and ep != "MEAN":
            loco_map[ep] = float(margin) if pd.notna(margin) else float("nan")

    print("LOCO margins loaded:", loco_map)

    rows = []
    for pkl_stem, loco_key in PKL_TO_LOCO.items():
        pkl_path = GRAPH_DIR / f"{pkl_stem}.pkl"
        if not pkl_path.exists():
            print(f"  SKIP {pkl_stem}: pkl not found")
            continue

        with open(pkl_path, "rb") as f:
            b = pickle.load(f)

        dev_dict = b.get("dev_bps_1m", {})
        nodes = b.get("active_node_strs", list(dev_dict.keys()))
        n_nodes = len(nodes)
        density = graph_density(dev_dict)

        margin = loco_map.get(loco_key, float("nan")) if loco_key else float("nan")

        episode_label = loco_key or pkl_stem
        print(f"  {episode_label:20s}  density={density:.3f}  GAT_margin={margin:+.3f}  n_nodes={n_nodes}")

        rows.append({
            "pkl_stem":      pkl_stem,
            "episode":       episode_label,
            "n_nodes":       n_nodes,
            "graph_density": round(density, 4),
            "gat_margin":    round(margin, 4) if not np.isnan(margin) else None,
            "in_loco":       loco_key is not None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "episode_density_vs_margin.csv", index=False)

    # Correlation among episodes that appear in LOCO and have valid density
    valid = df.dropna(subset=["graph_density", "gat_margin"])
    print(f"\nValid rows for correlation: {len(valid)}")

    summary = {"n_episodes_total": len(df), "n_valid_for_corr": len(valid)}

    if len(valid) >= 3:
        r_pearson, p_pearson = stats.pearsonr(valid["graph_density"], valid["gat_margin"])
        r_spearman, p_spearman = stats.spearmanr(valid["graph_density"], valid["gat_margin"])
        summary.update({
            "pearson_r":   round(float(r_pearson), 4),
            "pearson_p":   round(float(p_pearson), 4),
            "spearman_r":  round(float(r_spearman), 4),
            "spearman_p":  round(float(p_spearman), 4),
        })
        print(f"\nPearson  r={r_pearson:+.3f}  p={p_pearson:.3f}")
        print(f"Spearman r={r_spearman:+.3f}  p={p_spearman:.3f}")

        # Directional counts
        high_density = valid[valid["graph_density"] >= valid["graph_density"].median()]
        low_density  = valid[valid["graph_density"] <  valid["graph_density"].median()]
        summary["high_density_mean_margin"] = round(float(high_density["gat_margin"].mean()), 4)
        summary["low_density_mean_margin"]  = round(float(low_density["gat_margin"].mean()), 4)
        print(f"High-density mean GAT margin: {high_density['gat_margin'].mean():+.3f}")
        print(f"Low-density  mean GAT margin: {low_density['gat_margin'].mean():+.3f}")
    else:
        print("Too few valid rows for correlation — report descriptively")

    summary["episodes"] = df.to_dict(orient="records")
    (OUT_DIR / "graph_density_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT_DIR}/episode_density_vs_margin.csv")
    print(f"Saved: {OUT_DIR}/graph_density_summary.json")


if __name__ == "__main__":
    main()
