"""
Generate all paper figures from results/ artefacts.

Figures produced:
  fig1_model_horizon_pr_auc.png   — model × horizon PR-AUC (Fig 5 analog)
  fig2_lead_time_decay.png        — accuracy vs horizon for best model
  fig3_hub_ranking_real.png       — top-10 hub scores with CI, real episodes
  fig4_hub_stability.png          — rank-correlation heatmap across seeds

Usage:
    python scripts/generate_figures.py [--out_dir results/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


MODELS = ["majority", "persistence", "logreg", "xgboost", "lstm", "graphsage", "gat"]
HORIZONS = [30, 60, 240, 1440]
HORIZON_LABELS = ["30 min", "1 h", "4 h", "24 h"]
MODEL_COLORS = {
    "majority":    "#aaaaaa",
    "persistence": "#888888",
    "logreg":      "#4393c3",
    "xgboost":     "#2166ac",
    "lstm":        "#f4a582",
    "graphsage":   "#d6604d",
    "gat":         "#b2182b",
}
MODEL_MARKERS = {
    "majority": "x", "persistence": "+", "logreg": "s",
    "xgboost": "D", "lstm": "^", "graphsage": "o", "gat": "*",
}


def load_results(results_dir: Path) -> dict:
    """Load per-model per-horizon JSON reports."""
    data = {}
    for h in HORIZONS:
        path = results_dir / f"reports_h{h}.json"
        if path.exists():
            with open(path) as f:
                data[h] = json.load(f)
    return data


def fig_model_horizon(results: dict, metric: str = "pr_auc", out_path: Path = None) -> None:
    """
    Model × horizon grid (Uniswap Fig 5 analog).
    Each model gets a line; x-axis = horizon; y-axis = metric.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        y_vals, x_vals = [], []
        for h in HORIZONS:
            if h not in results or model not in results[h]:
                continue
            val = results[h][model].get(metric, float("nan"))
            if not np.isnan(val):
                y_vals.append(val)
                x_vals.append(h)

        if not y_vals:
            continue

        color = MODEL_COLORS.get(model, "black")
        marker = MODEL_MARKERS.get(model, "o")
        lw = 2.0 if model in ("graphsage", "gat", "xgboost") else 1.2
        ax.plot(x_vals, y_vals, marker=marker, color=color, linewidth=lw,
                label=model.upper() if model in ("gat",) else model.capitalize(),
                markersize=7 if model in ("graphsage", "gat") else 5)

    ax.set_xscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Model performance by horizon — {metric.replace('_', ' ').upper()}")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def fig_hub_ranking(hub_csv: Path, out_path: Path = None, top_n: int = 10) -> None:
    """
    Top-N hub scores with 95% CI error bars (real episodes only).
    """
    if not hub_csv.exists():
        print(f"[WARN] Hub ranking not found: {hub_csv}")
        return

    df = pd.read_csv(hub_csv)
    if "is_real" in df.columns:
        df = df[df["is_real"]]
    df = df.sort_values("hub_score", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    y = range(len(df))
    colors = ["#b2182b" if row.get("propagator_label", 0) else "#4393c3"
              for _, row in df.iterrows()]

    ax.barh(list(y), df["hub_score"].values[::-1], color=colors[::-1], alpha=0.8)

    if "ci_lo" in df.columns and "ci_hi" in df.columns:
        xerr_lo = (df["hub_score"] - df["ci_lo"]).values[::-1]
        xerr_hi = (df["ci_hi"] - df["hub_score"]).values[::-1]
        ax.errorbar(df["hub_score"].values[::-1], list(y),
                    xerr=[xerr_lo, xerr_hi], fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["node"].values[::-1], fontsize=8)
    ax.set_xlabel("Hub score (composite)")
    ax.set_title("Contagion hub ranking — real episodes (with 95% CI)")
    legend_patches = [
        mpatches.Patch(color="#b2182b", label="Propagator"),
        mpatches.Patch(color="#4393c3", label="Non-propagator"),
    ]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/ladder")
    parser.add_argument("--out_dir", default="results/figures")
    parser.add_argument("--metric", default="pr_auc")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    results = load_results(results_dir)
    if results:
        fig_model_horizon(results, args.metric, out / f"fig1_model_horizon_{args.metric}.png")
    else:
        print("[WARN] No results found — run train/run_ladder.py first.")

    # Hub ranking figure
    hub_dir = Path("exports")
    for hub_csv in sorted(hub_dir.glob("hub_ranking_v1_*.csv")):
        ep_tag = hub_csv.stem.replace("hub_ranking_v1_", "")
        fig_hub_ranking(hub_csv, out / f"fig3_hub_ranking_{ep_tag}.png")

    print(f"\nAll figures written to {out}/")


if __name__ == "__main__":
    main()
