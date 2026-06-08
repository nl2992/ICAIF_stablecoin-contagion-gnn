"""
Plan H — Multi-horizon PR-AUC decay curve for GAT vs XGBoost.

Computes LOCO PR-AUC at horizons 30, 60, 120, 360, 720, 1440 min for GAT and
XGBoost. Plots both curves with 1-std shaded bands and marks the earliest
horizon where GAT > XGBoost by >= 0.05.

Usage:
  python eval/lead_time_multi.py [--seeds 3] [--models gat,xgboost]

Outputs:
  results/eval/lead_time_loco_h{H}.csv          per-fold PR-AUC per horizon per model
  results/eval/lead_time_summary.csv             mean ± std per (model, horizon)
  results/figures/lead_time_decay.png            main figure (paper Fig 2)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402

HORIZONS = [30, 60, 240, 1440]   # must match labels.horizons_min in experiment.yaml

CLUSTERS = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022", "BUSD_winddown": "BUSD_2023",
    "USDT_Oct2018": "USDT_2018",
}


def evaluate_fold_horizon(held_cluster: str, all_eps: list, feat_names: list,
                           horizon: int, model_kind: str, seed: int) -> dict | None:
    """Return PR-AUC for one (fold, horizon, model, seed) combination."""
    held = [e for e in all_eps if cluster_of(e) == held_cluster]
    train = [e for e in all_eps if cluster_of(e) != held_cluster]
    Xtr, ytr, _ = tabular_from_episodes(train, horizon, feat_names)
    Xte, yte, _ = tabular_from_episodes(held, horizon, feat_names)

    if yte.sum() == 0:
        return None

    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    if model_kind == "xgboost":
        m = make_xgboost(scale_pos_weight=spw)
        m.fit(Xtr, ytr, verbose=False)
        p = m.predict_proba(Xte)[:, 1]
    elif model_kind in ("gat", "graphsage"):
        tr = GNNContagionTrainer(kind=model_kind, horizon=horizon, seed=seed,
                                  epochs=60, patience=8)
        tr.fit(train, None)
        p = tr.predict_episodes(held)
    else:
        raise ValueError(f"Unknown model: {model_kind}")

    pr = full_report(yte, p)["pr_auc"]
    return {
        "held_cluster": held_cluster, "horizon_min": horizon,
        "model": model_kind, "seed": seed, "pr_auc": round(pr, 5),
        "pos_rate": round(float(yte.mean()), 4),
    }


def plot_decay_curves(summary: pd.DataFrame, actionability_threshold: float,
                      out_path: Path) -> None:
    """Plot PR-AUC vs horizon with ±1 std bands and actionability marker."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"gat": "#d62728", "xgboost": "#1f77b4", "graphsage": "#ff7f0e"}
    markers = {"gat": "o", "xgboost": "s", "graphsage": "^"}

    for model_name, grp in summary.groupby("model"):
        grp = grp.sort_values("horizon_min")
        col = colors.get(model_name, "black")
        mk = markers.get(model_name, "D")
        ax.plot(grp["horizon_min"], grp["mean_pr_auc"],
                color=col, marker=mk, linewidth=2, label=model_name.upper(), markersize=6)
        ax.fill_between(
            grp["horizon_min"],
            grp["mean_pr_auc"] - grp["std_pr_auc"],
            grp["mean_pr_auc"] + grp["std_pr_auc"],
            alpha=0.15, color=col)

    if actionability_threshold is not None:
        ax.axvline(actionability_threshold, color="green", linestyle="--", alpha=0.7,
                   label=f"Actionability threshold ({actionability_threshold}min)")

    ax.set_xlabel("Prediction horizon (minutes)", fontsize=11)
    ax.set_ylabel("LOCO PR-AUC (mean ± 1 std)", fontsize=11)
    ax.set_title("PR-AUC vs Prediction Horizon — Graph vs Tabular", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([f"{h//60}h" if h >= 60 else f"{h}m" for h in HORIZONS])
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def find_actionability_threshold(summary: pd.DataFrame, min_gap: float = 0.05) -> int | None:
    """Find earliest horizon where GAT > XGBoost by >= min_gap."""
    if "gat" not in summary["model"].values or "xgboost" not in summary["model"].values:
        return None
    gat_s = summary[summary["model"] == "gat"].set_index("horizon_min")["mean_pr_auc"]
    xgb_s = summary[summary["model"] == "xgboost"].set_index("horizon_min")["mean_pr_auc"]
    for h in sorted(HORIZONS):
        if h in gat_s and h in xgb_s:
            if gat_s[h] - xgb_s[h] >= min_gap:
                return h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gat,xgboost")
    ap.add_argument("--horizons", default=None,
                    help="Comma-separated list of horizons (default: 30,60,120,360,720,1440)")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    model_list = [m.strip() for m in args.models.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")] if args.horizons else HORIZONS

    feat_names = load_feature_names()
    all_eps = list_episodes()
    clusters = sorted(set(cluster_of(e) for e in all_eps))

    Path("results/eval").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    all_rows = []
    for h in horizons:
        print(f"\n=== Horizon: {h} min ===")
        for model_kind in model_list:
            seeds = [0] if model_kind == "xgboost" else list(range(args.seeds))
            for clus in clusters:
                for seed in seeds:
                    result = evaluate_fold_horizon(clus, all_eps, feat_names, h, model_kind, seed)
                    if result is not None:
                        all_rows.append(result)
                        print(f"  {clus} | {model_kind} | seed={seed} | h={h}: PR-AUC={result['pr_auc']:.4f}")

    if not all_rows:
        print("No results collected.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv("results/eval/lead_time_loco_raw.csv", index=False)

    # Summary: mean ± std per (model, horizon) across folds and seeds
    summary = (df.groupby(["model", "horizon_min"])["pr_auc"]
               .agg(mean_pr_auc="mean", std_pr_auc="std", n_obs="count")
               .reset_index())
    summary["mean_pr_auc"] = summary["mean_pr_auc"].round(4)
    summary["std_pr_auc"] = summary["std_pr_auc"].round(4)
    summary.to_csv("results/eval/lead_time_summary.csv", index=False)
    print("\n=== Lead-Time Decay Summary ===")
    pivot = summary.pivot(index="horizon_min", columns="model", values="mean_pr_auc")
    print(pivot.round(4).to_string())

    # Actionability threshold
    at = find_actionability_threshold(summary)
    if at:
        print(f"\nActionability threshold: {at} min -- earliest horizon where GAT > XGBoost by >=0.05")
    else:
        print("\nNo actionability threshold found (GAT never exceeds XGBoost by >=0.05)")

    # Key finding
    gat_24h = summary[(summary["model"] == "gat") & (summary["horizon_min"] == 1440)]
    xgb_24h = summary[(summary["model"] == "xgboost") & (summary["horizon_min"] == 1440)]
    gat_30m = summary[(summary["model"] == "gat") & (summary["horizon_min"] == 30)]
    if len(gat_24h) > 0 and len(xgb_24h) > 0:
        gap = float(gat_24h["mean_pr_auc"]) - float(xgb_24h["mean_pr_auc"])
        print(f"GAT advantage at 24h: {gap:+.4f} PR-AUC")
    if len(gat_30m) > 0:
        print(f"GAT PR-AUC at 30min: {float(gat_30m['mean_pr_auc']):.4f} (near-chance expected)")

    plot_decay_curves(summary, at, Path("results/figures/lead_time_decay_multi.png"))


if __name__ == "__main__":
    main()
