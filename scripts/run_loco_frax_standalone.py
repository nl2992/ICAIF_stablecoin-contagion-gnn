"""Run FRAX_SVB as a standalone 5th LOCO fold.

In the standard run_loco.py, FRAX_SVB is clustered with USDC_SVB into the
SVB_2023 cluster, so there is never a fold that holds out FRAX_SVB alone.
This script creates a custom cluster map that assigns FRAX_SVB its own cluster
(FRAX_SVB_standalone), then runs LOCO to get a margin estimate for FRAX_SVB.

FRAX_SVB has graph_density=0.5462 (highest of all episodes).
If its GAT margin is also high, the r(density, margin) → 0.9+ with n=5.

Usage:
  python scripts/run_loco_frax_standalone.py --model gat --horizon 1440

Outputs:
  results/eval/loco_frax_standalone_h{H}.csv
  results/eval/density_bootstrap_ci_n5.json   (updated CI with 5th point)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402
from scgnn.utils.seeds import set_all_seeds  # noqa: E402

FRAX_DENSITY = 0.5462

EPISODES_5 = {"USDC_SVB", "FRAX_SVB", "DAI_FTX", "UST_Terra", "USDT_May2022"}

# Custom cluster map: FRAX_SVB gets its own standalone cluster
CLUSTERS_FRAX_STANDALONE = {
    "USDC_SVB":      "SVB_2023",           # USDC is still in SVB cluster
    "FRAX_SVB":      "FRAX_SVB_standalone", # FRAX standalone — this is the new fold
    "UST_Terra":     "Terra_2022",
    "USDT_May2022":  "Terra_2022",
    "DAI_FTX":       "FTX_2022",
}

# Known density/margin pairs from episode_density_vs_margin.csv (n=4 baseline)
BASELINE_EPISODES = [
    {"episode": "Terra_2022", "density": 0.3602, "gat_margin": 0.0878},
    {"episode": "SVB_2023",   "density": 0.3946, "gat_margin": 0.0466},
    {"episode": "USDT_2018",  "density": 0.3491, "gat_margin": 0.0452},
    {"episode": "FTX_2022",   "density": 0.0802, "gat_margin": 0.0028},
]


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xd = x - x.mean(); yd = y - y.mean()
    denom = np.sqrt((xd**2).sum() * (yd**2).sum())
    return float(np.dot(xd, yd) / denom) if denom > 1e-12 else 0.0


def bootstrap_ci(density: np.ndarray, margin: np.ndarray,
                 n_boot: int = 100_000, seed: int = 2025) -> dict:
    rng = np.random.default_rng(seed)
    n = len(density)
    obs = pearson_r(density, margin)
    boot_r = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_r[i] = pearson_r(density[idx], margin[idx])
    return {
        "observed_r": round(obs, 4),
        "n_episodes": n,
        "ci_95_lo": round(float(np.percentile(boot_r, 2.5)), 4),
        "ci_95_hi": round(float(np.percentile(boot_r, 97.5)), 4),
        "ci_90_lo": round(float(np.percentile(boot_r, 5.0)), 4),
        "ci_90_hi": round(float(np.percentile(boot_r, 95.0)), 4),
        "boot_mean_r": round(float(boot_r.mean()), 4),
        "pct_positive": round(float(np.mean(boot_r > 0)), 4),
        "pct_above_0p5": round(float(np.mean(boot_r > 0.5)), 4),
    }


def run_frax_fold(model_kind: str, horizon: int, seed: int = 42) -> dict:
    """Run the FRAX_SVB standalone fold and return margin estimate."""
    all_eps = list_episodes()
    episodes = [e for e in all_eps if e in EPISODES_5]
    if "FRAX_SVB" not in episodes:
        raise RuntimeError(
            "FRAX_SVB episode not found in list_episodes(). "
            "Ensure data/processed/graphs/FRAX_SVB.pkl exists."
        )

    feat_names = load_feature_names()

    # Only the FRAX_SVB standalone fold
    held = ["FRAX_SVB"]
    train = [e for e in episodes if e != "FRAX_SVB"]

    print(f"Train episodes: {train}")
    print(f"Test  episodes: {held}")

    Xtr, ytr, _ = tabular_from_episodes(train, horizon, feat_names)
    Xte, yte, _ = tabular_from_episodes(held, horizon, feat_names)

    print(f"Train: {len(Xtr)} samples, {float(ytr.mean()):.3%} positive")
    print(f"Test:  {len(Xte)} samples, {float(yte.mean()):.3%} positive")

    if yte.sum() == 0:
        print("WARNING: FRAX_SVB test set has no positive labels — margin will be NaN")

    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    # XGBoost baseline
    m = make_xgboost(scale_pos_weight=spw)
    m.fit(Xtr, ytr, verbose=False)
    xgb_pr = (full_report(yte, m.predict_proba(Xte)[:, 1])["pr_auc"]
               if yte.sum() > 0 else float("nan"))
    print(f"XGBoost PR-AUC: {xgb_pr:.5f}")

    # GNN
    set_all_seeds(seed)
    tr = GNNContagionTrainer(kind=model_kind, horizon=horizon, seed=seed,
                              epochs=80, patience=10)
    tr.fit(train, None)
    p = tr.predict_episodes(held)
    gnn_pr = (full_report(yte, p)["pr_auc"]
               if yte.sum() > 0 else float("nan"))
    print(f"GAT PR-AUC:     {gnn_pr:.5f}")

    margin = gnn_pr - xgb_pr if not (np.isnan(gnn_pr) or np.isnan(xgb_pr)) else float("nan")
    print(f"Margin (GAT - XGB): {margin:+.5f}")

    return {
        "held_cluster": "FRAX_SVB_standalone",
        "episodes": "FRAX_SVB",
        "n_test": int(len(yte)),
        "pos_rate": round(float(yte.mean()), 4) if len(yte) > 0 else 0.0,
        "xgboost": round(xgb_pr, 5),
        model_kind: round(gnn_pr, 5),
        "margin_vs_xgb": round(margin, 5),
        "graph_density": FRAX_DENSITY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gat", choices=["gat", "graphsage"])
    parser.add_argument("--horizon", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT = _ROOT / "results/eval"
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"FRAX_SVB STANDALONE LOCO  (model={args.model}, H={args.horizon})")
    print(f"Graph density: {FRAX_DENSITY} (highest of all episodes)")
    print("=" * 60)

    row = run_frax_fold(args.model, args.horizon, args.seed)

    # Save standalone fold result
    df = pd.DataFrame([row])
    out_csv = OUT / f"loco_frax_standalone_h{args.horizon}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # If we have a valid margin, update bootstrap CI with n=5
    frax_margin = row["margin_vs_xgb"]
    if not np.isnan(frax_margin):
        density_n5 = np.array([e["density"] for e in BASELINE_EPISODES] + [FRAX_DENSITY])
        margin_n5  = np.array([e["gat_margin"] for e in BASELINE_EPISODES] + [frax_margin])
        ci5 = bootstrap_ci(density_n5, margin_n5)
        ci5["frax_svb_density"] = FRAX_DENSITY
        ci5["frax_svb_margin"] = frax_margin
        ci5["episodes"] = BASELINE_EPISODES + [
            {"episode": "FRAX_SVB", "density": FRAX_DENSITY, "gat_margin": frax_margin}
        ]
        out_json = OUT / "density_bootstrap_ci_n5.json"
        out_json.write_text(json.dumps(ci5, indent=2))
        print(f"Saved n=5 bootstrap CI: {out_json}")
        print(f"  r(n=5) = {ci5['observed_r']:.4f}  "
              f"95% CI [{ci5['ci_95_lo']:.3f}, {ci5['ci_95_hi']:.3f}]  "
              f"{ci5['pct_positive']*100:.1f}% positive resamples")
    else:
        print("\nFRAX_SVB margin is NaN — no positive labels in test set.")
        print("Cannot update density correlation. Check that FRAX_SVB.pkl has crisis labels.")


if __name__ == "__main__":
    main()
