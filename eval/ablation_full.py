"""
Plan A — Full 4-condition component ablation with LOEO folds and 95% CIs.

Conditions:
  1. node-only tabular  : XGBoost  (no graph, no message passing)
  2. GAT, no edges      : ablate_edges=True  (node features + MLP, no topology)
  3. GAT, no node feat  : ablate_node_features=True  (topology only, zeroed features)
  4. full GAT           : node features + directed lead-lag graph edges

Evaluation: leave-one-cluster-out (LOCO) at h=1440 min, 5 seeds.

Outputs:
  results/eval/ablation_full.csv    — all conditions × folds × seeds
  results/eval/ablation_summary.csv — mean ± std per condition
  results/eval/ablation_ci.csv      — mean + 95 % CI per condition (bootstrap)

Usage:
  python eval/ablation_full.py [--seeds 5] [--horizon 1440]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.classical import make_xgboost  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402


CLUSTERS = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022", "BUSD_winddown": "BUSD_2023",
    "USDT_Oct2018": "USDT_2018",
}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple:
    """Return (mean, lower_95, upper_95) via percentile bootstrap."""
    rng = np.random.default_rng(0)
    boots = [rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), lo, hi


def run_one_fold(held_cluster: str, all_eps: list, feat_names: list,
                 horizon: int, seeds: list) -> list:
    """Run all 4 conditions for one LOCO fold over multiple seeds."""
    held = [e for e in all_eps if cluster_of(e) == held_cluster]
    train = [e for e in all_eps if cluster_of(e) != held_cluster]
    Xtr, ytr, _ = tabular_from_episodes(train, horizon, feat_names)
    Xte, yte, _ = tabular_from_episodes(held, horizon, feat_names)
    if yte.sum() == 0:
        print(f"  [skip] {held_cluster}: no positives in test set")
        return []

    rows = []
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)

    # ---- Condition 1: XGBoost (tabular, no graph) ----
    m = make_xgboost(scale_pos_weight=spw)
    m.fit(Xtr, ytr, verbose=False)
    pr = full_report(yte, m.predict_proba(Xte)[:, 1])["pr_auc"]
    rows.append({"held_cluster": held_cluster, "condition": "1_xgboost_no_graph",
                 "seed": "det", "pr_auc": round(pr, 5)})
    print(f"    cond1 xgboost: pr_auc={pr:.4f}")

    # ---- Conditions 2–4: GAT variants (seeded) ----
    for seed in seeds:
        for cond_label, ablate_e, ablate_n in [
            ("2_gat_no_edges",    True,  False),
            ("3_gat_no_node_feat",False, True),
            ("4_gat_full",        False, False),
        ]:
            tr = GNNContagionTrainer(
                kind="gat", horizon=horizon, seed=seed,
                epochs=80, patience=10,
                ablate_edges=ablate_e,
                ablate_node_features=ablate_n,
            )
            tr.fit(train, None)
            p = tr.predict_episodes(held)
            pr = full_report(yte, p)["pr_auc"]
            rows.append({"held_cluster": held_cluster, "condition": cond_label,
                         "seed": seed, "pr_auc": round(pr, 5)})
        print(f"    seed={seed} done")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=1440)
    args = ap.parse_args()

    feat_names = load_feature_names()
    all_eps = list_episodes()
    clusters = sorted(set(cluster_of(e) for e in all_eps))
    seeds = list(range(args.seeds))

    Path("results/eval").mkdir(parents=True, exist_ok=True)
    all_rows = []

    for clus in clusters:
        print(f"\n=== LOCO fold: hold out {clus} ===")
        rows = run_one_fold(clus, all_eps, feat_names, args.horizon, seeds)
        all_rows.extend(rows)

    if not all_rows:
        print("No results — all folds had zero positives.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv("results/eval/ablation_full.csv", index=False)
    print("\n=== Saved: results/eval/ablation_full.csv ===")

    # ---- Summary: mean ± std per condition (across folds, across seeds) ----
    valid = df[df["pr_auc"].notna()]
    summary_rows = []
    for cond, grp in valid.groupby("condition"):
        vals = grp["pr_auc"].values
        mean, lo, hi = bootstrap_ci(vals)
        summary_rows.append({
            "condition": cond,
            "mean_pr_auc": round(mean, 4),
            "std_pr_auc": round(float(np.std(vals)), 4),
            "ci95_lo": round(lo, 4),
            "ci95_hi": round(hi, 4),
            "n_obs": len(vals),
        })
    summary = pd.DataFrame(summary_rows).set_index("condition")
    summary.to_csv("results/eval/ablation_summary.csv")
    print("\n=== 4-Condition Ablation (LOCO, h=%d) ===" % args.horizon)
    print(summary.to_string())

    # ---- Delta vs XGBoost baseline (graph contribution) ----
    xgb_mean = summary.loc["1_xgboost_no_graph", "mean_pr_auc"] if "1_xgboost_no_graph" in summary.index else float("nan")
    print("\n=== Deltas vs XGBoost baseline ===")
    for cond in summary.index:
        if cond == "1_xgboost_no_graph":
            continue
        delta = summary.loc[cond, "mean_pr_auc"] - xgb_mean
        print(f"  {cond:35s}: {delta:+.4f}")

    full_gat = summary.loc["4_gat_full", "mean_pr_auc"] if "4_gat_full" in summary.index else float("nan")
    no_edge = summary.loc["2_gat_no_edges", "mean_pr_auc"] if "2_gat_no_edges" in summary.index else float("nan")
    print(f"\n  GRAPH topology contribution (cond4 - cond2): {full_gat - no_edge:+.4f} PR-AUC")


if __name__ == "__main__":
    main()
