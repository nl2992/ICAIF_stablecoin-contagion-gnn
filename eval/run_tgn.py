"""
Plan E runner — TGN-lite vs static GAT comparison (LOCO, h=1440).

Usage:
  python eval/run_tgn.py [--seeds 5] [--horizon 1440]

Outputs:
  results/eval/tgn_vs_gat_h{H}.csv   per-fold PR-AUC for TGN-lite and static GAT
  results/eval/tgn_verdict_h{H}.json  finding: does temporal memory add value?
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

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402
from scgnn.models.temporal_gnn import TGNContagionTrainer  # noqa: E402

CLUSTERS = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022", "BUSD_winddown": "BUSD_2023",
    "USDT_Oct2018": "USDT_2018",
}


def run_fold(held_cluster: str, all_eps: list, feat_names: list,
             horizon: int, seeds: list) -> list:
    held = [e for e in all_eps if cluster_of(e) == held_cluster]
    train = [e for e in all_eps if cluster_of(e) != held_cluster]

    _, yte, _ = tabular_from_episodes(held, horizon, feat_names)
    if yte.sum() == 0:
        print(f"  [skip] {held_cluster}: no positives in test set")
        return []

    rows = []
    for seed in seeds:
        # Static GAT
        gat = GNNContagionTrainer(kind="gat", horizon=horizon, seed=seed,
                                   epochs=80, patience=10)
        gat.fit(train, None)
        p_gat = gat.predict_episodes(held)
        pr_gat = full_report(yte, p_gat)["pr_auc"]

        # TGN-lite
        tgn = TGNContagionTrainer(horizon=horizon, seed=seed,
                                   epochs=80, patience=10, memory_dim=32)
        tgn.fit(train, None)
        p_tgn = tgn.predict_episodes(held)
        pr_tgn = full_report(yte, p_tgn)["pr_auc"]

        rows.append({
            "held_cluster": held_cluster, "seed": seed,
            "gat_pr_auc": round(pr_gat, 5),
            "tgn_pr_auc": round(pr_tgn, 5),
            "delta_tgn_vs_gat": round(pr_tgn - pr_gat, 5),
        })
        print(f"  {held_cluster} seed={seed}: GAT={pr_gat:.4f}, TGN={pr_tgn:.4f}, delta={pr_tgn - pr_gat:+.4f}")
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
        print(f"\n=== LOCO fold: {clus} ===")
        all_rows.extend(run_fold(clus, all_eps, feat_names, args.horizon, seeds))

    if not all_rows:
        print("No results.")
        return

    df = pd.DataFrame(all_rows)
    out = f"results/eval/tgn_vs_gat_h{args.horizon}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    valid = df[~df["gat_pr_auc"].isna()]
    gat_mean = float(valid["gat_pr_auc"].mean())
    tgn_mean = float(valid["tgn_pr_auc"].mean())
    delta_mean = float(valid["delta_tgn_vs_gat"].mean())
    wins = int((valid["delta_tgn_vs_gat"] > 0).sum())
    n = len(valid)

    verdict = {
        "gat_mean_pr_auc": round(gat_mean, 4),
        "tgn_mean_pr_auc": round(tgn_mean, 4),
        "mean_delta_tgn_minus_gat": round(delta_mean, 4),
        "tgn_wins_folds_seeds": wins,
        "total_folds_seeds": n,
        "finding": (
            f"TGN-lite {'improves' if delta_mean > 0.005 else 'does not improve'} over static GAT "
            f"(mean delta = {delta_mean:+.4f}). "
            + ("Temporal memory contributes beyond the 6h rolling-edge graph."
               if delta_mean > 0.005 else
               "Rolling 6h edges already capture temporal dynamics; GRU memory is redundant.")
        ),
    }
    Path(f"results/eval/tgn_verdict_h{args.horizon}.json").write_text(json.dumps(verdict, indent=2))
    print("\n=== TGN-lite vs Static GAT ===")
    print(f"  GAT mean PR-AUC: {gat_mean:.4f}")
    print(f"  TGN mean PR-AUC: {tgn_mean:.4f}")
    print(f"  Delta:           {delta_mean:+.4f}")
    print(f"  Finding: {verdict['finding']}")


if __name__ == "__main__":
    main()
