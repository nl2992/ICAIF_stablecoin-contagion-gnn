"""
Reviewer-grade polish on the benchmark outputs:

1. LIFT table — PR-AUC minus base rate (PR-AUC rises mechanically with the base rate
   across horizons, so absolute PR-AUC overstates skill). ROC-AUC (base-rate-free) added.
2. ROBUST verdict — recomputes the GNN-vs-XGB margin EXCLUDING degenerate folds
   (n_positive < 5, e.g. USDT_2018 has a single positive), so the PASS is not an artifact
   of one tiny fold.

Writes results/eval/lift_table.csv and results/eval/robust_verdict.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = [30, 60, 240, 1440]
GNN = ["graphsage", "gat"]
MIN_POS = 5


def lift_table() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        p = Path(f"results/ladder/pooled_results_h{h}.csv")
        if not p.exists():
            continue
        df = pd.read_csv(p, index_col=0)
        base = float(df["positive_rate"].iloc[0])
        for m in df.index:
            rows.append({
                "horizon_min": h, "model": m, "base_rate": round(base, 4),
                "pr_auc": round(float(df.loc[m, "pr_auc"]), 4),
                "pr_auc_lift": round(float(df.loc[m, "pr_auc"]) - base, 4),
                "roc_auc": round(float(df.loc[m, "roc_auc"]), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv("results/eval/lift_table.csv", index=False)
    return out


def robust_verdict() -> dict:
    df = pd.read_csv("results/eval/loeo_h1440.csv")
    df = df[~df["held_cluster"].isin(["MEAN"])].copy()
    df["n_pos"] = (df["pos_rate"].astype(float) * df["n_test"].astype(float)).round()
    kept = df[df["n_pos"] >= MIN_POS]
    dropped = df[df["n_pos"] < MIN_POS]["held_cluster"].tolist()
    verdict = {"protocol": "leave-one-cluster-out @ h=1440",
               "folds_kept": kept["held_cluster"].tolist(),
               "folds_dropped_low_power": dropped, "min_positives": MIN_POS}
    for g in GNN:
        margin = (kept[g] - kept["xgboost"]).astype(float)
        lift = (kept[g] - kept["pos_rate"].astype(float))
        verdict[g] = {
            "mean_margin_vs_xgb": round(float(margin.mean()), 4),
            "folds_win_ge_0.05": int((margin >= 0.05).sum()),
            "n_folds": int(len(margin)),
            "mean_lift_over_base": round(float(lift.mean()), 4),
            "beats_xgb_on_average": bool(margin.mean() > 0),
        }
    Path("results/eval/robust_verdict.json").write_text(json.dumps(verdict, indent=2))
    return verdict


if __name__ == "__main__":
    lt = lift_table()
    print("=== LIFT TABLE (PR-AUC over base rate; ROC-AUC base-rate-free) ===")
    print(lt[lt["horizon_min"] == 1440].to_string(index=False))
    print("\n=== ROBUST VERDICT (degenerate folds dropped) ===")
    print(json.dumps(robust_verdict(), indent=2))
