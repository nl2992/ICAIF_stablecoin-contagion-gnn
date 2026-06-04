"""
Multi-seed robustness for the headline GAT-vs-XGBoost comparison at h=1440.

Re-runs the leakage-safe held-out-SVB split AND the three positive-bearing LOEO folds
across N seeds, so the +PR-AUC margin of the GNN over XGBoost is reported with a
confidence interval rather than a single noisy point estimate.

Writes results/eval/multiseed_h1440.csv and _summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of, run_partition  # noqa: E402
from scgnn.data.dataset import list_episodes, load_feature_names  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402

H = 1440
SEEDS = [0, 1, 2, 3, 4]
MODELS = ["xgboost", "graphsage", "gat"]


def main():
    feat = load_feature_names()
    eps = list_episodes()
    # held-out SVB headline + the 3 folds that carry positives at 24h
    setups = {
        "headline_SVB": ([e for e in eps if cluster_of(e) not in ("SVB_2023", "FTX_2022")],
                         [e for e in eps if cluster_of(e) == "FTX_2022"],
                         [e for e in eps if cluster_of(e) == "SVB_2023"]),
        "loeo_FTX": ([e for e in eps if cluster_of(e) != "FTX_2022"], [],
                     [e for e in eps if cluster_of(e) == "FTX_2022"]),
        "loeo_Terra": ([e for e in eps if cluster_of(e) != "Terra_2022"], [],
                       [e for e in eps if cluster_of(e) == "Terra_2022"]),
    }
    rows = []
    for name, (tr, va, te) in setups.items():
        for seed in SEEDS:
            probs, yte, _ = run_partition(feat, tr, va, te, H, seed)
            base = float(yte.mean())
            rec = {"setup": name, "seed": seed, "base_rate": round(base, 4)}
            for m in MODELS:
                if m in probs:
                    rec[m] = full_report(yte, probs[m])["pr_auc"]
            rows.append(rec)
            print(name, "seed", seed, {m: round(rec.get(m, float('nan')), 3) for m in MODELS})
    df = pd.DataFrame(rows)
    df.to_csv("results/eval/multiseed_h1440.csv", index=False)

    # margin GAT - XGB and GraphSAGE - XGB, mean +/- std across seeds, per setup
    summary = {}
    for name in setups:
        sub = df[df["setup"] == name]
        s = {"base_rate": round(float(sub["base_rate"].mean()), 4)}
        for m in ["graphsage", "gat"]:
            if m in sub and "xgboost" in sub:
                margin = (sub[m] - sub["xgboost"]).dropna()
                lift = (sub[m] - sub["base_rate"]).dropna()
                s[m] = {
                    "pr_auc_mean": round(float(sub[m].mean()), 4),
                    "pr_auc_std": round(float(sub[m].std()), 4),
                    "margin_vs_xgb_mean": round(float(margin.mean()), 4),
                    "margin_vs_xgb_std": round(float(margin.std()), 4),
                    "lift_over_base_mean": round(float(lift.mean()), 4),
                }
        s["xgboost_pr_auc_mean"] = round(float(sub["xgboost"].mean()), 4)
        summary[name] = s
    Path("results/eval/multiseed_summary_h1440.json").write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
