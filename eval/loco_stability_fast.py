"""
Plan B (fast): sparse-episode sensitivity using existing LOCO results.

Reads the pre-computed loeo_h1440.csv (all-7 LOCO) and computes the
stable-5 mean by dropping the two sparse folds (BUSD_2023, USDT_2018).
No retraining needed for the comparison table.

For a fully-retrained stable-5 LOCO, use scripts/run_loco.py --compare_sets.
"""
import sys
sys.path.insert(0, "src")
import pandas as pd
import numpy as np
from pathlib import Path

H = 1440
SPARSE_FOLDS = {"BUSD_2023", "USDT_2018"}

# Load existing LOEO results
loeo = pd.read_csv(f"results/eval/loeo_h{H}.csv", index_col=0)
print("=== Existing LOCO results (all-7) ===")
print(loeo.round(4).to_string())

# Separate models from metadata columns
meta_cols = ["episodes", "n_test", "pos_rate"]
model_cols = [c for c in loeo.columns if c not in meta_cols]

# all-7 mean (valid folds only, pos_rate > 0)
all7_valid = loeo.drop(index="MEAN", errors="ignore")
all7_valid = all7_valid[all7_valid["pos_rate"] > 0]

# stable-5: drop sparse folds
stable5_valid = all7_valid[~all7_valid.index.isin(SPARSE_FOLDS)]

print(f"\nAll-7 valid folds ({len(all7_valid)}): {list(all7_valid.index)}")
print(f"Stable-5 valid folds ({len(stable5_valid)}): {list(stable5_valid.index)}")

# Comparison table
rows = []
for model in model_cols:
    if model not in all7_valid.columns:
        continue
    a7 = all7_valid[model].dropna()
    s5 = stable5_valid[model].dropna()
    rows.append({
        "model": model,
        "all7_mean_pr_auc": round(float(a7.mean()), 4),
        "all7_std": round(float(a7.std()), 4),
        "all7_n_folds": len(a7),
        "stable5_mean_pr_auc": round(float(s5.mean()), 4),
        "stable5_std": round(float(s5.std()), 4),
        "stable5_n_folds": len(s5),
        "delta_stable5_minus_all7": round(float(s5.mean() - a7.mean()), 4),
    })

comp = pd.DataFrame(rows).set_index("model")
Path("results/eval").mkdir(parents=True, exist_ok=True)
comp.to_csv(f"results/eval/loco_stability_fast_h{H}.csv")

print("\n=== Sparse-Episode Sensitivity (Plan B) ===")
print(comp.to_string())

# ---- Key comparison: GAT and XGBoost ----
print("\n=== Key comparison: GAT vs XGBoost ===")
for model in ["gat", "xgboost"]:
    if model not in comp.index:
        continue
    r = comp.loc[model]
    print(f"  {model}:")
    print(f"    all-7  : {r['all7_mean_pr_auc']:.4f} +/- {r['all7_std']:.4f}")
    print(f"    stable-5: {r['stable5_mean_pr_auc']:.4f} +/- {r['stable5_std']:.4f}")
    print(f"    delta  : {r['delta_stable5_minus_all7']:+.4f}")

# ---- GAT margin over XGBoost ----
print("\n=== GAT margin over XGBoost (LOCO mean PR-AUC) ===")
for subset, valid_df in [("all-7", all7_valid), ("stable-5", stable5_valid)]:
    if "gat" in valid_df.columns and "xgboost" in valid_df.columns:
        valid = valid_df[valid_df["pos_rate"] > 0]
        gat = valid["gat"].dropna()
        xgb = valid["xgboost"].dropna()
        common = gat.index.intersection(xgb.index)
        margin = (gat[common] - xgb[common]).mean()
        print(f"  {subset}: GAT mean={gat[common].mean():.4f}, XGB={xgb[common].mean():.4f}, "
              f"margin={margin:+.4f}")

# ---- Pre-registration note ----
print("\n=== Pre-registration note ===")
print("Episode quality criterion (set before final model runs):")
print("  Exclude if: pos_rate == 0 OR (pos_rate < 0.02 AND data_quality == 'incomplete')")
print("  Excluded: BUSD_winddown (pos_rate=0.0), USDT_Oct2018 (pos_rate=0.0103, pre-DeFi)")
print("  Evidence: data/processed/base_rates.csv and configs/episodes_v2.yaml")
