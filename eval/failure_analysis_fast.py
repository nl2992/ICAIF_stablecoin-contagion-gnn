"""
Plan G (fast): failure-mode analysis from pre-saved probability arrays.

Uses results/ladder/probs_{model}_h1440.npy on the SVB held-out test set.
Identifies the N most-confident FPs and FNs for each model, extracts
feature vectors and qualitative failure-mode labels.

No retraining — instant run.
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes
from scgnn.eval.metrics import full_report

# The SVB cluster is the held-out test set (same as run_benchmark.py)
CLUSTERS = {
    "USDC_SVB": "SVB_2023", "FRAX_SVB": "SVB_2023",
    "UST_Terra": "Terra_2022", "USDT_May2022": "Terra_2022",
    "DAI_FTX": "FTX_2022", "BUSD_winddown": "BUSD_2023",
    "USDT_Oct2018": "USDT_2018",
}

def cluster_of(ep):
    return CLUSTERS.get(ep, ep)

def classify_failure_mode(prob, true_label, rvol, threshold=0.5):
    pred = int(prob >= threshold)
    if pred == 1 and true_label == 0:
        return "FP_vol_spike" if rvol > 0.01 else "FP_spurious_edge"
    if pred == 0 and true_label == 1:
        return "FN_borderline" if prob > 0.3 else "FN_slow_onset"
    return "correct"

H = 1440
N_CASES = 15  # top-N FPs and FNs per model
MODELS = ["gat", "graphsage", "xgboost", "gru"]

feat_names = load_feature_names()
all_eps = list_episodes()
test_eps = [e for e in all_eps if cluster_of(e) == "SVB_2023"]

Xte, yte, meta = tabular_from_episodes(test_eps, H, feat_names)
meta = meta.reset_index(drop=True)

rvol_idx  = feat_names.index("rvol_1h")  if "rvol_1h"  in feat_names else 1
amihud_idx = feat_names.index("amihud")   if "amihud"   in feat_names else 4

print(f"Test set: {len(yte)} samples, {int(yte.sum())} positives (pos_rate={yte.mean():.4f})")

Path("results/eval").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

all_cases = []
for model_name in MODELS:
    probs = np.load(f"results/ladder/probs_{model_name}_h{H}.npy")
    rep = full_report(yte, probs)
    print(f"\n{model_name}: PR-AUC={rep['pr_auc']:.4f}")

    correct = (probs >= 0.5).astype(int) == yte.astype(int)
    misclassified_idx = np.where(~correct)[0]

    fp_idx = misclassified_idx[(yte[misclassified_idx] == 0)]
    fn_idx = misclassified_idx[(yte[misclassified_idx] == 1)]
    # Top-N by confidence
    fp_top = fp_idx[np.argsort(-probs[fp_idx])[:N_CASES]]
    fn_top = fn_idx[np.argsort(probs[fn_idx])[:N_CASES]]
    selected = np.concatenate([fp_top, fn_top])

    print(f"  FP={len(fp_idx)}, FN={len(fn_idx)} | showing top {N_CASES} each")

    for i in selected:
        x = Xte[i]
        feat_magnitudes = np.abs(x)
        top5 = feat_magnitudes.argsort()[::-1][:5]
        top_feats = [feat_names[j] for j in top5]
        fm = classify_failure_mode(float(probs[i]), int(yte[i]), float(x[rvol_idx]))
        row = meta.iloc[i].to_dict()
        row.update({
            "model": model_name,
            "prob": round(float(probs[i]), 4),
            "y_true": int(yte[i]),
            "error_type": "FP" if yte[i] == 0 else "FN",
            "failure_mode": fm,
            "top_feature_1": top_feats[0],
            "top_feature_1_val": round(float(x[top5[0]]), 4),
            "top_feature_2": top_feats[1] if len(top_feats) > 1 else "",
            "rvol_1h": round(float(x[rvol_idx]), 5),
            "amihud": round(float(x[amihud_idx]), 5),
        })
        all_cases.append(row)

df = pd.DataFrame(all_cases)
df.to_csv("results/eval/failure_cases_fast.csv", index=False)
print(f"\nSaved: results/eval/failure_cases_fast.csv ({len(df)} cases)")

# ---- Summary by model and failure mode ----
summary = (df.groupby(["model", "error_type", "failure_mode"])
           .agg(count=("prob", "count"),
                mean_prob=("prob", "mean"),
                top_feat=("top_feature_1", lambda x: x.value_counts().idxmax()))
           .reset_index())
summary.to_csv("results/eval/failure_summary_fast.csv", index=False)
print("\n=== Failure Mode Summary ===")
print(summary.to_string(index=False))

# ---- Gallery figure ----
fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5))
for ax, model_name in zip(axes, MODELS):
    sub = df[(df["model"] == model_name)].copy()
    top_fp = sub[sub["error_type"] == "FP"].nlargest(8, "prob")
    top_fn = sub[sub["error_type"] == "FN"].nsmallest(8, "prob")
    combined = pd.concat([top_fp, top_fn])
    colors_list = ["#d62728" if t == "FP" else "#1f77b4" for t in combined["error_type"]]
    labels = [f"{r['error_type']} {r.get('node','?')} p={r['prob']:.2f}" for _, r in combined.iterrows()]
    ax.barh(range(len(combined)), combined["prob"].values, color=colors_list, alpha=0.8)
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"{model_name.upper()}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
plt.suptitle("Top misclassifications (red=FP, blue=FN)", fontsize=10)
plt.tight_layout()
fig.savefig("results/figures/failure_gallery_fast.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/failure_gallery_fast.png")

# ---- 3 representative paper cases ----
print("\n=== Three representative failure cases for paper Section 5.4 ===")
modes_shown = set()
for _, row in df[df["model"] == "gat"].iterrows():
    fm = row["failure_mode"]
    if fm in modes_shown or fm == "correct":
        continue
    modes_shown.add(fm)
    print(f"\n  [{fm}]")
    print(f"    Node: {row.get('node','?')}, Episode: {row.get('episode','?')}, Snapshot: {row.get('snapshot','?')}")
    print(f"    Predicted: {row['prob']:.4f}, True: {row['y_true']}")
    print(f"    Top feature: {row['top_feature_1']} = {row['top_feature_1_val']:.4f}")
    print(f"    RVol1h: {row['rvol_1h']:.5f}, Amihud: {row['amihud']:.5f}")
    if len(modes_shown) >= 3:
        break
