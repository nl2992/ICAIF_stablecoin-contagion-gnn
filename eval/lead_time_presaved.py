"""Plan H (fast): lead-time decay from pre-saved probability arrays — no retraining."""
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scgnn.eval.metrics import full_report

horizons = [30, 60, 240, 1440]
models   = ["majority", "persistence", "logreg", "xgboost", "gru", "graphsage", "gat"]
colors   = {"gat": "#d62728", "graphsage": "#ff7f0e", "xgboost": "#1f77b4",
            "gru": "#2ca02c", "logreg": "#9467bd", "majority": "#aaa", "persistence": "#888"}

Path("results/figures").mkdir(parents=True, exist_ok=True)
rows = []
for h in horizons:
    y = np.load(f"results/ladder/test_labels_h{h}.npy")
    for m in models:
        p = np.load(f"results/ladder/probs_{m}_h{h}.npy")
        rep = full_report(y, p)
        pr  = rep["pr_auc"]
        roc = rep["roc_auc"]
        rows.append({"horizon_min": h, "model": m, "pr_auc": pr, "roc_auc": roc})
        print(f"h={h:4d} {m:12s}: PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")

df = pd.DataFrame(rows)
df.to_csv("results/eval/lead_time_presaved.csv", index=False)
print("\nSaved: results/eval/lead_time_presaved.csv")

# ---- Pivot table ----
pivot = df.pivot(index="horizon_min", columns="model", values="pr_auc")
print("\n=== PR-AUC by model and horizon ===")
print(pivot.round(4).to_string())

# ---- Figure ----
fig, ax = plt.subplots(figsize=(8, 5))
for m in ["gat", "graphsage", "xgboost", "gru"]:
    sub = df[df["model"] == m].sort_values("horizon_min")
    ax.plot(sub["horizon_min"], sub["pr_auc"], marker="o", label=m.upper(),
            color=colors.get(m, "black"), linewidth=2, markersize=6)

ax.set_xlabel("Prediction horizon (minutes)", fontsize=11)
ax.set_ylabel("PR-AUC (held-out SVB test)", fontsize=11)
ax.set_title("PR-AUC Decay vs Prediction Horizon", fontsize=12)
ax.set_xscale("log")
ax.set_xticks(horizons)
ax.set_xticklabels(["30m", "1h", "4h", "24h"])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
fig.savefig("results/figures/lead_time_decay_presaved.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/lead_time_decay_presaved.png")

# ---- Actionability analysis ----
print("\n=== GAT vs XGBoost gap by horizon ===")
gat_df = df[df["model"] == "gat"].set_index("horizon_min")["pr_auc"]
xgb_df = df[df["model"] == "xgboost"].set_index("horizon_min")["pr_auc"]
action_h = None
for h in sorted(horizons):
    gap = gat_df[h] - xgb_df[h]
    tag = "  <== ACTIONABILITY THRESHOLD" if gap >= 0.05 and action_h is None else ""
    print(f"  h={h:4d}min: GAT={gat_df[h]:.4f}  XGB={xgb_df[h]:.4f}  gap={gap:+.4f}{tag}")
    if gap >= 0.05 and action_h is None:
        action_h = h

if action_h:
    print(f"\nFinding: Graph-based prediction informative at >={action_h}min horizon.")
else:
    print("\nFinding: GAT gap never reaches +0.05 on held-out test (see LOCO for cross-validated estimate).")

# ---- Random chance baseline ----
base_rate = float(np.load("results/ladder/test_labels_h1440.npy").mean())
print(f"\nPositive rate (base rate) on test set: {base_rate:.4f}")
print(f"Random chance PR-AUC = base rate = {base_rate:.4f}")
print(f"GAT at 30min: {gat_df[30]:.4f} vs chance {base_rate:.4f}")
print(f"GAT at 24h:   {gat_df[1440]:.4f} vs chance {base_rate:.4f}")
