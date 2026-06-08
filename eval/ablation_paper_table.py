"""
Plan A (fast): assemble the 4-condition ablation table for the paper.

Reads existing ablation_graph.csv (conditions 1, 2, 4) and trains
condition 3 (GAT + no node features, topology only) with 3 seeds.
Uses the held-out SVB test set (same as ablation_graph.py).

Output: results/eval/ablation_4condition.csv  (paper Table 3)
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "eval")
import numpy as np
import pandas as pd
from pathlib import Path

from run_benchmark import cluster_of
from scgnn.data.dataset import list_episodes, load_feature_names, tabular_from_episodes
from scgnn.eval.metrics import full_report
from scgnn.models.gnn_trainer import GNNContagionTrainer

H = 1440
SEEDS = [0, 1, 2]

feat_names = load_feature_names()
all_eps = list_episodes()
train = [e for e in all_eps if cluster_of(e) not in ("SVB_2023", "FTX_2022")]
val   = [e for e in all_eps if cluster_of(e) == "FTX_2022"]
test  = [e for e in all_eps if cluster_of(e) == "SVB_2023"]

_, yte, _ = tabular_from_episodes(test, H, feat_names)
base_rate = float(yte.mean())

# Load existing 3-condition results
existing = pd.read_csv("results/eval/ablation_graph.csv")
print("Existing ablation_graph.csv:")
print(existing.to_string(index=False))

# Condition 3: GAT + no node features (topology only)
print("\nTraining condition 3: GAT + ablate_node_features=True ...")
scores_c3 = []
for seed in SEEDS:
    tr = GNNContagionTrainer(kind="gat", horizon=H, seed=seed, epochs=80, patience=10,
                              ablate_node_features=True)
    tr.fit(train, val)
    p = tr.predict_episodes(test)
    pr = full_report(yte, p)["pr_auc"]
    scores_c3.append(pr)
    print(f"  seed={seed}: PR-AUC={pr:.4f}")

c3_mean = float(np.mean(scores_c3))
c3_std  = float(np.std(scores_c3))

# Build the 4-condition paper table
rows = []

# Condition 1: XGBoost (no graph)
r = existing[existing["rung"] == "node-only (XGBoost)"].iloc[0]
rows.append({"condition": "1. Tabular (XGBoost, no graph)",
             "node_features": "yes", "graph_edges": "n/a", "model": "XGBoost",
             "pr_auc_mean": r["pr_auc_mean"], "pr_auc_std": r["pr_auc_std"]})

# Condition 2: GAT + node features, no edges
r = existing[existing["rung"] == "gat (no edges)"].iloc[0]
rows.append({"condition": "2. GAT, node feat only (no edges)",
             "node_features": "yes", "graph_edges": "no", "model": "GAT",
             "pr_auc_mean": r["pr_auc_mean"], "pr_auc_std": r["pr_auc_std"]})

# Condition 3: GAT + edges only, no node features
rows.append({"condition": "3. GAT, edges only (no node feat)",
             "node_features": "no", "graph_edges": "yes", "model": "GAT",
             "pr_auc_mean": round(c3_mean, 4), "pr_auc_std": round(c3_std, 4)})

# Condition 4: GAT full
r = existing[existing["rung"] == "gat (real edges)"].iloc[0]
rows.append({"condition": "4. GAT, full (node feat + edges)",
             "node_features": "yes", "graph_edges": "yes", "model": "GAT",
             "pr_auc_mean": r["pr_auc_mean"], "pr_auc_std": r["pr_auc_std"]})

df = pd.DataFrame(rows)
df["base_rate"] = round(base_rate, 4)

# Deltas vs XGBoost baseline
xgb_pr = df.loc[df["condition"].str.startswith("1"), "pr_auc_mean"].iloc[0]
df["delta_vs_xgboost"] = (df["pr_auc_mean"] - xgb_pr).round(4)

Path("results/eval").mkdir(parents=True, exist_ok=True)
df.to_csv("results/eval/ablation_4condition.csv", index=False)

print("\n" + "=" * 70)
print("=== 4-Condition Ablation (held-out SVB @24h) ===")
print("=" * 70)
print(df[["condition", "pr_auc_mean", "pr_auc_std", "delta_vs_xgboost"]].to_string(index=False))
print(f"\nBase rate (positive rate): {base_rate:.4f}")

# Key contributions
c2_mean = df.loc[df["condition"].str.startswith("2"), "pr_auc_mean"].iloc[0]
c4_mean = df.loc[df["condition"].str.startswith("4"), "pr_auc_mean"].iloc[0]
graph_contrib = c4_mean - c2_mean
node_contrib  = c4_mean - c3_mean

print(f"\nGraph topology contribution (cond4 - cond2): {graph_contrib:+.4f} PR-AUC")
print(f"Node feature contribution  (cond4 - cond3): {node_contrib:+.4f} PR-AUC")
print(f"Full GAT vs XGBoost:                         {c4_mean - xgb_pr:+.4f} PR-AUC")
print(f"\nHeadline: 'Directed lead-lag graph topology contributes {graph_contrib:+.4f} PR-AUC")
print(f" beyond node microstructure features alone (cond2 vs cond4).'")
