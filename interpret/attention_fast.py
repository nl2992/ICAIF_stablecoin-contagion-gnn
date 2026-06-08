"""
Plan F (fast): GAT attention-weight hub analysis.

Trains one GAT model on all training episodes, then extracts attention
weights on each test episode snapshot to rank asset-pair edges.
Uses the SVB held-out test set (same episodes as run_benchmark headline).

No LOCO retraining — single fit, instant attention extraction.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "eval")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from pathlib import Path

from run_benchmark import cluster_of
from scgnn.data.dataset import list_episodes, load_episode, load_feature_names, to_pyg_snapshots
from scgnn.models.gnn_trainer import GNNContagionTrainer

H = 1440
SEEDS = [0, 1, 2]
WINDOW_H = 48   # snapshots before peak to extract attention from

feat_names = load_feature_names()
all_eps = list_episodes()
train_eps = [e for e in all_eps if cluster_of(e) not in ("SVB_2023", "FTX_2022")]
val_eps   = [e for e in all_eps if cluster_of(e) == "FTX_2022"]
test_eps  = [e for e in all_eps if cluster_of(e) == "SVB_2023"]

Path("results/interpret").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

all_attention = []

for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    trainer = GNNContagionTrainer(kind="gat", horizon=H, seed=seed,
                                   epochs=80, patience=10)
    trainer.fit(train_eps, val_eps)
    trainer.model.eval()

    for ep_name in test_eps + [e for e in all_eps if cluster_of(e) != "SVB_2023"][:3]:
        b = load_episode(ep_name)
        node_strs = b["node_strs"]
        N = len(node_strs)
        labels = b["labels"][H]
        active = b["active"]
        S = len(b["snapshots"])

        # Find peak snapshot
        pos_rates = [labels[si][active[si]].mean() if active[si].any() else 0.0
                     for si in range(S)]
        peak_si = int(np.argmax(pos_rates))
        start_si = max(0, peak_si - WINDOW_H)

        edge_attn = defaultdict(list)
        with torch.no_grad():
            for si, d in to_pyg_snapshots(b, H):
                if si < start_si or si > peak_si:
                    continue
                if d.edge_index.shape[1] == 0:
                    continue
                try:
                    _, alpha, ei = trainer.model.forward_with_attention(
                        d.x, d.edge_index, d.edge_attr)
                    if alpha.dim() == 2:
                        alpha = alpha.mean(dim=1)
                    alpha_np = alpha.cpu().numpy()
                    ei_np = ei.cpu().numpy()
                    for k in range(ei_np.shape[1]):
                        src = node_strs[ei_np[0, k]] if ei_np[0, k] < N else f"n{ei_np[0,k]}"
                        dst = node_strs[ei_np[1, k]] if ei_np[1, k] < N else f"n{ei_np[1,k]}"
                        edge_attn[(src, dst)].append(float(alpha_np[k]))
                except Exception as exc:
                    print(f"  [warn] {ep_name} si={si}: {exc}")

        for (src, dst), vals in edge_attn.items():
            all_attention.append({
                "episode": ep_name, "seed": seed,
                "src_node": src, "dst_node": dst,
                "mean_attention": float(np.mean(vals)),
                "n_snapshots": len(vals),
            })
        print(f"  {ep_name}: {len(edge_attn)} unique edges in [{start_si},{peak_si}]")

if not all_attention:
    print("No attention data.")
    sys.exit(0)

full_df = pd.DataFrame(all_attention)
full_df.to_csv("results/interpret/attention_raw.csv", index=False)

# ---- Aggregate: mean attention per edge across episodes and seeds ----
agg = (full_df.groupby(["src_node", "dst_node"])["mean_attention"]
       .agg(mean_attn="mean", std_attn="std", n_obs="count")
       .reset_index()
       .sort_values("mean_attn", ascending=False)
       .reset_index(drop=True))

agg.to_csv("results/interpret/attention_hub_table.csv", index=False)

print("\n=== Top-15 Contagion Edge Pairs (mean GAT attention) ===")
print(agg.head(15).round(4).to_string(index=False))

# ---- Heatmap ----
node_order = sorted(set(agg["src_node"]) | set(agg["dst_node"]))
idx_map = {n: i for i, n in enumerate(node_order)}
mat = np.zeros((len(node_order), len(node_order)))
for _, row in agg.iterrows():
    si = idx_map.get(row["src_node"])
    di = idx_map.get(row["dst_node"])
    if si is not None and di is not None:
        mat[si, di] = row["mean_attn"]

row_sums = mat.sum(axis=1, keepdims=True)
mat_norm = np.where(row_sums > 0, mat / row_sums, 0)

fig, ax = plt.subplots(figsize=(max(6, len(node_order)*0.7), max(5, len(node_order)*0.6)))
im = ax.imshow(mat_norm, cmap="YlOrRd", vmin=0)
ax.set_xticks(range(len(node_order))); ax.set_yticks(range(len(node_order)))
ax.set_xticklabels(node_order, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(node_order, fontsize=7)
ax.set_xlabel("Target (dst)"); ax.set_ylabel("Source (src)")
ax.set_title("GAT attention weights averaged over crisis episodes\n(row-normalized, src->dst)")
plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized attention")
plt.tight_layout()
fig.savefig("results/figures/attention_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: results/figures/attention_heatmap.png")

# ---- Hub summary ----
print("\n=== Hub finding ===")
top_src = agg.groupby("src_node")["mean_attn"].sum().idxmax()
top_edge = agg.iloc[0]
xbar = agg["mean_attn"].mean()
top_mult = top_edge["mean_attn"] / xbar if xbar > 0 else float("nan")
print(f"Dominant source node: {top_src}")
print(f"Top edge: {top_edge['src_node']} -> {top_edge['dst_node']} "
      f"(mean={top_edge['mean_attn']:.4f}, {top_mult:.1f}x average)")
print(f"Saved: results/interpret/attention_hub_table.csv")
