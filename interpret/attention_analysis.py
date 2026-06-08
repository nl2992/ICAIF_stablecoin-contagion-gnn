"""
Plan F — GAT attention-weight analysis.

For each LOCO test fold, train GAT, then extract per-edge attention weights
during the 48h window before peak contagion. Aggregates across folds to rank
asset pairs by mean attention weight.

Attention weights are extracted from the final GAT layer using PyG's
`return_attention_weights=True` mechanism, exposed via
GATContagion.forward_with_attention().

Outputs:
  results/interpret/attention_per_fold.csv     edge-level attention per fold
  results/interpret/attention_hub_table.csv    top-N hub pairs ranked by mean attention
  results/interpret/attention_heatmap.png      NxN attention matrix (avg over crises)

Usage:
  python interpret/attention_analysis.py [--top_k 10] [--horizon 1440] [--window_h 48]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "eval"))

from run_benchmark import cluster_of  # noqa: E402
from scgnn.data.dataset import list_episodes, load_episode, load_feature_names, to_pyg_snapshots  # noqa: E402
from scgnn.eval.metrics import full_report  # noqa: E402
from scgnn.models.gnn_trainer import GNNContagionTrainer  # noqa: E402


def extract_attention_weights(
    model,
    episode_name: str,
    horizon: int,
    window_h: int = 48,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    For a trained GAT model, extract attention weights from all snapshots in the
    `window_h` hours BEFORE the snapshot of peak positive rate.

    Returns a DataFrame with columns: src_node, dst_node, mean_attention, n_snapshots
    """
    b = load_episode(episode_name)
    node_strs = b["node_strs"]
    N = len(node_strs)
    S = len(b["snapshots"])

    # Find snapshot of peak contagion signal (highest fraction of positives)
    labels = b["labels"][horizon]  # (S, N)
    active = b["active"]           # (S, N)
    pos_rates = []
    for si in range(S):
        m = active[si]
        pos_rates.append(labels[si][m].mean() if m.any() else 0.0)
    peak_si = int(np.argmax(pos_rates))
    start_si = max(0, peak_si - window_h)

    # Accumulate attention weights over the window
    edge_attention: dict[tuple, list] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for si, d in to_pyg_snapshots(b, horizon):
            if si < start_si or si > peak_si:
                continue
            d = d.to(device)
            if d.edge_index.shape[1] == 0:
                continue
            _, alpha, ei = model.model.forward_with_attention(
                d.x, d.edge_index, d.edge_attr)
            # alpha: (E, heads) or (E,); take mean over heads
            if alpha.dim() == 2:
                alpha = alpha.mean(dim=1)
            alpha_np = alpha.cpu().numpy()
            ei_np = ei.cpu().numpy()

            for k in range(ei_np.shape[1]):
                src_idx = int(ei_np[0, k])
                dst_idx = int(ei_np[1, k])
                src_name = node_strs[src_idx] if src_idx < N else f"n{src_idx}"
                dst_name = node_strs[dst_idx] if dst_idx < N else f"n{dst_idx}"
                edge_attention[(src_name, dst_name)].append(float(alpha_np[k]))

    rows = []
    for (src, dst), vals in edge_attention.items():
        rows.append({
            "episode": episode_name,
            "src_node": src,
            "dst_node": dst,
            "mean_attention": float(np.mean(vals)),
            "std_attention": float(np.std(vals)),
            "n_snapshots": len(vals),
        })
    return pd.DataFrame(rows)


def plot_attention_heatmap(
    hub_df: pd.DataFrame,
    node_order: list,
    out_path: Path,
) -> None:
    """Plot NxN attention heatmap averaged over all crisis episodes."""
    mat = np.zeros((len(node_order), len(node_order)))
    idx_map = {n: i for i, n in enumerate(node_order)}
    for _, row in hub_df.iterrows():
        si = idx_map.get(row["src_node"])
        di = idx_map.get(row["dst_node"])
        if si is not None and di is not None:
            mat[si, di] += row["mean_attention"]

    # Normalize rows so each source sums to 1 (or 0 if no outgoing edges)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.where(row_sums > 0, mat / row_sums, 0)

    fig, ax = plt.subplots(figsize=(max(6, len(node_order) * 0.7),
                                     max(5, len(node_order) * 0.6)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=mat.max())
    ax.set_xticks(range(len(node_order)))
    ax.set_yticks(range(len(node_order)))
    ax.set_xticklabels(node_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(node_order, fontsize=8)
    ax.set_xlabel("Target node (dst)")
    ax.set_ylabel("Source node (src)")
    ax.set_title("GAT attention weights averaged over crisis episodes\n(row-normalised; src → dst)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalised attention")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=10, help="Top-K edge pairs to report")
    ap.add_argument("--horizon", type=int, default=1440)
    ap.add_argument("--window_h", type=int, default=48,
                    help="Hours before peak contagion to extract attention from")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    feat_names = load_feature_names()
    all_eps = list_episodes()
    clusters = sorted(set(cluster_of(e) for e in all_eps))

    Path("results/interpret").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    all_attention = []

    for held_cluster in clusters:
        held = [e for e in all_eps if cluster_of(e) == held_cluster]
        train = [e for e in all_eps if cluster_of(e) != held_cluster]

        # Check if held has positives
        from scgnn.data.dataset import tabular_from_episodes
        _, yte, _ = tabular_from_episodes(held, args.horizon, feat_names)
        if yte.sum() == 0:
            print(f"  [skip] {held_cluster}: no positives")
            continue

        print(f"\n=== LOCO fold: {held_cluster} ===")
        for seed in range(args.seeds):
            trainer = GNNContagionTrainer(
                kind="gat", horizon=args.horizon, seed=seed,
                epochs=80, patience=10)
            trainer.fit(train, None)

            for ep_name in held:
                try:
                    atten_df = extract_attention_weights(
                        trainer, ep_name, args.horizon, args.window_h)
                    atten_df["held_cluster"] = held_cluster
                    atten_df["seed"] = seed
                    all_attention.append(atten_df)
                    print(f"  {ep_name} seed={seed}: {len(atten_df)} edge-attention entries")
                except Exception as e:
                    print(f"  [warn] {ep_name} seed={seed}: {e}")

    if not all_attention:
        print("No attention data collected. Check that GAT models have edges.")
        return

    full_df = pd.concat(all_attention, ignore_index=True)
    full_df.to_csv("results/interpret/attention_per_fold.csv", index=False)
    print(f"\nSaved: results/interpret/attention_per_fold.csv ({len(full_df)} rows)")

    # ---- Hub table: aggregate across folds and seeds ----
    agg = (full_df.groupby(["src_node", "dst_node"])["mean_attention"]
           .agg(["mean", "std", "count"])
           .reset_index()
           .rename(columns={"mean": "mean_attention", "std": "std_attention",
                             "count": "n_folds_seeds"})
           .sort_values("mean_attention", ascending=False)
           .reset_index(drop=True))

    top_k = agg.head(args.top_k)
    top_k.to_csv("results/interpret/attention_hub_table.csv", index=False)
    print(f"\n=== Top-{args.top_k} Contagion Edge Pairs (mean GAT attention) ===")
    print(top_k.round(4).to_string(index=False))

    # ---- Attention heatmap ----
    node_order = sorted(set(agg["src_node"]) | set(agg["dst_node"]))
    plot_attention_heatmap(agg, node_order,
                           Path("results/figures/attention_heatmap.png"))

    # ---- Hub summary ----
    print("\n=== Hub finding ===")
    if len(top_k) > 0:
        top_src = top_k.groupby("src_node")["mean_attention"].sum().idxmax()
        top_edge = top_k.iloc[0]
        print(f"  Dominant source node: {top_src}")
        print(f"  Highest-attention edge: {top_edge['src_node']} -> {top_edge['dst_node']}"
              f" (mean attention = {top_edge['mean_attention']:.4f})")
        xbar = agg["mean_attention"].mean()
        top_mult = top_edge["mean_attention"] / xbar if xbar > 0 else float("nan")
        print(f"  Top edge carries {top_mult:.1f}x average attention weight")


if __name__ == "__main__":
    main()
