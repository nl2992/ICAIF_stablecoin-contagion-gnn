"""
Graph visualizations:
  - Stress vs calm snapshot comparison (Uniswap Fig 3 analog)
  - Positive-rate-over-time (Uniswap Fig 4 analog)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scgnn.graphs.stats import snapshot_stats, positive_rate_over_time, is_expected_hub


def _layout(G: nx.DiGraph) -> dict:
    """Deterministic spring layout."""
    return nx.spring_layout(G.to_undirected(), seed=42, weight="weight")


def plot_snapshot_comparison(
    calm_graph: nx.DiGraph,
    stress_graph: nx.DiGraph,
    calm_label: str = "Calm",
    stress_label: str = "Stress",
    out_path: Optional[Path] = None,
) -> None:
    """Side-by-side comparison of calm vs stress snapshot (Fig 3 analog)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, G, label in [(axes[0], calm_graph, calm_label), (axes[1], stress_graph, stress_label)]:
        pos = _layout(G)
        weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
        node_colors = [
            "tomato" if is_expected_hub(n) else "steelblue"
            for n in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=300, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=[w * 3 for w in weights],
            alpha=0.5,
            edge_color="gray",
            arrows=True,
            arrowstyle="->",
            arrowsize=10,
        )
        stats = snapshot_stats(G)
        ax.set_title(
            f"{label}\nNodes: {stats['n_nodes']}  Edges: {stats['n_edges']}  "
            f"Density: {stats['density']:.3f}  Avg-degree: {stats['avg_degree']:.1f}",
            fontsize=9,
        )
        ax.axis("off")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_positive_rate_timeline(
    label_series: pd.Series,
    episode_spans: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
    window: str = "24h",
    out_path: Optional[Path] = None,
) -> None:
    """
    Rolling positive rate over time with episode stress windows shaded (Fig 4 analog).

    episode_spans: list of (start, end, episode_name) for shading.
    """
    rate = positive_rate_over_time(label_series, window)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(rate.index, rate.values, color="steelblue", linewidth=1.0, label="Positive rate (24h roll)")

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(episode_spans)))
    for (start, end, name), c in zip(episode_spans, colors):
        ax.axvspan(start, end, alpha=0.15, color=c, label=name)

    ax.set_ylabel("Positive rate")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Contagion label positive rate over time")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
