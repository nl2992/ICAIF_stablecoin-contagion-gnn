"""
Interpretability: XGBoost gain ranking + GNNExplainer hub analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.explain import Explainer, GNNExplainer


def plot_xgb_importance(
    importance: Dict[str, float],
    topk: int = 20,
    out_path: Optional[Path] = None,
) -> None:
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:topk]
    names, scores = zip(*items)
    fig, ax = plt.subplots(figsize=(8, topk * 0.4))
    ax.barh(range(len(names)), scores[::-1], color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Gain")
    ax.set_title(f"XGBoost Feature Importance (top {topk})")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_gnn_explainer(
    model: torch.nn.Module,
    data,
    node_idx: int,
    epochs: int = 200,
) -> dict:
    """
    Run GNNExplainer on a single node and return node/edge masks.
    Returns dict with 'node_mask' and 'edge_mask' tensors.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx, edge_attr=getattr(data, "edge_attr", None))
    return {
        "node_mask": explanation.node_mask,
        "edge_mask": explanation.edge_mask,
    }


def hub_centrality_report(
    G: nx.Graph,
    node_labels: Dict[str, int],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identify contagion hubs: nodes with high betweenness that are
    also classified as stress-propagators.
    """
    betweenness = nx.betweenness_centrality(G, weight="weight")
    degree = dict(G.degree(weight="weight"))
    rows = []
    for node in G.nodes():
        rows.append({
            "node": node,
            "betweenness": betweenness.get(node, 0.0),
            "weighted_degree": degree.get(node, 0.0),
            "propagator": node_labels.get(node, 0),
        })
    df = pd.DataFrame(rows).sort_values("betweenness", ascending=False)
    return df.head(top_n).reset_index(drop=True)
