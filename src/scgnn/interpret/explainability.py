"""
Interpretability layer.

1. XGBoost gain ranking — global feature importance.
2. GNNExplainer — per-node subgraph importance.
3. Hub centrality report — betweenness × propagator label.
4. Single-prediction trace — worked case study (USDC_SVB).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ------------------------------------------------------------------ XGBoost


def xgb_gain_ranking(model, feature_names: List[str], topk: int = 20) -> pd.DataFrame:
    scores = model.get_booster().get_score(importance_type="gain")
    rows = [{"feature": f, "gain": scores.get(f, 0.0)} for f in feature_names]
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return df.head(topk)


def plot_xgb_importance(
    df: pd.DataFrame,
    out_path: Optional[Path] = None,
    title: str = "XGBoost feature importance (gain)",
) -> None:
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.38)))
    ax.barh(df["feature"][::-1], df["gain"][::-1], color="steelblue")
    ax.set_xlabel("Gain")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ GNNExplainer


def run_gnn_explainer(
    model,
    data,
    node_idx: int,
    epochs: int = 200,
) -> dict:
    from torch_geometric.explain import Explainer, GNNExplainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )
    explanation = explainer(data.x, data.edge_index, index=node_idx, edge_attr=getattr(data, "edge_attr", None))
    return {"node_mask": explanation.node_mask, "edge_mask": explanation.edge_mask}


def plot_node_importance_heatmap(
    node_masks: Dict[str, np.ndarray],   # {node_str: feature_importance_vector}
    feature_names: List[str],
    out_path: Optional[Path] = None,
) -> None:
    """Heatmap of per-node feature importance (Uniswap Fig 6 analog)."""
    nodes = list(node_masks.keys())
    matrix = np.array([node_masks[n] for n in nodes])
    fig, ax = plt.subplots(figsize=(max(6, len(feature_names) * 0.5), max(4, len(nodes) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=7)
    ax.set_title("Per-node feature importance (GNNExplainer)")
    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Hub centrality


def hub_centrality_report(
    G: nx.Graph,
    propagator_labels: Dict[str, int],   # {node_str: 0 or 1}
    top_n: int = 10,
) -> pd.DataFrame:
    UG = G.to_undirected()
    betweenness = nx.betweenness_centrality(UG, weight="weight")
    degree = dict(G.degree(weight="weight"))
    rows = []
    for node in G.nodes():
        rows.append({
            "node": node,
            "betweenness": betweenness.get(node, 0.0),
            "weighted_degree": degree.get(node, 0.0),
            "propagator": propagator_labels.get(node, 0),
        })
    df = pd.DataFrame(rows).sort_values("betweenness", ascending=False).head(top_n)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ Case study


def trace_single_prediction(
    model,
    X_node: np.ndarray,            # feature vector for one (node, time) pair
    feature_names: List[str],
    node_str: str,
    t: pd.Timestamp,
    true_label: int,
    predicted_prob: float,
    top_k_features: int = 5,
    shap_values: Optional[np.ndarray] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Produce a human-readable trace of a single model prediction.
    Returns a DataFrame with the top contributing features.
    """
    if shap_values is not None:
        contrib = pd.Series(shap_values, index=feature_names).abs().sort_values(ascending=False)
    else:
        contrib = pd.Series(X_node, index=feature_names).abs().sort_values(ascending=False)

    top = contrib.head(top_k_features).reset_index()
    top.columns = ["feature", "contribution"]
    top["feature_value"] = [float(X_node[feature_names.index(f)]) for f in top["feature"]]

    summary = pd.DataFrame([{
        "node": node_str,
        "timestamp": str(t),
        "predicted_prob": predicted_prob,
        "true_label": true_label,
        "correct": int((predicted_prob >= 0.5) == bool(true_label)),
    }])
    print("=== Prediction trace ===")
    print(summary.to_string(index=False))
    print("\nTop contributing features:")
    print(top.to_string(index=False))

    if out_path:
        summary.to_csv(out_path.with_suffix(".summary.csv"), index=False)
        top.to_csv(out_path.with_suffix(".features.csv"), index=False)
    return top
