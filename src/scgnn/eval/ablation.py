"""
Ablation study framework.

Three pre-defined ablations:
  (a) real-only vs real+synthetic data
  (b) node-features-only vs +edge-features (tests edge contribution)
  (c) graph model (GraphSAGE) vs XGBoost on identical features (tests GNN framing)

Each ablation uses the same chronological split and PR-AUC as primary metric.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from scgnn.eval.metrics import full_report, results_table


def ablation_real_vs_synthetic(
    train_real_fn: Callable,      # () → (X_train, y_train)
    train_real_synth_fn: Callable,
    val_fn: Callable,             # () → (X_val, y_val)
    test_fn: Callable,
    model_factory: Callable,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Compare a model trained on real-only vs real+synthetic data.
    Returns results table with rows [real_only, real+synthetic].
    """
    X_val, y_val = val_fn()
    X_test, y_test = test_fn()
    reports = {}

    for label, train_fn in [("real_only", train_real_fn), ("real+synthetic", train_real_synth_fn)]:
        X_train, y_train = train_fn()
        model = model_factory()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test).astype(float)
        reports[f"{model_name}_{label}"] = full_report(y_test.ravel(), probs.ravel())

    return results_table(reports)


def ablation_edge_features(
    X_node_only: np.ndarray,
    X_with_edge: np.ndarray,
    y_train: np.ndarray,
    X_test_node: np.ndarray,
    X_test_edge: np.ndarray,
    y_test: np.ndarray,
    model_factory: Callable,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Compare node-features-only vs node+edge-features.
    """
    reports = {}
    for label, X_tr, X_te in [
        ("node_only", X_node_only, X_test_node),
        ("node+edge", X_with_edge, X_test_edge),
    ]:
        m = model_factory()
        m.fit(X_tr, y_train)
        probs = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_te).astype(float)
        reports[f"{model_name}_{label}"] = full_report(y_test.ravel(), probs.ravel())
    return results_table(reports)


def ablation_graph_vs_tabular(
    X_flat_train: np.ndarray,
    y_train: np.ndarray,
    X_flat_test: np.ndarray,
    y_test: np.ndarray,
    tabular_model_factory: Callable,
    graph_predict_fn: Callable,      # (X_flat_test) → probs
    tabular_name: str = "xgboost",
) -> pd.DataFrame:
    """
    The critical ablation: does GraphSAGE beat XGBoost on the *same* features?
    If not, the GNN framing is unjustified — we report this honestly.
    """
    reports = {}

    tab = tabular_model_factory()
    tab.fit(X_flat_train, y_train)
    tab_probs = tab.predict_proba(X_flat_test)[:, 1]
    reports[tabular_name] = full_report(y_test.ravel(), tab_probs.ravel())

    gnn_probs = graph_predict_fn(X_flat_test)
    reports["graphsage"] = full_report(y_test.ravel(), gnn_probs.ravel())

    df = results_table(reports)
    delta = df.loc["graphsage", "pr_auc"] - df.loc[tabular_name, "pr_auc"]
    df["delta_vs_tabular"] = [0.0, delta]
    return df
