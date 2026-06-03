"""
Evaluation metrics.

PRIMARY metric: PR-AUC (honest under class imbalance).
Reported alongside: weighted-F1, ROC-AUC, precision, recall.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def full_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    has_both = len(np.unique(y_true)) > 1
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "positive_rate": float(y_true.mean()),
        "n": len(y_true),
    }


def results_table(
    model_reports: Dict[str, dict],
    metrics: List[str] = ["pr_auc", "roc_auc", "weighted_f1", "precision", "recall"],
) -> pd.DataFrame:
    """Assemble the main results table (models × metrics)."""
    rows = []
    for name, report in model_reports.items():
        row = {"model": name}
        for m in metrics:
            row[m] = report.get(m, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def lead_time_table(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_dict: Dict[int, np.ndarray],
    label_dict: Dict[int, np.ndarray],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Compute pr_auc and weighted_f1 at each horizon.
    Returns DataFrame indexed by horizon_min.
    """
    rows = []
    for h in horizons:
        X, y = feature_dict[h], label_dict[h]
        probs = predict_fn(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
        report = full_report(y.ravel(), probs.ravel())
        rows.append({"horizon_min": h, **{k: report[k] for k in ["pr_auc", "roc_auc", "weighted_f1"]}})
    return pd.DataFrame(rows).set_index("horizon_min")


def threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    """PR-AUC and weighted-F1 at each classification threshold."""
    rows = []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        rows.append({"threshold": thr, "pr_auc": pr, "weighted_f1": f1})
    return pd.DataFrame(rows).set_index("threshold")
