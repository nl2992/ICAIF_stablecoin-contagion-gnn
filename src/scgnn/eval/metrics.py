"""Evaluation metrics and the lead-time accuracy decay analysis."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def full_classification_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def lead_time_decay(
    model,
    feature_dict: Dict[int, np.ndarray],   # {horizon_min: X}
    label_dict: Dict[int, np.ndarray],      # {horizon_min: y}
    horizons: List[int],
    predict_proba,
) -> pd.DataFrame:
    """
    For each horizon Δ, compute weighted-F1 and ROC-AUC.
    Returns a DataFrame with rows = horizon, cols = metric.
    """
    rows = []
    for h in horizons:
        X, y = feature_dict[h], label_dict[h]
        probs = predict_proba(X)
        if probs.ndim == 2:
            probs = probs[:, 1]
        report = full_classification_report(y, probs)
        rows.append({"horizon_min": h, **report})
    return pd.DataFrame(rows).set_index("horizon_min")
