"""
Binary contagion label construction.

Label y_{j,t+Δ} = 1 iff node j's peg deviation crosses `depeg_bps` threshold
within horizon Δ minutes after time t.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def make_labels(
    peg_deviations: Dict[str, pd.Series],
    horizons_min: List[int],
    depeg_bps: float = 10.0,
) -> Dict[int, pd.DataFrame]:
    """
    Returns {horizon: DataFrame(index=time, columns=node_ids)} of binary labels.
    For each horizon Δ, label[t, j] = 1 if max |dev[t:t+Δ]| > depeg_bps.
    """
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons_min:
        label_dict = {}
        for node, dev in peg_deviations.items():
            rolled = (
                dev.abs()
                .rolling(h, min_periods=1)
                .max()
                .shift(-h)           # look-forward
            )
            label_dict[node] = (rolled > depeg_bps).astype(int)
        out[h] = pd.DataFrame(label_dict)
    return out


def class_weights(labels: pd.Series) -> Dict[int, float]:
    counts = labels.value_counts()
    total = len(labels)
    return {cls: total / (2 * cnt) for cls, cnt in counts.items()}


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
