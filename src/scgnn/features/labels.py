"""
Label construction — pre-registered definitions from configs/experiment.yaml.

y_{j,t} = 1 (ONSET) if:
  1. node j is NOT stressed at t (not already in a depeg episode)
  2. node j ENTERS a stress period within [t+1, t+Δ]

"Stressed" at time t = |peg_deviation_bps| > threshold AND
  that condition has been True for ≥ sustained_min consecutive minutes.

This onset formulation avoids the trivial persistence label:
a model cannot trivially predict "it will be stressed" if it already is.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _stress_indicator(
    deviation_bps: pd.Series,
    threshold_bps: float,
    sustained_min: int = 10,
) -> pd.Series:
    """
    Binary series: 1 when |dev| > threshold for ≥ sustained_min consecutive minutes.
    Uses a rolling-min to require sustained breach.
    """
    above = (deviation_bps.abs() > threshold_bps).astype(int)
    # Rolling min over sustained_min: all minutes in window must be above threshold
    sustained = above.rolling(sustained_min, min_periods=sustained_min).min()
    return sustained.fillna(0).astype(int)


def make_onset_labels(
    peg_deviations: Dict[str, pd.Series],
    horizon_min: int,
    thresholds_bps: Dict[str, float],
    sustained_min: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Returns {node_id: binary array} of onset labels at each minute.

    y[t] = 1 iff:
      - stressed[t] == 0  (not currently in stress)
      - max(stressed[t+1 : t+horizon_min]) == 1  (enters stress within horizon)
    """
    out = {}
    for node, dev in peg_deviations.items():
        thr = thresholds_bps.get(node, thresholds_bps.get("default", 25.0))
        stressed = _stress_indicator(dev, thr, sustained_min)

        # Forward look: is there a stress onset within the next horizon_min minutes?
        future_max = stressed.rolling(horizon_min, min_periods=1).max().shift(-horizon_min)
        onset = ((stressed == 0) & (future_max == 1)).astype(int)
        out[node] = onset.fillna(0).values
    return out


def make_labels(
    peg_deviations: Dict[str, pd.Series],
    horizons_min: List[int],
    depeg_bps: float = 25.0,
) -> Dict[int, pd.DataFrame]:
    """
    Simpler (non-onset) labels for compatibility with existing code.
    y[t, j] = 1 if max |dev[t:t+Δ]| > depeg_bps.
    """
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons_min:
        label_dict = {}
        for node, dev in peg_deviations.items():
            rolled = dev.abs().rolling(h, min_periods=1).max().shift(-h)
            label_dict[node] = (rolled > depeg_bps).astype(int)
        out[h] = pd.DataFrame(label_dict)
    return out


def base_rate_table(
    labels: Dict[int, np.ndarray],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Compute positive rate per horizon.
    Warns if any horizon has < 2% positives (task may be degenerate).
    """
    rows = []
    for h in horizons:
        y = labels[h]
        flat = y.ravel() if hasattr(y, "ravel") else np.array(y)
        pos_rate = float(flat.mean())
        n_pos = int(flat.sum())
        n_total = len(flat)
        flag = "WARN: < 2%" if pos_rate < 0.02 else "OK"
        rows.append({
            "horizon_min": h,
            "positive_rate": pos_rate,
            "n_positive": n_pos,
            "n_total": n_total,
            "flag": flag,
        })
    df = pd.DataFrame(rows).set_index("horizon_min")
    return df


def class_weights(labels: pd.Series) -> Dict[int, float]:
    counts = labels.value_counts()
    total = len(labels)
    return {cls: total / (2.0 * cnt) for cls, cnt in counts.items()}


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
