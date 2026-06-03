"""
Contagion label construction.
Pre-registered definition — do NOT modify thresholds retroactively.

=======================================================================
EXACT LABEL RULE (v1, pre-registered 2025-06-03)
=======================================================================

We define the ONSET label, not the "active stress" label, because:
  - Active stress trivially persists (majority baseline near-perfect)
  - ONSET captures the causal question: "will stress SPREAD to this node?"

y_{j,t} = 1  iff ALL of the following hold:

  (a) ORIGIN EXCLUSION:
      j ≠ origin_node(episode)
      The node that triggered the episode cannot be labelled as receiving
      contagion — it IS the contagion source.  Mixing it in inflates
      positive rates and confounds the causal interpretation.

  (b) PRE-EXISTING STRESS MASK:
      stressed_{j,t-1} = 0
      If j was already off-peg before the shock window, any subsequent
      stress is continuation, not propagation.  We mask these out.
      "Stressed at t" = |price_j(t) − 1| > threshold_bps AND that
      condition has held for ≥ sustained_min consecutive minutes.

  (c) ONSET WITHIN HORIZON:
      ∃ t' ∈ (t, t+Δ] : stressed_{j,t'} = 1
      A new stress period begins for j within the next Δ minutes.

THRESHOLD CHOICE:
  We use a fixed BPS threshold, not a relative move or vol/Amihud spike,
  because:
  - Stablecoins are designed to be exactly $1; any departure is the signal.
  - Relative moves from an already-depegged coin are misleading.
  - Amihud/vol measure microstructure noise, not the peg event.

  Thresholds (pre-registered, asset-class-specific):
    fiat_backed:   25 bps  (USDC, USDT, TUSD, PYUSD, BUSD)
    crypto_backed: 75 bps  (DAI, FRAX)
    synthetic:     50 bps  (USDe)

  A sensitivity sweep over {10, 25, 50} bps is run in
  eval/label_sensitivity.py; hub rankings must be stable across this range.

SUSTAINED REQUIREMENT:
  sustained_min = 10  (10 consecutive 1-min bars above threshold)
  Prevents noise from causing transient crosses from being labeled
  as true contagion events.

=======================================================================
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ core primitive


def _stress_indicator(
    deviation_bps: pd.Series,
    threshold_bps: float,
    sustained_min: int = 10,
) -> pd.Series:
    """
    Binary: 1 when |dev| > threshold for ≥ sustained_min consecutive minutes.

    Uses rolling min so every minute in the window must exceed the threshold.
    Returns int series aligned to deviation_bps.index.
    """
    above = (deviation_bps.abs() > threshold_bps).astype(int)
    sustained = above.rolling(sustained_min, min_periods=sustained_min).min()
    return sustained.fillna(0).astype(int)


# ------------------------------------------------------------------ onset labels (primary)


def make_onset_labels(
    peg_deviations: Dict[str, pd.Series],
    horizon_min: int,
    thresholds_bps: Dict[str, float],
    sustained_min: int = 10,
    origin_node: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute onset labels for all nodes.

    Rules enforced:
      (a) Origin node always receives label=0 (excluded)
      (b) Pre-existing stress at t masks the label (t must be calm)
      (c) Stress onset must occur within (t, t+horizon_min]

    Parameters
    ----------
    peg_deviations : {node_str: bps series}
    horizon_min    : prediction horizon Δ
    thresholds_bps : {node_str: threshold}  — use asset-class defaults
    sustained_min  : consecutive-minutes requirement
    origin_node    : the episode trigger node (excluded from labels)

    Returns
    -------
    {node_str: binary np.ndarray}
    """
    out: Dict[str, np.ndarray] = {}
    for node, dev in peg_deviations.items():
        thr = thresholds_bps.get(node, 25.0)
        stressed = _stress_indicator(dev, thr, sustained_min)

        # (a) Origin exclusion
        if origin_node is not None and node == origin_node:
            out[node] = np.zeros(len(dev), dtype=np.int8)
            continue

        # (b) Pre-existing stress mask: node must be calm at t
        currently_calm = (stressed == 0).astype(int)

        # (c) Future stress onset: does stress begin within [t+1, t+Δ]?
        # Shift stressed backward by horizon: 1 if ANY stress in window
        future_max = stressed.rolling(horizon_min, min_periods=1).max().shift(-horizon_min)

        onset = (currently_calm * (future_max.fillna(0) > 0)).astype(np.int8)
        out[node] = onset.values
    return out


# ------------------------------------------------------------------ simplified labels (compatibility)


def make_labels(
    peg_deviations: Dict[str, pd.Series],
    horizons_min: List[int],
    depeg_bps: float = 25.0,
) -> Dict[int, pd.DataFrame]:
    """
    Active-stress label (without onset/mask logic) — kept for backward compatibility.
    Use make_onset_labels for all new code.
    """
    out: Dict[int, pd.DataFrame] = {}
    for h in horizons_min:
        label_dict = {}
        for node, dev in peg_deviations.items():
            rolled = dev.abs().rolling(h, min_periods=1).max().shift(-h)
            label_dict[node] = (rolled > depeg_bps).astype(int)
        out[h] = pd.DataFrame(label_dict)
    return out


# ------------------------------------------------------------------ base-rate reporting


def base_rate_table(
    labels: Dict[int, np.ndarray],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Positive rate per horizon.
    Warns if < 2% (task may be trivial) or > 50% (near-majority label flip).
    """
    rows = []
    for h in horizons:
        y = labels.get(h)
        if y is None:
            continue
        flat = y.ravel() if hasattr(y, "ravel") else np.array(y)
        pos_rate = float(flat.mean())
        n_pos = int(flat.sum())
        n_total = len(flat)
        if pos_rate < 0.02:
            flag = "WARN: <2% positives — task may be degenerate"
        elif pos_rate > 0.50:
            flag = "WARN: >50% positives — threshold may be too loose"
        else:
            flag = "OK"
        rows.append({
            "horizon_min": h,
            "positive_rate": round(pos_rate, 4),
            "n_positive": n_pos,
            "n_total": n_total,
            "imbalance_ratio": round((n_total - n_pos) / max(n_pos, 1), 1),
            "flag": flag,
        })
    return pd.DataFrame(rows).set_index("horizon_min")


def per_episode_base_rates(
    episode_labels: Dict[str, Dict[int, np.ndarray]],  # {ep_name: {horizon: y_array}}
    horizons: List[int],
) -> pd.DataFrame:
    """
    Base rate for every episode × horizon combination.
    This is the table that will catch degenerate horizons per episode.
    """
    rows = []
    for ep_name, label_dict in episode_labels.items():
        for h in horizons:
            y = label_dict.get(h)
            if y is None:
                continue
            flat = y.ravel()
            rows.append({
                "episode": ep_name,
                "horizon_min": h,
                "positive_rate": round(float(flat.mean()), 4),
                "n_positive": int(flat.sum()),
                "n_total": len(flat),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ asset-class threshold map


def default_threshold_map(
    node_str: str,
    cfg: Optional[dict] = None,
) -> float:
    """Return the pre-registered threshold for a node (by asset class)."""
    asset = node_str.split("/")[0].upper()
    if cfg:
        thr = cfg.get("labels", {}).get("thresholds_bps", {})
        if asset in ("USDC", "USDT", "TUSD", "PYUSD", "BUSD"):
            return float(thr.get("fiat_backed", 25.0))
        if asset in ("DAI", "FRAX"):
            return float(thr.get("crypto_backed", 75.0))
        if asset in ("USDE",):
            return float(thr.get("synthetic", 50.0))
    # Defaults
    if asset in ("USDC", "USDT", "TUSD", "PYUSD", "BUSD"):
        return 25.0
    if asset in ("DAI", "FRAX"):
        return 75.0
    return 50.0


# ------------------------------------------------------------------ utilities


def class_weights(labels: pd.Series) -> Dict[int, float]:
    counts = labels.value_counts()
    total = len(labels)
    return {int(cls): total / (2.0 * cnt) for cls, cnt in counts.items()}


def weighted_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
