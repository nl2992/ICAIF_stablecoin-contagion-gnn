"""Tests for onset label logic and base-rate computation."""
import numpy as np
import pandas as pd
import pytest

from scgnn.features.labels import (
    _stress_indicator,
    make_onset_labels,
    base_rate_table,
    class_weights,
)


def _dev_series(n=300, shock_start=100, shock_len=20, shock_bps=50.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    dev = pd.Series(rng.normal(0, 2, n), index=idx)
    dev.iloc[shock_start:shock_start + shock_len] += shock_bps
    return dev


def test_stress_indicator_detects_sustained_breach():
    dev = _dev_series(shock_bps=50.0)
    stress = _stress_indicator(dev, threshold_bps=25.0, sustained_min=10)
    assert stress.max() == 1
    # Should not trigger before the shock
    assert stress.iloc[:100].sum() == 0


def test_stress_indicator_no_false_trigger_below_threshold():
    dev = _dev_series(shock_bps=5.0)
    stress = _stress_indicator(dev, threshold_bps=25.0, sustained_min=10)
    assert stress.sum() == 0


def test_onset_labels_shape():
    dev = _dev_series()
    thresholds = {"USDC/binance": 25.0}
    labels = make_onset_labels({"USDC/binance": dev}, horizon_min=30,
                               thresholds_bps=thresholds, sustained_min=10)
    assert "USDC/binance" in labels
    assert len(labels["USDC/binance"]) == len(dev)


def test_onset_label_is_binary():
    dev = _dev_series()
    labels = make_onset_labels({"node": dev}, horizon_min=30,
                               thresholds_bps={"node": 25.0}, sustained_min=10)
    arr = labels["node"]
    assert set(arr).issubset({0, 1})


def test_onset_label_positive_rate_nonzero():
    dev = _dev_series(shock_bps=60.0, shock_len=30)
    labels = make_onset_labels({"node": dev}, horizon_min=60,
                               thresholds_bps={"node": 25.0}, sustained_min=5)
    assert labels["node"].mean() > 0


def test_base_rate_table_flags_low():
    labels = {30: np.array([0] * 99 + [1]), 60: np.array([0] * 80 + [1] * 20)}
    tbl = base_rate_table(labels, [30, 60])
    assert tbl.loc[30, "flag"].startswith("WARN")
    assert tbl.loc[60, "flag"] == "OK"
