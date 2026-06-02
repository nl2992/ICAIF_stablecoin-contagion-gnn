"""Unit tests for feature builders."""
import numpy as np
import pandas as pd
import pytest

from scgnn.features.node_features import (
    amihud_illiquidity,
    lop_wedge,
    price_ratio,
    realized_vol,
)
from scgnn.features.labels import class_weights, make_labels


def _dummy_series(n=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    return pd.Series(1.0 + rng.normal(0, 0.001, n), index=idx, name="USDC")


def test_price_ratio_at_peg():
    s = _dummy_series()
    ratio = price_ratio(s)
    assert ratio.mean() == pytest.approx(s.mean(), rel=1e-6)


def test_realized_vol_nonneg():
    s = _dummy_series()
    ret = s.pct_change().dropna()
    vol = realized_vol(ret, "30min").dropna()
    assert (vol >= 0).all()


def test_amihud_nonneg():
    s = _dummy_series()
    ret = s.pct_change().dropna()
    vol = pd.Series(np.abs(np.random.default_rng(1).normal(1e6, 1e5, len(ret))), index=ret.index)
    amihud = amihud_illiquidity(ret, vol, "30min").dropna()
    assert (amihud >= 0).all()


def test_lop_wedge_zero_when_equal():
    s = _dummy_series()
    wedge = lop_wedge({"binance": s, "coinbase": s})
    assert (wedge == 0).all()


def test_make_labels_shape():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=500, freq="1min", tz="UTC")
    devs = {
        "USDC/binance": pd.Series(rng.normal(0, 5, 500), index=idx),
        "USDT/binance": pd.Series(rng.normal(0, 3, 500), index=idx),
    }
    labels = make_labels(devs, horizons_min=[30, 60])
    assert set(labels.keys()) == {30, 60}
    assert labels[30].shape == (500, 2)


def test_class_weights_balanced():
    s = pd.Series([0] * 90 + [1] * 10)
    cw = class_weights(s)
    assert cw[0] < cw[1]
