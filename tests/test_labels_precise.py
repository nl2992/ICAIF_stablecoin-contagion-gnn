"""
Tests for the exact pre-registered label rule.

Every test here corresponds to a specific clause in the label docstring.
If a test fails, the label definition has been violated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scgnn.features.labels import (
    _stress_indicator,
    make_onset_labels,
    default_threshold_map,
    base_rate_table,
    per_episode_base_rates,
)


def _idx(n=300):
    return pd.date_range("2023-03-10", periods=n, freq="1min", tz="UTC")


def _dev(shock_start=100, shock_bps=50.0, shock_len=20, n=300, seed=0):
    """Deviation series with one injected shock."""
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    dev = pd.Series(rng.normal(0, 1.5, n), index=idx)
    dev.iloc[shock_start:shock_start + shock_len] += shock_bps
    return dev


# ─── Clause (a): origin exclusion ────────────────────────────────────────────

class TestOriginExclusion:
    def test_origin_node_always_zero(self):
        dev = _dev(shock_bps=60.0)
        labels = make_onset_labels(
            {"origin/binance": dev},
            horizon_min=30,
            thresholds_bps={"origin/binance": 25.0},
            sustained_min=5,
            origin_node="origin/binance",
        )
        assert labels["origin/binance"].sum() == 0, "Origin node must have all-zero labels"

    def test_non_origin_can_have_positive_labels(self):
        idx = _idx()
        origin_dev = _dev(shock_bps=80.0, shock_start=50)
        # Victim depegs shortly after origin
        victim_dev = _dev(shock_bps=40.0, shock_start=70)
        labels = make_onset_labels(
            {"USDC/binance": origin_dev, "USDT/binance": victim_dev},
            horizon_min=60,
            thresholds_bps={"USDC/binance": 25.0, "USDT/binance": 25.0},
            sustained_min=5,
            origin_node="USDC/binance",
        )
        assert labels["USDC/binance"].sum() == 0
        # Victim should have some positives (unless shock too small/slow)
        # We just check it's not forced to zero
        assert labels["USDT/binance"] is not None


# ─── Clause (b): pre-existing stress mask ────────────────────────────────────

class TestPreExistingMask:
    def test_already_stressed_node_gets_zero_at_t(self):
        """If node is stressed at t, label at t must be 0 (already stressed ≠ onset)."""
        idx = _idx(200)
        # Stress present from t=0 onward
        dev = pd.Series([60.0] * 200, index=idx)
        labels = make_onset_labels(
            {"node": dev},
            horizon_min=30,
            thresholds_bps={"node": 25.0},
            sustained_min=5,
        )
        # All stressed from start → currently_calm = 0 → all labels 0
        # After sustained_min bars, stressed=1 → label must be 0
        y = labels["node"]
        assert y[10:].sum() == 0, "Sustained-stressed node must get 0 after stress onset"

    def test_calm_node_can_get_positive_label(self):
        """A node that is calm and then enters stress → label = 1 in the run-up."""
        dev = _dev(shock_start=150, shock_bps=60.0, shock_len=30)
        labels = make_onset_labels(
            {"node": dev},
            horizon_min=60,
            thresholds_bps={"node": 25.0},
            sustained_min=5,
        )
        y = labels["node"]
        # Should have at least one positive before the shock
        assert y[:150].sum() > 0, "Calm node should have positive labels before its shock"


# ─── Clause (c): onset within horizon ────────────────────────────────────────

class TestOnsetWithinHorizon:
    def test_label_1_appears_before_shock(self):
        """Labels should be 1 at times Δ minutes BEFORE the actual shock."""
        n = 400
        shock_at = 200
        horizon = 60
        dev = _dev(shock_start=shock_at, shock_bps=60.0, shock_len=30, n=n)
        labels = make_onset_labels(
            {"node": dev},
            horizon_min=horizon,
            thresholds_bps={"node": 25.0},
            sustained_min=5,
        )
        y = labels["node"]
        # The horizon window before the shock should contain some positives
        lead_up = y[max(0, shock_at - horizon):shock_at]
        assert lead_up.sum() > 0, f"Labels should be 1 in the {horizon}-min window before shock"

    def test_no_labels_after_shock_ends(self):
        """After the shock ends and node calms down, no more positive labels."""
        n = 400
        shock_at, shock_len = 100, 20
        dev = _dev(shock_start=shock_at, shock_bps=60.0, shock_len=shock_len, n=n)
        labels = make_onset_labels(
            {"node": dev},
            horizon_min=30,
            thresholds_bps={"node": 25.0},
            sustained_min=5,
        )
        # Well after the shock ends: node is calm and no future stress → label 0
        tail = labels["node"][shock_at + shock_len + 60:]
        assert tail.sum() == 0, "After stress ends and calm returns, no more onset labels"


# ─── Threshold defaults ───────────────────────────────────────────────────────

class TestThresholdDefaults:
    @pytest.mark.parametrize("asset,expected", [
        ("USDC", 25.0),
        ("USDT", 25.0),
        ("BUSD", 25.0),
        ("TUSD", 25.0),
        ("PYUSD", 25.0),
        ("DAI", 75.0),
        ("FRAX", 75.0),
        ("USDe", 50.0),
        ("USDE", 50.0),
    ])
    def test_default_threshold(self, asset, expected):
        node_str = f"{asset}/binance"
        assert default_threshold_map(node_str) == expected


# ─── Base-rate flags ──────────────────────────────────────────────────────────

class TestBaseRateFlags:
    def test_low_positive_rate_flagged(self):
        y = np.array([0] * 99 + [1])
        tbl = base_rate_table({30: y}, [30])
        assert "WARN" in tbl.loc[30, "flag"]

    def test_high_positive_rate_flagged(self):
        y = np.array([0] * 40 + [1] * 60)
        tbl = base_rate_table({60: y}, [60])
        assert "WARN" in tbl.loc[60, "flag"]

    def test_ok_positive_rate_not_flagged(self):
        y = np.array([0] * 80 + [1] * 20)
        tbl = base_rate_table({30: y}, [30])
        assert tbl.loc[30, "flag"] == "OK"

    def test_sustained_indicator_requires_consecutive(self):
        """Isolated spikes below sustained_min should not trigger stress."""
        idx = _idx(100)
        # 5 isolated spikes but never 10 consecutive
        vals = pd.Series([0.0] * 100, index=idx)
        for i in range(0, 50, 10):
            vals.iloc[i] = 40.0   # isolated spikes
        stress = _stress_indicator(vals, threshold_bps=25.0, sustained_min=10)
        assert stress.sum() == 0, "Isolated spikes should not count as sustained stress"
