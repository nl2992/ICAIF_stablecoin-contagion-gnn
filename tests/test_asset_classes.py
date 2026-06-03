"""
Tests for asset→class mapping, threshold governance, and UST delisting fix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scgnn.features.asset_classes import (
    get_asset_class,
    threshold_for_node,
    threshold_map_for_nodes,
    uniform_threshold_map,
    relative_threshold_map,
    ASSET_CLASSES,
)
from scgnn.data.integrity import (
    is_delisting_artifact,
    classify_price_trajectory,
    ASSET_STATUS_REGISTRY,
)


# ─── Asset class mapping ──────────────────────────────────────────────────────

class TestAssetClassMapping:
    @pytest.mark.parametrize("asset,expected_class,expected_bps", [
        ("USDC",  "fiat_backed",   25.0),
        ("USDT",  "fiat_backed",   25.0),
        ("TUSD",  "fiat_backed",   25.0),
        ("PYUSD", "fiat_backed",   25.0),
        ("BUSD",  "fiat_backed",   25.0),
        ("DAI",   "crypto_backed", 75.0),
        ("FRAX",  "crypto_backed", 75.0),
        ("USDE",  "synthetic",     50.0),
    ])
    def test_registered_assets(self, asset, expected_class, expected_bps):
        cls = get_asset_class(asset)
        assert cls.name == expected_class
        assert cls.threshold_bps == expected_bps

    def test_unknown_asset_defaults_to_fiat_backed(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cls = get_asset_class("XYZSTABLE")
        assert cls.name == "fiat_backed"
        assert any("Unknown asset" in str(warning.message) for warning in w)

    def test_threshold_for_node_str(self):
        assert threshold_for_node("USDC/binance") == 25.0
        assert threshold_for_node("DAI/coinbase") == 75.0
        assert threshold_for_node("USDe/curve/3pool") == 50.0

    def test_threshold_map_for_nodes(self):
        nodes = ["USDC/binance", "DAI/coinbase", "FRAX/curve"]
        tmap = threshold_map_for_nodes(nodes)
        assert tmap["USDC/binance"] == 25.0
        assert tmap["DAI/coinbase"] == 75.0
        assert tmap["FRAX/curve"] == 75.0


# ─── Uniform and relative threshold arms ─────────────────────────────────────

class TestThresholdArms:
    def test_uniform_threshold_all_same(self):
        nodes = ["USDC/binance", "DAI/coinbase", "USDe/curve"]
        tmap = uniform_threshold_map(nodes, bps=30.0)
        assert all(v == 30.0 for v in tmap.values())

    def test_relative_threshold_shifts_per_class(self):
        nodes = ["USDC/binance", "DAI/coinbase", "USDe/curve"]
        tmap = relative_threshold_map(nodes, delta_bps=+15.0)
        assert tmap["USDC/binance"] == 40.0   # 25 + 15
        assert tmap["DAI/coinbase"] == 90.0   # 75 + 15
        assert tmap["USDe/curve"] == 65.0     # 50 + 15

    def test_relative_threshold_brackets_crypto(self):
        """The relative sweep DOES bracket the 75 bps crypto threshold, unlike {10,25,50}."""
        nodes = ["DAI/coinbase"]
        lo = relative_threshold_map(nodes, -25.0)["DAI/coinbase"]
        hi = relative_threshold_map(nodes, +25.0)["DAI/coinbase"]
        assert lo < 75.0 < hi, "Relative sweep should bracket the 75 bps crypto threshold"

    def test_relative_threshold_floor_at_1(self):
        nodes = ["USDC/binance"]
        tmap = relative_threshold_map(nodes, delta_bps=-100.0)
        assert tmap["USDC/binance"] >= 1.0, "Threshold must not go below 1 bps"


# ─── UST delisting artifact distinction ──────────────────────────────────────

class TestDelistingArtifact:
    def _ust_crisis_prices(self) -> pd.Series:
        """UST May 2022: rapid collapse from $1 to $0.01 over ~5 days."""
        idx = pd.date_range("2022-05-07", periods=7200, freq="1min", tz="UTC")
        # Days 0-2: stable; days 2-5: rapid collapse
        prices = np.ones(7200)
        prices[2880:] = np.linspace(1.0, 0.01, 7200 - 2880)
        return pd.Series(prices, index=idx)

    def _busd_winddown_prices(self) -> pd.Series:
        """BUSD post-Feb 2023: stays at $1 then volume dries up → 0 after ~40 days."""
        n = 58_000   # ~40 days in minutes; extends well past 30-day grace period
        idx = pd.date_range("2023-02-13", periods=n, freq="1min", tz="UTC")
        prices = np.ones(n)
        # Price goes to zero after 35 days (50 400 min) — past the 30-day grace
        prices[50_400:] = 0.0
        return pd.Series(prices, index=idx)

    def test_ust_crisis_not_flagged_as_artifact(self):
        """UST's rapid May 2022 collapse is a genuine crisis, not a delist artifact."""
        prices = self._ust_crisis_prices()
        # The crisis window is BEFORE the delist date grace period ends
        # → should NOT be flagged as artifact
        result = is_delisting_artifact(prices, "UST", wind_down_window_days=30)
        assert result is False, \
            "UST terminal collapse should NOT be flagged as delisting artifact"

    def test_busd_winddown_flagged_after_grace(self):
        """BUSD gradual wind-down (price=0 30+ days after delist) IS an artifact."""
        prices = self._busd_winddown_prices()
        result = is_delisting_artifact(prices, "BUSD", wind_down_window_days=30)
        assert result is True, "BUSD slow wind-down to zero should be flagged as artifact"

    def test_classify_ust_as_genuine_crisis(self):
        prices = self._ust_crisis_prices()
        traj = classify_price_trajectory(prices, "UST")
        assert traj == "genuine_crisis"

    def test_classify_stable_as_stable(self):
        idx = pd.date_range("2023-01-01", periods=500, freq="1min", tz="UTC")
        # noise std of 0.00003 → max deviation ≈ 0.3 bps → well below 5 bps stable threshold
        prices = pd.Series(1.0 + np.random.default_rng(0).normal(0, 0.00003, 500), index=idx)
        traj = classify_price_trajectory(prices, "USDC")
        assert traj == "stable"

    def test_active_asset_never_flagged(self):
        idx = pd.date_range("2023-03-10", periods=1000, freq="1min", tz="UTC")
        prices = pd.Series(1.0 + np.random.default_rng(1).normal(0, 0.001, 1000), index=idx)
        assert not is_delisting_artifact(prices, "USDC")
        assert not is_delisting_artifact(prices, "DAI")
