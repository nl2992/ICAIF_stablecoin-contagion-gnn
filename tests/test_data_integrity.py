"""Tests for data integrity checks: dead assets, 2018 features, missingness, imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scgnn.data.integrity import (
    is_delisting_artifact,
    get_available_features,
    TrainFitImputer,
    PRE_DEFI_CUTOFF,
    DEFI_ERA_FEATURES,
    DELISTED_ASSETS,
)
from scgnn.data.active_nodes import compute_active_mask
from scgnn.data.registry import NodeID, NodeRegistry


# ─── Dead asset detection ─────────────────────────────────────────────────────

class TestDeadAssets:
    def test_busd_detected_after_delist(self):
        # Series must extend past the 30-day grace period (43 200 min + buffer)
        idx = pd.date_range("2023-02-13", periods=50_000, freq="1min", tz="UTC")
        # Price stays at $1 during grace, then zeros after ~32 days
        prices = pd.Series([1.0] * 46_000 + [0.0] * 4_000, index=idx)
        assert is_delisting_artifact(prices, "BUSD")

    def test_active_asset_not_flagged(self):
        idx = pd.date_range("2023-03-10", periods=200, freq="1min", tz="UTC")
        prices = pd.Series(1.0 + np.random.default_rng(0).normal(0, 0.001, 200), index=idx)
        assert not is_delisting_artifact(prices, "USDC")

    def test_delisted_assets_set_contains_expected(self):
        assert "BUSD" in DELISTED_ASSETS
        assert "UST" in DELISTED_ASSETS
        assert "USDC" not in DELISTED_ASSETS


# ─── 2018 feature support ─────────────────────────────────────────────────────

class TestFeatureSupport:
    ALL_FEATURES = ["price_ratio", "rvol_1h", "tvl_usd_log", "lop_wedge",
                    "shared_lp_pct", "kyle_lambda", "ou_half_life"]

    def test_pre_defi_excludes_defi_features(self):
        pre_defi_start = pd.Timestamp("2018-10-14", tz="UTC")
        available = get_available_features(pre_defi_start, self.ALL_FEATURES)
        for f in DEFI_ERA_FEATURES:
            assert not any(f in a for a in available), f"{f} should be excluded pre-DeFi"

    def test_pre_defi_keeps_cex_features(self):
        pre_defi_start = pd.Timestamp("2018-10-14", tz="UTC")
        available = get_available_features(pre_defi_start, self.ALL_FEATURES)
        assert "price_ratio" in available
        assert "rvol_1h" in available

    def test_defi_era_has_all_features(self):
        defi_start = pd.Timestamp("2023-03-10", tz="UTC")
        available = get_available_features(defi_start, self.ALL_FEATURES)
        assert set(available) == set(self.ALL_FEATURES)

    def test_cutoff_date_is_2019(self):
        assert PRE_DEFI_CUTOFF.year == 2019


# ─── Imputation (no leakage) ──────────────────────────────────────────────────

class TestImputer:
    def test_fit_on_train_only(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, (100, 5))
        X_test = rng.normal(10, 2, (20, 5))
        X_test[0, 0] = np.nan

        imp = TrainFitImputer()
        imp.fit(X_train)
        X_test_imp = imp.transform(X_test)

        # Imputed value should be close to train median (~0), not test median (~10)
        assert abs(X_test_imp[0, 0]) < 2.0

    def test_structural_zero_filled_with_zero(self):
        X_train = np.array([[np.nan, 1.0], [np.nan, 2.0]])
        X_test = np.array([[np.nan, 3.0]])
        imp = TrainFitImputer()
        imp.fit(X_train, structural_zero_cols=[0])
        X_imp = imp.transform(X_test)
        assert X_imp[0, 0] == 0.0

    def test_no_nan_after_imputation(self):
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 10))
        X[rng.random((50, 10)) < 0.2] = np.nan
        imp = TrainFitImputer()
        X_imp = imp.fit_transform(X)
        assert not np.isnan(X_imp).any()


# ─── Active node filter ───────────────────────────────────────────────────────

class TestActiveNodes:
    def _make_registry(self):
        return NodeRegistry.from_config(["USDC", "BUSD"], ["binance"])

    def test_dead_asset_inactive(self):
        reg = self._make_registry()
        idx = pd.date_range("2023-03-10", periods=100, freq="1min", tz="UTC")
        # USDC active, BUSD dead (delist date 2023-02-13 < 2023-03-10)
        window = pd.DataFrame({
            "USDC/binance": 1.0 + np.random.default_rng(0).normal(0, 0.001, 100),
            "BUSD/binance": np.zeros(100),
        }, index=idx)
        ep_start = pd.Timestamp("2023-03-10", tz="UTC")
        mask = compute_active_mask(window, ep_start, reg)
        assert mask["USDC/binance"] is True
        assert mask["BUSD/binance"] is False

    def test_low_coverage_inactive(self):
        reg = NodeRegistry.from_config(["USDC"], ["binance"])
        idx = pd.date_range("2023-03-10", periods=100, freq="1min", tz="UTC")
        prices = pd.Series([1.0] * 10 + [np.nan] * 90, index=idx)
        window = pd.DataFrame({"USDC/binance": prices})
        ep_start = pd.Timestamp("2023-03-10", tz="UTC")
        mask = compute_active_mask(window, ep_start, reg, min_coverage_pct=80.0)
        assert mask["USDC/binance"] is False

    def test_active_node_passes(self):
        reg = NodeRegistry.from_config(["USDC"], ["binance"])
        idx = pd.date_range("2023-03-10", periods=200, freq="1min", tz="UTC")
        prices = pd.Series(1.0 + np.random.default_rng(0).normal(0, 0.001, 200), index=idx)
        window = pd.DataFrame({"USDC/binance": prices})
        ep_start = pd.Timestamp("2023-03-10", tz="UTC")
        mask = compute_active_mask(window, ep_start, reg)
        assert mask["USDC/binance"] is True
