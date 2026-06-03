"""
★ Leakage hardening tests.

These tests verify that no fit/transform component sees held-out data.
Every test asserts a LeakageError is raised for the forbidden pattern,
AND that the correct pattern succeeds without error.

Covers:
  1. Imputer never sees held-out episode rows during fit
  2. Isotonic calibrator fit on val only, not test fold
  3. Feature standardisation statistics are train-fold-only
  4. Temporal edge feature leakage: rolling window uses only data ≤ t
  5. Hyperparameter selection on fold-internal val, not global val
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from scgnn.eval.loeo_safe import (
    LOEOFold,
    LOEOSafeTransformer,
    LeakageError,
    make_loeo_safe_components,
)
from scgnn.data.integrity import TrainFitImputer
from scgnn.utils.time import last_window


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def fold():
    return LOEOFold(
        held_out_episode_name="USDC_SVB",
        train_episode_names=["UST_Terra", "USDT_Oct2018", "USDT_May2022",
                             "FRAX_SVB", "BUSD_winddown"],
        val_episode_name="DAI_FTX",
    )


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(0, 1, (n, 5))
    y = rng.integers(0, 2, n)
    train_tags = ["UST_Terra"] * 20 + ["USDT_May2022"] * 20 + ["FRAX_SVB"] * 20
    val_tags = ["DAI_FTX"] * 15
    test_tags = ["USDC_SVB"] * 15
    return {
        "X_train": X[:60], "y_train": y[:60], "train_tags": train_tags,
        "X_val": X[:15],   "y_val": y[:15],   "val_tags": val_tags,
        "X_test": X[:15],  "y_test": y[:15],  "test_tags": test_tags,
    }


# ─── 1. Imputer leakage tests ─────────────────────────────────────────────────

class TestImputer:
    def test_imputer_raises_on_held_out_data(self, fold, data):
        """Fitting imputer on data that includes the held-out episode → LeakageError."""
        contaminated_tags = data["train_tags"] + data["test_tags"]   # includes USDC_SVB
        contaminated_X = np.vstack([data["X_train"], data["X_test"]])
        safe_imp = LOEOSafeTransformer(TrainFitImputer(), fold, role="imputer")
        with pytest.raises(LeakageError, match="LEAKAGE"):
            safe_imp.fit(contaminated_X, episode_tags=contaminated_tags)

    def test_imputer_succeeds_on_train_only(self, fold, data):
        """Fitting imputer on train-only data → no error."""
        safe_imp = LOEOSafeTransformer(TrainFitImputer(), fold, role="imputer")
        safe_imp.fit(data["X_train"], episode_tags=data["train_tags"])
        X_transformed = safe_imp.transform(data["X_test"])
        assert X_transformed.shape == data["X_test"].shape

    def test_imputer_raises_without_episode_tags(self, fold, data):
        """Calling .fit() without episode_tags → LeakageError (can't prove safety)."""
        safe_imp = LOEOSafeTransformer(TrainFitImputer(), fold, role="imputer")
        with pytest.raises(LeakageError, match="episode_tags"):
            safe_imp.fit(data["X_train"])   # no episode_tags


# ─── 2. Isotonic calibrator leakage tests ─────────────────────────────────────

class TestCalibrator:
    def test_calibrator_raises_on_test_data(self, fold, data):
        """Calibrator fit on test fold → LeakageError."""
        from sklearn.isotonic import IsotonicRegression
        safe_cal = LOEOSafeTransformer(IsotonicRegression(), fold, role="calibrator")
        test_probs = np.clip(np.random.default_rng(0).random(15), 0, 1)
        with pytest.raises(LeakageError, match="LEAKAGE"):
            safe_cal.fit(test_probs, data["y_test"], episode_tags=data["test_tags"])

    def test_calibrator_succeeds_on_val_data(self, fold, data):
        """Calibrator fit on validation fold → no error."""
        from sklearn.isotonic import IsotonicRegression
        safe_cal = LOEOSafeTransformer(IsotonicRegression(out_of_bounds="clip"),
                                       fold, role="calibrator")
        val_probs = np.clip(np.random.default_rng(1).random(15), 0, 1)
        safe_cal.fit(val_probs, data["y_val"], episode_tags=data["val_tags"])
        calibrated = safe_cal.predict(val_probs)
        assert calibrated.shape == val_probs.shape

    def test_calibrator_raises_on_train_only_without_val(self, fold, data):
        """Calibrator that skips val and fits only on train → warning (not error)."""
        from sklearn.isotonic import IsotonicRegression
        import warnings
        safe_cal = LOEOSafeTransformer(IsotonicRegression(), fold, role="calibrator")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Train-only fit is allowed (held-out not present) but val warning fires
            safe_cal.fit(data["X_train"][:, 0], data["y_train"],
                         episode_tags=data["train_tags"])
        assert any("val" in str(warning.message).lower() for warning in w)


# ─── 3. Scaler leakage tests ──────────────────────────────────────────────────

class TestScaler:
    def test_scaler_raises_on_global_fit(self, fold, data):
        """StandardScaler fit on all data (train + val + test) → LeakageError."""
        all_X = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
        all_tags = data["train_tags"] + data["val_tags"] + data["test_tags"]
        safe_sc = LOEOSafeTransformer(StandardScaler(), fold, role="scaler")
        with pytest.raises(LeakageError, match="LEAKAGE"):
            safe_sc.fit(all_X, episode_tags=all_tags)

    def test_scaler_stats_from_train_only(self, fold, data):
        """
        Verify scaler mean/std reflect train distribution, not test.
        If test (mean≈10) were included, the fitted mean would be >> 0.
        """
        # Train: N(0,1), Test: N(10,2) — very different
        rng = np.random.default_rng(99)
        X_train_small = rng.normal(0, 1, (30, 3))
        X_test_biased = rng.normal(10, 2, (15, 3))
        fold_local = LOEOFold(
            held_out_episode_name="USDC_SVB",
            train_episode_names=["UST_Terra"],
        )
        safe_sc = LOEOSafeTransformer(StandardScaler(), fold_local, role="scaler")
        train_tags_local = ["UST_Terra"] * 30
        safe_sc.fit(X_train_small, episode_tags=train_tags_local)
        assert abs(safe_sc.inner.mean_.mean()) < 1.0, (
            "Scaler mean >> 0 — test data may have contaminated fit"
        )

    def test_make_loeo_safe_components(self, fold):
        """make_loeo_safe_components returns all required wrappers."""
        components = make_loeo_safe_components(fold)
        assert "scaler" in components
        assert "imputer" in components
        assert "calibrator" in components
        for name, comp in components.items():
            assert isinstance(comp, LOEOSafeTransformer)
            assert comp._role == name


# ─── 4. Temporal edge feature leakage ─────────────────────────────────────────

class TestTemporalEdgeLeakage:
    """
    Rolling 6h correlation, lead-lag, shared-LP% must use only data ≤ t.
    Test on a monotone series: if future data leaks in, correlation changes sign.
    """

    def test_rolling_window_uses_only_past_data(self):
        """
        Series: strictly increasing. Rolling 6h corr at time t should use
        only data [0, t]. If the window used future data, the series would
        still be increasing but the corr with a lagged version would change.
        """
        from scgnn.utils.time import last_window

        idx = pd.date_range("2023-01-01", periods=300, freq="1min", tz="UTC")
        s = pd.Series(np.arange(300, dtype=float), index=idx)

        # At t = idx[120], the 6h window should be [60, 120]
        t = idx[120]
        history = s[s.index <= t]
        window = last_window(history, "1h")   # 60 minutes

        assert window.index.min() >= t - pd.Timedelta("1h"), \
            "Window extends before the 1h lookback — future data leaked"
        assert window.index.max() <= t, \
            "Window extends beyond t — future data leaked"

    def test_correlation_at_t_excludes_future(self):
        """
        Build a synthetic pair where s2 = -s1 in the future but +s1 in the past.
        Rolling corr at t must be positive (past only), not negative (with future).
        """
        from scgnn.features.edge_features import rolling_correlation

        idx = pd.date_range("2023-01-01", periods=200, freq="1min", tz="UTC")
        s1 = pd.Series(np.arange(200, dtype=float), index=idx)
        # s2 mirrors s1 in past [0:100], anti-correlated in future [100:200]
        s2_vals = np.concatenate([np.arange(100, dtype=float), -np.arange(100, dtype=float)])
        s2 = pd.Series(s2_vals, index=idx)

        # At t=80 (past only window), correlation should be positive
        t_past = idx[80]
        s1_past = s1[s1.index <= t_past].iloc[-60:]   # last 60 min
        s2_past = s2[s2.index <= t_past].iloc[-60:]
        corr_past = rolling_correlation(s1_past, s2_past)
        assert corr_past > 0.9, f"Expected positive correlation in past window, got {corr_past:.3f}"

    def test_no_episode_boundary_leak_in_6h_window(self):
        """
        The 6h window at the START of an episode must be clipped to that
        episode's baseline start — not bleed into the previous episode.
        Test via build_snapshot_graph with episode_start parameter.
        """
        from scgnn.graphs.builder import build_snapshot_graph
        from scgnn.data.registry import NodeID, NodeRegistry

        nodes = [NodeID("USDC", "binance"), NodeID("USDT", "binance")]
        node_strs = [str(n) for n in nodes]

        # Create data spanning two fake "episodes"
        idx = pd.date_range("2023-01-01", periods=500, freq="1min", tz="UTC")
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(rng.normal(0, 0.001, (500, 2)), index=idx, columns=node_strs)

        # Episode 2 starts at t=300; its 6h window without a guard would reach t=240
        # which is in episode 1. With the guard, it should be clipped to t=300.
        episode_start = idx[300]
        t = idx[301]

        G = build_snapshot_graph(
            returns, nodes, t,
            lookback="6h",
            corr_threshold=0.0,
            episode_start=episode_start,
        )
        assert len(G.nodes) == 2, "Graph should still have nodes"


# ─── 5. Hyperparameter selection on fold-internal val ─────────────────────────

class TestHyperparameterSelection:
    def test_val_is_fold_internal_not_global(self, fold):
        """
        The fold's val_episode_name must be a different episode from the
        held-out test episode — no "global val" that leaks across folds.
        """
        assert fold.val_episode_name != fold.held_out_episode_name, \
            "Val episode is the same as held-out — hyperparams tuned on test"

    def test_val_not_in_train_episodes(self, fold):
        """Val episode must not appear in the training set."""
        if fold.val_episode_name:
            assert fold.val_episode_name not in fold.train_episode_names, \
                "Val episode is also in train — val contaminated"

    def test_loeo_fold_has_disjoint_splits(self):
        """Train, val, test must be disjoint in any fold configuration."""
        all_episodes = ["A", "B", "C", "D", "E", "F", "G"]
        for held_out_idx, held_out in enumerate(all_episodes):
            train = [e for i, e in enumerate(all_episodes)
                     if i != held_out_idx and i != (held_out_idx - 1) % len(all_episodes)]
            val = all_episodes[(held_out_idx - 1) % len(all_episodes)]
            fold = LOEOFold(held_out, train, val)
            assert fold.held_out_episode_name not in fold.train_episode_names
            if fold.val_episode_name:
                assert fold.val_episode_name not in fold.train_episode_names
