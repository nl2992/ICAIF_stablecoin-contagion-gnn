"""
Tests confirming propagator labels are computed from raw price data only —
not from model output (non-circular).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scgnn.hub.ranking import compute_propagator_labels


def _dev_series(n=300, shock_start=50, shock_bps=60.0, shock_len=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-03-10", periods=n, freq="1min", tz="UTC")
    dev = pd.Series(rng.normal(0, 2, n), index=idx)
    dev.iloc[shock_start:shock_start + shock_len] += shock_bps
    return dev


class TestPropagatorLabels:
    def test_origin_always_zero(self):
        origin = "USDC/binance"
        peg_devs = {origin: _dev_series(shock_bps=80.0)}
        labels = compute_propagator_labels(peg_devs, origin, {origin: 25.0})
        assert labels[origin] == 0

    def test_pre_existing_stress_excluded(self):
        """Node stressed BEFORE origin onset should get 0 (pre-existing, not contagion)."""
        origin = "USDC/binance"
        victim = "USDT/binance"
        # Origin shocks at t=100; victim was ALREADY stressed from t=0
        origin_dev = _dev_series(shock_start=100, shock_bps=80.0, n=300, seed=1)
        victim_dev = _dev_series(shock_start=5, shock_bps=60.0, n=300, seed=2)
        peg_devs = {origin: origin_dev, victim: victim_dev}
        labels = compute_propagator_labels(peg_devs, origin, {origin: 25.0, victim: 25.0}, sustained_min=5)
        # Victim was stressed before origin → should be excluded
        assert labels[victim] == 0

    def test_contagion_within_window_labeled_1(self):
        """Node that enters stress after origin and within window → 1."""
        origin = "USDC/binance"
        victim = "DAI/coinbase"
        # Origin shocks at t=50; victim shocks at t=80 (30 min later)
        origin_dev = _dev_series(shock_start=50, shock_bps=80.0, n=300, seed=3)
        victim_dev = _dev_series(shock_start=80, shock_bps=80.0, n=300, seed=4)
        peg_devs = {origin: origin_dev, victim: victim_dev}
        labels = compute_propagator_labels(
            peg_devs, origin,
            {origin: 25.0, victim: 75.0},   # DAI has 75 bps threshold
            sustained_min=5,
            propagation_window_min=1440,
        )
        assert labels[victim] == 1

    def test_victim_outside_window_labeled_0(self):
        """Node that enters stress 25h after origin → outside 24h window → 0."""
        origin = "USDC/binance"
        victim = "USDT/binance"
        # Origin at t=0, victim at t=0+25h=1500 min — well outside window
        n = 2000
        origin_dev = _dev_series(shock_start=5, shock_bps=80.0, n=n, seed=5)
        victim_dev = _dev_series(shock_start=1500, shock_bps=60.0, n=n, seed=6)
        peg_devs = {origin: origin_dev, victim: victim_dev}
        labels = compute_propagator_labels(
            peg_devs, origin,
            {origin: 25.0, victim: 25.0},
            sustained_min=5,
            propagation_window_min=1440,   # 24h = 1440 min
        )
        assert labels[victim] == 0

    def test_no_model_output_in_computation(self):
        """
        Structural test: compute_propagator_labels takes only price data as input.
        There is no parameter for model predictions — non-circularity enforced by API.
        """
        import inspect
        sig = inspect.signature(compute_propagator_labels)
        param_names = list(sig.parameters.keys())
        # Must not accept 'model', 'predictions', 'probs', 'hub_scores'
        forbidden = {"model", "predictions", "probs", "hub_scores", "y_pred"}
        assert not (set(param_names) & forbidden), \
            f"Propagator label function should not accept model output: {set(param_names) & forbidden}"
