"""
End-to-end smoke test on synthetic stub data.

Verifies the full pipeline runs without errors before any real data is pulled.
Run via:  python scripts/smoke_test.py
Also run in CI for every push.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scgnn.utils.seeds import set_all_seeds
from scgnn.data.registry import NodeID, NodeRegistry
from scgnn.features.node_features import build_node_feature_matrix
from scgnn.features.labels import make_labels, class_weights
from scgnn.graphs.builder import build_snapshot_graph, build_temporal_graph_sequence
from scgnn.models.baselines import MajorityClassifier, PersistenceClassifier
from scgnn.models.classical import make_logreg, make_xgboost


def _stub_price_series(n: int = 600, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    # Mostly near $1, occasional shock
    prices = 1.0 + rng.normal(0, 0.0005, n)
    prices[300:320] += 0.005   # inject a depeg
    return pd.Series(prices, index=idx)


def main() -> None:
    set_all_seeds(42)

    # --- registry ---
    registry = NodeRegistry.from_config(
        assets=["USDC", "USDT", "DAI"],
        venues=["binance", "coinbase"],
    )
    assert len(registry) == 6, f"expected 6 nodes, got {len(registry)}"

    # --- features ---
    close = _stub_price_series()
    returns = close.pct_change().fillna(0)
    volume = pd.Series(np.abs(np.random.default_rng(1).normal(1e6, 1e5, len(close))), index=close.index)
    feats = build_node_feature_matrix(
        close=close,
        volume=volume,
        returns=returns,
        signed_volume=volume * np.sign(returns),
        venue_prices={"binance": close, "coinbase": close * 1.0001},
    )
    assert feats.shape[0] == len(close)
    print(f"[OK] feature matrix: {feats.shape}")

    # --- labels ---
    dev = (close - 1.0) * 10_000  # bps
    labels = make_labels({"USDC/binance": dev, "USDT/binance": dev * 0.5}, horizons_min=[30, 60])
    assert 30 in labels and 60 in labels
    print(f"[OK] labels: {labels[60].shape}, positive rate @60min: {labels[60].values.mean():.3f}")

    # --- graph ---
    node_ids = list(registry)[:3]
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(2)
    ret_df = pd.DataFrame(
        rng.normal(0, 0.001, (n, len(node_ids))),
        index=idx,
        columns=[str(nd) for nd in node_ids],
    )
    G = build_snapshot_graph(ret_df, node_ids, ret_df.index[-1], corr_threshold=0.0)
    assert len(G.nodes) == 3
    print(f"[OK] snapshot graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # --- baselines ---
    X = np.random.default_rng(3).normal(0, 1, (100, 5))
    y = np.array([0] * 90 + [1] * 10)
    for name, clf in [("majority", MajorityClassifier()), ("logreg", make_logreg())]:
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 100
        print(f"[OK] {name} predictions: {preds.mean():.2f}")

    print("\n[PASS] All smoke tests passed.")


if __name__ == "__main__":
    main()
