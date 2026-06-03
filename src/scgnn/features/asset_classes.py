"""
Asset→class mapping table (pre-registered).

Edge cases documented:
  FRAX:  post-March 2023 FRAX V3 moved toward fully-backed, but during the
         FRAX_SVB episode (2023-03-11) it was still 80% USDC-backed; for ALL
         episodes we classify as crypto_backed (75 bps) because its peg
         mechanics relied on collateral, not pure fiat.
  USDe:  Ethena's synthetic dollar (delta-neutral strategy); classified as
         synthetic (50 bps) — tighter than crypto_backed because it aims for
         a hard peg via funding arbitrage, looser than fiat_backed because
         it has no fiat reserve.
  PYUSD: PayPal USD, 1:1 fiat-backed by Paxos; classified as fiat_backed (25 bps).
  BUSD:  Binance USD, 1:1 fiat-backed by Paxos; classified as fiat_backed (25 bps).
         Note: delisted after 2023-02-13; active nodes filter handles this.
  UST:   Algorithmic; NOT in node universe (collapsed); handled by is_delisting_artifact.
  USDT:  Despite periodic reserve concerns, classified as fiat_backed (25 bps) —
         this is the most conservative assignment and brackets all known events.

UNIFORM THRESHOLD ARM:
  For the robustness check, a single threshold of 25 bps is applied to ALL
  asset classes.  This tests whether per-class differentiation materially
  changes the hub ranking (if hubs are stable, the per-class design is
  defensible; if they flip, the classification is load-bearing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class AssetClass:
    name: str
    threshold_bps: float
    rationale: str


ASSET_CLASSES: Dict[str, AssetClass] = {
    # Fiat-backed (25 bps)
    "USDC":  AssetClass("fiat_backed", 25.0, "Circle USD, 1:1 fiat reserve"),
    "USDT":  AssetClass("fiat_backed", 25.0, "Tether USD, classified conservative"),
    "TUSD":  AssetClass("fiat_backed", 25.0, "TrueUSD, 1:1 fiat reserve"),
    "PYUSD": AssetClass("fiat_backed", 25.0, "PayPal USD, 1:1 fiat reserve (Paxos)"),
    "BUSD":  AssetClass("fiat_backed", 25.0, "Binance USD, 1:1 fiat reserve (Paxos); delisted 2023-02-13"),
    # Crypto-backed (75 bps)
    "DAI":   AssetClass("crypto_backed", 75.0, "MakerDAO, over-collateralised crypto + PSM"),
    "FRAX":  AssetClass("crypto_backed", 75.0, "FRAX v1/v2, partial algorithmic; FRAX v3 post-2023 trending fiat-backed but classified conservative for all episodes"),
    # Synthetic (50 bps)
    "USDE":  AssetClass("synthetic", 50.0, "Ethena USDe, delta-neutral synthetic"),
    "USDE_":  AssetClass("synthetic", 50.0, "alias"),   # handle both casings
}

# Canonical uppercase key lookup
def _normalise(asset: str) -> str:
    return asset.upper().rstrip("_")


def get_asset_class(asset: str) -> AssetClass:
    key = _normalise(asset)
    cls = ASSET_CLASSES.get(key)
    if cls is None:
        # Default to fiat_backed (most conservative) with a warning
        import warnings
        warnings.warn(
            f"Unknown asset '{asset}' — defaulting to fiat_backed (25 bps). "
            "Add it to ASSET_CLASSES in asset_classes.py.",
            stacklevel=2,
        )
        return AssetClass("fiat_backed", 25.0, "unknown asset — conservative default")
    return cls


def threshold_for_node(node_str: str) -> float:
    """Return the pre-registered threshold for a node string like 'USDC/binance'."""
    asset = node_str.split("/")[0]
    return get_asset_class(asset).threshold_bps


def threshold_map_for_nodes(node_strs: list[str]) -> Dict[str, float]:
    """Build a {node_str: threshold_bps} dict for a list of node strings."""
    return {ns: threshold_for_node(ns) for ns in node_strs}


def uniform_threshold_map(node_strs: list[str], bps: float = 25.0) -> Dict[str, float]:
    """
    Single-uniform-threshold arm: apply the same threshold to all nodes.
    Used in the robustness sweep alongside the per-class design.
    """
    return {ns: bps for ns in node_strs}


def relative_threshold_map(
    node_strs: list[str],
    delta_bps: float,
) -> Dict[str, float]:
    """
    Relative sensitivity arm: shift each class threshold by ±delta_bps.
    This brackets the pre-registered values without using absolute {10,25,50}
    which don't bracket the 75 bps crypto-backed threshold.

    E.g. delta_bps = -10 → fiat: 15, synthetic: 40, crypto: 65
         delta_bps = +15 → fiat: 40, synthetic: 65, crypto: 90
    """
    return {ns: max(1.0, threshold_for_node(ns) + delta_bps) for ns in node_strs}
