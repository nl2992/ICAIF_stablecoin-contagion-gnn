"""
Export schema v1 — the stable interface between Repo 1 (GNN) and Repo 2 (ABM).

Two artifact types:

  1. hub_ranking  — one row per node per episode, consumed by ABM to set
                    contagion-shock probability proportional to hub_score.

  2. calibration  — one row per episode × asset, consumed by ABM to set
                    OU parameters and shock amplitudes.

Schema is versioned. Breaking changes require a new SCHEMA_VERSION.
Non-breaking additions (new optional columns) are allowed within a version.

Both artifacts are written as CSV (primary) + JSON sidecar (metadata).
The ABM reads the CSV; the JSON is for lineage/audit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


SCHEMA_VERSION = "1"

# ------------------------------------------------------------------
# Hub-ranking schema
# ------------------------------------------------------------------

HUB_RANKING_COLUMNS = [
    # Identity
    "node",           # str  — "USDC/binance" or "USDC/curve/3pool"
    "asset",          # str  — e.g. "USDC"
    "venue",          # str  — e.g. "binance"
    "fee_tier",       # str  — e.g. "500" or "" for CEX
    # Ranking
    "rank",           # int  — 1 = highest hub score
    "hub_score",      # float — composite score in [0, 1]
    "ci_lo",          # float — 2.5th percentile across seeds
    "ci_hi",          # float — 97.5th percentile across seeds
    "ci_std",         # float — std dev across seeds
    # Components
    "gnn_mask_sum",   # float — total GNNExplainer node-mask weight
    "betweenness",    # float — betweenness centrality in stress graph
    "propagator_label",  # int — 1 if node was stress propagator in this episode
    "norm_gnn",       # float — normalized gnn_mask_sum
    "norm_bc_x_prop", # float — normalized betweenness × propagator
    # Provenance
    "episode_tag",    # str  — e.g. "USDC_SVB" or "synthetic_001"
    "is_real",        # bool — True for real historical episodes
]

# Columns the ABM *must* have (subset of above)
HUB_RANKING_REQUIRED = ["node", "asset", "venue", "rank", "hub_score", "ci_lo", "ci_hi", "episode_tag", "is_real"]

# ------------------------------------------------------------------
# Calibration schema
# ------------------------------------------------------------------

CALIBRATION_COLUMNS = [
    # Identity
    "episode",        # str  — episode name
    "asset",          # str  — asset experiencing the shock
    "venue",          # str  — venue (if venue-specific; else "all")
    # OU parameters (for ABM mean-reversion dynamics)
    "ou_half_life_min",     # float — mean-reversion half-life in minutes
    "ou_half_life_ci_lo",   # float
    "ou_half_life_ci_hi",   # float
    # Propagation
    "propagation_rho",      # float — empirical fraction of nodes that were stressed
    "propagation_lag_min",  # float — median lead-lag to first contagion node (minutes)
    # Shock amplitude
    "peak_depeg_bps",       # float — maximum |price − 1| in basis points during episode
    "shock_duration_min",   # float — total minutes the trigger was stressed
    # Provenance
    "episode_tag",          # str
    "is_real",              # bool
]

CALIBRATION_REQUIRED = [
    "episode", "asset", "ou_half_life_min", "propagation_rho",
    "peak_depeg_bps", "episode_tag", "is_real",
]


def validate_hub_ranking(df) -> List[str]:
    """Return list of missing required columns."""
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return ["not a DataFrame"]
    return [c for c in HUB_RANKING_REQUIRED if c not in df.columns]


def validate_calibration(df) -> List[str]:
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return ["not a DataFrame"]
    return [c for c in CALIBRATION_REQUIRED if c not in df.columns]


def write_schema_doc(out_dir: Path = Path("exports")) -> Path:
    """Write a human-readable schema document for Repo 2 consumers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Stable export interface between stablecoin-contagion-gnn (Repo 1) "
            "and the ABM (Repo 2). Breaking changes increment schema_version."
        ),
        "artifacts": {
            "hub_ranking": {
                "filename_pattern": f"hub_ranking_v{SCHEMA_VERSION}_{{episode_tag}}.csv",
                "required_columns": HUB_RANKING_REQUIRED,
                "all_columns": HUB_RANKING_COLUMNS,
                "notes": (
                    "The ABM uses hub_score (or ci_lo for conservative scenarios) "
                    "to set per-node shock propagation probability. "
                    "Filter is_real=True for strongest evidence."
                ),
            },
            "calibration": {
                "filename_pattern": f"calibration_v{SCHEMA_VERSION}_{{episode_tag}}.csv",
                "required_columns": CALIBRATION_REQUIRED,
                "all_columns": CALIBRATION_COLUMNS,
                "notes": (
                    "Provides OU half-life, propagation rho, and peak depeg spread "
                    "as empirical calibration targets for the ABM's stress dynamics."
                ),
            },
        },
        "node_id_convention": {
            "format": "asset/venue[/fee_tier]",
            "examples": ["USDC/binance", "USDT/curve/3pool", "DAI/uniswap_v3/500"],
            "note": "fee_tier is empty string for CEX nodes",
        },
    }
    path = out_dir / f"schema_v{SCHEMA_VERSION}.json"
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Schema doc written: {path}")
    return path
