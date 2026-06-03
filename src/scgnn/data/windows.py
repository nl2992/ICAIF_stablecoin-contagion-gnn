"""
Episode window builder.

For each episode, assembles a DataFrame of 1-min prices with:
  - a pre-event baseline (default 7 days of calm)
  - the stress window itself
  - coverage metadata

The window is the atomic unit fed into feature/label construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    name: str
    trigger: str
    trigger_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    split: str              # train | val | test
    notes: str = ""

    @property
    def baseline_start(self) -> pd.Timestamp:
        return self.start - pd.Timedelta(days=7)

    @property
    def full_window_start(self) -> pd.Timestamp:
        return self.baseline_start

    @property
    def duration_stress_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600


def load_episodes(path: Path) -> List[Episode]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    episodes = []
    for item in raw.get("episodes", []):
        episodes.append(Episode(
            name=item["name"],
            trigger=item["trigger"],
            trigger_type=item["trigger_type"],
            start=pd.Timestamp(item["start"], tz="UTC"),
            end=pd.Timestamp(item["end"], tz="UTC"),
            split=item["split"],
            notes=item.get("notes", ""),
        ))
    return episodes


def episodes_by_split(episodes: List[Episode]) -> Dict[str, List[Episode]]:
    out: Dict[str, List[Episode]] = {"train": [], "val": [], "test": []}
    for ep in episodes:
        out[ep.split].append(ep)
    return out


def build_episode_window(
    episode: Episode,
    price_grid: pd.DataFrame,
    min_coverage_pct: float = 80.0,
) -> Tuple[pd.DataFrame, float]:
    """
    Slice the pre-aligned price grid to the episode's full window
    (baseline + stress).  Returns (window_df, coverage_pct).
    """
    window = price_grid.loc[episode.full_window_start:episode.end].copy()
    coverage = float((window.notna().mean(axis=None)) * 100)
    if coverage < min_coverage_pct:
        logger.warning(
            "Episode %s: coverage %.1f%% < %.1f%% threshold",
            episode.name, coverage, min_coverage_pct,
        )
    return window, coverage


def compute_availability_matrix(
    episodes: List[Episode],
    price_grids: Dict[str, pd.DataFrame],   # {episode_name: aligned_price_df}
    min_coverage_pct: float = 80.0,
) -> pd.DataFrame:
    """
    Return a DataFrame of shape (episodes × nodes) with coverage percentages.
    Gate: episode × node cell must be ≥ min_coverage_pct to be usable.
    """
    rows = []
    for ep in episodes:
        if ep.name not in price_grids:
            continue
        grid = price_grids[ep.name]
        _, _ = build_episode_window(ep, grid, min_coverage_pct)
        window = grid.loc[ep.full_window_start:ep.end]
        row = {"episode": ep.name, "split": ep.split}
        for col in grid.columns:
            row[col] = float(window[col].notna().mean() * 100)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("episode")
    return df


def save_availability_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    # Also save a human-readable flag version
    flag_path = path.with_suffix(".flags.csv")
    min_cov = 80.0
    flag_df = df.copy()
    for col in df.select_dtypes("number").columns:
        flag_df[col] = df[col].apply(lambda v: "OK" if v >= min_cov else f"LOW:{v:.0f}%")
    flag_df.to_csv(flag_path)
    logger.info("Availability matrix saved to %s", path)
