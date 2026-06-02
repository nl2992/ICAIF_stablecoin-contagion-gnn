"""Episode definitions: real historical stress events + synthetic augmentation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class Episode:
    name: str
    trigger: str                # asset that first depeg'd
    start: pd.Timestamp
    end: pd.Timestamp
    synthetic: bool = False
    seed: Optional[int] = None

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600


REAL_EPISODES: List[Episode] = [
    Episode("UST_Terra",   "UST",  pd.Timestamp("2022-05-07", tz="UTC"), pd.Timestamp("2022-05-13", tz="UTC")),
    Episode("USDC_SVB",    "USDC", pd.Timestamp("2023-03-10", tz="UTC"), pd.Timestamp("2023-03-15", tz="UTC")),
    Episode("USDT_Oct18",  "USDT", pd.Timestamp("2018-10-14", tz="UTC"), pd.Timestamp("2018-10-16", tz="UTC")),
    Episode("USDT_May22",  "USDT", pd.Timestamp("2022-05-12", tz="UTC"), pd.Timestamp("2022-05-14", tz="UTC")),
    Episode("FRAX_SVB",    "FRAX", pd.Timestamp("2023-03-11", tz="UTC"), pd.Timestamp("2023-03-16", tz="UTC")),
    Episode("BUSD_wind",   "BUSD", pd.Timestamp("2023-02-13", tz="UTC"), pd.Timestamp("2023-02-17", tz="UTC")),
    Episode("DAI_crisis",  "DAI",  pd.Timestamp("2022-11-09", tz="UTC"), pd.Timestamp("2022-11-14", tz="UTC")),
]


def chronological_split(
    episodes: List[Episode],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[List[Episode], List[Episode], List[Episode]]:
    episodes = sorted(episodes, key=lambda e: e.start)
    n = len(episodes)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        episodes[:n_train],
        episodes[n_train:n_train + n_val],
        episodes[n_train + n_val:],
    )


class SyntheticEpisodeGenerator:
    """
    Bootstrap synthetic stress episodes by block-resampling 1-min price windows
    and injecting a depeg shock. Used to escape n=1 for rare events.
    """

    def __init__(
        self,
        price_data: dict[str, pd.Series],
        n_episodes: int = 200,
        window_hours: int = 72,
        shock_bps_range: tuple[int, int] = (10, 150),
        seed: int = 42,
    ):
        self.price_data = price_data
        self.n_episodes = n_episodes
        self.window_hours = window_hours
        self.shock_bps_range = shock_bps_range
        self.rng = np.random.default_rng(seed)

    def generate(self) -> List[Episode]:
        assets = list(self.price_data.keys())
        episodes = []
        for i in range(self.n_episodes):
            trigger = self.rng.choice(assets)
            seed_val = int(self.rng.integers(0, 2**31))
            # Placeholder timestamps — actual windows assigned at graph-build time
            episodes.append(Episode(
                name=f"synthetic_{i:04d}",
                trigger=trigger,
                start=pd.Timestamp("2020-01-01", tz="UTC"),
                end=pd.Timestamp("2020-01-04", tz="UTC"),
                synthetic=True,
                seed=seed_val,
            ))
        return episodes


def load_episode_manifest(path: Path) -> List[Episode]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    episodes = []
    for item in raw:
        episodes.append(Episode(
            name=item["name"],
            trigger=item["trigger"],
            start=pd.Timestamp(item["start"], tz="UTC"),
            end=pd.Timestamp(item["end"], tz="UTC"),
            synthetic=item.get("synthetic", False),
        ))
    return episodes


def save_episode_manifest(episodes: List[Episode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "name": e.name,
            "trigger": e.trigger,
            "start": str(e.start.date()),
            "end": str(e.end.date()),
            "synthetic": e.synthetic,
        }
        for e in episodes
    ]
    with open(path, "w") as f:
        yaml.safe_dump(records, f, default_flow_style=False)
