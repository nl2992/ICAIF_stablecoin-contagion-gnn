"""
Artifact stamping.

Every output file is stamped with:
  - git SHA (or 'dirty' suffix if uncommitted changes)
  - schema version
  - global seed
  - config hash (SHA-256 of experiment.yaml)
  - timestamp

This makes every artifact traceable to its exact pipeline state.
Reviewers can reproduce results by checking out the stamped commit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def git_sha(short: bool = True) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if short:
            sha = sha[:8]
        # Check for uncommitted changes
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def config_hash(config_path: Path = Path("configs/experiment.yaml")) -> str:
    if not config_path.exists():
        return "unknown"
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def make_stamp(
    schema_version: str = "1",
    seed: int = 42,
    config_path: Path = Path("configs/experiment.yaml"),
    extra: Optional[dict] = None,
) -> dict:
    stamp = {
        "git_sha": git_sha(),
        "schema_version": schema_version,
        "seed": seed,
        "config_hash": config_hash(config_path),
        "produced_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        stamp.update(extra)
    return stamp


def stamp_artifact(artifact_path: Path, stamp: dict) -> None:
    """Write a companion .stamp.json file alongside any output artifact."""
    stamp_path = artifact_path.with_suffix(".stamp.json")
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)


def stamp_dataframe(df: Any, stamp: dict) -> Any:
    """Attach stamp as DataFrame.attrs metadata."""
    df.attrs["stamp"] = stamp
    return df


def load_stamp(artifact_path: Path) -> Optional[dict]:
    stamp_path = artifact_path.with_suffix(".stamp.json")
    if not stamp_path.exists():
        return None
    with open(stamp_path) as f:
        return json.load(f)


def assert_stamp_matches(artifact_path: Path, expected_sha: str) -> None:
    """Raise if the artifact was produced from a different commit."""
    stamp = load_stamp(artifact_path)
    if stamp is None:
        raise FileNotFoundError(f"No stamp file found for {artifact_path}")
    actual = stamp.get("git_sha", "").replace("-dirty", "")
    expected = expected_sha.replace("-dirty", "")
    if not actual.startswith(expected[:8]):
        raise RuntimeError(
            f"Artifact {artifact_path.name} was produced from commit {actual}, "
            f"expected {expected}. Regenerate from the correct commit."
        )
