"""Tests for export schema validation and calibration."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scgnn.export.schema import (
    validate_hub_ranking,
    validate_calibration,
    write_schema_doc,
    HUB_RANKING_REQUIRED,
    CALIBRATION_REQUIRED,
    SCHEMA_VERSION,
)


def test_schema_version_is_string():
    assert isinstance(SCHEMA_VERSION, str)
    assert len(SCHEMA_VERSION) > 0


def test_write_schema_doc_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = write_schema_doc(Path(tmp))
        assert path.exists()
        import json
        with open(path) as f:
            doc = json.load(f)
        assert "artifacts" in doc
        assert "hub_ranking" in doc["artifacts"]
        assert "calibration" in doc["artifacts"]


def test_validate_hub_ranking_passes_with_all_required():
    df = pd.DataFrame({col: ["placeholder"] for col in HUB_RANKING_REQUIRED})
    assert validate_hub_ranking(df) == []


def test_validate_calibration_passes_with_all_required():
    df = pd.DataFrame({col: ["placeholder"] for col in CALIBRATION_REQUIRED})
    assert validate_calibration(df) == []


def test_validate_hub_ranking_catches_missing():
    df = pd.DataFrame({"node": ["x"], "hub_score": [0.5]})
    missing = validate_hub_ranking(df)
    assert "rank" in missing
    assert "episode_tag" in missing


def test_validate_not_dataframe():
    assert len(validate_hub_ranking({})) > 0
    assert len(validate_calibration([])) > 0
