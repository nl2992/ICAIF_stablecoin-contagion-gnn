"""
Data manifest — frozen empirical targets for reproducibility.

The manifest records:
  - source URL / exchange endpoint for every episode × (asset, venue)
  - fetch date (UTC ISO-8601)
  - SHA-256 checksum of the cached parquet file

After the initial data pull, commit data/manifest.json.  Any subsequent run
that finds a different checksum on disk raises an error — the empirical
targets are frozen and cannot silently drift.

Usage:
    python -m scgnn.data.manifest build   # build from data/raw/
    python -m scgnn.data.manifest verify  # verify checksums
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


_MANIFEST_PATH = Path("data/manifest.json")

# Source metadata for each venue × data type
SOURCE_REGISTRY: Dict[str, dict] = {
    "binance_ohlcv": {
        "type": "CEX OHLCV 1-min",
        "endpoint": "https://api.binance.com/api/v3/klines",
        "library": "ccxt>=4.2",
        "notes": "Rate-limited; full history available from 2017.",
    },
    "coinbase_ohlcv": {
        "type": "CEX OHLCV 1-min",
        "endpoint": "https://api.exchange.coinbase.com/products/{symbol}/candles",
        "library": "ccxt>=4.2",
        "notes": "300-candle limit per request; requires pagination.",
    },
    "kraken_ohlcv": {
        "type": "CEX OHLCV 1-min",
        "endpoint": "https://api.kraken.com/0/public/OHLC",
        "library": "ccxt>=4.2",
        "notes": "1-min history for 2018 may be incomplete; supplement with archives if needed.",
    },
    "uniswap_v3_thegraph": {
        "type": "DEX hourly pool data",
        "endpoint": "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
        "library": "requests>=2.31",
        "notes": "Hourly TVL + volume; free but rate-limited; API key recommended for production.",
    },
    "defillama_tvl": {
        "type": "Protocol TVL daily",
        "endpoint": "https://api.llama.fi/protocol/{slug}",
        "library": "requests>=2.31",
        "notes": "Free, no auth required.",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(raw_dir: Path = Path("data/raw")) -> dict:
    """
    Walk data/raw/, compute checksums, and return a manifest dict.
    Call this once after the initial data pull.
    """
    fetch_date = datetime.now(timezone.utc).isoformat()
    entries: List[dict] = []

    for parquet_file in sorted(raw_dir.rglob("*.parquet")):
        rel = parquet_file.relative_to(raw_dir)
        parts = rel.parts
        venue = parts[0] if len(parts) > 1 else "unknown"
        source_key = next(
            (k for k in SOURCE_REGISTRY if venue in k), "unknown"
        )
        entries.append({
            "file": str(rel),
            "venue": venue,
            "source": SOURCE_REGISTRY.get(source_key, {}),
            "fetch_date_utc": fetch_date,
            "sha256": _sha256(parquet_file),
            "size_bytes": parquet_file.stat().st_size,
        })

    manifest = {
        "schema_version": "1.0",
        "built_utc": fetch_date,
        "n_files": len(entries),
        "files": entries,
    }
    return manifest


def save_manifest(manifest: dict, path: Path = _MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {path}  ({manifest['n_files']} files)")


def verify_manifest(
    raw_dir: Path = Path("data/raw"),
    path: Path = _MANIFEST_PATH,
    strict: bool = True,
) -> bool:
    """
    Verify all files in the manifest still match their recorded checksums.
    strict=True raises RuntimeError on mismatch (use in CI/pipeline entry points).
    """
    if not path.exists():
        print(f"[WARN] No manifest found at {path} — run 'python -m scgnn.data.manifest build' first.")
        return False

    with open(path) as f:
        manifest = json.load(f)

    failures: List[str] = []
    for entry in manifest["files"]:
        full_path = raw_dir / entry["file"]
        if not full_path.exists():
            failures.append(f"MISSING: {entry['file']}")
            continue
        actual = _sha256(full_path)
        if actual != entry["sha256"]:
            failures.append(
                f"CHECKSUM MISMATCH: {entry['file']}\n"
                f"  expected: {entry['sha256']}\n"
                f"  actual:   {actual}"
            )

    if failures:
        msg = "Data manifest verification FAILED:\n" + "\n".join(failures)
        if strict:
            raise RuntimeError(msg)
        print(f"[ERROR] {msg}")
        return False

    print(f"[OK] Manifest verified: {len(manifest['files'])} files match recorded checksums.")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "build":
        m = build_manifest()
        save_manifest(m)
    elif cmd == "verify":
        ok = verify_manifest()
        sys.exit(0 if ok else 1)
    else:
        print(f"Unknown command: {cmd}. Use 'build' or 'verify'.")
        sys.exit(1)
