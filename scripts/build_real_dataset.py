"""
Build the REAL dataset (Binance + Coinbase 1-min) and write a provenance data card.

Usage:
    python scripts/build_real_dataset.py --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.data.realbuild import assemble  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def write_data_card(manifest: dict, out: Path) -> None:
    lines = [
        "# Data Card — stablecoin-contagion-gnn (REAL DATA)",
        "",
        f"_Built: {dt.datetime.utcnow().isoformat()}Z_",
        "",
        "## Provenance",
        "",
        "- **Binance** spot 1-minute klines (`api.binance.com/api/v3/klines`), USDT-quoted:",
        "  USDC, TUSD, USDP, FRAX, BUSD, FDUSD, UST.",
        "- **Coinbase** Exchange 1-minute candles (`api.exchange.coinbase.com`), USD-quoted:",
        "  DAI, USDT.",
        "- Binance USDT-quoted prices are converted to **true USD** by multiplying by the",
        "  Coinbase USDT/USD mid (so a USDT depeg is not hidden by the quote currency).",
        "- This is genuine market data, **not** synthetic. Raw pulls cached under `data/raw/`.",
        "",
        "## Scope & honest limitations",
        "",
        "- The contagion graph is **cross-asset on the deepest venues** (Binance + Coinbase),",
        "  a deliberate scoping choice driven by where reliable 1-min history exists.",
        "- Episodes requiring pre-2019 minute history (e.g. USDT Oct-2018) are **dropped** when",
        "  no venue clears the coverage gate — see `episodes_built` below.",
        "- Single venue per asset means the cross-venue LOP-wedge feature is ~0 (kept for schema",
        "  parity); contagion edges come from cross-asset lead-lag correlation, not LOP.",
        "",
        "## Built episodes",
        "",
        "| Episode | rows(train/val/test split) |",
        "|---|---|",
    ]
    for ep in manifest.get("episodes_built", []):
        lines.append(f"| {ep} | see base_rates.csv |")
    lines += [
        "",
        "## Shapes",
        "",
        f"- feature_dim (F): {manifest.get('feature_dim')}",
        f"- n_nodes (N): {manifest.get('n_nodes')}",
        f"- train/val/test samples: {manifest.get('train_samples')} / "
        f"{manifest.get('val_samples')} / {manifest.get('test_samples')}",
        "",
        "See `data/processed/base_rates.csv` for positive rates per episode x horizon and",
        "`data/processed/dataset_manifest.json` for the full manifest.",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--cache_dir", default="data/raw")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    manifest = assemble(cfg, Path(args.out_dir), Path(args.cache_dir))
    write_data_card(manifest, Path("data/data_card.md"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
