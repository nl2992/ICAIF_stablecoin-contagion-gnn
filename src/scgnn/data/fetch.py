"""Pull 1-min OHLCV + on-chain data for the node universe."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ASSETS = ["USDC", "USDT", "DAI", "FRAX", "TUSD", "USDe", "PYUSD", "BUSD"]
VENUES = ["binance", "coinbase", "kraken"]

_BINANCE_BASE = "https://api.binance.com/api/v3/klines"
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def fetch_binance_1min(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dest: Path,
) -> pd.DataFrame:
    """Download 1-min OHLCV from Binance for a single symbol."""
    rows = []
    limit = 1000
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        resp = requests.get(_BINANCE_BASE, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 60_000
        time.sleep(0.1)

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].astype(float)

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{symbol}_1min.parquet"
    df.to_parquet(out)
    return df


def fetch_curve_pool_tvl(pool_address: str, dest: Path) -> pd.DataFrame:
    """Pull hourly TVL for a Curve pool from DeFiLlama."""
    url = f"https://api.llama.fi/protocol/{pool_address}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    tvl_series = data.get("tvl", [])
    df = pd.DataFrame(tvl_series).rename(columns={"date": "timestamp", "totalLiquidityUSD": "tvl_usd"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp")

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{pool_address}_tvl.parquet"
    df.to_parquet(out)
    return df


def build_price_series(raw_dir: Path, asset: str) -> pd.Series:
    """Return a unified 1-min close-price series (mean across venues) for an asset."""
    frames = []
    for venue in VENUES:
        p = raw_dir / f"{asset}USDT_1min.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p)["close"])
    if not frames:
        raise FileNotFoundError(f"No raw data found for {asset} in {raw_dir}")
    return pd.concat(frames, axis=1).mean(axis=1).rename(asset)


def build_peg_deviation(price: pd.Series, peg: float = 1.0) -> pd.Series:
    """Return peg deviation in basis points."""
    return ((price - peg) / peg * 10_000).rename(f"{price.name}_bps")
