"""
Real 1-minute price acquisition via direct exchange REST endpoints.

Why not ccxt here: ccxt's ``load_markets()`` is slow / occasionally blocked in
sandboxed environments.  The two endpoints below are the minimal, reliable path
for the assets we actually need and are fully paginated + cached to parquet.

Venues
------
- Binance  spot klines  (quote = USDT)  -> asset/USDT 1-min close+volume
- Coinbase exchange candles (quote = USD) -> asset/USD 1-min close+volume

Peg reference
-------------
Binance prices are USDT-quoted.  We convert to *true USD* by multiplying by the
Coinbase USDT/USD mid (reindexed, ffilled), so a USDT depeg is not hidden.  When
USDT/USD is unavailable we fall back to USDT == $1 and record that in the data card.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (research; stablecoin-contagion-gnn)"}
_BINANCE = "https://api.binance.com/api/v3/klines"
_COINBASE = "https://api.exchange.coinbase.com/products/{}/candles"


def _get_json(url: str, timeout: int = 20, retries: int = 4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url[:90]} :: {last}")


# ----------------------------------------------------------------- Binance
def fetch_binance_1m(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                     cache_dir: Path) -> pd.DataFrame:
    """Paginated 1-min klines.  Returns DataFrame[close, volume] on a UTC index."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"binance_{symbol}_{start.date()}_{end.date()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    cursor, rows = start_ms, []
    while cursor < end_ms:
        url = (f"{_BINANCE}?symbol={symbol}&interval=1m"
               f"&startTime={cursor}&endTime={end_ms}&limit=1000")
        batch = _get_json(url)
        if not batch:
            break
        rows.extend(batch)
        nxt = int(batch[-1][0]) + 60_000
        if nxt <= cursor:
            break
        cursor = nxt
        time.sleep(0.20)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v",
                                     "ct", "qv", "n", "tb", "tq", "ig"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    out = (df.set_index("ts")[["c", "v"]]
             .rename(columns={"c": "close", "v": "volume"}).astype(float)
             .sort_index())
    out = out[~out.index.duplicated(keep="first")]
    out.to_parquet(cache)
    return out


# ----------------------------------------------------------------- Coinbase
def fetch_coinbase_1m(product: str, start: pd.Timestamp, end: pd.Timestamp,
                      cache_dir: Path) -> pd.DataFrame:
    """Paginated 1-min candles (max 300/request).  Returns DataFrame[close, volume]."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"coinbase_{product}_{start.date()}_{end.date()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    rows, cursor = [], start
    step = pd.Timedelta(minutes=300)
    while cursor < end:
        seg_end = min(cursor + step, end)
        url = (_COINBASE.format(product)
               + f"?granularity=60&start={cursor.isoformat()}&end={seg_end.isoformat()}")
        try:
            batch = _get_json(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("coinbase %s segment failed: %s", product, exc)
            batch = []
        # candle = [time, low, high, open, close, volume]
        for cdl in batch:
            rows.append((cdl[0], cdl[4], cdl[5]))
        cursor = seg_end
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    out = (df.set_index("ts").astype(float).sort_index())
    out = out[~out.index.duplicated(keep="first")]
    out.to_parquet(cache)
    return out


# ----------------------------------------------------------------- venue/symbol map
# (asset, venue) -> (fetcher_kind, exchange_symbol)
SYMBOL_MAP = {
    ("USDC", "binance"): ("binance", "USDCUSDT"),
    ("TUSD", "binance"): ("binance", "TUSDUSDT"),
    ("USDP", "binance"): ("binance", "USDPUSDT"),
    ("FRAX", "binance"): ("binance", "FRAXUSDT"),
    ("BUSD", "binance"): ("binance", "BUSDUSDT"),
    ("FDUSD", "binance"): ("binance", "FDUSDUSDT"),
    ("UST", "binance"): ("binance", "USTUSDT"),
    ("DAI", "coinbase"): ("coinbase", "DAI-USD"),
    ("USDT", "coinbase"): ("coinbase", "USDT-USD"),
}


def fetch_series(asset: str, venue: str, start: pd.Timestamp, end: pd.Timestamp,
                 cache_dir: Path) -> pd.DataFrame:
    key = (asset, venue)
    if key not in SYMBOL_MAP:
        return pd.DataFrame()
    kind, sym = SYMBOL_MAP[key]
    if kind == "binance":
        return fetch_binance_1m(sym, start, end, cache_dir)
    return fetch_coinbase_1m(sym, start, end, cache_dir)
