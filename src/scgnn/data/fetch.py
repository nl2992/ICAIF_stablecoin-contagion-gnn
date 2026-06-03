"""
Data acquisition layer.

CEX 1-min OHLCV → CCXT (Binance / Coinbase / Kraken).
DEX pool data   → The Graph subgraphs (Uniswap v3, Curve).
TVL             → DeFiLlama.

All outputs are written to data/raw/<venue>/<symbol>_1min.parquet and cached.
Gap logging: every gap > max_forward_fill_min is recorded in data/raw/gaps.csv.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ CEX / CCXT
_CCXT_TIMEFRAME = "1m"
_MAX_CANDLES = 1000      # CCXT per-request limit for most exchanges


def fetch_ccxt_ohlcv(
    exchange_id: str,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dest: Path,
    max_forward_fill_min: int = 5,
) -> pd.DataFrame:
    """
    Pull 1-min OHLCV from a CCXT exchange and cache to parquet.

    Handles pagination automatically.  Logs every gap > max_forward_fill_min.
    Falls back to the cached file if it already covers the window.
    """
    try:
        import ccxt
    except ImportError as e:
        raise ImportError("pip install ccxt") from e

    dest.mkdir(parents=True, exist_ok=True)
    cache_path = dest / f"{exchange_id}_{symbol.replace('/', '_')}_1min.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.DatetimeIndex(cached.index, tz="UTC")
        if cached.index[0] <= start and cached.index[-1] >= end:
            logger.info("Cache hit: %s", cache_path)
            return cached.loc[start:end]

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()

    if symbol not in exchange.symbols:
        logger.warning("%s not available on %s", symbol, exchange_id)
        return pd.DataFrame()

    all_rows = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    while cursor_ms < end_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, _CCXT_TIMEFRAME, since=cursor_ms, limit=_MAX_CANDLES
            )
        except Exception as exc:
            logger.error("CCXT error for %s/%s: %s", exchange_id, symbol, exc)
            time.sleep(5)
            continue
        if not candles:
            break
        all_rows.extend(candles)
        cursor_ms = int(candles[-1][0]) + 60_000
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Gap logging
    full_idx = pd.date_range(df.index[0], df.index[-1], freq="1min", tz="UTC")
    missing = full_idx.difference(df.index)
    if len(missing) > 0:
        gap_log = dest / "gaps.csv"
        gap_rows = pd.DataFrame(
            {"timestamp": missing, "venue": exchange_id, "symbol": symbol}
        )
        gap_rows.to_csv(gap_log, mode="a", header=not gap_log.exists(), index=False)
        logger.warning("%d missing 1-min bars for %s/%s", len(missing), exchange_id, symbol)

    # Forward-fill short gaps only
    df = df.reindex(full_idx)
    df = df.ffill(limit=max_forward_fill_min)
    df.to_parquet(cache_path)
    return df.loc[start:end]


# ------------------------------------------------------------------ DEX / The Graph


_UNISWAP_V3_SUBGRAPH = (
    "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
)
_CURVE_SUBGRAPH = (
    "https://api.thegraph.com/subgraphs/name/curvefi/curve"
)


def _gql_request(url: str, query: str, variables: dict) -> dict:
    resp = requests.post(url, json={"query": query, "variables": variables}, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", {})


def fetch_uniswap_v3_pool_hourly(
    pool_address: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dest: Path,
) -> pd.DataFrame:
    """Pull hourly volume + TVL for a Uniswap v3 pool via The Graph."""
    dest.mkdir(parents=True, exist_ok=True)
    cache_path = dest / f"univ3_{pool_address[:8]}_hourly.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        if df.index[0] <= start and df.index[-1] >= end:
            return df.loc[start:end]

    query = """
    query($pool: String!, $start: Int!, $end: Int!) {
      poolHourDatas(
        where: {pool: $pool, periodStartUnix_gte: $start, periodStartUnix_lte: $end}
        orderBy: periodStartUnix, orderDirection: asc, first: 1000
      ) {
        periodStartUnix
        tvlUSD
        volumeUSD
        token0Price
        token1Price
      }
    }
    """
    variables = {
        "pool": pool_address.lower(),
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
    }
    data = _gql_request(_UNISWAP_V3_SUBGRAPH, query, variables)
    rows = data.get("poolHourDatas", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["periodStartUnix"].astype(int), unit="s", utc=True)
    df = df.set_index("ts").drop(columns=["periodStartUnix"])
    df = df.astype(float)
    df.to_parquet(cache_path)
    return df.loc[start:end]


def fetch_curve_pool_hourly(
    pool_address: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dest: Path,
) -> pd.DataFrame:
    """Pull hourly data for a Curve pool via DeFiLlama (more reliable than subgraph)."""
    dest.mkdir(parents=True, exist_ok=True)
    cache_path = dest / f"curve_{pool_address[:8]}_hourly.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        if df.index[0] <= start and df.index[-1] >= end:
            return df.loc[start:end]

    url = f"https://api.llama.fi/protocol/curve-{pool_address}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("DeFiLlama error for %s: %s", pool_address, exc)
        return pd.DataFrame()

    tvl_data = data.get("tvl", [])
    if not tvl_data:
        return pd.DataFrame()

    df = pd.DataFrame(tvl_data)
    df["ts"] = pd.to_datetime(df["date"], unit="s", utc=True)
    df = df.set_index("ts").rename(columns={"totalLiquidityUSD": "tvl_usd"})
    df = df[["tvl_usd"]].sort_index()
    df.to_parquet(cache_path)
    return df.loc[start:end]


# ------------------------------------------------------------------ TVL / DeFiLlama


_DEFI_LLAMA_TVL = "https://api.llama.fi/protocol/{}"
_COIN_ID_MAP = {
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "FRAX": "frax",
    "TUSD": "true-usd",
    "USDe": "ethena-usde",
    "PYUSD": "paypal-usd",
    "BUSD": "binance-usd",
}


def fetch_protocol_tvl(protocol_slug: str, dest: Path) -> pd.DataFrame:
    """Pull daily TVL for a protocol from DeFiLlama."""
    dest.mkdir(parents=True, exist_ok=True)
    cache_path = dest / f"tvl_{protocol_slug}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df.index = pd.DatetimeIndex(df.index, tz="UTC")
        return df

    url = _DEFI_LLAMA_TVL.format(protocol_slug)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("DeFiLlama TVL error for %s: %s", protocol_slug, exc)
        return pd.DataFrame()

    tvl_data = resp.json().get("tvl", [])
    df = pd.DataFrame(tvl_data)
    df["ts"] = pd.to_datetime(df["date"], unit="s", utc=True)
    df = df.set_index("ts").rename(columns={"totalLiquidityUSD": "tvl_usd"})[["tvl_usd"]]
    df.to_parquet(cache_path)
    return df


# ------------------------------------------------------------------ Unified grid


def resample_to_common_grid(
    series_dict: Dict[str, pd.Series],
    freq: str = "1min",
    max_forward_fill_min: int = 5,
) -> pd.DataFrame:
    """
    Align all series onto a common UTC 1-min grid.
    Gaps > max_forward_fill_min are left as NaN (not filled).
    """
    if not series_dict:
        return pd.DataFrame()
    common = pd.date_range(
        start=min(s.index[0] for s in series_dict.values() if len(s)),
        end=max(s.index[-1] for s in series_dict.values() if len(s)),
        freq=freq,
        tz="UTC",
    )
    df = pd.DataFrame(index=common)
    for name, s in series_dict.items():
        s_reindexed = s.reindex(common)
        df[name] = s_reindexed.ffill(limit=max_forward_fill_min)
    return df


def coverage_pct(df: pd.DataFrame) -> pd.Series:
    """Return % non-NaN coverage per column."""
    return (df.notna().mean() * 100).rename("coverage_pct")
