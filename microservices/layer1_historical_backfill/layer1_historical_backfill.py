#!/usr/bin/env python3
"""
Layer 1 — Historical OHLCV Backfill
======================================
One-shot script: fetches N days of 1-minute candles from Binance Futures REST
and writes them to the same Parquet schema used by layer1_data_sink.

Why Binance FAPI:
  - /fapi/v1/klines does NOT require API key for public data
  - Returns futures OHLCV (mark candles): what the AI models actually trade
  - Rate limit: 2400 req/min weight, each klines call = 10 weight → safe

Schema (matches layer1_data_sink):
  open_time, open, high, low, close, volume, close_time, quote_volume,
  num_trades, taker_base, taker_quote, symbol, source, ts_ingested

Run:
  python layer1_historical_backfill.py                    # 7 days, top 15 symbols
  DAYS_BACK=30 python layer1_historical_backfill.py       # 30 days
  SYMBOLS=BTCUSDT,ETHUSDT python layer1_historical_backfill.py  # specific symbols
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import aiohttp

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s backfill %(message)s",
)
log = logging.getLogger("backfill")

# ── Config ───────────────────────────────────────────────────────────────
DATA_ROOT  = Path(os.getenv("DATA_ROOT", "/opt/quantum/data"))
DAYS_BACK  = int(os.getenv("DAYS_BACK", "7"))
INTERVAL   = os.getenv("INTERVAL", "1m")
LIMIT      = 1500          # max per Binance klines call
RATE_DELAY = 0.15          # seconds between requests (~6.6 req/s, well under 40/s limit)

_env_symbols = os.getenv("SYMBOLS", "")
TOP_SYMBOLS = [s.strip() for s in _env_symbols.split(",") if s.strip()] or [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "ADAUSDT", "LINKUSDT", "MATICUSDT",
    "DOTUSDT", "UNIUSDT", "LTCUSDT", "NEARUSDT", "INJUSDT",
]

BINANCE_FAPI = "https://fapi.binance.com"
BYBIT_REST   = "https://api.bybit.com"

# Parquet schema (matches layer1_data_sink)
PA_SCHEMA = pa.schema([
    pa.field("open_time",    pa.int64()),
    pa.field("open",         pa.float64()),
    pa.field("high",         pa.float64()),
    pa.field("low",          pa.float64()),
    pa.field("close",        pa.float64()),
    pa.field("volume",       pa.float64()),
    pa.field("close_time",   pa.int64()),
    pa.field("quote_volume", pa.float64()),
    pa.field("num_trades",   pa.int64()),
    pa.field("taker_base",   pa.float64()),
    pa.field("taker_quote",  pa.float64()),
    pa.field("symbol",       pa.string()),
    pa.field("source",       pa.string()),
    pa.field("ts_ingested",  pa.int64()),
])


# ── Binance FAPI Fetcher ─────────────────────────────────────────────────
async def fetch_binance_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[dict]:
    """Fetch all 1m candles for symbol in [start_ms, end_ms] from Binance FAPI."""
    rows = []
    cursor = start_ms
    request_count = 0

    while cursor < end_ms:
        url = (f"{BINANCE_FAPI}/fapi/v1/klines"
               f"?symbol={symbol}&interval={INTERVAL}"
               f"&startTime={cursor}&endTime={end_ms}&limit={LIMIT}")
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    log.warning(f"  [{symbol}] rate limited — sleeping 5s")
                    await asyncio.sleep(5)
                    continue
                if resp.status != 200:
                    log.error(f"  [{symbol}] HTTP {resp.status} — skipping batch")
                    break
                data = await resp.json()
        except Exception as e:
            log.error(f"  [{symbol}] fetch error: {e}")
            await asyncio.sleep(2)
            break

        if not data:
            break

        ts_now = int(time.time() * 1000)
        for k in data:
            rows.append({
                "open_time":    int(k[0]),
                "open":         float(k[1]),
                "high":         float(k[2]),
                "low":          float(k[3]),
                "close":        float(k[4]),
                "volume":       float(k[5]),
                "close_time":   int(k[6]),
                "quote_volume": float(k[7]),
                "num_trades":   int(k[8]),
                "taker_base":   float(k[9]),
                "taker_quote":  float(k[10]),
                "symbol":       symbol,
                "source":       "binance_fapi_hist",
                "ts_ingested":  ts_now,
            })

        last_open = int(data[-1][0])
        cursor = last_open + 60_000  # advance by 1 minute
        request_count += 1

        if len(data) < LIMIT:
            break  # reached end of available data

        await asyncio.sleep(RATE_DELAY)

    log.info(f"  [{symbol}] fetched {len(rows)} candles in {request_count} requests")
    return rows


# ── Bybit Fallback Fetcher ────────────────────────────────────────────────
async def fetch_bybit_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[dict]:
    """Bybit /v5/market/kline (fallback if Binance fails). Max 1000 per call."""
    rows = []
    cursor = start_ms
    BYBIT_LIMIT = 1000

    while cursor < end_ms:
        url = (f"{BYBIT_REST}/v5/market/kline"
               f"?category=linear&symbol={symbol}&interval=1"
               f"&start={cursor}&end={min(cursor + BYBIT_LIMIT * 60_000, end_ms)}"
               f"&limit={BYBIT_LIMIT}")
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()
        except Exception as e:
            log.error(f"  [{symbol}] bybit error: {e}")
            break

        klines = data.get("result", {}).get("list", [])
        if not klines:
            break

        ts_now = int(time.time() * 1000)
        for k in klines:
            # Bybit returns: [startTime, open, high, low, close, volume, turnover]
            rows.append({
                "open_time":    int(k[0]),
                "open":         float(k[1]),
                "high":         float(k[2]),
                "low":          float(k[3]),
                "close":        float(k[4]),
                "volume":       float(k[5]),
                "close_time":   int(k[0]) + 59_999,
                "quote_volume": float(k[6]) if len(k) > 6 else 0.0,
                "num_trades":   0,
                "taker_base":   0.0,
                "taker_quote":  0.0,
                "symbol":       symbol,
                "source":       "bybit_hist",
                "ts_ingested":  ts_now,
            })

        last_open = int(klines[-1][0])
        cursor = last_open + 60_000
        if len(klines) < BYBIT_LIMIT:
            break
        await asyncio.sleep(RATE_DELAY)

    log.info(f"  [{symbol}] bybit fetched {len(rows)} candles")
    return rows


# ── Parquet Writer ────────────────────────────────────────────────────────
def write_parquet(rows: List[dict], symbol: str):
    """Group rows by date and write/append to daily Parquet files."""
    if not rows or not HAS_PANDAS:
        return 0

    df = pd.DataFrame(rows)
    # Convert open_time (ms) to date strings
    df["_date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")

    written = 0
    sym_dir = DATA_ROOT / "ohlcv" / symbol / "1m"
    sym_dir.mkdir(parents=True, exist_ok=True)

    for date_str, group in df.groupby("_date"):
        day_df = group.drop(columns=["_date"]).sort_values("open_time").reset_index(drop=True)
        out_file = sym_dir / f"{date_str}.parquet"

        if out_file.exists():
            # Merge: load existing, concat, deduplicate
            existing = pd.read_parquet(out_file)
            merged = pd.concat([existing, day_df]).drop_duplicates(
                subset=["open_time"]
            ).sort_values("open_time").reset_index(drop=True)
            new_rows = len(merged) - len(existing)
            if new_rows <= 0:
                continue
            day_df = merged

        table = pa.Table.from_pandas(day_df, schema=PA_SCHEMA, preserve_index=False)
        pq.write_table(table, out_file, compression="snappy")
        written += len(day_df)

    return written


# ── Per-Symbol Backfill ───────────────────────────────────────────────────
async def backfill_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> dict:
    log.info(f"[{symbol}] backfilling {DAYS_BACK} days...")
    t0 = time.time()

    # Try Binance first
    rows = await fetch_binance_klines(session, symbol, start_ms, end_ms)

    # Bybit fallback if Binance returns very few rows
    if len(rows) < 100:
        log.warning(f"  [{symbol}] Binance returned only {len(rows)} rows — trying Bybit")
        rows = await fetch_bybit_klines(session, symbol, start_ms, end_ms)

    if not rows:
        return {"symbol": symbol, "status": "no_data", "rows": 0}

    written = write_parquet(rows, symbol)
    elapsed = round(time.time() - t0, 1)
    log.info(f"  [{symbol}] wrote {written} rows to Parquet in {elapsed}s")
    return {"symbol": symbol, "status": "ok", "rows": written, "elapsed_s": elapsed}


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    if not HAS_PANDAS:
        log.error("pandas / pyarrow not installed — cannot write Parquet. Run: pip install pandas pyarrow")
        return

    now_ms  = int(time.time() * 1000)
    start_ms = now_ms - DAYS_BACK * 24 * 3600 * 1000

    log.info(f"[L1-BACKFILL] Starting — {DAYS_BACK} days back, {len(TOP_SYMBOLS)} symbols")
    log.info(f"[L1-BACKFILL] Window: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).isoformat()} "
             f"→ {datetime.now(timezone.utc).isoformat()}")
    log.info(f"[L1-BACKFILL] Output: {DATA_ROOT}/ohlcv/<SYM>/1m/YYYY-MM-DD.parquet")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Use single session with connection pooling
    connector = aiohttp.TCPConnector(limit=5, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        results = []
        for sym in TOP_SYMBOLS:
            try:
                result = await backfill_symbol(session, sym, start_ms, now_ms)
                results.append(result)
            except Exception as e:
                log.error(f"[{sym}] FAILED: {e}", exc_info=True)
                results.append({"symbol": sym, "status": "error", "rows": 0})
            await asyncio.sleep(0.2)  # be polite

    # Summary
    total_rows = sum(r["rows"] for r in results)
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    log.info(f"\n[L1-BACKFILL] ════ COMPLETE ════")
    log.info(f"  Symbols: {ok_count}/{len(TOP_SYMBOLS)} OK")
    log.info(f"  Total rows written: {total_rows:,}")
    log.info(f"  Storage: {DATA_ROOT}/ohlcv/")
    for r in results:
        icon = "✅" if r["status"] == "ok" else "❌"
        log.info(f"  {icon} {r['symbol']:12s} {r['rows']:8,} rows  ({r['status']})")

    log.info("[L1-BACKFILL] Layer 3 backtest runner will now have data to work with.")


if __name__ == "__main__":
    asyncio.run(main())
