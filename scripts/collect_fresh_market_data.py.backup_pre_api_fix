"""
Collect fresh OHLCV data (no API keys) from Binance public endpoints and
save CSVs under data/market_data/latest/ for training.

Usage examples:
- python scripts/collect_fresh_market_data.py
- python scripts/collect_fresh_market_data.py --symbols BTCUSDT ETHUSDT --timeframe 1m --days 7
- python scripts/collect_fresh_market_data.py --endpoint spot --timeframe 15m --days 30
- python scripts/collect_fresh_market_data.py --symbols-file data/symbol_universe/high_volume_l1_l2_usdt_100.txt
- python scripts/collect_fresh_market_data.py --refresh-universe --refresh-top 120
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import aiohttp
import pandas as pd

BINANCE_SPOT_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUTURES_URL = "https://fapi.binance.com/fapi/v1/klines"
MAX_LIMIT = 1500  # Binance API per-request limit

# Default symbols mirror the training universe
DEFAULT_SYMBOLS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "OPUSDT",
    "ARBUSDT",
    "LINKUSDT",
    "UNIUSDT",
    "AAVEUSDT",
    "MKRUSDT",
    "ADAUSDT",
    "DOTUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "SUIUSDT",
    "INJUSDT",
    "TIAUSDT",
]

DEFAULT_UNIVERSE_FILE = Path("data/symbol_universe/high_volume_l1_l2_usdt_100.txt")

logger = logging.getLogger("collect_fresh_market_data")


def load_symbols_from_file(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {file_path}")
    symbols = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
    if not symbols:
        raise ValueError(f"Symbols file is empty: {file_path}")
    return symbols


def refresh_universe(symbols_file: Path, top: int, core_file: Optional[Path]) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "refresh_symbol_universe.py"),
        "--top",
        str(top),
        "--out",
        str(symbols_file),
    ]
    if core_file:
        cmd.extend(["--core-file", str(core_file)])

    logger.info("Refreshing symbol universe -> %s (top=%d)", symbols_file, top)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("refresh_symbol_universe failed")


async def fetch_klines(
    session: aiohttp.ClientSession,
    base_url: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    limit: int,
    delay: float,
) -> List[List]:
    """Paginate through klines until end time is reached."""
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    cursor = start_ms
    rows: List[List] = []

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }

        async with session.get(base_url, params=params, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()

        if not data:
            break

        rows.extend(data)
        cursor = data[-1][0] + 1  # advance past last candle
        await asyncio.sleep(delay)

    return rows


def klines_to_df(klines: List[List]) -> pd.DataFrame:
    """Convert Binance kline payload to a typed DataFrame."""
    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce", downcast="integer")

    df = df.dropna()
    return df


async def fetch_symbol(
    session: aiohttp.ClientSession,
    base_url: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    limit: int,
    delay: float,
    sem: asyncio.Semaphore,
) -> tuple[str, pd.DataFrame]:
    """Fetch one symbol with concurrency guard."""
    async with sem:
        logger.info("Fetching %s %s from %s to %s", symbol, interval, start.date(), end.date())
        klines = await fetch_klines(session, base_url, symbol, interval, start, end, limit, delay)
        df = klines_to_df(klines)
        if df.empty:
            logger.warning("No data returned for %s", symbol)
        else:
            logger.info("%s rows for %s", len(df), symbol)
        return symbol, df


async def collect_data(
    symbols: List[str],
    interval: str,
    days: int,
    endpoint: str,
    outdir: Path,
    concurrency: int,
    delay: float,
) -> None:
    base_url = BINANCE_FUTURES_URL if endpoint == "futures" else BINANCE_SPOT_URL
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    outdir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_symbol(session, base_url, sym, interval, start, end, MAX_LIMIT, delay, sem)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks)

    combined: List[pd.DataFrame] = []
    for symbol, df in results:
        if df.empty:
            continue
        fname = f"{symbol}_{interval}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        df.to_csv(outdir / fname, index=False)
        combined.append(df.assign(symbol=symbol))

    if combined:
        merged = pd.concat(combined, ignore_index=True)
        merged_file = outdir / f"combined_{interval}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        merged.to_csv(merged_file, index=False)
        logger.info("Wrote combined dataset: %s (%d rows)", merged_file, len(merged))
    else:
        logger.warning("No datasets were written (all empty responses)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fresh OHLCV data (no API keys required)")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to download (default: training universe or symbols-file)",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Path to symbols file (one symbol per line)",
    )
    parser.add_argument(
        "--timeframe",
        dest="timeframe",
        default="1m",
        help="Binance interval (e.g., 1m, 5m, 15m, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="How many trailing days to fetch",
    )
    parser.add_argument(
        "--endpoint",
        choices=["futures", "spot"],
        default="futures",
        help="Choose futures (default) or spot public endpoint",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/market_data/latest"),
        help="Where to write CSV files",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent symbol requests",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Sleep between paged requests to stay under rate limits",
    )
    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help="Refresh symbol universe file before download",
    )
    parser.add_argument(
        "--refresh-top",
        type=int,
        default=100,
        help="Top-N by 24h quote volume when refreshing universe",
    )
    parser.add_argument(
        "--refresh-core-file",
        type=Path,
        default=None,
        help="Core symbols file to keep when refreshing universe",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    symbols: List[str]

    # Refresh universe if requested
    if args.refresh_universe:
        target_file = args.symbols_file or DEFAULT_UNIVERSE_FILE
        refresh_universe(symbols_file=target_file, top=args.refresh_top, core_file=args.refresh_core_file)
        symbols = load_symbols_from_file(target_file)
    elif args.symbols_file:
        symbols = load_symbols_from_file(args.symbols_file)
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_SYMBOLS

    try:
        asyncio.run(
            collect_data(
                symbols=symbols,
                interval=args.timeframe,
                days=args.days,
                endpoint=args.endpoint,
                outdir=args.outdir,
                concurrency=max(1, args.concurrency),
                delay=max(0.05, args.delay),
            )
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as exc:
        logger.error("Failed to collect data: %s", exc)
        raise


if __name__ == "__main__":
    main()
