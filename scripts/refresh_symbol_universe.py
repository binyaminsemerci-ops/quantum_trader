"""
Refresh a curated symbol universe based on Binance spot data.
- Always include a core L1/L2/high-volume set
- Add top-N USDT symbols by 24h quote volume (status=TRADING)
- Write the resulting list (one symbol per line) to the target file

Usage examples:
  python scripts/refresh_symbol_universe.py
  python scripts/refresh_symbol_universe.py --top 120 --out data/symbol_universe/custom.txt
  python scripts/refresh_symbol_universe.py --core-file data/symbol_universe/high_volume_l1_l2_usdt.txt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Dict

import requests

EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
TICKER_24H_URL = "https://api.binance.com/api/v3/ticker/24hr"
DEFAULT_OUT = Path("data/symbol_universe/high_volume_l1_l2_usdt_100.txt")
DEFAULT_CORE_FILE = Path("data/symbol_universe/high_volume_l1_l2_usdt.txt")

# Fallback core list if core file is missing
DEFAULT_CORE_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "TRXUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "ARBUSDT",
    "OPUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "SUIUSDT",
    "INJUSDT",
    "TIAUSDT",
    "SEIUSDT",
    "STRKUSDT",
    "FILUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "ETCUSDT",
    "PEPEUSDT",
    "WIFUSDT",
    "RENDERUSDT",
    "TONUSDT",
]

logger = logging.getLogger("refresh_symbol_universe")


def load_core_symbols(core_path: Path) -> List[str]:
    if core_path.exists():
        symbols = [line.strip() for line in core_path.read_text().splitlines() if line.strip()]
        if symbols:
            return symbols
    logger.warning("Core file missing or empty, using built-in default core list")
    return DEFAULT_CORE_SYMBOLS


def fetch_exchange_info() -> Dict[str, Dict[str, str]]:
    resp = requests.get(EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    info = resp.json().get("symbols", [])
    return {s["symbol"]: s for s in info}


def fetch_tickers() -> Dict[str, float]:
    resp = requests.get(TICKER_24H_URL, timeout=30)
    resp.raise_for_status()
    tickers = resp.json()
    volumes = {}
    for t in tickers:
        sym = t.get("symbol")
        if not sym:
            continue
        try:
            volumes[sym] = float(t.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            volumes[sym] = 0.0
    return volumes


def build_universe(core: Iterable[str], top_n: int, quote: str) -> List[str]:
    core_list = list(dict.fromkeys(core))  # preserve order, remove dups
    info = fetch_exchange_info()
    tickers = fetch_tickers()

    candidates = []
    for symbol, meta in info.items():
        if meta.get("status") != "TRADING":
            continue
        if meta.get("quoteAsset") != quote:
            continue
        vol = tickers.get(symbol, 0.0)
        candidates.append((symbol, vol))

    candidates.sort(key=lambda x: x[1], reverse=True)

    top_symbols: List[str] = []
    for sym, _ in candidates:
        if sym in core_list:
            continue
        top_symbols.append(sym)
        if len(top_symbols) >= top_n:
            break

    result = core_list + top_symbols
    return result


def write_universe(symbols: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(symbols))
    logger.info("Wrote %d symbols to %s", len(symbols), out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh symbol universe from Binance volumes")
    parser.add_argument("--top", type=int, default=100, help="How many top-volume symbols to add")
    parser.add_argument("--quote", default="USDT", help="Quote asset filter (default: USDT)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output file path")
    parser.add_argument("--core-file", type=Path, default=DEFAULT_CORE_FILE, help="Core symbols file")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    core_symbols = load_core_symbols(args.core_file)
    universe = build_universe(core_symbols, top_n=args.top, quote=args.quote)
    write_universe(universe, args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Failed to refresh symbol universe: %s", exc)
        sys.exit(1)
