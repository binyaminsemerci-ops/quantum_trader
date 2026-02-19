"""
Print current Binance symbols. Defaults to USDT-quoted spot pairs with status=TRADING.
Usage:
  python scripts/list_binance_symbols.py               # USDT spot
  python scripts/list_binance_symbols.py --all         # all symbols
  python scripts/list_binance_symbols.py --quote BUSD  # filter by quote asset
"""
import argparse
import sys
import requests

EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

def fetch_symbols():
    resp = requests.get(EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    return resp.json().get("symbols", [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Print all trading symbols")
    parser.add_argument("--quote", default="USDT", help="Quote asset filter (default: USDT)")
    args = parser.parse_args()

    symbols = fetch_symbols()
    if args.all:
        filtered = [s["symbol"] for s in symbols if s.get("status") == "TRADING"]
    else:
        filtered = [
            s["symbol"]
            for s in symbols
            if s.get("status") == "TRADING" and s.get("quoteAsset") == args.quote
        ]

    print(f"Count: {len(filtered)}")
    print(" ".join(filtered))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
