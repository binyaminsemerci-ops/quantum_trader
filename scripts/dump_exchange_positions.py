#!/usr/bin/env python3
"""
Direct Exchange Position Dump - Ground Truth for Position State

Queries Binance Testnet directly (same API Governor uses) and displays
all non-zero positions with mark price and notional value.

**SECURITY**: Never prints credentials. Reads from environment or file.

Usage:
    # Via env-file (recommended)
    python3 dump_exchange_positions.py --env-file /etc/quantum/governor.env
    
    # Via systemd EnvironmentFile (gold standard)
    systemd-run --pipe --wait -p EnvironmentFile=/etc/quantum/governor.env \
        python3 dump_exchange_positions.py
    
    # With filtering
    python3 dump_exchange_positions.py --symbol ETHUSDT --max 10

Requirements:
    - BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET in env or file
    - python-binance library (pip install python-binance)

Output:
    Symbol | Side | Qty | Mark Price | Notional USD | Leverage

⚠️  NEVER pass credentials via CLI args or print them in output
"""

import os
import sys
import argparse
from binance.client import Client
from decimal import Decimal

def load_env_file(path):
    """
    Load environment variables from file (KEY=VALUE format).
    Supports comments (#), blank lines, quoted values.
    Only sets if not already in environment.
    
    SECURITY: Never prints values, only success/failure.
    """
    if not os.path.exists(path):
        print(f"❌ ERROR: Env file not found: {path}", file=sys.stderr)
        return False
    
    loaded_keys = []
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip blank lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Only set if not already in environment
                if key and value and key not in os.environ:
                    os.environ[key] = value
                    loaded_keys.append(key)
        
        if loaded_keys:
            print(f"✅ Loaded env-file: yes", file=sys.stderr)
            return True
        else:
            print(f"⚠️  Loaded env-file: no (keys already in environment)", file=sys.stderr)
            return True
    
    except Exception as e:
        print(f"❌ ERROR: Failed to load env file {path}: {e}", file=sys.stderr)
        return False

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Query Binance Testnet positions (ground truth)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='SECURITY: Never prints API credentials'
    )
    parser.add_argument(
        '--env-file',
        default='/etc/quantum/governor.env',
        help='Path to env file with BINANCE_TESTNET_API_KEY/SECRET (default: %(default)s)'
    )
    parser.add_argument(
        '--symbol',
        help='Filter to specific symbol (e.g., ETHUSDT)'
    )
    parser.add_argument(
        '--max',
        type=int,
        help='Max positions to display'
    )
    
    args = parser.parse_args()
    
    # Load credentials from env-file if not already in environment
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
    
    if not api_key or not api_secret:
        # Try loading from env-file
        if not load_env_file(args.env_file):
            sys.exit(2)
        
        # Re-check after loading
        api_key = os.getenv('BINANCE_TESTNET_API_KEY')
        api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ ERROR: Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET", file=sys.stderr)
        print(f"Tried: environment variables + {args.env_file}", file=sys.stderr)
        print("\nRecommended usage:", file=sys.stderr)
        print("  python3 dump_exchange_positions.py --env-file /etc/quantum/governor.env", file=sys.stderr)
        print("  systemd-run --pipe -p EnvironmentFile=/etc/quantum/governor.env python3 dump_exchange_positions.py", file=sys.stderr)
        sys.exit(2)
    
    # Initialize client
    try:
        client = Client(api_key, api_secret, testnet=True)
        client.futures_change_leverage(symbol='BTCUSDT', leverage=1)  # Auth test
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to Binance Testnet: {e}")
        sys.exit(1)
    
    # Fetch account info
    try:
        account = client.futures_account()
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch account info: {e}")
        sys.exit(1)
    
    # Fetch mark prices for all symbols
    try:
        mark_prices_raw = client.futures_mark_price()
        mark_prices = {item['symbol']: float(item['markPrice']) for item in mark_prices_raw}
    except Exception as e:
        print(f"⚠️  WARNING: Could not fetch mark prices: {e}")
        mark_prices = {}
    
    # Parse positions
    positions = account.get('positions', [])
    if not positions:
        print("⚠️  No positions returned from exchange")
        return
    
    # Filter non-zero positions
    THRESHOLD = 1e-8
    active_positions = []
    
    for pos in positions:
        symbol = pos.get('symbol', 'UNKNOWN')
        qty = float(pos.get('positionAmt', 0))
        leverage = int(pos.get('leverage', 0))
        
        # Use mark prices from API call (more reliable)
        mark_price = mark_prices.get(symbol, float(pos.get('markPrice', 0)))
        
        if abs(qty) > THRESHOLD:
            notional = abs(qty) * mark_price
            side = "LONG" if qty > 0 else "SHORT"
            active_positions.append({
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'mark_price': mark_price,
                'notional': notional,
                'leverage': leverage
            })
    
    # Display results
    print("\n" + "="*80)
    print("BINANCE TESTNET FUTURES - POSITION DUMP")
    print("="*80)
    print(f"Total symbols checked: {len(positions)}")
    print(f"Active positions (abs(qty) > {THRESHOLD}): {len(active_positions)}")
    print("="*80)
    
    if not active_positions:
        print("\n✅ ALL FLAT - No active positions found (qty threshold: {:.0e})".format(THRESHOLD))
        return
    
    # Apply symbol filter if specified
    if args.symbol:
        active_positions = [p for p in active_positions if p['symbol'] == args.symbol.upper()]
        if not active_positions:
            print(f"\n⚠️  No positions found for symbol: {args.symbol}")
            return
    
    # Apply max limit if specified
    if args.max and len(active_positions) > args.max:
        print(f"\n⚠️  Limiting to top {args.max} positions (total: {len(active_positions)})")
        active_positions = active_positions[:args.max]
    
    # Sort by notional descending
    active_positions.sort(key=lambda x: x['notional'], reverse=True)
    
    # Table header
    print("\n{:<12} {:<6} {:>15} {:>15} {:>15} {:>8}".format(
        "SYMBOL", "SIDE", "QTY", "MARK_PRICE", "NOTIONAL_USD", "LEV"
    ))
    print("-" * 80)
    
    # Table rows
    total_notional = Decimal(0)
    for pos in active_positions:
        print("{:<12} {:<6} {:>15.4f} {:>15.4f} {:>15.2f} {:>8}x".format(
            pos['symbol'],
            pos['side'],
            pos['qty'],
            pos['mark_price'],
            pos['notional'],
            pos['leverage']
        ))
        total_notional += Decimal(str(pos['notional']))
    
    # Footer
    print("-" * 80)
    print("{:<12} {:<6} {:>15} {:>15} {:>15.2f} {:>8}".format(
        "TOTAL", "", "", "", float(total_notional), ""
    ))
    print("="*80)
    
    # Additional stats
    print(f"\nActive symbols: {len(active_positions)}")
    print(f"Total notional exposure: ${total_notional:,.2f}")
    print(f"Query timestamp: {account.get('updateTime', 'unknown')}")
    print("\n✅ Direct exchange position dump complete")

if __name__ == '__main__':
    main()
