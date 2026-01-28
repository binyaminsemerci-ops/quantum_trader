#!/usr/bin/env python3
"""
Direct Exchange Position Dump - Ground Truth for Position State

Queries Binance Testnet directly (same API Governor uses) and displays
all non-zero positions with mark price and notional value.

**SECURITY**: Never prints credentials. Reads from environment only.

Usage:
    # Via systemd env file (recommended)
    sudo -u quantum systemd-run --unit=temp-dump --setenv=EnvironmentFile=/etc/quantum/governor.env \
        python3 /home/qt/quantum_trader/scripts/dump_exchange_positions.py
    
    # Or with explicit env vars
    BINANCE_TESTNET_API_KEY=xxx BINANCE_TESTNET_API_SECRET=yyy python3 dump_exchange_positions.py

Requirements:
    - BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET in env
    - python-binance library (pip install python-binance)

Output:
    Symbol | Side | Qty | Mark Price | Notional USD | Leverage

⚠️  NEVER pass credentials via CLI args or print them in output
"""

import os
import sys
from binance.client import Client
from decimal import Decimal

def main():
    # Load testnet credentials from environment ONLY (no CLI, no printing)
    api_key = os.getenv('BINANCE_TESTNET_API_KEY')
    api_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
    
    if not api_key or not api_secret:
        print("❌ ERROR: Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET")
        print("Set them in your environment or /etc/quantum/governor.env")
        print("\nRecommended usage:")
        print("  sudo systemd-run --setenv=EnvironmentFile=/etc/quantum/governor.env python3 dump_exchange_positions.py")
        sys.exit(1)
    
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
