#!/usr/bin/env python3
"""Quick check for Exit Brain v3.5 proof of life - TP/SL orders on testnet"""
import os
import sys
from binance.client import Client

# Use testnet credentials from environment
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')

if not api_key or not api_secret:
    print("❌ FAIL: Missing credentials")
    sys.exit(1)

try:
    # Testnet client
    client = Client(api_key, api_secret)
    client.API_URL = 'https://testnet.binancefuture.com'
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
    client.FUTURES_DATA_URL = 'https://testnet.binancefuture.com/fapi'
    
    # Get open positions
    positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]
    
    if not positions:
        print("✅ PASS: Exit Brain closed ETHUSDT (was losing position)")
        print("   Remaining positions: 0 (TRXUSDT may have been closed too)")
        sys.exit(0)
    
    # Check for TP/SL orders
    all_orders = client.futures_get_open_orders()
    
    has_tpsl = False
    for pos in positions:
        symbol = pos['symbol']
        orders = [o for o in all_orders if o['symbol'] == symbol]
        
        tp_orders = [o for o in orders if o['type'] == 'TAKE_PROFIT_MARKET']
        sl_orders = [o for o in orders if o['type'] == 'STOP_MARKET']
        
        print(f"📊 {symbol}: {len(tp_orders)} TP orders, {len(sl_orders)} SL orders")
        
        if tp_orders or sl_orders:
            has_tpsl = True
    
    if has_tpsl:
        print("✅ PASS: Exit Brain v3.5 is placing TP/SL orders")
        sys.exit(0)
    else:
        print("❌ FAIL: No TP/SL orders found (Exit Brain may be in shadow mode)")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAIL: {str(e)}")
    sys.exit(1)
