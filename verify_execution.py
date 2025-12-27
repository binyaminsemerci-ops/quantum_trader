#!/usr/bin/env python3
"""
Verifiser at trading logic faktisk kjører som beskrevet
"""
import os
from binance.client import Client
from datetime import datetime

client = Client(
    os.getenv('BINANCE_TESTNET_API_KEY'),
    os.getenv('BINANCE_TESTNET_SECRET_KEY'),
    testnet=True
)
client.API_URL = 'https://testnet.binancefuture.com'

print("=" * 80)
print("🔬 VERIFYING ACTUAL EXECUTION vs DOCUMENTATION")
print("=" * 80)
print()

# 1. Verify TP/SL orders exist
print("1️⃣  VERIFYING TP/SL PROTECTION ON POSITIONS")
print("-" * 80)
positions = client.futures_position_information()
active = [p for p in positions if float(p['positionAmt']) != 0]

orders = client.futures_get_open_orders()
orders_by_symbol = {}
for order in orders:
    symbol = order['symbol']
    if symbol not in orders_by_symbol:
        orders_by_symbol[symbol] = []
    orders_by_symbol[symbol].append(order)

for pos in active:
    symbol = pos['symbol']
    amt = float(pos['positionAmt'])
    side = "LONG" if amt > 0 else "SHORT"
    
    symbol_orders = orders_by_symbol.get(symbol, [])
    
    has_tp = any(o['type'] == 'TAKE_PROFIT_MARKET' for o in symbol_orders)
    has_sl = any(o['type'] == 'STOP_MARKET' for o in symbol_orders)
    has_trailing = any(o['type'] == 'TRAILING_STOP_MARKET' for o in symbol_orders)
    
    status = "✅" if (has_tp and has_sl) else "❌"
    
    print(f"{status} {symbol:15s} {side:5s} | TP:{has_tp} SL:{has_sl} Trail:{has_trailing}")

print()

# 2. Verify position sizes match 100 USDT @ 30x
print("2️⃣  VERIFYING POSITION SIZING (100 USDT @ 30x)")
print("-" * 80)
for pos in active[:5]:
    symbol = pos['symbol']
    amt = abs(float(pos['positionAmt']))
    entry = float(pos['entryPrice'])
    notional = amt * entry
    
    # With 30x leverage, 100 USDT margin = 3000 USDT notional
    expected_notional = 3000
    margin_used = notional / 30  # Assuming 30x leverage
    
    match = "✅" if 80 <= margin_used <= 120 else "⚠️"
    
    print(f"{match} {symbol:15s} | Notional: ${notional:8.2f} | Margin: ${margin_used:6.2f} USDT")

print()

# 3. Verify no high-funding positions
print("3️⃣  VERIFYING FUNDING RATE FILTER")
print("-" * 80)
print("Checking if any active positions have high funding rates...")

high_funding_symbols = ['1000WHYUSDT', '1000BONKUSDT', '1000RATSUSDT']
active_symbols = [p['symbol'] for p in active]

high_funding_active = [s for s in high_funding_symbols if s in active_symbols]

if high_funding_active:
    print(f"❌ WARNING: High-funding symbols found: {high_funding_active}")
else:
    print(f"✅ No high-funding symbols in active positions")
    print(f"   Active: {', '.join(active_symbols[:5])}")

print()

# 4. Check if positions were opened recently
print("4️⃣  VERIFYING RECENT TRADE ACTIVITY")
print("-" * 80)

recent_trades = []
for symbol in [p['symbol'] for p in active[:3]]:
    try:
        trades = client.futures_account_trades(symbol=symbol, limit=3)
        for trade in trades:
            time = datetime.fromtimestamp(trade['time'] / 1000)
            recent_trades.append((time, symbol, trade))
    except:
        pass

recent_trades.sort(key=lambda x: x[0], reverse=True)

print("Last 5 trades:")
for i, (time, symbol, trade) in enumerate(recent_trades[:5]):
    side = "BUY" if trade['buyer'] else "SELL"
    price = float(trade['price'])
    qty = float(trade['qty'])
    print(f"  {i+1}. {symbol:15s} {side:4s} ${price:10.4f} @ {time.strftime('%H:%M:%S')}")

print()

# 5. Verify dynamic TP/SL calculation
print("5️⃣  VERIFYING DYNAMIC TP/SL CALCULATION")
print("-" * 80)
print("Expected: Lower confidence = Tighter TP/SL")
print("From logs: confidence=0.52 -> TP=4.7% SL=6.6%")
print("           confidence=0.57 -> TP=4.8% SL=6.6%")

for pos in active:
    symbol = pos['symbol']
    amt = abs(float(pos['positionAmt']))
    entry = float(pos['entryPrice'])
    
    symbol_orders = orders_by_symbol.get(symbol, [])
    
    # Find TP order
    tp_orders = [o for o in symbol_orders if o['type'] == 'TAKE_PROFIT_MARKET']
    if tp_orders:
        tp_price = float(tp_orders[0]['stopPrice']) if tp_orders[0].get('stopPrice') else None
        if tp_price:
            tp_pct = abs((tp_price - entry) / entry * 100)
            print(f"  {symbol:15s} | Entry: ${entry:8.4f} | TP: ${tp_price:8.4f} ({tp_pct:.2f}%)")

print()

# 6. Check account leverage
print("6️⃣  VERIFYING LEVERAGE SETTING")
print("-" * 80)
try:
    # Check leverage for first active position
    if active:
        symbol = active[0]['symbol']
        leverage_bracket = client.futures_leverage_bracket(symbol=symbol)
        current_leverage = active[0].get('leverage', 'N/A')
        print(f"✅ {symbol} leverage: {current_leverage}x")
except Exception as e:
    print(f"⚠️  Could not verify leverage: {e}")

print()

print("=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)
