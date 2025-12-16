#!/usr/bin/env python3
"""Check if TP/SL orders are actually set on Binance and analyze all orders."""
import os
from binance.client import Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

# Get open positions
positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]

# Get all open orders
open_orders = client.futures_get_open_orders()

print("\n" + "=" * 80)
print("[SEARCH] SJEKKER TP/SL ORDRER PÅ BINANCE")
print("=" * 80)

print(f"\n[CHART] OPEN POSITIONS: {len(positions)}")
for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    pnl = float(p['unRealizedProfit'])
    
    print(f"\n{'─' * 80}")
    print(f"   Symbol: {symbol}")
    print(f"   Position: {amt} @ ${entry:.4f}")
    print(f"   P&L: ${pnl:.2f}")
    
    # Check for TP/SL orders on this symbol
    symbol_orders = [o for o in open_orders if o['symbol'] == symbol]
    
    if symbol_orders:
        print(f"   Orders: {len(symbol_orders)}")
        for order in symbol_orders:
            order_type = order['type']
            side = order['side']
            qty = float(order['origQty'])
            stop_price = float(order.get('stopPrice', 0))
            price = float(order.get('price', 0))
            
            if order_type == 'TAKE_PROFIT_MARKET':
                pct_from_entry = ((stop_price - entry) / entry * 100) if amt > 0 else ((entry - stop_price) / entry * 100)
                print(f"      [OK] TP: {qty} @ ${stop_price:.4f} (+{pct_from_entry:.2f}%)")
            elif order_type == 'STOP_MARKET':
                pct_from_entry = ((entry - stop_price) / entry * 100) if amt > 0 else ((stop_price - entry) / entry * 100)
                print(f"      🛑 SL: {qty} @ ${stop_price:.4f} (-{pct_from_entry:.2f}%)")
            elif order_type == 'TRAILING_STOP_MARKET':
                callback = float(order.get('priceRate', 0)) * 100
                print(f"      [TARGET] TRAILING: {qty}, callback {callback:.2f}%")
            else:
                print(f"      ❓ {order_type}: {side} {qty}")
    else:
        print(f"      [WARNING] INGEN ORDRER - Position har ikke TP/SL beskyttelse!")

print("\n" + "=" * 80)
print(f"[CLIPBOARD] ALLE ÅPNE ORDRER: {len(open_orders)} totalt")
print("=" * 80)

# Group orders by type
order_types = {}
for order in open_orders:
    order_type = order['type']
    order_types[order_type] = order_types.get(order_type, 0) + 1

for order_type, count in sorted(order_types.items(), key=lambda x: x[1], reverse=True):
    print(f"   {order_type}: {count} ordrer")

print("\n🤖 HVEM SATTE ORDRENE?")
print("=" * 80)

# Check order creation times to see if they're recent (from auto_set_tpsl.py or AI)
recent_orders = []
old_orders = []

for order in open_orders:
    order_time = datetime.fromtimestamp(order['time'] / 1000)
    age_minutes = (datetime.now() - order_time).total_seconds() / 60
    
    if age_minutes < 30:  # Last 30 minutes
        recent_orders.append(order)
    else:
        old_orders.append(order)

print(f"\n   📅 Nye ordrer (siste 30 min): {len(recent_orders)}")
print(f"   📅 Gamle ordrer (>30 min): {len(old_orders)}")

if recent_orders:
    print("\n   SISTE ORDRER:")
    for order in recent_orders[:10]:  # Show last 10
        order_time = datetime.fromtimestamp(order['time'] / 1000)
        age = (datetime.now() - order_time).total_seconds() / 60
        print(f"      • {order['symbol']} {order['type']} - {age:.1f} min siden")

print("\n💡 KONKLUSJON:")
if len(open_orders) > len(positions) * 2:
    print("   [WARNING] Du har MANGE ordrer - sannsynligvis både gamle og nye")
    print("   🤖 AI/auto_set_tpsl.py setter nye ordrer")
    print("   [MEMO] Anbefaling: Kanseller gamle ordrer for å rydde opp")
else:
    print("   [OK] Normal mengde ordrer for aktive posisjoner")

print("=" * 80 + "\n")
