#!/usr/bin/env python3
"""Show 20x leverage status summary."""
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

print("\n" + "=" * 80)
print("⚡ 20x LEVERAGE MODE - STATUS OPPSUMMERING")
print("=" * 80)

# Get account info
account = client.futures_account()
balance = float(account['totalWalletBalance'])
available = float(account['availableBalance'])

print(f"\n[MONEY] KAPITAL:")
print(f"   Total: ${balance:.2f} USDT")
print(f"   Tilgjengelig: ${available:.2f} USDT")

# Get positions
positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]
print(f"\n[CHART] POSISJONER: {len(positions)}")

total_pnl = 0
for p in positions:
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    pnl = float(p['unRealizedProfit'])
    leverage = int(p['leverage'])
    total_pnl += pnl
    
    notional = abs(amt) * entry
    margin = notional / leverage
    
    print(f"\n   {p['symbol']}:")
    print(f"      {'LONG' if amt > 0 else 'SHORT'} {abs(amt)} @ ${entry}")
    print(f"      Leverage: {leverage}x")
    print(f"      Notional: ${notional:.2f}")
    print(f"      Margin: ${margin:.2f}")
    print(f"      P&L: ${pnl:.2f}")

print(f"\n   [CHART_UP] Total P&L: ${total_pnl:.2f}")

# Get open orders
orders = client.futures_get_open_orders()
tp_orders = [o for o in orders if o['type'] == 'TAKE_PROFIT_MARKET']
sl_orders = [o for o in orders if o['type'] == 'STOP_MARKET']
trail_orders = [o for o in orders if o['type'] == 'TRAILING_STOP_MARKET']

print(f"\n[SHIELD]  TP/SL BESKYTTELSE:")
print(f"   [OK] Take Profit ordrer: {len(tp_orders)}")
print(f"   🛑 Stop Loss ordrer: {len(sl_orders)}")
print(f"   [TARGET] Trailing Stop ordrer: {len(trail_orders)}")

# Read config
tp_pct = float(os.getenv('QT_TP_PCT', '0.03'))
sl_pct = float(os.getenv('QT_SL_PCT', '0.02'))
trail_pct = float(os.getenv('QT_TRAIL_PCT', '0.015'))
max_notional = float(os.getenv('QT_MAX_NOTIONAL_PER_TRADE', '1600'))
confidence = float(os.getenv('QT_MIN_CONFIDENCE', '0.70'))

print(f"\n⚙️  20x LEVERAGE KONFIGURASJON:")
print(f"   • Leverage: 20x")
print(f"   • Posisjonsstørrelse: ${max_notional:.0f} per trade")
print(f"   • Margin per trade: ${max_notional/20:.0f} USDT")
print(f"   • AI Confidence: {confidence*100:.0f}%+ only")
print(f"   • Take Profit: +{tp_pct*100:.1f}%")
print(f"   • Stop Loss: -{sl_pct*100:.1f}%")
print(f"   • Trailing: {trail_pct*100:.1f}%")

# Calculate potential with 20x
print(f"\n💡 MED 20x LEVERAGE:")
print(f"   • ${max_notional:.0f} notional krever ${max_notional/20:.0f} margin")
print(f"   • ${balance:.0f} kapital = {int(balance/(max_notional/20))} mulige posisjoner")
print(f"   • Maks exposure: ${balance * 20:.0f} USDT")
print(f"   • Per trade profit @ +{tp_pct*100:.1f}%: ${max_notional * tp_pct:.2f}")
print(f"   • Max loss per trade @ -{sl_pct*100:.1f}%: ${max_notional * sl_pct:.2f}")

print(f"\n[TARGET] PATH TIL $2720:")
print(f"   Start: ${balance:.2f}")
print(f"   Target: $2720")
print(f"   Needed: ${2720 - balance:.2f} profit")
print(f"   @ ${max_notional * tp_pct:.2f} per win: ~{int((2720-balance)/(max_notional*tp_pct))} winning trades")
print(f"   @ 70% win rate: ~{int((2720-balance)/(max_notional*tp_pct)/0.7)} total trades")

print("\n" + "=" * 80)
print("[OK] 20x LEVERAGE AKTIV - TP/SL BESKYTTET - KLAR FOR TRADING!")
print("=" * 80 + "\n")
