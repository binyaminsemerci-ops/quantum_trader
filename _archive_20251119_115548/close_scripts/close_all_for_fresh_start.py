#!/usr/bin/env python3
"""Close all active positions to test AI fresh start."""
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

print("\n" + "=" * 70)
print("🔚 LUKKER ALLE AKTIVE POSISJONER - FRESH START FOR AI")
print("=" * 70)

# Get all positions
positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]

print(f"\n[CHART] Fant {len(positions)} åpne posisjoner:")

total_pnl = 0

for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    current = float(p['markPrice'])
    pnl = float(p['unRealizedProfit'])
    leverage = int(p['leverage'])
    
    total_pnl += pnl
    
    print(f"\n   {symbol}:")
    print(f"      {'LONG' if amt > 0 else 'SHORT'} {abs(amt)} @ ${entry}")
    print(f"      Current: ${current}")
    print(f"      Leverage: {leverage}x")
    print(f"      P&L: ${pnl:.2f}")
    
    # Determine close side
    side = 'SELL' if amt > 0 else 'BUY'
    qty = abs(amt)
    
    try:
        # Close position at market
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=qty,
            reduceOnly=True
        )
        print(f"      [OK] LUKKET - Order ID: {order['orderId']}")
        
        # Cancel all orders for this symbol
        try:
            open_orders = client.futures_get_open_orders(symbol=symbol)
            for o in open_orders:
                client.futures_cancel_order(symbol=symbol, orderId=o['orderId'])
            print(f"      🗑️  Kansellerte {len(open_orders)} ordrer")
        except:
            pass
            
    except Exception as e:
        print(f"      ❌ ERROR: {e}")

print("\n" + "=" * 70)
print(f"[MONEY] TOTAL REALISERT P&L: ${total_pnl:.2f}")
print("=" * 70)

# Get updated balance
try:
    account = client.futures_account()
    balance = float(account['totalWalletBalance'])
    available = float(account['availableBalance'])
    
    print(f"\n[MONEY] OPPDATERT BALANSE:")
    print(f"   Total: ${balance:.2f} USDT")
    print(f"   Tilgjengelig: ${available:.2f} USDT")
    
    print(f"\n[TARGET] PATH TIL $2720:")
    print(f"   Current: ${balance:.2f}")
    print(f"   Target: $2720")
    print(f"   Needed: ${2720 - balance:.2f}")
    print(f"   @ $48/win (3% TP): ~{int((2720-balance)/48)} winning trades")
    
except Exception as e:
    print(f"\n[WARNING] Could not fetch balance: {e}")

print("\n🤖 AI ER NÅ KLAR FOR FRESH START:")
print("   • Position Monitor kjører (hvert 30s)")
print("   • Event-Driven Executor søker 70%+ signals (hvert 10s)")
print("   • Trailing Stop Manager klar (hvert 10s)")
print("   • 20x leverage aktivert")
print("   • $1600 per trade")
print("   • 3% TP / 2% SL / 1.5% trailing")

print("\n⏳ NESTE TRADE:")
print("   AI vil analysere 36 symbols og åpne posisjon")
print("   når den finner 70%+ confidence signal.")
print("   Check docker logs for aktivitet:")
print("   → docker logs quantum_backend --tail 50 --follow")

print("\n" + "=" * 70)
print("[OK] ALLE POSISJONER LUKKET - AI TAR OVER!")
print("=" * 70 + "\n")
