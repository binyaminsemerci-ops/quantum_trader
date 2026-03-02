#!/usr/bin/env python3
"""Direct Binance testnet position closer — kjøres bare én gang"""
import sys
sys.path.insert(0, '/opt/quantum')

from dotenv import load_dotenv
load_dotenv('/etc/quantum/testnet.env')

import os
from binance.client import Client
from binance.enums import *

k = os.environ.get('BINANCE_API_KEY', '')
s = os.environ.get('BINANCE_API_SECRET', '')
if not k:
    print("FEIL: Ingen API-nøkkel funnet i /etc/quantum/testnet.env")
    sys.exit(1)

c = Client(k, s, testnet=True)
print(f"Koblet til Binance testnet (key={k[:6]}...)")

# Hent alle åpne posisjoner
positions = [p for p in c.futures_position_information() if abs(float(p['positionAmt'])) > 0]
print(f"\nÅpne posisjoner funnet: {len(positions)}")

if not positions:
    print("Ingen åpne posisjoner — systemet er allerede flatt!")
    sys.exit(0)

for p in positions:
    sym = p['symbol']
    amt = float(p['positionAmt'])
    side = 'LONG' if amt > 0 else 'SHORT'
    close_side = SIDE_SELL if amt > 0 else SIDE_BUY
    qty = abs(amt)
    upnl = float(p.get('unRealizedProfit', 0))
    print(f"\n  {sym}: {side} qty={qty} upnl=${upnl:.4f}")
    try:
        order = c.futures_create_order(
            symbol=sym,
            side=close_side,
            type=ORDER_TYPE_MARKET,
            quantity=qty,
            reduceOnly=True
        )
        print(f"  ✅ LUKKET: orderId={order['orderId']} status={order['status']}")
    except Exception as e:
        print(f"  ❌ FEILET: {e}")
        # Prøv uten reduceOnly
        try:
            order = c.futures_create_order(
                symbol=sym,
                side=close_side,
                type=ORDER_TYPE_MARKET,
                quantity=qty,
                positionSide='BOTH'
            )
            print(f"  ✅ LUKKET (fallback): orderId={order['orderId']} status={order['status']}")
        except Exception as e2:
            print(f"  ❌ FALLBACK FEILET: {e2}")

print("\n=== Verifisering etter lukking ===")
import time
time.sleep(2)
remaining = [p for p in c.futures_position_information() if abs(float(p['positionAmt'])) > 0]
print(f"Gjenværende åpne posisjoner: {len(remaining)}")
for p in remaining:
    print(f"  {p['symbol']} amt={p['positionAmt']}")
if not remaining:
    print("✅ Alle posisjoner lukket — systemet er flatt!")
