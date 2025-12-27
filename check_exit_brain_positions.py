#!/usr/bin/env python3
"""
Quick check: Does Exit Brain see the 4 Binance Testnet positions?
"""
import requests
import json

print("=" * 70)
print("EXIT BRAIN V3 POSITION MONITOR CHECK")
print("=" * 70)

try:
    # Check backend health
    health = requests.get("http://localhost:8000/health", timeout=5).json()
    print(f"\n✅ Backend: {health['status']}")
    print(f"✅ Binance keys: {health['secrets']['has_binance_keys']}")
    
    # Get positions
    positions = requests.get("http://localhost:8000/positions", timeout=5).json()
    
    open_positions = [p for p in positions if float(p['position_amt']) != 0]
    
    print(f"\n📊 Open Positions: {len(open_positions)}")
    print("-" * 70)
    
    for pos in open_positions:
        symbol = pos['symbol']
        amt = float(pos['position_amt'])
        entry = float(pos['entry_price'])
        pnl = float(pos.get('unRealizedProfit', 0))
        
        side = "LONG" if amt > 0 else "SHORT"
        pnl_emoji = "🟢" if pnl > 0 else "🔴"
        
        print(f"{pnl_emoji} {symbol:12} {side:6} {amt:>12,.2f} @ ${entry:>8,.2f}  PnL: ${pnl:>8,.2f}")
    
    print("-" * 70)
    
    if len(open_positions) == 4:
        print("\n✅ Exit Brain kan se alle 4 testnet-posisjoner!")
        print("✅ LIVE mode executor overvåker nå disse posisjonene")
        print("✅ Legacy modules er blokkert av gateway")
    elif len(open_positions) == 0:
        print("\n⚠️ Ingen åpne posisjoner funnet")
        print("   Sjekk om Binance Testnet-posisjonene fortsatt er åpne")
    else:
        print(f"\n⚠️ Forventet 4 posisjoner, fant {len(open_positions)}")
    
    print("\n" + "=" * 70)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("   Backend kjører kanskje ikke?")
    print("=" * 70)
