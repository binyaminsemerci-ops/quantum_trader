#!/usr/bin/env python3
"""Check current AI status and recent activity."""
import os
import subprocess
from binance.client import Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

print("\n" + "=" * 70)
print("🤖 AI STATUS - REAL-TIME")
print("=" * 70)

# Get current positions
positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]

print(f"\n[CHART] AKTIVE POSISJONER: {len(positions)}")
if positions:
    for p in positions:
        amt = float(p['positionAmt'])
        pnl = float(p['unRealizedProfit'])
        print(f"   • {p['symbol']}: {'LONG' if amt > 0 else 'SHORT'} {abs(amt)} | P&L: ${pnl:.2f}")
else:
    print("   ⏳ Ingen posisjoner ennå - AI søker etter 70%+ confidence signals")

# Get account info
account = client.futures_account()
balance = float(account['totalWalletBalance'])
available = float(account['availableBalance'])

print(f"\n[MONEY] BALANCE:")
print(f"   Total: ${balance:.2f} USDT")
print(f"   Tilgjengelig: ${available:.2f} USDT")

# Check backend logs for recent activity
print(f"\n[SEARCH] SISTE AI AKTIVITET:")
try:
    result = subprocess.run(
        ["docker", "logs", "quantum_backend", "--tail", "30"],
        capture_output=True,
        text=True,
        timeout=5
    )
    
    lines = result.stdout.split('\n')
    
    # Look for important messages
    for line in reversed(lines):
        if 'Strong signals' in line or 'confidence' in line:
            # Extract timestamp
            if '"timestamp"' in line:
                try:
                    import json
                    log = json.loads(line)
                    ts = log.get('timestamp', '')
                    msg = log.get('message', '')
                    print(f"   [TARGET] {ts[-12:-7]}: {msg}")
                except:
                    pass
        elif 'Position Monitor' in line:
            if '"timestamp"' in line:
                try:
                    import json
                    log = json.loads(line)
                    ts = log.get('timestamp', '')
                    msg = log.get('message', '')
                    print(f"   [SEARCH] {ts[-12:-7]}: {msg}")
                    break
                except:
                    pass
except Exception as e:
    print(f"   [WARNING] Could not read logs: {e}")

print(f"\n⚙️ AI KONFIGURASJON:")
print(f"   • Event-Driven: [OK] Aktiv (hvert 10s)")
print(f"   • Min Confidence: 70%")
print(f"   • Symbols: 36")
print(f"   • Cooldown: 120s mellom trades")
print(f"   • Position Size: $1600 (20x leverage)")
print(f"   • TP: 3% / SL: 2% / Trail: 1.5%")

print(f"\n[TARGET] NESTE TRADE:")
print(f"   AI analyserer kontinuerlig.")
print(f"   Åpner trade når den finner 70%+ confidence signal.")
print(f"   Tid siden siste sjekk: Se docker logs")

print(f"\n💡 LIVE MONITORING:")
print(f"   docker logs quantum_backend --tail 50 --follow")

print("\n" + "=" * 70)
print(f"⏰ Status oppdatert: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 70 + "\n")
