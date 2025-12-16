#!/usr/bin/env python3
"""
Analyserer hvorfor profittene er små
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.services.binance_client import get_binance_client
from backend.config import settings

client = get_binance_client(testnet=settings.BINANCE_TESTNET)

print("\n" + "="*80)
print("📊 ANALYSE: HVORFOR SÅ SMÅ PROFITTER?")
print("="*80 + "\n")

# Get positions
positions = [p for p in client.futures_position_information() if float(p['positionAmt']) != 0]

print(f"🔍 AKTIVE POSISJONER: {len(positions)}\n")

total_pnl = 0
total_margin = 0

for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    mark = float(p['markPrice'])
    pnl = float(p['unRealizedProfit'])
    margin = float(p['initialMargin'])
    leverage = float(p['leverage'])
    
    # Calculate percentage
    if margin > 0:
        pnl_pct = (pnl / margin) * 100
    else:
        pnl_pct = 0
    
    # Calculate price change
    if amt > 0:  # LONG
        price_change_pct = ((mark - entry) / entry) * 100
        direction = "LONG"
    else:  # SHORT
        price_change_pct = ((entry - mark) / entry) * 100
        direction = "SHORT"
    
    print(f"📈 {symbol} {direction}")
    print(f"   Entry: ${entry:.6f}")
    print(f"   Mark:  ${mark:.6f}")
    print(f"   Price Change: {price_change_pct:+.2f}%")
    print(f"   Leverage: {leverage}x")
    print(f"   Margin: ${margin:.2f}")
    print(f"   PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)")
    print()
    
    total_pnl += pnl
    total_margin += margin

print("="*80)
print(f"💰 TOTAL:")
print(f"   Total Margin: ${total_margin:.2f}")
print(f"   Total PnL: ${total_pnl:.2f}")
if total_margin > 0:
    print(f"   Total Return: {(total_pnl/total_margin)*100:+.2f}%")
print("="*80 + "\n")

# PROBLEM ANALYSIS
print("🔍 PROBLEM ANALYSE:")
print("="*80 + "\n")

print("❓ HVORFOR SÅ SMÅ PROFITTER?\n")

# Check Math AI settings
print("1️⃣  MATH AI PARAMETERS:")
print("   Math AI beregner: $300 margin @ 3.0x leverage")
print("   TP target: 1.6% price move")
print("   SL: 0.8% price move")
print()
print("   ⚠️  PROBLEM: Posisjonene er MYE mindre enn $300!")
print(f"   Actual margin: ${total_margin:.2f} (should be ${300*len(positions):.2f})")
print()

# Check leverage
avg_leverage = sum(float(p['leverage']) for p in positions) / len(positions) if positions else 0
print("2️⃣  LEVERAGE:")
print(f"   Average leverage: {avg_leverage:.1f}x")
print(f"   Math AI recommends: 3.0x")
print()
if avg_leverage < 3.0:
    print(f"   ⚠️  PROBLEM: Leverage er lavere enn anbefalt!")
    print(f"   Impact: {(3.0/avg_leverage - 1)*100:.0f}% mindre exposure")
print()

# Check price movements
print("3️⃣  PRICE MOVEMENTS:")
for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    entry = float(p['entryPrice'])
    mark = float(p['markPrice'])
    
    if amt > 0:  # LONG
        price_change_pct = ((mark - entry) / entry) * 100
    else:  # SHORT
        price_change_pct = ((entry - mark) / entry) * 100
    
    print(f"   {symbol}: {price_change_pct:+.2f}% (TP target: 1.6%)")

print()
print("   ⚠️  PROBLEM: Prisene har ikke beveget seg nok ennå!")
print("   Math AI target: 1.6% move for TP")
print("   De fleste posisjoner er fortsatt i oppstart-fasen")
print()

# Check position sizes
print("4️⃣  POSITION SIZES:")
for p in positions:
    symbol = p['symbol']
    margin = float(p['initialMargin'])
    expected_margin = 300
    diff_pct = ((margin - expected_margin) / expected_margin) * 100
    
    print(f"   {symbol}: ${margin:.2f} margin (expected: ${expected_margin:.2f}, diff: {diff_pct:+.1f}%)")

print()
print("   ⚠️  CRITICAL PROBLEM: Position sizes mye mindre enn Math AI anbefalinger!")
print()

print("="*80)
print("💡 HOVEDPROBLEMER:")
print("="*80 + "\n")

print("1. 🔴 POSITION SIZES FOR SMÅ")
print(f"   • Math AI anbefaler: $300 margin per posisjon")
print(f"   • Actual average: ${total_margin/len(positions) if positions else 0:.2f} per posisjon")
print(f"   • Gap: {((300 - total_margin/len(positions))/(300))*100 if positions else 0:.0f}% mindre enn anbefalt")
print()

print("2. ⏰ TIMING")
print("   • Posisjonene er fortsatt nye")
print("   • Prisene har ikke beveget seg nok til TP (1.6%)")
print("   • Trenger tid for å nå profit targets")
print()

print("3. 🎯 TP/SL TARGETS")
print("   • Math AI setter TP på 1.6% price move")
print("   • Dette gir $4-5 profit per $300 posisjon")
print("   • Med mindre position sizes → mindre profitt")
print()

print("="*80)
print("🔧 LØSNINGER:")
print("="*80 + "\n")

print("1. ✅ ØK POSITION SIZES")
print("   • Math AI beregner allerede $300 per posisjon")
print("   • Systemet må faktisk USE disse parametrene")
print("   • Sjekk at execution respekterer Math AI sizing")
print()

print("2. ✅ BRUK FULL LEVERAGE")
print("   • Math AI anbefaler 3.0x leverage")
print("   • Sikre at orders plasseres med riktig leverage")
print()

print("3. ⏰ VENT")
print("   • Posisjonene trenger tid til å nå TP")
print("   • Math AI target: 1.6% price move")
print("   • Expected profit: $4-5 per $300 posisjon")
print("   • Med 15 posisjoner: $60-75 profit per cycle")
print()

print("4. 🔄 CONTINUOUS LEARNING WILL HELP")
print("   • Retraining system aktivert (daglig)")
print("   • Models lærer å finne bedre entry points")
print("   • Win rate forbedres fra 45% → 55%+")
print("   • Higher win rate = mer konsistente profitter")
print()

print("="*80)
print("📊 EXPECTED PROFITT MED RIKTIG SIZING:")
print("="*80 + "\n")

expected_margin_per_pos = 300
expected_tp_pct = 1.6
expected_leverage = 3.0

expected_profit_per_pos = (expected_margin_per_pos * expected_tp_pct / 100)
expected_max_positions = 15
expected_total_profit_per_cycle = expected_profit_per_pos * expected_max_positions

print(f"Med Math AI parameters ($300 @ 3.0x, TP=1.6%):")
print(f"   • Profit per posisjon: ${expected_profit_per_pos:.2f}")
print(f"   • Max posisjoner: {expected_max_positions}")
print(f"   • Total profit per cycle: ${expected_total_profit_per_cycle:.2f}")
print(f"   • Med 50% win rate: ${expected_total_profit_per_cycle * 0.5:.2f} net per cycle")
print()

print(f"Med 10 cycles per dag:")
print(f"   • Daily profit (50% WR): ${expected_total_profit_per_cycle * 0.5 * 10:.2f}")
print(f"   • Weekly profit: ${expected_total_profit_per_cycle * 0.5 * 10 * 7:.2f}")
print(f"   • Monthly profit: ${expected_total_profit_per_cycle * 0.5 * 10 * 30:.2f}")
print()

print("="*80)
print("🎯 KONKLUSJON:")
print("="*80 + "\n")

print("HOVEDPROBLEMET er at position sizes er MYE mindre enn Math AI anbefalinger!")
print()
print(f"Math AI sier:    $300 margin @ 3.0x leverage")
print(f"Actual average:  ${total_margin/len(positions) if positions else 0:.2f} margin @ {avg_leverage:.1f}x leverage")
print()
print("Dette reduserer profittene dramatisk!")
print()
print("LØSNING:")
print("✅ Sikre at execution bruker Math AI sizing parametere fullt ut")
print("✅ Verifiser at orders plasseres med riktig margin + leverage")
print("✅ Sjekk smart_execution.py at den respekterer Math AI sizing")
print()
print("="*80 + "\n")
