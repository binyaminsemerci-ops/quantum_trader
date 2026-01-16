#!/usr/bin/env python3
"""
System Status Checker - Alle nye fikser og forbedringer
"""
import subprocess
import re
from datetime import datetime

print("="*80)
print(f"🤖 QUANTUM TRADER - SYSTEM STATUS RAPPORT")
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# Hent alle logs
logs = subprocess.run(
    ["journalctl", "-u", "quantum-backend.service"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='ignore'
).stderr

print("\n📊 TRADE AKTIVITET:")
print("-" * 80)

# Tell TRADE APPROVED
approved = len(re.findall(r'TRADE APPROVED', logs))
print(f"  ✅ Trades godkjent: {approved}")

# Tell closed positions
closed_matches = re.findall(r'Detected (\d+) closed positions', logs)
total_closed = sum(int(m) for m in closed_matches)
print(f"  🔒 Positions lukket: {total_closed}")

# Tell portfolio limit blocks
blocks = len(re.findall(r'Portfolio limit reached', logs))
print(f"  🚫 Blokkert (portfolio full): {blocks}")

print("\n🧠 RL AGENT & LÆRING:")
print("-" * 80)

# RL updates
rl_updates = len(re.findall(r'updating Meta-Strategy rewards', logs))
print(f"  📈 RL læring events: {rl_updates}")

# RL Q-table updates
q_updates = re.findall(r'Q-table.*?(\d+)\s+updates', logs)
if q_updates:
    print(f"  🎯 Q-table updates: {q_updates[-1]}")
else:
    print(f"  🎯 Q-table updates: Teller...")

# RL exploitation vs exploration
exploiting = len(re.findall(r'Exploiting:', logs))
exploring = len(re.findall(r'Exploring:', logs))
print(f"  🔍 RL Exploitation: {exploiting} | Exploration: {exploring}")

if exploiting + exploring > 0:
    exploit_pct = exploiting / (exploiting + exploring) * 100
    print(f"     Exploitation rate: {exploit_pct:.1f}%")

print("\n💡 SMART POSITION SIZER:")
print("-" * 80)

# Smart Position Sizer aktivitet
smart_logs = re.findall(r'Smart Position Sizer|SmartPositionSizer', logs)
if smart_logs:
    print(f"  ✅ Smart Sizer aktiv: {len(smart_logs)} kall")
else:
    print(f"  ⚠️  Smart Sizer: Ingen logs funnet (deployet nylig)")

# Win rate tracking
win_rate_logs = re.findall(r'win.*rate.*?(\d+\.?\d*)%', logs, re.IGNORECASE)
if win_rate_logs:
    print(f"  📊 Win rates tracked: {len(win_rate_logs)} entries")
    print(f"     Latest: {win_rate_logs[-1]}%")
else:
    print(f"  📊 Win rate tracking: Venter på closed trades...")

print("\n🎯 TP/SL FORBEDRINGER (RL-TPSL):")
print("-" * 80)

# RL TP/SL strategies
conservative = len(re.findall(r'Strategy=conservative', logs))
balanced = len(re.findall(r'Strategy=balanced', logs))
aggressive = len(re.findall(r'Strategy=aggressive', logs))

print(f"  🛡️  Conservative TP/SL: {conservative}")
print(f"  ⚖️  Balanced TP/SL: {balanced}")
print(f"  🚀 Aggressive TP/SL: {aggressive}")

# Extract TP/SL levels
tp_levels = re.findall(r'TP=(\d+\.?\d*)%', logs)
sl_levels = re.findall(r'SL=(\d+\.?\d*)%', logs)

if tp_levels and sl_levels:
    avg_tp = sum(float(tp) for tp in tp_levels) / len(tp_levels)
    avg_sl = sum(float(sl) for sl in sl_levels) / len(sl_levels)
    print(f"  📏 Avg TP: {avg_tp:.1f}% | Avg SL: {avg_sl:.1f}%")
    print(f"  🎲 Risk/Reward Ratio: {avg_tp/avg_sl:.2f}x")

print("\n⚙️ SYSTEM KONFIGURASJON:")
print("-" * 80)

# Portfolio limits
max_concurrent = re.findall(r'Max concurrent trades:\s*(\d+)', logs)
max_positions = re.findall(r'Max Positions:\s*(\d+)', logs)

if max_concurrent:
    print(f"  📦 Max concurrent trades: {max_concurrent[-1]}")
if max_positions:
    print(f"  📦 Max positions (PBA): {max_positions[-1]}")

# Confidence threshold
confidence = re.findall(r'confidence\s*>=\s*(\d+\.?\d*)', logs)
if confidence:
    print(f"  🎯 Confidence threshold: {confidence[-1]}")

print("\n🔄 SISTE 10 MINUTTER:")
print("-" * 80)

# Get recent activity
recent_logs = subprocess.run(
    ["journalctl", "-u", "quantum-backend.service", "--since", "10m"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='ignore'
).stderr

recent_approved = len(re.findall(r'TRADE APPROVED', recent_logs))
recent_closed = len(re.findall(r'Detected \d+ closed positions', recent_logs))
recent_blocks = len(re.findall(r'Portfolio limit reached', recent_logs))

print(f"  ✅ Trades approved: {recent_approved}")
print(f"  🔒 Positions closed: {recent_closed}")
print(f"  🚫 Blocked by limit: {recent_blocks}")

if recent_blocks > recent_approved * 2:
    print(f"\n  ⚠️  WARNING: Portfolio limit blokkerer mange trades!")
    print(f"     Mange positions er åpne men ikke lukket enda.")
elif recent_approved > 0:
    print(f"\n  ✅ System er aktivt og trader normalt")
else:
    print(f"\n  💤 Lite aktivitet siste 10 min (normal hvis low volatility)")

print("\n" + "="*80)
print("📝 KONKLUSJON:")
print("-" * 80)

if total_closed > 20:
    print(f"✅ RL Agent lærer: {total_closed} closed trades prosessert")
else:
    print(f"⏳ RL Agent warming up: {total_closed} trades så langt")

if exploiting > 0:
    print(f"✅ RL bruker lærte strategier ({exploit_pct:.0f}% exploitation)")

if balanced + conservative + aggressive > 50:
    print(f"✅ Dynamic TP/SL aktiv: {balanced + conservative + aggressive} justeringer")

if recent_approved > 0:
    print(f"✅ Systemet trader aktivt: {recent_approved} nye trades siste 10 min")

print("\n💡 NESTE STEG:")
print("-" * 80)
print("  1. La systemet kjøre 24-48 timer for mer læring")
print("  2. Smart Position Sizer vil blokkere trades hvis win rate < 30%")
print("  3. RL Agent vil optimalisere TP/SL basert på outcomes")
print("  4. Monitor performance med: journalctl -u quantum-backend.service | grep 'RL\\|Smart'")
print("="*80)

