#!/usr/bin/env python3
"""
FULL SYSTEM STATUS CHECK
- Current open positions and their PnL
- Total unrealized PnL
- RL Agent activity
- AI model predictions
- Smart Position Sizer status
- Learning activity
"""
import subprocess
import re
import json
from datetime import datetime

print("="*80)
print("🔍 QUANTUM TRADER - FULL SYSTEM STATUS")
print("="*80)
print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Get recent Docker logs
try:
    result = subprocess.run(
        ['docker', 'logs', 'quantum_backend', '--tail', '500', '2>&1'],
        capture_output=True,
        text=True,
        timeout=20
    )
    logs = result.stdout
except Exception as e:
    print(f"❌ Error fetching service logs: {e}")
    exit(1)

# 1. OPEN POSITIONS & PnL
print("📊 OPEN POSITIONS & PnL:")
print("-" * 80)

# Parse position monitor entries
position_pattern = r'\[CHART\]\s+(\w+):\s+PnL\s+([-+]?\d+\.?\d*)%\s+margin\s+\(\+\$([-+]?\d+\.?\d*)\s+USDT\)'
positions = re.findall(position_pattern, logs)

if not positions:
    print("⚠️  No open positions found in recent logs")
else:
    # Get unique positions (last occurrence)
    unique_positions = {}
    for symbol, pnl_pct, pnl_usdt in positions:
        unique_positions[symbol] = {
            'pnl_pct': float(pnl_pct),
            'pnl_usdt': float(pnl_usdt)
        }
    
    print(f"Total Open Positions: {len(unique_positions)}\n")
    print(f"{'Symbol':<12} {'PnL %':<12} {'PnL USDT':<12} {'Status':<10}")
    print("-" * 80)
    
    total_pnl = 0.0
    profitable = 0
    losing = 0
    
    for symbol, data in sorted(unique_positions.items(), key=lambda x: x[1]['pnl_pct'], reverse=True):
        pnl_pct = data['pnl_pct']
        pnl_usdt = data['pnl_usdt']
        
        status = "🟢 PROFIT" if pnl_pct > 0 else "🔴 LOSS"
        if pnl_pct > 0:
            profitable += 1
        else:
            losing += 1
        
        total_pnl += pnl_usdt
        print(f"{symbol:<12} {pnl_pct:>10.2f}% {pnl_usdt:>10.2f} USDT {status}")
    
    print("-" * 80)
    print(f"{'TOTAL':<12} {'':<12} {total_pnl:>10.2f} USDT")
    print(f"\n🎯 Win/Loss Split: {profitable} profitable, {losing} losing")
    
    if len(unique_positions) > 0:
        win_rate = (profitable / len(unique_positions)) * 100
        print(f"📊 Current Win Rate: {win_rate:.1f}%")
    
    if total_pnl > 0:
        print(f"✅ STATUS: IN PROFIT (+{total_pnl:.2f} USDT)")
    else:
        print(f"❌ STATUS: IN LOSS ({total_pnl:.2f} USDT)")

# 2. RL AGENT ACTIVITY
print("\n\n🤖 RL AGENT STATUS:")
print("-" * 80)

rl_exploit = len(re.findall(r'Exploiting.*balanced.*Q=', logs))
rl_explore = len(re.findall(r'Exploring', logs))

print(f"Exploitation actions: {rl_exploit}")
print(f"Exploration actions: {rl_explore}")

if rl_exploit > 0:
    print("✅ RL Agent is ACTIVE and using learned strategies")
else:
    print("⚠️  RL Agent not actively trading")

# Extract RL TP/SL settings
tpsl_match = re.search(r'TP=([\d.]+)%.*SL=([\d.]+)%', logs)
if tpsl_match:
    tp, sl = tpsl_match.groups()
    print(f"📍 Current TP/SL: TP={tp}%, SL={sl}%")

# 3. AI SIGNALS
print("\n\n🧠 AI ENSEMBLE SIGNALS:")
print("-" * 80)

signals_pattern = r'AI signals generated for \d+ symbols: BUY=(\d+) SELL=(\d+) HOLD=(\d+)'
signals_match = re.search(signals_pattern, logs)

if signals_match:
    buy, sell, hold = signals_match.groups()
    print(f"BUY:  {buy} signals")
    print(f"SELL: {sell} signals")
    print(f"HOLD: {hold} signals")
    
    # Check confidence
    conf_pattern = r'conf avg=([\d.]+) max=([\d.]+)'
    conf_match = re.search(conf_pattern, logs)
    if conf_match:
        avg_conf, max_conf = conf_match.groups()
        print(f"📊 Confidence: avg={avg_conf}, max={max_conf}")
else:
    print("⚠️  No recent AI signals found")

# Check for blocked signals
blocked_pattern = r'Portfolio Balancer blocked (\d+)/(\d+) signals'
blocked_match = re.search(blocked_pattern, logs)
if blocked_match:
    blocked, total = blocked_match.groups()
    print(f"⚠️  {blocked}/{total} signals BLOCKED (portfolio full)")

# 4. SMART POSITION SIZER
print("\n\n💡 SMART POSITION SIZER:")
print("-" * 80)

sps_logs = len(re.findall(r'Smart Position Sizer', logs, re.IGNORECASE))
if sps_logs > 0:
    print(f"✅ Active ({sps_logs} log entries)")
else:
    print("⚠️  No activity detected")

# Check for win rate tracking
win_rate_pattern = r'win rate.*?([\d.]+)%'
wr_matches = re.findall(win_rate_pattern, logs, re.IGNORECASE)
if wr_matches:
    latest_wr = wr_matches[-1]
    print(f"📊 Tracked Win Rate: {latest_wr}%")
    if float(latest_wr) < 30:
        print("❌ CRITICAL: Win rate <30% - Emergency stop should trigger!")
else:
    print("⚠️  No win rate data yet (waiting for closed positions)")

# 5. LEARNING ACTIVITY
print("\n\n📚 LEARNING & UPDATES:")
print("-" * 80)

# Q-table updates
qtable_updates = len(re.findall(r'Q-table.*updat', logs, re.IGNORECASE))
print(f"Q-table updates: {qtable_updates}")

# Meta-strategy updates
meta_updates = len(re.findall(r'Meta.*strategy', logs, re.IGNORECASE))
print(f"Meta-strategy updates: {meta_updates}")

# Closed positions (for learning)
closed_positions = len(re.findall(r'Position closed|Detected \d+ closed', logs, re.IGNORECASE))
print(f"Closed positions detected: {closed_positions}")

if closed_positions == 0:
    print("\n⚠️  NO CLOSED POSITIONS - Learning cannot progress!")
    print("💡 This means:")
    print("   - RL Agent cannot update Q-values")
    print("   - Smart Position Sizer has no data")
    print("   - No performance feedback loop")
else:
    print(f"\n✅ Learning active: {closed_positions} positions used for updates")

# 6. SYSTEM HEALTH
print("\n\n🏥 SYSTEM HEALTH:")
print("-" * 80)

# Check for errors
errors = len(re.findall(r'"level":\s*"ERROR"', logs))
warnings = len(re.findall(r'"level":\s*"WARN"', logs))

print(f"Errors (recent): {errors}")
print(f"Warnings (recent): {warnings}")

if errors > 10:
    print("❌ HIGH ERROR COUNT - System may have issues")
elif errors > 0:
    print("⚠️  Some errors detected - review logs")
else:
    print("✅ No errors in recent logs")

# Check for trade approvals
approvals = len(re.findall(r'TRADE APPROVED', logs))
print(f"\nTrade approvals (recent): {approvals}")

if approvals > 0:
    print("✅ System is actively approving trades")
else:
    print("⚠️  No recent trade approvals (portfolio may be full)")

# 7. PORTFOLIO STATUS
print("\n\n💼 PORTFOLIO STATUS:")
print("-" * 80)

# Check portfolio limit blocks
limit_blocks = len(re.findall(r'Portfolio limit reached', logs))
print(f"Portfolio limit blocks: {limit_blocks}")

if limit_blocks > 100:
    print("❌ CRITICAL: Portfolio constantly full - no rotation!")
    print("   IMPACT: High-quality signals cannot execute")
elif limit_blocks > 10:
    print("⚠️  Portfolio frequently full")
else:
    print("✅ Portfolio has capacity")

# 8. SUMMARY
print("\n\n" + "="*80)
print("📋 SUMMARY:")
print("="*80)

if len(unique_positions) == 0:
    print("❌ NO OPEN POSITIONS - System may not be trading")
elif total_pnl > 0:
    print(f"✅ PROFITABLE: {len(unique_positions)} positions, +{total_pnl:.2f} USDT unrealized")
else:
    print(f"⚠️  IN LOSS: {len(unique_positions)} positions, {total_pnl:.2f} USDT unrealized")

if closed_positions == 0:
    print("❌ NO POSITIONS CLOSING - TP/SL levels too wide")
    print("   ACTION NEEDED: Reduce TP from 6% to 3%, SL from 2.5% to 1.5%")

if rl_exploit == 0:
    print("⚠️  RL Agent not actively using strategies")
elif closed_positions == 0:
    print("⚠️  RL Agent active but cannot learn (no closed positions)")
else:
    print("✅ RL Agent learning from closed positions")

if limit_blocks > 100:
    print("❌ Portfolio rotation blocked - need faster position closes")

print("\n" + "="*80)
