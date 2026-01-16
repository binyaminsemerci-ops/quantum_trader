"""
Check how many trades are in active_trades and their states.
"""
import subprocess
import re

# Get last 500 lines of docker logs
result = subprocess.run(
    ['docker', 'logs', 'quantum_backend', '--tail', '500'],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='ignore'
)

logs = result.stdout + result.stderr

# Count how many trades are in active_trades
print("=" * 80)
print("🔍 ACTIVE_TRADES ANALYSIS")
print("=" * 80)

# Find active_trades dict size logs
active_trades_logs = []
for line in logs.split('\n'):
    if 'active_trades' in line.lower() or 'len(self.active_trades)' in line:
        active_trades_logs.append(line.strip())

if active_trades_logs:
    print("\n📊 Active Trades References:")
    for log in active_trades_logs[-10:]:  # Last 10
        print(f"  {log}")
else:
    print("\n❌ No active_trades dict size logs found")

# Find CLOSED state transitions
print("\n" + "=" * 80)
print("🔚 CLOSED TRADES (Last 20)")
print("=" * 80)

closed_logs = []
for line in logs.split('\n'):
    if 'CLOSED_SL' in line or 'CLOSED_TP' in line or 'CLOSED_TIME' in line or 'CLOSED_PARTIAL' in line:
        closed_logs.append(line.strip())

if closed_logs:
    for log in closed_logs[-20:]:
        print(f"  {log}")
else:
    print("\n✅ No closed trades in recent logs (no previous trading)")

# Check position count
print("\n" + "=" * 80)
print("📍 CURRENT POSITIONS")
print("=" * 80)

position_logs = []
for line in logs.split('\n'):
    if 'Current positions:' in line or 'BRIEFCASE' in line:
        position_logs.append(line.strip())

if position_logs:
    for log in position_logs[-10:]:
        print(f"  {log}")
else:
    print("\n❌ No position count logs found")

# Check max concurrent trades check
print("\n" + "=" * 80)
print("🚫 BLOCKED MESSAGES")
print("=" * 80)

blocked_logs = []
for line in logs.split('\n'):
    if 'Max concurrent trades reached' in line or 'BLOCKED' in line:
        blocked_logs.append(line.strip())

if blocked_logs:
    print(f"\n🔴 Found {len(blocked_logs)} BLOCKED messages")
    for log in blocked_logs[-10:]:
        print(f"  {log}")
else:
    print("\n✅ No BLOCKED messages (good!)")

# Conclusion
print("\n" + "=" * 80)
print("💡 DIAGNOSIS")
print("=" * 80)

# Try to extract numbers from blocked message
blocked_count = None
for line in blocked_logs[-5:]:
    match = re.search(r'Max concurrent trades reached:\s*(\d+)\s*/\s*(\d+)', line)
    if match:
        blocked_count = (int(match.group(1)), int(match.group(2)))
        break

# Try to extract from position logs
current_positions = None
for line in position_logs[-5:]:
    match = re.search(r'Current positions:\s*(\d+)/(\d+)', line)
    if match:
        current_positions = (int(match.group(1)), int(match.group(2)))
        break

if blocked_count and current_positions:
    print(f"\n🔴 BUG CONFIRMED:")
    print(f"   Position logs say: {current_positions[0]} positions open (out of {current_positions[1]} max)")
    print(f"   But GlobalRiskController thinks: {blocked_count[0]} concurrent trades (max {blocked_count[1]})")
    print(f"\n💥 Mismatch: {blocked_count[0]} != {current_positions[0]}")
    print(f"\n🔧 Root Cause:")
    print(f"   TradeLifecycleManager.active_trades has {blocked_count[0]} trades")
    print(f"   But _get_open_positions_info() should filter by state")
    print(f"   Likely: Closed trades not being removed from active_trades")
    print(f"           OR state filter not working properly")
elif blocked_count:
    print(f"\n🔴 GlobalRiskController blocking with: {blocked_count[0]}/{blocked_count[1]} trades")
elif current_positions:
    print(f"\n✅ System shows: {current_positions[0]} positions open")
else:
    print("\n⚠️  Insufficient log data to diagnose")

print("\n" + "=" * 80)

