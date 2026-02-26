#!/usr/bin/env python3
"""
FINAL FIX: Add guard to skip non-executable apply.result entries.
Only execute TradeIntent parse when PATH1B rebuilt signal_data with 'action'.
"""
import subprocess, shutil, time

SRC = "/home/qt/quantum_trader/services/execution_service.py"

with open(SRC) as f:
    content = f.read()

# ─── The anchor: SKIP block → TradeIntent filter block gap ────────────────
# Lines 1158-1168 (SKIP check → allowed_fields):
OLD = """            # P0 FIX Feb 19: Handle SKIP decisions (do not parse as TradeIntent)
            if signal_data.get("decision") == "SKIP":
                logger.debug(f"[PATH1B] ACK SKIP {symbol}: {signal_data.get('error', 'no_error')}")
                if stream_name and group_name:
                    try:
                        await eventbus.redis.xack(stream_name, group_name, msg_id)
                    except Exception as ack_err:
                        logger.error(f"Failed to ACK SKIP {msg_id}: {ack_err}")
                continue

            # C3-FIX: Unwrap 'payload' JSON envelope from AI engine trade.intent messages"""

NEW = """            # P0 FIX Feb 19: Handle SKIP decisions (do not parse as TradeIntent)
            if signal_data.get("decision") in ("SKIP", "BLOCKED"):
                logger.debug(f"[PATH1B] ACK SKIP/BLOCKED {symbol}: decision={signal_data.get('decision')} error={signal_data.get('error', 'no_error')}")
                if stream_name and group_name:
                    try:
                        await eventbus.redis.xack(stream_name, group_name, msg_id)
                    except Exception as ack_err:
                        logger.error(f"Failed to ACK SKIP {msg_id}: {ack_err}")
                continue

            # C3-FIX-2: Guard — only process entries that were rebuilt by PATH1B (have 'action')
            # apply.result entries with decision=None/absent are informational (not executed) reports
            # from intent-executor. They have no action/confidence and must not trigger order placement.
            if 'action' not in signal_data:
                logger.debug(f"[ACK-INFO] {symbol}: no action in signal_data (decision={signal_data.get('decision', 'absent')}), ACKing as informational")
                if stream_name and group_name:
                    try:
                        await eventbus.redis.xack(stream_name, group_name, msg_id)
                    except Exception:
                        pass
                continue

            # C3-FIX: Unwrap 'payload' JSON envelope from AI engine trade.intent messages"""

if OLD in content:
    backup = SRC + f".bak.guard.{int(time.time())}"
    shutil.copy(SRC, backup)
    print(f"✅ Backup: {backup}")
    patched = content.replace(OLD, NEW, 1)
    with open(SRC, "w") as f:
        f.write(patched)
    print("✅ Applied C3-FIX-2 guard patch")
else:
    # Try alternative anchor (if C3-FIX block is slightly different)
    import re
    # Show what we actually have around the skip check
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if "P0 FIX Feb 19" in line or "ACK SKIP" in line or "C3-FIX" in line:
            print(f"  LINE {i+1}: {repr(line)}")
    print("\n⚠ Exact string not found — showing key area:")
    for i, line in enumerate(lines[1155:1175], 1156):
        print(f"  {i}: {line}")
    import sys; sys.exit(1)

# Verify
with open(SRC) as f:
    new_content = f.read()
if "C3-FIX-2: Guard" in new_content:
    print("✅ C3-FIX-2 guard verified in file")
else:
    print("❌ Guard verification failed")
    import sys; sys.exit(1)

# Restart
print("\n=== Restart quantum-execution ===")
subprocess.run(["systemctl", "daemon-reload"])
subprocess.run(["systemctl", "restart", "quantum-execution"])
time.sleep(3)
status = subprocess.run(
    ["systemctl", "is-active", "quantum-execution"],
    capture_output=True, text=True
).stdout.strip()
print(f"  Status: {status}")

# Wait for AI cycle + check log
print("  Waiting 65s for AI cycle...")
time.sleep(65)

log = subprocess.run(
    ["tail", "-n", "80", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
).stdout

print("\n=== execution.log last 80 lines ===")
has_error = False
for line in log.splitlines():
    if "❌" in line or "TradeIntent" in line or "ERROR" in line:
        has_error = True
        print(f"  ERROR: {line.strip()[-200:]}")
    elif any(k in line for k in ["EXEC CLOSE", "execute_order", "placed", "FILLED", "SUBMIT", "order_result", "intent_executor"]):
        print(f"  EXEC:  {line.strip()[-200:]}")
    elif "ACK-INFO" in line or "SKIP/BLOCKED" in line:
        print(f"  SKIP:  {line.strip()[-150:]}")

if not has_error:
    print("  ✅ No TradeIntent errors in latest log")

# Check execution.result
import redis
r = redis.Redis()
info = r.xinfo_stream("quantum:stream:execution.result")
print(f"\n=== execution.result ===")
print(f"  Length: {info['length']}")
last = r.xrevrange("quantum:stream:execution.result", "+", "-", count=1)
if last:
    ts = last[0][0].decode()
    print(f"  Newest: {ts}")
