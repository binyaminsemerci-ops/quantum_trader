#!/usr/bin/env python3
"""
1. Show execution_service lines 975-1015 (message handling context)
2. Apply TWO-PART patch to execution_service:
   a) Unwrap 'payload' JSON from AI-engine trade.intent messages
   b) Map 'side' -> 'action'
3. Restart service + verify
"""
import subprocess, shutil, time, sys

SRC = "/home/qt/quantum_trader/services/execution_service.py"

with open(SRC) as f:
    lines = f.readlines()
    content = "".join(lines)

print("=== Lines 975-1015 (message handling context) ===")
for i, line in enumerate(lines[974:1015], 975):
    print(f"  {i}: {line.rstrip()}")

# ─── Patch: insert payload unwrap + side→action before allowed_fields ─────
# Anchor is the exact 3 lines starting the filter block (lines 1168-1170):
OLD_BLOCK = """            # Filter signal_data to only include TradeIntent fields
            # BRIDGE-PATCH: Added ai_size_usd, ai_leverage, ai_harvest_policy for AI-driven sizing
            # P0.4C: Added reason, reduce_only for exit flow audit trail
            allowed_fields = {"""

NEW_BLOCK = """            # C3-FIX: Unwrap 'payload' JSON envelope from AI engine trade.intent messages
            # AI engine writes all signal fields inside a 'payload' JSON string field
            _payload_raw = signal_data.get('payload')
            if _payload_raw and isinstance(_payload_raw, str) and _payload_raw.startswith('{'):
                try:
                    import json as _json
                    _payload_parsed = _json.loads(_payload_raw)
                    for _pk, _pv in _payload_parsed.items():
                        _pk = str(_pk)
                        if _pk not in signal_data:
                            signal_data[_pk] = _pv  # keep original types
                except Exception as _pe:
                    logger.warning(f"\u26a0 Failed to decode payload JSON for {symbol}: {_pe}")
            # C3-FIX: Map 'side' -> 'action' for trade.intent with AI engine schema
            if 'action' not in signal_data and 'side' in signal_data:
                signal_data['action'] = signal_data['side']
            # Filter signal_data to only include TradeIntent fields
            # BRIDGE-PATCH: Added ai_size_usd, ai_leverage, ai_harvest_policy for AI-driven sizing
            # P0.4C: Added reason, reduce_only for exit flow audit trail
            allowed_fields = {"""

if OLD_BLOCK in content:
    backup = SRC + f".bak.payload.{int(time.time())}"
    shutil.copy(SRC, backup)
    print(f"\n✅ Backup: {backup}")
    with open(SRC, "w") as f:
        f.write(content.replace(OLD_BLOCK, NEW_BLOCK, 1))
    print("✅ Applied payload-unwrap + side→action patch")
else:
    print("\n⚠ OLD_BLOCK not found — dumping actual lines 1168-1170:")
    for i, line in enumerate(lines[1167:1170], 1168):
        print(f"  {i}: {repr(line)}")
    sys.exit(1)

# ─── Verify patch was applied ──────────────────────────────────────────────
with open(SRC) as f:
    new_content = f.read()
if "C3-FIX: Unwrap 'payload'" in new_content and "C3-FIX: Map 'side'" in new_content:
    print("✅ Both C3-FIX patches verified in file")
else:
    print("❌ Patch verification FAILED")
    sys.exit(1)

# ─── Restart service ──────────────────────────────────────────────────────
print("\n=== Restart quantum-execution ===")
subprocess.run(["systemctl", "daemon-reload"])
subprocess.run(["systemctl", "restart", "quantum-execution"])
time.sleep(4)
status = subprocess.run(
    ["systemctl", "is-active", "quantum-execution"],
    capture_output=True, text=True
).stdout.strip()
print(f"  Status: {status}")

# Wait 15s for fresh trade.intent messages to arrive
print("  Waiting 15s for fresh trade.intent messages...")
time.sleep(15)

# ─── Check logs for results ────────────────────────────────────────────────
print("\n=== execution.log last 60 lines (filtered) ===")
result = subprocess.run(
    ["tail", "-n", "60", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
)
for line in result.stdout.splitlines():
    if any(k in line for k in [
        "TradeIntent", "❌", "placing", "Placing", "order", "FILLED", "EXEC",
        "execute_order", "intent", "SELL", "BUY", "ERROR", "WARN", "C3-FIX",
        "payload", "action", "execution.result", "submitted", "binance"
    ]):
        print(f"  {line.strip()[-200:]}")

# ─── Check execution.result stream for new entries ────────────────────────
import redis
r = redis.Redis()
info = r.xinfo_stream("quantum:stream:execution.result")
last_entry = r.xrevrange("quantum:stream:execution.result", "+", "-", count=1)

print(f"\n=== execution.result stream status ===")
print(f"  Length: {info['length']}")
if last_entry:
    msg_ts = last_entry[0][0].decode()
    print(f"  Newest entry ID: {msg_ts}")
    for k, v in last_entry[0][1].items():
        print(f"    {k.decode()}: {v.decode()[:80]}")
