#!/usr/bin/env python3
"""
Fix execution_service guard: skip action=FULL_CLOSE_PROPOSED (and other non-trade actions).
Also report on Binance API errors from apply_layer close attempts.
"""
import subprocess, shutil, time, redis

SRC = "/home/qt/quantum_trader/services/execution_service.py"

# ─── 1. Report Binance errors from apply_layer ────────────────────────────
r = redis.Redis()
entries = r.xrevrange("quantum:stream:apply.result", count=500)

print("=== Recent apply.result entries with Binance errors ===")
for mid, d in entries:
    if b'error' in d and (b'Binance' in d.get(b'error', b'') or b'execution_failed' in d.get(b'error', b'')):
        symbol = d.get(b'symbol', b'?').decode()
        action = d.get(b'action', b'?').decode()
        error = d.get(b'error', b'').decode()[:120]
        ts = mid.decode()
        age_ms = int(time.time() * 1000) - int(ts.split('-')[0])
        age_min = age_ms // 60000
        print(f"  [{age_min}m ago] {symbol} action={action} error={error[:80]}")

# Count by error type
from collections import Counter
err_types = Counter()
for _, d in entries:
    err = d.get(b'error', b'').decode()
    if 'Binance' in err or 'execution_failed' in err:
        if '-2022' in err:
            err_types['ReduceOnly rejected (-2022)'] += 1
        elif '-1111' in err:
            err_types['Precision error (-1111)'] += 1
        elif '-2010' in err:
            err_types['InsufficientMargin (-2010)'] += 1
        elif '-1121' in err:
            err_types['Invalid symbol (-1121)'] += 1
        else:
            short = err[:50]
            err_types[short] += 1

print("\n=== Binance error distribution (last 500 apply.result entries) ===")
for k, v in sorted(err_types.items(), key=lambda x: -x[1]):
    print(f"  {v}x {k}")

# ─── 2. Fix execution_service guard: check for valid trade actions ────────
with open(SRC) as f:
    lines = f.readlines()

# Find the existing guard lines
guard_start_idx = None
guard_end_idx = None
for i, line in enumerate(lines):
    if "C3-FIX-2: Guard" in line:
        guard_start_idx = i
    if guard_start_idx is not None and "'action' not in signal_data" in line:
        guard_end_idx = i

print(f"\n=== Guard section (lines {guard_start_idx+1} to {guard_end_idx+1}) ===")
if guard_start_idx and guard_end_idx:
    for i in range(guard_start_idx, guard_end_idx + 8):
        print(f"  {i+1}: {lines[i].rstrip()[:150]}")
    
    # Replace the guard condition
    old_guard_line = lines[guard_end_idx]
    indent = len(old_guard_line) - len(old_guard_line.lstrip())
    spaces = ' ' * indent

    # Update the C3-FIX-2 comment and condition
    # Find exact range: C3-FIX-2 comment through the old `if 'action' not in signal_data:` block
    # Collect all lines from guard_start_idx to (guard_end_idx + ~8 lines for the body)
    body_end_idx = guard_end_idx
    while body_end_idx < len(lines) and (
        "continue" not in lines[body_end_idx] or 
        body_end_idx <= guard_end_idx
    ):
        body_end_idx += 1

    print(f"\n  Body span: lines {guard_start_idx+1} to {body_end_idx+1}")
    for i in range(guard_start_idx, body_end_idx + 1):
        print(f"  {i+1}: {repr(lines[i].rstrip())[:130]}")

    # Replace the old guard block with updated one that checks for valid trade directions
    # The old guard: if 'action' not in signal_data: ...continue
    # New guard: keep old (for missing action) AND add check for invalid action values

    backup = SRC + f".bak.guardfix.{int(time.time())}"
    shutil.copy(SRC, backup)
    print(f"\n✅ Backup: {backup}")

    # Build new guard lines (replace from guard_start_idx to body_end_idx inclusive)
    new_guard_lines = [
        f"\n",
        f"{spaces}# C3-FIX-2: Guard — skip non-trade apply.result entries\n",
        f"{spaces}# Skip entries without action OR with non-directional action values\n",
        f"{spaces}# (e.g., action=FULL_CLOSE_PROPOSED from apply_layer close reports)\n",
        f"{spaces}_valid_trade_actions = ('BUY', 'SELL', 'LONG', 'SHORT', 'buy', 'sell', 'long', 'short')\n",
        f"{spaces}if signal_data.get('action') not in _valid_trade_actions:\n",
        f"{spaces}    _action_val = signal_data.get('action', '<absent>')\n",
        f"{spaces}    logger.debug(f\"[ACK-NONTRADE] {{symbol}}: action={{_action_val!r}} is not a trade direction, skipping\")\n",
        f"{spaces}    if stream_name and group_name:\n",
        f"{spaces}        try:\n",
        f"{spaces}            await eventbus.redis.xack(stream_name, group_name, msg_id)\n",
        f"{spaces}        except Exception:\n",
        f"{spaces}            pass\n",
        f"{spaces}    continue\n",
        f"\n",
    ]

    # Identify lines to remove: from guard_start_idx - 1 (blank line) to body_end_idx
    # (The blank line before guard was inserted by our code)
    remove_start = guard_start_idx - 1 if lines[guard_start_idx - 1].strip() == '' else guard_start_idx
    new_lines = lines[:remove_start] + new_guard_lines + lines[body_end_idx + 1:]

    with open(SRC, "w") as f:
        f.writelines(new_lines)
    print(f"✅ Replaced guard with valid-action check")

    # Verify
    with open(SRC) as f:
        content = f.read()
    if "_valid_trade_actions" in content and "ACK-NONTRADE" in content:
        print("✅ Guard replacement verified")
    else:
        print("❌ Guard replacement verification failed")
        import sys; sys.exit(1)
else:
    print("❌ Could not find guard section")
    import sys; sys.exit(1)

# ─── 3. Restart service ──────────────────────────────────────────────────
print("\n=== Restart quantum-execution ===")
subprocess.run(["systemctl", "daemon-reload"])
subprocess.run(["systemctl", "restart", "quantum-execution"])
time.sleep(3)
status = subprocess.run(
    ["systemctl", "is-active", "quantum-execution"],
    capture_output=True, text=True
).stdout.strip()
print(f"  Status: {status}")

print("  Waiting 70s for AI cycle...")
time.sleep(70)

log = subprocess.run(
    ["tail", "-n", "100", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
).stdout.splitlines()

errors = [l for l in log if "❌" in l or ("TradeIntent" in l and "ERROR" in l)]
guards = [l for l in log if "ACK-NONTRADE" in l or "ACK-INFO" in l or "ACK-BLOCKED" in l]

print(f"\n=== Log summary (last 100 lines) ===")
print(f"  TradeIntent errors: {len(errors)}")
print(f"  Non-trade ACK guard fires: {len(guards)} (debug level - may show 0)")

for l in errors[:5]:
    print(f"  ERR: {l.strip()[-150:]}")
