#!/usr/bin/env python3
"""Direct line-based patch for execution_service.py guard"""
import shutil, time

SRC = "/home/qt/quantum_trader/services/execution_service.py"

with open(SRC) as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Find the SKIP check line
skip_line_idx = None
c3fix_line_idx = None
for i, line in enumerate(lines):
    if "Handle SKIP decisions (do not parse as TradeIntent)" in line:
        skip_line_idx = i
    if "C3-FIX: Unwrap 'payload' JSON envelope" in line:
        c3fix_line_idx = i

print(f"SKIP check block starts: line {skip_line_idx + 1}")
print(f"C3-FIX unwrap starts: line {c3fix_line_idx + 1}")

# Verify: show lines skip_line_idx to c3fix_line_idx
for i in range(skip_line_idx, c3fix_line_idx + 1):
    print(f"  {i+1}: {repr(lines[i].rstrip())}")

# Find the `decision == "SKIP"` condition line
skip_cond_idx = None
for i in range(skip_line_idx, skip_line_idx + 3):
    if '"SKIP"' in lines[i]:
        skip_cond_idx = i
        break

print(f"\nSKIP condition line: {skip_cond_idx + 1}: {repr(lines[skip_cond_idx].rstrip())}")

# Find the `continue` line after the SKIP block (it's the last `continue` before c3fix_line_idx)
continue_idx = None
for i in range(skip_cond_idx, c3fix_line_idx):
    if lines[i].strip() == "continue":
        continue_idx = i  # will be the LAST one found

print(f"continue line (end of SKIP block): {continue_idx + 1}: {repr(lines[continue_idx].rstrip())}")

# Determine indentation prefix from line skip_line_idx
indent = ""
for ch in lines[skip_line_idx]:
    if ch == " ":
        indent += " "
    else:
        break

print(f"Indent: {repr(indent)} ({len(indent)} spaces)")

# Build new lines to insert AFTER continue_idx
GUARD_LINES = [
    "\n",  # blank line
    f"{indent}# C3-FIX-2: Guard — skip non-executable apply.result entries (no action = informational)\n",
    f"{indent}# apply.result entries with decision=None/absent/BLOCKED are intent_executor reports\n",
    f"{indent}# that the plan was NOT executed. They must not trigger TradeIntent order placement.\n",
    f"{indent}if signal_data.get('decision') == 'BLOCKED':\n",
    f"{indent}    logger.debug(f\"[ACK-BLOCKED] {{symbol}}: decision=BLOCKED, ACKing\")\n",
    f"{indent}    if stream_name and group_name:\n",
    f"{indent}        try:\n",
    f"{indent}            await eventbus.redis.xack(stream_name, group_name, msg_id)\n",
    f"{indent}        except Exception:\n",
    f"{indent}            pass\n",
    f"{indent}    continue\n",
    "\n",
    f"{indent}if 'action' not in signal_data:\n",
    f"{indent}    logger.debug(f\"[ACK-INFO] {{symbol}}: no action in signal_data (decision={{signal_data.get('decision', 'absent')}}), ACKing as informational\")\n",
    f"{indent}    if stream_name and group_name:\n",
    f"{indent}        try:\n",
    f"{indent}            await eventbus.redis.xack(stream_name, group_name, msg_id)\n",
    f"{indent}        except Exception:\n",
    f"{indent}            pass\n",
    f"{indent}    continue\n",
    "\n",
]

# Also fix the SKIP line to also handle BLOCKED in the same check
# (Optional: modify lines[skip_cond_idx] too)
old_skip_cond = lines[skip_cond_idx]
# Already handled via BLOCKED block above, no need to change

# Also fix: expand the SKIP debug log to mention BLOCKED
# Find the logger.debug line in the SKIP block
debug_idx = skip_cond_idx + 1
while "logger.debug" not in lines[debug_idx] and debug_idx < skip_cond_idx + 5:
    debug_idx += 1
if "ACK SKIP" in lines[debug_idx]:
    # Update the debug message to include SKIP/BLOCKED
    lines[debug_idx] = lines[debug_idx].replace(
        '"[PATH1B] ACK SKIP',
        '"[PATH1B] ACK SKIP/BLOCKED'
    )
    print(f"\nUpdated SKIP debug log: {repr(lines[debug_idx].rstrip())}")

backup = SRC + f".bak.guard2.{int(time.time())}"
shutil.copy(SRC, backup)
print(f"\n✅ Backup: {backup}")

# Insert guard lines AFTER continue_idx
new_lines = lines[:continue_idx + 1] + GUARD_LINES + lines[continue_idx + 1:]

with open(SRC, "w") as f:
    f.writelines(new_lines)

print(f"✅ Inserted {len(GUARD_LINES)} guard lines after line {continue_idx + 1}")

# Verify
with open(SRC) as f:
    content = f.read()
if "C3-FIX-2: Guard" in content and "action' not in signal_data" in content:
    print("✅ Guard verified in file")
else:
    print("❌ Guard verify failed")
    import sys; sys.exit(1)

# Restart
import subprocess
print("\n=== Restart quantum-execution ===")
subprocess.run(["systemctl", "daemon-reload"])
subprocess.run(["systemctl", "restart", "quantum-execution"])
import time as _time; _time.sleep(3)
status = subprocess.run(
    ["systemctl", "is-active", "quantum-execution"],
    capture_output=True, text=True
).stdout.strip()
print(f"  Status: {status}")

print("  Waiting 70s for AI cycle...")
_time.sleep(70)

log = subprocess.run(
    ["tail", "-n", "100", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
).stdout.splitlines()

errors = [l for l in log if "❌" in l or "TradeIntent" in l or ("ERROR" in l and "ACK" not in l)]
infos = [l for l in log if "ACK-INFO" in l or "ACK-BLOCKED" in l]
execs = [l for l in log if "EXEC CLOSE" in l or "execute_order" in l or "FILLED" in l]

print(f"\n=== Log summary (last 100 lines) ===")
print(f"  TradeIntent errors: {len(errors)}")
print(f"  ACK-INFO/BLOCKED guards fired: {len(infos)}")
print(f"  Execution events: {len(execs)}")

for l in errors[:5]:
    print(f"  ERR: {l.strip()[-150:]}")
for l in infos[:3]:
    print(f"  INFO: {l.strip()[-150:]}")
for l in execs[:3]:
    print(f"  EXEC: {l.strip()[-150:]}")

import redis as _redis
r = _redis.Redis()
info = r.xinfo_stream("quantum:stream:execution.result")
print(f"\n=== execution.result: length={info['length']}, newest={r.xrevrange('quantum:stream:execution.result', count=1)[0][0].decode()} ===")
