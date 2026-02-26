#!/usr/bin/env python3
"""
1. Check TradeIntent model + how action is mapped from side  
2. Scan execution.log for TradeIntent errors
3. Update quantum-execution EnvironmentFile to include intent-executor.env
"""
import subprocess, re, shutil, time

# ─── 1. TradeIntent model + side→action mapping ───────────────────────────
SRC = "/home/qt/quantum_trader/services/execution_service.py"
with open(SRC) as f:
    lines = f.readlines()
    content = f.read() if False else "".join(lines)

print("=== execution_service.py lines 1-40 (imports + TradeIntent) ===")
for i, line in enumerate(lines[:40], 1):
    if any(k in line for k in ["TradeIntent", "import", "from", "BINANCE", "BASE_URL", "testnet"]):
        print(f"  {i:4}: {line.rstrip()}")

# Find TradeIntent definition  
print("\n=== TradeIntent class definition ===")
for i, line in enumerate(lines, 1):
    if "class TradeIntent" in line or ("TradeIntent" in line and "BaseModel" in line):
        for j in range(i-1, min(i+30, len(lines))):
            print(f"  {j+1:4}: {lines[j].rstrip()}")
        break

# Check if there's a `side` → `action` validator
print("\n=== validator or field_validator for action/side ===")
for i, line in enumerate(lines, 1):
    if "validator" in line and ("action" in line or "side" in line):
        for j in range(max(0,i-2), min(i+5, len(lines))):
            print(f"  {j+1:4}: {lines[j].rstrip()}")

# Check trade.intent path for side→action mapping
print("\n=== Trade intent side→action mapping area (1161-1180) ===")
for i, line in enumerate(lines[1160:1180], 1161):
    print(f"  {i:4}: {line.rstrip()}")

# ─── 2. Scan execution.log ────────────────────────────────────────────────
print("\n=== execution.log: last ~2000 lines filtered ===")
result = subprocess.run(
    ["tail", "-n", "2000", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
)
relevant = []
for line in result.stdout.splitlines():
    if any(k in line for k in [
        "Failed to parse TradeIntent", "❌", "STALE", "EXEC CLOSE", "PATH1B EXEC",
        "order placed", "FILLED", "REJECTED", "Error", "error", "Exception",
        "Traceback", "placing order", "submit_order", "execute_order", "SELL", "BUY"
    ]):
        relevant.append(line)

print(f"  Total relevant lines in last 2000: {len(relevant)}")
for line in relevant[-30:]:
    print(f"  {line.strip()[-200:]}")

# ─── 3. Fix: Add `side` to allowed_fields + map to action ─────────────────
print("\n=== Patch: Add side→action remapping before TradeIntent parse ===")

OLD_FILTER = """            # Filter signal_data to only include TradeIntent fields
            # BRIDGE-PATCH: Added ai_size_usd, ai_leverage, ai_harvest_policy for AI-driven sizing
            # P0.4C: Added reason, reduce_only for exit flow audit trail
            allowed_fields = {
                'symbol', 'action', 'confidence', 'position_size_usd', 'leverage',
                'timestamp', 'source', 'stop_loss_pct', 'take_profit_pct',
                'entry_price', 'stop_loss', 'take_profit', 'quantity',
                'ai_size_usd', 'ai_leverage', 'ai_harvest_policy',  # BRIDGE-PATCH v1.1
                'reason', 'reduce_only'  # P0.4C exit flow
            }
            filtered_data = {k: v for k, v in signal_data.items() if k in allowed_fields}"""

NEW_FILTER = """            # Filter signal_data to only include TradeIntent fields
            # BRIDGE-PATCH: Added ai_size_usd, ai_leverage, ai_harvest_policy for AI-driven sizing
            # P0.4C: Added reason, reduce_only for exit flow audit trail
            # C3-FIX: Map 'side' → 'action' for trade.intent messages from AI engine
            if 'action' not in signal_data and 'side' in signal_data:
                signal_data['action'] = signal_data['side']
            allowed_fields = {
                'symbol', 'action', 'confidence', 'position_size_usd', 'leverage',
                'timestamp', 'source', 'stop_loss_pct', 'take_profit_pct',
                'entry_price', 'stop_loss', 'take_profit', 'quantity',
                'ai_size_usd', 'ai_leverage', 'ai_harvest_policy',  # BRIDGE-PATCH v1.1
                'reason', 'reduce_only'  # P0.4C exit flow
            }
            filtered_data = {k: v for k, v in signal_data.items() if k in allowed_fields}"""

if OLD_FILTER.strip() in content:
    backup = SRC + f".bak.side.{int(time.time())}"
    shutil.copy(SRC, backup)
    print(f"  Backup: {backup}")
    with open(SRC, "w") as f:
        f.write(content.replace(OLD_FILTER, NEW_FILTER))
    print(f"  ✅ Applied side→action mapping fix to execution_service.py")
else:
    print("  ⚠ Exact string not found (whitespace difference?) — showing actual lines 1161-1178:")
    for i, line in enumerate(lines[1160:1178], 1161):
        print(f"    {i}: {repr(line.rstrip())}")

# ─── 4. Update quantum-execution service to also load intent-executor.env ─
print("\n=== Update quantum-execution to load intent-executor.env ===")
UNIT = "/etc/systemd/system/quantum-execution.service"
with open(UNIT) as f:
    unit_content = f.read()

shutil.copy(UNIT, UNIT + f".bak.{int(time.time())}")

OLD_ENV_FILE = "EnvironmentFile=/etc/quantum/testnet.env"
NEW_ENV_FILE = "EnvironmentFile=/etc/quantum/testnet.env\nEnvironmentFile=/etc/quantum/intent-executor.env"

if OLD_ENV_FILE in unit_content and NEW_ENV_FILE not in unit_content:
    with open(UNIT, "w") as f:
        f.write(unit_content.replace(OLD_ENV_FILE, NEW_ENV_FILE))
    print(f"  ✅ Added intent-executor.env to quantum-execution EnvironmentFile")
elif NEW_ENV_FILE in unit_content:
    print(f"  Already has intent-executor.env")
else:
    print(f"  ⚠ Could not find '{OLD_ENV_FILE}' in unit file")
    # Show relevant lines
    for line in unit_content.splitlines():
        if "EnvironmentFile" in line:
            print(f"    {line}")

# Reload + restart
print("\n=== systemctl daemon-reload + restart quantum-execution ===")
subprocess.run(["systemctl", "daemon-reload"], check=True)
subprocess.run(["systemctl", "restart", "quantum-execution"], check=True)
import time
time.sleep(4)
status = subprocess.run(["systemctl", "is-active", "quantum-execution"],
                       capture_output=True, text=True).stdout.strip()
print(f"  Status: {status}")

# Show first few new log lines
time.sleep(2)
log_tail = subprocess.run(
    ["tail", "-n", "30", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
).stdout
for line in log_tail.splitlines()[-10:]:
    print(f"  LOG: {line.strip()[-200:]}")
