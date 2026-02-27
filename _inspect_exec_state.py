#!/usr/bin/env python3
"""Inspect execution_service.py state + investigate why guard doesn't fire"""

SRC = "/home/qt/quantum_trader/services/execution_service.py"

with open(SRC) as f:
    lines = f.readlines()

print("=== Lines 1155-1230 ===")
for i, line in enumerate(lines[1154:1230], 1155):
    print(f"  {i}: {line.rstrip()[:150]}")

# Check if C3-FIX-2 guard is present
guard_found = any("C3-FIX-2: Guard" in line for line in lines)
action_guard_found = any("'action' not in signal_data" in line for line in lines)
print(f"\nC3-FIX-2 guard found: {guard_found}")
print(f"action guard found: {action_guard_found}")

# Find and show the guard lines
for i, line in enumerate(lines, 1):
    if "'action' not in signal_data" in line:
        print(f"  Guard at line {i}: {line.rstrip()}")

# Check PIPELINE_DIAG line
for i, line in enumerate(lines, 1):
    if "PIPELINE_DIAG" in line:
        print(f"  PIPELINE_DIAG at line {i}: {line.rstrip()}")

# Check process env
import subprocess, os
result = subprocess.run(
    ["cat", "/proc/$(systemctl show -p MainPID --value quantum-execution)/environ"],
    capture_output=True, text=True, shell=True
)
if "PIPELINE_DIAG" in result.stdout:
    print("\nPIPELINE_DIAG is in process env")
else:
    print("\nPIPELINE_DIAG NOT in process env")

# Get PID a different way
import subprocess
pid_result = subprocess.run(
    ["systemctl", "show", "-p", "MainPID", "--value", "quantum-execution"],
    capture_output=True, text=True
)
pid = pid_result.stdout.strip()
print(f"Main PID: {pid}")
if pid and pid != "0":
    env_file = f"/proc/{pid}/environ"
    try:
        with open(env_file, "rb") as f:
            env_content = f.read().decode(errors="replace")
        env_vars = dict(v.split("=", 1) for v in env_content.split("\x00") if "=" in v)
        print(f"PIPELINE_DIAG in env: {env_vars.get('PIPELINE_DIAG', 'NOT SET')}")
        print(f"USE_BINANCE_TESTNET: {env_vars.get('USE_BINANCE_TESTNET', 'NOT SET')}")
        print(f"BINANCE_BASE_URL: {env_vars.get('BINANCE_BASE_URL', 'NOT SET')}")
    except Exception as e:
        print(f"Could not read /proc/{pid}/environ: {e}")
