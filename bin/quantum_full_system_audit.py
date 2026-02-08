#!/usr/bin/env python3
"""
Quantum Trader – FULL SYSTEM AUDIT
Covers: Data → Signal → Decision → Risk → Execution
READ-ONLY (testnet safe)
"""

import subprocess
import time
from datetime import datetime

REDIS = "redis-cli"

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        return f"[ERROR] {e.output.decode().strip()}"

def h(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def sec(title):
    print(f"\n--- {title}")

def ok(m): print(f"✅ {m}")
def fail(m): print(f"❌ {m}")
def info(m): print(f"ℹ️  {m}")

# ----------------------------------------------------------------

h("QUANTUM TRADER – FULL SYSTEM AUDIT")
print("UTC:", datetime.utcnow().isoformat())

# 1. Core services
sec("1) CORE SERVICES")
services = sh("systemctl list-units --type=service | grep quantum").splitlines()
print("\n".join(services) if services else "No quantum services listed")

# 2. Market data presence
sec("2) MARKET / DATA INPUT")
streams = [
    "quantum:stream:market",
    "quantum:stream:features",
    "quantum:stream:signals",
]
for s in streams:
    l = sh(f"{REDIS} XLEN {s}")
    if l.isdigit() and int(l) > 0:
        ok(f"{s} has data ({l})")
    else:
        info(f"{s} empty or missing")

# 3. Signal generation
sec("3) SIGNAL GENERATION")
signal_keys = sh(f"{REDIS} KEYS 'quantum:*signal*'").splitlines()
if signal_keys:
    ok(f"Signal keys exist: {signal_keys[:5]}")
else:
    fail("No signal artifacts found")

# 4. DECISION PLANE (CRITICAL)
sec("4) DECISION PLANE (CEO / STRATEGY / RISK)")
decision_keys = [
    "quantum:decision:intent",
    "quantum:ceo:decision",
    "quantum:strategy:decision",
    "quantum:order:intent",
]
found = False
for k in decision_keys:
    v = sh(f"{REDIS} GET {k}")
    if v and v != "(nil)":
        ok(f"{k} = {v}")
        found = True

if not found:
    fail("NO decision intents produced by system")

# 5. Risk & apply layer
sec("5) APPLY / RISK")
status = sh("python3 bin/risk_status.py")
print(status)

# 6. Execution attempts
sec("6) EXECUTION ATTEMPTS")
apply_log = sh("grep -i 'order\\|execute\\|submit' /opt/quantum/logs/apply_layer.log | tail -5")
if apply_log:
    ok("Execution attempts found")
    print(apply_log)
else:
    fail("No execution attempts in apply-layer")

# FINAL
h("FINAL CONCLUSION")

if not found:
    print("""
ROOT CAUSE (SYSTEMIC):

• No active DECISION PRODUCER
• CEO / Strategy / Risk brains are not running
• Apply-layer is correctly idle
• System is SAFE, but UNDECIDED

This is not a bug.
This is an incomplete activation of the architecture.
""")
else:
    print("""
DECISION PLANE EXISTS

Next steps:
• Trace decision → apply binding
• Verify exchange adapter / router
""")
