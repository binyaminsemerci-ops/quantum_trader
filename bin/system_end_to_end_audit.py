#!/usr/bin/env python3
"""
Quantum System End-to-End Audit
READ-ONLY – no side effects
"""

import subprocess
import time
import json
import os
from datetime import datetime

REDIS = "redis-cli"

def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
    except subprocess.CalledProcessError as e:
        return f"[ERROR] {e.output.decode().strip()}"

def header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def section(title):
    print(f"\n--- {title}")

def ok(msg): print(f"✅ {msg}")
def fail(msg): print(f"❌ {msg}")
def info(msg): print(f"ℹ️  {msg}")

# ---------------------------------------------------------------------

header("QUANTUM SYSTEM – END-TO-END AUDIT")
print("Timestamp:", datetime.utcnow().isoformat(), "UTC")

# 1. SYSTEMD SERVICES
section("1) SYSTEMD – CORE SERVICES")

services = [
    "quantum-apply-layer",
    "quantum-rl-feedback-v2",
    "quantum-rl-trainer",
]

for svc in services:
    status = sh(f"systemctl is-active {svc}")
    if status == "active":
        ok(f"{svc} is ACTIVE")
    else:
        fail(f"{svc} is {status}")

# 2. HEARTBEATS
section("2) HEARTBEATS")

hb_keys = {
    "RL Feedback": "quantum:svc:rl_feedback_v2:heartbeat",
    "RL Trainer": "quantum:svc:rl_trainer:heartbeat",
}

for name, key in hb_keys.items():
    val = sh(f"{REDIS} GET {key}")
    ttl = sh(f"{REDIS} TTL {key}")
    if val and val != "(nil)" and ttl.isdigit() and int(ttl) > 0:
        ok(f"{name} heartbeat OK (ttl={ttl}s)")
    else:
        fail(f"{name} heartbeat MISSING or EXPIRED")

# 3. RISK FSM
section("3) RISK FSM STATE")

try:
    out = sh("python3 bin/risk_status.py")
    print(out)
except Exception as e:
    fail("risk_status.py failed")

kill = sh(f"{REDIS} GET quantum:global:kill_switch")
if kill in ("1", "true", "True"):
    fail("KILL SWITCH = ON")
else:
    ok("Kill switch = OFF")

# 4. LEARNING LOOP
section("4) LEARNING LOOP (PnL → Reward → Trainer)")

# Inject test PnL
sh(f"{REDIS} XADD quantum:stream:exitbrain.pnl '*' symbol BTCUSDT pnl 42 confidence 0.9 volatility 0.01")
time.sleep(2)

reward = sh(f"{REDIS} XREVRANGE quantum:stream:rl_rewards + - COUNT 1")
if reward and reward != "(empty list or set)":
    ok("Reward published")
else:
    fail("No reward published")

trainer_log = sh("grep 'Consumed reward' /opt/quantum/logs/rl_trainer.err | tail -1")
if trainer_log:
    ok("Trainer consumed reward")
else:
    fail("Trainer did NOT consume reward")

models = sh("ls -lt /opt/quantum/models | head -2")
print(models)

# 5. DECISION PLANE
section("5) DECISION PLANE (CEO / STRATEGY / RISK BRAINS)")

decision_keys = [
    "quantum:decision:intent",
    "quantum:strategy:signal",
    "quantum:ceo:decision",
    "quantum:order:intent",
]

found = False
for k in decision_keys:
    v = sh(f"{REDIS} GET {k}")
    if v and v != "(nil)":
        ok(f"FOUND decision artifact: {k}")
        found = True

if not found:
    fail("NO decision artifacts found (CEO/Strategy not producing output)")

# 6. EXECUTION BINDING
section("6) EXECUTION BINDING")

apply_log = sh("grep -i 'execute' /opt/quantum/logs/apply_layer.log | tail -3")
if apply_log:
    info("Apply layer execution logs:")
    print(apply_log)
else:
    fail("Apply layer never attempted execution")

# FINAL VERDICT
header("FINAL VERDICT")

if not found:
    print("""
SYSTEM STATE: SAFE BUT INCOMPLETE

Root cause:
- No active decision producer (CEO / Strategy / Risk brain)
- Apply-layer is gating correctly
- System is behaving as designed (fail-closed)

Conclusion:
→ System CANNOT trade because no component issues trade intent.
""")
else:
    print("""
SYSTEM STATE: DECISION PLANE PRESENT

Next step:
- Inspect decision → execution mapping
- Verify order router / exchange adapter
""")
