#!/usr/bin/env python3
"""
Fix 1: UPDATE_LEDGER_AFTER_EXEC=true
Fix 2: Finn og sett min hold-tid / cooldown
"""
import redis, subprocess

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# ─── FIX 1: UPDATE_LEDGER_AFTER_EXEC ─────────────────────────
print("=== FIX 1: INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC ===")
print()

env_file = "/etc/quantum/intent-executor.env"
with open(env_file) as f:
    content = f.read()

old_val = "INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC=false"
new_val = "INTENT_EXECUTOR_UPDATE_LEDGER_AFTER_EXEC=true"

if old_val in content:
    content = content.replace(old_val, new_val)
    with open(env_file, "w") as f:
        f.write(content)
    print(f"✅ Endret: false → true")
else:
    print(f"⚠️  Linja ikke funnet. Nåværende state:")
    for line in content.splitlines():
        if "UPDATE_LEDGER" in line:
            print(f"  {line}")

# Verifiser
with open(env_file) as f:
    for line in f:
        if "UPDATE_LEDGER" in line:
            print(f"   Nå: {line.strip()}")

# ─── FINN MIN HOLD-TID KONFIG ─────────────────────────────────
print("\n=== FINN MIN HOLD-TID / COOLDOWN KONFIG ===")

import os, re

search_dirs = [
    "/home/qt/quantum_trader/microservices/autonomous_trader/",
    "/home/qt/quantum_trader/microservices/intent_executor/",
    "/home/qt/quantum_trader/microservices/apply_layer/",
]

hold_terms = ["min_hold", "cooldown", "hold_sec", "hold_time", "min_time",
              "HOLD_SEC", "MIN_HOLD", "COOLDOWN", "hold_bars", "hold_minutes"]

print("\n  Søker etter hold-tid parametere i kode:")
for d in search_dirs:
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith("__")]
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path) as f:
                    lines = f.readlines()
            except:
                continue
            for i, line in enumerate(lines):
                for term in hold_terms:
                    if term in line and not line.strip().startswith("#"):
                        print(f"  {path.split('quantum_trader/')[1]}:{i+1}  {line.strip()[:90]}")
                        break

# Sjekk Redis config for hold-relaterte nøkler
print("\n  Redis: hold/cooldown konfig nøkler:")
for pattern in ["quantum:config:*", "quantum:settings:*", "quantum:param:*"]:
    keys = r.keys(pattern)
    for key in sorted(keys):
        typ = r.type(key)
        if typ == "hash":
            data = r.hgetall(key)
            for k, v in data.items():
                if any(t.lower() in k.lower() for t in hold_terms + ["min_exit", "max_hold", "force_exit_after"]):
                    print(f"  {key} → {k} = {v}")
        elif typ == "string":
            val = r.get(key)
            if any(t.lower() in key.lower() for t in hold_terms):
                print(f"  {key} = {val}")

# Sjekk autonomous_trader config
print("\n  Autonomous trader: exit score logic søk:")
at_file = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"
with open(at_file) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if any(t in line for t in ["min_hold", "hold_sec", "hold_time", "hold_count",
                                "hold_bars", "hold_minutes", "HOLD_THRESHOLD"]):
        if not line.strip().startswith("#"):
            ctx_start = max(0, i-1)
            ctx_end = min(len(lines), i+2)
            for j in range(ctx_start, ctx_end):
                prefix = ">>>" if j == i else "   "
                print(f"  {prefix} L{j+1}: {lines[j].rstrip()[:90]}")
            print()
