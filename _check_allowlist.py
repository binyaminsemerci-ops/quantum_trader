#!/usr/bin/env python3
import redis, json, os

r = redis.Redis(decode_responses=True)

# 1. Check allowlists from env
print("=== ALLOWLISTS ===")
env_file = "/etc/quantum/intent-executor.env"
source_list = []
symbol_list = []
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line.startswith("INTENT_EXECUTOR_SOURCE_ALLOWLIST="):
            source_list = line.split("=", 1)[1].split(",")
            print(f"SOURCE_ALLOWLIST ({len(source_list)}):")
            for s in source_list:
                print(f"  {s.strip()}")
        elif line.startswith("INTENT_EXECUTOR_ALLOWLIST="):
            symbol_list = line.split("=", 1)[1].split(",")
            print(f"SYMBOL_ALLOWLIST ({len(symbol_list)}):")
            for s in symbol_list:
                print(f"  {s.strip()}")

# Check if AGLDUSDT is in symbol list
agld_in = any("AGLD" in s for s in symbol_list)
print(f"\nAGLDUSDT in symbol allowlist: {agld_in}")

# Check if c3_activation_audit is in source list
c3_in = any("c3_activation" in s for s in source_list)
print(f"c3_activation_audit in source allowlist: {c3_in}")

# 2. Get full apply.result payload for plan eaec32a2
print("\n=== apply.result full entry for plan eaec32a2 ===")
results = r.xrange("quantum:stream:apply.result", count=2000)
for sid, data in reversed(results):
    raw = data.get("payload", "")
    pid_direct = data.get("plan_id", "")
    if "eaec32a2" in raw or pid_direct.startswith("eaec32a2"):
        print(f"Stream ID: {sid}")
        if raw:
            try:
                p = json.loads(raw)
                for k, v in sorted(p.items()):
                    val = str(v)
                    if len(val) > 120:
                        val = val[:120] + "..."
                    print(f"  {k}: {val}")
            except:
                print(f"  (raw): {raw[:500]}")
        else:
            for k, v in sorted(data.items()):
                print(f"  {k}: {v}")
        break

# 3. Check what the original manual plan looked like
print("\n=== apply.plan.manual entry 1771982238240-0 ===")
entry = r.xrange("quantum:stream:apply.plan.manual", "-", "+", count=10)
for sid, data in entry:
    if "1771982238240" in sid:
        raw = data.get("payload", "")
        if raw:
            p = json.loads(raw)
            print(f"  symbol: {p.get('symbol')}")
            print(f"  action: {p.get('action')}")
            print(f"  source: {p.get('source')}")
            print(f"  plan_id: {p.get('plan_id')}")
            print(f"  steps: {p.get('steps')}")
