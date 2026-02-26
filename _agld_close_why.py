#!/usr/bin/env python3
import redis, json, subprocess

r = redis.Redis(decode_responses=True)

# Get full apply.result entry for eaec32a2
print("--- apply.result full payload for plan eaec32a2 ---")
results = r.xrange("quantum:stream:apply.result", count=2000)
for sid, data in reversed(results):
    raw_payload = data.get("payload", "")
    plan_id_direct = data.get("plan_id", "")
    if "eaec32a2" in str(data) or "eaec32a2" in raw_payload or plan_id_direct.startswith("eaec32a2"):
        print(f"  Stream ID: {sid}")
        print(f"  Raw data keys: {list(data.keys())}")
        if raw_payload:
            try:
                p = json.loads(raw_payload)
                for k, v in p.items():
                    print(f"    {k}: {v}")
            except:
                print(f"    raw: {raw_payload[:500]}")
        else:
            for k, v in data.items():
                print(f"    {k}: {v}")
        break

# Check journalctl around 01:17:22 more broadly
print("\n--- journalctl around AGLDUSDT close (all lines 01:17:20-01:17:35) ---")
result = subprocess.run(
    ["journalctl", "-u", "quantum-intent-executor", "--no-pager",
     "--since", "2026-02-25 01:17:15", "--until", "2026-02-25 01:17:45"],
    capture_output=True, text=True
)
for line in result.stdout.splitlines():
    print(f"  {line.strip()}")

# Also check what BINANCE_BASE_URL is now loaded by the service
print("\n--- Verify env loaded by service ---")
import os
env_file = "/etc/quantum/intent-executor.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if "BINANCE_BASE_URL" in line or "BINANCE_API_KEY" in line:
                k = line.split("=")[0]
                v = line.split("=", 1)[1] if "=" in line else ""
                if "KEY" in k or "SECRET" in k:
                    print(f"  {k}={v[:8]}...")
                else:
                    print(f"  {k}={v}")
