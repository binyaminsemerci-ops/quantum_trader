#!/usr/bin/env python3
"""
Read intent_executor source for BINANCE_BASE_URL, check creds, show AGLDUSDT position.
"""
import subprocess, os, redis, json

r = redis.Redis(decode_responses=True)

# 1) Show intent_executor BINANCE config section
print("=== intent_executor/main.py — Binance config lines ===")
with open("/home/qt/quantum_trader/microservices/intent_executor/main.py") as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    if any(x in line for x in ["BINANCE_BASE_URL", "BINANCE_API_KEY", "BINANCE_API_SECRET",
                                "BINANCE_TESTNET", "testnet.binance", "fapi.binance",
                                "os.getenv(\"BINANCE"]):
        print(f"  {i:4d}: {line.rstrip()}")

# 2) Check creds directory
print("\n=== /etc/quantum/creds/ ===")
try:
    result = subprocess.run(["ls", "-la", "/etc/quantum/creds/"], capture_output=True, text=True)
    print(result.stdout)
    # Show contents of non-secret files
    for fname in os.listdir("/etc/quantum/creds/"):
        fpath = f"/etc/quantum/creds/{fname}"
        if os.path.isfile(fpath):
            print(f"\n  --- {fname} ---")
            try:
                with open(fpath) as fh:
                    for line in fh:
                        l = line.strip()
                        if not l or l.startswith("#"):
                            print(f"    {l}")
                            continue
                        k = l.split("=")[0]
                        v = l.split("=", 1)[1] if "=" in l else ""
                        if any(x in k.upper() for x in ["SECRET", "PASSWORD", "TOKEN"]):
                            print(f"    {k}=***MASKED***")
                        elif "KEY" in k.upper():
                            print(f"    {k}={v[:8]}...")
                        else:
                            print(f"    {l}")
            except:
                print(f"    (cannot read)")
except Exception as e:
    print(f"  ERROR: {e}")

# 3) Show AGLDUSDT position
print("\n=== AGLDUSDT position ===")
agld_keys = list(r.keys("quantum:position:AGLD*"))
for k in agld_keys:
    if r.exists(k):
        t = r.type(k)
        if t == "hash":
            pos = r.hgetall(k)
        elif t == "string":
            raw = r.get(k)
            try:
                pos = json.loads(raw)
            except:
                pos = {"raw": raw}
        print(f"  {k} [{t}]:")
        for fk, fv in sorted(pos.items()):
            print(f"    {fk}: {fv}")

# Also get AGLDUSDT ledger entries
print("\n=== AGLDUSDT ledger ===")
for k in sorted(r.keys("quantum:position:ledger:AGLDUSDT*")):
    print(f"  {k}")
    raw = r.get(k)
    try:
        d = json.loads(raw)
        for fk, fv in sorted(d.items()):
            print(f"    {fk}: {fv}")
    except:
        print(f"  raw: {raw[:200] if raw else None}")
