#!/usr/bin/env python3
"""Check current -1111 status and diagnose XRPUSDT precision issue."""
import redis, re, time
from collections import Counter

r = redis.Redis()

# --- 1. Recent errors last 5 min ---
cutoff = (int(time.time()) - 300) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff))
errors = []
for _, d in entries:
    err = d.get(b"error", b"").decode()
    if err:
        sym = d.get(b"symbol", b"?").decode()
        m = re.search(r'"code"\s*:\s*(-?[0-9]+)', err)
        errors.append((sym, m.group(1) if m else "?"))

print(f"=== Errors last 5 min (of {len(entries)} entries) ===")
for (s, c), n in Counter(errors).most_common():
    print(f"  {s} code={c} x{n}")
if not errors:
    print("  ✅ ZERO errors — -1111 already gone")

# --- 2. Check if XRPUSDT has anything remaining in Redis ---
print("\n=== XRPUSDT Redis keys ===")
for key in r.scan_iter("*xrp*", count=200):
    key_str = key.decode().lower()
    if "xrp" in key_str:
        typ = r.type(key).decode()
        ttl = r.ttl(key)
        print(f"  {key.decode()} [{typ}] TTL={ttl}")

# --- 3. Check apply_layer source - how it gets XRPUSDT quantities ---
import subprocess
result = subprocess.run(
    ["grep", "-rn", "lot_size\|step_size\|stepSize\|precision\|quantity.*round\|round.*quantity",
     "/home/qt/quantum_trader/services/"],
    capture_output=True, text=True
)
print("\n=== apply_layer: quantity rounding code ===")
for line in result.stdout.splitlines()[:15]:
    print(f"  {line.strip()[:120]}")

# --- 4. Find apply_layer service file ---
result2 = subprocess.run(
    ["find", "/home/qt/quantum_trader/services", "-name", "apply_layer*", "-o", "-name", "*harvest_brain*"],
    capture_output=True, text=True
)
print("\n=== apply_layer / harvest_brain service files ===")
for f in result2.stdout.splitlines():
    print(f"  {f}")
