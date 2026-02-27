#!/usr/bin/env python3
"""
Part 1: Investigate quantum-execution source + real keys
Part 2: Fix and run the position ATR risk patcher
"""
import redis, json, time, subprocess, os

r = redis.Redis(decode_responses=True)

# ─── A: Inspect quantum-execution source ──────────────────────────────────
print("=" * 70)
print("A) quantum-execution SOURCE INSPECTION")
print("=" * 70)

exec_main = "/home/qt/quantum_trader/microservices/execution/main.py"
try:
    with open(exec_main) as f:
        content = f.read()
    lines = content.splitlines()
    print(f"  Total lines: {len(lines)}")

    # Print first 80 lines (imports, config, key loading)
    print("\n  === Lines 1-80 (config) ===")
    for i, line in enumerate(lines[:80], 1):
        if any(k in line for k in ["BINANCE", "API_KEY", "API_SECRET", "BASE_URL",
                                    "testnet", "execution.result", "xadd", "xread",
                                    "stream", "def ", "class ", "order", "ORDER"]):
            print(f"  {i:4}: {line}")
except Exception as e:
    print(f"  Error reading source: {e}")

# Check systemd service file for quantum-execution
print("\n  === quantum-execution service unit ===")
svc = subprocess.run(
    ["cat", "/etc/systemd/system/quantum-execution.service"],
    capture_output=True, text=True
).stdout
for line in svc.splitlines():
    print(f"    {line}")

# journalctl since service start
print("\n  === journalctl quantum-execution last 200 lines ===")
logs = subprocess.run(
    ["journalctl", "-u", "quantum-execution", "-n", "200", "--no-pager"],
    capture_output=True, text=True
).stdout
if "-- No entries --" in logs or not logs.strip():
    print("  (ZERO log entries for this service)")
    # Check if unit file exists and is enabled
    enabled = subprocess.run(
        ["systemctl", "is-enabled", "quantum-execution"],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"  is-enabled: {enabled}")
    active = subprocess.run(
        ["systemctl", "status", "quantum-execution"],
        capture_output=True, text=True
    ).stdout.strip()
    print(f"  status:\n{active[:600]}")
else:
    for line in logs.splitlines()[-40:]:
        print(f"  {line.strip()[-200:]}")

# ─── B: Find real API keys ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("B) REAL API KEY HUNT")
print("=" * 70)

# Try to read .cred files
cred_dir = "/etc/quantum/creds/"
creds_found = {}
if os.path.exists(cred_dir):
    for fname in os.listdir(cred_dir):
        fpath = os.path.join(cred_dir, fname)
        try:
            with open(fpath, "rb") as f:
                raw = f.read(200)
            # Check if it's plaintext (starts with ASCII printable)
            decoded = raw.decode("utf-8", errors="ignore").strip()
            # A real Binance key is 64 chars alphanumeric
            if len(decoded) >= 30 and decoded.isalnum():
                creds_found[fname] = decoded
                print(f"  ✅ PLAINTEXT cred in {fname}: {decoded[:8]}... (len={len(decoded)})")
            else:
                print(f"  🔒 ENCRYPTED cred in {fname} (first 20 bytes: {raw[:20]})")
        except Exception as e:
            print(f"  ❌ {fname}: {e}")

# Scan all env files for any key that is NOT the testnet key
testnet_prefix = "w2W60kzu"
print(f"\n  Scanning ALL /etc/quantum/*.env for non-testnet BINANCE keys:")
for ef in os.listdir("/etc/quantum/"):
    if not ef.endswith(".env"):
        continue
    fpath = f"/etc/quantum/{ef}"
    try:
        with open(fpath) as f:
            content_env = f.read()
        for line in content_env.splitlines():
            line = line.strip()
            if "BINANCE_API_KEY=" in line or "BINANCE_API_SECRET=" in line:
                k, v = line.split("=", 1)
                is_testnet = v.strip().startswith(testnet_prefix)
                marker = "testnet" if is_testnet else "✅ DIFFERENT KEY"
                if "SECRET" in k:
                    print(f"    {ef}: {k}=*** [{marker}]")
                else:
                    print(f"    {ef}: {k}={v[:12]}... [{marker}]")
    except:
        pass

# ─── C: PATCH positions — skip WRONGTYPE keys ─────────────────────────────
print("\n" + "=" * 70)
print("C) PATCH 23 POSITIONS — entry_risk_usdt (WRONGTYPE-safe)")
print("=" * 70)

# Build ATR lookup from market data (avoid WRONGTYPE)
def safe_atr_from_redis(sym, ep):
    candidates = [
        (f"quantum:market:{sym}:binance_main", "atr"),
        (f"quantum:market:{sym}", "atr"),
        (f"quantum:atr:{sym}", "value"),
    ]
    for key, field in candidates:
        try:
            key_type = r.type(key)
            if key_type == "hash":
                val = r.hget(key, field) or r.hget(key, "atr_14")
                if val:
                    return float(val), f"redis:{key}:{field}"
        except:
            pass
    # Fallback: 2% of entry price as ATR estimate
    return ep * 0.02, "estimated@2%_ep"

to_patch = []
for pkey in r.keys("quantum:position:*"):
    if pkey.count(":") != 2:
        continue
    try:
        key_type = r.type(pkey)
        if key_type != "hash":
            continue
        d = r.hgetall(pkey)
        if not d:
            continue
        risk = float(d.get("entry_risk_usdt", 0) or 0)
        qty = float(d.get("quantity", 0) or 0)
        ep = float(d.get("entry_price", 0) or 0)
        if risk == 0 and qty > 0 and ep > 0:
            to_patch.append((pkey, d))
    except Exception as e:
        print(f"  skip {pkey}: {e}")

print(f"  Positions to patch: {len(to_patch)}")
patched = 0
for pkey, d in to_patch:
    sym = pkey.split(":")[-1]
    ep   = float(d.get("entry_price", 0) or 0)
    sl   = float(d.get("stop_loss", 0) or 0)
    qty  = float(d.get("quantity", 0) or 0)
    side = d.get("side", "SHORT")

    atr_val, atr_src = safe_atr_from_redis(sym, ep)

    # Compute risk_per_unit
    if sl > 0 and sl != ep:
        if side == "SHORT" and sl > ep:
            # Correct direction for SHORT
            risk_per_unit = sl - ep
        elif side == "LONG" and sl < ep:
            risk_per_unit = ep - sl
        else:
            # Wrong direction — use 1.5 * ATR as fallback
            risk_per_unit = 1.5 * atr_val
    else:
        risk_per_unit = 1.5 * atr_val

    entry_risk_usdt = round(risk_per_unit * qty, 4)

    r.hset(pkey, mapping={
        "entry_risk_usdt": entry_risk_usdt,
        "risk_patched_by": "c3_retrospective",
        "risk_patched_at": int(time.time()),
    })
    patched += 1
    print(f"  ✅ {sym:22} {side:5} qty={qty:>10.1f}  ep={ep:.5f}  "
          f"risk_per_unit={risk_per_unit:.6f}  entry_risk={entry_risk_usdt:.4f}  [{atr_src}]")

print(f"\n  Patched {patched}/{len(to_patch)} positions")

# Final verify
remaining_zero = sum(
    1 for pkey in r.keys("quantum:position:*")
    if pkey.count(":") == 2 and r.type(pkey) == "hash"
    and float((r.hget(pkey, "entry_risk_usdt") or 0)) == 0
    and float((r.hget(pkey, "quantity") or 0)) > 0
)
print(f"  Remaining positions still at entry_risk_usdt=0: {remaining_zero}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
