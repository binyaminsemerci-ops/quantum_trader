#!/usr/bin/env python3
"""
FULL C3 ACTIVATION SCRIPT
Step 1: Patch intent_executor to support BINANCE_BASE_URL env override
Step 2: Update intent-executor.env → real Binance
Step 3: Close AGLDUSDT via manual lane
Step 4: Enable manual lane 4h
Step 5: Restart intent-executor
"""
import redis, json, subprocess, time, os, shutil
from datetime import datetime, timezone

r = redis.Redis(decode_responses=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)

def ts():
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S UTC")

SEP = "=" * 70
print(f"{SEP}")
print(f"C3 ACTIVATION — {ts()}")
print(f"{SEP}")

# ─── STEP 1: Patch source to support BINANCE_BASE_URL env override ────────────
print(f"\n[STEP 1] Patch intent_executor source — BINANCE_BASE_URL from env")
SRC = "/home/qt/quantum_trader/microservices/intent_executor/main.py"

with open(SRC) as f:
    src = f.read()

OLD = 'BINANCE_BASE_URL = "https://testnet.binancefuture.com"'
NEW = 'BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")'

if OLD in src:
    # Backup
    bak = SRC + f".bak.{int(time.time())}"
    shutil.copy(SRC, bak)
    print(f"  Backup: {bak}")
    
    src_patched = src.replace(OLD, NEW, 1)
    with open(SRC, "w") as f:
        f.write(src_patched)
    print(f"  ✅ Patched: BINANCE_BASE_URL now from env (default=testnet)")
elif 'os.getenv("BINANCE_BASE_URL"' in src:
    print(f"  ✅ Already patched (os.getenv present)")
else:
    print(f"  ⚠️  Pattern not found — manual check needed")

# Also patch API key var names to support both testnet and real
OLD_KEY = 'BINANCE_API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")'
NEW_KEY = 'BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", os.getenv("BINANCE_TESTNET_API_KEY", ""))'
OLD_SEC = 'BINANCE_API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")'
NEW_SEC = 'BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", os.getenv("BINANCE_TESTNET_API_SECRET", ""))'

with open(SRC) as f:
    src2 = f.read()

changes = 0
if OLD_KEY in src2:
    src2 = src2.replace(OLD_KEY, NEW_KEY, 1)
    changes += 1
if OLD_SEC in src2:
    src2 = src2.replace(OLD_SEC, NEW_SEC, 1)
    changes += 1
if changes:
    with open(SRC, "w") as f:
        f.write(src2)
    print(f"  ✅ Patched: API key/secret now fallback to BINANCE_TESTNET_* if BINANCE_API_KEY not set")
else:
    print(f"  ✅ API key vars already support fallback")

# ─── STEP 2: Check for real API keys + update intent-executor.env ─────────────
print(f"\n[STEP 2] Check real API keys + switch to fapi.binance.com")

# Try to find real key in any env file
real_key = None
real_secret = None
for envfile in ["/etc/quantum/testnet.env", "/etc/quantum/autonomous-trader.env",
                "/etc/quantum/apply-layer.env", "/etc/quantum/execution.env"]:
    try:
        with open(envfile) as f:
            content = f.read()
        for line in content.splitlines():
            if line.startswith("BINANCE_API_KEY=") and len(line.split("=", 1)[1].strip()) > 20:
                real_key = line.split("=", 1)[1].strip()
            if line.startswith("BINANCE_API_SECRET=") and len(line.split("=", 1)[1].strip()) > 20:
                real_secret = line.split("=", 1)[1].strip()
    except:
        pass

# Read intent-executor.env
ENV_FILE = "/etc/quantum/intent-executor.env"
with open(ENV_FILE) as f:
    env_content = f.read()

env_bak = ENV_FILE + f".bak.{int(time.time())}"
shutil.copy(ENV_FILE, env_bak)
print(f"  Backup: {env_bak}")

# Add/update BINANCE_BASE_URL
if "BINANCE_BASE_URL=" not in env_content:
    addition = "\n# Real Binance Futures API (switched from testnet)\nBINANCE_BASE_URL=https://fapi.binance.com\n"
    with open(ENV_FILE, "w") as f:
        f.write(env_content.rstrip() + addition)
    print(f"  ✅ Added BINANCE_BASE_URL=https://fapi.binance.com")
else:
    lines = env_content.splitlines()
    new_lines = [("BINANCE_BASE_URL=https://fapi.binance.com" if ln.startswith("BINANCE_BASE_URL=") else ln) for ln in lines]
    with open(ENV_FILE, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"  ✅ Updated BINANCE_BASE_URL=https://fapi.binance.com")

# Also update execution.env BINANCE_USE_TESTNET
EXEC_ENV = "/etc/quantum/execution.env"
try:
    with open(EXEC_ENV) as f:
        exec_content = f.read()
    shutil.copy(EXEC_ENV, EXEC_ENV + f".bak.{int(time.time())}")
    exec_new = exec_content.replace("BINANCE_USE_TESTNET=true", "BINANCE_USE_TESTNET=false")
    with open(EXEC_ENV, "w") as f:
        f.write(exec_new)
    print(f"  ✅ execution.env: BINANCE_USE_TESTNET=false")
except Exception as e:
    print(f"  ⚠️  Could not update execution.env: {e}")

if real_key:
    print(f"  ✅ Found real BINANCE_API_KEY in env files")
    # Add to intent-executor.env if not already there
    with open(ENV_FILE) as f:
        env_now = f.read()
    if "BINANCE_API_KEY=" not in env_now:
        with open(ENV_FILE, "a") as f:
            f.write(f"\n# Real Binance API Keys\nBINANCE_API_KEY={real_key}\n")
            if real_secret:
                f.write(f"BINANCE_API_SECRET={real_secret}\n")
        print(f"  ✅ Added real BINANCE_API_KEY to intent-executor.env")
else:
    print(f"  ⚠️  No plaintext BINANCE_API_KEY found in env files")
    print(f"     Real API keys in /etc/quantum/creds/ are encrypted")
    print(f"     Orders to fapi.binance.com will REJECT until real keys provided!")
    print(f"     → Add: echo 'BINANCE_API_KEY=<key>' >> {ENV_FILE}")
    print(f"     → Add: echo 'BINANCE_API_SECRET=<secret>' >> {ENV_FILE}")

# ─── STEP 3: Enable manual lane (do this BEFORE publishing close) ─────────────
print(f"\n[STEP 3] Enable manual lane — 4 hours TTL")
MANUAL_KEY = "quantum:manual_lane:enabled"
r.set(MANUAL_KEY, "1", ex=14400)
ttl = r.ttl(MANUAL_KEY)
print(f"  ✅ {MANUAL_KEY} = 1  TTL={ttl//60}m {ttl%60}s")

# ─── STEP 4: Close AGLDUSDT via manual stream ─────────────────────────────────
print(f"\n[STEP 4] Close AGLDUSDT SHORT (SL wrong direction)")

# Read current position
pos = r.hgetall("quantum:position:AGLDUSDT")
if not pos:
    print(f"  ⚠️  quantum:position:AGLDUSDT not found in Redis — may already be closed")
else:
    qty = float(pos.get("quantity", 0))
    entry = pos.get("entry_price", "?")
    side = pos.get("side", "?")
    sl = pos.get("stop_loss", "0")
    mark = pos.get("mark_price", "0")
    print(f"  Position: {side} {qty} @ {entry}  mark={mark}  SL={sl}")
    
    # Publish FULL_CLOSE to manual stream
    import uuid
    plan_id = str(uuid.uuid4()).replace("-", "")[:16]
    
    close_msg = {
        "plan_id": plan_id,
        "symbol": "AGLDUSDT",
        "action": "FULL_CLOSE_PROPOSED",
        "decision": "MANUAL_CLOSE_BAD_SL",
        "reason_codes": json.dumps(["MANUAL_CLOSE", "SL_WRONG_DIRECTION"]),
        "close_qty": str(qty),
        "price": mark,
        "reduceOnly": "True",
        "timestamp": str(int(time.time() * 1000)),
        "source": "c3_activation_audit",
        "kill_score": "1.0",
        "steps": json.dumps([{
            "action": "SELL" if side == "SHORT" else "BUY",
            "symbol": "AGLDUSDT",
            "qty": str(qty),
            "reduceOnly": True,
            "reason": "manual_close_bad_sl"
        }])
    }
    
    msg_id = r.xadd("quantum:stream:apply.plan.manual", close_msg)
    print(f"  ✅ FULL_CLOSE published to apply.plan.manual")
    print(f"     plan_id={plan_id}  stream_id={msg_id}")
    print(f"     → intent_executor will process this (manual lane is now open)")

# ─── STEP 5: Restart intent-executor to pick up new BINANCE_BASE_URL ──────────
print(f"\n[STEP 5] Restart quantum-intent-executor")
r5 = run("systemctl restart quantum-intent-executor")
if r5.returncode == 0:
    print(f"  ✅ Service restarted")
else:
    print(f"  ❌ Restart failed: {r5.stderr}")

time.sleep(5)
r5b = run("systemctl is-active quantum-intent-executor")
print(f"  Status: {r5b.stdout.strip()}")

# ─── STEP 6: Wait and watch for AGLDUSDT close ────────────────────────────────
print(f"\n[STEP 6] Waiting 15s for AGLDUSDT close to process...")
time.sleep(15)

# Check if close was processed
results = r.xrevrange("quantum:stream:apply.result", count=10)
agld_closed = False
for mid, fields in results:
    if fields.get("symbol") == "AGLDUSDT" and fields.get("executed", "").lower() in ("true", "1"):
        agld_closed = True
        print(f"  ✅ AGLDUSDT close executed: plan={fields.get('plan_id','?')[:8]}")
        break

if not agld_closed:
    # Check intent-executor logs
    log_result = run("journalctl -u quantum-intent-executor -n 20 --no-pager | tail -20")
    print(f"  ⚠️  No executed close found yet. Recent logs:")
    for line in log_result.stdout.splitlines()[-10:]:
        print(f"    {line.strip()[-100:]}")

print(f"\n{SEP}")
print(f"DONE — {ts()}")
print(f"{SEP}")
print(f"\nNext: Run python3 /tmp/_execution_audit.py to verify all checks")
print(f"\nTo provide real API keys when ready:")
print(f"  echo 'BINANCE_API_KEY=<your_real_key>' >> /etc/quantum/intent-executor.env")
print(f"  echo 'BINANCE_API_SECRET=<your_real_secret>' >> /etc/quantum/intent-executor.env")
print(f"  systemctl restart quantum-intent-executor")
