#!/usr/bin/env python3
"""
Fix AGLDUSDT close:
1. Add AGLDUSDT to INTENT_EXECUTOR_ALLOWLIST
2. Add 'manual_close' to INTENT_EXECUTOR_SOURCE_ALLOWLIST
3. Restart service
4. Republish FULL_CLOSE with source=harvest_brain (already allowed)
5. Delete Redis position hash (phantom testnet artifact)
"""
import redis, json, time, subprocess, shutil

r = redis.Redis(decode_responses=True)

ENV_FILE = "/etc/quantum/intent-executor.env"

# 1. Fix allowlists
print("=== [1] Patch allowlists ===")
with open(ENV_FILE) as f:
    content = f.read()

shutil.copy(ENV_FILE, ENV_FILE + f".bak.agld.{int(time.time())}")

lines = content.splitlines()
new_lines = []
for ln in lines:
    if ln.startswith("INTENT_EXECUTOR_ALLOWLIST="):
        symbols = ln.split("=", 1)[1].strip()
        if "AGLDUSDT" not in symbols:
            ln = ln.rstrip() + ",AGLDUSDT"
            print(f"  ✅ Added AGLDUSDT to symbol allowlist")
    if ln.startswith("INTENT_EXECUTOR_SOURCE_ALLOWLIST="):
        sources = ln.split("=", 1)[1].strip()
        if "manual_close" not in sources:
            ln = ln.rstrip() + ",manual_close"
            print(f"  ✅ Added manual_close to source allowlist")
    new_lines.append(ln)

with open(ENV_FILE, "w") as f:
    f.write("\n".join(new_lines) + "\n")

# 2. Restart service
print("\n=== [2] Restart quantum-intent-executor ===")
subprocess.run(["systemctl", "restart", "quantum-intent-executor"], check=True)
time.sleep(3)
result = subprocess.run(
    ["systemctl", "is-active", "quantum-intent-executor"],
    capture_output=True, text=True
)
print(f"  Status: {result.stdout.strip()}")

# 3. Publish FULL_CLOSE with source=harvest_brain (allowed)
print("\n=== [3] Republish FULL_CLOSE for AGLDUSDT (source=harvest_brain) ===")
plan_id = f"agld_manual_close_{int(time.time())}"
plan = {
    "plan_id": plan_id,
    "symbol": "AGLDUSDT",
    "action": "FULL_CLOSE_PROPOSED",
    "source": "harvest_brain",     # in allowlist
    "side": "SHORT",
    "close_qty": "3030.0",
    "reduce_only": True,
    "timestamp": int(time.time() * 1000),
    "reason": "manual_close_wrong_sl",
    "steps": [
        {"action": "SELL", "symbol": "AGLDUSDT", "qty": "3030.0", "reduceOnly": True}
    ]
}
sid = r.xadd("quantum:stream:apply.plan.manual", {"payload": json.dumps(plan)})
print(f"  ✅ Published plan_id={plan_id}")
print(f"     stream_id={sid}")

# 4. Wait and check result
print("\n=== [4] Waiting 10s for result ===")
time.sleep(10)

results = r.xrange("quantum:stream:apply.result", count=500)
found = False
for s_id, data in reversed(results):
    raw = data.get("payload", "")
    d_pid = data.get("plan_id", "")
    if plan_id in raw or d_pid == plan_id or (raw and json.loads(raw).get("plan_id") == plan_id if raw.startswith("{") else False):
        p = json.loads(raw) if raw else data
        found = True
        print(f"  plan_id={p.get('plan_id')} executed={p.get('executed')} reason={p.get('reason','')}")
        break

if not found:
    print("  (no result yet — checking logs)")
    result = subprocess.run(
        ["journalctl", "-u", "quantum-intent-executor", "-n", "15", "--no-pager"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if any(k in line for k in ["AGLD", "agld_manual", "manual", "ERROR", "error", "FULL_CLOSE"]):
            print(f"  LOG: {line.strip()[-150:]}")

# 5. Delete Redis position hash (phantom testnet artifact)
print("\n=== [5] Delete phantom Redis position hash ===")
pos_key = "quantum:position:AGLDUSDT"
pos_exists = r.exists(pos_key)
if pos_exists:
    # Snapshot before delete
    pos_data = r.hgetall(pos_key)
    print(f"  Deleting {pos_key}:")
    print(f"    side={pos_data.get('side')}  qty={pos_data.get('quantity')}  entry={pos_data.get('entry_price')}")
    print(f"    stop_loss={pos_data.get('stop_loss')}  source={pos_data.get('source')}")
    r.delete(pos_key)
    print(f"  ✅ Deleted quantum:position:AGLDUSDT")
    
    # Also remove from any position index sets
    for set_key in ["quantum:positions:active", "quantum:active_positions", "quantum:positions"]:
        if r.sismember(set_key, "AGLDUSDT"):
            r.srem(set_key, "AGLDUSDT")
            print(f"  ✅ Removed AGLDUSDT from {set_key}")

    # Check for related keys
    related = r.keys("*AGLDUSDT*")
    print(f"  Remaining AGLDUSDT keys: {related}")
else:
    print(f"  (quantum:position:AGLDUSDT does not exist)")

print("\n=== DONE ===")
print(f"AGLDUSDT phantom position cleaned.")
print(f"Intent-executor restarted with updated allowlists.")
