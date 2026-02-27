#!/usr/bin/env python3
"""Post-fix diagnosis — compare system state vs before fixes."""
import redis
import time
import subprocess

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def ts_age(ts_ms):
    now_ms = int(time.time() * 1000)
    return round((now_ms - int(ts_ms.split("-")[0])) / 1000, 1)

print("=" * 65)
print("POST-FIX SYSTEM DIAGNOSIS")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 65)

# ── 1. STREAMS ──────────────────────────────────────────────────
print("\n[1] STREAM STATUS")
streams = [
    "quantum:stream:position.snapshot",
    "quantum:stream:harvest.intents",
    "quantum:stream:reconcile.events",
    "quantum:stream:apply.result",
]
for s in streams:
    length = r.xlen(s)
    last = r.xrevrange(s, "+", "-", count=1)
    if last:
        msg_id, fields = last[0]
        age = ts_age(msg_id)
        sym = fields.get("symbol", fields.get("event_type", "?"))
        print(f"  {s}")
        print(f"    len={length}  last={age}s ago  sym={sym}")
    else:
        print(f"  {s}  len={length}  EMPTY")

# ── 2. REDIS POSITIONS ──────────────────────────────────────────
print("\n[2] REDIS POSITIONS (quantum:position:{sym})")
cursor = 0
pos_keys = []
while True:
    cursor, keys = r.scan(cursor, match="quantum:position:*", count=200)
    for k in keys:
        if ":snapshot:" not in k and ":ledger:" not in k and ":claim:" not in k:
            pos_keys.append(k)
    if cursor == 0:
        break

pos_keys.sort()
print(f"  Total open position keys: {len(pos_keys)}")
for pk in pos_keys:
    sym = pk.replace("quantum:position:", "")
    data = r.hgetall(pk)
    side = data.get("side", "?")
    qty  = data.get("quantity", "?")
    ep   = data.get("entry_price", "?")
    upnl = float(data.get("unrealized_pnl", 0))
    risk_miss = data.get("risk_missing", "0")
    has_ledger = r.exists(f"quantum:position:ledger:{sym}")
    has_snap   = r.exists(f"quantum:position:snapshot:{sym}")
    hold       = r.get(f"quantum:position:hold:{sym}")
    print(f"  {sym:20s} {side:5s} qty={qty:>10s} ep={ep:>10s}  pnl={upnl:+.3f}"
          f"  risk_missing={risk_miss}  ledger={bool(has_ledger)}  snap={bool(has_snap)}"
          f"  hold={'YES:'+hold if hold else 'no'}")

# ── 3. RECONCILE EVENTS (last 10) ────────────────────────────────
print("\n[3] RECONCILE EVENTS (last 10)")
events = r.xrevrange("quantum:stream:reconcile.events", "+", "-", count=10)
if not events:
    print("  EMPTY")
for msg_id, fields in events:
    age = ts_age(msg_id)
    etype = fields.get("event_type", "?")
    sym   = fields.get("symbol", "?")
    reason = fields.get("reason", "")
    print(f"  {age:>7.1f}s ago  {etype:25s}  {sym:20s}  {reason}")

# ── 4. HARVEST BRAIN LOG TAIL ────────────────────────────────────
print("\n[4] HARVEST BRAIN LOG (last 15 lines)")
try:
    out = subprocess.check_output(
        ["tail", "-15", "/var/log/quantum/harvest_brain.log"],
        stderr=subprocess.DEVNULL
    ).decode(errors="replace")
    for line in out.strip().splitlines():
        print("  " + line[-120:])
except Exception as e:
    print(f"  ERROR: {e}")

# ── 5. RECONCILE ENGINE LOG (last 10 lines) ──────────────────────
print("\n[5] RECONCILE ENGINE LOG (last 10 lines)")
try:
    out = subprocess.check_output(
        ["journalctl", "-u", "quantum-reconcile-engine", "--no-pager", "-n", "10", "--output=cat"],
        stderr=subprocess.DEVNULL
    ).decode(errors="replace")
    for line in out.strip().splitlines():
        print("  " + line[-120:])
except Exception as e:
    print(f"  ERROR: {e}")

# ── 6. SERVICE STATUS ────────────────────────────────────────────
print("\n[6] SERVICE STATUS")
services = [
    "quantum-balance-tracker",
    "quantum-position-state-brain",
    "quantum-reconcile-engine",
    "quantum-harvest-brain",
    "quantum-intent-executor",
    "quantum-apply-layer",
]
for svc in services:
    try:
        result = subprocess.check_output(
            ["systemctl", "is-active", svc],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        result = "UNKNOWN"
    print(f"  {svc:40s} {result}")

# ── 7. POSITION LEDGERS ─────────────────────────────────────────
print("\n[7] LEDGER STATUS for all open positions")
for pk in pos_keys:
    sym = pk.replace("quantum:position:", "")
    ledger = r.hgetall(f"quantum:position:ledger:{sym}")
    if ledger:
        la = ledger.get("ledger_amt", ledger.get("last_known_amt", "?"))
        ls = ledger.get("ledger_side", ledger.get("last_side", "?"))
        src = ledger.get("source", "?")
        print(f"  {sym:20s}  ledger_amt={la:>10s}  side={ls:5s}  source={src}")
    else:
        print(f"  {sym:20s}  *** NO LEDGER ***")

print("\n" + "=" * 65)
print("DONE")
