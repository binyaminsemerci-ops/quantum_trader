#!/usr/bin/env python3
"""Deep investigation of all 4 diagnosis findings"""
import subprocess, redis, json, time, sys

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
now_ms = int(time.time() * 1000)

# ============================================================
# FINDING 1: All running quantum services
# ============================================================
print("=" * 70)
print("FINDING 1: ALL QUANTUM SERVICES (running + dead)")
print("=" * 70)
out = subprocess.check_output(
    ["systemctl", "list-units", "--type=service", "--all", "--no-pager"],
    stderr=subprocess.DEVNULL
).decode()
for line in out.splitlines():
    if "quantum" in line.lower():
        print(" ", line.strip())

# ============================================================
# FINDING 2: Why harvest brain produces no intents
# ============================================================
print("\n" + "=" * 70)
print("FINDING 2: HARVEST BRAIN LOG (last 60 lines)")
print("=" * 70)
out = subprocess.check_output(
    ["journalctl", "-u", "quantum-harvest-brain", "--no-pager", "-n", "60"],
    stderr=subprocess.DEVNULL
).decode()
# Show only actionable lines
for line in out.splitlines():
    if any(x in line for x in ["emitted", "EMIT", "SKIP", "HOLD", "ERROR", "WARNING",
                                 "evaluate", "no intents", "harvest_action", "PARTIAL",
                                 "FULL_CLOSE", "EMERGENCY", "MOVE_SL", "min_notional",
                                 "SKIP_MIN", "positions found", "position loaded"]):
        print(" ", line.strip()[-160:])

# Show raw last 10 lines regardless
print("\n  [Last 10 raw lines]:")
for line in out.splitlines()[-10:]:
    print(" ", line.strip()[-160:])

# ============================================================
# FINDING 3: Ghost purge events missing from stream
# ============================================================
print("\n" + "=" * 70)
print("FINDING 3: RECONCILE.EVENTS STREAM (last 20)")
print("=" * 70)
try:
    msgs = r.xrevrange("quantum:stream:reconcile.events", count=20)
    for msg_id, data in msgs:
        ts_ms = int(msg_id.split("-")[0])
        age   = round((now_ms - ts_ms) / 1000)
        print(f"  [{age:>5}s ago] event={data.get('event','?'):<20} symbol={data.get('symbol','?'):<20} reason={data.get('reason','?')}")
except Exception as e:
    print(f"  ERROR: {e}")

# ============================================================
# FINDING 4: Harvest brain — what's in positions vs thresholds
# ============================================================
print("\n" + "=" * 70)
print("FINDING 4: POSITION DETAILS vs HARVEST THRESHOLDS")
print("=" * 70)
pos_keys = sorted([
    k for k in r.keys("quantum:position:*")
    if ":snapshot:" not in k and ":ledger:" not in k and ":claim:" not in k
])
for key in pos_keys:
    d = r.hgetall(key)
    sym    = key.replace("quantum:position:", "")
    side   = d.get("side", "?")
    qty    = d.get("quantity", "?")
    entry  = d.get("entry_price", "?")
    sl     = d.get("stop_loss", "?")
    tp     = d.get("take_profit", "?")
    pnl    = d.get("unrealized_pnl", "?")
    lev    = d.get("leverage", "?")
    risk   = d.get("entry_risk_usdt", "?")
    opened = d.get("opened_at", "?")
    try:
        pnl_f   = float(pnl)
        entry_f = float(entry)
        sl_f    = float(sl)
        tp_f    = float(tp)
        qty_f   = float(qty)
        risk_f  = float(risk)
        # R-multiple approximation
        if side == "SHORT":
            r_net = (entry_f - sl_f) / abs(entry_f - sl_f) if sl_f != entry_f else 0
            pnl_r = pnl_f / risk_f if risk_f else 0
        else:
            pnl_r = pnl_f / risk_f if risk_f else 0
        age_min = round((time.time() - int(opened)) / 60) if opened != "?" else "?"
        print(f"  {sym:<22} {side:<6} entry={entry:<14} sl={sl:<14} pnl={pnl_f:+.4f} R={pnl_r:+.2f} lev={lev} age={age_min}min")
    except Exception as ex:
        print(f"  {sym:<22} raw: {d}")

# ============================================================
# FINDING 4b: Snapshot freshness for open positions
# ============================================================
print("\n── Snapshot freshness ───────────────────────────────────────────")
for key in pos_keys:
    sym = key.replace("quantum:position:", "")
    snap_key = f"quantum:position:snapshot:{sym}"
    snap = r.hgetall(snap_key)
    if snap:
        ts = snap.get("ts_epoch") or snap.get("sync_timestamp") or snap.get("ts") or "?"
        try:
            age_s = round(time.time() - int(ts))
            print(f"  {sym:<22} snapshot age={age_s}s")
        except:
            print(f"  {sym:<22} snapshot ts={ts}")
    else:
        print(f"  {sym:<22} NO SNAPSHOT")

# ============================================================
# FINDING 4c: Check what stream harvest brain reads from
# ============================================================
print("\n── Harvest Brain input streams (last msg age) ───────────────────")
input_streams = [
    "quantum:stream:position.snapshot",
    "quantum:position:snapshot:*",  # just count
    "quantum:stream:market.tick",
    "quantum:stream:portfolio.state",
]
for s in input_streams:
    if "*" in s:
        cnt = len(r.keys(s))
        print(f"  {s:<45} count={cnt}")
        continue
    try:
        msgs = r.xrevrange(s, count=1)
        if msgs:
            msg_id, _ = msgs[0]
            ts_ms = int(msg_id.split("-")[0])
            age   = round((now_ms - ts_ms) / 1000)
            length = r.xlen(s)
            print(f"  {s:<45} len={length:<6} last={age}s ago")
        else:
            print(f"  {s:<45} EMPTY")
    except Exception as e:
        print(f"  {s:<45} ERROR: {e}")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
