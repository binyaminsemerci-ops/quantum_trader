#!/usr/bin/env python3
"""Full execution layer diagnosis"""
import subprocess, redis, json, time, sys

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

SERVICES = [
    "quantum-harvest-brain",
    "quantum-reconcile-engine",
    "quantum-apply-layer",
    "quantum-intent-executor",
    "quantum-exit-brain",
    "quantum-position-state-brain",
    "quantum-risk-guard",
]

STREAMS = [
    "quantum:stream:harvest.intents",
    "quantum:stream:reconcile.close",
    "quantum:stream:apply.plans",
    "quantum:stream:execution.results",
    "quantum:stream:reconcile.alert",
    "quantum:stream:reconcile.events",
]

def svc_status(name):
    try:
        out = subprocess.check_output(
            ["systemctl", "is-active", name], stderr=subprocess.DEVNULL
        ).decode().strip()
        if out == "active":
            ts = subprocess.check_output(
                ["systemctl", "show", name, "--property=ExecMainStartTimestamp"],
                stderr=subprocess.DEVNULL
            ).decode().strip().replace("ExecMainStartTimestamp=", "")
            return f"✅ ACTIVE  (since {ts})"
        return f"❌ {out.upper()}"
    except:
        return "❓ UNKNOWN"

print("=" * 70)
print("EXECUTION LAYER DIAGNOSTICS")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 70)

# --- 1. Services ---
print("\n── 1. SERVICES ─────────────────────────────────────────────────")
for svc in SERVICES:
    print(f"  {svc:<40} {svc_status(svc)}")

# --- 2. Open positions (Redis vs Binance) ---
print("\n── 2. REDIS POSITIONS ───────────────────────────────────────────")
pos_keys = sorted([
    k for k in r.keys("quantum:position:*")
    if ":snapshot:" not in k and ":ledger:" not in k and ":claim:" not in k
])
print(f"  Count: {len(pos_keys)}")
for key in pos_keys:
    d = r.hgetall(key)
    sym  = key.replace("quantum:position:", "")
    side = d.get("side", "?")
    qty  = d.get("quantity", "?")
    pnl  = d.get("unrealized_pnl", "?")
    try:
        flag = f" (pnl={float(pnl):+.3f})"
    except:
        flag = ""
    print(f"  {sym:<22} {side:<6} qty={qty}{flag}")

# --- 3. Dedup keys ---
print("\n── 3. DEDUP KEYS (harvest cooldowns) ────────────────────────────")
dedup_keys = r.keys("quantum:dedup:harvest:*")
print(f"  Active dedup locks: {len(dedup_keys)}")
for k in sorted(dedup_keys)[:20]:
    ttl = r.ttl(k)
    print(f"  {k}  TTL={ttl}s")

# --- 4. Streams (last message age) ---
print("\n── 4. STREAMS ───────────────────────────────────────────────────")
now_ms = int(time.time() * 1000)
for stream in STREAMS:
    try:
        msgs = r.xrevrange(stream, count=1)
        if msgs:
            msg_id, _ = msgs[0]
            ts_ms = int(msg_id.split("-")[0])
            age   = round((now_ms - ts_ms) / 1000)
            length = r.xlen(stream)
            print(f"  {stream:<45} len={length:<6} last={age}s ago")
        else:
            print(f"  {stream:<45} EMPTY")
    except Exception as e:
        print(f"  {stream:<45} ERROR: {e}")

# --- 5. Recent harvest intents (last 5) ---
print("\n── 5. LAST 5 HARVEST INTENTS ────────────────────────────────────")
try:
    msgs = r.xrevrange("quantum:stream:harvest.intents", count=5)
    if msgs:
        for msg_id, data in msgs:
            ts_ms  = int(msg_id.split("-")[0])
            age    = round((now_ms - ts_ms) / 1000)
            sym    = data.get("symbol", "?")
            intent = data.get("intent_type", "?")
            qty    = data.get("qty", "?")
            notional = data.get("notional", "?")
            deduped  = data.get("deduplicated", "?")
            print(f"  [{age}s ago] {sym:<20} {intent:<22} qty={qty} notional={notional} dedup={deduped}")
    else:
        print("  No messages in stream")
except Exception as e:
    print(f"  ERROR: {e}")

# --- 6. Recent ghost purge events ---
print("\n── 6. RECENT GHOST PURGE EVENTS ─────────────────────────────────")
try:
    msgs = r.xrevrange("quantum:stream:reconcile.events", count=20)
    ghost_msgs = [(mid, d) for mid, d in msgs if d.get("event") == "GHOST_PURGE"]
    if ghost_msgs:
        for msg_id, data in ghost_msgs[:10]:
            ts_ms = int(msg_id.split("-")[0])
            age   = round((now_ms - ts_ms) / 1000)
            print(f"  [{age}s ago] GHOST_PURGE {data.get('symbol','?')}  reason={data.get('reason','?')}")
    else:
        print("  No GHOST_PURGE events found")
except Exception as e:
    print(f"  ERROR: {e}")

# --- 7. Reconcile hold keys ---
print("\n── 7. RECONCILE HOLDS ───────────────────────────────────────────")
hold_keys = r.keys("quantum:reconcile:hold:*")
print(f"  Active holds: {len(hold_keys)}")
for k in sorted(hold_keys):
    ttl = r.ttl(k)
    sym = k.replace("quantum:reconcile:hold:", "")
    print(f"  {sym:<20} TTL={ttl}s")

# --- 8. Circuit breaker / kill switch ---
print("\n── 8. CIRCUIT BREAKERS ──────────────────────────────────────────")
kb = r.get("quantum:global:kill_switch")
print(f"  global kill_switch : {kb or 'OFF'}")
cb = r.hgetall("quantum:circuit_breaker")
if cb:
    print(f"  circuit_breaker    : {cb}")
else:
    print(f"  circuit_breaker    : not set")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
