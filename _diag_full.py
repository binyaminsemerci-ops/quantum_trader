import redis, time, subprocess

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

print("=== 1. SERVICES ===")
for svc in ["quantum-reconcile-engine","quantum-harvest-brain","quantum-position-state-brain",
            "quantum-balance-tracker","quantum-intent-executor","quantum-apply-layer"]:
    res = subprocess.run(["systemctl","is-active",svc], capture_output=True, text=True)
    print(f"  {svc}: {res.stdout.strip()}")

print()
print("=== 2. POSITION SNAPSHOT STREAM ===")
entries = r.xrevrange("quantum:stream:position.snapshot", count=1)
if entries:
    eid, data = entries[0]
    age = int(time.time()) - int(eid.split("-")[0])//1000
    print(f"  Last entry: {age}s ago  (OK if <30s)")
else:
    print("  EMPTY - stream dead!")

print()
print("=== 3. OPEN POSITIONS + LEDGER HEALTH ===")
pos_keys = [k for k in r.keys("quantum:position:*")
            if ":" not in k.replace("quantum:position:","")]
for pk in sorted(pos_keys):
    sym = pk.replace("quantum:position:","")
    pos    = r.hgetall(pk)
    ledger = r.hgetall(f"quantum:position:ledger:{sym}")
    snap   = r.hgetall(f"quantum:position:snapshot:{sym}")
    l_amt  = ledger.get("last_known_amt", ledger.get("ledger_amt","MISSING"))
    l_side = ledger.get("last_side",      ledger.get("ledger_side","MISSING"))
    snap_src = snap.get("source","no-snap")
    ledger_ok = l_amt not in ("MISSING","0","0.0","") and l_side not in ("MISSING","FLAT","NONE","")
    status = "OK" if ledger_ok else "BAD"
    print(f"  [{status}] {sym}: ledger_amt={l_amt} ledger_side={l_side} | snap_src={snap_src} pos_side={pos.get('side','?')}")

print()
print("=== 4. HARVEST BRAIN LAST 5 INTENTS ===")
entries = r.xrevrange("quantum:stream:trade.intent", count=5)
if entries:
    for eid, data in entries:
        age = int(time.time()) - int(eid.split("-")[0])//1000
        print(f"  {data.get('symbol','?')} action={data.get('action','?')} r={data.get('r_level','?')} age={age}s")
else:
    print("  NO INTENTS YET")

print()
print("=== 5. RECONCILE EVENTS (last 10) ===")
entries = r.xrevrange("quantum:stream:reconcile.event", count=10)
for eid, data in entries:
    age = int(time.time()) - int(eid.split("-")[0])//1000
    reason = data.get("reason","")
    print(f"  {data.get('symbol','?')} {data.get('event_type','?')} {age}s ago  {reason}")
if not entries:
    print("  none")

print()
print("=== 6. HOLD KEYS ===")
hold_keys = r.keys("quantum:hold:*")
for hk in sorted(hold_keys):
    ttl    = r.ttl(hk)
    reason = r.get(hk) or "?"
    print(f"  {hk}  TTL={ttl}s  reason={reason}")
if not hold_keys:
    print("  none")

print()
print("=== 7. GHOST PURGE COUNT (last reconcile.event stream) ===")
entries = r.xrevrange("quantum:stream:reconcile.event", count=100)
ghosts = [e for e in entries if e[1].get("event_type") == "GHOST_PURGE"]
print(f"  Ghost purge events in last 100: {len(ghosts)}")
if ghosts:
    eid, data = ghosts[0]
    age = int(time.time()) - int(eid.split("-")[0])//1000
    print(f"  Last ghost: {data.get('symbol')} {age}s ago")
