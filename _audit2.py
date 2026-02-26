#!/usr/bin/env python3
"""Full system audit v2 - no subprocesses, pure Redis."""
import redis, time, re
from collections import Counter

r = redis.Redis()
now = int(time.time())

print("=" * 60)
print("QUANTUM TRADER — SYSTEM AUDIT")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print("=" * 60)

# ── 1. Redis key summary ──────────────────────────────────────
print("\n=== 1. REDIS KEY GROUPS ===")
key_groups = {}
for key in r.scan_iter("quantum:*", count=1000):
    parts = key.decode().split(":")
    group = ":".join(parts[:3]) if len(parts) >= 3 else ":".join(parts[:2])
    key_groups[group] = key_groups.get(group, 0) + 1
for g, cnt in sorted(key_groups.items(), key=lambda x: -x[1])[:25]:
    print(f"  {g:<50} {cnt:>4}")

# ── 2. Stream health ──────────────────────────────────────────
print("\n=== 2. STREAM HEALTH ===")
streams = [
    "quantum:stream:apply.result",
    "quantum:stream:apply.plan",
    "quantum:stream:apply.plan.manual",
    "quantum:stream:trade.intent",
    "quantum:stream:execution",
    "quantum:stream:ai.signal",
    "quantum:stream:market.data",
    "quantum:stream:harvest.proposal",
]
for s in streams:
    try:
        info = r.xinfo_stream(s)
        length = info.get("length", 0)
        last_entry = info.get("last-entry") or info.get("last_entry")
        if last_entry:
            last_ms = int(last_entry[0].decode().split("-")[0])
            age_h = (now - last_ms // 1000) / 3600
            age_str = f"{age_h:.1f}h ago" if age_h > 1 else f"{(age_h*60):.0f}m ago"
        else:
            age_str = "empty"
        print(f"  {s:<45} len={length:>6}  last={age_str}")
    except Exception as e:
        print(f"  {s:<45} MISSING")

# ── 3. Recent error rates ─────────────────────────────────────
print("\n=== 3. BINANCE ERRORS LAST 10 MIN ===")
cutoff = (now - 600) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff))
binance_errs = []
other_errs = Counter()
for _, d in entries:
    err = d.get(b"error", b"").decode()
    if "Binance API error" in err:
        sym = d.get(b"symbol", b"?").decode()
        m = re.search(r'"code"\s*:\s*(-?[0-9]+)', err)
        binance_errs.append((sym, m.group(1) if m else "?"))
    elif err and err not in ("", "null"):
        other_errs[err[:50]] += 1

if binance_errs:
    for (s, c), n in Counter(binance_errs).most_common():
        print(f"  ❌ {s} code={c} x{n}")
else:
    print("  ✅ ZERO Binance errors")
print(f"  apply.result entries in 10 min: {len(entries)}")
if other_errs:
    print("  Other errors:")
    for e, n in other_errs.most_common(5):
        print(f"    {e} x{n}")

# ── 4. Active positions ───────────────────────────────────────
print("\n=== 4. ACTIVE POSITIONS ===")
pos_keys = [k for k in r.scan_iter("quantum:position:*", count=500)
            if not any(x in k for x in [b"snapshot", b"backup", b"phantom"])]
if pos_keys:
    for k in pos_keys:
        try:
            data = {kk.decode(): vv.decode() for kk, vv in r.hgetall(k).items()}
            sym = data.get("symbol", k.decode().split(":")[-1])
            side = data.get("side", "?")
            qty = data.get("quantity", data.get("qty", "?"))
            entry = data.get("entry_price", data.get("open_price", "?"))
            risk = data.get("entry_risk_usdt", "?")
            pnl = data.get("unrealized_pnl", "?")
            print(f"  {sym:<12} {side:<6} qty={qty}  entry={entry}  risk_usdt={risk}  pnl={pnl}")
        except:
            print(f"  {k.decode()}")
else:
    print("  No active positions")

# ── 5. Manual lane ────────────────────────────────────────────
print("\n=== 5. MANUAL LANE ===")
ml = r.get("quantum:manual_lane:enabled")
ttl = r.ttl("quantum:manual_lane:enabled")
if ml:
    print(f"  ✅ ACTIVE  TTL={ttl}s ({ttl/3600:.2f}h remaining)")
else:
    print("  ❌ INACTIVE")

# ── 6. Harvest proposals ─────────────────────────────────────
print("\n=== 6. HARVEST PROPOSALS ===")
found = False
for key in r.scan_iter("quantum:harvest:proposal:*", count=200):
    found = True
    sym = key.decode().split(":")[-1]
    data = {k.decode(): v.decode() for k, v in r.hgetall(key).items()}
    action = data.get("harvest_action", "?")
    entry = data.get("position_entry_price", "?")
    computed = data.get("computed_at_utc", "?")[:19]
    print(f"  {sym:<12} action={action:<25} entry={entry}  @{computed}")
if not found:
    print("  No proposals found")

# ── 7. Portfolio / risk state ─────────────────────────────────
print("\n=== 7. PORTFOLIO / RISK ===")
pf = r.hgetall("quantum:state:portfolio")
if pf:
    for k, v in sorted(pf.items()):
        print(f"  {k.decode():<35} = {v.decode()}")
else:
    print("  (quantum:state:portfolio missing)")
for k in ["quantum:portfolio:drawdown", "quantum:portfolio:balance",
          "quantum:portfolio:peak_balance", "quantum:risk:drawdown_pct",
          "quantum:circuit_breaker:active"]:
    v = r.get(k)
    if v:
        print(f"  {k.split(':',1)[1]:<35} = {v.decode()}")

# ── 8. Intent-bridge / P3.5 stats ──────────────────────────
print("\n=== 8. P3.5 GUARD STATS ===")
# p35 bucket count
p35_count = sum(1 for _ in r.scan_iter("quantum:p35:*", count=500))
print(f"  {'p35_bucket_keys':<35} = {p35_count}")
# intent bridge seen dedup count
ib_count = sum(1 for _ in r.scan_iter("quantum:intent_bridge:seen:*", count=500))
print(f"  {'intent_bridge_seen_keys':<35} = {ib_count}")
# permit p33 count (approved signals)
p33_count = sum(1 for _ in r.scan_iter("quantum:permit:p33:*", count=500))
print(f"  {'permit_p33_approved_keys':<35} = {p33_count}")
# Sample one p35 key for kelly/edge
for k in r.scan_iter("quantum:p35:*", count=10):
    typ = r.type(k).decode()
    if typ == "hash":
        data = {kk.decode(): vv.decode() for kk, vv in r.hgetall(k).items()}
        print(f"  sample p35: {k.decode().split(':')[-1]} -> {data}")
        break
    elif typ == "string":
        print(f"  sample p35: {k.decode()} = {r.get(k).decode()[:80]}")
        break
# legacy stats keys
for k in ["quantum:stats:p35_guard_blocked", "quantum:stats:executed_true",
          "quantum:intent_executor:stats"]:
    v = r.get(k) if r.exists(k) and r.type(k).decode() == "string" else None
    if v:
        print(f"  {k.split(':')[-1]:<35} = {v.decode()}")

# ── 9. Circuit breaker ────────────────────────────────────────
print("\n=== 9. CIRCUIT BREAKER ===")
cb_keys = list(r.scan_iter("quantum:circuit_breaker:*", count=100))
if cb_keys:
    for k in cb_keys:
        typ = r.type(k).decode()
        if typ == "string":
            print(f"  {k.decode():<45} = {r.get(k).decode()}")
        elif typ == "hash":
            data = {kk.decode(): vv.decode() for kk, vv in r.hgetall(k).items()}
            print(f"  {k.decode()}: {data}")
else:
    print("  No circuit breaker keys")

print("\n=== AUDIT COMPLETE ===")
