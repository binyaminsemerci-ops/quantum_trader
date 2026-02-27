#!/usr/bin/env python3
"""
Full system audit — Feb 25, 2026
Covers: services, Redis state, execution pipeline, errors, guard blocks, positions
"""
import redis, re, time, subprocess
from collections import Counter
from datetime import datetime, timezone

r = redis.Redis()
now = int(time.time())
now_ms = now * 1000

SEP = "=" * 60

# ─── 1. SERVICES ─────────────────────────────────────────────
print(f"\n{SEP}")
print("1. SYSTEMD SERVICES")
print(SEP)
res = subprocess.run(
    ["systemctl", "list-units", "--type=service", "--all", "--no-pager",
     "--plain", "--no-legend"],
    capture_output=True, text=True
)
quantum_svcs = [l for l in res.stdout.splitlines() if "quantum" in l.lower()]
for line in quantum_svcs:
    parts = line.split()
    name = parts[0] if parts else "?"
    active = parts[2] if len(parts) > 2 else "?"
    sub = parts[3] if len(parts) > 3 else "?"
    status = "✅" if sub == "running" else "❌"
    print(f"  {status} {name:<55} {sub}")

# ─── 2. REDIS KEY INVENTORY ──────────────────────────────────
print(f"\n{SEP}")
print("2. REDIS KEY INVENTORY")
print(SEP)
categories = {
    "positions":    ("quantum:position:*", 0),
    "harvest_prop": ("quantum:harvest:proposal:*", 0),
    "harvest_heat": ("quantum:harvest:heat:*", 0),
    "harvest_v2":   ("quantum:harvest_v2:state:*", 0),
    "snapshots":    ("quantum:position:snapshot:*", 0),
    "harvest_plan": ("quantum:harvest:plan:*", 0),
    "apply_plan":   ("quantum:stream:apply.plan*", 0),
}
for label, (pattern, _) in categories.items():
    keys = list(r.scan_iter(pattern, count=500))
    categories[label] = (pattern, len(keys))
    syms = sorted({k.decode().split(":")[-1] for k in keys})
    sym_str = ", ".join(syms[:10]) if syms else "—"
    print(f"  {label:<20} {len(keys):>3} keys   [{sym_str}]")

# ─── 3. EXECUTION PIPELINE LAST 10 MIN ───────────────────────
print(f"\n{SEP}")
print("3. EXECUTION PIPELINE — last 10 min")
print(SEP)
cutoff = (now - 600) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff))
decisions = Counter()
errors = Counter()
for _, d in entries:
    dec = d.get(b"decision", b"None").decode()
    err = d.get(b"error", b"").decode()
    decisions[dec] += 1
    if err:
        m = re.search(r'"code"\s*:\s*(-?[0-9]+)', err)
        if m:
            errors[f"Binance_{m.group(1)}"] += 1
        elif err not in ("", "no_position", "kill_score_close_ok"):
            errors[err[:40]] += 1

print(f"  Total apply.result entries: {len(entries)}")
print(f"  Decisions: {dict(decisions.most_common(6))}")
print(f"  Binance errors: {dict(errors) if errors else 'ZERO ✅'}")

# ─── 4. INTENT-EXECUTOR GUARD BLOCKS ─────────────────────────
print(f"\n{SEP}")
print("4. INTENT-EXECUTOR — P3.5_GUARD stats")
print(SEP)
guard_key = "quantum:intent_executor:stats"
if r.exists(guard_key):
    stats = {k.decode(): v.decode() for k, v in r.hgetall(guard_key).items()}
    for k, v in sorted(stats.items()):
        print(f"  {k:<35} {v}")
else:
    # Try reading from execution log
    res2 = subprocess.run(
        ["grep", "-a", "p35_guard_blocked\|kelly\|min_edge\|executed_true",
         "/var/log/quantum/execution.log"],
        capture_output=True, text=True
    )
    lines = res2.stdout.strip().splitlines()
    for l in lines[-5:]:
        print(f"  {l.strip()[:100]}")

# ─── 5. MANUAL LANE ──────────────────────────────────────────
print(f"\n{SEP}")
print("5. MANUAL LANE")
print(SEP)
ml = r.get("quantum:manual_lane:enabled")
ml_ttl = r.ttl("quantum:manual_lane:enabled")
if ml:
    exp_utc = datetime.fromtimestamp(now + ml_ttl, tz=timezone.utc).strftime("%H:%M UTC")
    print(f"  Status: ACTIVE ✅  value={ml.decode()}  TTL={ml_ttl}s (~{ml_ttl//3600}h{(ml_ttl%3600)//60}m, expires {exp_utc})")
else:
    print("  Status: INACTIVE ❌")

# ─── 6. POSITIONS IN REDIS ───────────────────────────────────
print(f"\n{SEP}")
print("6. OPEN POSITIONS IN REDIS")
print(SEP)
pos_keys = list(r.scan_iter("quantum:position:*", count=500))
pos_keys = [k for k in pos_keys if b"snapshot" not in k and b"backup" not in k and b"phantom" not in k]
if not pos_keys:
    print("  No open positions ✅")
for key in sorted(pos_keys):
    sym = key.decode().split(":")[-1]
    typ = r.type(key).decode()
    if typ == "hash":
        d = {k.decode(): v.decode() for k, v in r.hgetall(key).items()}
        side = d.get("side", "?")
        qty = d.get("quantity", d.get("qty", "?"))
        entry = d.get("entry_price", d.get("open_price", "?"))
        pnl = d.get("unrealized_pnl", "?")
        risk = d.get("entry_risk_usdt", "?")
        print(f"  {sym:<12} side={side} qty={qty} entry={entry} upnl={pnl} risk_usdt={risk}")
    else:
        print(f"  {sym:<12} [{typ}]")

# ─── 7. RECENT EXECUTION LOG ─────────────────────────────────
print(f"\n{SEP}")
print("7. EXECUTION LOG — last 5 lines")
print(SEP)
res3 = subprocess.run(
    ["tail", "-5", "/var/log/quantum/execution.log"],
    capture_output=True, text=True
)
for l in res3.stdout.strip().splitlines():
    print(f"  {l.strip()[:110]}")

# ─── 8. HARVEST PROPOSAL PUBLISHER ───────────────────────────
print(f"\n{SEP}")
print("8. HARVEST PROPOSAL SERVICE — current symbols")
print(SEP)
res4 = subprocess.run(["grep", "SYMBOLS", "/etc/quantum/harvest-proposal.env"],
                      capture_output=True, text=True)
print(f"  {res4.stdout.strip()}")

# ─── 9. SUMMARY ──────────────────────────────────────────────
print(f"\n{SEP}")
print("9. SUMMARY")
print(SEP)
print(f"  Timestamp: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Binance errors (last 10min): {sum(errors.values()) if errors else 0}")
print(f"  Phantom positions: {sum(1 for k in pos_keys if True)} total pos keys")
print(f"  Manual lane: {'ACTIVE' if ml else 'INACTIVE'}")
print(f"  Harvest publisher symbols: {res4.stdout.strip().replace('SYMBOLS=','')}")
