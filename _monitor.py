#!/usr/bin/env python3
"""
Live monitor: trades siste 30 min, PnL, aktive posisjoner, cooldown-status, service health
"""
import redis, time, json
from datetime import datetime, timezone

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
now = int(time.time() * 1000)
cutoff_30m = now - (30 * 60 * 1000)
cutoff_1h  = now - (60 * 60 * 1000)
cutoff_24h = now - (24 * 3600 * 1000)

def ts_to_str(ms):
    return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime('%H:%M:%S')

print("=" * 65)
print(f"LIVE MONITOR  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 65)

# ── 1. Service health ──────────────────────────────────────────
import subprocess
services = ['quantum-intent-executor', 'quantum-apply-layer', 'quantum-autonomous-trader']
print("\n[SERVICES]")
for svc in services:
    r2 = subprocess.run(['systemctl', 'is-active', svc], capture_output=True, text=True)
    status = r2.stdout.strip()
    icon = "✅" if status == "active" else "❌"
    print(f"  {icon} {svc.replace('quantum-','')}: {status}")

# ── 2. Aktive posisjoner ───────────────────────────────────────
print("\n[AKTIVE POSISJONER]")
pos_keys = r.keys("quantum:position:*")
active = []
for k in pos_keys:
    try:
        d = r.hgetall(k)
        qty = float(d.get('quantity', 0))
        status = d.get('status', '')
        if qty > 0 and status not in ('CLOSED', 'CLOSED_GHOST', ''):
            sym = k.split(':')[-1]
            side = d.get('side', '?')
            entry = float(d.get('entry_price', 0))
            r_net = d.get('R_net', '?')
            pnl = d.get('unrealized_pnl', '?')
            active.append((sym, side, entry, r_net, pnl))
    except:
        pass

if active:
    for sym, side, entry, r_net, pnl in active:
        print(f"  {sym:15s} {side:5s} entry={entry:.4f}  R={r_net}  uPnL={pnl}")
else:
    print("  (ingen aktive posisjoner)")

# ── 3. Trades siste 30 min ────────────────────────────────────
print("\n[TRADES SISTE 30 MIN]")
msgs = r.xrange('quantum:stream:trade.closed', min=f"{cutoff_30m}-0", max='+')
trades_30m = []
for msg_id, fields in msgs:
    try:
        pnl = float(fields.get('pnl_usd', 0))
        sym = fields.get('symbol', '?')
        reason = fields.get('reason', '?')
        r_net = float(fields.get('R_net', 0))
        ts = ts_to_str(msg_id.split('-')[0])
        trades_30m.append((ts, sym, pnl, r_net, reason))
    except:
        pass

if trades_30m:
    total_pnl = sum(t[2] for t in trades_30m)
    wins = sum(1 for t in trades_30m if t[2] > 0)
    wr = wins / len(trades_30m) * 100 if trades_30m else 0
    print(f"  Antall: {len(trades_30m)}  |  PnL: ${total_pnl:+.2f}  |  WR: {wr:.0f}%")
    for ts, sym, pnl, r_val, reason in trades_30m[-10:]:
        icon = "+" if pnl > 0 else "-"
        print(f"  {ts}  {sym:14s}  ${pnl:+6.2f}  R={r_val:.2f}  [{reason}]")
    if len(trades_30m) > 10:
        print(f"  ... (+{len(trades_30m)-10} flere)")
else:
    print("  Ingen trades siste 30 min")

# ── 4. PnL siste 1t og 24t ────────────────────────────────────
print("\n[PNL SAMMENDRAG]")
def calc_pnl(cutoff):
    msgs = r.xrange('quantum:stream:trade.closed', min=f"{cutoff}-0", max='+')
    trades = []
    for _, fields in msgs:
        try:
            trades.append(float(fields.get('pnl_usd', 0)))
        except:
            pass
    if not trades:
        return 0, 0, 0
    wins = sum(1 for p in trades if p > 0)
    return sum(trades), len(trades), wins/len(trades)*100

pnl_1h, n_1h, wr_1h = calc_pnl(cutoff_1h)
pnl_24h, n_24h, wr_24h = calc_pnl(cutoff_24h)
print(f"  1t:   ${pnl_1h:+7.2f}  ({n_1h} trades, WR={wr_1h:.0f}%)")
print(f"  24t:  ${pnl_24h:+7.2f}  ({n_24h} trades, WR={wr_24h:.0f}%)")

# ── 5. Churn sjekk — trades < 5 min etter forrige ─────────────
print("\n[CHURN CHECK siste 1t]")
msgs_1h = r.xrange('quantum:stream:trade.closed', min=f"{cutoff_1h}-0", max='+')
by_sym = {}
for msg_id, fields in msgs_1h:
    sym = fields.get('symbol', '?')
    ts = int(msg_id.split('-')[0])
    if sym not in by_sym:
        by_sym[sym] = []
    by_sym[sym].append(ts)

churn = []
for sym, tss in by_sym.items():
    tss.sort()
    for i in range(1, len(tss)):
        gap_min = (tss[i] - tss[i-1]) / 60000
        if gap_min < 5:
            churn.append((sym, gap_min))

if churn:
    print(f"  ⚠️  {len(churn)} churn-tilfeller (< 5 min gap):")
    for sym, gap in churn[:5]:
        print(f"    {sym}: {gap:.1f} min gap")
else:
    print("  ✅ Ingen churn (alle gaps ≥ 5 min)")

# ── 6. Aktive cooldowns ───────────────────────────────────────
print("\n[COOLDOWNS AKTIVE]")
cd_keys = r.keys("quantum:cooldown:open:*")
if cd_keys:
    print(f"  {len(cd_keys)} symboler i open-cooldown:")
    for k in sorted(cd_keys)[:8]:
        ttl = r.ttl(k)
        sym = k.split(':')[-1]
        print(f"    {sym:14s}  TTL={ttl}s ({ttl//60}m {ttl%60}s)")
    if len(cd_keys) > 8:
        print(f"    ... (+{len(cd_keys)-8} til)")
else:
    print("  Ingen aktive open-cooldowns")

# ── 7. Intent stats ────────────────────────────────────────────
print("\n[INTENT PIPELINE siste 30 min]")
intent_msgs = r.xrange('quantum:stream:intent', min=f"{cutoff_30m}-0", max='+')
harvest_msgs = r.xrange('quantum:stream:intent.harvest', min=f"{cutoff_30m}-0", max='+') if r.exists('quantum:stream:intent.harvest') else []
print(f"  Intents generert: {len(intent_msgs)}")
print(f"  Harvest intents:  {len(harvest_msgs)}")

print("\n" + "=" * 65)
