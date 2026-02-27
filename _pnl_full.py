#!/usr/bin/env python3
import redis, json
from datetime import datetime, timezone, timedelta
from collections import defaultdict

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=" * 60)
print("REALIZED PnL — KOMPLETT HISTORIKK")
print("=" * 60)

# Hent altinn
all_entries = r.xrange("quantum:stream:trade.closed", min="-", max="+")
total = len(all_entries)
print(f"Totalt i stream: {total} lukkede handler")

if not all_entries:
    print("Ingen data.")
    exit()

# Parse alle
trades = []
for eid, fields in all_entries:
    ts = datetime.fromtimestamp(int(eid.split('-')[0])/1000, tz=timezone.utc)
    trades.append({
        "ts": ts,
        "symbol": fields.get("symbol", "?"),
        "side": fields.get("side", "?"),
        "entry": float(fields.get("entry_price", 0) or 0),
        "exit": float(fields.get("exit_price", 0) or 0),
        "pnl_pct": float(fields.get("pnl_percent", 0) or 0),
        "pnl_usd": float(fields.get("pnl_usd", 0) or 0),
        "r_net": float(fields.get("R_net", 0) or 0),
        "reason": fields.get("reason", ""),
        "order_id": fields.get("order_id", ""),
        "source": fields.get("source", ""),
    })

# Første og siste trade
print(f"Tidligste: {trades[0]['ts'].strftime('%Y-%m-%d %H:%M UTC')}  ({trades[0]['symbol']})")
print(f"Nyligste:  {trades[-1]['ts'].strftime('%Y-%m-%d %H:%M UTC')}  ({trades[-1]['symbol']})")

def summarize(trade_list, label):
    if not trade_list:
        print(f"\n  {label}: ingen handler")
        return
    total_usd = sum(t["pnl_usd"] for t in trade_list)
    total_r = sum(t["r_net"] for t in trade_list)
    wins = sum(1 for t in trade_list if t["pnl_usd"] > 0)
    losses = sum(1 for t in trade_list if t["pnl_usd"] <= 0)
    best = max(trade_list, key=lambda x: x["pnl_usd"])
    worst = min(trade_list, key=lambda x: x["pnl_usd"])
    wr = wins / len(trade_list) * 100 if trade_list else 0

    print(f"\n── {label} ({len(trade_list)} handler) ──")
    print(f"  Realized PnL:     ${total_usd:+.2f}")
    print(f"  Sum R:            {total_r:+.2f}R")
    print(f"  Win rate:         {wins}W / {losses}L  = {wr:.0f}%")
    print(f"  Avg per trade:    ${total_usd/len(trade_list):+.2f}")
    print(f"  Best:  {best['symbol']:12} ${best['pnl_usd']:+.2f}  {best['ts'].strftime('%m-%d %H:%M')}")
    print(f"  Worst: {worst['symbol']:12} ${worst['pnl_usd']:+.2f}  {worst['ts'].strftime('%m-%d %H:%M')}")
    return total_usd, total_r, wins, losses

now = datetime.now(timezone.utc)

# Perioder
h1 = [t for t in trades if t["ts"] >= now - timedelta(hours=1)]
h6 = [t for t in trades if t["ts"] >= now - timedelta(hours=6)]
h24 = [t for t in trades if t["ts"] >= now - timedelta(hours=24)]
all_time = trades

summarize(h1, "SISTE TIME")
summarize(h6, "SISTE 6 TIMER")
summarize(h24, "SISTE 24 TIMER")

# All time per dag
print("\n── DAGLIG BREAKDOWN (all time) ──")
by_day = defaultdict(list)
for t in trades:
    day = t["ts"].strftime("%Y-%m-%d")
    by_day[day].append(t)

total_all = 0.0
for day in sorted(by_day.keys()):
    day_trades = by_day[day]
    pnl = sum(t["pnl_usd"] for t in day_trades)
    r_sum = sum(t["r_net"] for t in day_trades)
    wins = sum(1 for t in day_trades if t["pnl_usd"] > 0)
    total_all += pnl
    print(f"  {day}  {len(day_trades):4}x  PnL=${pnl:+8.2f}  R={r_sum:+6.2f}  W={wins}/{len(day_trades)}")

print(f"\n  {'ALL TIME':10}  {total:4}x  PnL=${total_all:+8.2f}")

# Siste times detaljer
if h1:
    print(f"\n── SISTE TIME — HVER TRADE ──")
    for t in h1:
        print(f"  {t['ts'].strftime('%H:%M:%S')}  {t['symbol']:12}  {t['side']:5}  "
              f"${t['pnl_usd']:+7.2f}  {t['pnl_pct']:+5.2f}%  R={t['r_net']:+.2f}  "
              f"{t['reason'][:50]}")

# Symbol summary all time
print(f"\n── TOP 10 SYMBOLER (all time, etter PnL) ──")
by_sym = defaultdict(list)
for t in trades:
    by_sym[t["symbol"]].append(t)

sym_pnl = [(sym, sum(t["pnl_usd"] for t in ts), len(ts)) for sym, ts in by_sym.items()]
sym_pnl.sort(key=lambda x: x[1], reverse=True)

for sym, pnl, n in sym_pnl[:10]:
    print(f"  {sym:15}  {n:3}x  ${pnl:+8.2f}")

print(f"\n── BOTTOM 5 SYMBOLER (worst) ──")
for sym, pnl, n in sym_pnl[-5:]:
    print(f"  {sym:15}  {n:3}x  ${pnl:+8.2f}")
