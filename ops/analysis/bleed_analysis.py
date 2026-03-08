#!/usr/bin/env python3
"""
Capital Bleed Root Cause Analysis
===================================
Pulls last 48h of trade data from:
  1. Redis trade.closed stream (all exits)
  2. Binance testnet /fapi/v1/userTrades (fills)
  3. Binance testnet /fapi/v1/income (realized PnL, fees, funding)
  4. Redis positions + balance snapshot
  5. Redis recent proposals/intents (win/loss breakdown)

Run: python ops/analysis/bleed_analysis.py
"""
import os
import sys
import time
import json
import hashlib
import hmac
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import redis

# ── Config ──────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Binance testnet
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
BINANCE_BASE       = "https://testnet.binancefuture.com"

DAYS_BACK = 2
NOW_MS = int(time.time() * 1000)
START_MS = NOW_MS - int(DAYS_BACK * 24 * 3600 * 1000)

# ── Binance helpers ──────────────────────────────────────────────────────────
def _sign(params: dict) -> str:
    q = urllib.parse.urlencode(sorted(params.items()))
    return hmac.new(BINANCE_API_SECRET.encode(), q.encode(), hashlib.sha256).hexdigest()

def binance_get(path: str, params: dict) -> dict | list:
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    params["signature"] = _sign(params)
    url = BINANCE_BASE + path + "?" + urllib.parse.urlencode(sorted(params.items()))
    req = urllib.request.Request(url, headers={"X-MBX-APIKEY": BINANCE_API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

# ── Redis helpers ─────────────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def stream_range(stream: str, start_ms: int, count: int = 2000) -> list[dict]:
    """Return list of dicts from a Redis stream, newest first."""
    start_id = f"{start_ms}-0"
    entries = r.xrange(stream, min=start_id, max="+", count=count)
    results = []
    for entry_id, fields in entries:
        fields["_id"] = entry_id
        fields["_ts_ms"] = int(entry_id.split("-")[0])
        results.append(fields)
    return results

def ts_label(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Binance testnet income (realized PnL, fees, funding)
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  CAPITAL BLEED ROOT CAUSE ANALYSIS  —  last 48h")
print("=" * 72)

print("\n── 1. BINANCE TESTNET INCOME (realized PnL + fees + funding) ──────────")
income_types = ["REALIZED_PNL", "COMMISSION", "FUNDING_FEE"]
income_totals = {}
income_by_symbol = defaultdict(lambda: defaultdict(float))
income_rows = []

for itype in income_types:
    data = binance_get("/fapi/v1/income", {"incomeType": itype, "startTime": START_MS, "limit": 1000})
    if isinstance(data, list):
        total = sum(float(x.get("income", 0)) for x in data)
        income_totals[itype] = total
        for x in data:
            sym = x.get("symbol", "UNKNOWN")
            income_by_symbol[sym][itype] += float(x.get("income", 0))
            income_rows.append({
                "ts": ts_label(int(x.get("time", 0))),
                "type": itype,
                "symbol": sym,
                "income": float(x.get("income", 0)),
            })
    else:
        income_totals[itype] = None
        print(f"  ERROR fetching {itype}: {data}")

pnl_total   = income_totals.get("REALIZED_PNL") or 0
fee_total   = income_totals.get("COMMISSION") or 0
fund_total  = income_totals.get("FUNDING_FEE") or 0
net_total   = pnl_total + fee_total + fund_total

print(f"  Realized PnL:   {pnl_total:+.4f} USDT  ({len([r for r in income_rows if r['type']=='REALIZED_PNL'])} fills)")
print(f"  Commissions:    {fee_total:+.4f} USDT  ({len([r for r in income_rows if r['type']=='COMMISSION'])} fills)")
print(f"  Funding fees:   {fund_total:+.4f} USDT  ({len([r for r in income_rows if r['type']=='FUNDING_FEE'])} events)")
print(f"  ─────────────────────────────────")
print(f"  NET (48h):      {net_total:+.4f} USDT")

# ── Fee/PnL ratio
if pnl_total and pnl_total != 0:
    fee_ratio = abs(fee_total) / abs(pnl_total) * 100
    print(f"\n  Fee drag:       {fee_ratio:.1f}% of gross PnL eaten by commission")
if fund_total < 0:
    print(f"  ⚠ Funding drag:  paying {abs(fund_total):.4f} USDT in funding (wrong side?)")

# ── Per-symbol breakdown
print("\n  Per-symbol net (48h):")
sym_nets = {}
for sym, types in income_by_symbol.items():
    net = sum(types.values())
    sym_nets[sym] = net

for sym, net in sorted(sym_nets.items(), key=lambda x: x[1]):
    p  = income_by_symbol[sym].get("REALIZED_PNL", 0)
    c  = income_by_symbol[sym].get("COMMISSION", 0)
    f  = income_by_symbol[sym].get("FUNDING_FEE", 0)
    print(f"  {sym:<12}  net={net:+.4f}  pnl={p:+.4f}  fee={c:+.4f}  fund={f:+.4f}")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Binance testnet user trades (individual fills)
# ────────────────────────────────────────────────────────────────────────────
print("\n── 2. BINANCE TESTNET USER TRADES (last 48h sample) ──────────────────")
# Get all symbols that had income
symbols_traded = list(income_by_symbol.keys()) or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
all_trades = []
for sym in symbols_traded[:15]:  # cap at 15 symbols
    trades = binance_get("/fapi/v1/userTrades", {"symbol": sym, "startTime": START_MS, "limit": 200})
    if isinstance(trades, list):
        for t in trades:
            all_trades.append({
                "ts": ts_label(int(t.get("time", 0))),
                "symbol": t.get("symbol"),
                "side": t.get("side"),
                "positionSide": t.get("positionSide"),
                "price": float(t.get("price", 0)),
                "qty": float(t.get("qty", 0)),
                "realizedPnl": float(t.get("realizedPnl", 0)),
                "commission": float(t.get("commission", 0)),
                "maker": t.get("maker"),
                "buyer": t.get("buyer"),
            })

# Wins vs losses
wins   = [t for t in all_trades if t["realizedPnl"] > 0]
losses = [t for t in all_trades if t["realizedPnl"] < 0]
flat   = [t for t in all_trades if t["realizedPnl"] == 0]
total_fills = len(all_trades)
total_pnl_fills = sum(t["realizedPnl"] for t in all_trades)
total_fees_fills = sum(t["commission"] for t in all_trades)

print(f"  Total fills: {total_fills}  wins={len(wins)}  losses={len(losses)}  flat={len(flat)}")
print(f"  Gross PnL from fills: {total_pnl_fills:+.4f} USDT")
print(f"  Commission from fills: {total_fees_fills:+.4f} USDT")

if losses:
    avg_loss = sum(t["realizedPnl"] for t in losses) / len(losses)
    avg_win  = sum(t["realizedPnl"] for t in wins) / len(wins) if wins else 0
    print(f"\n  Avg win:  {avg_win:+.4f} USDT")
    print(f"  Avg loss: {avg_loss:+.4f} USDT")
    if avg_win and avg_loss:
        rr = abs(avg_win / avg_loss)
        print(f"  R:R ratio: {rr:.2f}  (need >1.0 to be profitable)")

# Taker/maker split
takers = [t for t in all_trades if not t["maker"]]
makers = [t for t in all_trades if t["maker"]]
taker_pct = len(takers) / total_fills * 100 if total_fills else 0
print(f"  Taker fills: {len(takers)} ({taker_pct:.0f}%)")
print(f"  Maker fills: {len(makers)}")
if takers:
    taker_fee = sum(t["commission"] for t in takers)
    print(f"  Taker commission total: {taker_fee:+.4f} USDT")

# Worst individual losses
print("\n  Worst 10 individual fills by realized PnL:")
for t in sorted(all_trades, key=lambda x: x["realizedPnl"])[:10]:
    print(f"  {t['ts']}  {t['symbol']:<12} {t['side']:<5} qty={t['qty']:.4f}  pnl={t['realizedPnl']:+.4f}  fee={t['commission']:+.4f}  maker={t['maker']}")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Redis trade.closed stream
# ────────────────────────────────────────────────────────────────────────────
print("\n── 3. REDIS trade.closed (last 48h) ───────────────────────────────────")
closed_events = stream_range("quantum:stream:trade.closed", START_MS, count=500)

pnl_vals   = []
r_vals     = []
reason_counts = defaultdict(int)
source_counts = defaultdict(int)
symbol_pnl = defaultdict(float)

for ev in closed_events:
    try:
        pnl = float(ev.get("pnl_usd", 0))
        r_net = float(ev.get("R_net", 0))
        reason = ev.get("reason", "unknown")
        source = ev.get("source", "unknown")
        symbol = ev.get("symbol", "?")
        pnl_vals.append(pnl)
        r_vals.append(r_net)
        reason_counts[reason] += 1
        source_counts[source] += 1
        symbol_pnl[symbol] += pnl
    except Exception:
        pass

n_events = len(closed_events)
n_wins_r  = sum(1 for p in pnl_vals if p > 0)
n_loss_r  = sum(1 for p in pnl_vals if p < 0)
total_redis_pnl = sum(pnl_vals)

print(f"  Events in 48h: {n_events}")
print(f"  Total PnL (Redis): {total_redis_pnl:+.4f} USDT")
print(f"  Wins: {sum(1 for p in pnl_vals if p > 0)}  Losses: {sum(1 for p in pnl_vals if p < 0)}")
if r_vals:
    print(f"  Avg R_net: {sum(r_vals)/len(r_vals):.3f}")

print("\n  Exit reasons:")
for reason, cnt in sorted(reason_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"  {cnt:4d}x  {reason}")

print("\n  Sources:")
for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {cnt:4d}x  {src}")

print("\n  PnL by symbol (Redis):")
for sym, pnl in sorted(symbol_pnl.items(), key=lambda x: x[1]):
    print(f"  {sym:<12} {pnl:+.4f} USDT")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Redis proposals / harvest signals
# ────────────────────────────────────────────────────────────────────────────
print("\n── 4. HARVEST PROPOSALS (entry signal quality, 48h) ──────────────────")
proposals = stream_range("quantum:stream:harvest.proposal", START_MS, count=500)
prop_decisions = defaultdict(int)
prop_confidence = []
prop_by_symbol = defaultdict(int)

for p in proposals:
    dec = p.get("decision", p.get("action", "unknown"))
    prop_decisions[dec] += 1
    try:
        conf = float(p.get("confidence", p.get("score", 0)))
        if conf > 0:
            prop_confidence.append(conf)
    except Exception:
        pass
    sym = p.get("symbol", "?")
    prop_by_symbol[sym] += 1

print(f"  Total proposals: {len(proposals)}")
print(f"  Decisions: {dict(prop_decisions)}")
if prop_confidence:
    avg_conf = sum(prop_confidence) / len(prop_confidence)
    low_conf = sum(1 for c in prop_confidence if c < 0.55)
    print(f"  Avg confidence: {avg_conf:.3f}  low-conf (<0.55): {low_conf}/{len(prop_confidence)}")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Open positions and unrealized PnL
# ────────────────────────────────────────────────────────────────────────────
print("\n── 5. CURRENT OPEN POSITIONS ─────────────────────────────────────────")
open_pos = binance_get("/fapi/v2/positionRisk", {"recvWindow": 5000})
if isinstance(open_pos, list):
    active = [p for p in open_pos if float(p.get("positionAmt", 0)) != 0]
    total_unrealized = sum(float(p.get("unRealizedProfit", 0)) for p in active)
    print(f"  Open positions: {len(active)}")
    print(f"  Total unrealized: {total_unrealized:+.4f} USDT")
    for p in sorted(active, key=lambda x: float(x.get("unRealizedProfit", 0))):
        sym   = p.get("symbol")
        side  = "LONG" if float(p.get("positionAmt", 0)) > 0 else "SHORT"
        amt   = float(p.get("positionAmt", 0))
        upnl  = float(p.get("unRealizedProfit", 0))
        entry = float(p.get("entryPrice", 0))
        liq   = float(p.get("liquidationPrice", 0))
        lev   = p.get("leverage", "?")
        print(f"  {sym:<12} {side:<5} amt={amt:.4f}  entry={entry:.4f}  upnl={upnl:+.4f}  liq={liq:.2f}  lev={lev}x")
else:
    print(f"  ERROR: {open_pos}")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — Account balance
# ────────────────────────────────────────────────────────────────────────────
print("\n── 6. ACCOUNT BALANCE ────────────────────────────────────────────────")
balances = binance_get("/fapi/v2/balance", {})
if isinstance(balances, list):
    for b in balances:
        if float(b.get("balance", 0)) > 0:
            print(f"  {b['asset']:<10}  balance={float(b['balance']):.4f}  available={float(b.get('availableBalance', 0)):.4f}  unrealized={float(b.get('crossUnPnl', 0)):+.4f}")
else:
    print(f"  ERROR: {balances}")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — Fee analysis: churn rate
# ────────────────────────────────────────────────────────────────────────────
print("\n── 7. CHURN / FEE EFFICIENCY ─────────────────────────────────────────")
total_volume = sum(t["price"] * t["qty"] for t in all_trades)
if total_volume > 0 and fee_total != 0:
    effective_fee_rate = abs(total_fees_fills) / total_volume * 100
    print(f"  Total notional traded: {total_volume:.2f} USDT")
    print(f"  Total commission: {total_fees_fills:+.4f} USDT")
    print(f"  Effective fee rate: {effective_fee_rate:.4f}% per fill")
    trades_per_hour = total_fills / 48
    print(f"  Trade frequency: {trades_per_hour:.1f} fills/hour")
    daily_fee = abs(total_fees_fills) / 2
    print(f"  Daily fee burn: ~{daily_fee:.4f} USDT/day")
    fees_needed_to_break_even = abs(total_fees_fills)
    print(f"  Need >{fees_needed_to_break_even:.4f} USDT gross PnL just to break even on fees")

# ── Rapid open/close detection
if len(all_trades) > 1:
    sorted_trades = sorted(all_trades, key=lambda x: x["ts"])
    rapid = 0
    for i in range(1, len(sorted_trades)):
        if sorted_trades[i]["symbol"] == sorted_trades[i-1]["symbol"]:
            rapid += 1
    print(f"\n  Same-symbol consecutive fills: {rapid} (potential churn indicator)")

# ────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — Summary / Root cause
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  ROOT CAUSE SUMMARY")
print("=" * 72)

causes = []

if fee_total and pnl_total:
    if abs(fee_total) > abs(pnl_total) * 0.3:
        causes.append(f"FEES eating >{abs(fee_total)/abs(pnl_total)*100:.0f}% of gross PnL — churn suspected")

if fund_total < -1:
    causes.append(f"FUNDING FEES: paying {abs(fund_total):.4f} USDT — likely holding wrong-side positions overnight")

if losses and wins:
    rr = abs(avg_win / avg_loss) if avg_loss else 0
    if rr < 1.0:
        causes.append(f"BAD R:R = {rr:.2f} — losing more per loss than winning per win")
    win_rate = len(wins) / (len(wins) + len(losses))
    if win_rate < 0.5:
        causes.append(f"LOW WIN RATE = {win_rate:.0%} — model is wrong more than half the time")

if total_fills > 0 and abs(total_fees_fills) > abs(total_pnl_fills):
    causes.append(f"FEE DESTRUCTION: fees ({total_fees_fills:+.4f}) exceed gross PnL ({total_pnl_fills:+.4f})")

if prop_confidence and sum(1 for c in prop_confidence if c < 0.55) / len(prop_confidence) > 0.4:
    causes.append(f"WEAK SIGNALS: {sum(1 for c in prop_confidence if c < 0.55)/len(prop_confidence):.0%} of proposals below 55% confidence")

if isinstance(open_pos, list) and active:
    negative_pos = [p for p in active if float(p.get("unRealizedProfit", 0)) < -2]
    if negative_pos:
        causes.append(f"STUCK LOSERS: {len(negative_pos)} open positions with unrealized loss > -2 USDT not being cut")

for i, cause in enumerate(causes, 1):
    print(f"  {i}. {cause}")

if not causes:
    print("  No dominant cause detected — need more data or API keys not configured")

print()
