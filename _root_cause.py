#!/usr/bin/env python3
"""
Root-cause deep dive — ikke symboler, men atferd.
Analyserer trade-timing, side-switching, hold-tid, PnL-distribusjon,
fee-ratio, reversal-rate og churn-indikator.
"""
import requests, hmac, hashlib, time, os, json
from urllib.parse import urlencode
from datetime import datetime, timezone
from collections import defaultdict

for line in open("/etc/quantum/testnet.env"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k = k.strip()
    if k:
        os.environ[k] = v.strip()

KEY = os.environ.get("BINANCE_TESTNET_API_KEY", "")
SEC = os.environ.get("BINANCE_TESTNET_API_SECRET",
      os.environ.get("BINANCE_TESTNET_SECRET_KEY", ""))
BASE = "https://testnet.binancefuture.com"

def sign(params):
    qs = urlencode(params)
    params["signature"] = hmac.new(SEC.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return params

def get(path, params=None):
    if params is None: params = {}
    params["timestamp"] = int(time.time() * 1000)
    sign(params)
    r = requests.get(BASE + path, params=params,
                     headers={"X-MBX-APIKEY": KEY}, timeout=15)
    return r.json()

NOW_MS   = int(time.time() * 1000)
START_MS = NOW_MS - 2 * 24 * 3600 * 1000
STEP     = 4 * 3600 * 1000

SEP  = "═" * 70
SEP2 = "─" * 70

def ts(ms):
    return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ── Fetch all trades (userTrades, paginated by symbol) ────────────────────────
print("Fetching userTrades per symbol...")
target_syms = [
    "ADAUSDT","1000CHEEMSUSDT","AAVEUSDT","1MBABYDOGEUSDT",
    "ALICEUSDT","1INCHUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","LINKUSDT",
    "SOLUSDT","ETHUSDT","BTCUSDT","0GUSDT","ALTUSDT","DOTUSDT"
]
all_trades = []  # list of trade dicts enriched with symbol
for sym in target_syms:
    rows = get("/fapi/v1/userTrades", {"symbol": sym, "startTime": START_MS, "limit": 1000})
    if isinstance(rows, list):
        for r in rows:
            r["symbol"] = sym
        all_trades.extend(rows)
        print(f"  {sym}: {len(rows)} fills")
    time.sleep(0.08)

all_trades.sort(key=lambda x: int(x["time"]))
print(f"  Total: {len(all_trades)} fills across {len(target_syms)} symbols\n")

# ── Per-symbol order reconstruction ──────────────────────────────────────────
# Group fills by orderId to reconstruct orders
print(SEP)
print("  ORDER-LEVEL ANALYSIS")
print(SEP)

sym_orders = defaultdict(list)
for t in all_trades:
    sym_orders[t["symbol"]].append(t)

# For each symbol, reconstruct open/close cycles
print("\n  Hold-time analysis (per symbol, sampled):")
print(f"  {'Symbol':<22} {'Orders':>6}  {'Avg hold':>10}  {'Min hold':>9}  {'Max hold':>9}  {'<1min':>6}  {'<5min':>6}")
print(f"  {SEP2}")

global_hold_times = []
global_lt1min = 0
global_lt5min = 0
global_orders = 0

for sym, fills in sorted(sym_orders.items(), key=lambda x: -len(x[1])):
    if len(fills) < 4:
        continue
    # Group by orderId
    orders_by_id = defaultdict(list)
    for f in fills:
        orders_by_id[str(f["orderId"])].append(f)

    order_times = sorted([min(int(f["time"]) for f in v)
                          for v in orders_by_id.values()])

    # Approximate hold time = time between consecutive order executions
    hold_times = []
    for i in range(1, len(order_times)):
        h = (order_times[i] - order_times[i-1]) / 1000  # seconds
        if h > 0:
            hold_times.append(h)

    if not hold_times:
        continue

    avg_h = sum(hold_times) / len(hold_times)
    min_h = min(hold_times)
    max_h = max(hold_times)
    lt1   = sum(1 for h in hold_times if h < 60)
    lt5   = sum(1 for h in hold_times if h < 300)
    n_ord = len(order_times)

    global_hold_times.extend(hold_times)
    global_lt1min += lt1
    global_lt5min += lt5
    global_orders += n_ord

    def fmt_s(s):
        if s < 60: return f"{s:.0f}s"
        if s < 3600: return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"

    print(f"  {sym:<22} {n_ord:>6}  {fmt_s(avg_h):>10}  {fmt_s(min_h):>9}  {fmt_s(max_h):>9}  {lt1:>6}  {lt5:>6}")

if global_hold_times:
    g_avg = sum(global_hold_times)/len(global_hold_times)
    g_min = min(global_hold_times)
    pct_lt1 = global_lt1min / max(global_orders, 1) * 100
    pct_lt5 = global_lt5min / max(global_orders, 1) * 100
    print(f"  {SEP2}")
    print(f"  {'TOTAL/AVG':<22} {global_orders:>6}  {fmt_s(g_avg):>10}  {fmt_s(g_min):>9}  {'':>9}  {pct_lt1:.0f}%    {pct_lt5:.0f}%")

# ── Side-flip analysis ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  SIDE-FLIP ANALYSIS  (how often does system reverse direction?)")
print(SEP)

for sym, fills in sorted(sym_orders.items(), key=lambda x: -len(x[1]))[:6]:
    if len(fills) < 6:
        continue
    # Sequence of sides by time
    sides = [f["side"] for f in sorted(fills, key=lambda x: int(x["time"]))]
    flips = sum(1 for i in range(1, len(sides)) if sides[i] != sides[i-1])
    flip_rate = flips / max(len(sides)-1, 1) * 100
    print(f"  {sym:<22}  fills={len(sides):<5}  flips={flips:<5}  flip_rate={flip_rate:.1f}%")

# ── PnL distribution ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PnL DISTRIBUTION PER FILL")
print(SEP)

rpnl_vals = [float(t.get("realizedPnl", 0)) for t in all_trades]
nonzero = [v for v in rpnl_vals if abs(v) > 0.0001]

if nonzero:
    pos = [v for v in nonzero if v > 0]
    neg = [v for v in nonzero if v < 0]
    zero = len(rpnl_vals) - len(nonzero)

    print(f"  Total fills          : {len(rpnl_vals)}")
    print(f"  Zero PnL fills       : {zero}  ({zero/len(rpnl_vals)*100:.1f}%)  ← these are OPENING fills")
    print(f"  Winning fills        : {len(pos)}  ({len(pos)/len(nonzero)*100:.1f}%)")
    print(f"  Losing fills         : {len(neg)}  ({len(neg)/len(nonzero)*100:.1f}%)")
    print(f"")
    if pos:
        print(f"  Avg win  per fill    : +{sum(pos)/len(pos):.4f} USDT")
        print(f"  Max win  per fill    : +{max(pos):.4f} USDT")
    if neg:
        print(f"  Avg loss per fill    : {sum(neg)/len(neg):.4f} USDT")
        print(f"  Max loss per fill    : {min(neg):.4f} USDT")
    if pos and neg:
        win_rate = len(pos)/len(nonzero)
        avg_win  = sum(pos)/len(pos)
        avg_loss = abs(sum(neg)/len(neg))
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
        rr_ratio   = avg_win / avg_loss
        print(f"")
        print(f"  Win rate             : {win_rate*100:.1f}%")
        print(f"  Avg win / Avg loss   : {rr_ratio:.3f}  (needs >1.0 to be profitable at this win rate)")
        print(f"  Expectancy per fill  : {expectancy:+.4f} USDT")
        print(f"  {'✅ Positive edge' if expectancy > 0 else '🔴 NEGATIVE EDGE — system loses money on every trade in expectation'}")

# ── Fee as % of gross ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  FEE BURDEN ANALYSIS")
print(SEP)

total_comm  = sum(float(t.get("commission", 0)) for t in all_trades)
gross_vol   = sum(float(t.get("quoteQty", 0)) for t in all_trades)
total_rpnl  = sum(float(t.get("realizedPnl", 0)) for t in all_trades)

print(f"  Gross trading volume : {gross_vol:>14.2f} USDT")
print(f"  Total fees paid      : {total_comm:>+14.4f} USDT")
print(f"  Total realized PnL   : {total_rpnl:>+14.4f} USDT")
if gross_vol > 0:
    print(f"  Effective fee rate   : {total_comm/gross_vol*100:.4f}%  per notional")
if abs(total_rpnl) > 0:
    print(f"  Fee / |gross PnL|    : {total_comm/abs(total_rpnl)*100:.1f}%")
print(f"  Break-even req.      : system needs >{total_comm:.2f} USDT gross PnL just to cover fees")

# ── Timing: are losses concentrated? ─────────────────────────────────────────
print(f"\n{SEP}")
print("  ARE LOSSES RANDOM OR SYSTEMATIC?")
print(f"  (Autocorrelation check — do losses come in streaks?)")
print(SEP)

close_fills = [t for t in all_trades if abs(float(t.get("realizedPnl",0))) > 0.0001]
close_fills.sort(key=lambda x: int(x["time"]))
if len(close_fills) > 10:
    outcomes = [1 if float(t["realizedPnl"]) > 0 else 0 for t in close_fills]
    # Count runs
    runs = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] != outcomes[i-1]:
            runs += 1
    expected_runs = 1 + 2*len([o for o in outcomes if o==1])*len([o for o in outcomes if o==0])/len(outcomes)
    n_wins  = sum(outcomes)
    n_loss  = len(outcomes) - n_wins
    print(f"  Closing fills analyzed : {len(close_fills)}")
    print(f"  Wins: {n_wins}  Losses: {n_loss}  Run count: {runs}")
    print(f"  Expected runs (random) : {expected_runs:.0f}")
    if runs < expected_runs * 0.85:
        print(f"  ⚠️  FEWER runs than random → losses come in STREAKS (momentum following losses)")
    elif runs > expected_runs * 1.15:
        print(f"  ⚠️  MORE runs than random → wins and losses alternate (mean-reversion chop)")
    else:
        print(f"  Results are roughly random in sequence")

    # Loss streak analysis
    max_streak = cur_streak = 0
    for o in outcomes:
        if o == 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    print(f"  Longest losing streak  : {max_streak} consecutive losing fills")

    # PnL per hour for closing fills
    print(f"\n  PnL/hour from closing fills only:")
    hourly_close = defaultdict(lambda: [0,0,0])  # [pnl, wins, losses]
    for t in close_fills:
        h = datetime.fromtimestamp(int(t["time"])/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:00")
        pnl = float(t["realizedPnl"])
        hourly_close[h][0] += pnl
        if pnl > 0: hourly_close[h][1] += 1
        else:       hourly_close[h][2] += 1
    pos_hours = sum(1 for v in hourly_close.values() if v[0] > 0)
    neg_hours = sum(1 for v in hourly_close.values() if v[0] < 0)
    print(f"  Profitable hours: {pos_hours}  /  Losing hours: {neg_hours}")
    pct_green = pos_hours / max(pos_hours+neg_hours,1) * 100
    print(f"  % hours profitable: {pct_green:.1f}%")

# ── Root cause summary ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ROOT CAUSE DIAGNOSIS")
print(SEP)

avg_hold_sec = sum(global_hold_times)/max(len(global_hold_times),1)
print(f"""
  1. STRATEGY EDGE:
     Avg fill expectancy = {total_rpnl/max(len(all_trades),1):+.4f} USDT/fill
     Win rate on closing fills = {len(pos)/max(len(nonzero),1)*100:.1f}%
     Risk/reward ratio = {(sum(pos)/max(len(pos),1)) / max(abs(sum(neg)/max(len(neg),1)),0.0001):.3f}

  2. HOLD TIME:
     Avg time between orders = {fmt_s(avg_hold_sec)}
     Orders held <1 min = {pct_lt1:.0f}%  |  <5 min = {pct_lt5:.0f}%

  3. FEE DRAG:
     Total fees = {total_comm:.2f} USDT on {gross_vol:.0f} USDT volume
     Fee rate = {total_comm/max(gross_vol,1)*100:.4f}% per notional

  4. VOLUME vs EDGE:
     System generates enormous volume but has no positive expectancy.
     The strategy is optimized for ACTIVITY not PROFIT.
""")
print(SEP)
