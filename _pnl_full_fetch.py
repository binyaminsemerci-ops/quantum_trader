#!/usr/bin/env python3
"""
Full paginated PnL fetch — last 2 days, all income types.
Bypasses the 1000-row API limit by paginating via endTime stepping.
"""
import requests, hmac, hashlib, time, os, json
from urllib.parse import urlencode
from datetime import datetime, timezone
from collections import defaultdict

# Load env
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

def ts(ms):
    return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

NOW_MS      = int(time.time() * 1000)
START_MS    = NOW_MS - 2 * 24 * 3600 * 1000   # 48h back
CHUNK_H     = 4                                 # paginate in 4-hour windows

SEP  = "═" * 68
SEP2 = "─" * 68

# ── Paginated income fetch ────────────────────────────────────────────────────
def fetch_income_paginated(inc_type):
    all_rows = []
    end = NOW_MS
    start = START_MS
    step = CHUNK_H * 3600 * 1000
    cursor = start
    while cursor < NOW_MS:
        window_end = min(cursor + step, NOW_MS)
        rows = get("/fapi/v1/income", {
            "incomeType": inc_type,
            "startTime":  cursor,
            "endTime":    window_end,
            "limit":      1000
        })
        if isinstance(rows, list):
            all_rows.extend(rows)
        cursor = window_end + 1
        time.sleep(0.1)
    # deduplicate by tranId
    seen = set()
    deduped = []
    for r in all_rows:
        tid = r.get("tranId", r.get("tradeId", id(r)))
        if tid not in seen:
            seen.add(tid)
            deduped.append(r)
    return deduped

print(f"\n{SEP}")
print("  Fetching all income — paginated (4h windows) ...")
print(f"  Window: {ts(START_MS)} → {ts(NOW_MS)} UTC")
print(SEP)

realized_rows   = fetch_income_paginated("REALIZED_PNL")
commission_rows = fetch_income_paginated("COMMISSION")
funding_rows    = fetch_income_paginated("FUNDING_FEE")

print(f"  Rows fetched:")
print(f"    REALIZED_PNL : {len(realized_rows)}")
print(f"    COMMISSION   : {len(commission_rows)}")
print(f"    FUNDING_FEE  : {len(funding_rows)}")

# ── Totals ────────────────────────────────────────────────────────────────────
total_r = sum(float(x.get("income", 0)) for x in realized_rows)
total_c = sum(float(x.get("income", 0)) for x in commission_rows)
total_f = sum(float(x.get("income", 0)) for x in funding_rows)
net     = total_r + total_c + total_f

print(f"\n{SEP}")
print("  FULL 2-DAY PnL SUMMARY")
print(SEP)
print(f"  Realized PnL (fills)  : {total_r:>+13.4f} USDT")
print(f"  Commissions (fees)    : {total_c:>+13.4f} USDT")
print(f"  Funding fees          : {total_f:>+13.4f} USDT")
print(f"  {SEP2}")
print(f"  NET TOTAL             : {net:>+13.4f} USDT")

# ── Per-hour breakdown ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  HOURLY PnL (realized only)")
print(SEP)
hourly = defaultdict(float)
for row in realized_rows:
    h = datetime.fromtimestamp(int(row["time"])/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:00")
    hourly[h] += float(row.get("income", 0))

running = 0.0
for h in sorted(hourly):
    v = hourly[h]
    running += v
    bar = ("▓" * int(abs(v)/3)) if abs(v) > 1 else ""
    sign_char = "+" if v >= 0 else "-"
    print(f"  {h}  {v:>+9.2f} USDT  running={running:>+9.2f}  {bar}")

# ── Per-symbol breakdown ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PER-SYMBOL PnL  (realized + fees combined)")
print(SEP)
sym_r = defaultdict(float)
sym_c = defaultdict(float)
sym_f = defaultdict(float)
for row in realized_rows:
    sym_r[row.get("symbol","?")] += float(row.get("income",0))
for row in commission_rows:
    sym_c[row.get("symbol","?")] += float(row.get("income",0))
for row in funding_rows:
    sym_f[row.get("symbol","?")] += float(row.get("income",0))

all_syms = sorted(set(list(sym_r) + list(sym_c) + list(sym_f)))
sym_net  = {s: sym_r.get(s,0) + sym_c.get(s,0) + sym_f.get(s,0) for s in all_syms}
sorted_syms = sorted(all_syms, key=lambda s: sym_net[s])

winners = [s for s in sorted_syms if sym_net[s] > 0]
losers  = [s for s in sorted_syms if sym_net[s] < 0]

print(f"\n  TOP LOSERS ({len(losers)} symbols in loss):")
for s in sorted_syms[:15]:
    r_ = sym_r.get(s,0); c_ = sym_c.get(s,0); f_ = sym_f.get(s,0)
    n_ = sym_net[s]
    print(f"  🔴 {s:<22}  rPnL={r_:>+9.4f}  fees={c_:>+8.4f}  fund={f_:>+7.4f}  NET={n_:>+9.4f}")

print(f"\n  TOP WINNERS ({len(winners)} symbols in profit):")
for s in sorted(winners, key=lambda s: -sym_net[s])[:10]:
    r_ = sym_r.get(s,0); c_ = sym_c.get(s,0)
    n_ = sym_net[s]
    print(f"  ✅ {s:<22}  rPnL={r_:>+9.4f}  fees={c_:>+8.4f}  NET={n_:>+9.4f}")

# ── Fee analysis ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  FEE ANALYSIS")
print(SEP)
fee_by_sym = defaultdict(float)
for row in commission_rows:
    fee_by_sym[row.get("symbol","?")] += abs(float(row.get("income",0)))
top_fee_syms = sorted(fee_by_sym, key=lambda s: -fee_by_sym[s])[:10]
print(f"  Total fees paid: {abs(total_c):.4f} USDT over {len(commission_rows)} transactions")
print(f"  Average per transaction: {abs(total_c)/max(len(commission_rows),1):.4f} USDT")
print(f"\n  Highest fee-generating symbols:")
for s in top_fee_syms:
    print(f"    {s:<22}  {fee_by_sym[s]:.4f} USDT")

# ── Trade count per symbol ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TRADE COUNT PER SYMBOL (by fill count from income stream)")
print(SEP)
trade_cnt = defaultdict(int)
for row in realized_rows:
    trade_cnt[row.get("symbol","?")] += 1
top_traded = sorted(trade_cnt, key=lambda s: -trade_cnt[s])[:15]
print(f"  Total fill events: {len(realized_rows)}")
for s in top_traded:
    fee_drag = fee_by_sym.get(s, 0)
    pnl_per  = sym_r.get(s,0) / max(trade_cnt[s],1)
    print(f"  {s:<22}  fills={trade_cnt[s]:<5}  rPnL/fill={pnl_per:>+7.4f}  fee_total={fee_drag:.4f}")

print(f"\n{SEP}")
print("  CONCLUSION")
print(SEP)
total_fills = len(realized_rows)
fee_to_pnl  = abs(total_c) / max(abs(total_r), 0.01)
print(f"  Net loss          : {net:>+.2f} USDT")
print(f"  Fee-to-PnL ratio  : {fee_to_pnl:.2%}  (fees as % of gross |PnL|)")
print(f"  Total fill events : {total_fills}")
print(f"  Avg PnL per fill  : {total_r/max(total_fills,1):>+.4f} USDT")
print(f"  Symbols traded    : {len(all_syms)}")
print(SEP)
