#!/usr/bin/env python3
"""Binance testnet realized PnL analysis — root cause of capital bleed"""
import requests, time, hmac, hashlib, json
from datetime import datetime, timezone
from collections import defaultdict

API_KEY    = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
API_SECRET = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'
BASE       = 'https://testnet.binancefuture.com'

def sign_params(params):
    qs  = '&'.join(f'{k}={v}' for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return qs + '&signature=' + sig

def get(path, params=None):
    if params is None:
        params = {}
    params['timestamp']  = int(time.time() * 1000)
    params['recvWindow'] = 10000
    qs = sign_params(params)
    r  = requests.get(BASE + path + '?' + qs,
                      headers={'X-MBX-APIKEY': API_KEY}, timeout=15)
    return r.json()

SEP = '=' * 70

# ── 1. Account snapshot ────────────────────────────────────────────────────
acc = get('/fapi/v2/account')
print(SEP)
print('ACCOUNT SNAPSHOT')
print(SEP)
wallet = unrealized = margin_used = 0.0
for a in acc.get('assets', []):
    if a['asset'] == 'USDT':
        wallet       = float(a['walletBalance'])
        unrealized   = float(a['unrealizedProfit'])
        margin_used  = float(a['initialMargin'])
        maint_margin = float(a.get('maintMargin', 0))
        avail        = float(a.get('availableBalance', 0))
        print(f'Wallet balance   : {wallet:>12.4f} USDT')
        print(f'Unrealized PnL   : {unrealized:>+12.4f} USDT')
        print(f'Equity           : {wallet+unrealized:>12.4f} USDT')
        print(f'Initial margin   : {margin_used:>12.4f} USDT')
        print(f'Maint margin     : {maint_margin:>12.4f} USDT')
        print(f'Available balance: {avail:>12.4f} USDT')

# ── 2. Open positions ──────────────────────────────────────────────────────
print(f'\n{SEP}')
print('OPEN POSITIONS')
print(SEP)
open_pos = []
for p in acc.get('positions', []):
    amt = float(p['positionAmt'])
    if abs(amt) > 0.0001:
        open_pos.append(p)
        entry    = float(p['entryPrice'])
        upnl     = float(p['unrealizedProfit'])
        lev      = p['leverage']
        notional = abs(float(p['notional']))
        side     = 'LONG' if amt > 0 else 'SHORT'
        print(f"  {p['symbol']:12s} {side:5s}  amt={amt:+.4f}  entry={entry:.4f}  "
              f"uPnL={upnl:+.4f}  lev={lev}x  notional={notional:.2f}")
if not open_pos:
    print('  (no open positions)')

# ── 3. Realized PnL last 14 days ───────────────────────────────────────────
print(f'\n{SEP}')
print('REALIZED PnL — LAST 14 DAYS (chronological)')
print(SEP)
start_ms = int((time.time() - 14 * 86400) * 1000)
income = get('/fapi/v1/income', {'incomeType': 'REALIZED_PNL',
                                  'startTime': start_ms, 'limit': 1000})
total_rpnl = 0.0
by_symbol  = defaultdict(float)
trade_rows = []
if isinstance(income, list):
    for i in income:
        sym  = i['symbol']
        val  = float(i['income'])
        ts   = datetime.fromtimestamp(i['time']/1000, tz=timezone.utc).strftime('%m-%d %H:%M')
        total_rpnl     += val
        by_symbol[sym] += val
        trade_rows.append((i['time'], sym, val, ts))
    trade_rows.sort(key=lambda x: x[0])
    print(f'Count            : {len(trade_rows)} fills')
    print(f'Total realized   : {total_rpnl:+.4f} USDT')
    print(f'\nBy symbol (worst first):')
    for sym, v in sorted(by_symbol.items(), key=lambda x: x[1]):
        count = sum(1 for r in trade_rows if r[1] == sym)
        wins  = sum(1 for r in trade_rows if r[1] == sym and r[2] > 0)
        print(f'  {sym:12s}: {v:+.8f} USDT  ({wins}/{count} wins)')
    print(f'\nLast 50 realized fills:')
    for row in trade_rows[-50:]:
        marker = '✓' if row[2] >= 0 else '✗'
        print(f'  {marker} {row[3]}  {row[1]:12s}  {row[2]:+.6f}')
else:
    print(f'  ERROR: {income}')

# ── 4. Commission (fees) ───────────────────────────────────────────────────
print(f'\n{SEP}')
print('COMMISSION (FEES) — LAST 14 DAYS')
print(SEP)
comm = get('/fapi/v1/income', {'incomeType': 'COMMISSION',
                                'startTime': start_ms, 'limit': 1000})
total_fee = 0.0
fee_by_sym = defaultdict(float)
fee_count  = defaultdict(int)
if isinstance(comm, list):
    for c in comm:
        total_fee         += float(c['income'])
        fee_by_sym[c['symbol']] += float(c['income'])
        fee_count[c['symbol']]  += 1
    print(f'Total fees       : {total_fee:+.4f} USDT ({len(comm)} events)')
    print(f'\nBy symbol (worst first):')
    for sym, v in sorted(fee_by_sym.items(), key=lambda x: x[1]):
        print(f'  {sym:12s}: {v:+.6f} USDT  ({fee_count[sym]} fills)')

# ── 5. Funding fees ────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('FUNDING FEES — LAST 14 DAYS')
print(SEP)
fund = get('/fapi/v1/income', {'incomeType': 'FUNDING_FEE',
                                'startTime': start_ms, 'limit': 1000})
total_fund = 0.0
fund_by_sym = defaultdict(float)
if isinstance(fund, list):
    for f in fund:
        total_fund          += float(f['income'])
        fund_by_sym[f['symbol']] += float(f['income'])
    print(f'Total funding    : {total_fund:+.4f} USDT ({len(fund)} events)')
    print(f'\nBy symbol:')
    for sym, v in sorted(fund_by_sym.items(), key=lambda x: x[1]):
        print(f'  {sym:12s}: {v:+.6f} USDT')

# ── 6. Trade-level win/loss stats ──────────────────────────────────────────
print(f'\n{SEP}')
print('WIN / LOSS BREAKDOWN')
print(SEP)
if trade_rows:
    wins   = [r for r in trade_rows if r[2] > 0]
    losses = [r for r in trade_rows if r[2] < 0]
    break_e = [r for r in trade_rows if r[2] == 0]
    total_w = sum(r[2] for r in wins)
    total_l = sum(r[2] for r in losses)
    avg_w   = total_w / len(wins)   if wins   else 0
    avg_l   = total_l / len(losses) if losses else 0
    win_rate = len(wins) / len(trade_rows) * 100 if trade_rows else 0
    pf = abs(total_w / total_l) if total_l != 0 else float('inf')
    print(f'Total trades     : {len(trade_rows)}')
    print(f'Wins             : {len(wins)} ({win_rate:.1f}%)')
    print(f'Losses           : {len(losses)}')
    print(f'Break-even       : {len(break_e)}')
    print(f'Total gross win  : {total_w:+.4f} USDT')
    print(f'Total gross loss : {total_l:+.4f} USDT')
    print(f'Net PnL          : {total_rpnl:+.4f} USDT')
    print(f'Total fees       : {total_fee:+.4f} USDT')
    print(f'Total funding    : {total_fund:+.4f} USDT')
    print(f'Complete net     : {total_rpnl+total_fee+total_fund:+.4f} USDT')
    print(f'Avg win          : {avg_w:+.6f} USDT')
    print(f'Avg loss         : {avg_l:+.6f} USDT')
    print(f'Win/loss ratio   : {abs(avg_w/avg_l):.3f}' if avg_l != 0 else 'N/A')
    print(f'Profit factor    : {pf:.3f}')

    # Streak analysis
    streak_cur = 0
    streak_type = None
    max_loss_streak = 0
    cur_loss_streak = 0
    for r in trade_rows:
        if r[2] < 0:
            cur_loss_streak += 1
            max_loss_streak = max(max_loss_streak, cur_loss_streak)
        else:
            cur_loss_streak = 0
    print(f'Max loss streak  : {max_loss_streak}')

# ── 7. Recent order history (fills) for size analysis ─────────────────────
print(f'\n{SEP}')
print('RECENT ORDER FILLS — SIZE & SLIPPAGE ANALYSIS')
print(SEP)
orders = get('/fapi/v1/userTrades', {'startTime': start_ms, 'limit': 500})
if isinstance(orders, list):
    print(f'Total fills (14d): {len(orders)}')
    # Group by symbol, analyze notional sizes
    notionals = defaultdict(list)
    for o in orders:
        n = float(o['qty']) * float(o['price'])
        notionals[o['symbol']].append(n)
    print(f'\nAvg notional per fill by symbol:')
    for sym in sorted(notionals.keys()):
        vals = notionals[sym]
        avg  = sum(vals)/len(vals)
        print(f'  {sym:12s}: avg={avg:.2f} USDT  n={len(vals)}  min={min(vals):.2f}  max={max(vals):.2f}')
else:
    print(f'  ERROR: {orders}')

print(f'\n{SEP}')
print('DONE')
print(SEP)
