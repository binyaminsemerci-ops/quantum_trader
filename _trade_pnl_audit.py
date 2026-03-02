#!/usr/bin/env python3
"""
Trade & PnL Audit — Binance Testnet vs Internal System (Redis)
Fetches last 2 days from Binance and compares with Redis ledger/positions.
"""
import os, sys, json, time, hmac, hashlib, redis as redis_lib
from urllib.parse import urlencode
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
    import requests

# ── Load env ──────────────────────────────────────────────────────────────────
env_file = '/etc/quantum/testnet.env'
API_KEY = SECRET = ''
for line in open(env_file):
    line = line.strip()
    if line.startswith('#') or '=' not in line:
        continue
    k, _, v = line.partition('=')
    k, v = k.strip(), v.strip()
    os.environ[k] = v
    if k == 'BINANCE_TESTNET_API_KEY':    API_KEY = v
    if k == 'BINANCE_TESTNET_API_SECRET': SECRET  = v
    if k == 'BINANCE_TESTNET_SECRET_KEY' and not SECRET: SECRET = v

USE_TESTNET = os.environ.get('BINANCE_TESTNET', 'true').lower() in ('true', '1', 'yes')
BASE_URL = 'https://testnet.binancefuture.com' if USE_TESTNET else 'https://fapi.binance.com'

SEP  = '═' * 72
SEP2 = '─' * 72

print(f'Mode    : {"TESTNET" if USE_TESTNET else "LIVE"}')
print(f'Endpoint: {BASE_URL}')
print(f'Key OK  : {bool(API_KEY and len(API_KEY) > 10)}')
print(f'Secret  : {bool(SECRET and len(SECRET) > 10)}')


# ── API helpers ───────────────────────────────────────────────────────────────
def sign(params: dict) -> str:
    qs = urlencode(params)
    return hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()

def api(path, params=None, method='GET'):
    if params is None:
        params = {}
    params['timestamp'] = int(time.time() * 1000)
    params['signature'] = sign(params)
    try:
        r = requests.get(BASE_URL + path, params=params,
                         headers={'X-MBX-APIKEY': API_KEY}, timeout=12)
        return r.json()
    except Exception as e:
        return {'error': str(e)}

def ts_str(ts_ms) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(ts_ms)


NOW_MS          = int(time.time() * 1000)
TWO_DAYS_MS     = NOW_MS - 2 * 24 * 3600 * 1000

# ── Redis ─────────────────────────────────────────────────────────────────────
r = redis_lib.Redis(host='localhost', port=6379, db=0, decode_responses=True)


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  1. BINANCE ACCOUNT SUMMARY')
print(f'{SEP}')
acct = api('/fapi/v2/account')
if 'error' in acct or 'code' in acct:
    print(f'  ERROR: {acct}')
    wallet = unrealized = available = 0.0
else:
    wallet      = float(acct.get('totalWalletBalance', 0))
    marg_bal    = float(acct.get('totalMarginBalance', 0))
    unrealized  = float(acct.get('totalUnrealizedProfit', 0))
    available   = float(acct.get('availableBalance', 0))
    print(f'  Wallet Balance       : {wallet:>14.4f} USDT')
    print(f'  Margin Balance       : {marg_bal:>14.4f} USDT')
    print(f'  Total Unrealized PnL : {unrealized:>+14.4f} USDT')
    print(f'  Available Balance    : {available:>14.4f} USDT')
    print(f'  Can Trade            : {acct.get("canTrade")}')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  2. BINANCE OPEN POSITIONS')
print(f'{SEP}')
pos_raw = api('/fapi/v2/positionRisk')
if isinstance(pos_raw, dict):
    print(f'  ERROR: {pos_raw}')
    open_pos_binance = []
else:
    open_pos_binance = [p for p in pos_raw if abs(float(p.get('positionAmt', 0))) > 0]

binance_pos_map = {}
for p in open_pos_binance:
    sym  = p['symbol']
    amt  = float(p['positionAmt'])
    side = 'LONG' if amt > 0 else 'SHORT'
    entry = float(p['entryPrice'])
    upnl  = float(p['unRealizedProfit'])
    lev   = p.get('leverage', '?')
    binance_pos_map[sym] = {'amt': amt, 'side': side, 'entry': entry, 'upnl': upnl, 'leverage': lev}
    liq  = float(p.get('liquidationPrice', 0))
    print(f'  {sym:<18} {side:<6} qty={abs(amt):<10.2f} entry={entry:<12.4f} uPnL={upnl:>+10.4f} USDT  lev={lev}x  liq={liq:.4f}')

if not open_pos_binance:
    print('  (no open positions on Binance)')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  3. INTERNAL SYSTEM OPEN POSITIONS (Redis)')
print(f'{SEP}')
pos_keys = sorted([
    k for k in r.keys('quantum:position:*')
    if ':snapshot:' not in k and ':ledger:' not in k
])
sys_pos_map = {}
for pk in pos_keys:
    d    = r.hgetall(pk)
    sym  = d.get('symbol', pk.split(':')[-1])
    side = d.get('side', '?')
    qty  = d.get('quantity', d.get('qty', '?'))
    entry= d.get('entry_price', '?')
    upnl = d.get('unrealized_pnl', '?')
    lev  = d.get('leverage', '?')
    src  = d.get('source', '?')
    sys_pos_map[sym] = {'side': side, 'qty': qty, 'entry': entry, 'upnl': upnl, 'leverage': lev, 'source': src}
    print(f'  {sym:<18} {side:<6} qty={qty:<10}  entry={entry:<12}  uPnL={upnl}  lev={lev}x  src={src}')

if not pos_keys:
    print('  (no open positions in Redis)')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  4. POSITION COMPARISON  (Binance vs System)')
print(f'{SEP}')
all_syms = sorted(set(list(binance_pos_map) + list(sys_pos_map)))
match_cnt = disc_cnt = 0
for sym in all_syms:
    b = binance_pos_map.get(sym)
    s = sys_pos_map.get(sym)
    if b and s:
        try:
            b_qty  = abs(float(b['amt']))
            s_qty  = abs(float(s['qty']))
            side_ok = b['side'] == s['side']
            qty_ok  = abs(b_qty - s_qty) < max(0.01, 0.01 * max(b_qty, s_qty))
            if qty_ok and side_ok:
                print(f'  ✅ {sym:<18} MATCH    B: {b_qty} {b["side"]}  S: {s_qty} {s["side"]}')
                match_cnt += 1
            else:
                mismatch = []
                if not qty_ok:  mismatch.append(f'qty B={b_qty} S={s_qty} diff={abs(b_qty-s_qty):.4f}')
                if not side_ok: mismatch.append(f'side B={b["side"]} S={s["side"]}')
                print(f'  ⚠️  {sym:<18} MISMATCH  {" | ".join(mismatch)}')
                disc_cnt += 1
        except Exception as e:
            print(f'  ❓ {sym:<18} compare error: {e}')
    elif b and not s:
        print(f'  🔴 {sym:<18} BINANCE ONLY  qty={abs(b["amt"])} {b["side"]}  (MISSING in system)')
        disc_cnt += 1
    else:
        print(f'  🟡 {sym:<18} SYSTEM ONLY   qty={s["qty"]} {s["side"]}  (not tracked on Binance)')
        disc_cnt += 1

print(f'\n  Result: {match_cnt} match  |  {disc_cnt} discrepancy')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  5. BINANCE TRADE HISTORY  (last 2 days)')
print(f'{SEP}')

track_syms = sorted(set(list(binance_pos_map) + list(sys_pos_map) + ['ARCUSDT', 'RIVERUSDT']))
binance_trades_map = {}
total_binance_trades = 0

for sym in track_syms:
    trades = api('/fapi/v1/userTrades', {'symbol': sym, 'startTime': TWO_DAYS_MS, 'limit': 100})
    if isinstance(trades, list) and trades:
        binance_trades_map[sym] = trades
        total_binance_trades += len(trades)
        print(f'\n  {sym}  ({len(trades)} fill(s)):')
        for t in trades:
            rpnl = float(t.get('realizedPnl', 0))
            comm = float(t.get('commission', 0))
            comm_a = t.get('commissionAsset', '')
            print(f'    {ts_str(t.get("time", 0))}  {t.get("side","?"):<5}  qty={t.get("qty","?"):<10}  '
                  f'price={float(t.get("price",0)):<12.4f}  rPnL={rpnl:>+10.4f}  '
                  f'fee={comm:.4f} {comm_a}  orderId={t.get("orderId","?")}')
    elif isinstance(trades, dict) and ('error' in trades or 'code' in trades):
        print(f'  {sym}: {trades}')

print(f'\n  Total fills found (Binance 2d): {total_binance_trades}')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  6. BINANCE INCOME / REALIZED PnL  (last 2 days)')
print(f'{SEP}')

def fetch_income(inc_type):
    rows = api('/fapi/v1/income', {'incomeType': inc_type, 'startTime': TWO_DAYS_MS, 'limit': 1000})
    if isinstance(rows, list):
        return rows
    return []

realized_rows  = fetch_income('REALIZED_PNL')
commission_rows= fetch_income('COMMISSION')
funding_rows   = fetch_income('FUNDING_FEE')

total_r  = sum(float(x.get('income', 0)) for x in realized_rows)
total_c  = sum(float(x.get('income', 0)) for x in commission_rows)
total_f  = sum(float(x.get('income', 0)) for x in funding_rows)

print(f'  Realized PnL (trade fills) : {total_r:>+14.4f} USDT  ({len(realized_rows)} entries)')
print(f'  Commissions (fees paid)    : {total_c:>+14.4f} USDT  ({len(commission_rows)} entries)')
print(f'  Funding Fees               : {total_f:>+14.4f} USDT  ({len(funding_rows)} entries)')
print(f'  {SEP2}')
net_binance = total_r + total_c + total_f
print(f'  Net P&L (after fees+fund)  : {net_binance:>+14.4f} USDT')

# Daily breakdown for realized
if realized_rows:
    print(f'\n  Realized PnL — daily per symbol:')
    daily = {}
    for row in realized_rows:
        day = datetime.fromtimestamp(int(row['time'])/1000, tz=timezone.utc).strftime('%Y-%m-%d')
        sym = row.get('symbol', '?')
        amt = float(row.get('income', 0))
        daily.setdefault(day, {}).setdefault(sym, 0.0)
        daily[day][sym] += amt
    for day in sorted(daily):
        day_total = sum(daily[day].values())
        print(f'\n  {day}  (day total: {day_total:>+.4f} USDT)')
        for sym in sorted(daily[day]):
            print(f'    {sym:<18}  rPnL={daily[day][sym]:>+10.4f} USDT')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  7. INTERNAL SYSTEM PnL  (Redis ledger)')
print(f'{SEP}')
ledger_keys = sorted([k for k in r.keys('quantum:ledger:*') if k != 'quantum:ledger:seen_orders'])
sys_ledger_total = 0.0
sys_pnl_by_sym   = {}
for lk in ledger_keys:
    d    = r.hgetall(lk)
    sym  = lk.split(':')[-1]
    tpnl = float(d.get('total_pnl_usdt', 0) or 0)
    wins = d.get('winning_trades', '?')
    losses = d.get('losing_trades', '?')
    total_t = d.get('total_trades', '?')
    fees = float(d.get('total_fees_usdt', 0) or 0)
    vol  = float(d.get('total_volume_usdt', 0) or 0)
    sys_ledger_total += tpnl
    sys_pnl_by_sym[sym] = tpnl
    print(f'  {sym:<18}  PnL={tpnl:>+10.4f} USDT  trades={total_t}  W/L={wins}/{losses}  fees={fees:.4f}  vol={vol:.2f}')

# Position-level ledger
print(f'\n  Position ledger entries:')
for pk in sorted(r.keys('quantum:position:ledger:*')):
    d    = r.hgetall(pk)
    sym  = pk.split(':')[-1]
    side = d.get('ledger_side', d.get('side', '?'))
    qty  = d.get('ledger_amt', d.get('qty', '?'))
    entry= d.get('avg_entry_price', d.get('entry_price', '?'))
    upnl = d.get('unrealized_pnl', '?')
    last_ord = d.get('last_order_id', '?')
    print(f'  {sym:<18}  {side:<6} qty={qty:<10}  entry={entry:<12}  uPnL={upnl}  lastOrder={last_ord}')

print(f'\n  System ledger PnL total: {sys_ledger_total:>+10.4f} USDT')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  8. PnL COMPARISON  (Binance 2d realized vs System ledger)')
print(f'{SEP}')

binance_pnl_by_sym = {}
for row in realized_rows:
    sym = row.get('symbol', '?')
    binance_pnl_by_sym[sym] = binance_pnl_by_sym.get(sym, 0.0) + float(row.get('income', 0))

all_pnl_syms = sorted(set(list(binance_pnl_by_sym) + list(sys_pnl_by_sym)))
if not all_pnl_syms:
    print('  No realized PnL data in either source (no closed trades in 2d window)')
else:
    for sym in all_pnl_syms:
        bp = binance_pnl_by_sym.get(sym)
        sp = sys_pnl_by_sym.get(sym)
        b_s = f'{bp:>+10.4f}' if bp is not None else '    N/A   '
        s_s = f'{sp:>+10.4f}' if sp is not None else '    N/A   '
        if bp is not None and sp is not None:
            diff = bp - sp
            flag = '✅' if abs(diff) < 0.05 else ('⚠️ ' if abs(diff) < 2.0 else '🔴')
            print(f'  {flag} {sym:<18}  Binance={b_s}  System={s_s}  diff={diff:>+8.4f} USDT')
        elif bp is not None:
            print(f'  🆕 {sym:<18}  Binance={b_s}  System=    N/A    (not in ledger)')
        else:
            print(f'  📦 {sym:<18}  Binance=    N/A     System={s_s}  (ledger only)')

print(f'\n  Binance 2d realized PnL : {sum(binance_pnl_by_sym.values()):>+10.4f} USDT')
print(f'  Binance 2d commissions  : {total_c:>+10.4f} USDT')
print(f'  Binance 2d funding fees : {total_f:>+10.4f} USDT')
print(f'  System ledger PnL total : {sys_ledger_total:>+10.4f} USDT')
gap = sum(binance_pnl_by_sym.values()) - sys_ledger_total
print(f'  Gap (B_realized - S_ledger): {gap:>+10.4f} USDT')
if abs(gap) > 0.1:
    print(f'  ⚠️  Gap likely explained by: trades not yet tracked in system ledger,')
    print(f'     or ledger span exceeding 2-day Binance window.')


# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('  9. OPEN ORDERS ON BINANCE')
print(f'{SEP}')
open_orders = api('/fapi/v1/openOrders')
if isinstance(open_orders, list):
    print(f'  Open orders: {len(open_orders)}')
    for o in open_orders:
        print(f'  {ts_str(o.get("time",0))}  {o["symbol"]:<18} {o.get("side","?"):<6} '
              f'{o.get("type","?"):<12} qty={o.get("origQty","?"):<10} '
              f'price={o.get("price","0")}  status={o.get("status","?")}')
    if not open_orders:
        print('  (no open orders)')
else:
    print(f'  {open_orders}')

print(f'\n{SEP}')
print('  AUDIT COMPLETE')
print(SEP)
