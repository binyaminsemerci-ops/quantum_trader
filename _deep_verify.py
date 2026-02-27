#!/usr/bin/env python3
"""Deep verify: Binance order history vs Redis closed trades"""
import hashlib, hmac, time, urllib.parse, json, os, redis as rlib
import urllib.request

def get_env(path):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, _, v = line.partition('=')
                    env[k.strip()] = v.strip()
    except: pass
    return env

env = {}
for p in ['/etc/quantum/intent-executor.env', '/etc/quantum/apply-layer.env']:
    env.update(get_env(p))

API_KEY    = env.get('BINANCE_TESTNET_API_KEY', '')
API_SECRET = env.get('BINANCE_TESTNET_API_SECRET', '')
BASE_URL   = 'https://testnet.binancefuture.com'

def signed_request(path, params=None):
    params = params or {}
    params['timestamp'] = int(time.time() * 1000)
    params['recvWindow'] = 10000
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE_URL}{path}?{qs}&signature={sig}"
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}
    except Exception as e:
        return {"error": str(e)}

# 1. Income history - NO time filter, all types
print("=== ALL INCOME (no time filter, limit=200) ===")
all_income = signed_request('/fapi/v1/income', {'limit': 200})
if isinstance(all_income, list):
    by_type = {}
    for item in all_income:
        t = item.get('incomeType', '?')
        inc = float(item.get('income', 0))
        by_type.setdefault(t, []).append(inc)
    for t, vals in sorted(by_type.items()):
        print(f"  {t:30s}: count={len(vals):3d}  total={sum(vals):+.4f} USDT")
    
    # Show realized PnL entries
    rpnl = [i for i in all_income if i.get('incomeType') == 'REALIZED_PNL']
    print(f"\n  REALIZED_PNL entries: {len(rpnl)}")
    for item in rpnl[-20:]:
        ts = time.strftime('%m-%d %H:%M', time.gmtime(item['time']/1000))
        print(f"    {ts}  {item['symbol']:20s}  {float(item['income']):+.4f}  tradeId={item.get('tradeId','?')}")
else:
    print(f"  ERROR: {all_income}")

# 2. All orders for symbols in Redis closed trades
print("\n=== BINANCE ORDER HISTORY (closed trade symbols) ===")
r = rlib.Redis(host='127.0.0.1', port=6379, decode_responses=True)
msgs = r.xrevrange('quantum:stream:trade.closed', count=20)

symbols_to_check = set()
redis_trades = []
print(f"  Redis stream has {len(msgs)} closed trade entries:")
for mid, d in msgs:
    ts_s = time.strftime('%m-%d %H:%M:%S', time.gmtime(int(mid.split('-')[0])/1000))
    try:
        p = json.loads(d.get('payload', '{}'))
    except:
        p = d
    sym = p.get('symbol', d.get('symbol', '?'))
    pnl = p.get('pnl_usd', p.get('pnl', p.get('realized_pnl', p.get('pnl_usdt', '?'))))
    side = p.get('side', p.get('direction', '?'))
    order_id = p.get('order_id', p.get('orderId', p.get('close_order_id', '?')))
    entry = p.get('entry_price', '?')
    exit_p = p.get('exit_price', p.get('close_price', '?'))
    reason = p.get('close_reason', p.get('reason', '?'))
    redis_trades.append({'ts': ts_s, 'sym': sym, 'pnl': pnl, 'order_id': order_id})
    print(f"  {ts_s}  {sym:20s}  pnl={pnl}  side={side}  entry={entry}  exit={exit_p}  reason={reason}  order_id={order_id}")
    symbols_to_check.add(sym)

# 3. Check Binance order history per symbol
print("\n=== BINANCE ALL ORDERS per symbol ===")
today_start = int((time.time() - 86400) * 1000)
for sym in sorted(symbols_to_check):
    if sym == '?':
        continue
    orders = signed_request('/fapi/v1/allOrders', {
        'symbol': sym,
        'startTime': today_start,
        'limit': 20
    })
    if isinstance(orders, list) and orders:
        filled = [o for o in orders if o['status'] == 'FILLED']
        print(f"\n  {sym}: {len(filled)} FILLED orders (of {len(orders)} total)")
        for o in filled[-5:]:
            ts_s = time.strftime('%H:%M:%S', time.gmtime(o['time']/1000))
            print(f"    {ts_s}  {o['side']:4s}  qty={o['executedQty']}  avg={o['avgPrice']}  id={o['orderId']}")
    elif isinstance(orders, dict) and 'error' in orders:
        print(f"  {sym}: error={orders['error'][:80]}")

# 4. How does monitor calculate PnL?
print("\n=== MONITOR PNL SOURCE ===")
# Find what key the monitor reads for PnL
for key_pattern in ['quantum:pnl*', 'quantum:stats*', 'quantum:metrics*', 'qt:pnl*']:
    keys = r.keys(key_pattern)
    for k in keys[:5]:
        t = r.type(k)
        if t == 'string':
            print(f"  {k}: {r.get(k)}")
        elif t == 'hash':
            print(f"  {k}: {r.hgetall(k)}")
