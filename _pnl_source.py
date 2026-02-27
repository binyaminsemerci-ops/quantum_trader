#!/usr/bin/env python3
"""Find exact PnL source in monitor + get real Binance fills"""
import hashlib, hmac, time, urllib.parse, json, redis as rlib
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

r = rlib.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Read monitor.py to find how PnL is calculated
print("=== Monitor PnL calculation source ===")
try:
    with open('/tmp/_monitor.py') as f:
        code = f.read()
    for i, line in enumerate(code.splitlines(), 1):
        if any(w in line.lower() for w in ['pnl', '21', 'trade', 'wr', 'winrate', 'realized', 'summary']):
            print(f"  L{i:3d}: {line.rstrip()}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Dump raw bytes of a trade.closed stream entry
print("\n=== Raw trade.closed stream entry (first 3) ===")
msgs = r.xrevrange('quantum:stream:trade.closed', count=3)
for mid, d in msgs:
    print(f"\n  ID: {mid}")
    for k, v in d.items():
        print(f"    {k!r}: {v!r}")

# 3. Get actual user trades (fills) from Binance for ETHUSDT today
print("\n=== Binance userTrades for ETHUSDT (last 24h) ===")
today_ms = int((time.time() - 86400) * 1000)
trades = signed_request('/fapi/v1/userTrades', {
    'symbol': 'ETHUSDT',
    'startTime': today_ms,
    'limit': 50
})
total_pnl = 0.0
if isinstance(trades, list):
    for t in trades:
        pnl = float(t.get('realizedPnl', 0))
        total_pnl += pnl
        ts = time.strftime('%H:%M:%S', time.gmtime(t['time']/1000))
        print(f"  {ts}  {t['side']:4s}  qty={t['qty']}  price={t['price']}  "
              f"realizedPnl={pnl:+.4f}  commission={t.get('commission','?')} {t.get('commissionAsset','')}")
    print(f"  ETHUSDT total realized PnL today: {total_pnl:+.4f}")
else:
    print(f"  {trades}")

# 4. Get all realized PnL from userTrades for multiple symbols
print("\n=== Total realized PnL from userTrades (5 most active symbols) ===")
symbols = ['ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'ACTUSDT', '1000CHEEMSUSDT']
grand_total = 0.0
grand_comm = 0.0
for sym in symbols:
    trades = signed_request('/fapi/v1/userTrades', {
        'symbol': sym,
        'startTime': today_ms,
        'limit': 50
    })
    if isinstance(trades, list):
        sym_pnl = sum(float(t.get('realizedPnl', 0)) for t in trades)
        sym_comm = sum(float(t.get('commission', 0)) for t in trades)
        grand_total += sym_pnl
        grand_comm += sym_comm
        if sym_pnl != 0 or sym_comm != 0:
            print(f"  {sym:20s}  realizedPnl={sym_pnl:+.4f}  commission={sym_comm:.4f}  net={sym_pnl - sym_comm:+.4f}")
    else:
        print(f"  {sym:20s}  error={trades.get('error','?')[:60]}")

print(f"\n  GRAND TOTAL realized: {grand_total:+.4f}")
print(f"  GRAND TOTAL commission: {grand_comm:.4f}")
print(f"  NET: {grand_total - grand_comm:+.4f}")
