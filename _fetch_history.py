#!/usr/bin/env python3
"""Fetch trade history + income history from Binance testnet and compare with Redis logs"""
import hashlib, hmac, time, urllib.parse, json, os
import urllib.request

# Read credentials from env files
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

if not API_KEY:
    print("ERROR: No API key found"); exit(1)

print(f"API_KEY: {API_KEY[:16]}...")
print(f"BASE_URL: {BASE_URL}\n")

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
    except Exception as e:
        return {"error": str(e)}

# 1. Account balance
print("=== ACCOUNT BALANCE ===")
bal = signed_request('/fapi/v2/balance')
if isinstance(bal, list):
    for b in bal:
        if float(b.get('balance', 0)) > 0:
            asset = b.get('asset')
            balance = b.get('balance')
            avail = b.get('availableBalance')
            upnl = b.get('crossUnPnl', 0)
            print(f"  {asset}: balance={balance}  available={avail}  unrealizedPnL={upnl}")
else:
    print(f"  {bal}")

# 2. Open positions
print("\n=== OPEN POSITIONS (Binance) ===")
positions = signed_request('/fapi/v2/positionRisk')
open_pos = []
if isinstance(positions, list):
    for p in positions:
        amt = float(p.get('positionAmt', 0))
        if abs(amt) > 0:
            open_pos.append(p)
            sym = p.get('symbol')
            side = 'LONG' if amt > 0 else 'SHORT'
            entry = p.get('entryPrice')
            upnl = p.get('unRealizedProfit')
            lev = p.get('leverage')
            print(f"  {sym:20s} {side:5s} amt={amt}  entry={entry}  uPnL={upnl}  lev={lev}x")
    if not open_pos:
        print("  (no open positions)")
else:
    print(f"  {positions}")

# 3. Recent trades (trade history)
print("\n=== TRADE HISTORY (last 50 fills, all symbols) ===")
# Get income history instead — covers all symbols
income = signed_request('/fapi/v1/income', {
    'incomeType': 'REALIZED_PNL',
    'limit': 50,
    'startTime': int((time.time() - 86400) * 1000)  # last 24h
})
total_pnl = 0.0
if isinstance(income, list):
    print(f"  Found {len(income)} REALIZED_PNL entries (last 24h):")
    for item in income:
        sym = item.get('symbol', '?')
        inc = float(item.get('income', 0))
        total_pnl += inc
        ts = item.get('time', 0)
        t_str = time.strftime('%H:%M:%S', time.gmtime(ts/1000))
        trade_id = item.get('tradeId', '?')
        print(f"  {t_str}  {sym:20s}  PnL={inc:+.4f} USDT  tradeId={trade_id}")
    print(f"\n  TOTAL realized PnL (24h): {total_pnl:+.4f} USDT")
else:
    print(f"  {income}")

# Also get commission income
print("\n=== COMMISSION (last 24h) ===")
comm = signed_request('/fapi/v1/income', {
    'incomeType': 'COMMISSION',
    'limit': 50,
    'startTime': int((time.time() - 86400) * 1000)
})
total_comm = 0.0
if isinstance(comm, list):
    for item in comm:
        inc = float(item.get('income', 0))
        total_comm += inc
    print(f"  Total commission paid: {total_comm:.4f} USDT")
else:
    print(f"  {comm}")

print(f"\n  NET (PnL + commission): {total_pnl + total_comm:+.4f} USDT")

# 4. Compare with Redis logs
print("\n=== REDIS TRADE LOG COMPARISON ===")
import redis as rlib
r = rlib.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# Check trade log stream
stream_keys = ['quantum:stream:trade.closed', 'quantum:stream:trade.executed', 
               'quantum:stream:position.closed', 'quantum:stream:order.filled']
for sk in stream_keys:
    msgs = r.xrevrange(sk, count=10)
    if msgs:
        print(f"\n  Stream {sk} ({len(msgs)} recent):")
        for mid, d in msgs[:5]:
            ts_s = time.strftime('%H:%M:%S', time.gmtime(int(mid.split('-')[0])/1000))
            p = {}
            try: p = json.loads(d.get('payload', '{}'))
            except: p = d
            sym = p.get('symbol', d.get('symbol', '?'))
            pnl = p.get('pnl_usd', p.get('pnl', p.get('realized_pnl', '?')))
            print(f"    {ts_s}  {sym}  pnl={pnl}")

# Check trade keys
trade_keys = sorted(r.keys('quantum:trade:*'))
if trade_keys:
    print(f"\n  quantum:trade:* keys: {len(trade_keys)}")
    for k in trade_keys[-10:]:
        d = r.hgetall(k)
        if d:
            print(f"    {k}: {d}")
