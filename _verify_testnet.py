#!/usr/bin/env python3
"""Verify testnet config and recent trades"""
import redis, json, os

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# 1. Check all env files for testnet setting
print("=== Testnet config ===")
for envfile in [
    '/etc/quantum/autonomous-trader.env',
    '/etc/quantum/apply-layer.env',
    '/etc/quantum/ai-engine.env',
    '/etc/quantum/intent-executor.env',
]:
    try:
        with open(envfile) as f:
            for line in f:
                if any(w in line.upper() for w in ['TESTNET', 'BASE_URL', 'BINANCE_URL', 'FAPI']):
                    print(f"  {envfile.split('/')[-1]}: {line.strip()}")
    except FileNotFoundError:
        pass

# 2. Check signal injector uses correct URL
print("\n=== Signal injector source URL ===")
try:
    with open('/opt/quantum/signal_injector.py') as f:
        for i, line in enumerate(f, 1):
            if 'fapi' in line.lower() or 'binance' in line.lower() or 'testnet' in line.lower():
                print(f"  L{i}: {line.strip()}")
except: pass

# 3. Check last 5 closed trades - were they on testnet?
print("\n=== Last 5 closed trades ===")
trade_keys = sorted(r.keys('quantum:trade:*'))
for k in trade_keys[-5:]:
    t = r.type(k)
    if t == 'hash':
        d = r.hgetall(k)
        print(f"  {k.split(':')[-1]}  symbol={d.get('symbol','?')}  side={d.get('side','?')}  "
              f"pnl={d.get('pnl_usd','?')}  testnet={d.get('testnet','?')}  "
              f"order_id={d.get('order_id','?')[:20] if d.get('order_id') else '?'}")

# 4. Check intent-executor for binance URL
print("\n=== Intent executor binance URL ===")
for base in ['/home/qt/quantum_trader', '/opt/quantum']:
    for dirpath, _, files in os.walk(f'{base}/microservices/intent_executor'):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath) as f:
                        for i, line in enumerate(f, 1):
                            if 'testnet' in line.lower() or 'fapi.binance' in line.lower() or 'base_url' in line.lower():
                                print(f"  {fname}:{i}: {line.strip()}")
                except: pass
