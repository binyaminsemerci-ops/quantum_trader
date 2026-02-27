import hmac, hashlib, time, urllib.request, urllib.parse, json
from datetime import datetime

API_KEY = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
API_SECRET = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'
BASE = 'https://testnet.binancefuture.com'

ts = int(time.time() * 1000)
# Feb 23 00:00 UTC = 1740268800000
start_time = 1740268800000
params = {'limit': 1000, 'startTime': start_time, 'incomeType': 'REALIZED_PNL', 'timestamp': ts}
qs = urllib.parse.urlencode(params)
sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
url = f'{BASE}/fapi/v1/income?{qs}&signature={sig}'
req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
try:
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read())
except Exception as e:
    print(f'ERROR: {e}')
    data = []

if not data:
    print('Ingen data returnert')
else:
    print(f"{'Tid':20s}  {'Type':25s}  {'Symbol':12s}  {'Income':>12s}  Info")
    print('-'*90)
    totals = {}
    for d in data:
        ts_ms = d.get('time', 0)
        t = datetime.utcfromtimestamp(ts_ms/1000).strftime('%Y-%m-%d %H:%M:%S')
        inc_type = d.get('incomeType', '')
        symbol = d.get('symbol', '')
        income = float(d.get('income', 0))
        info = d.get('info', '')
        totals[inc_type] = totals.get(inc_type, 0) + income
        print(f"{t}  {inc_type:25s}  {symbol:12s}  {income:+12.6f}  {info}")
    print('-'*90)
    print("TOTALER:")
    for k, v in sorted(totals.items()):
        print(f"  {k:25s}: {v:+.6f} USDT")
    print(f"  {'NETTO TOTAL':25s}: {sum(totals.values()):+.6f} USDT")
