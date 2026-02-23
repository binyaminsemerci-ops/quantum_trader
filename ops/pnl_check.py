import hmac, hashlib, time, urllib.request, urllib.parse, json
from datetime import datetime

API_KEY = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
API_SECRET = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'
BASE = 'https://testnet.binancefuture.com'

print("=== INCOME TYPES SJEKK ===")
for itype in ['REALIZED_PNL', 'COMMISSION', 'FUNDING_FEE', 'CROSS_COLLATERAL_TRANSFER']:
    ts = int(time.time() * 1000)
    params = {'limit': 10, 'incomeType': itype, 'timestamp': ts}
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f'{BASE}/fapi/v1/income?{qs}&signature={sig}'
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        if data:
            total = sum(float(d['income']) for d in data)
            last = datetime.utcfromtimestamp(data[-1]['time']/1000).strftime('%Y-%m-%d %H:%M')
            print(f"  {itype:25s}: {len(data)} rader, siste={last}, sum={total:+.4f}")
            for d in data:
                t = datetime.utcfromtimestamp(d['time']/1000).strftime('%m-%d %H:%M')
                print(f"    {t}  {d.get('symbol',''):12s}  {float(d['income']):+10.4f}  {d.get('info','')}")
        else:
            print(f"  {itype:25s}: ingen data")
    except Exception as e:
        print(f"  {itype:25s}: FEIL - {e}")

# Hent de 20 siste income-oppføringene uansett type
print("\n=== 20 SISTE INCOME OPPFORINGER ===")
ts = int(time.time() * 1000)
params = {'limit': 20, 'timestamp': ts}
qs = urllib.parse.urlencode(params)
sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
url = f'{BASE}/fapi/v1/income?{qs}&signature={sig}'
req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
resp = urllib.request.urlopen(req, timeout=10)
data = json.loads(resp.read())
for d in data:
    t = datetime.utcfromtimestamp(d['time']/1000).strftime('%Y-%m-%d %H:%M:%S')
    print(f"  {t}  {d.get('incomeType',''):20s}  {d.get('symbol',''):12s}  {float(d.get('income',0)):+10.4f}")
