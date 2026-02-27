import hmac, hashlib, time, urllib.request, urllib.parse, json
from datetime import datetime

API_KEY = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
API_SECRET = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'
BASE = 'https://testnet.binancefuture.com'

# Hent userTrades for hvert symbol fra Feb 23
symbols = ['INJUSDT', 'LTCUSDT', 'AIUSDT', 'ADAUSDT', 'XRPUSDT', 'ALICEUSDT', 'AEVOUSDT', 'BTCUSDT']

print(f"{'Tid':20s}  {'Symbol':12s}  {'Side':5s}  {'Pris':>10s}  {'Qty':>10s}  {'RealizedPnL':>12s}  {'Fee':>8s}")
print('-'*90)

total_realized = 0.0
total_fees = 0.0

for sym in symbols:
    ts = int(time.time() * 1000)
    params = {'symbol': sym, 'limit': 50, 'timestamp': ts}
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f'{BASE}/fapi/v1/userTrades?{qs}&signature={sig}'
    req = urllib.request.Request(url, headers={'X-MBX-APIKEY': API_KEY})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        trades = json.loads(resp.read())
        if not trades:
            continue
        for t in trades:
            dt = datetime.utcfromtimestamp(t['time']/1000).strftime('%Y-%m-%d %H:%M:%S')
            side = 'BUY' if t['buyer'] else 'SELL'
            pnl = float(t.get('realizedPnl', 0))
            fee = float(t.get('commission', 0))
            qty = float(t['qty'])
            price = float(t['price'])
            total_realized += pnl
            total_fees += fee
            marker = ' <-- TAP' if pnl < -0.01 else (' <-- GEVINST' if pnl > 0.01 else '')
            print(f"{dt:20s}  {sym:12s}  {side:5s}  {price:10.5f}  {qty:10.3f}  {pnl:+12.6f}  {fee:8.6f}{marker}")
    except Exception as e:
        print(f"  {sym}: FEIL - {e}")

print('-'*90)
print(f"  Total Realized PnL:  {total_realized:+.4f} USDT")
print(f"  Total Fees (taker):  {total_fees:+.6f} USDT")
print(f"  Netto etter fees:    {total_realized - total_fees:+.4f} USDT")
