import os, hmac, hashlib, time, requests

api_key = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
api_secret = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'

base_url = 'https://testnet.binancefuture.com'
timestamp = int(time.time() * 1000)
query = f'timestamp={timestamp}'
signature = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()

headers = {'X-MBX-APIKEY': api_key}
url = f'{base_url}/fapi/v2/positionRisk?{query}&signature={signature}'
r = requests.get(url, headers=headers)
data = r.json()

if isinstance(data, list):
    positions = [p for p in data if float(p.get('positionAmt', 0)) != 0]
    print(f'ACTIVE_POSITIONS={len(positions)}')
    for p in positions:
        sym = p['symbol']
        amt = p['positionAmt']
        price = p['entryPrice']
        print(f'{sym}: {amt} @ {price}')
else:
    print('ERROR:', data)
