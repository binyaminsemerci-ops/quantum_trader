import hmac, hashlib, urllib.parse, urllib.request, urllib.error, json, time

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"
BASE = "https://testnet.binancefuture.com"

def binance_get(path, params={}):
    params["timestamp"] = int(time.time() * 1000)
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE}{path}?{qs}&signature={sig}"
    req = urllib.request.Request(url)
    req.add_header("X-MBX-APIKEY", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()}

# 1. Check DOGEUSDT position
pos = binance_get("/fapi/v2/positionRisk", {"symbol": "DOGEUSDT"})
for p in pos if isinstance(pos, list) else [pos]:
    print(f"Position: qty={p.get('positionAmt')} side={p.get('positionSide')} entry={p.get('entryPrice')}")

# 2. Check open orders
orders = binance_get("/fapi/v1/openOrders", {"symbol": "DOGEUSDT"})
print(f"Open orders: {len(orders) if isinstance(orders, list) else 'ERROR: '+str(orders)}")
for o in (orders if isinstance(orders, list) else []):
    print(f"  orderType={o.get('type')} side={o.get('side')} stopPrice={o.get('stopPrice')} qty={o.get('origQty')}")

# 3. Try a plain MARKET buy as sanity check
print("\n=== ACCOUNT INFO ===")
acc = binance_get("/fapi/v2/account", {})
print(f"totalWalletBalance={acc.get('totalWalletBalance')} availableBalance={acc.get('availableBalance')}")
