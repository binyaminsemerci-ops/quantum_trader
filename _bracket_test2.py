import hmac, hashlib, urllib.parse, urllib.request, urllib.error, json, time

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"
BASE = "https://testnet.binancefuture.com"

def post_order(params, label, endpoint="/fapi/v1/order"):
    params["timestamp"] = int(time.time() * 1000)
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    data = (qs + "&signature=" + sig).encode()
    req = urllib.request.Request(f"{BASE}{endpoint}", data=data, method="POST")
    req.add_header("X-MBX-APIKEY", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            res = json.loads(r.read())
            print(f"{label} SUCCESS: orderId={res.get('orderId','?')} status={res.get('status','?')}")
    except urllib.error.HTTPError as e:
        print(f"{label} ERROR {e.code}:", e.read().decode())

# Check supported order types for DOGEUSDT
req = urllib.request.Request(f"{BASE}/fapi/v1/exchangeInfo")
with urllib.request.urlopen(req) as r:
    info = json.loads(r.read())
for sym in info["symbols"]:
    if sym["symbol"] == "DOGEUSDT":
        print("Supported orderTypes:", sym.get("orderTypes"))
        print("Filters:", [f["filterType"] for f in sym.get("filters",[])])
        break

print()
# Try STOP (with price) instead of STOP_MARKET
post_order({"symbol": "DOGEUSDT", "side": "BUY", "type": "STOP",
    "price": "0.110", "stopPrice": "0.1095", "quantity": "974",
    "timeInForce": "GTC", "reduceOnly": "true"}, "STOP limit-order SL")

# Try TAKE_PROFIT (with price) instead of TAKE_PROFIT_MARKET
post_order({"symbol": "DOGEUSDT", "side": "BUY", "type": "TAKE_PROFIT",
    "price": "0.100", "stopPrice": "0.1005", "quantity": "974",
    "timeInForce": "GTC", "reduceOnly": "true"}, "TAKE_PROFIT limit-order TP")
