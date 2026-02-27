import hmac, hashlib, urllib.parse, urllib.request, urllib.error, json, time

API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"
BASE = "https://testnet.binancefuture.com"

def test_order(params, label):
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(f"{BASE}/fapi/v1/order", data=data, method="POST")
    req.add_header("X-MBX-APIKEY", API_KEY)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            r = json.loads(resp.read())
            print(f"{label} SUCCESS: orderId={r.get('orderId')} status={r.get('status')}")
    except urllib.error.HTTPError as e:
        print(f"{label} ERROR {e.code}:", e.read().decode())

# Test: quantity + reduceOnly (no closePosition)
test_order({"symbol": "DOGEUSDT", "side": "BUY", "type": "STOP_MARKET",
    "stopPrice": "0.110", "quantity": "974", "reduceOnly": "true",
    "timestamp": int(time.time() * 1000)}, "SL qty+reduceOnly")

test_order({"symbol": "DOGEUSDT", "side": "BUY", "type": "TAKE_PROFIT_MARKET",
    "stopPrice": "0.098", "quantity": "974", "reduceOnly": "true",
    "timestamp": int(time.time() * 1000)}, "TP qty+reduceOnly")
