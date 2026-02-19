#!/usr/bin/env python3
import os, hmac, hashlib, urllib.parse, urllib.request, json, time

api_key = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
api_secret = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"
base_url = "https://testnet.binancefuture.com"

params = {"timestamp": int(time.time() * 1000)}
query_string = urllib.parse.urlencode(params)
signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
params["signature"] = signature

url = f"{base_url}/fapi/v2/positionRisk?{urllib.parse.urlencode(params)}"
req = urllib.request.Request(url, method="GET")
req.add_header("X-MBX-APIKEY", api_key)

with urllib.request.urlopen(req, timeout=10) as response:
    positions = json.loads(response.read().decode())
    non_zero = [p for p in positions if abs(float(p.get("positionAmt", 0))) > 0]
    print(f"Total positions: {len(positions)}")
    print(f"Non-zero positions: {len(non_zero)}")
    for p in non_zero[:10]:
        sym = p["symbol"]
        amt = p["positionAmt"]
        entry = p["entryPrice"]
        print(f"  {sym}: amt={amt} entry={entry}")
