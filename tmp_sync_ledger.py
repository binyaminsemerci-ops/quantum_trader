import os, time, json, hmac, hashlib, urllib.parse, urllib.request, subprocess

base = "https://testnet.binancefuture.com"
kv = {}
for line in open("/etc/quantum/apply-layer.env").read().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

api_key = kv["BINANCE_TESTNET_API_KEY"]
api_sec = kv["BINANCE_TESTNET_API_SECRET"]
allow = {"BTCUSDT", "ETHUSDT", "TRXUSDT"}

params = {"timestamp": int(time.time() * 1000)}
qs = urllib.parse.urlencode(params)
sig = hmac.new(api_sec.encode(), qs.encode(), hashlib.sha256).hexdigest()
params["signature"] = sig
url = f"{base}/fapi/v2/positionRisk?{urllib.parse.urlencode(params)}"
req = urllib.request.Request(url)
req.add_header("X-MBX-APIKEY", api_key)

with urllib.request.urlopen(req, timeout=10) as r:
    pos = json.loads(r.read().decode())

for p in pos:
    sym = p.get("symbol")
    if sym not in allow:
        continue
    amt = float(p.get("positionAmt", "0") or "0")
    side = "LONG" if amt > 0 else ("SHORT" if amt < 0 else "FLAT")
    subprocess.check_call(
        ["redis-cli", "HSET", f"quantum:position:ledger:{sym}",
         "last_known_amt", str(amt),
         "last_side", side,
         "updated_at", str(int(time.time()))],
        stdout=subprocess.DEVNULL
    )
    print(f"LEDGER_SYNC {sym} amt={amt} side={side}")

print("OK - Ledger synced")
