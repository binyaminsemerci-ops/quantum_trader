import requests, hmac, hashlib, time, os
from urllib.parse import urlencode
for line in open("/etc/quantum/testnet.env"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    k = k.strip()
    if k:
        os.environ[k] = v.strip()
KEY = os.environ.get("BINANCE_TESTNET_API_KEY", "")
SEC = os.environ.get("BINANCE_TESTNET_API_SECRET", os.environ.get("BINANCE_TESTNET_SECRET_KEY", ""))
p = {"timestamp": int(time.time() * 1000)}
qs = urlencode(p)
p["signature"] = hmac.new(SEC.encode(), qs.encode(), hashlib.sha256).hexdigest()
r = requests.get("https://testnet.binancefuture.com/fapi/v2/positionRisk",
                 params=p, headers={"X-MBX-APIKEY": KEY}, timeout=10)
open_pos = [x for x in r.json() if abs(float(x.get("positionAmt", 0))) > 0]
for x in open_pos:
    print(x["symbol"], x["positionAmt"], "entry="+x["entryPrice"], "uPnL="+x["unRealizedProfit"])
if not open_pos:
    print("NO_OPEN_POSITIONS")
