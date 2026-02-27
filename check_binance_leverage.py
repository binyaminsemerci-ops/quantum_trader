"""Check current Binance testnet leverage for active symbols."""
import time, hmac, hashlib, urllib.parse, urllib.request, json, subprocess

pid_out = subprocess.run(
    ['systemctl','show','quantum-intent-executor.service','--property=MainPID'],
    capture_output=True, text=True
)
pid = pid_out.stdout.strip().split('=')[-1]
proc_env = {}
try:
    with open(f'/proc/{pid}/environ') as f:
        for item in f.read().split('\x00'):
            if '=' in item:
                k, _, v = item.partition('=')
                proc_env[k] = v
except Exception as e:
    print(f"Could not read proc env: {e}")

KEY    = proc_env.get('BINANCE_API_KEY', '')
SECRET = proc_env.get('BINANCE_API_SECRET', '')
BASE   = proc_env.get('BINANCE_BASE_URL', 'https://testnet.binancefuture.com')

print(f"BASE={BASE}  KEY_OK={bool(KEY)}")

# GET /fapi/v2/positionRisk
params = {"timestamp": int(time.time() * 1000)}
qs  = urllib.parse.urlencode(params)
sig = hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
url = f"{BASE}/fapi/v2/positionRisk?{qs}&signature={sig}"
req = urllib.request.Request(url, method="GET")
req.add_header("X-MBX-APIKEY", KEY)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read().decode())
    # Filter to symbols with non-zero positions or recently used symbols
    relevant = [d for d in data if float(d.get("positionAmt", 0)) != 0]
    if not relevant:
        # show first 10 to see leverage values
        relevant = data[:10]
    for d in relevant:
        print(f"  {d['symbol']:20s}  leverage={d.get('leverage')}  posAmt={d.get('positionAmt')}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}: {body}")
except Exception as ex:
    print(f"ERR: {ex}")

# Also try set_leverage for ALTUSDT=6 and show full response
print("\n--- Testing set_leverage ALTUSDT=6 ---")
params2 = {
    "symbol": "ALTUSDT",
    "leverage": 6,
    "timestamp": int(time.time() * 1000)
}
qs2  = urllib.parse.urlencode(params2)
sig2 = hmac.new(SECRET.encode(), qs2.encode(), hashlib.sha256).hexdigest()
params2["signature"] = sig2
url2 = f"{BASE}/fapi/v1/leverage?{urllib.parse.urlencode(params2)}"
req2 = urllib.request.Request(url2, method="POST")
req2.add_header("X-MBX-APIKEY", KEY)
try:
    with urllib.request.urlopen(req2, timeout=10) as r:
        print("OK:", r.read().decode())
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}: {body}")
except Exception as ex:
    print(f"ERR: {ex}")
