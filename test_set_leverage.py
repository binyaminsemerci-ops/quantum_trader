"""Test /fapi/v1/leverage endpoint directly using the running service's env"""
import time, hmac, hashlib, urllib.parse, urllib.request, subprocess

# Get env from running process
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

print(f"BASE   = {BASE}")
print(f"KEY    = {'*' * 8 + KEY[-4:] if KEY else 'MISSING'}")
print(f"SECRET = {'*' * 8 + SECRET[-4:] if SECRET else 'MISSING'}")

for symbol, lev in [("ALTUSDT", 6), ("ACHUSDT", 6)]:
    params = {
        "symbol": symbol,
        "leverage": lev,
        "timestamp": int(time.time() * 1000)
    }
    qs  = urllib.parse.urlencode(params)
    sig = hmac.new(SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    url = f"{BASE}/fapi/v1/leverage?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="POST")
    req.add_header("X-MBX-APIKEY", KEY)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            print(f"[{symbol}] OK: {r.read().decode()}")
    except urllib.error.HTTPError as e:
        print(f"[{symbol}] HTTP {e.code}: {e.read().decode()}")
    except Exception as ex:
        print(f"[{symbol}] ERR: {ex}")
