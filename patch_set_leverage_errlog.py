"""Patch _set_leverage to log the Binance error body and treat -4028 as success."""
import sys

path = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(path) as f:
    src = f.read()

old = '''\
    def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set Binance Futures leverage for a symbol before an entry order.
        Non-fatal: warns and continues if Binance rejects (e.g. already set)."""
        try:
            params = {
                "symbol": symbol,
                "leverage": leverage,
                "timestamp": int(time.time() * 1000)
            }
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            url = f"{BINANCE_BASE_URL}/fapi/v1/leverage?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, method="POST")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                actual = result.get("leverage", leverage)
                logger.info(f"[LEVERAGE] {symbol}: set to {actual}x ✅")
                return True
        except Exception as e:
            logger.warning(f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal): {e}")
            return False'''

new = '''\
    def _set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set Binance Futures leverage for a symbol before an entry order.
        Non-fatal: warns and continues if Binance rejects.
        Binance -4028 (Leverage not changed) is treated as success."""
        try:
            params = {
                "symbol": symbol,
                "leverage": leverage,
                "timestamp": int(time.time() * 1000)
            }
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            url = f"{BINANCE_BASE_URL}/fapi/v1/leverage?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, method="POST")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                actual = result.get("leverage", leverage)
                logger.info(f"[LEVERAGE] {symbol}: set to {actual}x OK")
                return True
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                err  = json.loads(body)
                code = err.get("code", 0)
                msg  = err.get("msg", body)
            except Exception:
                code, msg = 0, body
            # -4028 = "Leverage not changed" means already at the correct value
            if code == -4028:
                logger.info(f"[LEVERAGE] {symbol}: already at {leverage}x (no change needed)")
                return True
            logger.warning(f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal) code={code} msg={msg}")
            return False
        except Exception as e:
            logger.warning(f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal): {e}")
            return False'''

if old in src:
    src = src.replace(old, new, 1)
    print("PATCH applied")
else:
    print("ERROR: anchor not found")
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print("File written OK")
