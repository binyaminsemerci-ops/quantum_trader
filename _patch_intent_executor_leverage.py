import re

PATH = '/home/qt/quantum_trader/microservices/intent_executor/main.py'
with open(PATH, 'r') as f:
    src = f.read()

# =============================================================================
# CHANGE 1: Add _set_leverage() method before _execute_binance_order
# =============================================================================
ANCHOR1_OLD = '    def _execute_binance_order(self, symbol: str, side: str, qty: float, reduce_only: bool = True) -> Dict:'
ANCHOR1_NEW = '''    def _set_leverage(self, symbol: str, leverage: int) -> bool:
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
                logger.info(f"[LEVERAGE] {symbol}: set to {actual}x \u2705")
                return True
        except Exception as e:
            logger.warning(f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal): {e}")
            return False

    def _execute_binance_order(self, symbol: str, side: str, qty: float, reduce_only: bool = True) -> Dict:'''

if ANCHOR1_OLD not in src:
    print("ERROR: anchor1 not found")
    exit(1)
src = src.replace(ANCHOR1_OLD, ANCHOR1_NEW, 1)
print("CHANGE1 applied: _set_leverage method added")

# =============================================================================
# CHANGE 2: Extract plan_leverage from event_data after reduce_only parse
# =============================================================================
ANCHOR2_OLD = (
    '            # Log warning if field was missing (indicates old/malformed plan)\n'
    '            if b"reduceOnly" not in event_data:'
)
ANCHOR2_NEW = (
    '            # Extract leverage from plan (LeverageEngine via intent-bridge)\n'
    '            _lev_str = event_data.get(b"leverage", b"1").decode()\n'
    '            try:\n'
    '                plan_leverage = max(1, min(125, int(float(_lev_str))))\n'
    '            except (ValueError, TypeError):\n'
    '                plan_leverage = 1\n'
    '\n'
    '            # Log warning if field was missing (indicates old/malformed plan)\n'
    '            if b"reduceOnly" not in event_data:'
)

if ANCHOR2_OLD not in src:
    print("ERROR: anchor2 not found")
    exit(1)
src = src.replace(ANCHOR2_OLD, ANCHOR2_NEW, 1)
print("CHANGE2 applied: plan_leverage extraction added")

# =============================================================================
# CHANGE 3: Call _set_leverage before entry orders
# =============================================================================
ANCHOR3_OLD = (
    '            # Execute Binance order\n'
    '            logger.info(f"\U0001f680 Executing Binance order: {symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only}")'
)
ANCHOR3_NEW = (
    '            # Set leverage on Binance before entry orders (LeverageEngine value)\n'
    '            if not reduce_only:\n'
    '                self._set_leverage(symbol, plan_leverage)\n'
    '\n'
    '            # Execute Binance order\n'
    '            logger.info(f"\U0001f680 Executing Binance order: {symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only} leverage={plan_leverage}x")'
)

if ANCHOR3_OLD not in src:
    print("ERROR: anchor3 not found - trying alternate...")
    # Try finding it with grep
    import subprocess
    result = subprocess.run(['grep', '-n', 'Execute Binance order', PATH], capture_output=True, text=True)
    print(result.stdout)
    exit(1)
src = src.replace(ANCHOR3_OLD, ANCHOR3_NEW, 1)
print("CHANGE3 applied: _set_leverage call before entry")

with open(PATH, 'w') as f:
    f.write(src)
print("File written OK")
