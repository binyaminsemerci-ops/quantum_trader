"""
Patch intent_executor/main.py to call set_leverage before every entry order.
Run on VPS: python3 /tmp/patch_executor_leverage.py
"""
import sys

path = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(path, "r") as f:
    src = f.read()

errors = []

# ── CHANGE 1: insert _set_leverage() method just before _execute_binance_order ──
old1 = (
    "    def _execute_binance_order(self, symbol: str, side: str, "
    "qty: float, reduce_only: bool = True) -> Dict:"
)
new1 = """\
    def _set_leverage(self, symbol: str, leverage: int) -> bool:
        \"\"\"POST /fapi/v1/leverage to set Binance Futures leverage before entry.
        Non-fatal: warns and continues if Binance rejects (e.g. already set).\"\"\"
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
            url = (
                f"{BINANCE_BASE_URL}/fapi/v1/leverage"
                f"?{urllib.parse.urlencode(params)}"
            )
            req = urllib.request.Request(url, method="POST")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                logger.info(
                    f"[LEVERAGE] {symbol}: set to {result.get('leverage')}x OK"
                )
                return True
        except Exception as e:
            logger.warning(
                f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal): {e}"
            )
            return False

    def _execute_binance_order(self, symbol: str, side: str, \
qty: float, reduce_only: bool = True) -> Dict:"""

if old1 in src:
    src = src.replace(old1, new1, 1)
    print("CHANGE1 applied: _set_leverage method added")
else:
    errors.append("anchor1 not found: _execute_binance_order def line")

# ── CHANGE 2: extract plan_leverage from event_data in process_plan ──
# Insert right after the reduce_only line, before the "Log warning" comment
old2 = (
    'reduce_only = reduce_only_str in ("true", "1", "yes")\n'
    '\n'
    '            # Log warning if field was missing'
)
new2 = (
    'reduce_only = reduce_only_str in ("true", "1", "yes")\n'
    '\n'
    '            # Extract leverage from plan (LeverageEngine value via intent-bridge)\n'
    '            leverage_str = event_data.get(b"leverage", b"1").decode()\n'
    '            try:\n'
    '                plan_leverage = max(1, min(125, int(float(leverage_str))))\n'
    '            except (ValueError, TypeError):\n'
    '                plan_leverage = 1\n'
    '\n'
    '            # Log warning if field was missing'
)

if old2 in src:
    src = src.replace(old2, new2, 1)
    print("CHANGE2 applied: plan_leverage extracted from event_data")
else:
    errors.append("anchor2 not found: reduce_only_str block")

# ── CHANGE 3: call _set_leverage before _execute_binance_order ──
old3 = (
    '            # Execute Binance order\n'
    '            logger.info(f"\U0001f680 Executing Binance order: '
    '{symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only}")'
)
new3 = (
    '            # Set correct leverage on Binance before entry (LeverageEngine value, non-fatal)\n'
    '            if not reduce_only:\n'
    '                self._set_leverage(symbol, plan_leverage)\n'
    '\n'
    '            # Execute Binance order\n'
    '            logger.info(f"\U0001f680 Executing Binance order: '
    '{symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only} leverage={plan_leverage}x")'
)

if old3 in src:
    src = src.replace(old3, new3, 1)
    print("CHANGE3 applied: _set_leverage called before order")
else:
    errors.append("anchor3 not found: Execute Binance order block")

if errors:
    print("ERRORS:", errors)
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print("SUCCESS: File written OK")
