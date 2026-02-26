"""Apply changes 1 and 3 only (change 2 is already done)."""
import sys

path = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(path, "r") as f:
    src = f.read()

errors = []

# ── CHANGE 1: insert _set_leverage() just before _execute_binance_order ──
if "def _set_leverage" in src:
    print("CHANGE1 already applied - skipping")
else:
    old1 = (
        "    def _execute_binance_order(self, symbol: str, side: str, "
        "qty: float, reduce_only: bool = True) -> Dict:"
    )
    new1 = (
        "    def _set_leverage(self, symbol: str, leverage: int) -> bool:\n"
        '        """POST /fapi/v1/leverage — set Binance Futures leverage before entry.\n'
        "        Non-fatal: warns and continues if Binance rejects.\"\"\"\n"
        "        try:\n"
        "            params = {\n"
        '                "symbol": symbol,\n'
        '                "leverage": leverage,\n'
        '                "timestamp": int(time.time() * 1000)\n'
        "            }\n"
        "            query_string = urllib.parse.urlencode(params)\n"
        "            signature = hmac.new(\n"
        "                BINANCE_API_SECRET.encode(),\n"
        "                query_string.encode(),\n"
        "                hashlib.sha256\n"
        "            ).hexdigest()\n"
        '            params["signature"] = signature\n'
        "            url = (\n"
        '                f"{BINANCE_BASE_URL}/fapi/v1/leverage"\n'
        '                f"?{urllib.parse.urlencode(params)}"\n'
        "            )\n"
        '            req = urllib.request.Request(url, method="POST")\n'
        '            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)\n'
        "            with urllib.request.urlopen(req, timeout=10) as response:\n"
        "                result = json.loads(response.read().decode())\n"
        "                logger.info(\n"
        "                    f\"[LEVERAGE] {symbol}: set to "
        "{result.get('leverage')}x OK\"\n"
        "                )\n"
        "                return True\n"
        "        except Exception as e:\n"
        "            logger.warning(\n"
        '                f"[LEVERAGE] {symbol}: set_leverage failed (non-fatal): {e}"\n'
        "            )\n"
        "            return False\n"
        "\n"
        "    def _execute_binance_order(self, symbol: str, side: str, "
        "qty: float, reduce_only: bool = True) -> Dict:"
    )
    if old1 in src:
        src = src.replace(old1, new1, 1)
        print("CHANGE1 applied: _set_leverage method added")
    else:
        errors.append("anchor1 not found")

# ── CHANGE 3: call _set_leverage before _execute_binance_order ──
if "if not reduce_only:\n                self._set_leverage" in src:
    print("CHANGE3 already applied - skipping")
else:
    # Find the exact execute comment to insert before it
    marker3 = "            # Execute Binance order\n"
    idx = src.find(marker3)
    if idx == -1:
        errors.append("anchor3 not found: '# Execute Binance order' comment")
    else:
        insert = (
            "            # Set correct leverage on Binance before entry "
            "(LeverageEngine value, non-fatal)\n"
            "            if not reduce_only:\n"
            "                self._set_leverage(symbol, plan_leverage)\n"
            "\n"
        )
        src = src[:idx] + insert + src[idx:]
        # Also patch the log line to show leverage
        old_log = (
            'logger.info(f"\U0001f680 Executing Binance order: '
            '{symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only}")'
        )
        new_log = (
            'logger.info(f"\U0001f680 Executing Binance order: '
            '{symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only} '
            'leverage={plan_leverage}x")'
        )
        if old_log in src:
            src = src.replace(old_log, new_log, 1)
        print("CHANGE3 applied: _set_leverage called before order")

if errors:
    print("ERRORS:", errors)
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print("SUCCESS: all changes written")
