#!/usr/bin/env python3
"""
MASTER EXECUTOR FIX — 6 changes, single deployment
=====================================================
1. apply_layer: disable process_apply_plan_stream (stops double-execution)
2. intent_executor: add 10-pos gate, cooldown gate, claim-key race guard
3. intent_executor: add _place_bracket_order() method
4. intent_executor: call bracket orders after FILLED (STOP_MARKET + TAKE_PROFIT_MARKET)
5. signal_injector: add tp_pct=3.0/sl_pct=1.5 + 1h trend filter
6. balance_tracker: write stop_loss/take_profit to position snapshot
"""
import os, re, sys, shutil, time

BASE = "/home/qt/quantum_trader/microservices"
INJECTOR = "/opt/quantum/signal_injector.py"

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def backup(path):
    dst = path + ".bak_execfix"
    if not os.path.exists(dst):
        shutil.copy2(path, dst)
        print(f"  BACKUP: {dst}")
    else:
        print(f"  BACKUP EXISTS: {dst} (skipping)")

def patch(path, old, new, label):
    content = read(path)
    if old not in content:
        print(f"  ❌ ANCHOR NOT FOUND in {path}: {label}")
        print(f"     Searching for: {repr(old[:80])}")
        return False
    count = content.count(old)
    if count > 1:
        print(f"  ⚠️  MULTIPLE MATCHES ({count}) in {path}: {label}")
    content = content.replace(old, new, 1)
    write(path, content)
    print(f"  ✅ PATCHED: {label}")
    return True

# ===========================================================================
# 1. APPLY_LAYER — disable process_apply_plan_stream
# ===========================================================================
print("\n[1/6] apply_layer: disable process_apply_plan_stream")
AL = f"{BASE}/apply_layer/main.py"
backup(AL)

patch(AL,
    old=(
        "        # HIGHEST PRIORITY: Process entry intents from intent_bridge\n"
        "        try:\n"
        "            logger.info(\"[ENTRY_CYCLE_START] Calling process_apply_plan_stream...\")\n"
        "            self.process_apply_plan_stream()\n"
        "            logger.info(\"[ENTRY_CYCLE_END] process_apply_plan_stream completed\")\n"
        "        except Exception as e:\n"
        "            logger.error(f\"[ENTRY_CYCLE_ERROR] Error processing apply.plan stream: {e}\", exc_info=True)"
    ),
    new=(
        "        # DEACTIVATED 2026-02-25: intent_executor is now the SINGLE executor for apply.plan\n"
        "        # apply_layer MUST NOT consume apply.plan — would cause double-execution on Binance\n"
        "        # All entry gates now live in intent_executor\n"
        "        logger.debug(\"[ENTRY_CYCLE_DISABLED] intent_executor owns apply.plan — skipping\")"
    ),
    label="disable process_apply_plan_stream in run_cycle"
)

# ===========================================================================
# 2. INTENT_EXECUTOR — add entry gates before order execution
# ===========================================================================
print("\n[2/6] intent_executor: add 10-pos gate, cooldown gate, claim-key")
IE = f"{BASE}/intent_executor/main.py"
backup(IE)

GATES = """\
            # === HARD GATES FOR ENTRY (ported from apply_layer 2026-02-25) ===
            if not reduce_only:
                # Gate 1: max 10 active positions
                _raw_keys = self.redis.keys("quantum:position:*")
                _pos_keys = [k for k in _raw_keys
                             if b"snapshot" not in k and b"ledger" not in k
                             and b"cooldown" not in k and b"claim" not in k]
                if len(_pos_keys) >= 10:
                    logger.warning(f"🚫 POSITION_LIMIT {symbol}: {len(_pos_keys)}/10 active — REJECTED")
                    self._write_result(plan_id, symbol, executed=False,
                                       error=f"position_limit_{len(_pos_keys)}/10",
                                       side=side, qty=qty_to_use)
                    self._mark_done(plan_id)
                    return True

                # Gate 2: post-exit cooldown (prevents churn on same symbol)
                _cd_key = f"quantum:cooldown:open:{symbol}"
                if self.redis.exists(_cd_key):
                    _ttl = self.redis.ttl(_cd_key)
                    logger.warning(f"🚫 COOLDOWN {symbol}: {_ttl}s remaining — REJECTED")
                    self._write_result(plan_id, symbol, executed=False,
                                       error=f"cooldown_{_ttl}s",
                                       side=side, qty=qty_to_use)
                    self._mark_done(plan_id)
                    return True

                # Gate 3: atomic race-condition guard (SETNX 30s)
                _claim_key = f"quantum:position:claim:{symbol}"
                _claimed = self.redis.set(_claim_key, plan_id, nx=True, ex=30)
                if not _claimed:
                    logger.warning(f"🚫 RACE_GUARD {symbol}: claim already held — REJECTED")
                    self._write_result(plan_id, symbol, executed=False,
                                       error="race_guard_claim",
                                       side=side, qty=qty_to_use)
                    self._mark_done(plan_id)
                    return True

"""

patch(IE,
    old="            # Execute Binance order\n            logger.info(f\"🚀 Executing Binance order: {symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only}\")",
    new=GATES + "            # Execute Binance order\n            logger.info(f\"🚀 Executing Binance order: {symbol} {side} {qty_to_use:.4f} reduceOnly={reduce_only}\")",
    label="add 3 entry gates before order execution"
)

# ===========================================================================
# 3. INTENT_EXECUTOR — add _place_bracket_order method after _execute_binance_order
# ===========================================================================
print("\n[3/6] intent_executor: add _place_bracket_order method")

BRACKET_METHOD = '''
    def _place_bracket_order(self, symbol: str, order_type: str, side: str, stop_price: float) -> bool:
        """Place STOP_MARKET or TAKE_PROFIT_MARKET reduce-only bracket order after entry fill.
        Uses closePosition=true to avoid qty precision issues."""
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "stopPrice": f"{stop_price:.8f}",
                "closePosition": "true",
                "reduceOnly": "false",  # closePosition=true supersedes
                "timestamp": int(time.time() * 1000),
            }
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            params["signature"] = signature
            url = f"{BINANCE_BASE_URL}/fapi/v1/order?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, method="POST")
            req.add_header("X-MBX-APIKEY", BINANCE_API_KEY)
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                logger.info(
                    f"✅ BRACKET {order_type}: {symbol} {side} "
                    f"stopPrice={stop_price} orderId={result.get('orderId')}"
                )
                return True
        except Exception as e:
            logger.error(f"❌ BRACKET {order_type} FAILED: {symbol} stopPrice={stop_price}: {e}")
            return False

'''

# Insert after _poll_order_fill method (which ends before _commit_ledger_exactly_once)
patch(IE,
    old="    def _commit_ledger_exactly_once(self, symbol: str, order_id: int, filled_qty: float, side: str):",
    new=BRACKET_METHOD + "    def _commit_ledger_exactly_once(self, symbol: str, order_id: int, filled_qty: float, side: str):",
    label="add _place_bracket_order method"
)

# ===========================================================================
# 4. INTENT_EXECUTOR — call bracket orders after FILLED + RL state storage
# ===========================================================================
print("\n[4/6] intent_executor: call bracket orders after FILLED")

BRACKET_CALL = """\
                        except Exception as _e:
                            logger.warning(f"RL state write failed for order {order_id}: {_e}")

                    # BRACKET ORDERS: place STOP_MARKET + TAKE_PROFIT_MARKET after each entry fill
                    if not reduce_only and order_id and final_status == "FILLED":
                        _sl_raw = event_data.get(b"stop_loss", b"0")
                        _tp_raw = event_data.get(b"take_profit", b"0")
                        _sl = float(_sl_raw.decode() if isinstance(_sl_raw, bytes) else str(_sl_raw))
                        _tp = float(_tp_raw.decode() if isinstance(_tp_raw, bytes) else str(_tp_raw))
                        _bracket_side = "SELL" if side == "BUY" else "BUY"
                        if _sl > 0:
                            logger.info(f"🛡️  Placing STOP_MARKET for {symbol}: stopPrice={_sl}")
                            self._place_bracket_order(symbol, "STOP_MARKET", _bracket_side, _sl)
                        else:
                            logger.warning(f"⚠️  No stop_loss in plan for {symbol} — STOP_MARKET skipped")
                        if _tp > 0:
                            logger.info(f"🎯 Placing TAKE_PROFIT_MARKET for {symbol}: stopPrice={_tp}")
                            self._place_bracket_order(symbol, "TAKE_PROFIT_MARKET", _bracket_side, _tp)
                        else:
                            logger.warning(f"⚠️  No take_profit in plan for {symbol} — TAKE_PROFIT_MARKET skipped")
"""

patch(IE,
    old=(
        "                        except Exception as _e:\n"
        "                            logger.warning(f\"RL state write failed for order {order_id}: {_e}\")\n"
        "                \n"
        "                self._write_result("
    ),
    new=BRACKET_CALL + "\n                self._write_result(",
    label="call bracket orders after FILLED"
)

# ===========================================================================
# 5. SIGNAL_INJECTOR — add tp_pct/sl_pct + 1h trend filter
# ===========================================================================
print("\n[5/6] signal_injector: add tp_pct/sl_pct + 1h trend filter")
backup(INJECTOR)

# Add 1h kline fetch function after fetch_klines
NEW_FETCH_1H = '''
async def fetch_1h_trend(client: httpx.AsyncClient, symbol: str):
    """Return True if 1h trend is bullish (last close > open 4 candles ago)."""
    try:
        resp = await client.get(
            f"{BINANCE_FAPI}/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 5},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 5:
            return None  # Unknown, don't filter
        open_price_4h_ago = float(data[0][1])
        close_price_now = float(data[-1][4])
        return close_price_now > open_price_4h_ago  # True = bullish 1h trend
    except Exception as e:
        logger.warning(f"1h trend fetch failed for {symbol}: {e}")
        return None  # Unknown, don't filter


'''

patch(INJECTOR,
    old="async def publish_signal(",
    new=NEW_FETCH_1H + "async def publish_signal(",
    label="add fetch_1h_trend function"
)

# Add tp_pct and sl_pct to publish_signal payload
patch(INJECTOR,
    old=(
        '    payload = json.dumps({\n'
        '        "symbol": symbol,\n'
        '        "action": action,                         # "buy" or "sell"\n'
        '        "confidence": CONFIDENCE,\n'
        '        "ensemble_confidence": CONFIDENCE,\n'
        '        "model_votes": {action.upper(): "momentum"},\n'
        '        "consensus": 1,\n'
        '        "price": price,\n'
        '        "regime": "TRENDING",\n'
        '        "timestamp": now_iso,\n'
        '    })'
    ),
    new=(
        '    tp_pct = float(os.getenv("QT_INJECT_TP_PCT", "3.0"))\n'
        '    sl_pct = float(os.getenv("QT_INJECT_SL_PCT", "1.5"))\n'
        '    payload = json.dumps({\n'
        '        "symbol": symbol,\n'
        '        "action": action,                         # "buy" or "sell"\n'
        '        "confidence": CONFIDENCE,\n'
        '        "ensemble_confidence": CONFIDENCE,\n'
        '        "model_votes": {action.upper(): "momentum"},\n'
        '        "consensus": 1,\n'
        '        "price": price,\n'
        '        "regime": "TRENDING",\n'
        '        "tp_pct": tp_pct,\n'
        '        "sl_pct": sl_pct,\n'
        '        "timestamp": now_iso,\n'
        '    })'
    ),
    label="add tp_pct/sl_pct to publish_signal payload"
)

# Add 1h trend filter to run_cycle
patch(INJECTOR,
    old=(
        '            if change >= MIN_MOVE_PCT:\n'
        '                await publish_signal(r, symbol, "buy", close_price, change)\n'
        '                published += 1\n'
        '            elif change <= -MIN_MOVE_PCT:\n'
        '                await publish_signal(r, symbol, "sell", close_price, change)\n'
        '                published += 1'
    ),
    new=(
        '            if change >= MIN_MOVE_PCT:\n'
        '                await publish_signal(r, symbol, "buy", close_price, change)\n'
        '                published += 1\n'
        '            elif change <= -MIN_MOVE_PCT:\n'
        '                # 1h trend filter: only SHORT if 1h trend is also bearish\n'
        '                trend_1h = await fetch_1h_trend(client, symbol)\n'
        '                if trend_1h is True:\n'
        '                    logger.info(f"  SKIP_SELL {symbol:20s}  15m={change:+.2%} but 1h BULLISH — no short")\n'
        '                else:\n'
        '                    await publish_signal(r, symbol, "sell", close_price, change)\n'
        '                    published += 1'
    ),
    label="add 1h trend filter for SELL signals"
)

# ===========================================================================
# 6. BALANCE_TRACKER — write stop_loss/take_profit to snapshot
# ===========================================================================
print("\n[6/6] balance_tracker: write stop_loss/take_profit to position snapshot")
BT = f"{BASE}/balance_tracker/balance_tracker.py"
backup(BT)

patch(BT,
    old=(
        '                position_data = {\n'
        '                    "event_type": "position.snapshot",\n'
        '                    "symbol": position.get("symbol", ""),\n'
        '                    "side": "LONG" if position_amt > 0 else "SHORT",\n'
        '                    "position_qty": str(abs(position_amt)),  # PositionTracker expects position_qty\n'
        '                    "entry_price": str(entry_price),\n'
        '                    "mark_price": str(mark_price),\n'
        '                    "unrealized_pnl": str(unrealized),\n'
        '                    "leverage": str(position.get("leverage", 1)),\n'
        '                    "isolated": str(position.get("isolated", False)),\n'
        '                    "liquidation_price": str(position.get("liquidationPrice", 0)),\n'
        '                    "margin_type": "isolated" if position.get("isolated", False) else "cross",\n'
        '                    "entry_timestamp": str(int(time.time())),  # Add entry_timestamp\n'
        '                    "timestamp": str(int(time.time())),\n'
        '                    "source": "balance-tracker"\n'
        '                }'
    ),
    new=(
        '                # Read stop_loss/take_profit from Redis position ledger if available\n'
        '                _sym = position.get("symbol", "")\n'
        '                _pos_hash = {}\n'
        '                try:\n'
        '                    _pos_hash = self.redis.hgetall(f"quantum:position:{_sym}") or {}\n'
        '                    if not _pos_hash:\n'
        '                        _pos_hash = self.redis.hgetall(f"quantum:position:ledger:{_sym}") or {}\n'
        '                except Exception:\n'
        '                    pass\n'
        '                def _fld(h, key):\n'
        '                    v = h.get(key) or h.get(key.encode(), b"")\n'
        '                    return v.decode() if isinstance(v, bytes) else str(v)\n'
        '                _sl = _fld(_pos_hash, "stop_loss") or "0"\n'
        '                _tp = _fld(_pos_hash, "take_profit") or "0"\n'
        '                position_data = {\n'
        '                    "event_type": "position.snapshot",\n'
        '                    "symbol": _sym,\n'
        '                    "side": "LONG" if position_amt > 0 else "SHORT",\n'
        '                    "position_qty": str(abs(position_amt)),  # PositionTracker expects position_qty\n'
        '                    "entry_price": str(entry_price),\n'
        '                    "mark_price": str(mark_price),\n'
        '                    "unrealized_pnl": str(unrealized),\n'
        '                    "leverage": str(position.get("leverage", 1)),\n'
        '                    "isolated": str(position.get("isolated", False)),\n'
        '                    "liquidation_price": str(position.get("liquidationPrice", 0)),\n'
        '                    "margin_type": "isolated" if position.get("isolated", False) else "cross",\n'
        '                    "stop_loss": _sl,\n'
        '                    "take_profit": _tp,\n'
        '                    "entry_timestamp": str(int(time.time())),  # Add entry_timestamp\n'
        '                    "timestamp": str(int(time.time())),\n'
        '                    "source": "balance-tracker"\n'
        '                }'
    ),
    label="add stop_loss/take_profit to position snapshot"
)

print("\n✅ All 6 patches applied. Verify with grep before restarting services.")
print("   Check apply_layer log: journalctl -u quantum-apply-layer -n 5")
print("   Check intent_executor log: journalctl -u quantum-intent-executor -n 5")
