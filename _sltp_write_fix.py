import shutil, re

filepath = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
shutil.copy2(filepath, filepath + ".bak_sltp_write")

with open(filepath, "r") as f:
    content = f.read()

# Replace the entire BRACKET ORDERS section with Redis position hash write
old = """                    # BRACKET ORDERS: place STOP_MARKET + TAKE_PROFIT_MARKET after each entry fill
                    if not reduce_only and order_id and final_status == "FILLED":
                        _sl_raw = event_data.get(b"stop_loss", b"0")
                        _tp_raw = event_data.get(b"take_profit", b"0")
                        _sl = float(_sl_raw.decode() if isinstance(_sl_raw, bytes) else str(_sl_raw))
                        _tp = float(_tp_raw.decode() if isinstance(_tp_raw, bytes) else str(_tp_raw))
                        _bracket_side = "SELL" if side == "BUY" else "BUY"
                        if _sl > 0:
                            logger.info(f"\U0001f6e1\ufe0f  Placing STOP_MARKET for {symbol}: stopPrice={_sl}")
                            self._place_bracket_order(symbol, "STOP_MARKET", _bracket_side, _sl)
                        else:
                            logger.warning(f"\u26a0\ufe0f  No stop_loss in plan for {symbol} \u2014 STOP_MARKET skipped")
                        if _tp > 0:
                            logger.info(f"\U0001f3af Placing TAKE_PROFIT_MARKET for {symbol}: stopPrice={_tp}")
                            self._place_bracket_order(symbol, "TAKE_PROFIT_MARKET", _bracket_side, _tp)
                        else:
                            logger.warning(f"\u26a0\ufe0f  No take_profit in plan for {symbol} \u2014 TAKE_PROFIT_MARKET skipped")"""

new = """                    # WRITE POSITION HASH for harvest_brain software SL/TP monitoring (2026-02-26)
                    # Binance testnet does not support STOP_MARKET/TAKE_PROFIT_MARKET orders (-4120).
                    # Instead: write quantum:position:{symbol} so harvest_brain can monitor and
                    # fire MARKET orders when SL/TP price is crossed.
                    if not reduce_only and order_id and final_status == "FILLED":
                        try:
                            _sl_raw = event_data.get(b"stop_loss", b"0")
                            _tp_raw = event_data.get(b"take_profit", b"0")
                            _lev_raw = event_data.get(b"leverage", b"1")
                            _sl = float(_sl_raw.decode() if isinstance(_sl_raw, bytes) else str(_sl_raw))
                            _tp = float(_tp_raw.decode() if isinstance(_tp_raw, bytes) else str(_tp_raw))
                            _lev = float(_lev_raw.decode() if isinstance(_lev_raw, bytes) else str(_lev_raw))
                            _pos_side = "LONG" if side == "BUY" else "SHORT"
                            _risk = abs(float(mark_price) - _sl) * qty_to_use if _sl > 0 else abs(float(mark_price) * 0.02) * qty_to_use
                            _pos_hash = {
                                "symbol": symbol,
                                "side": _pos_side,
                                "quantity": str(qty_to_use),
                                "entry_price": str(mark_price),
                                "stop_loss": str(_sl),
                                "take_profit": str(_tp) if _tp > 0 else "",
                                "leverage": str(_lev),
                                "entry_risk_usdt": str(round(_risk, 6)),
                                "atr_value": "0.02",
                                "volatility_factor": "1.0",
                                "risk_missing": "0" if _sl > 0 else "1",
                                "opened_at": str(int(time.time())),
                                "order_id": str(order_id),
                                "source": "intent_executor_fill",
                            }
                            self.redis.hset(f"quantum:position:{symbol}", mapping=_pos_hash)
                            logger.info(
                                f"\U0001f4be POS_HASH written: {symbol} {_pos_side} "
                                f"sl={_sl} tp={_tp} risk_usdt={_risk:.4f}"
                            )
                        except Exception as _poswrite_err:
                            logger.warning(f"POS_HASH write failed for {symbol}: {_poswrite_err}")"""

count = content.count(old)
if count == 1:
    content = content.replace(old, new)
    with open(filepath, "w") as f:
        f.write(content)
    print("SUCCESS: bracket order replaced with Redis position hash write")
else:
    # Fuzzy match approach
    # Find the BRACKET ORDERS comment line and replace from there to the last _place_bracket_order
    import re
    m = re.search(r'( +# BRACKET ORDERS: place STOP_MARKET.*?)(\n +self\._write_result)', content, re.DOTALL)
    if m:
        old_fuzzy = m.group(1)
        content = content.replace(old_fuzzy, new.rstrip())
        with open(filepath, "w") as f:
            f.write(content)
        print(f"SUCCESS via fuzzy match ({len(old_fuzzy)} chars replaced)")
    else:
        print(f"ERROR: found {count} occurrences, fuzzy match failed")
        print("Searching for BRACKET ORDERS comment...")
        idx = content.find("# BRACKET ORDERS:")
        if idx > 0: print(repr(content[idx:idx+300]))
