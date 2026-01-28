import os, re

fp = "/home/qt/quantum_trader/services/execution_service.py"
s = open(fp, "r", encoding="utf-8").read()

# ----------------------------
# Phase 3: reduceOnly enforcement
# Safer than passing None: only include reduceOnly when True
# ----------------------------
# Find the futures_create_order call block and rewrite it to use kwargs
pattern = r"""
market_order\s*=\s*binance_client\.futures_create_order\(
\s*symbol=intent\.symbol,
\s*side=side_binance,
\s*type="MARKET",
\s*quantity=quantity
\s*\)
"""
m = re.search(pattern, s, flags=re.VERBOSE)
if not m:
    raise SystemExit("❌ Could not find futures_create_order(...) block (pattern mismatch).")

replacement = """market_kwargs = dict(
                symbol=intent.symbol,
                side=side_binance,
                type="MARKET",
                quantity=quantity,
            )
            if getattr(intent, "reduce_only", False):
                market_kwargs["reduceOnly"] = True

            market_order = binance_client.futures_create_order(**market_kwargs)
"""
s = re.sub(pattern, replacement, s, count=1, flags=re.VERBOSE)
print("✅ Phase 3 applied: reduceOnly enforced via kwargs (only when True)")

# ----------------------------
# Phase 4: CLOSE_EXECUTED log + optional trade.closed publish (feature-flag)
# Insert right after the existing 'TERMINAL STATE: FILLED' logger block.
# Avoid multiline f-strings; use a single formatted string.
# ----------------------------
needle = '✅ TERMINAL STATE: FILLED'
idx = s.find(needle)
if idx == -1:
    raise SystemExit("❌ Could not find 'TERMINAL STATE: FILLED' marker.")

# Find the end of that logger.info(...) block by scanning forward to the next blank line after it.
post = s[idx:]
# heuristic: insert after the first occurrence of ')\n\n' following the marker
endpos = post.find(")\n\n")
if endpos == -1:
    raise SystemExit("❌ Could not locate end of TERMINAL STATE logger block.")
insert_at = idx + endpos + len(")\n\n")

insert_code = """# P0.4C: close-order proof (reduceOnly chain)
            if getattr(intent, "reduce_only", False):
                src = getattr(intent, "source", None) or "unknown"
                rsn = getattr(intent, "reason", None) or "unknown"
                # NOTE: execution_service uses intent.side in your logs already; fall back safely
                side_val = getattr(intent, "side", None) or getattr(intent, "action", None) or "UNKNOWN"

                logger.info(
                    "✅ CLOSE_EXECUTED symbol=%s side=%s reduceOnly=True source=%s reason=%s qty=%s exit_price=%.6f order_id=%s"
                    % (intent.symbol, side_val, src, rsn, str(actual_qty), float(execution_price), str(order_id))
                )

                # Optional: publish trade.closed ONLY if explicitly enabled to avoid double-publish
                if os.getenv("P04C_PUBLISH_TRADE_CLOSED", "false").lower() == "true":
                    try:
                        closed_event = {
                            "symbol": intent.symbol,
                            "side": side_val,
                            "quantity": actual_qty,
                            "exit_price": execution_price,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "source": src,
                            "reason": rsn,
                            "exit_order_id": order_id,
                            "status": "closed",
                            "fee_usd": fee_usd,
                        }
                        await eventbus.publish(stream="quantum:stream:trade.closed", data=closed_event)
                        logger.info(
                            "📣 CLOSED_PUBLISHED symbol=%s source=%s reason=%s exit_order_id=%s"
                            % (intent.symbol, src, rsn, str(order_id))
                        )
                    except Exception as e:
                        logger.error("❌ Failed to publish trade.closed: %s" % (str(e),))

"""

# Only insert once: guard if already present
if "✅ CLOSE_EXECUTED" in s:
    print("⚠️ Phase 4 already present: CLOSE_EXECUTED found; skipping insert")
else:
    s = s[:insert_at] + insert_code + s[insert_at:]
    print("✅ Phase 4 applied: CLOSE_EXECUTED log + optional trade.closed publish inserted")

open(fp, "w", encoding="utf-8").write(s)
print("✅ Patch complete: execution_service.py updated")
