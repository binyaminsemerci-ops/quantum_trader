#!/usr/bin/env python3
"""P0.4C: Add reduceOnly + trade.closed publishing"""

filepath = "/home/qt/quantum_trader/services/execution_service.py"

with open(filepath) as f:
    content = f.read()

# PHASE 3: Add reduceOnly parameter
old_call = """            market_order = binance_client.futures_create_order(
                symbol=intent.symbol,
                side=side_binance,
                type="MARKET",
                quantity=quantity
            )"""

new_call = """            market_order = binance_client.futures_create_order(
                symbol=intent.symbol,
                side=side_binance,
                type="MARKET",
                quantity=quantity,
                reduceOnly=intent.reduce_only if intent.reduce_only else None
            )"""

if old_call in content:
    content = content.replace(old_call, new_call)
    print("✅ Phase 3: Added reduceOnly parameter")
else:
    print("⚠️ Phase 3: Pattern not found (maybe already applied?)")

# PHASE 4: Add CLOSE_EXECUTED log and trade.closed publishing
# Find the line after TERMINAL STATE: FILLED log
marker = """            # P0 FIX: Log terminal state for watchdog monitoring (Phase 2)
            logger.info(
                f"✅ TERMINAL STATE: FILLED | {intent.symbol} {intent.side} | "
                f"OrderID={order_id} | trace_id={trace_id}"
            )

            # 7. Update stats"""

insert_code = """            # P0 FIX: Log terminal state for watchdog monitoring (Phase 2)
            logger.info(
                f"✅ TERMINAL STATE: FILLED | {intent.symbol} {intent.side} | "
                f"OrderID={order_id} | trace_id={trace_id}"
            )

            # P0.4C: If reduce_only, log and publish trade.closed
            if intent.reduce_only:
                close_log_msg = (
                    f"✅ CLOSE_EXECUTED symbol={intent.symbol} side={intent.side} "
                    f"reduceOnly=True source={intent.source or 'unknown'} "
                    f"reason={intent.reason or 'unknown'} qty={actual_qty} "
                    f"exit_price={execution_price:.6f} binance_order_id={order_id}"
                )
                logger.info(close_log_msg)
                
                # Publish to trade.closed stream
                try:
                    closed_event = {
                        "symbol": intent.symbol,
                        "side": intent.side,
                        "quantity": actual_qty,
                        "exit_price": execution_price,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "source": intent.source or "unknown",
                        "reason": intent.reason or "unknown",
                        "exit_order_id": order_id,
                        "status": "closed",
                        "fee_usd": fee_usd
                    }
                    await eventbus.publish(
                        stream="quantum:stream:trade.closed",
                        data=closed_event
                    )
                    logger.info(
                        f"📣 CLOSED_PUBLISHED symbol={intent.symbol} "
                        f"source={intent.source or 'unknown'} "
                        f"reason={intent.reason or 'unknown'} "
                        f"exit_order_id={order_id}"
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to publish trade.closed: {e}")

            # 7. Update stats"""

if marker in content:
    content = content.replace(marker, insert_code)
    print("✅ Phase 4: Added CLOSE_EXECUTED + trade.closed publishing")
else:
    print("⚠️ Phase 4: Marker not found")

# Write result
with open(filepath, "w") as f:
    f.write(content)

print("✅ P0.4C patches applied!")
