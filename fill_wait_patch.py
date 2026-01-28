#!/usr/bin/env python3
"""
P0 Fix: Add fill-wait polling to execution_service.py
This ensures orders are actually FILLED before publishing execution results.
"""

def apply_fill_wait_patch():
    filepath = "/home/qt/quantum_trader/services/execution_service.py"
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Step 1: Find insertion point for wait_for_fill function (before first def)
    insert_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def round_quantity"):
            insert_line = i
            break
    
    if insert_line is None:
        print("❌ Could not find insertion point")
        return False
    
    # Step 2: Add wait_for_fill function
    wait_for_fill_code = [
        "\n",
        "async def wait_for_fill(binance_client, symbol: str, order_id: str, max_wait_seconds: int = 20) -> dict:\n",
        "    \"\"\"\n",
        "    Poll Binance API until order is filled or timeout.\n",
        "    Returns the filled order details when status=FILLED with non-zero executedQty and avgPrice.\n",
        "    Raises TimeoutError if order not filled within max_wait_seconds.\n",
        "    \"\"\"\n",
        "    import time as time_module\n",
        "    \n",
        "    start_time = time_module.time()\n",
        "    poll_interval = 0.5  # 500ms between polls\n",
        "    \n",
        "    logger.info(f\"⏳ WAITING_FOR_FILL orderId={order_id} symbol={symbol} (max {max_wait_seconds}s)\")\n",
        "    \n",
        "    while time_module.time() - start_time < max_wait_seconds:\n",
        "        try:\n",
        "            order_status = binance_client.futures_get_order(\n",
        "                symbol=symbol,\n",
        "                orderId=int(order_id)\n",
        "            )\n",
        "            \n",
        "            status = order_status.get(\"status\")\n",
        "            executed_qty = float(order_status.get(\"executedQty\", 0))\n",
        "            avg_price = float(order_status.get(\"avgPrice\", 0))\n",
        "            elapsed = time_module.time() - start_time\n",
        "            \n",
        "            logger.info(\n",
        "                f\"[FILL_POLL] orderId={order_id} status={status} \"\n",
        "                f\"executedQty={executed_qty} avgPrice={avg_price:.4f} \"\n",
        "                f\"elapsed={elapsed:.1f}s\"\n",
        "            )\n",
        "            \n",
        "            # Check for fill completion\n",
        "            if status == \"FILLED\" and executed_qty > 0 and avg_price > 0:\n",
        "                logger.info(\n",
        "                    f\"✅ FILL_CONFIRMED orderId={order_id} \"\n",
        "                    f\"avgPrice={avg_price:.4f} executedQty={executed_qty} \"\n",
        "                    f\"after {elapsed:.1f}s\"\n",
        "                )\n",
        "                return order_status\n",
        "            \n",
        "            # Check for failure states\n",
        "            elif status in [\"CANCELED\", \"EXPIRED\", \"REJECTED\"]:\n",
        "                raise Exception(f\"Order {order_id} failed with status: {status}\")\n",
        "            \n",
        "            # Still NEW or PARTIALLY_FILLED - keep polling\n",
        "            await asyncio.sleep(poll_interval)\n",
        "            \n",
        "        except Exception as e:\n",
        "            if \"failed with status\" in str(e):\n",
        "                raise  # Re-raise order failure\n",
        "            logger.warning(f\"[FILL_POLL] Error checking order {order_id}: {e}\")\n",
        "            await asyncio.sleep(poll_interval)\n",
        "    \n",
        "    # Timeout reached\n",
        "    raise TimeoutError(f\"Order {order_id} not filled within {max_wait_seconds}s\")\n",
        "\n",
        "\n",
    ]
    
    lines[insert_line:insert_line] = wait_for_fill_code
    print(f"✅ Added wait_for_fill function before line {insert_line}")
    
    # Step 3: Find and replace execution logic
    # Look for the pattern: order_id = str(market_order["orderId"])
    execution_start = None
    for i, line in enumerate(lines):
        if 'order_id = str(market_order["orderId"])' in line:
            execution_start = i
            break
    
    if execution_start is None:
        print("❌ Could not find execution logic to replace")
        with open(filepath, "w") as f:
            f.writelines(lines)
        return False
    
    # Find the end of the old block (up to "# NOTE: TP/SL are NOT placed")
    execution_end = None
    for i in range(execution_start, min(execution_start + 20, len(lines))):
        if "# NOTE: TP/SL are NOT placed" in lines[i]:
            execution_end = i
            break
    
    if execution_end is None:
        print("❌ Could not find end of execution block")
        with open(filepath, "w") as f:
            f.writelines(lines)
        return False
    
    # Replace lines from execution_start to execution_end with new logic
    new_execution_logic = [
        "            order_id = str(market_order[\"orderId\"])\n",
        "            \n",
        "            # P0 FIX: Check if order needs fill confirmation\n",
        "            initial_status = market_order.get(\"status\")\n",
        "            initial_qty = float(market_order.get(\"executedQty\", 0))\n",
        "            initial_price = float(market_order.get(\"avgPrice\", 0))\n",
        "            \n",
        "            # If order not immediately filled, wait for fill confirmation\n",
        "            if initial_status != \"FILLED\" or initial_qty == 0 or initial_price == 0:\n",
        "                logger.info(\n",
        "                    f\"⏳ Order placed but not immediately filled: orderId={order_id} \"\n",
        "                    f\"status={initial_status} executedQty={initial_qty} avgPrice={initial_price}\"\n",
        "                )\n",
        "                \n",
        "                try:\n",
        "                    # Poll for fill confirmation (max 20s)\n",
        "                    filled_order = await wait_for_fill(binance_client, intent.symbol, order_id, max_wait_seconds=20)\n",
        "                    execution_price = float(filled_order[\"avgPrice\"])\n",
        "                    actual_qty = float(filled_order[\"executedQty\"])\n",
        "                    \n",
        "                    logger.info(\n",
        "                        f\"✅ BINANCE MARKET ORDER FILLED: {intent.symbol} {intent.side} | \"\n",
        "                        f\"OrderID={order_id} | \"\n",
        "                        f\"Price=${execution_price:.4f} | \"\n",
        "                        f\"Qty={actual_qty}\"\n",
        "                    )\n",
        "                    \n",
        "                except TimeoutError as e:\n",
        "                    logger.error(f\"❌ Fill timeout: {e}\")\n",
        "                    # Publish pending_fill status (not filled)\n",
        "                    result = ExecutionResult(\n",
        "                        symbol=intent.symbol,\n",
        "                        action=intent.side,\n",
        "                        entry_price=0.0,  # Not filled\n",
        "                        position_size_usd=0.0,  # Not filled\n",
        "                        leverage=intent.leverage,\n",
        "                        timestamp=datetime.utcnow().isoformat() + \"Z\",\n",
        "                        order_id=order_id,\n",
        "                        status=\"pending_fill\",  # Timeout status\n",
        "                        slippage_pct=0.0,\n",
        "                        fee_usd=0.0\n",
        "                    )\n",
        "                    await eventbus.publish_execution(result)\n",
        "                    logger.info(f\"⚠️  TERMINAL STATE: PENDING_FILL | {intent.symbol} | orderId={order_id} | trace_id={trace_id}\")\n",
        "                    return\n",
        "                    \n",
        "                except Exception as e:\n",
        "                    logger.error(f\"❌ Fill confirmation failed: {e}\")\n",
        "                    # Publish rejected status\n",
        "                    result = ExecutionResult(\n",
        "                        symbol=intent.symbol,\n",
        "                        action=intent.side,\n",
        "                        entry_price=intent.entry_price,\n",
        "                        position_size_usd=0.0,\n",
        "                        leverage=intent.leverage,\n",
        "                        timestamp=datetime.utcnow().isoformat() + \"Z\",\n",
        "                        order_id=order_id,\n",
        "                        status=\"rejected\",\n",
        "                        slippage_pct=0.0,\n",
        "                        fee_usd=0.0\n",
        "                    )\n",
        "                    await eventbus.publish_execution(result)\n",
        "                    logger.info(f\"🚫 TERMINAL STATE: REJECTED | {intent.symbol} | Reason: {e} | trace_id={trace_id}\")\n",
        "                    return\n",
        "            else:\n",
        "                # Order filled immediately\n",
        "                execution_price = initial_price\n",
        "                actual_qty = initial_qty\n",
        "                \n",
        "                logger.info(\n",
        "                    f\"✅ BINANCE MARKET ORDER FILLED (immediate): {intent.symbol} {intent.side} | \"\n",
        "                    f\"OrderID={order_id} | \"\n",
        "                    f\"Price=${execution_price:.4f} | \"\n",
        "                    f\"Qty={actual_qty}\"\n",
        "                )\n",
        "\n",
    ]
    
    lines[execution_start:execution_end] = new_execution_logic
    print(f"✅ Replaced execution logic (lines {execution_start}-{execution_end})")
    
    # Write back
    with open(filepath, "w") as f:
        f.writelines(lines)
    
    print("✅ Fill-wait patch applied successfully")
    return True

if __name__ == "__main__":
    try:
        success = apply_fill_wait_patch()
        if success:
            print("\n✅ PATCH COMPLETE - Ready for testing")
        else:
            print("\n⚠️  PARTIAL PATCH - Manual review needed")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
