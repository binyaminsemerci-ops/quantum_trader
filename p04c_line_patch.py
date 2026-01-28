#!/usr/bin/env python3
import re

fp = "/home/qt/quantum_trader/services/execution_service.py"
with open(fp, "r", encoding="utf-8") as f:
    lines = f.readlines()

# PHASE 3: Add reduceOnly via kwargs
# Find line with "market_order = binance_client.futures_create_order("
for i, line in enumerate(lines):
    if "market_order = binance_client.futures_create_order(" in line:
        # Find the closing paren
        j = i
        while ")" not in lines[j] or "symbol=" in lines[j]:
            j += 1
        # Now j is at the closing paren line
        # Replace the block
        indent = " " * 12
        new_block = [
            f"{indent}market_kwargs = dict(\n",
            f"{indent}    symbol=intent.symbol,\n",
            f"{indent}    side=side_binance,\n",
            f'{indent}    type="MARKET",\n',
            f"{indent}    quantity=quantity,\n",
            f"{indent})\n",
            f"{indent}if getattr(intent, \"reduce_only\", False):\n",
            f'{indent}    market_kwargs["reduceOnly"] = True\n',
            f"\n",
            f"{indent}market_order = binance_client.futures_create_order(**market_kwargs)\n",
        ]
        # Replace lines i through j
        lines[i:j+1] = new_block
        print(f"✅ Phase 3: Replaced lines {i+1}-{j+1} with reduceOnly kwargs logic")
        break

# PHASE 4: Insert after "TERMINAL STATE: FILLED" logger block
# Find line 995 (the line after ")")
for i, line in enumerate(lines):
    if "TERMINAL STATE: FILLED" in line:
        # Find the closing paren of this logger.info
        j = i
        while True:
            if ")" in lines[j] and "trace_id" in lines[j]:
                j += 1  # Line after the closing paren
                break
            j += 1
        
        # Insert the new code block
        indent = " " * 12
        insert_lines = [
            "\n",
            f"{indent}# P0.4C: close-order proof (reduceOnly chain)\n",
            f"{indent}if getattr(intent, \"reduce_only\", False):\n",
            f'{indent}    src = getattr(intent, "source", None) or "unknown"\n',
            f'{indent}    rsn = getattr(intent, "reason", None) or "unknown"\n',
            f"{indent}    side_val = getattr(intent, \"side\", None) or getattr(intent, \"action\", None) or \"UNKNOWN\"\n",
            f"\n",
            f"{indent}    logger.info(\n",
            f'{indent}        "✅ CLOSE_EXECUTED symbol=%s side=%s reduceOnly=True source=%s reason=%s qty=%s exit_price=%.6f order_id=%s"\n',
            f"{indent}        % (intent.symbol, side_val, src, rsn, str(actual_qty), float(execution_price), str(order_id))\n",
            f"{indent}    )\n",
            f"\n",
            f'{indent}    # Optional: publish trade.closed ONLY if explicitly enabled\n',
            f'{indent}    if os.getenv("P04C_PUBLISH_TRADE_CLOSED", "false").lower() == "true":\n',
            f"{indent}        try:\n",
            f"{indent}            closed_event = {{\n",
            f'{indent}                "symbol": intent.symbol,\n',
            f'{indent}                "side": side_val,\n',
            f'{indent}                "quantity": actual_qty,\n',
            f'{indent}                "exit_price": execution_price,\n',
            f'{indent}                "timestamp": datetime.utcnow().isoformat() + "Z",\n',
            f'{indent}                "source": src,\n',
            f'{indent}                "reason": rsn,\n',
            f'{indent}                "exit_order_id": order_id,\n',
            f'{indent}                "status": "closed",\n',
            f'{indent}                "fee_usd": fee_usd,\n',
            f"{indent}            }}\n",
            f'{indent}            await eventbus.publish(stream="quantum:stream:trade.closed", data=closed_event)\n',
            f"{indent}            logger.info(\n",
            f'{indent}                "📣 CLOSED_PUBLISHED symbol=%s source=%s reason=%s exit_order_id=%s"\n',
            f"{indent}                % (intent.symbol, src, rsn, str(order_id))\n",
            f"{indent}            )\n",
            f"{indent}        except Exception as e:\n",
            f'{indent}            logger.error("❌ Failed to publish trade.closed: %s" % (str(e),))\n',
            "\n",
        ]
        
        # Insert at position j
        lines[j:j] = insert_lines
        print(f"✅ Phase 4: Inserted CLOSE_EXECUTED block after line {j}")
        break

with open(fp, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Patch complete!")
