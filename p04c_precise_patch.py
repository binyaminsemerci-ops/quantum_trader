#!/usr/bin/env python3
import re

fp = "/home/qt/quantum_trader/services/execution_service.py"
with open(fp, "r", encoding="utf-8") as f:
    lines = f.readlines()

# PHASE 3: Add reduceOnly via kwargs
for i, line in enumerate(lines):
    if "market_order = binance_client.futures_create_order(" in line:
        # Find the closing paren (should be within 10 lines)
        j = i
        while j < len(lines) and j < i + 10:
            if lines[j].strip() == ")":
                break
            j += 1
        
        if j >= len(lines) or j >= i + 10:
            print(f"⚠️ Could not find closing paren for futures_create_order at line {i+1}")
            break
        
        # Replace lines i through j (inclusive)
        indent = " " * 12
        new_block = [
            f"{indent}market_kwargs = dict(\n",
            f"{indent}    symbol=intent.symbol,\n",
            f"{indent}    side=side_binance,\n",
            f'{indent}    type="MARKET",\n',
            f"{indent}    quantity=quantity,\n",
            f"{indent})\n",
            f"{indent}if getattr(intent, 'reduce_only', False):\n",
            f'{indent}    market_kwargs["reduceOnly"] = True\n',
            f"\n",
            f"{indent}market_order = binance_client.futures_create_order(**market_kwargs)\n",
        ]
        lines[i:j+1] = new_block
        print(f"✅ Phase 3: Replaced lines {i+1}-{j+1} with reduceOnly kwargs")
        break

# PHASE 4: Insert before "# 7. Update stats"
for i, line in enumerate(lines):
    if line.strip() == "# 7. Update stats":
        # Insert before this line
        indent = " " * 12
        insert_lines = [
            f"\n",
            f"{indent}# P0.4C: close-order proof (reduceOnly chain)\n",
            f"{indent}if getattr(intent, 'reduce_only', False):\n",
            f"{indent}    src = getattr(intent, 'source', None) or 'unknown'\n",
            f"{indent}    rsn = getattr(intent, 'reason', None) or 'unknown'\n",
            f"{indent}    side_val = getattr(intent, 'side', None) or getattr(intent, 'action', None) or 'UNKNOWN'\n",
            f"\n",
            f"{indent}    logger.info(\n",
            f"{indent}        '✅ CLOSE_EXECUTED symbol=%s side=%s reduceOnly=True source=%s reason=%s qty=%s exit_price=%.6f order_id=%s'\n",
            f"{indent}        % (intent.symbol, side_val, src, rsn, str(actual_qty), float(execution_price), str(order_id))\n",
            f"{indent}    )\n",
            f"\n",
            f"{indent}    # Optional: publish trade.closed ONLY if explicitly enabled\n",
            f"{indent}    if os.getenv('P04C_PUBLISH_TRADE_CLOSED', 'false').lower() == 'true':\n",
            f"{indent}        try:\n",
            f"{indent}            closed_event = {{\n",
            f"{indent}                'symbol': intent.symbol,\n",
            f"{indent}                'side': side_val,\n",
            f"{indent}                'quantity': actual_qty,\n",
            f"{indent}                'exit_price': execution_price,\n",
            f"{indent}                'timestamp': datetime.utcnow().isoformat() + 'Z',\n",
            f"{indent}                'source': src,\n",
            f"{indent}                'reason': rsn,\n",
            f"{indent}                'exit_order_id': order_id,\n",
            f"{indent}                'status': 'closed',\n",
            f"{indent}                'fee_usd': fee_usd,\n",
            f"{indent}            }}\n",
            f"{indent}            await eventbus.publish(stream='quantum:stream:trade.closed', data=closed_event)\n",
            f"{indent}            logger.info(\n",
            f"{indent}                '📣 CLOSED_PUBLISHED symbol=%s source=%s reason=%s exit_order_id=%s'\n",
            f"{indent}                % (intent.symbol, src, rsn, str(order_id))\n",
            f"{indent}            )\n",
            f"{indent}        except Exception as e:\n",
            f"{indent}            logger.error('❌ Failed to publish trade.closed: %s' % (str(e),))\n",
            f"\n",
        ]
        
        lines[i:i] = insert_lines
        print(f"✅ Phase 4: Inserted CLOSE_EXECUTED block before line {i+1}")
        break

with open(fp, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("✅ Patch complete!")
