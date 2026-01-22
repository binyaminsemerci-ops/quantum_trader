#!/usr/bin/env python3
"""Fix execution_service.py to query order status after placement"""
import sys

file_path = "/home/qt/quantum_trader/services/execution_service.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the lines to replace
old_code = '''        order_id = str(market_order["orderId"])
        execution_price = float(market_order["avgPrice"]) if "avgPrice" in market_order else intent.entry_price
        actual_qty = float(market_order["executedQty"])'''

new_code = '''        order_id = str(market_order["orderId"])
        
        # CRITICAL FIX: Market orders may return status=NEW before filling
        # Query the order again to get actual fill data
        import time
        time.sleep(0.5)  # Wait 500ms for fill
        filled_order = binance_client.futures_get_order(symbol=intent.symbol, orderId=order_id)
        
        execution_price = float(filled_order.get("avgPrice", 0.0))
        actual_qty = float(filled_order.get("executedQty", 0.0))
        
        logger.info(
            f"🔍 After query: status={filled_order.get('status')}, "
            f"avgPrice={execution_price:.4f}, executedQty={actual_qty}"
        )
        
        if execution_price == 0.0 or actual_qty == 0.0:
            execution_price = intent.entry_price  # Fallback
        if actual_qty == 0.0:
            actual_qty = intent.position_size_usd / execution_price  # Fallback'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed execution_service.py")
    sys.exit(0)
else:
    print("❌ Could not find code to replace")
    sys.exit(1)
