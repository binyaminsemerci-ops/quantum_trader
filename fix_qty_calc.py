#!/usr/bin/env python3
"""Fix HarvestBrain to calculate qty from position_size_usd / entry_price"""
import sys

file_path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''            # Extract fill details
            qty = float(exec_event.get('qty', 0))
            price = float(exec_event.get('price', 0))
            entry_price = float(exec_event.get('entry_price', price))
            stop_loss = float(exec_event.get('stop_loss', 0))
            take_profit = float(exec_event.get('take_profit', 0))

            if qty <= 0:
                logger.debug(f"Skipping execution: qty={qty}")
                return False'''

new_code = '''            # Extract fill details
            qty = float(exec_event.get('qty', 0))
            price = float(exec_event.get('price', 0))
            entry_price = float(exec_event.get('entry_price', price))
            
            # If qty not provided, calculate from position_size_usd / entry_price
            if qty == 0 and entry_price > 0:
                position_size_usd = float(exec_event.get('position_size_usd', 0))
                if position_size_usd > 0:
                    qty = position_size_usd / entry_price
                    logger.debug(f"🔍 Calculated qty={qty:.4f} from position_size_usd={position_size_usd:.2f} / entry_price={entry_price:.4f}")
            
            stop_loss = float(exec_event.get('stop_loss', 0))
            take_profit = float(exec_event.get('take_profit', 0))

            if qty <= 0:
                logger.debug(f"Skipping execution: qty={qty}")
                return False'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed HarvestBrain to calculate qty from position_size_usd")
    sys.exit(0)
else:
    print("❌ Could not find code to replace")
    sys.exit(1)
