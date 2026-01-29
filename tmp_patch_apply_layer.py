#!/usr/bin/env python3
"""Patch apply_layer/main.py to read calibrated action from P2.6 Heat Gate"""

import sys

# Read the file
with open("/home/qt/quantum_trader/microservices/apply_layer/main.py", "r") as f:
    content = f.read()

# Find the target section
target = '''            proposal = {
                "harvest_action": data.get("harvest_action"),'''

replacement = '''            # P0.FIX: Read calibrated action if available (from P2.6 Heat Gate)
            # If calibrated=1, use "action" field (calibrated), else fall back to "harvest_action"
            is_calibrated = data.get("calibrated") == "1"
            if is_calibrated and data.get("action"):
                action = data.get("action")
                logger.debug(f"{symbol}: Using calibrated action={action} (original={data.get('original_action', 'N/A')})")
            else:
                action = data.get("harvest_action")
            
            proposal = {
                "harvest_action": action,  # Use calibrated action if available,'''

# Replace
if target in content:
    content = content.replace(target, replacement)
    print("✓ Replaced harvest_action read logic")
else:
    print("✗ Target not found!")
    sys.exit(1)

# Also add tracking fields after reason_codes
target2 = '''                "reason_codes": data.get("reason_codes", "").split(","),
            }'''

replacement2 = '''                "reason_codes": data.get("reason_codes", "").split(","),
                "p26_calibrated": is_calibrated,  # Track if calibrated
                "p26_original_action": data.get("original_action") if is_calibrated else None,
            }'''

if target2 in content:
    content = content.replace(target2, replacement2)
    print("✓ Added p26_calibrated tracking fields")
else:
    print("✗ Could not add tracking fields")

# Write back
with open("/home/qt/quantum_trader/microservices/apply_layer/main.py", "w") as f:
    f.write(content)

print("✓ Patch applied successfully")
