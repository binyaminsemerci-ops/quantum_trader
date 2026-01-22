#!/usr/bin/env python3
"""Fix HarvestBrain to read 'action' field instead of 'side'"""
import sys

file_path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''            else:
                # Flat dict format (direct from stream)
                symbol = exec_event.get('symbol', '').strip()
                side = exec_event.get('side', '').strip()
                status = exec_event.get('status', '').upper()'''

new_code = '''            else:
                # Flat dict format (direct from stream)
                symbol = exec_event.get('symbol', '').strip()
                side = exec_event.get('side', '') or exec_event.get('action', '')  # Try 'side' first, fall back to 'action'
                side = side.strip()
                status = exec_event.get('status', '').upper()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Fixed HarvestBrain to read 'action' field")
    sys.exit(0)
else:
    print("❌ Could not find code to replace")
    sys.exit(1)
