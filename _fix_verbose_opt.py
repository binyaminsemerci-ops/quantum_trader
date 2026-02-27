#!/usr/bin/env python3
"""
Fix: Enable verbose traceback logging in the /opt/quantum harvest_brain.py
This is needed because the service runs from /opt/quantum, not /home/qt/quantum_trader
"""
import sys

path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(path, "r") as f:
    content = f.read()

old = 'logger.debug(f"Error evaluating position {pos_key}: {e}")'
new = 'logger.error(f"TRACEBACK evaluating {pos_key}: {e}", exc_info=True)'

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("VERBOSE_OK: traceback logging enabled in /opt/quantum service")
else:
    # Find what the handler actually looks like
    idx = content.find("Error evaluating position")
    if idx >= 0:
        snippet = content[max(0, idx-80):idx+150]
        print(f"NOT_FOUND. Handler context:\n{snippet}")
    else:
        print("ERROR: 'Error evaluating position' string not found in file at all")
    sys.exit(1)
