#!/usr/bin/env python3
"""Revert verbose diagnostic logging back to debug level in /opt/quantum service."""
import sys

path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(path) as f:
    content = f.read()

old = 'logger.error(f"TRACEBACK evaluating {pos_key}: {e}", exc_info=True)'
new = 'logger.debug(f"Error evaluating position {pos_key}: {e}")'

if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("REVERTED: verbose logging back to debug level")
elif new in content:
    print("ALREADY_REVERTED: debug level already in place")
else:
    print("WARNING: neither form found — check manually")
    sys.exit(1)
