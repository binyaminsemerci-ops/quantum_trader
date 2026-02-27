#!/usr/bin/env python3
"""
Fix position_provider.py in harvest_v2:
Change the _validate method to accept positions where risk_missing=1
but entry_risk_usdt > 0 (retrospectively patched risk is still valid
for shadow/research evaluation purposes).
"""

import re

path = "/opt/quantum/microservices/harvest_v2/feeds/position_provider.py"

with open(path, "r") as f:
    content = f.read()

old = "        if pos.entry_risk_usdt <= 0 or pos.risk_missing == 1:\n            return \"INVALID_RISK\""
new = "        # Accept risk_missing=1 if entry_risk_usdt > 0 (retrospectively patched)\n        if pos.entry_risk_usdt <= 0:\n            return \"INVALID_RISK\""

if old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("PATCHED: position_provider.py risk_missing check relaxed for shadow mode")
else:
    print("ERROR: target string not found - check manually")
    # Show context
    for i, line in enumerate(content.splitlines(), 1):
        if "INVALID_RISK" in line or "risk_missing" in line:
            print(f"  Line {i}: {line}")
