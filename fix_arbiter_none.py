#!/usr/bin/env python3
"""Fix NoneType bug in Arbiter agent."""

file_path = "/home/qt/quantum_trader/ai_engine/agents/arbiter_agent.py"

with open(file_path, 'r') as f:
    content = f.read()

# Fix 1: indicators can be None even with default
old_indicators = "indicators = market_data.get('indicators', {})"
new_indicators = "indicators = market_data.get('indicators') or {}"

# Fix 2: regime can be None even with default
old_regime = "regime = market_data.get('regime', {})"
new_regime = "regime = market_data.get('regime') or {}"

changes = 0
if old_indicators in content:
    content = content.replace(old_indicators, new_indicators)
    changes += 1
    print(f"✅ Fixed indicators None check")
else:
    if new_indicators in content:
        print("⏭️ indicators already fixed")
    else:
        print("⚠️ Could not find indicators line")

if old_regime in content:
    content = content.replace(old_regime, new_regime)
    changes += 1
    print(f"✅ Fixed regime None check")
else:
    if new_regime in content:
        print("⏭️ regime already fixed")
    else:
        print("⚠️ Could not find regime line")

if changes > 0:
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"\n✅ Applied {changes} fixes to Arbiter")
else:
    print("\n⏭️ No changes needed")
