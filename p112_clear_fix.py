#!/usr/bin/env python3
"""P1.1.2 Micro-hotfix: Add .clear() to rank gauge to prevent stale labels"""

with open("/home/qt/quantum_trader/microservices/safety_telemetry/main.py", "r") as f:
    content = f.read()

# Find and replace the rank gauge section
old_code = '''                # P1.1: Set individual rank gauges
                for i, (symbol, count) in enumerate(top5):'''

new_code = '''                # P1.1.2: Clear stale labels before setting ranks
                safety_rate_symbol_rank_gauge.clear()
                
                # P1.1: Set individual rank gauges
                for i, (symbol, count) in enumerate(top5):'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ Added safety_rate_symbol_rank_gauge.clear()")
else:
    print("⚠️ Could not find exact pattern, trying alternative...")
    # Try alternative pattern
    old_alt = "                # P1.1: Set individual rank gauges\n                for i, (symbol, count) in enumerate(top5):"
    new_alt = "                # P1.1.2: Clear stale labels before setting ranks\n                safety_rate_symbol_rank_gauge.clear()\n                \n                # P1.1: Set individual rank gauges\n                for i, (symbol, count) in enumerate(top5):"
    
    if old_alt in content:
        content = content.replace(old_alt, new_alt)
        print("✅ Added safety_rate_symbol_rank_gauge.clear() (alt pattern)")
    else:
        print("❌ Pattern not found")

with open("/home/qt/quantum_trader/microservices/safety_telemetry/main.py", "w") as f:
    f.write(content)

print("✅ P1.1.2 micro-hotfix applied")
