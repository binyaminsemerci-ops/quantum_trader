#!/usr/bin/env python3
"""Enable Meta-Strategy Selector with Q-learning"""

file_path = '/home/qt/quantum_trader/microservices/ai_engine/service.py'

with open(file_path, 'r') as f:
    content = f.read()

# Enable Meta-Strategy
old = "if False and self.meta_strategy_selector:  # Disabled temporarily"
new = "if self.meta_strategy_selector:  # ✅ ENABLED: Q-learning strategy selection"

if old in content:
    content = content.replace(old, new)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Meta-Strategy Selector ENABLED")
else:
    if "if self.meta_strategy_selector:  # ✅ ENABLED" in content:
        print("✅ Already enabled")
    else:
        print("❌ Could not find target line")

# Verify
with open(file_path, 'r') as f:
    content = f.read()
if "if self.meta_strategy_selector:  # ✅ ENABLED" in content:
    print("✅ Verified: Meta-Strategy is now ENABLED")
else:
    print("⚠️ Verification failed")

print("Done!")
