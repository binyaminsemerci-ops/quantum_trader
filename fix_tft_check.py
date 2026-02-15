#!/usr/bin/env python3
"""Fix TFT weight by simplify check - always add if missing"""
import re

file_path = '/home/qt/quantum_trader/ai_engine/ensemble_manager.py'

with open(file_path, 'r') as f:
    content = f.read()

# Replace the complex check with a simple one
old = "if 'tft' not in calibrated_weights and hasattr(self, 'tft_agent') and self.tft_agent is not None:"
new = "if 'tft' not in calibrated_weights:  # TFT is 5th model - always include"

if old in content:
    content = content.replace(old, new)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Simplified TFT weight check")
elif "'tft' not in calibrated_weights:  # TFT" in content:
    print("✅ Already simplified")
else:
    # Try another pattern
    old2 = "if 'tft' in self.enabled_models"
    if old2 in content:
        content = content.replace(old2, "if True  # Always add TFT")
        with open(file_path, 'w') as f:
            f.write(content)
        print("✅ Fixed enabled_models check")
    else:
        print("⚠️ Looking for pattern...")
        # Find and show the relevant section
        idx = content.find("# Ensure TFT weight")
        if idx > 0:
            print(content[idx:idx+300])

print("Done!")
