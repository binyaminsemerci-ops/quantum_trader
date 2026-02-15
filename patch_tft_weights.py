#!/usr/bin/env python3
"""Patch to ensure TFT weight is included when using calibrated weights"""
import re

file_path = '/home/qt/quantum_trader/ai_engine/ensemble_manager.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find and patch _load_dynamic_weights to add tft if missing
old_code = '''if calibrated_weights:
                logger.info(f"[Calibration] ✅ Using calibrated weights: {calibrated_weights}")
                return calibrated_weights'''

new_code = '''if calibrated_weights:
                # Ensure TFT weight is included (may be missing from older calibrations)
                if 'tft' not in calibrated_weights and hasattr(self, 'tft_agent') and self.tft_agent is not None:
                    # Redistribute weights to include TFT at 20%
                    calibrated_weights = {k: v * 0.8 for k, v in calibrated_weights.items()}
                    calibrated_weights['tft'] = 0.20
                    logger.info(f"[Calibration] 🔧 Added TFT weight (20%), redistributed: {calibrated_weights}")
                else:
                    logger.info(f"[Calibration] ✅ Using calibrated weights: {calibrated_weights}")
                return calibrated_weights'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Patched _load_dynamic_weights to include TFT")
else:
    # Check if already patched
    if 'Added TFT weight' in content:
        print("✅ Already patched")
    else:
        print("❌ Could not find target code block")

# Also verify TFT weight at line 1076 - use .get() instead of direct access
old_aggregate = "weight = self.weights[model_name]"
new_aggregate = "weight = self.weights.get(model_name, 0.20)  # Default 20% for new models"

if old_aggregate in content:
    content = content.replace(old_aggregate, new_aggregate)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Made weight access resilient with .get()")
elif "weight = self.weights.get(model_name" in content:
    print("✅ Weight access already resilient")
else:
    print("⚠️ Could not find weight access line")

print("Done!")
