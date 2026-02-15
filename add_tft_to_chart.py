#!/usr/bin/env python3
"""Add TFT to CHART output and base_predictions"""

file_path = "/home/qt/quantum_trader/ai_engine/ensemble_manager.py"

with open(file_path, "r") as f:
    content = f.read()

changes = 0

# Fix 1: Line 713 - add tft to base_predictions loop
old1 = "for model_key in ['xgb', 'lgbm', 'nhits', 'patchtst']:"
new1 = "for model_key in ['xgb', 'lgbm', 'nhits', 'patchtst', 'tft']:"
if old1 in content:
    content = content.replace(old1, new1)
    changes += 1
    print(f"✅ Fixed line 713: Added tft to base_predictions loop")

# Fix 2: Line 952 - add TFT to model_abbrev dict
old2 = "model_abbrev = {'xgb': 'XGB', 'lgbm': 'LGBM', 'nhits': 'NH', 'patchtst': 'PT'}[model_key]"
new2 = "model_abbrev = {'xgb': 'XGB', 'lgbm': 'LGBM', 'nhits': 'NH', 'patchtst': 'PT', 'tft': 'TFT'}[model_key]"
if old2 in content:
    content = content.replace(old2, new2)
    changes += 1
    print(f"✅ Fixed line 952: Added TFT to model_abbrev")

if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} TFT-related fixes")
else:
    print("⚠️ No changes needed")
