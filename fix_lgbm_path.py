#!/usr/bin/env python3
"""
QSC MODE FIX: Correct LGBM model path to use latest trained model
"""
import glob
from pathlib import Path

file_path = 'ai_engine/agents/lgbm_agent.py'

# Find latest LGBM model
models = sorted(glob.glob('models/lightgbm_v*.pkl'), reverse=True)
if not models:
    print("❌ No LGBM model files found in models/")
    exit(1)

latest_model = models[0]
print(f"📦 Latest LGBM model: {latest_model}")

with open(file_path, 'r') as f:
    content = f.read()

# Fix model path
old_path = 'self.model_path = model_path or str(latest_model) if latest_model else "ai_engine/models/lgbm_model.pkl"'
new_path = f'self.model_path = model_path or str(latest_model) if latest_model else "{latest_model}"'

if old_path in content:
    content = content.replace(old_path, new_path)
    print("✅ LGBM model path CORRECTED")
else:
    print("⚠️  Model path line not found or already modified")

with open(file_path, 'w') as f:
    f.write(content)

print(f"✅ File updated: {file_path}")
print(f"✅ New default model path: {latest_model}")
