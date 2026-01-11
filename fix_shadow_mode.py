#!/usr/bin/env python3
"""
QSC MODE FIX: Disable PatchTST shadow mode for canary deployment
"""
import re

file_path = 'ai_engine/agents/patchtst_agent.py'

with open(file_path, 'r') as f:
    content = f.read()

# Fix 1: Disable shadow mode
old_shadow = "shadow_mode = os.getenv('PATCHTST_SHADOW_ONLY', 'false').lower() == 'true'"
new_shadow = "shadow_mode = False  # QSC MODE: All models must vote"

if old_shadow in content:
    content = content.replace(old_shadow, new_shadow)
    print("✅ PatchTST shadow mode DISABLED")
else:
    print("⚠️  Shadow mode line not found or already modified")

with open(file_path, 'w') as f:
    f.write(content)

print(f"✅ File updated: {file_path}")
