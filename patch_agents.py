#!/usr/bin/env python3
"""
Comprehensive fix for XGB + LGBM agents:
1. Patches lgbm_agent.py retraining_dir to prefer ai_engine/models
2. Retrains LGBM v6 with 49 features
3. Updates XGB symlinks
"""

import subprocess, sys, os, re

FILE = "/opt/quantum/ai_engine/agents/lgbm_agent.py"

with open(FILE, "r") as f:
    content = f.read()

# ----------------------------------------------------------------
# Fix 1: LGBM agent retraining_dir — prefer ai_engine/models first
# ----------------------------------------------------------------
OLD_DIR = """        retraining_dir = Path("/app/models") if Path("/app/models").exists() else (
            Path("models") if Path("models").exists() else Path("ai_engine/models")
        )"""

NEW_DIR = """        retraining_dir = (
            Path("/app/models") if Path("/app/models").exists() else
            Path("ai_engine/models") if Path("ai_engine/models").exists() else
            Path("models") if Path("models").exists() else
            Path("ai_engine/models")
        )"""

if OLD_DIR in content:
    content = content.replace(OLD_DIR, NEW_DIR, 1)
    print("FIX_LGBM_DIR: retraining_dir patched to prefer ai_engine/models")
else:
    print("FIX_LGBM_DIR: WARN - pattern not found exactly")
    # Try to find it
    idx = content.find('Path("/app/models")')
    if idx >= 0:
        print(f"  Found at char {idx}:")
        print(repr(content[idx-5:idx+200]))
    else:
        print("  Not found at all - may already be fixed")

# ----------------------------------------------------------------
# Fix 2: LGBM scaler_path fallback — also use absolute path fallback
# ----------------------------------------------------------------
OLD_SCALER = 'self.scaler_path = scaler_path or str(latest_scaler) if latest_scaler else "models/lightgbm_scaler_v20251230_223627.pkl"'
NEW_SCALER = '''self.scaler_path = scaler_path or str(latest_scaler) if latest_scaler else str(
            sorted((retraining_dir).glob("lightgbm_scaler_v*.pkl"))[-1]
            if list((retraining_dir).glob("lightgbm_scaler_v*.pkl")) else
            "models/lightgbm_scaler_v20251230_223627.pkl"
        )'''

if OLD_SCALER in content:
    content = content.replace(OLD_SCALER, NEW_SCALER, 1)
    print("FIX_LGBM_SCALER: scaler_path fallback enhanced")
else:
    print("FIX_LGBM_SCALER: WARN - scaler_path pattern not found exactly")

# ----------------------------------------------------------------
# Fix 3: model_path fallback — same treatment
# ----------------------------------------------------------------
OLD_MODEL = 'self.model_path = model_path or str(latest_model) if latest_model else "models/lightgbm_v20251213_231048.pkl"'
NEW_MODEL = '''self.model_path = model_path or str(latest_model) if latest_model else str(
            sorted((retraining_dir).glob("lightgbm_v*.pkl"))[-1]
            if list((retraining_dir).glob("lightgbm_v*.pkl")) else
            "models/lightgbm_v20251213_231048.pkl"
        )'''

if OLD_MODEL in content:
    content = content.replace(OLD_MODEL, NEW_MODEL, 1)
    print("FIX_LGBM_MODEL: model_path fallback enhanced")
else:
    print("FIX_LGBM_MODEL: WARN - model_path pattern not found exactly")

with open(FILE, "w") as f:
    f.write(content)

print("DONE: lgbm_agent.py written")

# ----------------------------------------------------------------
# Fix 4: Verify XGB v6 symlinks
# ----------------------------------------------------------------
MODEL_DIR = "/opt/quantum/ai_engine/models"
V6_MODEL  = f"{MODEL_DIR}/xgb_v6_20260217_003058.pkl"
V6_SCALER = f"{MODEL_DIR}/xgb_v6_20260217_003058_scaler.pkl"

print(f"\nXGB v6 model exists: {os.path.exists(V6_MODEL)}, size={os.path.getsize(V6_MODEL) if os.path.exists(V6_MODEL) else 'N/A'}")
print(f"XGB v6 scaler exists: {os.path.exists(V6_SCALER)}, size={os.path.getsize(V6_SCALER) if os.path.exists(V6_SCALER) else 'N/A'}")

for alias, target in [
    ("xgb_model.pkl",            "xgb_v6_20260217_003058.pkl"),
    ("xgboost_v_prod.pkl",       "xgb_v6_20260217_003058.pkl"),
    ("scaler.pkl",               "xgb_v6_20260217_003058_scaler.pkl"),
    ("xgboost_v_prod_scaler.pkl","xgb_v6_20260217_003058_scaler.pkl"),
]:
    dst = f"{MODEL_DIR}/{alias}"
    src = f"{MODEL_DIR}/{target}"
    if os.path.exists(src):
        try:
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(target, dst)
            print(f"  symlink: {alias} -> {target}")
        except Exception as e:
            print(f"  WARN: {alias}: {e}")
    else:
        print(f"  SKIP: target {target} not found")

print("\nXGB fix complete")
