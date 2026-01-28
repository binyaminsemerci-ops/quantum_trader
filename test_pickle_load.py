#!/usr/bin/env python3
"""Test loading nhits/patchtst pickle files with proper imports."""

import sys
sys.path.insert(0, '/home/qt/quantum_trader')

# CRITICAL: Import Dummy classes BEFORE loading pickles
from ai_engine.agents.unified_agents import DummyNHiTS, DummyPatchTST

import joblib
import os

model_dir = '/home/qt/quantum_trader/models'

# Find nhits model
nhits_files = [f for f in os.listdir(model_dir) if 'nhits' in f and f.endswith('.pkl') and 'scaler' not in f]
patchtst_files = [f for f in os.listdir(model_dir) if 'patchtst' in f and f.endswith('.pkl') and 'scaler' not in f]

print("="*80)
print("TESTING PICKLE LOADING WITH DUMMY CLASSES")
print("="*80)

if nhits_files:
    nhits_path = os.path.join(model_dir, nhits_files[0])
    print(f"\nLoading N-HiTS from: {nhits_path}")
    try:
        model = joblib.load(nhits_path)
        print(f"✅ N-HiTS loaded successfully: {type(model)}")
        print(f"   Test prediction: {model.predict([[1]*10])}")
    except Exception as e:
        print(f"❌ N-HiTS load failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ No N-HiTS model files found")

if patchtst_files:
    patchtst_path = os.path.join(model_dir, patchtst_files[0])
    print(f"\nLoading PatchTST from: {patchtst_path}")
    try:
        model = joblib.load(patchtst_path)
        print(f"✅ PatchTST loaded successfully: {type(model)}")
        print(f"   Test prediction: {model.predict([[1]*10])}")
    except Exception as e:
        print(f"❌ PatchTST load failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ No PatchTST model files found")
