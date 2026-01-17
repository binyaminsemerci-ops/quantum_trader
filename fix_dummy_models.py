#!/usr/bin/env python3
"""Re-save Dummy models with correct module path for pickle."""

import sys
import os
import joblib

# Set proper Python path
sys.path.insert(0, '/home/qt/quantum_trader')
os.chdir('/home/qt/quantum_trader')

# Import Dummy classes from the CORRECT module
from ai_engine.agents.unified_agents import DummyNHiTS, DummyPatchTST

# Also need scalers (use existing ones or create dummy ones)
from sklearn.preprocessing import StandardScaler
import numpy as np

print("="*80)
print("RE-SAVING DUMMY MODELS WITH CORRECT MODULE PATH")
print("="*80)

# Create timestamp for new files
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create model instances
nhits_model = DummyNHiTS(mean_pred=11.08)
patchtst_model = DummyPatchTST(mean_pred=11.08)

# Create dummy scalers (31 features to match other models)
scaler = StandardScaler()
scaler.fit(np.random.rand(10, 31))  # Dummy fit

# Save models with proper module path
nhits_path = f'models/nhits_v{timestamp}.pkl'
patchtst_path = f'models/patchtst_v{timestamp}.pkl'
nhits_scaler_path = f'models/nhits_v{timestamp}_scaler.pkl'
patchtst_scaler_path = f'models/patchtst_v{timestamp}_scaler.pkl'

joblib.dump(nhits_model, nhits_path)
joblib.dump(scaler, nhits_scaler_path)
print(f"✅ Saved N-HiTS model to {nhits_path}")

joblib.dump(patchtst_model, patchtst_path)
joblib.dump(scaler, patchtst_scaler_path)
print(f"✅ Saved PatchTST model to {patchtst_path}")

# Test loading to verify module path
print("\n" + "="*80)
print("VERIFICATION - Loading models back")
print("="*80)

loaded_nhits = joblib.load(nhits_path)
loaded_patchtst = joblib.load(patchtst_path)

print(f"N-HiTS type: {type(loaded_nhits)}")
print(f"N-HiTS module: {type(loaded_nhits).__module__}")
print(f"N-HiTS prediction test: {loaded_nhits.predict([[1]*10])}")

print(f"\nPatchTST type: {type(loaded_patchtst)}")
print(f"PatchTST module: {type(loaded_patchtst).__module__}")
print(f"PatchTST prediction test: {loaded_patchtst.predict([[1]*10])}")

print("\n" + "="*80)
print("SUCCESS! Models saved with correct module path:")
print(f"  {nhits_path}")
print(f"  {patchtst_path}")
print("="*80)
