#!/usr/bin/env python3
"""Quick diagnostic: Check if XGB and DLinear files exist and ensemble config"""
import os, sys

model_dir = "/home/qt/quantum_trader/ai_engine/models"
agents_file = "/home/qt/quantum_trader/ai_engine/agents/unified_agents.py"

print("=== MODEL FILES ===")
if os.path.isdir(model_dir):
    files = [f for f in os.listdir(model_dir) if not f.startswith("scaler_v")]
    xgb_files = [f for f in files if f.startswith("xgb_v") and not any(x in f for x in ["_scaler","_meta","_features"])]
    dlinear_files = [f for f in files if f.startswith("dlinear_v") and not any(x in f for x in ["_scaler","_meta","_features"])]
    lgbm_files = [f for f in files if f.startswith("lightgbm_v") and not any(x in f for x in ["_scaler","_meta","_features"])]
    nhits_files = [f for f in files if f.startswith("nhits_v") and not any(x in f for x in ["_scaler","_meta","_features"])]
    patchtst_files = [f for f in files if f.startswith("patchtst_v") and not any(x in f for x in ["_scaler","_meta","_features"])]
    tft_files = [f for f in files if f.startswith("tft_v") and not any(x in f for x in ["_scaler","_meta","_features"])]
    
    print(f"XGB models: {sorted(xgb_files)[-1] if xgb_files else 'NONE'}")
    print(f"LGBM models: {sorted(lgbm_files)[-1] if lgbm_files else 'NONE'}")
    print(f"NHiTS models: {sorted(nhits_files)[-1] if nhits_files else 'NONE'}")
    print(f"PatchTST models: {sorted(patchtst_files)[-1] if patchtst_files else 'NONE'}")
    print(f"TFT models: {sorted(tft_files)[-1] if tft_files else 'NONE'}")
    print(f"DLinear models: {sorted(dlinear_files)[-1] if dlinear_files else 'NONE'}")

print("\n=== XGB PREFIX IN UNIFIED_AGENTS ===")
if os.path.exists(agents_file):
    for i, line in enumerate(open(agents_file)):
        if "XGBoostAgent" in line or "xgboost_v" in line or "xgb_v" in line:
            print(f"  L{i+1}: {line.rstrip()}")

print("\n=== ENV FILE ===")
env_file = "/etc/quantum/ai-engine.env"
for line in open(env_file):
    if any(k in line for k in ["ENSEMBLE_MODELS", "MODEL_PATH", "SCALER_PATH"]):
        print(f"  {line.rstrip()}")

print("\n=== SERVICE FILE ===")
svc_file = "/etc/systemd/system/quantum-ai-engine.service"
for line in open(svc_file):
    if any(k in line for k in ["WorkingDirectory", "ExecStart", "PYTHONPATH"]):
        print(f"  {line.rstrip()}")

print("\n=== VENV CHECK ===")
import subprocess
r = subprocess.run(["/home/qt/quantum_trader_venv/bin/python", "-c", "import xgboost; print('xgb OK')"], 
                   capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
r2 = subprocess.run(["/home/qt/quantum_trader_venv/bin/python", "-c", "import torch; print('torch OK')"], 
                    capture_output=True, text=True)
print(r2.stdout.strip() or r2.stderr.strip())

print("\nDONE")
