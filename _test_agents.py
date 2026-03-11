#!/usr/bin/env python3
"""Test agent loading - diagnose XGB scaler and LGBM model failures"""
import sys, traceback
sys.path.insert(0, '/home/qt/quantum_trader')

print("=== TESTING AGENT LOADING ===")

# Test XGBoostAgent
print("\n--- XGBoostAgent ---")
try:
    from ai_engine.agents.unified_agents import XGBoostAgent
    xgb = XGBoostAgent()
    print(f"model type: {type(xgb.model)}")
    print(f"scaler type: {type(xgb.scaler)}")
    print(f"model_dir: {xgb.model_dir}")
    print(f"prefix: {xgb.prefix}")
    import os
    files = [f for f in os.listdir(xgb.model_dir) 
             if f.startswith(xgb.prefix) and not any(x in f for x in ['_scaler','_meta','_features'])]
    print(f"matching files: {sorted(files)}")
    if xgb._find_latest():
        print(f"_find_latest result: {xgb._find_latest()}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# Test LightGBMAgent
print("\n--- LightGBMAgent ---")
try:
    from ai_engine.agents.unified_agents import LightGBMAgent
    lgbm = LightGBMAgent()
    print(f"model type: {type(lgbm.model)}")
    print(f"scaler type: {type(lgbm.scaler)}")
    print(f"model_dir: {lgbm.model_dir}")
    print(f"prefix: {lgbm.prefix}")
    import os
    files = [f for f in os.listdir(lgbm.model_dir) 
             if f.startswith(lgbm.prefix) and not any(x in f for x in ['_scaler','_meta','_features'])]
    print(f"matching files: {sorted(files)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# Test DLinearAgent
print("\n--- DLinearAgent ---")
try:
    from ai_engine.agents.unified_agents import DLinearAgent
    d = DLinearAgent()
    print(f"model type: {type(d.model)}")
    print(f"scaler type: {type(d.scaler)}")
    if d.model:
        print("DLinear LOADED SUCCESSFULLY!")
    else:
        import os
        files = [f for f in os.listdir(d.model_dir) 
                 if f.startswith(d.prefix) and not any(x in f for x in ['_scaler','_meta','_features'])]
        print(f"matching files: {sorted(files)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("\n=== DONE ===")
