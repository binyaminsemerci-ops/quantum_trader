#!/usr/bin/env python3
"""Test XGBoost and LightGBM prediction with different input formats"""
import pickle
import pandas as pd
import numpy as np
import json

def test_xgboost():
    print("="*70)
    print("TESTING XGBOOST AGENT")
    print("="*70)
    
    # Load model, scaler, metadata
    print("\n[1] Loading files...")
    with open("models/xgboost_v20260205_231035_v5.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/xgboost_v20260205_231035_v5_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/xgboost_v20260205_231035_v5_meta.json") as f:
        meta = json.load(f)
    
    print(f"  Model type: {type(model).__name__}")
    print(f"  Model classes: {model.classes_}")
    print(f"  Scaler n_features: {scaler.n_features_in_}")
    print(f"  Metadata features: {len(meta['features'])}")
    print(f"  Feature names: {meta['features'][:5]}...")
    
    # Create test data
    features = meta['features']
    test_data = {f: np.random.randn() for f in features}
    test_df = pd.DataFrame([test_data])
    
    print(f"\n[2] Test DataFrame:")
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns)[:5]}...")
    
    # TEST A: Numpy array (current agent method)
    print("\n[3] TEST A: scaler.transform() -> numpy array")
    try:
        X_numpy = scaler.transform(test_df)
        print(f"  Scaled type: {type(X_numpy)}")
        print(f"  Scaled shape: {X_numpy.shape}")
        pred = model.predict(X_numpy)
        print(f"  ✅ SUCCESS: prediction = {pred}")
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:150]}")
        error_type_a = str(e)
    
    # TEST B: DataFrame with column names
    print("\n[4] TEST B: DataFrame with feature names preserved")
    try:
        X_scaled = scaler.transform(test_df)
        X_df = pd.DataFrame(X_scaled, columns=features)
        print(f"  DataFrame shape: {X_df.shape}")
        print(f"  DataFrame columns: {list(X_df.columns)[:5]}...")
        pred = model.predict(X_df)
        print(f"  ✅ SUCCESS: prediction = {pred} (class={pred[0]})")
        class_name = ['SELL', 'HOLD', 'BUY'][pred[0]]
        print(f"  Action: {class_name}")
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:150]}")
    
    # TEST C: Check if model has feature_names_in_
    print("\n[5] MODEL INSPECTION:")
    if hasattr(model, 'feature_names_in_'):
        print(f"  feature_names_in_: {model.feature_names_in_[:5]}...")
    else:
        print(f"  feature_names_in_: NOT FOUND")
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"  enable_categorical: {params.get('enable_categorical', 'N/A')}")

def test_lightgbm():
    print("\n" + "="*70)
    print("TESTING LIGHTGBM AGENT")
    print("="*70)
    
    print("\n[1] Loading files...")
    with open("models/lightgbm_v20260205_231055_v5.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/lightgbm_v20260205_231055_v5_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/lightgbm_v20260205_231055_v5_meta.json") as f:
        meta = json.load(f)
    
    print(f"  Model type: {type(model).__name__}")
    print(f"  Metadata features: {len(meta['features'])}")
    
    features = meta['features']
    test_data = {f: np.random.randn() for f in features}
    test_df = pd.DataFrame([test_data])
    
    # TEST A: Numpy array
    print("\n[2] TEST A: numpy array")
    try:
        X_numpy = scaler.transform(test_df)
        pred = model.predict(X_numpy)
        print(f"  Result type: {type(pred)}, shape: {np.array(pred).shape if hasattr(pred, 'shape') else 'N/A'}")
        print(f"  ✅ SUCCESS: prediction = {pred}")
        print(f"  First value: {pred[0] if hasattr(pred, '__getitem__') else pred}")
    except Exception as e:
        print(f"  ❌ FAILED: {str(e)[:200]}")

if __name__ == "__main__":
    import sys
    import os
    
    # Change to quantum_trader directory
    os.chdir("/home/qt/quantum_trader")
    
    try:
        test_xgboost()
    except Exception as e:
        print(f"\nXGBoost test crashed: {e}")
    
    try:
        test_lightgbm()
    except Exception as e:
        print(f"\nLightGBM test crashed: {e}")
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
