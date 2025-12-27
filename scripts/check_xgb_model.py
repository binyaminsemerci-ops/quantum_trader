#!/usr/bin/env python3
"""Check XGBoost model feature configuration"""
import pickle
import sys

model_path = "/app/ai_engine/models/xgb_model.pkl"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Model type: {type(model).__name__}")
    
    if hasattr(model, 'n_features_in_'):
        print(f"Expected features: {model.n_features_in_}")
    else:
        print("Expected features: unknown (attribute not found)")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")
    
    print(f"\nModel loaded successfully from {model_path}")
    sys.exit(0)
    
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
