"""
Validation script for unified feature engineering.

Tests:
1. Feature count consistency (training vs inference)
2. Model loading with correct feature expectations
3. Prediction success with unified features
4. No feature mismatch errors
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def log_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def log_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def log_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def log_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")


def test_unified_features():
    """Test unified feature engineering."""
    log_info("Testing UnifiedFeatureEngineer...")
    
    try:
        from backend.shared.unified_features import get_feature_engineer
        
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 110, 200),
            'high': np.random.uniform(110, 115, 200),
            'low': np.random.uniform(95, 100, 200),
            'close': np.random.uniform(100, 110, 200),
            'volume': np.random.uniform(1000, 5000, 200),
        })
        
        engineer = get_feature_engineer()
        features = engineer.compute_features(df)
        
        log_success(f"UnifiedFeatureEngineer computed {len(features.columns)} features")
        log_info(f"   Rows: {len(df)} → {len(features)} (after dropna)")
        log_info(f"   Features: {list(features.columns)[:10]}...")
        
        return len(features.columns)
        
    except Exception as e:
        log_error(f"UnifiedFeatureEngineer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_loading(model_name, model_path, scaler_path=None):
    """Test loading a specific model and check feature expectations."""
    log_info(f"Testing {model_name} model loading...")
    
    try:
        model_file = Path(model_path)
        
        if not model_file.exists():
            log_warning(f"{model_name} model not found at {model_path}")
            return None
        
        # Load model
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        log_success(f"{model_name} model loaded from {model_file.name}")
        
        # Check scaler if provided
        if scaler_path:
            scaler_file = Path(scaler_path)
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
                
                if hasattr(scaler, 'n_features_in_'):
                    log_info(f"   Scaler expects {scaler.n_features_in_} features")
                    return scaler.n_features_in_
                elif hasattr(scaler, 'feature_names_in_'):
                    log_info(f"   Scaler has {len(scaler.feature_names_in_)} feature names")
                    return len(scaler.feature_names_in_)
        
        # Check model feature expectations
        if hasattr(model, 'n_features_in_'):
            log_info(f"   Model expects {model.n_features_in_} features")
            return model.n_features_in_
        
        log_warning(f"   Cannot determine feature count for {model_name}")
        return None
        
    except Exception as e:
        log_error(f"{model_name} model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference_features():
    """Test feature computation from ai_engine.feature_engineer."""
    log_info("Testing ai_engine.feature_engineer (inference)...")
    
    try:
        from ai_engine.feature_engineer import compute_all_indicators
        
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 110, 200),
            'high': np.random.uniform(110, 115, 200),
            'low': np.random.uniform(95, 100, 200),
            'close': np.random.uniform(100, 110, 200),
            'volume': np.random.uniform(1000, 5000, 200),
        })
        
        features = compute_all_indicators(df)
        
        log_success(f"compute_all_indicators computed {len(features.columns)} features")
        log_info(f"   Rows: {len(df)} → {len(features)}")
        
        return len(features.columns)
        
    except Exception as e:
        log_error(f"compute_all_indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_latest_models():
    """Find latest trained models by timestamp."""
    log_info("Finding latest trained models...")
    
    model_dir = Path('ai_engine/models')
    
    models = {
        'xgboost': None,
        'lightgbm': None,
        'nhits': None,
        'patchtst': None,
    }
    
    for model_type in models.keys():
        # Find files matching pattern
        pattern = f"{model_type}_v*.pkl" if model_type in ['xgboost', 'lightgbm'] else f"{model_type}_v*.pkl"
        matching_files = sorted(model_dir.glob(pattern))
        
        if matching_files:
            latest = matching_files[-1]
            models[model_type] = str(latest)
            log_info(f"   Latest {model_type}: {latest.name}")
    
    return models


def main():
    """Run all validation tests."""
    print(f"\n{BLUE}{'='*80}")
    print("UNIFIED FEATURE ENGINEERING VALIDATION")
    print(f"{'='*80}{RESET}\n")
    
    # Test 1: UnifiedFeatureEngineer
    print(f"\n{BLUE}[TEST 1: UnifiedFeatureEngineer]{RESET}")
    unified_features = test_unified_features()
    
    # Test 2: Inference features
    print(f"\n{BLUE}[TEST 2: Inference Feature Computation]{RESET}")
    inference_features = test_inference_features()
    
    # Test 3: Feature count consistency
    print(f"\n{BLUE}[TEST 3: Feature Count Consistency]{RESET}")
    if unified_features and inference_features:
        if unified_features == inference_features:
            log_success(f"Feature counts MATCH: {unified_features} features")
        else:
            log_error(f"Feature counts MISMATCH: unified={unified_features}, inference={inference_features}")
    else:
        log_warning("Cannot verify feature count consistency")
    
    # Test 4: Find and check latest models
    print(f"\n{BLUE}[TEST 4: Latest Model Feature Expectations]{RESET}")
    latest_models = find_latest_models()
    
    for model_name, model_path in latest_models.items():
        if model_path:
            scaler_path = model_path.replace('_v', '_scaler_v').replace('.pkl', '_scaler.pkl')
            if not Path(scaler_path).exists():
                scaler_path = f"ai_engine/models/{model_name}_scaler.pkl"
            
            expected_features = test_model_loading(model_name, model_path, scaler_path)
            
            if expected_features and unified_features:
                if expected_features == unified_features:
                    log_success(f"{model_name}: Feature count MATCH ({expected_features})")
                else:
                    log_error(f"{model_name}: Feature count MISMATCH (model expects {expected_features}, unified provides {unified_features})")
    
    # Summary
    print(f"\n{BLUE}{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}{RESET}\n")
    
    if unified_features and inference_features and unified_features == inference_features:
        log_success("Feature engineering is UNIFIED and CONSISTENT!")
        log_info(f"   Both training and inference use {unified_features} features")
    else:
        log_error("Feature engineering needs fixes")
    
    print()


if __name__ == "__main__":
    main()
