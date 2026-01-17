#!/usr/bin/env python3
"""Test ensemble manager initialization to see why N-HiTS/PatchTST aren't loading."""

import sys
sys.path.insert(0, '/home/qt/quantum_trader')

from ai_engine.ensemble_manager import EnsembleManager

print("="*80)
print("TESTING ENSEMBLE MANAGER INITIALIZATION")
print("="*80)

try:
    ensemble = EnsembleManager(
        weights=None,  # Equal weights
        min_consensus=3,
        enabled_models=['xgb', 'lgbm', 'nhits', 'patchtst'],
        xgb_model_path='models/xgboost_v20260116_131235.pkl',
        xgb_scaler_path='models/xgboost_v20260116_131235_scaler.pkl'
    )
    
    print("\n" + "="*80)
    print("INITIALIZATION SUCCESSFUL!")
    print("="*80)
    print(f"Loaded agents: {list(ensemble.__dict__.keys())}")
    
    # Test prediction
    import numpy as np
    X_test = np.random.rand(1, 31)  # 31 features
    
    result = ensemble.predict(
        symbol='TESTUSDT',
        features=X_test[0],
        current_positions={},
        market_data={'price': 100.0}
    )
    
    print(f"\nPrediction result: {result}")
    
except Exception as e:
    print(f"\n{'='*80}")
    print(f"INITIALIZATION FAILED!")
    print(f"{'='*80}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
