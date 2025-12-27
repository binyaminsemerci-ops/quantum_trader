"""
Test Unified Feature Predictions
Verify XGBoost and LightGBM work with 55 unified features
"""

import sys
import os
sys.path.insert(0, '/app')

import pandas as pd
import numpy as np
from backend.shared.unified_features import get_feature_engineer
from ai_engine.agents.xgb_agent import XGBAgent
from ai_engine.agents.lgbm_agent import LightGBMAgent
from datetime import datetime, timedelta


def load_test_data(symbol="BTCUSDT", rows=200):
    """Load recent OHLCV data for testing"""
    try:
        # Try universe data first
        df = pd.read_parquet(f"/app/data/universe/{symbol}.parquet")
        print(f"✅ Loaded {len(df)} candles from universe data")
        return df.tail(rows)
    except:
        # Fallback to fetching live data
        print(f"⚠️ Universe data not found, using synthetic data")
        dates = pd.date_range(end=datetime.now(), periods=rows, freq='5T')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': 50000 + np.random.randn(rows) * 100,
            'high': 50100 + np.random.randn(rows) * 100,
            'low': 49900 + np.random.randn(rows) * 100,
            'close': 50000 + np.random.randn(rows) * 100,
            'volume': 1000 + np.random.randn(rows) * 100
        })
        return df


def test_feature_computation(symbol="BTCUSDT"):
    """Test that unified features produce 55 features"""
    print(f"\n{'='*60}")
    print(f"TEST 1: Feature Computation for {symbol}")
    print(f"{'='*60}")
    
    # Load data
    df = load_test_data(symbol)
    print(f"📊 Loaded {len(df)} candles")
    
    # Compute unified features
    engineer = get_feature_engineer()
    features = engineer.compute_features(df)
    
    # Count features (exclude base OHLCV columns)
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    feature_cols = [col for col in features.columns if col not in base_cols]
    
    print(f"✅ Total columns: {len(features.columns)}")
    print(f"✅ Feature columns: {len(feature_cols)}")
    print(f"✅ Expected: 55 features")
    print(f"✅ Match: {len(feature_cols) == 55}")
    
    # Check for NaN/Inf
    nan_count = features[feature_cols].isna().sum().sum()
    inf_count = np.isinf(features[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    
    print(f"✅ NaN values: {nan_count}")
    print(f"✅ Inf values: {inf_count}")
    
    return features, feature_cols


def test_xgboost_prediction(features, feature_cols):
    """Test XGBoost prediction with unified features"""
    print(f"\n{'='*60}")
    print(f"TEST 2: XGBoost Prediction")
    print(f"{'='*60}")
    
    try:
        agent = XGBAgent()
        
        # Get latest features
        latest = features[feature_cols].iloc[-1:].values
        print(f"📊 Input shape: {latest.shape}")
        print(f"📊 Feature count: {latest.shape[1]}")
        
        # Make prediction
        result = agent.predict(features.tail(1))
        
        print(f"✅ XGBoost prediction successful!")
        print(f"✅ Signal: {result['signal']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ Direction: {result.get('direction', 'N/A')}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ XGBoost prediction FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_lightgbm_prediction(features, feature_cols):
    """Test LightGBM prediction with unified features"""
    print(f"\n{'='*60}")
    print(f"TEST 3: LightGBM Prediction")
    print(f"{'='*60}")
    
    try:
        agent = LightGBMAgent()
        
        # Get latest features
        latest = features[feature_cols].iloc[-1:].values
        print(f"📊 Input shape: {latest.shape}")
        print(f"📊 Feature count: {latest.shape[1]}")
        
        # Make prediction
        result = agent.predict(features.tail(1))
        
        print(f"✅ LightGBM prediction successful!")
        print(f"✅ Signal: {result['signal']}")
        print(f"✅ Confidence: {result['confidence']:.3f}")
        print(f"✅ Direction: {result.get('direction', 'N/A')}")
        
        return True, result
        
    except Exception as e:
        print(f"❌ LightGBM prediction FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_ensemble_consensus(xgb_result, lgbm_result):
    """Test ensemble consensus between models"""
    print(f"\n{'='*60}")
    print(f"TEST 4: Ensemble Consensus")
    print(f"{'='*60}")
    
    if xgb_result and lgbm_result:
        xgb_signal = xgb_result['signal']
        lgbm_signal = lgbm_result['signal']
        
        consensus = xgb_signal == lgbm_signal
        
        print(f"📊 XGBoost: {xgb_signal} (conf={xgb_result['confidence']:.3f})")
        print(f"📊 LightGBM: {lgbm_signal} (conf={lgbm_result['confidence']:.3f})")
        print(f"✅ Consensus: {consensus}")
        
        if consensus:
            avg_conf = (xgb_result['confidence'] + lgbm_result['confidence']) / 2
            print(f"✅ Average Confidence: {avg_conf:.3f}")
            
            if avg_conf >= 0.6:
                print(f"✅ HIGH QUALITY SIGNAL (conf >= 0.6)")
            else:
                print(f"⚠️ LOW CONFIDENCE SIGNAL (conf < 0.6)")
        else:
            print(f"⚠️ NO CONSENSUS - models disagree")
        
        return consensus
    else:
        print(f"❌ Cannot test consensus - predictions failed")
        return False


def test_multiple_symbols():
    """Test predictions on multiple symbols"""
    print(f"\n{'='*60}")
    print(f"TEST 5: Multiple Symbol Validation")
    print(f"{'='*60}")
    
    test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    results = []
    
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        try:
            df = load_test_data(symbol)
            engineer = get_feature_engineer()
            features = engineer.compute_features(df)
            
            # Test XGBoost
            xgb_agent = XGBAgent()
            xgb_result = xgb_agent.predict(features.tail(1))
            
            # Test LightGBM
            lgbm_agent = LightGBMAgent()
            lgbm_result = lgbm_agent.predict(features.tail(1))
            
            consensus = xgb_result['signal'] == lgbm_result['signal']
            
            results.append({
                'symbol': symbol,
                'xgb_signal': xgb_result['signal'],
                'lgbm_signal': lgbm_result['signal'],
                'consensus': consensus,
                'avg_confidence': (xgb_result['confidence'] + lgbm_result['confidence']) / 2
            })
            
            print(f"✅ {symbol}: XGB={xgb_result['signal']}, LGBM={lgbm_result['signal']}, Consensus={consensus}")
            
        except Exception as e:
            print(f"❌ {symbol} FAILED: {e}")
            results.append({
                'symbol': symbol,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Multiple Symbol Test")
    print(f"{'='*60}")
    
    successful = [r for r in results if 'error' not in r]
    consensus_count = sum(1 for r in successful if r['consensus'])
    
    print(f"✅ Successful predictions: {len(successful)}/{len(test_symbols)}")
    print(f"✅ Consensus rate: {consensus_count}/{len(successful)} = {consensus_count/len(successful)*100:.1f}%")
    
    if successful:
        avg_conf = np.mean([r['avg_confidence'] for r in successful])
        print(f"✅ Average confidence: {avg_conf:.3f}")
    
    return results


def main():
    """Run all tests"""
    print(f"\n{'#'*60}")
    print(f"# UNIFIED FEATURE PREDICTION TEST SUITE")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # Test 1: Feature computation
    features, feature_cols = test_feature_computation("BTCUSDT")
    
    if len(feature_cols) != 55:
        print(f"\n❌ CRITICAL: Feature count mismatch! Expected 55, got {len(feature_cols)}")
        print(f"❌ ABORT: Cannot proceed with predictions")
        return False
    
    # Test 2: XGBoost prediction
    xgb_success, xgb_result = test_xgboost_prediction(features, feature_cols)
    
    # Test 3: LightGBM prediction
    lgbm_success, lgbm_result = test_lightgbm_prediction(features, feature_cols)
    
    # Test 4: Ensemble consensus
    if xgb_success and lgbm_success:
        consensus = test_ensemble_consensus(xgb_result, lgbm_result)
    else:
        consensus = False
    
    # Test 5: Multiple symbols
    multi_results = test_multiple_symbols()
    
    # Final verdict
    print(f"\n{'#'*60}")
    print(f"# FINAL VERDICT")
    print(f"{'#'*60}")
    
    all_passed = (
        len(feature_cols) == 55 and
        xgb_success and
        lgbm_success and
        len([r for r in multi_results if 'error' not in r]) >= 3
    )
    
    if all_passed:
        print(f"✅✅✅ ALL TESTS PASSED ✅✅✅")
        print(f"✅ Feature engineering: UNIFIED (55 features)")
        print(f"✅ XGBoost predictions: WORKING")
        print(f"✅ LightGBM predictions: WORKING")
        print(f"✅ Multiple symbols: VALIDATED")
        print(f"\n🚀 READY FOR PAPER TRADING!")
    else:
        print(f"❌❌❌ TESTS FAILED ❌❌❌")
        print(f"❌ Feature count: {'PASS' if len(feature_cols) == 55 else 'FAIL'}")
        print(f"❌ XGBoost: {'PASS' if xgb_success else 'FAIL'}")
        print(f"❌ LightGBM: {'PASS' if lgbm_success else 'FAIL'}")
        print(f"\n🛑 DO NOT DEPLOY - FIX ISSUES FIRST!")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
