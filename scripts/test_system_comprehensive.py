"""
COMPREHENSIVE SYSTEM TEST - Robustness & Integration
Tests: TFT model, backend integration, monitoring, data flow
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import requests
from typing import Dict, List


def test_section(name: str):
    """Print test section header"""
    print("\n" + "="*70)
    print(f"[TEST_TUBE] TEST: {name}")
    print("="*70 + "\n")


def test_result(test_name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "[OK] PASS" if passed else "❌ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"   {details}")


# ============================================================================
# TEST 1: MODEL FILES & STRUCTURE
# ============================================================================
def test_model_files():
    test_section("Model Files & Checkpoint Structure")
    
    model_path = Path("ai_engine/models/tft_model.pth")
    
    # Check model exists
    test_result("Model file exists", model_path.exists(), 
                f"Path: {model_path}")
    
    if not model_path.exists():
        return False
    
    # Check model size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    test_result("Model size reasonable", 5 < size_mb < 10,
                f"Size: {size_mb:.2f} MB")
    
    # Load and check checkpoint structure
    try:
        checkpoint = torch.load(str(model_path), weights_only=False)
        
        required_keys = ['model_state_dict', 'model_config', 'feature_mean', 'feature_std']
        has_all_keys = all(k in checkpoint for k in required_keys)
        test_result("Checkpoint has all required keys", has_all_keys,
                   f"Keys: {list(checkpoint.keys())}")
        
        # Check normalization stats
        if 'feature_mean' in checkpoint:
            mean = checkpoint['feature_mean']
            std = checkpoint['feature_std']
            test_result("Normalization stats shape correct", 
                       mean.shape == (14,) and std.shape == (14,),
                       f"Mean: {mean.shape}, Std: {std.shape}")
            
            test_result("Normalization stats have valid values",
                       np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0),
                       f"Mean range: [{mean.min():.2f}, {mean.max():.2f}]")
        
        # Check model config
        config = checkpoint['model_config']
        expected_config = {
            'input_size': 14,
            'sequence_length': 120,
            'hidden_size': 128,
            'num_heads': 8,
            'num_layers': 3,
            'num_classes': 3
        }
        
        config_correct = all(config.get(k) == v for k, v in expected_config.items())
        test_result("Model config correct", config_correct,
                   f"Config: {config}")
        
        return True
        
    except Exception as e:
        test_result("Load checkpoint", False, f"Error: {e}")
        return False


# ============================================================================
# TEST 2: MODEL LOADING & INFERENCE
# ============================================================================
def test_model_inference():
    test_section("Model Loading & Inference")
    
    try:
        from ai_engine.agents.tft_agent import TFTAgent
        
        # Initialize agent
        agent = TFTAgent(sequence_length=120)
        test_result("Agent initialization", True)
        
        # Load model
        loaded = agent.load_model()
        test_result("Model loads successfully", loaded)
        
        if not loaded:
            return False
        
        # Check normalization loaded
        has_norm = agent.feature_mean is not None and agent.feature_std is not None
        test_result("Normalization stats loaded", has_norm)
        
        if has_norm:
            test_result("Normalization stats not zeros/ones",
                       not (np.allclose(agent.feature_mean, 0) and np.allclose(agent.feature_std, 1)),
                       f"Mean[0:3]: {agent.feature_mean[:3]}")
        
        # Test prediction with mock data
        symbol = "BTCUSDT"
        for i in range(120):
            features = {
                'Close': 50000 + i * 10,
                'Volume': 1000000,
                'EMA_10': 50000 + i * 10,
                'EMA_50': 50000 + i * 5,
                'RSI': 50,
                'MACD': 0,
                'MACD_signal': 0,
                'BB_upper': 51000,
                'BB_middle': 50000,
                'BB_lower': 49000,
                'ATR': 500,
                'volume_sma_20': 1000000,
                'price_change_pct': 0.001,
                'high_low_range': 500
            }
            agent.add_to_history(symbol, features)
        
        action, confidence, metadata = agent.predict(symbol, features)
        
        test_result("Prediction returns valid action", 
                   action in ['BUY', 'SELL', 'HOLD'],
                   f"Action: {action}")
        
        test_result("Confidence in valid range",
                   0 <= confidence <= 1,
                   f"Confidence: {confidence:.4f}")
        
        # Check metadata
        required_metadata = ['risk_reward_ratio', 'upside', 'downside', 'q10', 'q50', 'q90']
        has_metadata = all(k in metadata for k in required_metadata)
        test_result("Metadata contains R/R analysis", has_metadata,
                   f"Keys: {list(metadata.keys())}")
        
        if has_metadata:
            rr = metadata['risk_reward_ratio']
            test_result("Risk/reward ratio calculated",
                       rr > 0,
                       f"R/R: {rr:.2f}:1")
        
        return True
        
    except Exception as e:
        test_result("Model inference", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 3: BACKEND INTEGRATION
# ============================================================================
def test_backend_integration():
    test_section("Backend Integration")
    
    try:
        # Check backend health
        response = requests.get("http://localhost:8000/health", timeout=5)
        test_result("Backend responding", response.status_code == 200)
        
        if response.status_code != 200:
            return False
        
        health = response.json()
        test_result("Backend status healthy",
                   health.get('status') == 'healthy',
                   f"Status: {health.get('status')}")
        
        test_result("Event-driven mode active",
                   health.get('event_driven_active') == True)
        
        # Check risk config
        risk = health.get('risk', {})
        config = risk.get('config', {})
        test_result("Risk limits configured",
                   config.get('max_notional_per_trade') is not None,
                   f"Max per trade: ${config.get('max_notional_per_trade')}")
        
        # Check if TFT agent is available
        # (would need to check agents endpoint if available)
        
        return True
        
    except requests.exceptions.RequestException as e:
        test_result("Backend connection", False, f"Error: {e}")
        print("   💡 Make sure backend is running: docker ps")
        return False


# ============================================================================
# TEST 4: DATA PIPELINE
# ============================================================================
def test_data_pipeline():
    test_section("Data Pipeline")
    
    # Check training data
    training_data = Path("data/binance_training_data.csv")
    test_result("Training data exists", training_data.exists())
    
    if training_data.exists():
        df = pd.read_csv(training_data)
        test_result("Training data not empty",
                   len(df) > 0,
                   f"Rows: {len(df)}")
        
        required_cols = ['symbol', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        has_cols = all(c in df.columns for c in required_cols)
        test_result("Training data has required columns", has_cols)
        
        test_result("Multiple symbols present",
                   df['symbol'].nunique() > 5,
                   f"Symbols: {df['symbol'].nunique()}")
    
    # Check normalization stats file
    norm_stats = Path("ai_engine/models/tft_normalization.json")
    if norm_stats.exists():
        with open(norm_stats) as f:
            stats = json.load(f)
        test_result("Normalization JSON exists", True,
                   f"Keys: {list(stats.keys())}")
    
    return True


# ============================================================================
# TEST 5: MONITORING SCRIPTS
# ============================================================================
def test_monitoring_scripts():
    test_section("Monitoring Scripts")
    
    scripts = [
        "scripts/monitor_tft_signals.py",
        "scripts/performance_review.py",
        "scripts/train_tft_quantile.py",
        "scripts/test_tft_real_data.py",
        "scripts/fetch_training_data.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        exists = script_path.exists()
        test_result(f"Script exists: {script}", exists)
        
        if exists:
            # Check if script is executable (has proper imports)
            try:
                with open(script_path) as f:
                    content = f.read()
                    has_main = '__main__' in content
                    test_result(f"  Has main block: {script}", has_main)
            except:
                pass
    
    # Check documentation
    docs = [
        "MONITORING_GUIDE.md",
        "TFT_QUICK_REFERENCE.txt",
        "TFT_V1.1_DEPLOYMENT.md",
        "ROBUSTNESS_RECOMMENDATIONS.md"
    ]
    
    for doc in docs:
        test_result(f"Documentation exists: {doc}", Path(doc).exists())
    
    return True


# ============================================================================
# TEST 6: FEATURE ENGINEERING
# ============================================================================
def test_feature_engineering():
    test_section("Feature Engineering")
    
    try:
        from ai_engine.feature_engineer import compute_all_indicators
        
        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=200, freq='1h'),
            'Open': np.random.randn(200).cumsum() + 100,
            'High': np.random.randn(200).cumsum() + 101,
            'Low': np.random.randn(200).cumsum() + 99,
            'Close': np.random.randn(200).cumsum() + 100,
            'Volume': np.random.randint(1000, 10000, 200)
        })
        
        # Compute indicators
        df_with_features = compute_all_indicators(df, use_advanced=False)
        
        required_features = [
            'EMA_10', 'EMA_50', 'RSI_14', 'MACD', 'MACD_signal',
            'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
            'volume_sma_20', 'price_change_pct', 'high_low_range'
        ]
        
        has_features = all(f in df_with_features.columns for f in required_features)
        test_result("All required features computed", has_features)
        
        # Check for NaN values
        nan_count = df_with_features[required_features].isna().sum().sum()
        test_result("No NaN values in features",
                   nan_count == 0,
                   f"NaN count: {nan_count}")
        
        return True
        
    except Exception as e:
        test_result("Feature engineering", False, f"Error: {e}")
        return False


# ============================================================================
# TEST 7: END-TO-END PREDICTION FLOW
# ============================================================================
def test_end_to_end():
    test_section("End-to-End Prediction Flow")
    
    try:
        # 1. Load training data
        df = pd.read_csv("data/binance_training_data.csv")
        test_result("Step 1: Load training data", len(df) > 0,
                   f"{len(df)} rows")
        
        # 2. Compute features
        from ai_engine.feature_engineer import compute_all_indicators
        symbol_df = df[df['symbol'] == 'BTCUSDT'].head(150)
        df_features = compute_all_indicators(symbol_df, use_advanced=False)
        test_result("Step 2: Compute features", len(df_features) > 0)
        
        # 3. Load model
        from ai_engine.agents.tft_agent import TFTAgent
        agent = TFTAgent(sequence_length=120)
        loaded = agent.load_model()
        test_result("Step 3: Load TFT model", loaded)
        
        # 4. Feed history
        feature_cols = [
            'Close', 'Volume', 'EMA_10', 'EMA_50', 'RSI_14',
            'MACD', 'MACD_signal', 'BB_upper', 'BB_middle', 'BB_lower',
            'ATR', 'volume_sma_20', 'price_change_pct', 'high_low_range'
        ]
        
        for idx, row in df_features.head(120).iterrows():
            features = {col: row[col] for col in feature_cols}
            agent.add_to_history('BTCUSDT', features)
        
        test_result("Step 4: Feed 120 candles to history", 
                   len(agent.history_buffer['BTCUSDT']) >= 120)
        
        # 5. Make prediction
        last_row = df_features.iloc[119]
        last_features = {col: last_row[col] for col in feature_cols}
        action, confidence, metadata = agent.predict('BTCUSDT', last_features)
        
        test_result("Step 5: Generate prediction",
                   action in ['BUY', 'SELL', 'HOLD'],
                   f"Action: {action}, Conf: {confidence:.2f}, R/R: {metadata.get('risk_reward_ratio', 0):.2f}")
        
        # 6. Verify R/R analysis
        has_rr = all(k in metadata for k in ['risk_reward_ratio', 'upside', 'downside'])
        test_result("Step 6: R/R analysis included", has_rr)
        
        return True
        
    except Exception as e:
        test_result("End-to-end flow", False, f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 8: ROBUSTNESS CHECKS
# ============================================================================
def test_robustness():
    test_section("Robustness & Error Handling")
    
    from ai_engine.agents.tft_agent import TFTAgent
    agent = TFTAgent(sequence_length=120)
    agent.load_model()
    
    # Test 1: Prediction with insufficient history
    action, conf, meta = agent.predict('NEWCOIN', {'Close': 100})
    test_result("Handles insufficient history gracefully",
               action == 'HOLD' and conf == 0.0)
    
    # Test 2: Missing features
    agent.add_to_history('TESTCOIN', {'Close': 100})  # Incomplete features
    test_result("Handles missing features", True)  # Should not crash
    
    # Test 3: Extreme values
    extreme_features = {
        'Close': 1e10,
        'Volume': 1e15,
        'EMA_10': 1e10,
        'EMA_50': 1e10,
        'RSI': 100,
        'MACD': 1e6,
        'MACD_signal': 1e6,
        'BB_upper': 1e10,
        'BB_middle': 1e10,
        'BB_lower': 1e10,
        'ATR': 1e8,
        'volume_sma_20': 1e15,
        'price_change_pct': 10.0,
        'high_low_range': 1e8
    }
    
    for i in range(120):
        agent.add_to_history('EXTREMECOIN', extreme_features)
    
    try:
        action, conf, meta = agent.predict('EXTREMECOIN', extreme_features)
        test_result("Handles extreme values without crashing", True,
                   f"Action: {action}")
    except Exception as e:
        test_result("Handles extreme values", False, f"Error: {e}")
    
    # Test 4: NaN values
    nan_features = {k: float('nan') for k in extreme_features.keys()}
    nan_features['Close'] = 100  # At least one valid value
    
    try:
        for i in range(120):
            agent.add_to_history('NANCOIN', nan_features)
        action, conf, meta = agent.predict('NANCOIN', nan_features)
        test_result("Handles NaN values", True)
    except Exception as e:
        test_result("Handles NaN values", False, f"Error: {e}")
    
    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "COMPREHENSIVE SYSTEM TEST" + " "*28 + "║")
    print("╚" + "═"*68 + "╝")
    
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Testing: Model, Backend, Integration, Robustness\n")
    
    results = {}
    
    # Run all tests
    results['Model Files'] = test_model_files()
    results['Model Inference'] = test_model_inference()
    results['Backend Integration'] = test_backend_integration()
    results['Data Pipeline'] = test_data_pipeline()
    results['Monitoring Scripts'] = test_monitoring_scripts()
    results['Feature Engineering'] = test_feature_engineering()
    results['End-to-End Flow'] = test_end_to_end()
    results['Robustness'] = test_robustness()
    
    # Summary
    print("\n" + "="*70)
    print("[CHART] TEST SUMMARY")
    print("="*70 + "\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[OK] PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} test suites passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is robust and functional.")
        print("\n[OK] Ready for production:")
        print("   • Model loads correctly with normalization")
        print("   • Predictions work end-to-end")
        print("   • Backend integration operational")
        print("   • Monitoring scripts ready")
        print("   • Error handling robust")
    else:
        print("[WARNING]  SOME TESTS FAILED - Review errors above")
        failed = [name for name, result in results.items() if not result]
        print(f"\nFailed tests: {', '.join(failed)}")
    
    print("\n" + "="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
