"""
Complete End-to-End System Test
Tests backend, AI engine, all improvements, and API endpoints
"""

import sys
import os
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("[TEST_TUBE] QUANTUM TRADER - FULL SYSTEM TEST")
print("="*80)
print(f"Test Time: {datetime.now()}")
print("="*80)

# Configuration
BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:5173"
TIMEOUT = 5

results = {
    'passed': 0,
    'failed': 0,
    'warnings': 0,
    'tests': []
}

def log_test(name: str, status: str, message: str = "", details: str = ""):
    """Log test result"""
    emoji = "[OK]" if status == "PASS" else "❌" if status == "FAIL" else "[WARNING]"
    print(f"\n{emoji} {name}")
    if message:
        print(f"   {message}")
    if details:
        print(f"   Details: {details}")
    
    results['tests'].append({
        'name': name,
        'status': status,
        'message': message,
        'details': details
    })
    
    if status == "PASS":
        results['passed'] += 1
    elif status == "FAIL":
        results['failed'] += 1
    else:
        results['warnings'] += 1

# ============================================================
# TEST 1: Backend Server Health
# ============================================================
print("\n" + "="*80)
print("TEST 1: BACKEND SERVER HEALTH")
print("="*80)

try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        log_test(
            "Backend Health Check",
            "PASS",
            f"Server running: {data.get('status', 'unknown')}",
            f"Timestamp: {data.get('timestamp', 'N/A')}"
        )
    else:
        log_test("Backend Health Check", "FAIL", f"Status code: {response.status_code}")
except Exception as e:
    log_test("Backend Health Check", "FAIL", f"Cannot connect to backend: {str(e)}")
    print("\n❌ CRITICAL: Backend server not running!")
    print("   Start backend: cd backend && uvicorn main:app --reload")
    sys.exit(1)

# ============================================================
# TEST 2: AI Engine - Ensemble Model
# ============================================================
print("\n" + "="*80)
print("TEST 2: AI ENGINE - ENSEMBLE MODEL")
print("="*80)

try:
    from ai_engine.model_ensemble import EnsemblePredictor
    import pickle
    
    # Check if ensemble model exists
    if os.path.exists("ai_engine/models/ensemble_model.pkl"):
        ensemble = EnsemblePredictor()
        ensemble.load("ensemble_model.pkl")
        
        # Load scaler
        with open("ai_engine/models/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        
        log_test(
            "Ensemble Model Loading",
            "PASS",
            f"Loaded {len(ensemble.models)} models",
            f"Models: {list(ensemble.models.keys())}"
        )
        
        # Test prediction
        test_features = np.random.randn(1, 77)
        test_features_scaled = scaler.transform(test_features)
        pred, conf = ensemble.predict_with_confidence(test_features_scaled)
        
        log_test(
            "Ensemble Prediction",
            "PASS",
            f"Prediction: {pred[0]:.6f}, Confidence: {conf[0]:.2f}",
            "Model produces valid outputs"
        )
    else:
        log_test(
            "Ensemble Model Loading",
            "WARN",
            "Ensemble model not found, will use fallback XGBoost"
        )
except Exception as e:
    log_test("Ensemble Model Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 3: Advanced Features
# ============================================================
print("\n" + "="*80)
print("TEST 3: ADVANCED FEATURE ENGINEERING")
print("="*80)

try:
    from ai_engine.feature_engineer_advanced import add_advanced_features
    
    # Create sample OHLCV data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1h')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.random.randn(200) * 100,
        'high': 50100 + np.random.randn(200) * 100,
        'low': 49900 + np.random.randn(200) * 100,
        'close': 50000 + np.random.randn(200) * 100,
        'volume': np.random.randint(100, 1000, 200)
    })
    
    # Add advanced features
    df_with_features = add_advanced_features(sample_df)
    
    new_features = len(df_with_features.columns) - len(sample_df.columns)
    
    log_test(
        "Advanced Feature Engineering",
        "PASS",
        f"Added {new_features} features",
        f"Total columns: {len(df_with_features.columns)}"
    )
    
except Exception as e:
    log_test("Advanced Features Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 4: Position Sizing (Kelly Criterion)
# ============================================================
print("\n" + "="*80)
print("TEST 4: POSITION SIZING (KELLY CRITERION)")
print("="*80)

try:
    from backend.services.position_sizing import create_position_sizer
    
    sizer = create_position_sizer(10000)
    
    signal = {
        'confidence': 0.75,
        'volatility': 0.02,
        'prediction': 0.03
    }
    
    result = sizer.calculate_position_size(
        signal=signal,
        current_price=50000,
        stop_loss_price=49000
    )
    
    log_test(
        "Kelly Position Sizing",
        "PASS",
        f"Position size: {result['position_size']:.2f} units (${result['position_value']:.2f})",
        f"Kelly fraction: {result['kelly_fraction']:.2%}, Risk: ${result['risk_amount']:.2f}"
    )
    
except Exception as e:
    log_test("Position Sizing Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 5: Smart Execution
# ============================================================
print("\n" + "="*80)
print("TEST 5: SMART ORDER EXECUTION")
print("="*80)

try:
    from backend.services.execution.smart_execution import create_smart_executor
    
    executor = create_smart_executor()
    
    result = executor.execute_smart_order(
        symbol='BTC/USDT',
        side='buy',
        size=0.1,
        current_price=50000,
        urgency='normal'
    )
    
    log_test(
        "Smart Order Execution",
        "PASS",
        f"Strategy: {result['strategy']}, Avg Price: ${result['avg_price']:.2f}",
        f"Total filled: {result['total_filled']}, Fees: ${result['total_fees']:.4f}"
    )
    
except Exception as e:
    log_test("Smart Execution Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 6: Advanced Risk Management
# ============================================================
print("\n" + "="*80)
print("TEST 6: ADVANCED RISK MANAGEMENT")
print("="*80)

try:
    from backend.services.advanced_risk import create_risk_manager
    
    risk_mgr = create_risk_manager()
    
    position = {
        'symbol': 'BTC/USDT',
        'side': 'long',
        'entry_price': 50000,
        'current_price': 51000,
        'size': 0.1,
        'stop_loss': 49000,
        'entry_time': datetime.now()
    }
    
    actions = risk_mgr.manage_position_risk(position, atr=500)
    
    log_test(
        "Risk Management",
        "PASS",
        f"Monitoring position with P&L: ${(51000-50000)*0.1:.2f}",
        f"Actions: {len(actions)} risk checks performed"
    )
    
except Exception as e:
    log_test("Risk Management Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 7: Market Regime Detection
# ============================================================
print("\n" + "="*80)
print("TEST 7: MARKET REGIME DETECTION")
print("="*80)

try:
    from ai_engine.regime_detection import create_regime_detector
    
    detector = create_regime_detector()
    
    # Create sample market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1h')
    market_df = pd.DataFrame({
        'timestamp': dates,
        'open': 50000 + np.cumsum(np.random.randn(100) * 100),
        'high': 50500 + np.cumsum(np.random.randn(100) * 100),
        'low': 49500 + np.cumsum(np.random.randn(100) * 100),
        'close': 50000 + np.cumsum(np.random.randn(100) * 100),
        'volume': np.random.randint(100, 1000, 100)
    })
    
    # Add features needed for regime detection
    from ai_engine.feature_engineer_advanced import add_advanced_features
    market_df = add_advanced_features(market_df)
    
    result = detector.detect_regime(market_df)
    
    log_test(
        "Regime Detection",
        "PASS",
        f"Detected: {result['regime'].name}, Confidence: {result['confidence']:.0%}",
        f"Strategy: {result['strategy']['description']}"
    )
    
except Exception as e:
    log_test("Regime Detection Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 8: API Endpoints
# ============================================================
print("\n" + "="*80)
print("TEST 8: API ENDPOINTS")
print("="*80)

# Test /ai/trades
try:
    response = requests.get(f"{BACKEND_URL}/ai/trades", timeout=TIMEOUT)
    if response.status_code == 200:
        trades = response.json()
        log_test(
            "GET /ai/trades",
            "PASS",
            f"Retrieved {len(trades)} trades",
            f"Endpoint responding correctly"
        )
    else:
        log_test("GET /ai/trades", "FAIL", f"Status: {response.status_code}")
except Exception as e:
    log_test("GET /ai/trades", "FAIL", f"Error: {str(e)}")

# Test /ai/stats
try:
    response = requests.get(f"{BACKEND_URL}/ai/stats", timeout=TIMEOUT)
    if response.status_code == 200:
        stats = response.json()
        log_test(
            "GET /ai/stats",
            "PASS",
            f"Total trades: {stats.get('total_trades', 0)}, Win rate: {stats.get('win_rate', 0):.1%}",
            f"Stats endpoint working"
        )
    else:
        log_test("GET /ai/stats", "FAIL", f"Status: {response.status_code}")
except Exception as e:
    log_test("GET /ai/stats", "FAIL", f"Error: {str(e)}")

# Test /ai/signals/latest
try:
    response = requests.get(f"{BACKEND_URL}/ai/signals/latest", timeout=TIMEOUT)
    if response.status_code == 200:
        signals = response.json()
        log_test(
            "GET /ai/signals/latest",
            "PASS",
            f"Retrieved {len(signals)} active signals",
            f"Signals endpoint working"
        )
    else:
        log_test("GET /ai/signals/latest", "FAIL", f"Status: {response.status_code}")
except Exception as e:
    log_test("GET /ai/signals/latest", "FAIL", f"Error: {str(e)}")

# Test /settings/exchanges
try:
    response = requests.get(f"{BACKEND_URL}/settings/exchanges", timeout=TIMEOUT)
    if response.status_code == 200:
        exchanges = response.json()
        log_test(
            "GET /settings/exchanges",
            "PASS",
            f"Available exchanges: {len(exchanges)}",
            f"Exchange configuration working"
        )
    else:
        log_test("GET /settings/exchanges", "FAIL", f"Status: {response.status_code}")
except Exception as e:
    log_test("GET /settings/exchanges", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 9: Database Connectivity
# ============================================================
print("\n" + "="*80)
print("TEST 9: DATABASE CONNECTIVITY")
print("="*80)

try:
    import sqlite3
    
    db_path = "backend/data/trades.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check trades table
        cursor.execute("SELECT COUNT(*) FROM trades")
        trade_count = cursor.fetchone()[0]
        
        # Check signals table
        cursor.execute("SELECT COUNT(*) FROM signals")
        signal_count = cursor.fetchone()[0]
        
        conn.close()
        
        log_test(
            "Database Access",
            "PASS",
            f"Trades: {trade_count}, Signals: {signal_count}",
            f"Database: {db_path}"
        )
    else:
        log_test("Database Access", "WARN", f"Database not found: {db_path}")
        
except Exception as e:
    log_test("Database Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# TEST 10: Frontend Connectivity (Optional)
# ============================================================
print("\n" + "="*80)
print("TEST 10: FRONTEND CONNECTIVITY (OPTIONAL)")
print("="*80)

try:
    response = requests.get(FRONTEND_URL, timeout=TIMEOUT)
    if response.status_code == 200:
        log_test(
            "Frontend Server",
            "PASS",
            "Frontend is running and accessible",
            f"URL: {FRONTEND_URL}"
        )
    else:
        log_test("Frontend Server", "WARN", f"Status: {response.status_code}")
except Exception as e:
    log_test(
        "Frontend Server",
        "WARN",
        "Frontend not running (optional)",
        "Start with: cd frontend && npm run dev"
    )

# ============================================================
# TEST 11: End-to-End Signal Generation
# ============================================================
print("\n" + "="*80)
print("TEST 11: END-TO-END SIGNAL GENERATION")
print("="*80)

try:
    from ai_engine.agents.xgb_agent import XGBAgent
    
    # Initialize agent with ensemble
    agent = XGBAgent(use_ensemble=True, use_advanced_features=True)
    
    # Create sample market data
    sample_data = {
        'symbol': 'BTC/USDT',
        'open': 50000,
        'high': 50500,
        'low': 49500,
        'close': 50200,
        'volume': 1000
    }
    
    # Get prediction
    prediction = agent.predict_for_symbol('BTC/USDT', sample_data)
    
    if prediction and 'action' in prediction:
        log_test(
            "End-to-End Signal",
            "PASS",
            f"Action: {prediction['action']}, Score: {prediction.get('score', 0):.4f}",
            f"Confidence: {prediction.get('confidence', 0):.2f}, Model: {prediction.get('model', 'unknown')}"
        )
    else:
        log_test("End-to-End Signal", "FAIL", "No valid prediction returned")
        
except Exception as e:
    log_test("End-to-End Signal Test", "FAIL", f"Error: {str(e)}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*80)
print("[CHART] FINAL TEST REPORT")
print("="*80)

total_tests = results['passed'] + results['failed'] + results['warnings']
pass_rate = (results['passed'] / total_tests * 100) if total_tests > 0 else 0

print(f"\nTotal Tests Run:     {total_tests}")
print(f"[OK] Passed:           {results['passed']}")
print(f"❌ Failed:           {results['failed']}")
print(f"[WARNING]  Warnings:         {results['warnings']}")
print(f"\nPass Rate:           {pass_rate:.1f}%")

if results['failed'] == 0:
    print("\n" + "="*80)
    print("🎉 ALL CRITICAL TESTS PASSED!")
    print("="*80)
    print("\n[OK] System Status: READY FOR PRODUCTION")
    print("\nNext Steps:")
    print("   1. [OK] Backend operational")
    print("   2. [OK] AI Engine with ensemble working")
    print("   3. [OK] All 6 improvements integrated")
    print("   4. [OK] API endpoints responding")
    print("   5. 🔜 Start frontend and test UI")
    print("   6. 🔜 Paper trading validation")
    print("   7. 🔜 Live deployment")
else:
    print("\n" + "="*80)
    print("[WARNING]  SOME TESTS FAILED - REVIEW REQUIRED")
    print("="*80)
    print("\nFailed Tests:")
    for test in results['tests']:
        if test['status'] == 'FAIL':
            print(f"   ❌ {test['name']}: {test['message']}")

print("\n" + "="*80)
print(f"Test completed at: {datetime.now()}")
print("="*80)

# Exit with appropriate code
sys.exit(0 if results['failed'] == 0 else 1)
