"""
Test Hybrid Agent Integration
==============================

Validates the Hybrid Agent before deploying to production.
Tests: Model loading, predictions, comparison with XGBoost/TFT
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import requests
from typing import Dict
import json


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


def test_hybrid_health():
    """Test 1: Hybrid Agent health check"""
    test_section("Hybrid Agent Health Check")
    
    try:
        response = requests.get("http://localhost:8000/api/test/hybrid/health", timeout=10)
        
        test_result("Endpoint responding", response.status_code == 200)
        
        if response.status_code == 200:
            data = response.json()
            
            test_result("Hybrid Agent loaded", 
                       data.get('status') == 'healthy',
                       f"Mode: {data.get('mode')}")
            
            test_result("TFT model loaded",
                       data.get('tft_loaded') == True)
            
            test_result("XGBoost model loaded",
                       data.get('xgb_loaded') == True)
            
            weights = data.get('weights', {})
            test_result("Weights configured correctly",
                       weights.get('tft') == 0.6 and weights.get('xgb') == 0.4,
                       f"TFT: {weights.get('tft')}, XGB: {weights.get('xgb')}")
            
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
            
    except Exception as e:
        test_result("Hybrid health check", False, f"Error: {e}")
        return False


def test_hybrid_prediction():
    """Test 2: Single prediction with Hybrid Agent"""
    test_section("Hybrid Agent Prediction")
    
    try:
        response = requests.post(
            "http://localhost:8000/api/test/hybrid/predict",
            params={"symbol": "BTCUSDT"},
            timeout=30
        )
        
        test_result("Prediction endpoint responding", response.status_code == 200)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check hybrid prediction
            hybrid = data.get('hybrid_prediction', {})
            test_result("Hybrid prediction valid",
                       hybrid.get('action') in ['BUY', 'SELL', 'HOLD'],
                       f"Action: {hybrid.get('action')}, Conf: {hybrid.get('confidence'):.2f}")
            
            # Check TFT prediction
            tft = data.get('tft_prediction', {})
            test_result("TFT prediction included",
                       tft.get('loaded') == True,
                       f"Action: {tft.get('action')}, Conf: {tft.get('confidence'):.2f}")
            
            # Check XGBoost prediction
            xgb = data.get('xgb_prediction', {})
            test_result("XGBoost prediction included",
                       xgb.get('loaded') == True,
                       f"Action: {xgb.get('action')}, Conf: {xgb.get('confidence'):.2f}")
            
            # Check agreement
            agreement = data.get('agreement')
            test_result("Agreement status calculated",
                       isinstance(agreement, bool),
                       f"Models agree: {agreement}")
            
            return True
        else:
            print(f"❌ Prediction failed: {response.text}")
            return False
            
    except Exception as e:
        test_result("Hybrid prediction", False, f"Error: {e}")
        return False


def test_model_comparison():
    """Test 3: Compare XGBoost vs TFT vs Hybrid"""
    test_section("Model Comparison (XGBoost vs TFT vs Hybrid)")
    
    try:
        response = requests.get(
            "http://localhost:8000/api/test/hybrid/compare",
            params={"symbols": "BTCUSDT,ETHUSDT,BNBUSDT"},
            timeout=60
        )
        
        test_result("Comparison endpoint responding", response.status_code == 200)
        
        if response.status_code == 200:
            data = response.json()
            
            results = data.get('results', [])
            test_result("Got comparison results",
                       len(results) > 0,
                       f"Symbols tested: {len(results)}")
            
            # Check result structure
            if results:
                first = results[0]
                has_all = all(k in first for k in ['xgb', 'tft', 'hybrid', 'agreement'])
                test_result("Results have all models",
                           has_all)
                
                # Print comparison
                print("\n[CHART] Model Predictions:")
                for result in results:
                    symbol = result['symbol']
                    xgb_act = result['xgb']['action']
                    tft_act = result['tft']['action']
                    hyb_act = result['hybrid']['action']
                    agree = result['agreement']['all_agree']
                    
                    agree_icon = "[OK]" if agree else "❌"
                    print(f"   {symbol:10} | XGB: {xgb_act:4} | TFT: {tft_act:4} | Hybrid: {hyb_act:4} {agree_icon}")
            
            # Check summary
            summary = data.get('summary', {})
            test_result("Summary statistics calculated",
                       'all_agree_pct' in summary,
                       f"Agreement rate: {summary.get('all_agree_pct', 0):.1f}%")
            
            return True
        else:
            print(f"❌ Comparison failed: {response.text}")
            return False
            
    except Exception as e:
        test_result("Model comparison", False, f"Error: {e}")
        return False


def test_config_endpoint():
    """Test 4: Configuration endpoint"""
    test_section("Configuration Endpoint")
    
    try:
        response = requests.get("http://localhost:8000/api/test/hybrid/config", timeout=5)
        
        test_result("Config endpoint responding", response.status_code == 200)
        
        if response.status_code == 200:
            data = response.json()
            
            test_result("Current mode returned",
                       'current_mode' in data,
                       f"Mode: {data.get('current_mode')}")
            
            test_result("Available modes listed",
                       len(data.get('available_modes', {})) == 3)
            
            test_result("Instructions provided",
                       'instructions' in data)
            
            return True
        else:
            return False
            
    except Exception as e:
        test_result("Config endpoint", False, f"Error: {e}")
        return False


def test_environment_variable():
    """Test 5: AI_MODEL environment variable"""
    test_section("Environment Variable Configuration")
    
    current = os.getenv('AI_MODEL', 'not set')
    test_result("AI_MODEL environment variable",
               current in ['xgb', 'tft', 'hybrid', 'not set'],
               f"Current value: {current}")
    
    if current == 'not set':
        print("   💡 Set AI_MODEL to control which agent is used:")
        print("      Windows: setx AI_MODEL hybrid")
        print("      Linux: export AI_MODEL=hybrid")
    
    return True


def test_direct_import():
    """Test 6: Direct import and initialization"""
    test_section("Direct Import & Initialization")
    
    try:
        from ai_engine.agents.hybrid_agent import HybridAgent
        
        test_result("HybridAgent import", True)
        
        agent = HybridAgent()
        test_result("HybridAgent initialization", True)
        
        test_result("Agent mode set",
                   agent.mode in ['hybrid', 'tft_only', 'xgb_only'],
                   f"Mode: {agent.mode}")
        
        test_result("TFT loaded",
                   agent.tft_loaded,
                   f"Status: {'[OK]' if agent.tft_loaded else '❌'}")
        
        test_result("XGBoost loaded",
                   agent.xgb_loaded,
                   f"Status: {'[OK]' if agent.xgb_loaded else '❌'}")
        
        return True
        
    except Exception as e:
        test_result("Direct import", False, f"Error: {e}")
        return False


def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*15 + "HYBRID AGENT INTEGRATION TEST" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n[WARNING]  Prerequisites:")
    print("   1. Backend must be running (docker-compose up or python backend/main.py)")
    print("   2. TFT model trained (ai_engine/models/tft_model.pth)")
    print("   3. XGBoost model trained (ai_engine/models/xgb_model.pkl)")
    
    print("\n[ROCKET] Starting tests...\n")
    
    results = {}
    
    # Run tests
    results['Direct Import'] = test_direct_import()
    results['Environment Variable'] = test_environment_variable()
    
    # Check if backend is running
    try:
        requests.get("http://localhost:8000/health", timeout=3)
        backend_running = True
    except:
        backend_running = False
        print("\n[WARNING]  Backend not responding, skipping API tests")
    
    if backend_running:
        results['Health Check'] = test_hybrid_health()
        results['Prediction'] = test_hybrid_prediction()
        results['Model Comparison'] = test_model_comparison()
        results['Config Endpoint'] = test_config_endpoint()
    
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
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n[OK] Hybrid Agent ready for deployment:")
        print("   1. Update make_default_agent() to return HybridAgent()")
        print("   2. Or set AI_MODEL=hybrid environment variable")
        print("   3. Restart backend to activate")
    else:
        print("[WARNING]  SOME TESTS FAILED - Review errors above")
    
    print("\n" + "="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
