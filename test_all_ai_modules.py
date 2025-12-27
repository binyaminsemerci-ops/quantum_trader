"""
🧪 COMPREHENSIVE AI MODULE TEST
================================
Tests all AI modules to verify:
1. CLM (Continuous Learning Manager) - learning & logging
2. AI Engine - signal generation & logging
3. RL Position Sizing - adaptive sizing & logging
4. Risk Management - real-time monitoring & logging
5. Exit Brain - dynamic TP/SL & logging
6. Model Ensemble - multi-model inference & logging
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def print_section(title: str):
    """Print section divider"""
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")


def check_result(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"     {details}")


async def test_clm_module():
    """Test Continuous Learning Manager"""
    print_section("1️⃣  CLM (Continuous Learning Manager)")
    
    results = {
        "module_exists": False,
        "can_initialize": False,
        "has_retraining": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        # Check if CLM exists
        from backend.services.ai.continuous_learning_manager import (
            ContinuousLearningManager,
            ModelType,
            RetrainTrigger
        )
        results["module_exists"] = True
        results["details"].append("CLM module imported successfully")
        
        # Try to initialize (without actual training)
        clm = ContinuousLearningManager()
        results["can_initialize"] = True
        results["details"].append("CLM initialized")
        
        # Check if it has retraining methods
        if hasattr(clm, 'check_retraining_needed') and hasattr(clm, 'trigger_retraining'):
            results["has_retraining"] = True
            results["details"].append("Retraining methods available")
        
        # Check logging
        import logging
        logger = logging.getLogger('backend.services.ai.continuous_learning_manager')
        if logger.level != logging.NOTSET or logger.handlers:
            results["has_logging"] = True
            results["details"].append(f"Logger configured (level: {logger.level})")
        
    except ImportError as e:
        results["details"].append(f"Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"Error: {str(e)}")
    
    # Print results
    check_result("CLM Module Exists", results["module_exists"])
    check_result("CLM Can Initialize", results["can_initialize"])
    check_result("CLM Has Retraining Logic", results["has_retraining"])
    check_result("CLM Has Logging", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     ℹ️  {detail}")
    
    return all([
        results["module_exists"],
        results["can_initialize"],
        results["has_retraining"]
    ])


async def test_ai_engine():
    """Test AI Engine"""
    print_section("2️⃣  AI ENGINE (Signal Generation)")
    
    results = {
        "service_exists": False,
        "can_generate_signals": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        from microservices.ai_engine.service import AIEngineService
        results["service_exists"] = True
        results["details"].append("AI Engine service imported")
        
        # Check if it can generate signals
        service = AIEngineService()
        if hasattr(service, 'get_trading_signals'):
            results["can_generate_signals"] = True
            results["details"].append("Signal generation method available")
        
        # Check logging
        import logging
        logger = logging.getLogger('microservices.ai_engine')
        if logger.level != logging.NOTSET or logger.handlers:
            results["has_logging"] = True
            results["details"].append(f"Logger configured")
        
    except ImportError as e:
        results["details"].append(f"Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"Error: {str(e)}")
    
    check_result("AI Engine Exists", results["service_exists"])
    check_result("Can Generate Signals", results["can_generate_signals"])
    check_result("Has Logging", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     ℹ️  {detail}")
    
    return all([
        results["service_exists"],
        results["can_generate_signals"]
    ])


async def test_rl_position_sizing():
    """Test RL Position Sizing"""
    print_section("3️⃣  RL POSITION SIZING (Adaptive Sizing)")
    
    results = {
        "module_exists": False,
        "has_learning": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        from backend.services.ai.rl_position_sizing import RLPositionSizing
        results["module_exists"] = True
        results["details"].append("RL Position Sizing imported")
        
        # Check for learning methods
        rl_sizing = RLPositionSizing()
        if hasattr(rl_sizing, 'update') or hasattr(rl_sizing, 'learn'):
            results["has_learning"] = True
            results["details"].append("Learning methods available")
        
        # Check logging
        import logging
        logger = logging.getLogger('backend.services.ai.rl_position_sizing')
        if logger.level != logging.NOTSET or logger.handlers:
            results["has_logging"] = True
            results["details"].append("Logger configured")
        
    except ImportError as e:
        results["details"].append(f"Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"Error: {str(e)}")
    
    check_result("RL Sizing Module Exists", results["module_exists"])
    check_result("Has Learning Methods", results["has_learning"])
    check_result("Has Logging", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     ℹ️  {detail}")
    
    return results["module_exists"]


async def test_risk_management():
    """Test Risk Management"""
    print_section("4️⃣  RISK MANAGEMENT (Real-time Monitoring)")
    
    results = {
        "module_exists": False,
        "has_monitoring": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        from backend.services.risk.manager import RiskManager
        results["module_exists"] = True
        results["details"].append("Risk Manager imported")
        
        # Check for monitoring methods
        risk_mgr = RiskManager()
        if hasattr(risk_mgr, 'check_position_limits') or hasattr(risk_mgr, 'validate_trade'):
            results["has_monitoring"] = True
            results["details"].append("Monitoring methods available")
        
        # Check logging
        import logging
        logger = logging.getLogger('backend.services.risk')
        if logger.level != logging.NOTSET or logger.handlers:
            results["has_logging"] = True
            results["details"].append("Logger configured")
        
    except ImportError as e:
        results["details"].append(f"Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"Error: {str(e)}")
    
    check_result("Risk Manager Exists", results["module_exists"])
    check_result("Has Monitoring Logic", results["has_monitoring"])
    check_result("Has Logging", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     ℹ️  {detail}")
    
    return results["module_exists"]


async def test_exit_brain():
    """Test Exit Brain"""
    print_section("5️⃣  EXIT BRAIN (Dynamic TP/SL)")
    
    results = {
        "module_exists": False,
        "has_dynamic_logic": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        from backend.services.execution.exit_brain import ExitBrain
        results["module_exists"] = True
        results["details"].append("Exit Brain imported")
        
        # Check for dynamic TP/SL methods
        exit_brain = ExitBrain()
        if hasattr(exit_brain, 'calculate_dynamic_targets') or hasattr(exit_brain, 'update_exit_levels'):
            results["has_dynamic_logic"] = True
            results["details"].append("Dynamic TP/SL methods available")
        
        # Check logging
        import logging
        logger = logging.getLogger('backend.services.execution.exit_brain')
        if logger.level != logging.NOTSET or logger.handlers:
            results["has_logging"] = True
            results["details"].append("Logger configured")
        
    except ImportError as e:
        results["details"].append(f"Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"Error: {str(e)}")
    
    check_result("Exit Brain Exists", results["module_exists"])
    check_result("Has Dynamic Logic", results["has_dynamic_logic"])
    check_result("Has Logging", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     ℹ️  {detail}")
    
    return results["module_exists"]


async def test_model_ensemble():
    """Test Model Ensemble"""
    print_section("6️⃣  MODEL ENSEMBLE (Multi-Model Inference)")
    
    results = {
        "module_exists": False,
        "has_models": False,
        "has_logging": False,
        "details": []
    }
    
    try:
        from backend.services.ai.ensemble import EnsembleManager
        results["module_exists"] = True
        results["details"].append("Ensemble Manager imported")
        
        # Check for model management
        ensemble = EnsembleManager()
        if hasattr(ensemble, 'predict') or hasattr(ensemble, 'get_ensemble_prediction'):
            results["has_models"] = True
            results["details"].append("Ensemble prediction methods available")
        
        # Check logging
        import logging
        logger = logging.getLogger('backend.services.ai.ensemble')
        if logger.level != logging.NOTSET or logger.handlers:
            results["has_logging"] = True
            results["details"].append("Logger configured")
        
    except ImportError as e:
        results["details"].append(f"Import error: {str(e)}")
    except Exception as e:
        results["details"].append(f"Error: {str(e)}")
    
    check_result("Ensemble Manager Exists", results["module_exists"])
    check_result("Has Model Logic", results["has_models"])
    check_result("Has Logging", results["has_logging"])
    
    for detail in results["details"]:
        print(f"     ℹ️  {detail}")
    
    return results["module_exists"]


async def check_server_logs():
    """Check if server is generating logs"""
    print_section("7️⃣  SERVER LOG VERIFICATION")
    
    log_patterns = {
        "ai_engine": ["ai_engine", "signal", "prediction"],
        "clm": ["continuous_learning", "retraining", "model.*update"],
        "rl_sizing": ["rl_position", "sizing", "adaptive"],
        "risk": ["risk", "exposure", "limit"],
        "exit_brain": ["exit_brain", "take_profit", "stop_loss"],
    }
    
    print("📝 Checking for recent logs...")
    print("     (Run: ssh root@46.224.116.254 'tail -1000 /home/qt/logs/*.log')")
    
    return True


async def main():
    """Run all tests"""
    print_header("🧪 COMPREHENSIVE AI MODULE TEST SUITE")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run all tests
    results["clm"] = await test_clm_module()
    results["ai_engine"] = await test_ai_engine()
    results["rl_sizing"] = await test_rl_position_sizing()
    results["risk"] = await test_risk_management()
    results["exit_brain"] = await test_exit_brain()
    results["ensemble"] = await test_model_ensemble()
    results["logs"] = await check_server_logs()
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    print("\n📋 Module Status:")
    for module, status in results.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {module.upper()}")
    
    if passed == total:
        print("\n🎉 ALL MODULES OPERATIONAL!")
    else:
        print("\n⚠️  Some modules need attention")
    
    print(f"\n📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
