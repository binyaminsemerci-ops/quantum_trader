#!/usr/bin/env python3
"""
ESS Standalone Verification Script
==================================
Tests ESS functionality independently of backend startup.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()


async def verify_ess_imports():
    """Verify all ESS modules import successfully."""
    print("🔍 Verifying ESS imports...")
    try:
        from backend.services.risk.emergency_stop_system import (
            EmergencyStopSystem,
            EmergencyStopController,
            DrawdownEmergencyEvaluator,
            SystemHealthEmergencyEvaluator,
            ExecutionErrorEmergencyEvaluator,
            DataFeedEmergencyEvaluator,
            ManualTriggerEmergencyEvaluator,
            EmergencyState,
            ESSStatus,
        )
        from backend.services.ess_alerters import ESSAlertManager
        print("✅ All ESS modules import successfully")
        return True
    except ImportError as e:
        print(f"❌ ESS import failed: {e}")
        return False


async def verify_ess_configuration():
    """Verify ESS environment configuration."""
    print("\n🔍 Verifying ESS configuration...")
    
    config = {
        "QT_ESS_ENABLED": os.getenv("QT_ESS_ENABLED", "false"),
        "QT_ESS_MAX_DAILY_LOSS": os.getenv("QT_ESS_MAX_DAILY_LOSS", "10.0"),
        "QT_ESS_MAX_EQUITY_DD": os.getenv("QT_ESS_MAX_EQUITY_DD", "15.0"),
        "QT_ESS_MAX_SL_HITS": os.getenv("QT_ESS_MAX_SL_HITS", "5"),
        "QT_ESS_CHECK_INTERVAL": os.getenv("QT_ESS_CHECK_INTERVAL", "5"),
    }
    
    print("📋 Current ESS Configuration:")
    for key, value in config.items():
        print(f"   {key} = {value}")
    
    if config["QT_ESS_ENABLED"].lower() != "true":
        print("⚠️  WARNING: QT_ESS_ENABLED is not 'true'")
        return False
    
    print("✅ ESS configuration looks good")
    return True


async def verify_ess_basic_functionality():
    """Test basic ESS initialization."""
    print("\n🔍 Testing basic ESS functionality...")
    
    try:
        from backend.services.risk.emergency_stop_system import (
            EmergencyStopSystem,
            EmergencyStopController,
            ManualTriggerEmergencyEvaluator,
        )
        from backend.services.event_bus import InMemoryEventBus
        
        # Create minimal dependencies
        event_bus = InMemoryEventBus()
        
        # Create real controller instead of stub
        controller = EmergencyStopController(
            exchange=None,  # Not needed for test
            event_bus=event_bus,
            policy_store=None  # Not needed for test
        )
        
        # Create evaluator
        evaluator = ManualTriggerEmergencyEvaluator()
        
        # Create ESS instance
        ess = EmergencyStopSystem(
            evaluators=[evaluator],
            controller=controller,
            policy_store=None,
            check_interval_sec=1
        )
        
        print("✅ ESS instance created successfully")
        
        # Test manual trigger evaluator
        print("🧪 Testing manual trigger evaluator...")
        evaluator.trigger("Test trigger from verification script")
        
        # Check if evaluator detects the trigger
        triggered, reason = await evaluator.check()
        if triggered:
            print(f"✅ Manual trigger works - Evaluator detected: {reason}")
        else:
            print("❌ Manual trigger didn't work")
            return False
        
        # Reset and verify
        evaluator.reset()
        triggered, reason = await evaluator.check()
        if not triggered:
            print("✅ Reset works - Evaluator cleared")
        else:
            print("⚠️  Reset didn't clear trigger (unexpected)")
        
        # Test controller
        print("🧪 Testing emergency stop controller...")
        if not controller.is_active:
            print("✅ Controller starts inactive")
        
        # Activate
        await controller.activate("Test activation")
        if controller.is_active:
            print("✅ Controller activation works")
        else:
            print("❌ Controller activation failed")
            return False
        
        # Reset
        await controller.reset("test_verifier")
        if not controller.is_active:
            print("✅ Controller reset works")
        else:
            print("❌ Controller reset failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ESS functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_ess_alerters():
    """Test ESS alert manager initialization."""
    print("\n🔍 Testing ESS alerters...")
    
    try:
        from backend.services.ess_alerters import (
            ESSAlertManager,
            SlackAlerter,
            SMSAlerter,
            EmailAlerter,
        )
        
        # Use ESSAlertManager.from_env() to create alert manager properly
        alert_manager = ESSAlertManager.from_env()
        
        # Check how many alerters were configured
        alerter_count = len(alert_manager.alerters)
        
        if alerter_count > 0:
            print(f"✅ Alert manager created with {alerter_count} alerter(s)")
            for alerter in alert_manager.alerters:
                alerter_name = alerter.__class__.__name__
                print(f"   ✅ {alerter_name} configured")
        else:
            print("   ℹ️  No alerters configured (all disabled or credentials missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Alerter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all verification tests."""
    print("=" * 70)
    print("ESS STANDALONE VERIFICATION")
    print("=" * 70)
    
    results = []
    
    # Test imports
    results.append(("Imports", await verify_ess_imports()))
    
    # Test configuration
    results.append(("Configuration", await verify_ess_configuration()))
    
    # Test functionality
    results.append(("Functionality", await verify_ess_basic_functionality()))
    
    # Test alerters
    results.append(("Alerters", await verify_ess_alerters()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED - ESS is ready for deployment")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Fix issues before deploying ESS")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
