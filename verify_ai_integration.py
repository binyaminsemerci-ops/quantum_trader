"""
AI-OS INTEGRATION VERIFICATION SCRIPT
======================================

Quick script to verify that all AI-OS subsystems are properly integrated and callable.

Usage:
    python verify_ai_integration.py
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.services.system_services import get_ai_services, AISystemConfig
from backend.services.integration_hooks import (
    pre_trade_universe_filter,
    pre_trade_risk_check,
    pre_trade_portfolio_check,
    pre_trade_confidence_adjustment,
    pre_trade_position_sizing,
    execution_order_type_selection,
    execution_slippage_check,
    post_trade_position_classification,
    post_trade_amplification_check,
    periodic_self_healing_check,
    periodic_ai_hfos_coordination,
    get_integration_summary,
)


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"ℹ️  {text}")


async def test_service_registry():
    """Test service registry initialization"""
    print_header("SERVICE REGISTRY TEST")
    
    try:
        # Get AI services singleton
        services = get_ai_services()
        print_success("AISystemServices singleton accessible")
        
        # Check config
        config = services.config
        print_success(f"Configuration loaded - Stage: {config.integration_stage.value}")
        
        # Get status
        status = services.get_status()
        print_success("Service registry status accessible")
        
        print_info(f"Integration Stage: {status['integration_stage']}")
        print_info(f"Emergency Brake: {status['emergency_brake']}")
        
        print("\nService Status:")
        for service, state in status.get('services_status', {}).items():
            if state == "initialized":
                print_success(f"  {service}: {state}")
            elif state == "disabled":
                print_warning(f"  {service}: {state}")
            elif state == "failed":
                print_error(f"  {service}: {state}")
            else:
                print_info(f"  {service}: {state}")
        
        return True
    
    except Exception as e:
        print_error(f"Service registry test failed: {e}")
        return False


async def test_pre_trade_hooks():
    """Test pre-trade integration hooks"""
    print_header("PRE-TRADE HOOKS TEST")
    
    hooks_passed = 0
    hooks_total = 5
    
    # Test universe filter
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        filtered = await pre_trade_universe_filter(symbols)
        print_success(f"pre_trade_universe_filter: {len(symbols)} → {len(filtered)} symbols")
        hooks_passed += 1
    except Exception as e:
        print_error(f"pre_trade_universe_filter failed: {e}")
    
    # Test risk check
    try:
        signal = {"symbol": "BTCUSDT", "action": "BUY", "confidence": 0.75}
        allowed, reason = await pre_trade_risk_check("BTCUSDT", signal, {})
        status = "ALLOWED" if allowed else f"BLOCKED ({reason})"
        print_success(f"pre_trade_risk_check: {status}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"pre_trade_risk_check failed: {e}")
    
    # Test portfolio check
    try:
        signal = {"symbol": "ETHUSDT", "action": "BUY", "confidence": 0.80}
        allowed, reason = await pre_trade_portfolio_check("ETHUSDT", signal, {})
        status = "ALLOWED" if allowed else f"BLOCKED ({reason})"
        print_success(f"pre_trade_portfolio_check: {status}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"pre_trade_portfolio_check failed: {e}")
    
    # Test confidence adjustment
    try:
        signal = {"symbol": "BTCUSDT", "confidence": 0.75}
        adjusted = await pre_trade_confidence_adjustment(signal, 0.72)
        print_success(f"pre_trade_confidence_adjustment: 0.72 → {adjusted:.2f}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"pre_trade_confidence_adjustment failed: {e}")
    
    # Test position sizing
    try:
        signal = {"symbol": "BTCUSDT", "confidence": 0.80}
        adjusted = await pre_trade_position_sizing("BTCUSDT", signal, 5000.0)
        print_success(f"pre_trade_position_sizing: $5000.00 → ${adjusted:.2f}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"pre_trade_position_sizing failed: {e}")
    
    print_info(f"\nPre-trade hooks: {hooks_passed}/{hooks_total} passed")
    return hooks_passed == hooks_total


async def test_execution_hooks():
    """Test execution integration hooks"""
    print_header("EXECUTION HOOKS TEST")
    
    hooks_passed = 0
    hooks_total = 2
    
    # Test order type selection
    try:
        signal = {"symbol": "BTCUSDT", "confidence": 0.75}
        order_type = await execution_order_type_selection("BTCUSDT", signal, "MARKET")
        print_success(f"execution_order_type_selection: MARKET → {order_type}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"execution_order_type_selection failed: {e}")
    
    # Test slippage check
    try:
        acceptable, reason = await execution_slippage_check("BTCUSDT", 50000.0, 50050.0)
        status = "ACCEPTABLE" if acceptable else f"REJECTED ({reason})"
        print_success(f"execution_slippage_check: {status}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"execution_slippage_check failed: {e}")
    
    print_info(f"\nExecution hooks: {hooks_passed}/{hooks_total} passed")
    return hooks_passed == hooks_total


async def test_post_trade_hooks():
    """Test post-trade integration hooks"""
    print_header("POST-TRADE HOOKS TEST")
    
    hooks_passed = 0
    hooks_total = 2
    
    # Test position classification
    try:
        position = {
            "symbol": "BTCUSDT",
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "quantity": 0.1
        }
        classified = await post_trade_position_classification(position)
        print_success(f"post_trade_position_classification: {classified.get('symbol', 'UNKNOWN')}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"post_trade_position_classification failed: {e}")
    
    # Test amplification check
    try:
        position = {
            "symbol": "BTCUSDT",
            "entry_price": 50000.0,
            "current_price": 51000.0,
            "unrealized_pnl": 100.0
        }
        recommendation = await post_trade_amplification_check(position)
        status = f"Recommendation: {recommendation}" if recommendation else "No amplification"
        print_success(f"post_trade_amplification_check: {status}")
        hooks_passed += 1
    except Exception as e:
        print_error(f"post_trade_amplification_check failed: {e}")
    
    print_info(f"\nPost-trade hooks: {hooks_passed}/{hooks_total} passed")
    return hooks_passed == hooks_total


async def test_periodic_hooks():
    """Test periodic integration hooks"""
    print_header("PERIODIC HOOKS TEST")
    
    hooks_passed = 0
    hooks_total = 2
    
    # Test self-healing check
    try:
        await periodic_self_healing_check()
        print_success("periodic_self_healing_check: Executed successfully")
        hooks_passed += 1
    except Exception as e:
        print_error(f"periodic_self_healing_check failed: {e}")
    
    # Test AI-HFOS coordination
    try:
        await periodic_ai_hfos_coordination()
        print_success("periodic_ai_hfos_coordination: Executed successfully")
        hooks_passed += 1
    except Exception as e:
        print_error(f"periodic_ai_hfos_coordination failed: {e}")
    
    print_info(f"\nPeriodic hooks: {hooks_passed}/{hooks_total} passed")
    return hooks_passed == hooks_total


async def test_integration_summary():
    """Test integration summary"""
    print_header("INTEGRATION SUMMARY TEST")
    
    try:
        summary = get_integration_summary()
        print_success("Integration summary accessible")
        
        print_info(f"Stage: {summary['stage']}")
        print_info(f"Enabled Subsystems: {len(summary['enabled_subsystems'])}")
        print_info(f"Emergency Brake: {summary['emergency_brake']}")
        print_info(f"AI-HFOS Active: {summary['ai_hfos_active']}")
        
        print("\nEnabled Subsystems:")
        for subsystem in summary['enabled_subsystems']:
            print_success(f"  {subsystem}")
        
        return True
    
    except Exception as e:
        print_error(f"Integration summary test failed: {e}")
        return False


async def main():
    """Run all verification tests"""
    print(f"{Colors.BOLD}AI-OS INTEGRATION VERIFICATION{Colors.END}")
    print(f"{'='*70}\n")
    
    results = {
        "Service Registry": await test_service_registry(),
        "Pre-Trade Hooks": await test_pre_trade_hooks(),
        "Execution Hooks": await test_execution_hooks(),
        "Post-Trade Hooks": await test_post_trade_hooks(),
        "Periodic Hooks": await test_periodic_hooks(),
        "Integration Summary": await test_integration_summary(),
    }
    
    # Final summary
    print_header("VERIFICATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED - Integration is OPERATIONAL{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Some tests failed - Review errors above{Colors.END}\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
