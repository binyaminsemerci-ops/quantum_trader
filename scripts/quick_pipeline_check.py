#!/usr/bin/env python3
"""
Quick Pipeline Health Check
============================
Lightweight check of critical pipeline components without waiting for full startup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_risk_v3():
    """Check Risk v3 modules load"""
    try:
        from backend.services.risk_v3 import RiskOrchestrator
        from backend.services.risk_v3.models import RiskLimits
        print("✓ Risk v3: Import OK")
        
        # Test profile scaling
        limits = RiskLimits(
            max_leverage=20.0,
            max_positions=10,
            max_daily_drawdown_pct=0.05
        )
        print(f"✓ Risk v3: Limits created (leverage={limits.max_leverage}x)")
        return True
    except Exception as e:
        print(f"✗ Risk v3: {e}")
        return False


def check_execution():
    """Check Execution modules load"""
    try:
        from backend.services.execution.event_driven_executor import EventDrivenExecutor
        print("✓ Execution: Import OK")
        return True
    except Exception as e:
        print(f"✗ Execution: {e}")
        return False


def check_portfolio():
    """Check Portfolio modules load"""
    try:
        from backend.services.execution.positions import PortfolioPositionService
        print("✓ Portfolio: Import OK")
        return True
    except Exception as e:
        print(f"✗ Portfolio: {e}")
        return False


def check_dashboard_bff():
    """Check Dashboard BFF modules load"""
    try:
        from backend.api.dashboard import bff_routes
        print("✓ Dashboard BFF: Import OK")
        
        # Check route exists
        routes = [r.path for r in bff_routes.router.routes]
        if "/trading" in routes:
            print("✓ Dashboard BFF: /trading route exists")
        return True
    except Exception as e:
        print(f"✗ Dashboard BFF: {e}")
        return False


def check_ai_modules():
    """Check AI modules"""
    try:
        from backend.domains.ai import AI_MODULES, runtime_coverage
        coverage = runtime_coverage()
        print(f"✓ AI Modules: {len(AI_MODULES)} registered, {coverage:.1f}% in runtime")
        return True
    except Exception as e:
        print(f"✗ AI Modules: {e}")
        return False


def main():
    print("="*60)
    print("PIPELINE COMPONENT HEALTH CHECK")
    print("="*60)
    
    results = {
        "AI Modules": check_ai_modules(),
        "Risk v3": check_risk_v3(),
        "Execution": check_execution(),
        "Portfolio": check_portfolio(),
        "Dashboard BFF": check_dashboard_bff(),
    }
    
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for component, status in results.items():
        status_str = "PASS" if status else "FAIL"
        print(f"{status_str:6} | {component}")
    
    print()
    print(f"Result: {passed}/{total} components healthy ({passed/total*100:.0f}%)")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
