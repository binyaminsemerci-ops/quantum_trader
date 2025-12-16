"""
Quantum Trader v2.0 Pre-Flight Check Script

EPIC-PREFLIGHT-001: Pre-flight validation before enabling REAL trading.

Usage:
    python scripts/preflight_check.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main():
    """
    Run all pre-flight checks and report results.
    
    Returns exit code 0 if all checks pass, 1 if any fail.
    """
    print("=" * 80)
    print("QUANTUM TRADER v2.0 - PRE-FLIGHT CHECK")
    print("=" * 80)
    print()
    
    # Import here to avoid circular dependencies
    from backend.preflight.checks import run_all_preflight_checks
    
    print("Running pre-flight checks...\n")
    
    try:
        results = await run_all_preflight_checks()
    except Exception as e:
        print(f"❌ CRITICAL: Pre-flight check runner failed: {e}")
        return 1
    
    # Display results
    passed = 0
    failed = 0
    
    for result in results:
        if result.success:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status} | {result.name}")
        print(f"       Reason: {result.reason}")
        
        if result.details:
            for key, value in result.details.items():
                print(f"       {key}: {value}")
        print()
    
    # Summary
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        print("\n⚠️  PRE-FLIGHT CHECK FAILED - DO NOT ENABLE REAL TRADING")
        return 1
    else:
        print("\n✅ PRE-FLIGHT CHECK PASSED - System ready for trading")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
