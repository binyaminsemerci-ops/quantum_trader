#!/usr/bin/env python3
"""
P0 MarketState SPEC v1.0 - Verification Script
Runs all proof commands and generates final report
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, desc):
    """Run command and capture output"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {desc}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║             P0 MarketState LOCKED SPEC v1.0 — VERIFICATION SCRIPT            ║
║                                                                              ║
║   Purpose: Verify that all SPEC v1.0 components work correctly              ║
║   Tests: Unit tests + Replay tool + Integration check                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # Test 1: Unit tests
    results['unit_tests'] = run_command(
        "python -m pytest ai_engine/tests/test_market_state_spec.py -v",
        "Unit Tests (14 tests)"
    )
    
    # Test 2: Replay TREND
    results['replay_trend'] = run_command(
        "python ops/replay_market_state.py --synthetic --regime trend",
        "Replay Script - TREND Regime"
    )
    
    # Test 3: Replay MEAN_REVERT
    results['replay_mr'] = run_command(
        "python ops/replay_market_state.py --synthetic --regime mean_revert",
        "Replay Script - MEAN_REVERT Regime"
    )
    
    # Test 4: Replay CHOP
    results['replay_chop'] = run_command(
        "python ops/replay_market_state.py --synthetic --regime chop",
        "Replay Script - CHOP Regime"
    )
    
    # Test 5: Import check
    results['import'] = run_command(
        "python -c \"from ai_engine.market_state import MarketState, DEFAULT_THETA; print('✅ Import successful'); print(f'Default window: {DEFAULT_THETA[\\\"vol\\\"][\\\"window\\\"]}'\"",
        "Import Check"
    )
    
    # Final Report
    print(f"\n\n{'='*80}")
    print("VERIFICATION REPORT")
    print(f"{'='*80}\n")
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:20s} {status}")
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED")
        print("\nP0 MarketState LOCKED SPEC v1.0 is PRODUCTION-READY")
        print("\nFiles:")
        print("  - ai_engine/market_state.py (15KB)")
        print("  - ai_engine/tests/test_market_state_spec.py (7.4KB)")
        print("  - ops/replay_market_state.py (6KB)")
        print("\nCommit: cb99feb7 + 941d6c3b")
        print("\nReady for:")
        print("  - Integration with AI Engine")
        print("  - Live price feed connection")
        print("  - Position sizing / leverage adjustment")
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("\nPlease review errors above and fix before deploying")
        sys.exit(1)
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
