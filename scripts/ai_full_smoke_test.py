#!/usr/bin/env python3
"""
AI Full Smoke Test
==================
Test all 24 AI modules with synthetic inputs.
Target: 100% pass rate.

Usage: python scripts/ai_full_smoke_test.py
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.domains.ai import (
    run_full_ai_healthcheck,
    calculate_health_score,
    AI_MODULES
)


async def main():
    print("=" * 80)
    print("AI MODULE SMOKE TEST - Quantum Trader v2.0")
    print("=" * 80)
    print(f"Total modules registered: {len(AI_MODULES)}")
    print()
    
    # Run health checks
    print("Running health checks on all modules...")
    print()
    
    results = await run_full_ai_healthcheck()
    
    # Calculate score
    score = calculate_health_score(results)
    
    # Print results
    print("-" * 80)
    print("RESULTS BY MODULE")
    print("-" * 80)
    
    passed = []
    failed = []
    
    for result in sorted(results, key=lambda r: (not r.ok, r.name)):
        status = "PASS" if result.ok else "FAIL"
        latency = f"{result.latency_ms:.2f}ms" if result.latency_ms else "N/A"
        
        print(f"{status:8} | {result.name:35} | {latency:10}")
        
        if not result.ok:
            print(f"         | Error: {result.error}")
            failed.append((result.name, result.error))
        else:
            passed.append(result.name)
    
    print("-" * 80)
    print()
    print("SUMMARY")
    print("-" * 80)
    print(f"Total:      {score['total']}")
    print(f"Passed:     {score['pass']} ({score['pass_rate']:.1f}%)")
    print(f"Failed:     {score['fail']}")
    if score['avg_latency_ms']:
        print(f"Avg Latency: {score['avg_latency_ms']:.2f}ms")
    print()
    
    if failed:
        print("FAILED MODULES:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print()
    
    # Exit code
    if score['pass_rate'] == 100.0:
        print("SUCCESS: ALL TESTS PASSED! System ready for production.")
        return 0
    else:
        print(f"WARNING: {score['fail']} module(s) failed. Fix required.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
