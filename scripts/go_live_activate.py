#!/usr/bin/env python3
"""
GO-LIVE Activation Script

This script performs all pre-flight checks and activates real trading
if all safety conditions are met.

Usage:
    python scripts/go_live_activate.py

Exit codes:
    0 - GO-LIVE activated successfully
    1 - Pre-flight checks failed or activation denied
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.preflight.checks import run_all_preflight_checks
from backend.go_live.activation import go_live_activate


def print_banner():
    """Print activation banner."""
    print()
    print("=" * 80)
    print("QUANTUM TRADER v2.0 - GO-LIVE ACTIVATION")
    print("=" * 80)
    print()
    print("⚠️  WARNING: This will enable REAL TRADING with REAL MONEY")
    print()


async def main():
    """Main activation workflow."""
    parser = argparse.ArgumentParser(
        description="GO-LIVE Activation Script - Activate real trading"
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="Senior Operator (PROMPT 10)",
        help="Name of operator performing activation (for audit trail)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks (NOT RECOMMENDED)",
    )
    args = parser.parse_args()

    print_banner()

    # Step 1: Run preflight checks (unless skipped)
    if not args.skip_preflight:
        print("🔍 Running preflight checks...")
        print()

        results = await run_all_preflight_checks()
        failed = [r for r in results if not r.success]

        # Display results
        for result in results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} | {result.name}")
            if not result.success:
                print(f"         Reason: {result.reason}")
                if result.details:
                    for key, value in result.details.items():
                        print(f"         {key}: {value}")

        print()
        print("=" * 80)
        print(f"PREFLIGHT RESULTS: {len(results) - len(failed)} passed, {len(failed)} failed")
        print("=" * 80)
        print()

        if failed:
            print("❌ PREFLIGHT CHECKS FAILED - Cannot activate GO-LIVE")
            print()
            print("Failed checks:")
            for fail in failed:
                print(f"  - {fail.name}: {fail.reason}")
            print()
            print("💡 Fix the above issues and try again.")
            print()
            return 1

        print("✅ All preflight checks PASSED")
        print()
    else:
        print("⚠️  SKIPPING preflight checks (--skip-preflight flag)")
        print()

    # Step 2: Attempt activation
    print("🚀 Attempting GO-LIVE activation...")
    print()

    try:
        success = await go_live_activate(operator=args.operator)

        if not success:
            print()
            print("=" * 80)
            print("❌ GO-LIVE ACTIVATION FAILED")
            print("=" * 80)
            print()
            print("Possible reasons:")
            print("  - activation_enabled=false in config/go_live.yaml")
            print("  - Risk state is not OK")
            print("  - ESS is active")
            print("  - Insufficient testnet history")
            print()
            print("💡 Check logs for details and resolve issues.")
            print()
            return 1

        print()
        print("=" * 80)
        print("✅ GO-LIVE ACTIVATION SUCCESSFUL")
        print("=" * 80)
        print()
        print("REAL TRADING is now ENABLED.")
        print()
        print("Next steps:")
        print("  1. Monitor Risk & Resilience Dashboard continuously")
        print("  2. Watch for first real order execution")
        print("  3. Follow 'First Hour Monitoring' checklist")
        print("  4. Document activation in operational log")
        print()
        print("To DEACTIVATE:")
        print("  - Run: python scripts/go_live_deactivate.py")
        print("  - Or delete: go_live.active")
        print("  - Or edit config: activation_enabled=false")
        print()
        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ GO-LIVE ACTIVATION FAILED (EXCEPTION)")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
