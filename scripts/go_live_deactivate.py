#!/usr/bin/env python3
"""
GO-LIVE Deactivation Script

This script deactivates REAL TRADING by:
  1. Removing activation marker file
  2. Logging deactivation reason
  3. Updating rollback metadata

Usage:
    python scripts/go_live_deactivate.py [--reason REASON]

Exit codes:
    0 - Deactivation successful
    1 - Deactivation failed
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.go_live.activation import go_live_deactivate


def print_banner():
    """Print deactivation banner."""
    print()
    print("=" * 80)
    print("QUANTUM TRADER v2.0 - GO-LIVE DEACTIVATION")
    print("=" * 80)
    print()


async def main():
    """Main deactivation logic."""
    import argparse

    parser = argparse.ArgumentParser(description="Deactivate GO-LIVE mode (disable REAL TRADING)")
    parser.add_argument(
        "--reason",
        type=str,
        default="manual_deactivation",
        help="Reason for deactivation (for audit trail)",
    )
    args = parser.parse_args()

    print_banner()

    print(f"🛑 Deactivating GO-LIVE (reason: {args.reason})...")
    print()

    try:
        success = await go_live_deactivate(reason=args.reason)

        if not success:
            print()
            print("=" * 80)
            print("❌ GO-LIVE DEACTIVATION FAILED")
            print("=" * 80)
            print()
            print("💡 Check logs for details.")
            print()
            return 1

        print()
        print("=" * 80)
        print("✅ GO-LIVE DEACTIVATION SUCCESSFUL")
        print("=" * 80)
        print()
        print("REAL TRADING is now DISABLED.")
        print()
        print("Next steps:")
        print("  1. Verify no real orders are being placed")
        print("  2. Check Risk & Resilience Dashboard")
        print("  3. Document deactivation in operational log")
        print()
        print("To REACTIVATE:")
        print("  - Run: python scripts/go_live_activate.py")
        print()
        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ GO-LIVE DEACTIVATION FAILED (EXCEPTION)")
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
