#!/usr/bin/env python3
"""
CI Assertion: Verify system is FLAT (no open positions)

Confirms that after panic_close execution:
1. mock:open_positions = 0
2. system:state:trading.halted = true
"""

import os
import sys
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POSITIONS_KEY = "mock:open_positions"  # CI mock key
TRADING_HALT_KEY = "system:state:trading"


def main():
    print("=" * 50)
    print("ASSERTION: System is FLAT")
    print("=" * 50)
    
    r = redis.from_url(REDIS_URL)
    
    errors = []
    
    # Check 1: Open positions should be 0
    positions = r.get(POSITIONS_KEY)
    if positions is None:
        print("⚠️ Position key not found (may be OK if EEW deleted it)")
        open_positions = 0
    else:
        open_positions = int(positions)
    
    print(f"Open positions: {open_positions}")
    
    if open_positions != 0:
        errors.append(f"Open positions = {open_positions}, expected 0")
    
    # Check 2: Trading should be halted
    halt_state = r.hgetall(TRADING_HALT_KEY)
    
    if not halt_state:
        errors.append(f"Trading halt state not set in {TRADING_HALT_KEY}")
    else:
        halted = halt_state.get(b"halted", b"").decode()
        reason = halt_state.get(b"reason", b"").decode()
        positions_closed = halt_state.get(b"positions_closed", b"0").decode()
        requires_reset = halt_state.get(b"requires_manual_reset", b"").decode()
        
        print(f"Trading halt state:")
        print(f"  halted: {halted}")
        print(f"  reason: {reason}")
        print(f"  positions_closed: {positions_closed}")
        print(f"  requires_manual_reset: {requires_reset}")
        
        if halted != "true":
            errors.append(f"halted = {halted}, expected 'true'")
        
        if requires_reset != "true":
            errors.append(f"requires_manual_reset = {requires_reset}, expected 'true'")
    
    if errors:
        print("=" * 50)
        print("❌ FAIL: System is NOT flat")
        for err in errors:
            print(f"   - {err}")
        print("")
        print("CRITICAL: Positions may still be open!")
        print("Manual intervention required.")
        sys.exit(1)
    
    print("=" * 50)
    print("✅ PASS: System is FLAT")
    print("   - Open positions: 0")
    print("   - Trading: HALTED")
    print("   - Manual reset required: YES")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
