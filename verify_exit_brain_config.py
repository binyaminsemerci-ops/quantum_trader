"""
Quick verification that Exit Brain V3 Shadow Mode is active
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config.exit_mode import (
    get_exit_mode,
    get_exit_executor_mode,
    is_exit_brain_live_rollout_enabled,
    is_exit_brain_live_fully_enabled,
)

def main():
    print("\n" + "="*60)
    print("EXIT BRAIN V3 CONFIGURATION CHECK")
    print("="*60)
    
    # Get current configuration
    exit_mode = get_exit_mode()
    executor_mode = get_exit_executor_mode()
    rollout_enabled = is_exit_brain_live_rollout_enabled()
    fully_enabled = is_exit_brain_live_fully_enabled()
    
    print(f"\n📊 Current Configuration:")
    print(f"   EXIT_MODE: {exit_mode}")
    print(f"   EXIT_EXECUTOR_MODE: {executor_mode}")
    print(f"   EXIT_BRAIN_V3_LIVE_ROLLOUT: {'ENABLED' if rollout_enabled else 'DISABLED'}")
    print(f"   Fully LIVE: {'YES' if fully_enabled else 'NO'}")
    
    # Determine operational state
    if exit_mode == "LEGACY":
        state = "LEGACY"
        emoji = "🔵"
        description = "Traditional exit system active"
    elif exit_mode == "EXIT_BRAIN_V3":
        if fully_enabled:
            state = "EXIT_BRAIN_V3_LIVE"
            emoji = "🔴"
            description = "AI controls exits, legacy modules blocked"
        else:
            state = "EXIT_BRAIN_V3_SHADOW"
            emoji = "🟡"
            description = "AI observes, legacy modules active"
    else:
        state = "UNKNOWN"
        emoji = "⚪"
        description = "Unknown configuration"
    
    print(f"\n{emoji} Operational State: {state}")
    print(f"   {description}")
    
    # Expected for Shadow Mode deployment
    print(f"\n✅ Expected Configuration (Shadow Mode):")
    print(f"   EXIT_MODE: EXIT_BRAIN_V3")
    print(f"   EXIT_EXECUTOR_MODE: SHADOW")
    print(f"   EXIT_BRAIN_V3_LIVE_ROLLOUT: DISABLED")
    
    # Verification
    print(f"\n🔍 Verification:")
    checks = []
    
    if exit_mode == "EXIT_BRAIN_V3":
        checks.append(("✅", "EXIT_MODE correctly set to EXIT_BRAIN_V3"))
    else:
        checks.append(("❌", f"EXIT_MODE is {exit_mode}, expected EXIT_BRAIN_V3"))
    
    if executor_mode == "SHADOW":
        checks.append(("✅", "EXIT_EXECUTOR_MODE correctly set to SHADOW"))
    else:
        checks.append(("⚠️ ", f"EXIT_EXECUTOR_MODE is {executor_mode}, expected SHADOW"))
    
    if not rollout_enabled:
        checks.append(("✅", "EXIT_BRAIN_V3_LIVE_ROLLOUT correctly DISABLED"))
    else:
        checks.append(("⚠️ ", "EXIT_BRAIN_V3_LIVE_ROLLOUT is ENABLED (not safe for shadow mode)"))
    
    if not fully_enabled:
        checks.append(("✅", "System NOT in LIVE mode (safe for shadow)"))
    else:
        checks.append(("🔴", "System IS in LIVE mode (AI placing orders!)"))
    
    for emoji, msg in checks:
        print(f"   {emoji} {msg}")
    
    # Next steps
    print(f"\n📋 Next Steps:")
    if state == "EXIT_BRAIN_V3_SHADOW":
        print(f"   1. Backend must be restarted to apply configuration")
        print(f"   2. Monitor shadow logs: backend/data/exit_brain_shadow.jsonl")
        print(f"   3. Watch for [EXIT_BRAIN_SHADOW] messages in logs")
        print(f"   4. Run for 24-48 hours before proceeding to LIVE")
    elif state == "EXIT_BRAIN_V3_LIVE":
        print(f"   🔴 LIVE MODE ACTIVE - AI is placing orders!")
        print(f"   Monitor closely for next 1-2 hours")
    elif state == "LEGACY":
        print(f"   Configuration not applied. Check .env file and restart backend.")
    
    print("\n" + "="*60)
    
    return 0 if state == "EXIT_BRAIN_V3_SHADOW" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
