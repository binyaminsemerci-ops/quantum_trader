"""
Exit Mode Configuration
========================

Central configuration for exit strategy ownership and execution.

EXIT_MODE determines which system owns exit decisions and execution:
- LEGACY: Traditional position_monitor + hybrid_tpsl behavior
- EXIT_BRAIN_V3: Exit Brain orchestrator owns exit decisions

This is Phase 1 - observability and soft guards only.
Later phases will enforce stricter ownership boundaries.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Valid exit modes
EXIT_MODE_LEGACY = "LEGACY"
EXIT_MODE_EXIT_BRAIN_V3 = "EXIT_BRAIN_V3"

VALID_EXIT_MODES = [EXIT_MODE_LEGACY, EXIT_MODE_EXIT_BRAIN_V3]

# Valid executor modes (for Exit Brain v3)
EXIT_EXECUTOR_MODE_SHADOW = "SHADOW"
EXIT_EXECUTOR_MODE_LIVE = "LIVE"

VALID_EXIT_EXECUTOR_MODES = [EXIT_EXECUTOR_MODE_SHADOW, EXIT_EXECUTOR_MODE_LIVE]

# Rollout safety flag values
ROLLOUT_ENABLED = "ENABLED"
ROLLOUT_DISABLED = "DISABLED"


def get_exit_mode() -> str:
    """
    Get current exit mode from environment.
    
    Returns:
        "LEGACY" or "EXIT_BRAIN_V3"
        
    Defaults to LEGACY if not set or invalid.
    """
    mode = os.getenv("EXIT_MODE", EXIT_MODE_LEGACY).upper()
    
    if mode not in VALID_EXIT_MODES:
        logger.warning(
            f"[EXIT_MODE] Invalid EXIT_MODE='{mode}' in environment. "
            f"Valid values: {VALID_EXIT_MODES}. Defaulting to {EXIT_MODE_LEGACY}."
        )
        return EXIT_MODE_LEGACY
    
    return mode


def is_exit_brain_mode() -> bool:
    """
    Check if running in Exit Brain V3 mode.
    
    Returns:
        True if EXIT_MODE == "EXIT_BRAIN_V3"
    """
    return get_exit_mode() == EXIT_MODE_EXIT_BRAIN_V3


def is_legacy_exit_mode() -> bool:
    """
    Check if running in Legacy exit mode.
    
    Returns:
        True if EXIT_MODE == "LEGACY"
    """
    return get_exit_mode() == EXIT_MODE_LEGACY


# ============================================================================
# EXIT BRAIN PROFILE CONFIGURATION
# ============================================================================

def get_exit_brain_profile() -> str:
    """
    Get current Exit Brain risk management profile from environment.
    
    Only relevant when EXIT_MODE == "EXIT_BRAIN_V3".
    
    Returns:
        "DEFAULT" or "CHALLENGE_100"
        
    Defaults to "DEFAULT" if not set or invalid.
    """
    profile = os.getenv("EXIT_BRAIN_PROFILE", "DEFAULT").upper()
    
    valid_profiles = ["DEFAULT", "CHALLENGE_100"]
    if profile not in valid_profiles:
        logger.warning(
            f"[EXIT_MODE] Invalid EXIT_BRAIN_PROFILE='{profile}' in environment. "
            f"Valid values: {valid_profiles}. Defaulting to DEFAULT."
        )
        return "DEFAULT"
    
    return profile


def is_challenge_100_profile() -> bool:
    """
    Check if Exit Brain is using CHALLENGE_100 risk management profile.
    
    Only meaningful when EXIT_MODE == "EXIT_BRAIN_V3".
    
    Returns:
        True if EXIT_BRAIN_PROFILE == "CHALLENGE_100"
    """
    return get_exit_brain_profile() == "CHALLENGE_100"


def get_exit_executor_mode() -> str:
    """
    Get current Exit Brain executor mode from environment.
    
    Only relevant when EXIT_MODE == "EXIT_BRAIN_V3".
    
    Returns:
        "SHADOW" or "LIVE"
        
    Defaults to SHADOW if not set or invalid.
    """
    mode = os.getenv("EXIT_EXECUTOR_MODE", EXIT_EXECUTOR_MODE_SHADOW).upper()
    
    if mode not in VALID_EXIT_EXECUTOR_MODES:
        logger.warning(
            f"[EXIT_MODE] Invalid EXIT_EXECUTOR_MODE='{mode}' in environment. "
            f"Valid values: {VALID_EXIT_EXECUTOR_MODES}. Defaulting to {EXIT_EXECUTOR_MODE_SHADOW}."
        )
        return EXIT_EXECUTOR_MODE_SHADOW
    
    return mode


def is_exit_executor_shadow_mode() -> bool:
    """
    Check if Exit Brain executor should run in SHADOW mode.
    
    Returns:
        True if EXIT_EXECUTOR_MODE == "SHADOW"
    """
    return get_exit_executor_mode() == EXIT_EXECUTOR_MODE_SHADOW


def is_exit_executor_live_mode() -> bool:
    """
    Check if Exit Brain executor is configured for LIVE mode.
    
    NOTE: This checks config only. For actual LIVE behavior,
    use is_exit_brain_live_fully_enabled() which also checks rollout flag.
    
    Returns:
        True if EXIT_EXECUTOR_MODE == "LIVE"
    """
    return get_exit_executor_mode() == EXIT_EXECUTOR_MODE_LIVE


def is_exit_brain_live_rollout_enabled() -> bool:
    """
    Check if Exit Brain v3 LIVE rollout safety flag is enabled.
    
    This is an additional kill-switch for LIVE mode.
    
    Returns:
        True if EXIT_BRAIN_V3_LIVE_ROLLOUT == "ENABLED"
    """
    flag = os.getenv("EXIT_BRAIN_V3_LIVE_ROLLOUT", ROLLOUT_DISABLED).upper()
    return flag == ROLLOUT_ENABLED


def is_exit_executor_kill_switch_active() -> bool:
    """
    Check if Exit Brain executor kill-switch is active.
    
    This is a fail-closed safety mechanism that forces shadow mode
    regardless of other settings. When active, NO orders will be placed.
    
    Returns:
        True if EXIT_EXECUTOR_KILL_SWITCH == "true" (case-insensitive)
    """
    kill_switch = os.getenv("EXIT_EXECUTOR_KILL_SWITCH", "false").lower()
    return kill_switch in ("true", "1", "yes", "on", "enabled")


def get_exit_rollout_pct() -> int:
    """
    Get Exit Brain LIVE rollout percentage (0-100).
    
    This controls what percentage of symbols get LIVE mode.
    The rest operate in SHADOW mode (deterministic per symbol).
    
    Returns:
        Integer 0-100 (default: 0 = all SHADOW)
    """
    try:
        pct = int(os.getenv("EXIT_LIVE_ROLLOUT_PCT", "0"))
        return max(0, min(100, pct))  # Clamp to 0-100
    except (ValueError, TypeError):
        logger.warning("[EXIT_MODE] Invalid EXIT_LIVE_ROLLOUT_PCT, defaulting to 0")
        return 0


def is_symbol_in_live_rollout(symbol: str) -> bool:
    """
    Check if a symbol is included in the LIVE rollout percentage.
    
    Uses deterministic hash-based selection:
    - hash(symbol) % 100 < EXIT_LIVE_ROLLOUT_PCT → LIVE
    - Otherwise → SHADOW
    
    This ensures consistent behavior per symbol across restarts.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        
    Returns:
        True if symbol is in LIVE rollout, False if SHADOW
    """
    rollout_pct = get_exit_rollout_pct()
    
    # 0% = all SHADOW, 100% = all LIVE
    if rollout_pct == 0:
        return False
    if rollout_pct == 100:
        return True
    
    # Deterministic hash-based rollout
    symbol_hash = hash(symbol) % 100
    return symbol_hash < rollout_pct


def is_exit_brain_live_fully_enabled(symbol: str = None) -> bool:
    """
    Check if Exit Brain v3 LIVE mode is FULLY enabled.
    
    Enforcement hierarchy (KILL_SWITCH > MODE > ROLLOUT > DEFAULT):
    1. EXIT_EXECUTOR_KILL_SWITCH == "true" → SHADOW (fail-closed)
    2. EXIT_MODE != "EXIT_BRAIN_V3" → SHADOW
    3. EXIT_EXECUTOR_MODE != "LIVE" → SHADOW
    4. EXIT_BRAIN_V3_LIVE_ROLLOUT != "ENABLED" → SHADOW
    5. If symbol provided: check EXIT_LIVE_ROLLOUT_PCT (hash-based)
    
    Args:
        symbol: Optional trading symbol for rollout percentage check
        
    Returns:
        True if all conditions are met and symbol is in LIVE rollout
    """
    # PRIORITY 1: Kill-switch overrides everything (fail-closed)
    if is_exit_executor_kill_switch_active():
        return False
    
    # PRIORITY 2: Base mode checks
    if not (
        is_exit_brain_mode() and
        is_exit_executor_live_mode() and
        is_exit_brain_live_rollout_enabled()
    ):
        return False
    
    # PRIORITY 3: Rollout percentage (if symbol provided)
    if symbol is not None:
        return is_symbol_in_live_rollout(symbol)
    
    # No symbol = global check passed
    return True


# Log mode on module import for visibility
_current_mode = get_exit_mode()
_executor_mode = get_exit_executor_mode()
_live_rollout = "ENABLED" if is_exit_brain_live_rollout_enabled() else "DISABLED"
_brain_profile = get_exit_brain_profile()
_kill_switch = "ACTIVE" if is_exit_executor_kill_switch_active() else "OFF"
_rollout_pct = get_exit_rollout_pct()

logger.info(
    f"[EXIT_MODE] Configuration loaded: "
    f"EXIT_MODE={_current_mode}, "
    f"EXIT_EXECUTOR_MODE={_executor_mode}, "
    f"EXIT_BRAIN_V3_LIVE_ROLLOUT={_live_rollout}, "
    f"ROLLOUT_PCT={_rollout_pct}%, "
    f"EXIT_BRAIN_PROFILE={_brain_profile}, "
    f"KILL_SWITCH={_kill_switch}"
)

# Log control state to Redis for audit trail
try:
    import redis
    import json
    from datetime import datetime, timezone
    
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_mode": _current_mode,
        "executor_mode": _executor_mode,
        "kill_switch": _kill_switch == "ACTIVE",
        "live_rollout": _live_rollout,
        "rollout_pct": _rollout_pct,
        "brain_profile": _brain_profile
    }
    
    # Store as list (latest first, keep last 100)
    key = "quantum:ops:exitbrain:control"
    r.lpush(key, json.dumps(state))
    r.ltrim(key, 0, 99)  # Keep last 100 entries
    
    logger.info(f"[EXIT_MODE] Control state logged to Redis: {key}")
except Exception as e:
    logger.debug(f"[EXIT_MODE] Redis audit logging not available: {e}")

# Check if LIVE mode is fully enabled
if is_exit_executor_kill_switch_active():
    logger.warning(
        "[EXIT_MODE] 🔴 KILL-SWITCH ACTIVE 🔴 "
        "Exit Brain forced to SHADOW mode. No orders will be placed. "
        "Set EXIT_EXECUTOR_KILL_SWITCH=false to re-enable."
    )
elif is_exit_brain_live_fully_enabled():
    logger.warning(
        "[EXIT_MODE] 🔴 EXIT BRAIN V3 LIVE MODE ACTIVE 🔴 "
        "AI executor will place real orders. Legacy exit modules will be blocked."
    )
elif is_exit_brain_mode() and is_exit_executor_live_mode() and not is_exit_brain_live_rollout_enabled():
    logger.info(
        "[EXIT_MODE] ⚠️  EXIT_EXECUTOR_MODE=LIVE but EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED. "
        "Executor will run in SHADOW mode (safety fallback)."
    )
elif is_exit_brain_mode():
    logger.info(
        f"[EXIT_MODE] Exit Brain v3 active in {_executor_mode} mode"
    )

# Also check consistency with EXIT_BRAIN_V3_ENABLED flag
_exit_brain_enabled = os.getenv("EXIT_BRAIN_V3_ENABLED", "false").lower() == "true"
if _current_mode == EXIT_MODE_EXIT_BRAIN_V3 and not _exit_brain_enabled:
    logger.warning(
        "[EXIT_MODE] EXIT_MODE=EXIT_BRAIN_V3 but EXIT_BRAIN_V3_ENABLED=false. "
        "Consider setting EXIT_BRAIN_V3_ENABLED=true for consistency."
    )
elif _current_mode == EXIT_MODE_LEGACY and _exit_brain_enabled:
    logger.warning(
        "[EXIT_MODE] EXIT_MODE=LEGACY but EXIT_BRAIN_V3_ENABLED=true. "
        "This may cause mixed behavior. Consider aligning flags."
    )
