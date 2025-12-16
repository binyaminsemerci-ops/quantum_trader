"""
GO-LIVE Activation Logic

This module implements the core activation logic for enabling REAL TRADING.

CRITICAL: This is a SAFETY LAYER. It prevents accidental real trading by:
  1. Checking activation_enabled flag in YAML config
  2. Running preflight checks (if required)
  3. Verifying global risk state
  4. Verifying ESS is inactive
  5. Creating activation marker file (go_live.active)

The Execution Service checks for this marker file before placing real orders.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml

from backend.preflight.checks import run_all_preflight_checks
from backend.preflight.types import PreflightResult

logger = logging.getLogger(__name__)

# Activation marker file path (project root)
GO_LIVE_MARKER_FILE = Path(__file__).parent.parent.parent / "go_live.active"
GO_LIVE_CONFIG_FILE = Path(__file__).parent.parent.parent / "config" / "go_live.yaml"


async def go_live_activate(operator: Optional[str] = None) -> bool:
    """
    Activate GO-LIVE mode for REAL TRADING.

    This function:
      1. Reads go_live.yaml config
      2. Checks activation_enabled flag
      3. Runs preflight checks (if required)
      4. Verifies risk state requirements
      5. Creates go_live.active marker file
      6. Updates activation metadata in YAML

    Args:
        operator: Name of operator performing activation (for audit trail)

    Returns:
        True if activation successful, False otherwise
    """
    logger.info("🚀 Starting GO-LIVE activation sequence...")

    # Load config
    try:
        with open(GO_LIVE_CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {GO_LIVE_CONFIG_FILE}")
        return False
    except yaml.YAMLError as e:
        logger.error(f"❌ Invalid YAML config: {e}")
        return False

    # Check 1: activation_enabled flag
    if not cfg.get("activation_enabled", False):
        logger.warning("❌ activation_enabled=false in config - cannot activate")
        logger.info("💡 To enable: edit config/go_live.yaml → set activation_enabled: true")
        return False

    logger.info("✅ Activation enabled in config")

    # Check 2: Run preflight checks (if required)
    if cfg.get("required_preflight", True):
        logger.info("🔍 Running preflight checks...")
        results = await run_all_preflight_checks()
        failed = [r for r in results if not r.success]

        if failed:
            logger.error(f"❌ Preflight checks FAILED ({len(failed)}/{len(results)}):")
            for fail in failed:
                logger.error(f"   - {fail.name}: {fail.reason}")
            return False

        logger.info(f"✅ Preflight checks PASSED ({len(results)}/{len(results)})")

    # Check 3: Verify risk state (stub - would integrate with Global Risk v3)
    required_risk_state = cfg.get("require_risk_state", "OK")
    try:
        current_risk_state = await _get_current_risk_state()
        if current_risk_state != required_risk_state:
            logger.error(
                f"❌ Risk state mismatch: current={current_risk_state}, "
                f"required={required_risk_state}"
            )
            return False
        logger.info(f"✅ Risk state is {current_risk_state}")
    except Exception as e:
        logger.error(f"❌ Failed to get risk state: {e}")
        return False

    # Check 4: Verify ESS is inactive (stub - would integrate with ESS)
    if cfg.get("require_ess_inactive", True):
        try:
            ess_active = await _check_ess_active()
            if ess_active:
                logger.error("❌ Emergency Stop System is ACTIVE - cannot activate GO-LIVE")
                return False
            logger.info("✅ ESS is inactive")
        except Exception as e:
            logger.error(f"❌ Failed to check ESS status: {e}")
            return False

    # Check 5: Verify testnet history (if required)
    if cfg.get("require_testnet_history", True):
        min_trades = cfg.get("min_testnet_trades", 3)
        try:
            testnet_count = await _get_testnet_trade_count()
            if testnet_count < min_trades:
                logger.error(
                    f"❌ Insufficient testnet history: {testnet_count} trades "
                    f"(minimum {min_trades} required)"
                )
                return False
            logger.info(f"✅ Testnet history verified ({testnet_count} trades)")
        except Exception as e:
            logger.error(f"❌ Failed to verify testnet history: {e}")
            return False

    # All checks passed - create activation marker file
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        operator_name = operator or "unknown"

        marker_content = f"""# GO-LIVE ACTIVATION MARKER
# This file indicates that REAL TRADING is ENABLED.
# DO NOT delete or modify this file unless you intend to DISABLE real trading.

activated: true
timestamp: {timestamp}
operator: {operator_name}
environment: {cfg.get('environment', 'unknown')}
allowed_profiles: {', '.join(cfg.get('allowed_profiles', []))}

# This file is checked by the Execution Service before placing real orders.
# If this file does not exist, all real trading is BLOCKED.
"""

        with open(GO_LIVE_MARKER_FILE, "w") as f:
            f.write(marker_content)

        logger.info(f"✅ Created activation marker: {GO_LIVE_MARKER_FILE}")

    except Exception as e:
        logger.error(f"❌ Failed to create activation marker: {e}")
        return False

    # Update activation metadata in config
    try:
        cfg["last_activation_timestamp"] = timestamp
        cfg["last_activation_operator"] = operator_name
        cfg["activation_count"] = cfg.get("activation_count", 0) + 1

        with open(GO_LIVE_CONFIG_FILE, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info("✅ Updated activation metadata in config")

    except Exception as e:
        logger.warning(f"⚠️ Failed to update config metadata: {e}")
        # Non-critical - activation still succeeded

    logger.info("🎉 GO-LIVE ACTIVATION SUCCESSFUL")
    logger.info(f"   Marker file: {GO_LIVE_MARKER_FILE}")
    logger.info(f"   Operator: {operator_name}")
    logger.info(f"   Timestamp: {timestamp}")
    logger.info(f"   Allowed profiles: {', '.join(cfg.get('allowed_profiles', []))}")

    return True


async def go_live_deactivate(reason: Optional[str] = None) -> bool:
    """
    Deactivate GO-LIVE mode (disable REAL TRADING).

    This function:
      1. Removes go_live.active marker file
      2. Updates rollback metadata in YAML

    Args:
        reason: Reason for deactivation (for audit trail)

    Returns:
        True if deactivation successful, False otherwise
    """
    logger.info("🛑 Starting GO-LIVE deactivation...")

    # Remove marker file
    try:
        if GO_LIVE_MARKER_FILE.exists():
            GO_LIVE_MARKER_FILE.unlink()
            logger.info(f"✅ Removed activation marker: {GO_LIVE_MARKER_FILE}")
        else:
            logger.warning("⚠️ Activation marker does not exist (already deactivated)")

    except Exception as e:
        logger.error(f"❌ Failed to remove activation marker: {e}")
        return False

    # Update rollback metadata
    try:
        with open(GO_LIVE_CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["last_rollback_timestamp"] = datetime.now(timezone.utc).isoformat()
        cfg["last_rollback_reason"] = reason or "manual_deactivation"

        with open(GO_LIVE_CONFIG_FILE, "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

        logger.info("✅ Updated rollback metadata in config")

    except Exception as e:
        logger.warning(f"⚠️ Failed to update config metadata: {e}")
        # Non-critical - deactivation still succeeded

    logger.info("✅ GO-LIVE DEACTIVATION SUCCESSFUL")
    return True


def is_go_live_active() -> bool:
    """
    Check if GO-LIVE is currently active.

    Returns:
        True if go_live.active marker file exists, False otherwise
    """
    return GO_LIVE_MARKER_FILE.exists()


async def _get_current_risk_state() -> str:
    """
    Get current global risk state.

    TODO: Integrate with Global Risk v3 service
    For now, returns stub value.

    Returns:
        Risk state string ("OK", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    """
    # Stub implementation - would call Global Risk v3 API
    # from backend.domains.risk_v3.service import get_current_risk_state
    # risk = await get_current_risk_state()
    # return risk.global_status

    logger.debug("Using stub risk state (OK) - TODO: integrate Global Risk v3")
    return "OK"


async def _check_ess_active() -> bool:
    """
    Check if Emergency Stop System is active.

    TODO: Integrate with ESS service
    For now, returns stub value.

    Returns:
        True if ESS is active (trading blocked), False otherwise
    """
    # Stub implementation - would call ESS API
    # from backend.services.risk.emergency_stop_system import is_ess_active
    # return await is_ess_active()

    logger.debug("Using stub ESS check (inactive) - TODO: integrate ESS")
    return False


async def _get_testnet_trade_count() -> int:
    """
    Get count of successful testnet trades.

    TODO: Integrate with analytics/database
    For now, returns stub value.

    Returns:
        Number of successful testnet trades
    """
    # Stub implementation - would query database
    # from backend.analytics.testnet import get_successful_trade_count
    # return await get_successful_trade_count()

    logger.debug("Using stub testnet count (10) - TODO: integrate analytics")
    return 10  # Stub: assume sufficient testnet history
