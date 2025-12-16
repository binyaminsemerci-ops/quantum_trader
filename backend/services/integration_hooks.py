"""
AI SYSTEM INTEGRATION HOOKS
============================

Integration points for AI subsystems in the trading loop.

This module provides:
- Pre-trade hooks (before signal execution)
- During-execution hooks (order creation & management)
- Post-trade hooks (position monitoring)
- Portfolio-level hooks (exposure management)

All hooks are NO-OP by default and activated via feature flags.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from backend.services.system_services import get_ai_services, SubsystemMode

logger = logging.getLogger(__name__)


# ============================================================================
# PRE-TRADE HOOKS (Before signal execution)
# ============================================================================

async def pre_trade_universe_filter(symbols: List[str]) -> List[str]:
    """
    Filter symbols through Universe OS before processing signals.
    
    Args:
        symbols: Original symbol list
    
    Returns:
        Filtered symbol list (respects blacklist, whitelist, etc.)
    
    Stage 1 (OBSERVE): Log what would be filtered, return original list
    Stage 2+ (ADVISORY/ENFORCED): Apply filtering
    """
    services = get_ai_services()
    
    if not services.config.universe_os_enabled:
        return symbols
    
    mode = services.config.universe_os_mode
    
    if mode == SubsystemMode.OFF:
        return symbols
    
    if mode == SubsystemMode.OBSERVE:
        # Stage 1: Only log, don't filter
        logger.info(f"[Universe OS] OBSERVE mode - would process {len(symbols)} symbols")
        return symbols
    
    # Stage 2+: Apply filtering (TODO: implement actual Universe OS integration)
    logger.info(f"[Universe OS] {mode.value} mode - processing {len(symbols)} symbols")
    
    # Placeholder: In production, integrate with selection_engine.py
    # filtered_symbols = await universe_os.filter_symbols(symbols)
    
    return symbols


async def pre_trade_risk_check(
    symbol: str,
    signal: Dict[str, Any],
    current_positions: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Check if trade is allowed through Risk OS and AI-HFOS.
    
    Args:
        symbol: Trading symbol
        signal: AI signal dict
        current_positions: Current open positions
    
    Returns:
        (allowed: bool, reason: str)
    
    Stage 1 (OBSERVE): Always allow, log decision
    Stage 2+ (ADVISORY/ENFORCED): Enforce risk checks
    """
    services = get_ai_services()
    
    # Check emergency brake first (always enforced)
    if services.config.emergency_brake_active:
        logger.warning(f"[Risk OS] Emergency brake ACTIVE - blocking {symbol}")
        return False, "Emergency brake active"
    
    # Check AI-HFOS directives
    if services.config.ai_hfos_enabled and services.ai_hfos_integration:
        try:
            output = services.ai_hfos_integration.last_output
            if output:
                # Check global directives
                if not output.global_directives.allow_new_trades:
                    reason = f"AI-HFOS blocked: {output.system_risk_mode.value} mode"
                    logger.warning(f"[AI-HFOS] Blocking {symbol} - {reason}")
                    
                    if services.config.ai_hfos_mode == SubsystemMode.ENFORCED:
                        return False, reason
                    else:
                        logger.info(f"[AI-HFOS] OBSERVE mode - would block {symbol}")
        except Exception as e:
            logger.error(f"[AI-HFOS] Error checking directives: {e}")
    
    # All checks passed
    return True, "All risk checks passed"


async def pre_trade_portfolio_check(
    symbol: str,
    signal: Dict[str, Any],
    current_positions: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Check if trade is allowed through Portfolio Balancer AI.
    
    Args:
        symbol: Trading symbol
        signal: AI signal dict
        current_positions: Current open positions
    
    Returns:
        (allowed: bool, reason: str)
    
    Stage 1 (OBSERVE): Always allow, log decision
    Stage 2+ (ADVISORY/ENFORCED): Apply portfolio limits
    """
    services = get_ai_services()
    
    if not services.config.pba_enabled:
        return True, "PBA not enabled"
    
    mode = services.config.pba_mode
    
    if mode == SubsystemMode.OFF:
        return True, "PBA disabled"
    
    # TODO: Integrate with actual Portfolio Balancer
    # For now, basic position count check
    max_positions = 15  # Updated portfolio limit
    
    if len(current_positions) >= max_positions:
        reason = f"Portfolio limit reached: {len(current_positions)} positions"
        logger.warning(f"[PBA] {reason}")
        
        if mode == SubsystemMode.ENFORCED:
            return False, reason
        else:
            logger.info(f"[PBA] OBSERVE mode - would block {symbol}")
    
    return True, "Portfolio check passed"


async def pre_trade_confidence_adjustment(
    signal: Dict[str, Any],
    base_threshold: float
) -> float:
    """
    Adjust confidence threshold based on AI-HFOS and Orchestrator.
    
    Args:
        signal: AI signal dict
        base_threshold: Base confidence threshold
    
    Returns:
        Adjusted confidence threshold
    
    Stage 1 (OBSERVE): Return base threshold, log adjustment
    Stage 2+ (ADVISORY/ENFORCED): Apply adjustments
    """
    services = get_ai_services()
    
    adjusted_threshold = base_threshold
    
    # Check AI-HFOS override
    if services.config.ai_hfos_enabled and services.ai_hfos_integration:
        try:
            output = services.ai_hfos_integration.last_output
            if output and output.global_directives.adjust_confidence_threshold:
                adjusted_threshold = output.global_directives.adjust_confidence_threshold
                
                logger.info(
                    f"[AI-HFOS] Confidence threshold adjusted: "
                    f"{base_threshold:.2f} → {adjusted_threshold:.2f}"
                )
        except Exception as e:
            logger.error(f"[AI-HFOS] Error adjusting confidence: {e}")
    
    return adjusted_threshold


async def pre_trade_position_sizing(
    symbol: str,
    signal: Dict[str, Any],
    base_size_usd: float
) -> float:
    """
    Adjust position size based on AI-HFOS and Portfolio Balancer.
    
    Args:
        symbol: Trading symbol
        signal: AI signal dict
        base_size_usd: Base position size in USD
    
    Returns:
        Adjusted position size in USD
    
    Stage 1 (OBSERVE): Return base size, log adjustment
    Stage 2+ (ADVISORY/ENFORCED): Apply scaling
    """
    services = get_ai_services()
    
    adjusted_size = base_size_usd
    
    # Check AI-HFOS scaling
    if services.config.ai_hfos_enabled and services.ai_hfos_integration:
        try:
            output = services.ai_hfos_integration.last_output
            if output:
                scale = output.global_directives.scale_position_sizes
                adjusted_size = base_size_usd * scale
                
                logger.info(
                    f"[AI-HFOS] Position size scaled for {symbol}: "
                    f"${base_size_usd:.2f} → ${adjusted_size:.2f} ({scale:.1%})"
                )
        except Exception as e:
            logger.error(f"[AI-HFOS] Error scaling position: {e}")
    
    return adjusted_size


# ============================================================================
# DURING-EXECUTION HOOKS (Order creation & management)
# ============================================================================

async def execution_order_type_selection(
    symbol: str,
    signal: Dict[str, Any],
    default_order_type: str
) -> str:
    """
    Select order type based on AELM and AI-HFOS directives.
    
    Args:
        symbol: Trading symbol
        signal: AI signal dict
        default_order_type: Default order type (MARKET, LIMIT, etc.)
    
    Returns:
        Order type to use
    
    Stage 1 (OBSERVE): Return default, log selection
    Stage 2+ (ADVISORY/ENFORCED): Apply AELM/AI-HFOS preferences
    """
    services = get_ai_services()
    
    order_type = default_order_type
    
    # Check AI-HFOS execution directives
    if services.config.ai_hfos_enabled and services.ai_hfos_integration:
        try:
            output = services.ai_hfos_integration.last_output
            if output:
                # Check if LIMIT orders are enforced
                if output.execution_directives.enforce_limit_orders:
                    order_type = "LIMIT"
                    logger.info(f"[AI-HFOS] Forcing LIMIT order for {symbol}")
                elif output.execution_directives.order_type_preference:
                    order_type = output.execution_directives.order_type_preference
        except Exception as e:
            logger.error(f"[AI-HFOS] Error selecting order type: {e}")
    
    return order_type


async def execution_slippage_check(
    symbol: str,
    expected_price: float,
    actual_price: float
) -> Tuple[bool, str]:
    """
    Check if slippage is acceptable based on AELM and AI-HFOS.
    
    Args:
        symbol: Trading symbol
        expected_price: Expected execution price
        actual_price: Actual execution price
    
    Returns:
        (acceptable: bool, reason: str)
    
    Stage 1 (OBSERVE): Always allow, log slippage
    Stage 2+ (ADVISORY/ENFORCED): Enforce slippage caps
    """
    services = get_ai_services()
    
    slippage_bps = abs((actual_price - expected_price) / expected_price) * 10000
    
    # Check AI-HFOS slippage cap
    max_slippage_bps = 15.0  # Default
    
    if services.config.ai_hfos_enabled and services.ai_hfos_integration:
        try:
            output = services.ai_hfos_integration.last_output
            if output:
                max_slippage_bps = output.execution_directives.max_slippage_bps
        except Exception as e:
            logger.error(f"[AI-HFOS] Error checking slippage: {e}")
    
    if slippage_bps > max_slippage_bps:
        reason = f"Excessive slippage: {slippage_bps:.1f} bps > {max_slippage_bps:.1f} bps cap"
        logger.warning(f"[AELM] {symbol} - {reason}")
        
        if services.config.aelm_mode == SubsystemMode.ENFORCED:
            return False, reason
        else:
            logger.info(f"[AELM] OBSERVE mode - would reject {symbol}")
    
    return True, "Slippage acceptable"


# ============================================================================
# POST-TRADE HOOKS (Position monitoring)
# ============================================================================

async def post_trade_position_classification(
    position: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classify position through Position Intelligence Layer.
    
    Args:
        position: Position data dict
    
    Returns:
        Position with classification metadata added
    
    Stage 1 (OBSERVE): Log classification only
    Stage 2+ (ADVISORY/ENFORCED): Add classification to position
    """
    services = get_ai_services()
    
    if not services.config.pil_enabled:
        return position
    
    # TODO: Integrate with actual PIL
    # For now, placeholder
    logger.info(f"[PIL] Would classify position: {position.get('symbol', 'UNKNOWN')}")
    
    return position


async def post_trade_amplification_check(
    position: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check for amplification opportunities through PAL.
    
    Args:
        position: Position data dict
    
    Returns:
        Amplification recommendation dict or None
    
    Stage 1 (OBSERVE): Log opportunities only
    Stage 2+ (ADVISORY/ENFORCED): Return recommendations
    """
    services = get_ai_services()
    
    if not services.config.pal_enabled or not services.pal:
        return None
    
    mode = services.config.pal_mode
    
    if mode == SubsystemMode.OFF:
        return None
    
    # TODO: Integrate with actual PAL analysis
    # For now, placeholder
    logger.info(f"[PAL] Checking amplification for: {position.get('symbol', 'UNKNOWN')}")
    
    return None


# ============================================================================
# PORTFOLIO-LEVEL HOOKS
# ============================================================================

async def portfolio_exposure_check() -> Dict[str, Any]:
    """
    Get portfolio exposure analysis from PBA.
    
    Returns:
        Exposure analysis dict with limits and recommendations
    
    Stage 1 (OBSERVE): Log exposure only
    Stage 2+ (ADVISORY/ENFORCED): Return actionable limits
    """
    services = get_ai_services()
    
    if not services.config.pba_enabled:
        return {"status": "disabled"}
    
    # TODO: Integrate with actual PBA
    # For now, placeholder
    logger.info("[PBA] Checking portfolio exposure...")
    
    return {
        "status": "ok",
        "total_exposure_pct": 0.0,
        "max_exposure_pct": 20.0,
        "recommendations": []
    }


async def portfolio_rebalance_recommendations() -> List[Dict[str, Any]]:
    """
    Get rebalancing recommendations from PBA.
    
    Returns:
        List of rebalancing action dicts
    
    Stage 1 (OBSERVE): Log recommendations only
    Stage 2+ (ADVISORY/ENFORCED): Return executable recommendations
    """
    services = get_ai_services()
    
    if not services.config.pba_enabled:
        return []
    
    mode = services.config.pba_mode
    
    if mode == SubsystemMode.OFF:
        return []
    
    # TODO: Integrate with actual PBA
    # For now, placeholder
    logger.info("[PBA] Generating rebalance recommendations...")
    
    return []


# ============================================================================
# PERIODIC/META-LEVEL HOOKS
# ============================================================================

async def periodic_self_healing_check():
    """
    Run periodic self-healing system check.
    
    Stage 1 (OBSERVE): Log issues only
    Stage 2+ (PROTECTIVE): Apply recovery actions
    """
    services = get_ai_services()
    
    if not services.config.self_healing_enabled or not services.self_healing:
        return
    
    try:
        # Run health check
        report = services.self_healing.check_system_health()
        
        if report.overall_status != "HEALTHY":
            logger.warning(
                f"[Self-Healing] System health: {report.overall_status} - "
                f"{len(report.detected_issues)} issues detected"
            )
            
            # In OBSERVE mode, just log
            if services.config.self_healing_mode == SubsystemMode.OBSERVE:
                for issue in report.detected_issues[:3]:  # Log first 3
                    logger.info(f"[Self-Healing] Issue: {issue.issue_type.value}")
    
    except Exception as e:
        logger.error(f"[Self-Healing] Check failed: {e}")


async def periodic_ai_hfos_coordination():
    """
    Run periodic AI-HFOS coordination cycle.
    
    This is the supreme meta-level coordination that:
    - Collects data from all subsystems
    - Resolves conflicts
    - Issues unified directives
    
    Stage 1 (OBSERVE): Coordination runs but directives are advisory only
    Stage 3+ (COORDINATION): Directives are enforced
    """
    services = get_ai_services()
    
    if not services.config.ai_hfos_enabled or not services.ai_hfos_integration:
        return
    
    try:
        # Run coordination cycle
        await services.ai_hfos_integration.run_coordination_cycle()
        
        # Log summary
        status = services.ai_hfos_integration.get_system_status()
        logger.info(
            f"[AI-HFOS] Coordination complete - "
            f"Risk Mode: {status.get('risk_mode', 'UNKNOWN')}, "
            f"Health: {status.get('health', 'UNKNOWN')}"
        )
    
    except Exception as e:
        logger.error(f"[AI-HFOS] Coordination failed: {e}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def should_enforce_mode(mode: SubsystemMode) -> bool:
    """Check if a subsystem mode should be enforced."""
    return mode in (SubsystemMode.ADVISORY, SubsystemMode.ENFORCED)


def get_integration_summary() -> Dict[str, Any]:
    """Get summary of current integration state."""
    services = get_ai_services()
    
    return {
        "stage": services.config.integration_stage.value,
        "enabled_subsystems": [
            name for name, status in services._services_status.items()
            if status in ("initialized", "available", "using_existing")
        ],
        "emergency_brake": services.config.emergency_brake_active,
        "ai_hfos_active": services.config.ai_hfos_enabled
    }
