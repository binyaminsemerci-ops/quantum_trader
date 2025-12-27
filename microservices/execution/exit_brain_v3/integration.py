"""
Exit Brain v3 Integration Helpers - Bridge to existing execution layers.

Translates ExitPlan into parameters for:
- dynamic_tpsl.py
- trailing_stop_manager.py / dynamic_trailing_rearm.py
- position_monitor.py
"""

import logging
from typing import Dict, Tuple, List, Optional

from .models import (
    ExitContext,
    ExitPlan,
    ExitLeg,
    ExitKind
)

logger = logging.getLogger(__name__)


def to_dynamic_tpsl(exit_plan: ExitPlan, ctx: ExitContext) -> Dict:
    """
    Convert ExitPlan to parameters for dynamic_tpsl.py.
    
    Returns dict compatible with DynamicTPSLCalculator output:
    {
        "tp_percent": float,
        "sl_percent": float,
        "trail_percent": float,
        "partial_tp": bool,
        "rationale": str,
        "confidence": float
    }
    """
    # Get primary TP and SL legs
    primary_tp = exit_plan.get_primary_tp()
    primary_sl = exit_plan.get_primary_sl()
    
    # Extract percentages
    tp_percent = primary_tp.trigger_pct if primary_tp else 0.03
    sl_percent = abs(primary_sl.trigger_pct) if primary_sl else 0.025
    
    # Check for trailing
    trail_legs = exit_plan.get_legs_by_kind(ExitKind.TRAIL)
    trail_percent = trail_legs[0].trail_callback if trail_legs else 0.015
    
    # Check if partial exits configured
    partial_tp = len(exit_plan.get_legs_by_kind(ExitKind.TP)) > 1
    
    result = {
        "tp_percent": tp_percent,
        "sl_percent": sl_percent,
        "trail_percent": trail_percent,
        "partial_tp": partial_tp,
        "rationale": exit_plan.reason,
        "confidence": exit_plan.confidence
    }
    
    logger.debug(
        f"[EXIT BRAIN → TPSL] {exit_plan.symbol}: "
        f"TP={tp_percent:.2%}, SL={sl_percent:.2%}, Trail={trail_percent:.2%}"
    )
    
    return result


def to_trailing_config(exit_plan: ExitPlan, ctx: ExitContext) -> Optional[Dict]:
    """
    Convert ExitPlan to trailing stop configuration.
    
    Returns dict for trailing_stop_manager or dynamic_trailing_rearm:
    {
        "enabled": bool,
        "callback_pct": float,
        "activation_pct": float,  # When to start trailing
        "size_pct": float         # Portion of position to trail
    }
    
    Returns None if no trailing configured.
    """
    trail_legs = exit_plan.get_legs_by_kind(ExitKind.TRAIL)
    
    if not trail_legs:
        return None
    
    # Use first trailing leg (typically TP3)
    trail_leg = trail_legs[0]
    
    # Determine activation threshold (start trailing when in profit)
    # Usually starts at 50% of primary TP
    primary_tp = exit_plan.get_primary_tp()
    activation_pct = (primary_tp.trigger_pct * 0.5) if primary_tp else 0.015
    
    config = {
        "enabled": True,
        "callback_pct": trail_leg.trail_callback,
        "activation_pct": activation_pct,
        "size_pct": trail_leg.size_pct
    }
    
    logger.debug(
        f"[EXIT BRAIN → TRAIL] {exit_plan.symbol}: "
        f"Callback={config['callback_pct']:.2%}, "
        f"Activates @ {config['activation_pct']:.2%}"
    )
    
    return config


def to_partial_exit_config(exit_plan: ExitPlan, ctx: ExitContext) -> List[Dict]:
    """
    Convert ExitPlan to partial exit ladder configuration.
    
    Returns list of partial exit levels for position_monitor:
    [
        {
            "trigger_pct": float,      # PnL % to trigger
            "size_pct": float,         # Portion to exit (0.0-1.0)
            "kind": str,               # "TP" or "TRAIL"
            "reason": str
        },
        ...
    ]
    
    Sorted by trigger_pct (ascending).
    """
    partial_exits = []
    
    # Extract all TP legs
    for leg in exit_plan.legs:
        if leg.kind in [ExitKind.TP, ExitKind.TRAIL]:
            partial_exits.append({
                "trigger_pct": leg.trigger_pct or 999.0,  # Trail = very high
                "size_pct": leg.size_pct,
                "kind": leg.kind.value,
                "reason": leg.reason,
                "trail_callback": leg.trail_callback if leg.kind == ExitKind.TRAIL else None
            })
    
    # Sort by trigger level
    partial_exits.sort(key=lambda x: x["trigger_pct"])
    
    logger.debug(
        f"[EXIT BRAIN → PARTIAL] {exit_plan.symbol}: "
        f"{len(partial_exits)} exit levels configured"
    )
    
    return partial_exits


def build_context_from_position(
    position: Dict,
    rl_hints: Optional[Dict] = None,
    risk_context: Optional[Dict] = None,
    market_data: Optional[Dict] = None
) -> ExitContext:
    """
    Build ExitContext from raw position data + optional hints.
    
    Args:
        position: Binance position dict (from futures_position_information)
        rl_hints: Optional RL v3 output {"tp_pct": X, "sl_pct": Y, "confidence": Z}
        risk_context: Optional Risk v3 state {"mode": "NORMAL", "max_loss": 0.025}
        market_data: Optional market metrics {"volatility": 0.02, "trend": 0.5}
        
    Returns:
        ExitContext ready for exit_brain.build_exit_plan()
    """
    symbol = position["symbol"]
    amt = float(position["positionAmt"])
    entry_price = float(position["entryPrice"])
    mark_price = float(position["markPrice"])
    leverage = float(position.get("leverage", 1))
    
    # Calculate PnL
    is_long = amt > 0
    if is_long:
        unrealized_pnl = (mark_price - entry_price) * amt
    else:
        unrealized_pnl = (entry_price - mark_price) * abs(amt)
    
    margin = abs(amt * entry_price) / leverage
    pnl_pct = (unrealized_pnl / margin * 100) if margin > 0 else 0.0
    
    # Extract hints
    rl_tp = rl_hints.get("tp_pct") if rl_hints else None
    rl_sl = rl_hints.get("sl_pct") if rl_hints else None
    rl_conf = rl_hints.get("confidence") if rl_hints else None
    
    risk_mode = risk_context.get("mode", "NORMAL") if risk_context else "NORMAL"
    max_loss = risk_context.get("max_loss", 0.025) if risk_context else 0.025
    
    volatility = market_data.get("volatility", 0.02) if market_data else 0.02
    trend = market_data.get("trend", 0.0) if market_data else 0.0
    
    # Infer market regime from volatility and trend
    if volatility > 0.04:
        regime = "VOLATILE"
    elif abs(trend) > 0.6:
        regime = "TRENDING"
    elif volatility < 0.015:
        regime = "RANGE_BOUND"
    else:
        regime = "NORMAL"
    
    ctx = ExitContext(
        symbol=symbol,
        side="LONG" if is_long else "SHORT",
        entry_price=entry_price,
        size=abs(amt),
        leverage=leverage,
        current_price=mark_price,
        unrealized_pnl_pct=pnl_pct,
        unrealized_pnl_usd=unrealized_pnl,
        volatility=volatility,
        trend_strength=trend,
        market_regime=regime,
        rl_tp_hint=rl_tp,
        rl_sl_hint=rl_sl,
        rl_confidence=rl_conf,
        risk_mode=risk_mode,
        max_loss_pct=max_loss
    )
    
    logger.debug(
        f"[EXIT BRAIN] Context built for {symbol}: "
        f"PnL={pnl_pct:.2f}%, Regime={regime}, Risk={risk_mode}"
    )
    
    return ctx
