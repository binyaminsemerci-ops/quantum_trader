#!/usr/bin/env python3
"""
P1 Risk Kernel: Stops/Trailing Proposal Engine (CALC-ONLY)

Consumes MarketState outputs + position snapshots → proposes SL/TP/trailing updates.
NO orders, NO execution, NO API calls. Pure calculation with audit trail.

LOCKED SPEC v1.0:
- Regime-weighted stop distances using k_sl/k_tp/k_trail multipliers
- Monotonic SL tightening (never loosen)
- Peak/trough-based trailing
- Optional TP extension on strong trends
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import time


@dataclass
class PositionSnapshot:
    """Immutable position state for risk calculation"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    peak_price: float  # highest price since entry (LONG) or lowest (SHORT)
    trough_price: float  # lowest price since entry (LONG) or highest (SHORT)
    age_sec: float
    unrealized_pnl: float
    current_sl: Optional[float] = None  # existing SL, if any
    current_tp: Optional[float] = None  # existing TP, if any


# DEFAULT_THETA for risk kernel
DEFAULT_THETA_RISK = {
    "sl_min_pct": 0.005,  # 0.5% minimum SL distance
    "tp_min_pct": 0.01,   # 1.0% minimum TP distance
    
    # k_sl: sigma multipliers per regime (trend/mr/chop)
    "k_sl": {
        "trend": 1.5,  # wider stops in trending markets
        "mr": 0.8,     # tighter stops in mean-reverting
        "chop": 1.0,   # neutral in choppy
    },
    
    # k_tp: sigma multipliers for TP distance
    "k_tp": {
        "trend": 2.5,  # wider targets in trending
        "mr": 1.2,     # tighter targets in MR
        "chop": 1.5,   # neutral
    },
    
    # k_trail: sigma multipliers for trailing gap
    "k_trail": {
        "trend": 1.2,  # tighter trailing in trends
        "mr": 1.5,     # looser in MR
        "chop": 1.3,   # neutral
    },
    
    # Optional TP extension on strong trends
    "tp_extend_gain": 0.5,  # scale factor for TS-based extension
    "tp_extend_max": 0.3,   # max 30% extension
    
    "monotonic_sl": True,   # SL may only tighten, never loosen
    "cooldown_sec": 60,     # min seconds between proposals (not enforced here)
    "eps": 1e-8,
}


def compute_proposal(
    symbol: str,
    market_state: Dict[str, Any],
    position: PositionSnapshot,
    theta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute proposed SL/TP updates for a position given current market state.
    
    Args:
        symbol: Trading symbol (must match position.symbol)
        market_state: Dict with keys: sigma, mu, ts, regime_probs {trend,mr,chop}, features (optional)
        position: PositionSnapshot with current position state
        theta: Risk config override (uses DEFAULT_THETA_RISK if None)
    
    Returns:
        Dict with:
            - proposed_sl: float
            - proposed_tp: float
            - reason_codes: list[str] (audit trail)
            - audit: dict with all intermediate calculations
            - meta: dict with timestamp, symbol, side
    """
    if theta is None:
        theta = DEFAULT_THETA_RISK
    
    if symbol != position.symbol:
        raise ValueError(f"Symbol mismatch: {symbol} != {position.symbol}")
    
    if position.side not in ("LONG", "SHORT"):
        raise ValueError(f"Invalid side: {position.side}")
    
    # Extract market state
    sigma = market_state.get("sigma", 0.0)
    mu = market_state.get("mu", 0.0)
    ts = market_state.get("ts", 0.0)
    regime_probs = market_state.get("regime_probs", {})
    
    pi_trend = regime_probs.get("trend", 0.0)
    pi_mr = regime_probs.get("mr", 0.0)
    pi_chop = regime_probs.get("chop", 0.0)
    
    # Extract theta params
    sl_min_pct = theta.get("sl_min_pct", 0.005)
    tp_min_pct = theta.get("tp_min_pct", 0.01)
    k_sl = theta.get("k_sl", {})
    k_tp = theta.get("k_tp", {})
    k_trail = theta.get("k_trail", {})
    tp_extend_gain = theta.get("tp_extend_gain", 0.5)
    tp_extend_max = theta.get("tp_extend_max", 0.3)
    monotonic_sl = theta.get("monotonic_sl", True)
    eps = theta.get("eps", 1e-8)
    
    # Compute weighted stop distance
    k_sl_weighted = (
        k_sl.get("trend", 1.0) * pi_trend +
        k_sl.get("mr", 1.0) * pi_mr +
        k_sl.get("chop", 1.0) * pi_chop
    )
    stop_dist_pct = max(sl_min_pct, k_sl_weighted * sigma)
    
    # Compute weighted TP distance with optional trend extension
    k_tp_weighted = (
        k_tp.get("trend", 1.0) * pi_trend +
        k_tp.get("mr", 1.0) * pi_mr +
        k_tp.get("chop", 1.0) * pi_chop
    )
    
    # Optional TP extension on strong trends
    tp_extension_factor = 1.0
    if pi_trend > 0.3 and ts > 0.1:  # only extend if trend is meaningful
        ts_clamped = max(0.0, min(1.0, ts))
        extension = tp_extend_gain * ts_clamped * pi_trend
        extension = min(extension, tp_extend_max)  # cap at max
        tp_extension_factor = 1.0 + extension
    
    tp_dist_pct = max(tp_min_pct, k_tp_weighted * sigma * tp_extension_factor)
    
    # Compute weighted trail gap
    k_trail_weighted = (
        k_trail.get("trend", 1.0) * pi_trend +
        k_trail.get("mr", 1.0) * pi_mr +
        k_trail.get("chop", 1.0) * pi_chop
    )
    trail_gap_pct = max(sl_min_pct, k_trail_weighted * sigma)
    
    # Compute proposed SL/TP based on side
    reason_codes = []
    
    if position.side == "LONG":
        # LONG: SL below current, TP above
        raw_sl = position.current_price * (1.0 - stop_dist_pct)
        raw_tp = position.current_price * (1.0 + tp_dist_pct)
        
        # Trailing SL based on peak
        trail_sl = position.peak_price * (1.0 - trail_gap_pct)
        
        # Use tightest SL (max)
        proposed_sl = max(raw_sl, trail_sl)
        if trail_sl > raw_sl + eps:
            reason_codes.append("trail_active")
        
        proposed_tp = raw_tp
        
        # Monotonic SL tightening
        if monotonic_sl and position.current_sl is not None:
            if proposed_sl < position.current_sl - eps:
                reason_codes.append("sl_tightened_monotonic")
                proposed_sl = position.current_sl
            elif proposed_sl > position.current_sl + eps:
                reason_codes.append("sl_tightening")
        
    else:  # SHORT
        # SHORT: SL above current, TP below
        raw_sl = position.current_price * (1.0 + stop_dist_pct)
        raw_tp = position.current_price * (1.0 - tp_dist_pct)
        
        # Trailing SL based on trough
        trail_sl = position.trough_price * (1.0 + trail_gap_pct)
        
        # Use tightest SL (min for SHORT)
        proposed_sl = min(raw_sl, trail_sl)
        if trail_sl < raw_sl - eps:
            reason_codes.append("trail_active")
        
        proposed_tp = raw_tp
        
        # Monotonic SL tightening
        if monotonic_sl and position.current_sl is not None:
            if proposed_sl > position.current_sl + eps:
                reason_codes.append("sl_tightened_monotonic")
                proposed_sl = position.current_sl
            elif proposed_sl < position.current_sl - eps:
                reason_codes.append("sl_tightening")
    
    # Add regime context to reason codes
    dominant_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
    reason_codes.append(f"regime_{dominant_regime}")
    
    if tp_extension_factor > 1.0 + eps:
        reason_codes.append("tp_extended")
    
    # Build audit dict
    audit = {
        "inputs": {
            "sigma": sigma,
            "mu": mu,
            "ts": ts,
            "regime_probs": regime_probs,
            "current_price": position.current_price,
            "peak_price": position.peak_price,
            "trough_price": position.trough_price,
            "existing_sl": position.current_sl,
            "existing_tp": position.current_tp,
        },
        "intermediates": {
            "k_sl_weighted": k_sl_weighted,
            "k_tp_weighted": k_tp_weighted,
            "k_trail_weighted": k_trail_weighted,
            "stop_dist_pct": stop_dist_pct,
            "tp_dist_pct": tp_dist_pct,
            "trail_gap_pct": trail_gap_pct,
            "tp_extension_factor": tp_extension_factor,
            "raw_sl": raw_sl,
            "raw_tp": raw_tp,
            "trail_sl": trail_sl,
        },
        "theta": {
            "sl_min_pct": sl_min_pct,
            "tp_min_pct": tp_min_pct,
            "k_sl": k_sl,
            "k_tp": k_tp,
            "k_trail": k_trail,
            "monotonic_sl": monotonic_sl,
        }
    }
    
    return {
        "proposed_sl": proposed_sl,
        "proposed_tp": proposed_tp,
        "reason_codes": reason_codes,
        "audit": audit,
        "meta": {
            "timestamp": time.time(),
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "current_price": position.current_price,
            "age_sec": position.age_sec,
        }
    }


def format_proposal(proposal: Dict[str, Any]) -> str:
    """Format proposal dict for human-readable logging"""
    meta = proposal["meta"]
    lines = [
        f"Proposal for {meta['symbol']} ({meta['side']}):",
        f"  Entry: ${meta['entry_price']:.4f}  Current: ${meta['current_price']:.4f}  Age: {meta['age_sec']:.0f}s",
        f"  Proposed SL: ${proposal['proposed_sl']:.4f}",
        f"  Proposed TP: ${proposal['proposed_tp']:.4f}",
        f"  Reasons: {', '.join(proposal['reason_codes'])}",
        f"  Regime: {proposal['audit']['inputs']['regime_probs']}",
        f"  Sigma: {proposal['audit']['inputs']['sigma']:.6f}  TS: {proposal['audit']['inputs']['ts']:.4f}",
    ]
    return "\n".join(lines)
