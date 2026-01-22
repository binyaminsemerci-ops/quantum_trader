"""
P2 Harvest Proposal Engine (calc-only)

Pure calculation module for profit harvesting proposals:
- Partial exit recommendations (25%, 50%, 75%)
- Profit lock SL tightening
- Kill score (edge collapse detection)
- Full close proposal

NO trading side-effects. NO orders. NO execution.
"""

import math
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class HarvestTheta:
    """Harvest engine tunables (theta.harvest.*)"""
    # Risk unit fallback
    fallback_stop_pct: float = 0.02  # 2% if P1 stop_dist_pct not available
    
    # Cost estimation
    cost_bps: float = 10.0  # 10 bps per round-trip
    
    # Tranche triggers (in R_net units)
    T1_R: float = 2.0  # Trigger for PARTIAL_25
    T2_R: float = 4.0  # Trigger for PARTIAL_50
    T3_R: float = 6.0  # Trigger for PARTIAL_75
    
    # Tranche weights (softmax inputs)
    u1: float = 0.0  # Weight for tranche 1
    u2: float = 0.0  # Weight for tranche 2
    u3: float = 0.0  # Weight for tranche 3
    softmax_temp: float = 1.0  # Temperature for softmax
    
    # Profit lock
    lock_R: float = 1.5  # Move SL to BE+ at this R_net
    be_plus_pct: float = 0.002  # 0.2% above breakeven
    
    # Kill score components
    trend_min: float = 0.3  # Minimum p_trend to avoid regime flip penalty
    sigma_ref: float = 0.01  # Reference sigma for spike detection
    sigma_spike_cap: float = 2.0  # Cap for sigma spike ratio
    ts_ref: float = 0.3  # Reference ts for drop detection
    ts_drop_cap: float = 0.5  # Cap for ts drop
    max_age_sec: float = 86400.0  # 24h max age for penalty
    
    # Kill score weights
    k_regime_flip: float = 1.0
    k_sigma_spike: float = 0.5
    k_ts_drop: float = 0.5
    k_age_penalty: float = 0.3
    
    kill_threshold: float = 0.6  # Trigger FULL_CLOSE_PROPOSED at this K


@dataclass
class PositionSnapshot:
    """Position state for harvest computation"""
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    current_price: float
    peak_price: float  # Highest price reached (LONG) or lowest (SHORT)
    trough_price: float  # Lowest price reached (LONG) or highest (SHORT)
    age_sec: float
    unrealized_pnl: float
    current_sl: Optional[float] = None
    current_tp: Optional[float] = None


@dataclass
class MarketState:
    """Market state for regime detection"""
    sigma: float
    ts: float
    p_trend: float
    p_mr: float
    p_chop: float


@dataclass
class P1Proposal:
    """Optional P1 proposal for risk_unit calculation"""
    stop_dist_pct: float
    proposed_sl: Optional[float] = None
    proposed_tp: Optional[float] = None


def compute_tranche_weights(theta: HarvestTheta, regime_probs: MarketState) -> List[float]:
    """
    Compute softmax tranche weights.
    
    Returns list of 3 weights summing to 1.0
    """
    u = [theta.u1, theta.u2, theta.u3]
    
    # Apply softmax
    exp_vals = [math.exp(val / theta.softmax_temp) for val in u]
    total = sum(exp_vals)
    
    if total < 1e-9:
        return [1/3, 1/3, 1/3]  # Uniform fallback
    
    return [e / total for e in exp_vals]


def compute_risk_unit(
    position: PositionSnapshot,
    p1_proposal: Optional[P1Proposal],
    theta: HarvestTheta
) -> float:
    """
    Compute risk_unit for R_net normalization.
    
    Prefer P1 stop_dist_pct if available, else use fallback.
    """
    if p1_proposal and p1_proposal.stop_dist_pct:
        risk_unit = position.entry_price * p1_proposal.stop_dist_pct
    else:
        risk_unit = position.entry_price * theta.fallback_stop_pct
    
    return max(risk_unit, 1e-9)  # Avoid division by zero


def compute_R_net(
    position: PositionSnapshot,
    risk_unit: float,
    theta: HarvestTheta
) -> tuple[float, float]:
    """
    Compute normalized R_net after cost estimation.
    
    Returns (R_net, cost_est)
    """
    cost_est = (theta.cost_bps / 10000.0) * position.entry_price
    R_net = (position.unrealized_pnl - cost_est) / risk_unit
    
    return R_net, cost_est


def determine_harvest_action(R_net: float, theta: HarvestTheta) -> str:
    """
    Determine harvest action based on R_net triggers.
    
    Returns: NONE | PARTIAL_25 | PARTIAL_50 | PARTIAL_75
    """
    if R_net >= theta.T3_R:
        return "PARTIAL_75"
    elif R_net >= theta.T2_R:
        return "PARTIAL_50"
    elif R_net >= theta.T1_R:
        return "PARTIAL_25"
    else:
        return "NONE"


def compute_profit_lock_sl(
    position: PositionSnapshot,
    R_net: float,
    theta: HarvestTheta
) -> Optional[float]:
    """
    Propose new SL for profit locking.
    
    Returns new_sl if profit lock triggered, else None.
    Must be monotonic tightening.
    """
    if R_net < theta.lock_R:
        return None
    
    # Compute BE+ level
    if position.side == "LONG":
        be_plus = position.entry_price * (1 + theta.be_plus_pct)
        if position.current_sl is not None:
            new_sl = max(position.current_sl, be_plus)
        else:
            new_sl = be_plus
        # Only propose if it tightens
        if position.current_sl is None or new_sl > position.current_sl:
            return new_sl
    else:  # SHORT
        be_plus = position.entry_price * (1 - theta.be_plus_pct)
        if position.current_sl is not None:
            new_sl = min(position.current_sl, be_plus)
        else:
            new_sl = be_plus
        # Only propose if it tightens
        if position.current_sl is None or new_sl < position.current_sl:
            return new_sl
    
    return None


def compute_kill_score(
    position: PositionSnapshot,
    market_state: MarketState,
    theta: HarvestTheta
) -> tuple[float, Dict[str, float]]:
    """
    Compute kill score K (edge collapse indicator).
    
    Returns (K, components_dict)
    K ∈ [0,1] via sigmoid
    """
    components = {}
    
    # 1. Regime flip: trend→chop/mr
    if market_state.p_trend < theta.trend_min:
        regime_flip = 1.0 if (market_state.p_chop + market_state.p_mr) > 0.5 else 0.0
    else:
        regime_flip = 0.0
    components["regime_flip"] = regime_flip
    
    # 2. Sigma spike
    sigma_ratio = market_state.sigma / theta.sigma_ref
    sigma_spike = max(0.0, min(sigma_ratio - 1.0, theta.sigma_spike_cap))
    components["sigma_spike"] = sigma_spike
    
    # 3. TS drop
    ts_drop = max(0.0, min(theta.ts_ref - market_state.ts, theta.ts_drop_cap))
    components["ts_drop"] = ts_drop
    
    # 4. Age penalty
    age_penalty = max(0.0, min(position.age_sec / theta.max_age_sec, 1.0))
    components["age_penalty"] = age_penalty
    
    # Weighted sum
    z = (
        theta.k_regime_flip * regime_flip +
        theta.k_sigma_spike * sigma_spike +
        theta.k_ts_drop * ts_drop +
        theta.k_age_penalty * age_penalty
    )
    
    # Sigmoid to [0,1]
    K = 1.0 / (1.0 + math.exp(-z))
    
    return K, components


def compute_harvest_proposal(
    position: PositionSnapshot,
    market_state: MarketState,
    p1_proposal: Optional[P1Proposal] = None,
    theta: Optional[HarvestTheta] = None
) -> Dict[str, Any]:
    """
    Main entry point: compute harvest proposal (calc-only).
    
    Returns dict with:
    - harvest_action: NONE | PARTIAL_25 | PARTIAL_50 | PARTIAL_75 | FULL_CLOSE_PROPOSED
    - new_sl_proposed: float or None
    - R_net, risk_unit, cost_est: float
    - kill_score: float
    - reason_codes: list of strings
    - audit: dict of inputs + intermediates
    """
    if theta is None:
        theta = HarvestTheta()
    
    reason_codes = []
    
    # 1. Compute risk_unit
    risk_unit = compute_risk_unit(position, p1_proposal, theta)
    
    # 2. Compute R_net
    R_net, cost_est = compute_R_net(position, risk_unit, theta)
    
    # 3. Determine harvest action
    harvest_action = determine_harvest_action(R_net, theta)
    if harvest_action != "NONE":
        reason_codes.append(f"harvest_{harvest_action.lower()}")
    
    # 4. Profit lock SL
    new_sl_proposed = compute_profit_lock_sl(position, R_net, theta)
    if new_sl_proposed is not None:
        reason_codes.append("profit_lock")
    
    # 5. Kill score
    kill_score, k_components = compute_kill_score(position, market_state, theta)
    if kill_score >= theta.kill_threshold:
        harvest_action = "FULL_CLOSE_PROPOSED"
        reason_codes.append("kill_score_triggered")
        if k_components["regime_flip"] > 0:
            reason_codes.append("regime_flip")
        if k_components["sigma_spike"] > 0.5:
            reason_codes.append("sigma_spike")
        if k_components["ts_drop"] > 0.2:
            reason_codes.append("ts_drop")
        if k_components["age_penalty"] > 0.5:
            reason_codes.append("age_penalty")
    
    # 6. Tranche weights (for audit)
    tranche_weights = compute_tranche_weights(theta, market_state)
    
    # Build output
    output = {
        "harvest_action": harvest_action,
        "new_sl_proposed": new_sl_proposed,
        "R_net": R_net,
        "risk_unit": risk_unit,
        "cost_est": cost_est,
        "kill_score": kill_score,
        "reason_codes": reason_codes,
        "audit": {
            # Inputs
            "position": asdict(position),
            "market_state": asdict(market_state),
            "p1_proposal": asdict(p1_proposal) if p1_proposal else None,
            "theta": asdict(theta),
            # Intermediates
            "tranche_weights": tranche_weights,
            "k_components": k_components,
        }
    }
    
    return output


# Example usage and validation
if __name__ == "__main__":
    # Test scenario: LONG position with profit
    pos = PositionSnapshot(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        current_price=105.0,
        peak_price=106.0,
        trough_price=99.0,
        age_sec=3600.0,
        unrealized_pnl=5.0,
        current_sl=99.0,
        current_tp=110.0
    )
    
    market = MarketState(
        sigma=0.015,
        ts=0.35,
        p_trend=0.5,
        p_mr=0.2,
        p_chop=0.3
    )
    
    p1 = P1Proposal(stop_dist_pct=0.02)
    
    result = compute_harvest_proposal(pos, market, p1)
    
    print("=== P2 Harvest Proposal (calc-only) ===")
    print(f"Harvest Action: {result['harvest_action']}")
    print(f"New SL Proposed: {result['new_sl_proposed']}")
    print(f"R_net: {result['R_net']:.2f}R")
    print(f"Kill Score: {result['kill_score']:.3f}")
    print(f"Reason Codes: {', '.join(result['reason_codes'])}")
    print(f"\nAudit Trail:")
    print(f"  Risk Unit: ${result['risk_unit']:.4f}")
    print(f"  Cost Est: ${result['cost_est']:.4f}")
    print(f"  Tranche Weights: {result['audit']['tranche_weights']}")
    print(f"  K Components: {result['audit']['k_components']}")
