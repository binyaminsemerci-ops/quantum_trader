"""
DYNAMIC TP/SL SYSTEM
====================

AI-driven take-profit and stop-loss calculation based on:
- Signal confidence
- Market volatility
- Position age
- Trend strength

Overrides legacy exit policy engine when enabled.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# [NEW] TP v3: Import Risk v3 Integration
try:
    from backend.services.risk_management.risk_v3_integration import RiskV3Integrator
    RISK_V3_AVAILABLE = True
    logger_risk_v3 = logging.getLogger(__name__ + ".risk_v3")
    logger_risk_v3.info("[OK] Risk v3 Integration available")
except ImportError as e:
    RISK_V3_AVAILABLE = False
    logger_risk_v3 = logging.getLogger(__name__ + ".risk_v3")
    logger_risk_v3.warning(f"[WARNING] Risk v3 Integration not available: {e}")


# [NEW] EXIT BRAIN V3: Unified exit orchestrator
EXIT_BRAIN_V3_ENABLED = False
try:
    import os
    EXIT_BRAIN_V3_ENABLED = os.getenv("EXIT_BRAIN_V3_ENABLED", "false").lower() == "true"
    
    if EXIT_BRAIN_V3_ENABLED:
        from backend.domains.exits.exit_brain_v3 import ExitBrainV3
        from backend.domains.exits.exit_brain_v3.models import ExitContext
        from backend.domains.exits.exit_brain_v3.integration import to_dynamic_tpsl
        logger.info("[EXIT BRAIN] Exit Brain v3 integration active in dynamic_tpsl")
except ImportError as e:
    EXIT_BRAIN_V3_ENABLED = False
    logger.warning(f"[EXIT BRAIN] Exit Brain v3 not available: {e}")


@dataclass
class DynamicTPSLOutput:
    """Dynamic TP/SL calculation result"""
    tp_percent: float
    sl_percent: float
    trail_percent: float
    partial_tp: bool
    rationale: str
    confidence: float
    timestamp: datetime


class DynamicTPSLCalculator:
    """
    Calculates optimal TP/SL based on signal strength and market conditions.
    
    Integration:
    - Called by event_driven_executor during trade execution
    - Overrides Exit Policy Engine when QT_USE_AI_DYNAMIC_TPSL=true
    - Works in harmony with AI-HFOS risk modes
    """
    
    def __init__(self):
        """Initialize Dynamic TP/SL Calculator"""
        self.logger = logging.getLogger(__name__)
        
        # Base parameters (will be scaled by confidence and volatility)
        # OPTIMIZED for 3-5x leverage with $300 trades - need REAL profits!
        self.base_tp = 0.06  # 6.0% base TP (30% profit with 5x leverage)
        self.base_sl = 0.025  # 2.5% base SL (12.5% loss with 5x leverage)
        self.base_trail = 0.02  # 2.0% trailing
        
        # Minimum absolute distances (prevents micro exits)
        self.min_tp_pct = 0.05  # 5.0% MINIMUM TP (25% profit with 5x!)
        self.min_sl_pct = 0.015  # 1.5% minimum SL (7.5% loss with 5x)
        
        # Confidence scaling factors
        self.min_confidence = 0.40  # Accept lower confidence (was 0.65)
        self.max_confidence = 0.95
        
        # Risk-reward ratios
        self.min_rr_ratio = 1.5
        self.max_rr_ratio = 4.0  # Allow higher R:R (was 3.0)
        
        # [NEW] TP v3: Initialize Risk v3 Integrator
        self.risk_v3_integrator = None
        if RISK_V3_AVAILABLE:
            self.risk_v3_integrator = RiskV3Integrator()
            logger_risk_v3.info("[OK] Risk v3 Integrator initialized")
        
        # [NEW] EXIT BRAIN V3: Initialize orchestrator
        self.exit_brain = None
        if EXIT_BRAIN_V3_ENABLED:
            self.exit_brain = ExitBrainV3()
            self.logger.info("[EXIT BRAIN] Exit Brain v3 orchestrator initialized")
        
        self.logger.info("[Dynamic TP/SL] Calculator initialized with AI-driven volatility scaling")
    
    def calculate(
        self,
        signal_confidence: float,
        action: str,
        market_conditions: Optional[Dict[str, Any]] = None,
        risk_mode: str = "NORMAL",
        risk_v3_context: Optional[Dict[str, Any]] = None,
        rl_tp_suggestion: Optional[float] = None,
        symbol: Optional[str] = None,
        entry_price: Optional[float] = None,
        size: Optional[float] = None,
        leverage: Optional[float] = None
    ) -> DynamicTPSLOutput:
        """
        Calculate dynamic TP/SL based on signal confidence and context.
        
        [EXIT BRAIN V3] If enabled, delegates to Exit Brain orchestrator.
        Otherwise uses legacy confidence/volatility scaling.
        
        Args:
            signal_confidence: AI signal confidence (0.0 to 1.0)
            action: Trade action (BUY/SELL/LONG/SHORT)
            market_conditions: Optional market data (volatility, etc.)
            risk_mode: Current AI-HFOS risk mode
            risk_v3_context: Risk v3 adjustments (ESS, systemic risk)
            rl_tp_suggestion: RL-suggested TP percentage
            symbol: Trading symbol (for Exit Brain)
            entry_price: Entry price (for Exit Brain)
            size: Position size (for Exit Brain)
            leverage: Leverage (for Exit Brain)
        
        Returns:
            DynamicTPSLOutput with TP/SL percentages
        """
        # [EXIT BRAIN V3] If enabled, build context and delegate to orchestrator
        if EXIT_BRAIN_V3_ENABLED and self.exit_brain and symbol and entry_price and size:
            try:
                # Build ExitContext
                ctx = ExitContext(
                    symbol=symbol,
                    side="LONG" if action in ["BUY", "LONG"] else "SHORT",
                    entry_price=entry_price,
                    size=size,
                    leverage=leverage or 1.0,
                    current_price=entry_price,  # At entry
                    unrealized_pnl_pct=0.0,  # New position
                    volatility=market_conditions.get("volatility", 0.02) if market_conditions else 0.02,
                    trend_strength=market_conditions.get("trend", 0.0) if market_conditions else 0.0,
                    market_regime=market_conditions.get("regime", "NORMAL") if market_conditions else "NORMAL",
                    rl_tp_hint=rl_tp_suggestion,
                    rl_confidence=signal_confidence,
                    signal_confidence=signal_confidence,
                    risk_mode=risk_mode
                )
                
                # Get exit plan from Exit Brain
                import asyncio
                loop = asyncio.get_event_loop()
                plan = loop.run_until_complete(self.exit_brain.build_exit_plan(ctx))
                
                # Convert to DynamicTPSLOutput format
                result_dict = to_dynamic_tpsl(plan, ctx)
                
                self.logger.info(
                    f"[EXIT BRAIN] {symbol}: TP={result_dict['tp_percent']:.2%}, "
                    f"SL={result_dict['sl_percent']:.2%}, Trail={result_dict['trail_percent']:.2%}"
                )
                
                return DynamicTPSLOutput(
                    tp_percent=result_dict["tp_percent"],
                    sl_percent=result_dict["sl_percent"],
                    trail_percent=result_dict["trail_percent"],
                    partial_tp=result_dict["partial_tp"],
                    rationale=result_dict["rationale"],
                    confidence=result_dict["confidence"],
                    timestamp=datetime.now(timezone.utc)
                )
            except Exception as e:
                self.logger.warning(f"[EXIT BRAIN] Failed to build plan: {e}, falling back to legacy")
                # Fall through to legacy calculation
        
        # [LEGACY PATH] Original confidence/volatility scaling
        # Normalize confidence to scaling factor (0.5 to 1.5)
        confidence_normalized = (signal_confidence - self.min_confidence) / (
            self.max_confidence - self.min_confidence
        )
        confidence_normalized = max(0.0, min(1.0, confidence_normalized))
        confidence_scale = 0.5 + confidence_normalized  # 0.5 to 1.5
        
        # Volatility scaling (from market conditions)
        volatility_scale = 1.0
        if market_conditions and 'volatility' in market_conditions:
            # Scale TP/SL based on volatility: high vol = wider targets
            volatility = market_conditions['volatility']
            # volatility typically 0.005-0.05 (0.5%-5%)
            volatility_scale = 0.8 + (volatility / 0.03) * 0.4  # 0.8 to 1.2
            volatility_scale = max(0.8, min(1.5, volatility_scale))
        
        # Combined scaling
        combined_scale = confidence_scale * volatility_scale
        
        # Apply risk mode multipliers
        risk_multipliers = self._get_risk_mode_multipliers(risk_mode)
        
        # Calculate TP (scales UP with confidence and volatility)
        tp_percent = self.base_tp * combined_scale * risk_multipliers["tp"]
        
        # [TP v3] Apply RL suggestion if available
        if rl_tp_suggestion and rl_tp_suggestion > 0:
            # Blend RL suggestion with confidence-based TP
            tp_percent = (tp_percent * 0.6) + (rl_tp_suggestion * 0.4)
            self.logger.debug(f"[RL v3] Blended TP with RL suggestion: {rl_tp_suggestion:.2%}")
        
        # Calculate SL (scales DOWN with confidence - tighter SL for high confidence)
        sl_scale = 1.5 - (confidence_scale - 0.5)  # 1.5 to 0.5 (inverse)
        sl_percent = self.base_sl * sl_scale * volatility_scale * risk_multipliers["sl"]
        
        # [NEW] TP v3: Query Risk v3 real-time metrics if available
        tp_adjustment_factor = 1.0
        if self.risk_v3_integrator:
            try:
                risk_context = self.risk_v3_integrator.get_risk_context()
                
                # Apply TP adjustment based on risk metrics
                tp_adjustment_factor = self.risk_v3_integrator.get_tp_adjustment_factor(risk_context)
                
                if tp_adjustment_factor != 1.0:
                    logger_risk_v3.info(
                        f"[TP v3] Risk adjustment: ESS={risk_context.ess_factor:.2f}, "
                        f"Systemic={risk_context.systemic_risk_level:.2f}, "
                        f"TP factor={tp_adjustment_factor:.2f}"
                    )
                
                # Override legacy risk_v3_context param with live data
                risk_v3_context = {
                    'ess_factor': risk_context.ess_factor,
                    'systemic_risk_level': risk_context.systemic_risk_level,
                    'correlation_risk': risk_context.correlation_risk,
                    'portfolio_heat': risk_context.portfolio_heat
                }
            except Exception as e:
                logger_risk_v3.warning(f"[TP v3] Risk context query failed: {e}")
        
        # [TP v3] Apply Risk v3 adjustments (legacy fallback or live data)
        if risk_v3_context:
            ess_factor = risk_v3_context.get('ess_factor', 1.0)
            systemic_risk = risk_v3_context.get('systemic_risk_level', 0.0)
            
            # High ESS → tighten TP to reduce exposure
            if ess_factor > 1.5:
                tp_adjustment = 0.85 * tp_adjustment_factor
                tp_percent *= tp_adjustment
                self.logger.info(f"[Risk v3] TP tightened due to ESS={ess_factor:.2f} (adjustment={tp_adjustment})")
            
            # Systemic risk detected → defensive TP
            if systemic_risk > 0.7:
                tp_percent *= 0.75
                sl_percent *= 1.2
                self.logger.warning(f"[Risk v3] Defensive mode: systemic_risk={systemic_risk:.2f}")
        else:
            # No legacy risk context, apply live adjustment factor
            tp_percent *= tp_adjustment_factor
        
        # ENFORCE MINIMUM DISTANCES (prevent micro exits!)
        tp_percent = max(tp_percent, self.min_tp_pct)
        sl_percent = max(sl_percent, self.min_sl_pct)
        
        # Ensure minimum risk-reward ratio
        current_rr = tp_percent / sl_percent
        if current_rr < self.min_rr_ratio:
            tp_percent = sl_percent * self.min_rr_ratio
        elif current_rr > self.max_rr_ratio:
            sl_percent = tp_percent / self.max_rr_ratio
        
        # Calculate trailing stop (proportional to TP)
        trail_percent = tp_percent * 0.4  # 40% of TP distance
        
        # Partial TP for high-confidence trades
        partial_tp = signal_confidence >= 0.80
        
        # Build rationale
        rationale = (
            f"Confidence: {signal_confidence:.1%} → "
            f"TP: {tp_percent:.1%}, SL: {sl_percent:.1%}, "
            f"R:R = {tp_percent/sl_percent:.1f}x, "
            f"Mode: {risk_mode}"
        )
        
        output = DynamicTPSLOutput(
            tp_percent=tp_percent,
            sl_percent=sl_percent,
            trail_percent=trail_percent,
            partial_tp=partial_tp,
            rationale=rationale,
            confidence=signal_confidence,
            timestamp=datetime.now(timezone.utc)
        )
        
        self.logger.info(
            f"🎯 [Dynamic TP/SL] {action} @ {signal_confidence:.1%} → "
            f"TP: {tp_percent:.2%}, SL: {sl_percent:.2%}, Trail: {trail_percent:.2%}, "
            f"R:R: {tp_percent/sl_percent:.2f}x"
        )
        
        return output
    
    def _get_risk_mode_multipliers(self, risk_mode: str) -> Dict[str, float]:
        """
        Get TP/SL multipliers based on AI-HFOS risk mode.
        
        Returns:
            Dict with 'tp' and 'sl' multipliers
        """
        multipliers = {
            "NORMAL": {"tp": 1.0, "sl": 1.0},
            "OPTIMISTIC": {"tp": 1.15, "sl": 0.9},  # Wider TP, tighter SL
            "AGGRESSIVE": {"tp": 1.30, "sl": 0.85},  # Much wider TP, tight SL
            "CRITICAL": {"tp": 0.70, "sl": 1.3}  # Narrow TP, wider SL (defensive)
        }
        
        return multipliers.get(risk_mode, multipliers["NORMAL"])
    
    def adjust_for_position_age(
        self,
        original_tpsl: DynamicTPSLOutput,
        position_age_minutes: int,
        unrealized_pnl_pct: float
    ) -> Tuple[float, float]:
        """
        Adjust TP/SL based on position age and performance.
        
        Args:
            original_tpsl: Original TP/SL calculation
            position_age_minutes: How long position has been open
            unrealized_pnl_pct: Current unrealized PnL
        
        Returns:
            (adjusted_tp, adjusted_sl) as percentages
        """
        tp = original_tpsl.tp_percent
        sl = original_tpsl.sl_percent
        
        # If position is winning and mature, tighten SL (lock in profits)
        if unrealized_pnl_pct > 0.02 and position_age_minutes > 30:
            # Move SL to breakeven or better
            sl = min(sl, unrealized_pnl_pct * 0.3)  # SL at 30% of current profit
            self.logger.info(
                f"🎯 [Dynamic TP/SL] Tightening SL for winning position: "
                f"{original_tpsl.sl_percent:.2%} → {sl:.2%}"
            )
        
        # If position is old and breakeven, tighten both
        elif abs(unrealized_pnl_pct) < 0.005 and position_age_minutes > 60:
            tp = tp * 0.8
            sl = sl * 0.8
            self.logger.info(
                f"🎯 [Dynamic TP/SL] Tightening range for stale position: "
                f"TP {original_tpsl.tp_percent:.2%} → {tp:.2%}, "
                f"SL {original_tpsl.sl_percent:.2%} → {sl:.2%}"
            )
        
        return tp, sl


# Global singleton
_dynamic_tpsl_calculator: Optional[DynamicTPSLCalculator] = None


def get_dynamic_tpsl_calculator() -> DynamicTPSLCalculator:
    """Get or create Dynamic TP/SL Calculator singleton"""
    global _dynamic_tpsl_calculator
    if _dynamic_tpsl_calculator is None:
        _dynamic_tpsl_calculator = DynamicTPSLCalculator()
    return _dynamic_tpsl_calculator


def calculate_dynamic_tpsl(
    confidence: float,
    action: str,
    risk_mode: str = "NORMAL"
) -> Dict[str, Any]:
    """
    Convenience function to calculate dynamic TP/SL.
    
    Returns:
        Dict with tp_percent, sl_percent, trail_percent, partial_tp
    """
    calculator = get_dynamic_tpsl_calculator()
    output = calculator.calculate(confidence, action, risk_mode=risk_mode)
    
    return {
        "tp_percent": output.tp_percent,
        "sl_percent": output.sl_percent,
        "trail_percent": output.trail_percent,
        "partial_tp": output.partial_tp,
        "rationale": output.rationale
    }
