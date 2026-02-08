"""
AI Engine Exit Evaluator

🧠 Intelligent exit decision making based on:
- Market regime changes
- Volatility structure  
- Ensemble confidence degradation
- R-momentum (profit acceleration)
- Position age
- Extreme profit levels
- **FEE IMPACT** (trading fees + funding fees accumulation)

This replaces hardcoded R-level thresholds with AI-driven dynamic decisions.

🔧 FIX DEPLOYED FEB 7 2026:
- Rebalanced scoring: reduced hold factors, increased exit factors
- Emergency exit added: R > 8 → immediate CLOSE
- Dynamic profit scaling: R > 5 → +4 points, R > 3 → +3 points
- Lowered thresholds: CLOSE requires exit > hold + 2 (was +3)
- Lowered thresholds: PARTIAL requires exit >= hold - 1 (was exit > hold)
- Result: Positions with R > 3 will now trigger partial closes
- Result: Positions with R > 8 will immediately full close

🔧 FEE-AWARENESS FIX FEB 8 2026:
- Calculate R_net_after_fees accounting for:
  - Entry/exit trading fees (~0.04% each)
  - Funding fees (0.01% per 8h cycle)
- Exit pressure increases for positions held >24h (funding accumulation)
- FEE_PROTECTION: Force close if net R < 1.0 even if gross R > 0
- Prevents break-even/loss trades after fee deduction
- All profit thresholds now use fee-adjusted R values
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExitEvaluation:
    """Result of exit evaluation"""
    action: str  # HOLD, PARTIAL_CLOSE, CLOSE
    percentage: float  # 0.0-1.0
    reason: str
    factors: Dict[str, Any]
    current_regime: str
    hold_score: int
    exit_score: int
    timestamp: int


class ExitEvaluator:
    """
    AI-driven exit evaluation
    
    Evaluates positions and recommends dynamic exit actions based on
    multiple AI-derived factors instead of hardcoded R-level thresholds.
    """
    
    def __init__(self, regime_detector=None, vse=None, ensemble=None):
        self.regime_detector = regime_detector
        self.vse = vse  # Volatility Structure Engine
        self.ensemble = ensemble
        logger.info("[ExitEvaluator] Initialized with AI components")
    
    async def evaluate_exit(self, position_data: Dict[str, Any]) -> ExitEvaluation:
        """
        🧠 AI-driven exit evaluation
        
        Args:
            position_data: Dict with keys:
                - symbol: str
                - side: "LONG"/"SHORT"
                - entry_price: float
                - current_price: float
                - position_qty: float
                - entry_timestamp: int
                - age_sec: int
                - R_net: float
                - R_history: List[float] (recent R values)
                - entry_regime: str (optional)
                - entry_confidence: float (optional)
                - stop_loss: float (optional)
                - take_profit: float (optional)
                - peak_price: float (optional)
        
        Returns:
            ExitEvaluation with action, percentage, reason, and factors
        """
        symbol = position_data["symbol"]
        R_net = position_data["R_net"]
        age_sec = position_data["age_sec"]
        side = position_data["side"]
        current_price = position_data["current_price"]
        entry_price = position_data["entry_price"]
        
        # === FACTOR EVALUATION ===
        
        # FACTOR 1: REGIME STATUS
        regime_changed, current_regime = await self._evaluate_regime(
            symbol, position_data.get("entry_regime", "TRENDING")
        )
        
        # FACTOR 2: VOLATILITY DYNAMICS
        vol_expanding = await self._evaluate_volatility(symbol)
        
        # FACTOR 3: ENSEMBLE CONFIDENCE
        confidence_degraded = await self._evaluate_confidence(
            symbol, position_data.get("entry_confidence", 0.7)
        )
        
        # FACTOR 4: R-MOMENTUM
        momentum_strong = self._evaluate_r_momentum(
            position_data.get("R_history", [R_net])
        )
        
        # FACTOR 5: PEAK DISTANCE (price action strength)
        near_peak = self._evaluate_peak_distance(
            side, current_price, position_data.get("peak_price", current_price)
        )
        
        # FACTOR 6: TIME DECAY & FEE ACCUMULATION
        age_hours = age_sec / 3600
        position_old = (age_hours > 6)
        position_fresh = (age_hours < 1)
        
        # 💰 FEE IMPACT: Estimate accumulated fees
        # Entry fee: ~0.04%, Exit fee: ~0.04%, Funding: ~0.01% per 8h
        funding_cycles = int(age_hours / 8)
        estimated_fee_pct = 0.04 + 0.04 + (funding_cycles * 0.01)  # Total fee %
        
        # Fee impact on R (as percentage of risk taken)
        # If we risked 2% to make this trade, 0.1% fee = 0.05R eaten
        fee_impact_on_R = estimated_fee_pct / 2.0  # Assume 2% risk typical
        
        # Adjust R_net for fees (realistic profit after costs)
        R_net_after_fees = R_net - fee_impact_on_R
        
        # Flag positions where fees are eating profits
        fees_eating_profit = (R_net > 0 and R_net_after_fees < 1.0)
        position_held_too_long = (age_hours > 24)  # Funding fees accumulating
        
        # === SCORING SYSTEM ===
        hold_score = 0
        exit_score = 0
        
        # 🛡️ HOLD FACTORS (positive for keeping position) - REBALANCED
        if not regime_changed:
            hold_score += 2  # Regime intact (reduced from 3)
        if vol_expanding:
            hold_score += 1  # Volatility growing (reduced from 2)
        if momentum_strong:
            hold_score += 2  # Strong R acceleration (reduced from 3)
        if not confidence_degraded:
            hold_score += 1  # Models still confident (reduced from 2)
        if position_fresh and R_net > 1:
            hold_score += 1  # Let it breathe (reduced from 2)
        if near_peak:
            hold_score += 1  # Still at peak, momentum intact (reduced from 2)
        
        # 💰 EXIT FACTORS (positive for taking profit) - ENHANCED
        if regime_changed:
            exit_score += 5  # Regime flipped (increased from 4)
        if not vol_expanding:
            exit_score += 3  # Volatility contracting (increased from 2)
        if confidence_degraded:
            exit_score += 4  # Models diverging (increased from 3)
        if position_old and R_net < 5:
            exit_score += 3  # Old position, rotate capital (increased from 2)
        
        # 🚨 FEE-BASED EXIT PRESSURE
        if fees_eating_profit:
            exit_score += 4  # Fees consuming profits, close before break-even!
        if position_held_too_long:
            exit_score += 2  # 24+ hours = funding fees accumulating
        
        # 🚨 PROFIT-BASED EXITS (dynamic scaling, fee-adjusted)
        if R_net_after_fees > 8:
            exit_score += 6  # EXTREME profits after fees - emergency exit
        elif R_net_after_fees > 5:
            exit_score += 4  # Very high profits after fees
        elif R_net_after_fees > 3:
            exit_score += 3  # High profits after fees
        elif R_net_after_fees > 2:
            exit_score += 1  # Decent profits after fees
        elif R_net_after_fees < 1 and R_net > 0:
            exit_score += 3  # Gross profit but net break-even = close NOW
        
        if not near_peak:
            exit_score += 3  # Far from peak, momentum dead (increased from 2)
        
        # === DECISION LOGIC ===
        
        # 🚨 EMERGENCY EXIT: Extreme profits (fee-adjusted) override all logic
        if R_net_after_fees > 8:
            action = "CLOSE"
            percentage = 1.0
            reason = f"EMERGENCY_EXIT_R={R_net_after_fees:.1f}_after_fees (gross_R={R_net:.1f}, fees={fee_impact_on_R:.2f}R, exit={exit_score} vs hold={hold_score})"
        
        # 🚨 FEE PROTECTION: Close positions where fees eating all profits
        elif fees_eating_profit and R_net > 0:
            action = "CLOSE"
            percentage = 1.0
            reason = f"FEE_PROTECTION_net_R={R_net_after_fees:.1f} (gross={R_net:.1f}, fees_eaten={fee_impact_on_R:.2f}R over {age_hours:.1f}h)"
        
        # 💪 STRONG EXIT SIGNAL (lowered threshold from +3 to +2)
        elif exit_score > hold_score + 2:
            if regime_changed or confidence_degraded:
                action = "CLOSE"
                percentage = 1.0  # 100%
                reason = f"regime_flip+confidence_lost (exit={exit_score} vs hold={hold_score})"
            else:
                action = "PARTIAL_CLOSE"
                # Dynamic percentage based on exit pressure
                percentage = min(0.75, (exit_score / 12))  # 30-75%
                reason = f"exit_score={exit_score}_conditions_weakening"
        
        # ⚖️ MODERATE EXIT SIGNAL (lowered threshold to allow earlier exits)
        elif exit_score >= hold_score - 1:
            action = "PARTIAL_CLOSE"
            # Scale with R-level: higher R = more profit taking
            base_pct = 0.25 + min(0.30, R_net / 20)  # 25-55%
            # Boost for strong exit scores
            score_boost = max(0, (exit_score - hold_score) * 0.10)
            percentage = min(0.75, base_pct + score_boost)
            reason = f"profit_taking_R={R_net:.1f} (exit={exit_score} vs hold={hold_score})"
        
        # 🛡️ HOLD POSITION
        else:
            action = "HOLD"
            percentage = 0.0
            reason = f"momentum_strong (hold={hold_score} vs exit={exit_score})"
        
        return ExitEvaluation(
            action=action,
            percentage=round(percentage, 2),
            reason=reason,
            factors={
                "regime_changed": regime_changed,
                "vol_expanding": vol_expanding,
                "momentum_strong": momentum_strong,
                "confidence_degraded": confidence_degraded,
                "near_peak": near_peak,
                "position_old": position_old,
                "position_fresh": position_fresh,
                "position_held_too_long": position_held_too_long,
                "fees_eating_profit": fees_eating_profit,
                "hold_score": hold_score,
                "exit_score": exit_score,
                "age_hours": round(age_hours, 1),
                "R_net": round(R_net, 2),
                "R_net_after_fees": round(R_net_after_fees, 2),
                "estimated_fee_pct": round(estimated_fee_pct, 3),
                "fee_impact_on_R": round(fee_impact_on_R, 2),
                "funding_cycles": funding_cycles
            },
            current_regime=current_regime,
            hold_score=hold_score,
            exit_score=exit_score,
            timestamp=int(time.time())
        )
    
    async def _evaluate_regime(self, symbol: str, entry_regime: str) -> tuple:
        """Check if market regime has changed since entry"""
        if not self.regime_detector:
            return False, "UNKNOWN"
        
        try:
            current_regime = self.regime_detector.get_regime(symbol)
            regime_name = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)
            changed = (regime_name != entry_regime)
            return changed, regime_name
        except Exception as e:
            logger.warning(f"[ExitEval] Regime check failed for {symbol}: {e}")
            return False, "UNKNOWN"
    
    async def _evaluate_volatility(self, symbol: str) -> bool:
        """Check if volatility is expanding (positive momentum signal)"""
        if not self.vse:
            return True  # Default optimistic if no VSE
        
        try:
            structure = await self.vse.get_structure(symbol)
            if structure:
                atr_gradient = structure.get("atr_gradient", 0)
                return atr_gradient > 0.1  # Positive gradient = expanding
        except Exception as e:
            logger.warning(f"[ExitEval] Volatility check failed for {symbol}: {e}")
        
        return True  # Fail-open
    
    async def _evaluate_confidence(self, symbol: str, entry_confidence: float) -> bool:
        """Check if ensemble confidence has degraded significantly"""
        if not self.ensemble:
            return False  # Default: no degradation if no ensemble
        
        try:
            # Re-run ensemble to get current confidence
            current_signal = await self.ensemble.get_signal(symbol)
            if current_signal:
                current_conf = current_signal.get("confidence", entry_confidence)
                degraded = (current_conf < entry_confidence - 0.15)  # 15% drop threshold
                return degraded
        except Exception as e:
            logger.warning(f"[ExitEval] Confidence check failed for {symbol}: {e}")
        
        return False  # Fail-open
    
    def _evaluate_r_momentum(self, R_history: list) -> bool:
        """Check if R is accelerating (strong momentum)"""
        if len(R_history) < 3:
            return False
        
        # Calculate R acceleration over last 3 measurements
        R_acceleration = (R_history[-1] - R_history[-3]) / 2
        return R_acceleration > 0.3  # Strong positive acceleration
    
    def _evaluate_peak_distance(self, side: str, current_price: float, peak_price: float) -> bool:
        """Check if price is near peak (strong momentum)"""
        if peak_price == 0 or current_price == 0:
            return True  # Assume near peak if no data
        
        if side == 'LONG':
            distance = (peak_price - current_price) / peak_price
        else:
            distance = (current_price - peak_price) / peak_price
        
        return distance < 0.02  # Within 2% of peak

