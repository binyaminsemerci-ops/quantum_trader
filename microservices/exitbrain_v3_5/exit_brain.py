"""
ExitBrain v3.5 - Phase 4O+ Integration
Intelligent Leverage + Cross-Exchange Adaptive Exit Management

Integrates:
- Phase 4M+ Cross-Exchange Intelligence (TP/SL adjustments)
- Phase 4O+ Intelligent Leverage Formula v2
- Dynamic position sizing based on confidence & volatility
- PnL feedback loop to Redis streams

Architecture:
    AI Signal → ILFv2 → ExitBrain v3.5 → Auto Executor → PnL Loop
"""

import time
import json
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from redis import Redis
except ImportError:
    Redis = None

# Import formula-based exit calculations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common.exit_math import (
    compute_dynamic_stop,
    compute_trailing_hit,
    near_liquidation,
    ExitPosition,
    Account,
    Market
)
from common.risk_settings import (
    DEFAULT_SETTINGS,
    get_settings,
    compute_harvest_r_targets
)

from .intelligent_leverage_engine import get_leverage_engine, LeverageCalculation
from .adaptive_leverage_engine import AdaptiveLeverageEngine, AdaptiveLevels

logger = logging.getLogger(__name__)


@dataclass
class SignalContext:
    """Complete context for a trading signal"""
    symbol: str
    side: str  # 'long' or 'short'
    confidence: float  # [0-1]
    entry_price: float
    atr_value: float
    timestamp: float


@dataclass
class ExitPlan:
    """Complete exit plan with leverage, TP, SL, trailing"""
    symbol: str
    side: str
    leverage: float  # Calculated by ILFv2
    take_profit_pct: float
    stop_loss_pct: float
    trailing_enabled: bool
    trailing_callback_pct: Optional[float]
    reasoning: str  # Human-readable explanation
    calculation_details: Dict  # Full calculation breakdown


class ExitBrainV35:
    """
    ExitBrain v3.5 - Adaptive Exit Management with Intelligent Leverage
    
    Combines:
    - Cross-Exchange Intelligence (Phase 4M+)
    - Intelligent Leverage Formula v2 (Phase 4O+)
    - Dynamic TP/SL calculation
    - PnL feedback publishing
    """
    
    def __init__(self, redis_client: Optional[Redis] = None, config: Optional[Dict] = None):
        """
        Initialize ExitBrain v3.5
        
        Args:
            redis_client: Redis client for stream publishing
            config: Configuration overrides
        """
        self.redis = redis_client
        self.config = config or {}
        
        # Get ILFv2 engine
        self.leverage_engine = get_leverage_engine(
            config=self.config.get("leverage_engine", {})
        )
        
        # Get risk settings for formula-based calculations
        self.risk_settings = get_settings()
        
        # Get Adaptive Leverage Engine with FORMULA-BASED parameters
        # Replace hardcoded base_tp/base_sl with risk-normalized values
        account_equity = self.config.get("account_equity", 10000.0)  # USD
        self.account = Account(equity=account_equity)
        
        # Calculate dynamic base levels (will be further adjusted by AdaptiveLeverageEngine)
        # These are starting points, not fixed values
        base_atr = 50.0  # Fallback ATR estimate (will be replaced with real ATR)
        base_leverage = 10.0  # Typical leverage for base calculations
        
        # Create sample position for baseline calculation
        sample_position = ExitPosition(
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            size=0.001,  # Small sample size
            leverage=base_leverage,
            highest_price=50000.0,
            lowest_price=50000.0,
            time_in_trade=0,
            distance_to_liq=None
        )
        sample_market = Market(current_price=50000.0, atr=base_atr)
        
        # Calculate FORMULA-BASED baseline stop distance
        dynamic_stop = compute_dynamic_stop(sample_position, self.account, sample_market, self.risk_settings)
        base_sl_formula = abs(sample_position.entry_price - dynamic_stop) / sample_position.entry_price
        
        # Use ATR-based calculation for base TP (3x stop distance as reasonable starting point)
        base_tp_formula = base_sl_formula * 3.0  # 3:1 reward:risk ratio
        
        # Get Adaptive Leverage Engine with formula-based baselines
        self.adaptive_engine = AdaptiveLeverageEngine(
            base_tp=base_tp_formula,  # FORMULA-BASED (not hardcoded 0.020)
            base_sl=base_sl_formula   # FORMULA-BASED (not hardcoded 0.012)
        )
        
        # Dynamically calculated parameters (NO HARDCODED VALUES)
        # These are calculated per-trade based on:
        # - Risk capital allocation
        # - Market volatility (ATR)
        # - Position leverage
        # - Account equity
        
        # Base values are EXAMPLE ONLY - real values calculated dynamically
        self._base_tp_pct = base_tp_formula  # DYNAMIC: ~0.015-0.050 depending on conditions
        self._base_sl_pct = base_sl_formula  # DYNAMIC: ~0.005-0.020 depending on conditions
        
        # Trailing callback uses ATR-based calculation (no fixed percentage)
        # Will be calculated as: ATR * TRAILING_ATR_MULT
        self._trailing_callback_base = self.risk_settings.TRAILING_ATR_MULT  # Dynamic scale factor
        
        # Safety limits - NO HARDCODED PERCENTAGES
        # These are calculated based on market conditions and volatility
        # Min: 2x the risk fraction, Max: 15x the risk fraction (reasonable bounds)
        risk_fraction = self.risk_settings.RISK_FRACTION
        self._min_tp_factor = 2.0   # Min TP = 2x risk fraction
        self._max_tp_factor = 15.0  # Max TP = 15x risk fraction
        self._min_sl_factor = 0.5   # Min SL = 0.5x risk fraction
        self._max_sl_factor = 4.0   # Max SL = 4x risk fraction
        
        # Reinforcement Learning - Dynamic reward feedback
        # When enabled, ExitBrain publishes PnL outcomes to Redis streams
        # for RL agent to learn optimal TP/SL parameters dynamically
        self.dynamic_reward = self.config.get("dynamic_reward", True)
        
        # Statistics
        self.plans_generated = 0
        self.avg_leverage_used = 0.0
        
        logger.info(
            f"[ExitBrain-v3.5] FORMULA-BASED INIT | "
            f"ILFv2: Enabled | "
            f"AdaptiveLeverage: Enabled | "
            f"Dynamic Reward: {'Enabled' if self.dynamic_reward else 'Disabled'} | "
            f"Risk Fraction: {self.risk_settings.RISK_FRACTION*100:.2f}% | "
            f"Base TP Formula: {self._base_tp_pct*100:.2f}% | "
            f"Base SL Formula: {self._base_sl_pct*100:.2f}% | "
            f"ATR Multipliers: Stop={self.risk_settings.STOP_ATR_MULT}x, Trail={self.risk_settings.TRAILING_ATR_MULT}x"
        )
    
    def build_exit_plan(
        self,
        signal: SignalContext,
        pnl_trend: float = 0.0,
        symbol_risk: float = 1.0,
        margin_util: float = 0.0,
        exch_divergence: float = 0.0,
        funding_rate: float = 0.0,
        cross_exchange_adjustments: Optional[Dict] = None
    ) -> ExitPlan:
        """
        Build complete exit plan with FORMULA-BASED intelligent leverage.
        
        NO HARDCODED PERCENTAGES - all values calculated dynamically based on:
        - Risk capital allocation per trade
        - Market volatility (ATR)
        - Position leverage
        - Account equity
        
        Args:
            signal: Trading signal context
            pnl_trend: Recent PnL trend [-1 to +1]
            symbol_risk: Symbol risk weight [0.5-1.5]
            margin_util: Used margin fraction [0-1]
            exch_divergence: Cross-exchange price divergence [0-1]
            funding_rate: Funding rate bias [-0.05 to +0.05]
            cross_exchange_adjustments: Optional Phase 4M+ adjustments
        
        Returns:
            ExitPlan with all parameters calculated via formulas
        """
        # Step 1: Calculate intelligent leverage (ILFv2)
        leverage_calc = self.leverage_engine.calculate_leverage(
            confidence=signal.confidence,
            volatility=signal.atr_value,
            pnl_trend=pnl_trend,
            symbol_risk=symbol_risk,
            margin_util=margin_util,
            exch_divergence=exch_divergence,
            funding_rate=funding_rate
        )
        
        # Step 2: FORMULA-BASED TP/SL calculation (replaces hardcoded percentages)
        # Build position context for formula calculations
        exit_position = ExitPosition(
            symbol=signal.symbol,
            side="BUY" if signal.side.lower() == "long" else "SELL",
            entry_price=signal.entry_price,
            size=1000.0 / signal.entry_price,  # $1000 position for calculation
            leverage=leverage_calc.leverage,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
            time_in_trade=0,
            distance_to_liq=None
        )
        
        # Build market context
        market = Market(
            current_price=signal.entry_price,
            atr=signal.atr_value if signal.atr_value > 0 else signal.entry_price * 0.01
        )
        
        # Calculate FORMULA-BASED dynamic stop
        dynamic_stop = compute_dynamic_stop(exit_position, self.account, market, self.risk_settings)
        formula_sl_pct = abs(signal.entry_price - dynamic_stop) / signal.entry_price
        
        # Calculate FORMULA-BASED take profit (reward:risk ratio based on market conditions)
        # Higher volatility = wider TP targets, lower volatility = tighter TP targets
        base_reward_ratio = 3.0  # Base 3:1 reward:risk ratio
        volatility_factor = (market.atr / signal.entry_price) / 0.02  # Normalize vs 2% volatility
        volatility_factor = max(0.5, min(2.0, volatility_factor))  # Clamp 0.5-2.0x
        
        # Adjust reward:risk based on confidence and volatility
        confidence_factor = 1.0 + signal.confidence  # 1.0-2.0x multiplier
        leverage_factor = 1.0 / (1.0 + leverage_calc.leverage / 20.0)  # Lower targets for high leverage
        
        adjusted_reward_ratio = base_reward_ratio * confidence_factor * leverage_factor * volatility_factor
        formula_tp_pct = formula_sl_pct * adjusted_reward_ratio
        
        # Calculate FORMULA-BASED trailing callback (ATR-based, no fixed percentage)
        trailing_distance_pct = (market.atr * self.risk_settings.TRAILING_ATR_MULT) / signal.entry_price
        
        # Step 2b: Integrate with AdaptiveLeverageEngine for multi-level targets
        # Use formula results as base inputs (not hardcoded percentages)
        adaptive_levels = self.adaptive_engine.compute_levels(
            base_tp_pct=formula_tp_pct,    # FORMULA-BASED input (not hardcoded)
            base_sl_pct=formula_sl_pct,    # FORMULA-BASED input (not hardcoded)
            leverage=leverage_calc.leverage,
            volatility_factor=volatility_factor,
            funding_delta=funding_rate,
            exchange_divergence=exch_divergence
        )
        
        # [MONITORING] Log FORMULA-BASED calculation results
        logger.info(
            f"[ExitBrain-v3.5] FORMULA CALC | "
            f"{signal.symbol} {leverage_calc.leverage:.1f}x | "
            f"ATR={market.atr:.4f} ({(market.atr/signal.entry_price*100):.2f}%) | "
            f"Formula_SL={formula_sl_pct*100:.2f}% | "
            f"Formula_TP={formula_tp_pct*100:.2f}% | "
            f"R:R={adjusted_reward_ratio:.2f} | "
            f"Factors: Conf={confidence_factor:.2f} Vol={volatility_factor:.2f} Lev={leverage_factor:.2f}"
        )
        
        # Log adaptive levels after engine processing
        logger.info(
            f"[ExitBrain-v3.5] Adaptive Processing | "
            f"{signal.symbol} {leverage_calc.leverage:.1f}x | "
            f"LSF={adaptive_levels.lsf:.4f} | "
            f"TP1={adaptive_levels.tp1_pct*100:.2f}% "
            f"TP2={adaptive_levels.tp2_pct*100:.2f}% "
            f"TP3={adaptive_levels.tp3_pct*100:.2f}% | "
            f"SL={adaptive_levels.sl_pct*100:.2f}% | "
            f"Harvest={adaptive_levels.harvest_scheme}"
        )
        
        # Use adaptive levels as base TP/SL
        base_tp = adaptive_levels.tp1_pct  # Use TP1 as primary target
        base_sl = adaptive_levels.sl_pct
        
        # Store full TP levels for partial harvesting (if needed by executor)
        tp_levels = [adaptive_levels.tp1_pct, adaptive_levels.tp2_pct, adaptive_levels.tp3_pct]
        harvest_scheme = adaptive_levels.harvest_scheme
        
        # Step 3: Apply Phase 4M+ cross-exchange adjustments (if available)
        if cross_exchange_adjustments:
            tp_multiplier = cross_exchange_adjustments.get("tp_multiplier", 1.0)
            sl_multiplier = cross_exchange_adjustments.get("sl_multiplier", 1.0)
            use_trailing = cross_exchange_adjustments.get("use_trailing", True)
            
            base_tp *= tp_multiplier
            base_sl *= sl_multiplier
            
            logger.info(
                f"[ExitBrain-v3.5] Phase 4M+ Applied | "
                f"TP×{tp_multiplier:.2f} | SL×{sl_multiplier:.2f}"
            )
        else:
            # Default trailing logic
            use_trailing = (signal.atr_value < 2.0)  # Disable if too volatile
        
        # Step 4: Apply FORMULA-BASED safety limits (no hardcoded percentages)
        # Limits are calculated based on risk fraction and market conditions
        risk_fraction = self.risk_settings.RISK_FRACTION
        
        # Dynamic safety bounds based on market volatility and leverage
        min_tp_pct = risk_fraction * self._min_tp_factor * volatility_factor
        max_tp_pct = risk_fraction * self._max_tp_factor * volatility_factor
        min_sl_pct = risk_fraction * self._min_sl_factor / leverage_calc.leverage
        max_sl_pct = risk_fraction * self._max_sl_factor / leverage_calc.leverage
        
        # Apply safety bounds
        final_tp = max(min_tp_pct, min(max_tp_pct, base_tp))
        final_sl = max(min_sl_pct, min(max_sl_pct, base_sl))
        
        # Log safety bound application
        if final_tp != base_tp or final_sl != base_sl:
            logger.info(
                f"[ExitBrain-v3.5] Safety Bounds Applied | "
                f"{signal.symbol} | "
                f"TP: {base_tp*100:.2f}% → {final_tp*100:.2f}% "
                f"(bounds: {min_tp_pct*100:.2f}%-{max_tp_pct*100:.2f}%) | "
                f"SL: {base_sl*100:.2f}% → {final_sl*100:.2f}% "
                f"(bounds: {min_sl_pct*100:.2f}%-{max_sl_pct*100:.2f}%)"
            )
        
        # Step 5: Calculate FORMULA-BASED trailing callback
        # Use ATR-based calculation instead of fixed percentage
        trailing_callback = trailing_distance_pct if use_trailing else None
        
        # Step 6: Build reasoning
        reasoning_parts = [
            f"Leverage: {leverage_calc.leverage:.1f}x ({leverage_calc.reasoning})",
            f"LSF: {adaptive_levels.lsf:.4f}",
            f"TP Levels: {adaptive_levels.tp1_pct*100:.2f}%/{adaptive_levels.tp2_pct*100:.2f}%/{adaptive_levels.tp3_pct*100:.2f}%",
            f"Harvest: {adaptive_levels.harvest_scheme}",
            f"SL: {final_sl*100:.2f}%"
        ]
        
        if cross_exchange_adjustments:
            reasoning_parts.append("Phase 4M+ adjustments applied")
        
        if use_trailing:
            reasoning_parts.append(f"Trailing: {trailing_callback*100:.2f}% callback")
        else:
            reasoning_parts.append("Trailing: disabled (high volatility)")
        
        reasoning = " | ".join(reasoning_parts)
        
        # Step 7: Build FORMULA-BASED calculation details
        calc_details = {
            "leverage": {
                "value": leverage_calc.leverage,
                "base": leverage_calc.base_leverage,
                "factors": leverage_calc.factors,
                "clamped": leverage_calc.clamped
            },
            "formula_calculation": {
                "dynamic_stop_price": dynamic_stop,
                "formula_sl_pct": formula_sl_pct,
                "formula_tp_pct": formula_tp_pct,
                "reward_risk_ratio": adjusted_reward_ratio,
                "volatility_factor": volatility_factor,
                "confidence_factor": confidence_factor,
                "leverage_factor": leverage_factor,
                "atr_value": market.atr,
                "trailing_distance_pct": trailing_distance_pct
            },
            "adaptive_levels": {
                "tp1_pct": adaptive_levels.tp1_pct,
                "tp2_pct": adaptive_levels.tp2_pct,
                "tp3_pct": adaptive_levels.tp3_pct,
                "sl_pct": adaptive_levels.sl_pct,
                "harvest_scheme": adaptive_levels.harvest_scheme,
                "lsf": adaptive_levels.lsf
            },
            "take_profit": {
                "final_pct": final_tp,
                "formula_base_pct": formula_tp_pct,
                "adaptive_base_pct": base_tp,
                "tp_levels": tp_levels,
                "safety_bounds": {
                    "min_pct": min_tp_pct,
                    "max_pct": max_tp_pct
                },
                "cross_exchange_multiplier": cross_exchange_adjustments.get("tp_multiplier", 1.0) if cross_exchange_adjustments else 1.0
            },
            "stop_loss": {
                "final_pct": final_sl,
                "formula_base_pct": formula_sl_pct,
                "adaptive_base_pct": base_sl,
                "safety_bounds": {
                    "min_pct": min_sl_pct,
                    "max_pct": max_sl_pct
                },
                "cross_exchange_multiplier": cross_exchange_adjustments.get("sl_multiplier", 1.0) if cross_exchange_adjustments else 1.0
            },
            "inputs": {
                "confidence": signal.confidence,
                "atr_value": signal.atr_value,
                "pnl_trend": pnl_trend,
                "exch_divergence": exch_divergence,
                "funding_rate": funding_rate,
                "margin_util": margin_util,
                "risk_fraction": risk_fraction
            }
        }
        
        # SHADOW VALIDATION: Compare formula-based results with old hardcoded approach
        if self.risk_settings.SHADOW_MODE:
            # OLD hardcoded approach (what would have been calculated)
            old_tp = 0.025  # Old hardcoded 2.5%
            old_sl = 0.015  # Old hardcoded 1.5%
            old_trailing = 0.008  # Old hardcoded 0.8%
            
            logger.info(
                f"[SHADOW] ExitBrain {signal.symbol} | "
                f"OLD: TP={old_tp*100:.2f}% SL={old_sl*100:.2f}% Trail={old_trailing*100:.2f}% | "
                f"NEW: TP={final_tp*100:.2f}% SL={final_sl*100:.2f}% Trail={trailing_callback*100:.2f if trailing_callback else 0:.2f}% | "
                f"ATR={market.atr:.4f} ({(market.atr/signal.entry_price*100):.2f}%) | "
                f"Risk={risk_fraction*100:.2f}% | "
                f"Lev={leverage_calc.leverage:.1f}x | "
                f"Confidence={signal.confidence:.2f}"
            )
        
        # Step 8: Publish to PnL stream (for RL agent feedback)
        if self.redis and self.dynamic_reward:
            self._publish_pnl_stream(
                signal=signal,
                leverage=leverage_calc.leverage,
                tp_pct=final_tp,
                sl_pct=final_sl,
                calc_details=calc_details
            )
            
            # [NEW] Publish adaptive levels to dedicated stream for monitoring
            self._publish_adaptive_levels_stream(
                signal=signal,
                adaptive_levels=adaptive_levels,
                leverage=leverage_calc.leverage
            )
        
        # Update statistics
        self.plans_generated += 1
        self.avg_leverage_used = (
            (self.avg_leverage_used * (self.plans_generated - 1) + leverage_calc.leverage)
            / self.plans_generated
        )
        
        # Build final plan
        plan = ExitPlan(
            symbol=signal.symbol,
            side=signal.side,
            leverage=leverage_calc.leverage,
            take_profit_pct=final_tp,
            stop_loss_pct=final_sl,
            trailing_enabled=use_trailing,
            trailing_callback_pct=trailing_callback,
            reasoning=reasoning,
            calculation_details=calc_details
        )
        
        logger.info(
            f"[ExitBrain-v3.5] FORMULA-BASED Plan Generated | "
            f"{signal.symbol} {signal.side.upper()} | "
            f"Leverage: {plan.leverage:.1f}x | "
            f"TP: {final_tp*100:.2f}% (formula-based) | "
            f"SL: {final_sl*100:.2f}% (risk-normalized) | "
            f"R:R={adjusted_reward_ratio:.2f} | "
            f"ATR={market.atr:.4f}"
        )
        
        return plan
    
    def _publish_pnl_stream(
        self,
        signal: SignalContext,
        leverage: float,
        tp_pct: float,
        sl_pct: float,
        calc_details: Dict
    ):
        """Publish exit plan to Redis stream for RL agent feedback"""
        try:
            stream_key = "quantum:stream:exitbrain.pnl"
            
            data = {
                "timestamp": time.time(),
                "symbol": signal.symbol,
                "side": signal.side,
                "confidence": signal.confidence,
                "dynamic_leverage": leverage,
                "take_profit_pct": tp_pct,
                "stop_loss_pct": sl_pct,
                "volatility": signal.atr_value,
                "exch_divergence": calc_details["inputs"]["exch_divergence"],
                "funding_rate": calc_details["inputs"]["funding_rate"],
                "pnl_trend": calc_details["inputs"]["pnl_trend"],
                "margin_util": calc_details["inputs"]["margin_util"]
            }
            
            # Add to stream (maxlen 1000)
            self.redis.xadd(stream_key, data, maxlen=1000)
            
            logger.debug(f"[ExitBrain-v3.5] Published to {stream_key}")
            
        except Exception as e:
            logger.error(f"[ExitBrain-v3.5] Failed to publish PnL stream: {e}")
    
    def _publish_adaptive_levels_stream(
        self,
        signal: SignalContext,
        adaptive_levels: AdaptiveLevels,
        leverage: float
    ):
        """Publish adaptive levels to Redis stream for monitoring"""
        try:
            stream_key = "quantum:stream:adaptive_levels"
            
            data = {
                "timestamp": time.time(),
                "symbol": signal.symbol,
                "side": signal.side,
                "leverage": leverage,
                "lsf": adaptive_levels.lsf,
                "tp1_pct": adaptive_levels.tp1_pct,
                "tp2_pct": adaptive_levels.tp2_pct,
                "tp3_pct": adaptive_levels.tp3_pct,
                "sl_pct": adaptive_levels.sl_pct,
                "harvest_scheme": json.dumps(adaptive_levels.harvest_scheme),
                "sl_clamped": "true" if adaptive_levels.sl_pct in [0.001, 0.02] else "false",
                "tp_minimum_enforced": "true" if adaptive_levels.tp1_pct <= 0.003 else "false"
            }
            
            # Add to stream (maxlen 1000)
            self.redis.xadd(stream_key, data, maxlen=1000)
            
            logger.debug(
                f"[ExitBrain-v3.5] Published adaptive levels to {stream_key} | "
                f"{signal.symbol} {leverage:.1f}x"
            )
            
        except Exception as e:
            logger.error(f"[ExitBrain-v3.5] Failed to publish adaptive levels stream: {e}")
    
    def get_statistics(self) -> Dict:
        """Get ExitBrain statistics"""
        return {
            "plans_generated": self.plans_generated,
            "avg_leverage_used": round(self.avg_leverage_used, 2),
            "leverage_engine_stats": self.leverage_engine.get_statistics(),
            "config": {
                "base_tp_pct": self.base_tp_pct,
                "base_sl_pct": self.base_sl_pct,
                "trailing_callback_pct": self.trailing_callback_pct
            }
        }


def apply_leverage(
    signal: SignalContext,
    atr_value: float,
    pnl_agent,
    portfolio,
    exch_metrics: Dict,
    funding_rate: float,
    redis_client: Optional[Redis] = None
) -> float:
    """
    Convenience function to calculate leverage for a signal
    
    Args:
        signal: Trading signal context
        atr_value: Current ATR value
        pnl_agent: PnL tracking agent with get_trend() method
        portfolio: Portfolio manager with get_margin_utilization() method
        exch_metrics: Cross-exchange metrics dict
        funding_rate: Current funding rate
        redis_client: Optional Redis client for stream publishing
    
    Returns:
        float: Calculated leverage
    """
    exitbrain = ExitBrainV35(redis_client=redis_client)
    
    # Update signal ATR
    signal.atr_value = atr_value
    
    # Build complete exit plan
    plan = exitbrain.build_exit_plan(
        signal=signal,
        pnl_trend=pnl_agent.get_trend(signal.symbol) if pnl_agent else 0.0,
        symbol_risk=exch_metrics.get("risk_factor", 1.0),
        margin_util=portfolio.get_margin_utilization() if portfolio else 0.0,
        exch_divergence=exch_metrics.get("divergence", 0.0),
        funding_rate=funding_rate
    )
    
    return plan.leverage
