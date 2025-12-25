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
        
        # Get Adaptive Leverage Engine
        self.adaptive_engine = AdaptiveLeverageEngine(
            base_tp=self.config.get("base_tp_pct", 0.01),
            base_sl=self.config.get("base_sl_pct", 0.005)
        )
        
        # Base exit parameters (kept for backward compatibility)
        self.base_tp_pct = self.config.get("base_tp_pct", 0.02)  # 2%
        self.base_sl_pct = self.config.get("base_sl_pct", 0.01)  # 1%
        self.trailing_callback_pct = self.config.get("trailing_callback_pct", 0.005)  # 0.5%
        
        # Safety limits
        self.min_tp_pct = self.config.get("min_tp_pct", 0.003)  # 0.3%
        self.max_tp_pct = self.config.get("max_tp_pct", 0.10)   # 10%
        self.min_sl_pct = self.config.get("min_sl_pct", 0.0015)  # 0.15%
        self.max_sl_pct = self.config.get("max_sl_pct", 0.05)   # 5%
        
        # Reinforcement Learning - Dynamic reward feedback
        # When enabled, ExitBrain publishes PnL outcomes to Redis streams
        # for RL agent to learn optimal TP/SL parameters dynamically
        self.dynamic_reward = self.config.get("dynamic_reward", True)
        
        # Statistics
        self.plans_generated = 0
        self.avg_leverage_used = 0.0
        
        logger.info(
            f"[ExitBrain-v3.5] Initialized | "
            f"ILFv2: Enabled | "
            f"AdaptiveLeverage: Enabled | "
            f"Dynamic Reward: {'Enabled' if self.dynamic_reward else 'Disabled'} | "
            f"Base TP: {self.base_tp_pct*100:.1f}% | "
            f"Base SL: {self.base_sl_pct*100:.1f}%"
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
        Build complete exit plan with intelligent leverage
        
        Args:
            signal: Trading signal context
            pnl_trend: Recent PnL trend [-1 to +1]
            symbol_risk: Symbol risk weight [0.5-1.5]
            margin_util: Used margin fraction [0-1]
            exch_divergence: Cross-exchange price divergence [0-1]
            funding_rate: Funding rate bias [-0.05 to +0.05]
            cross_exchange_adjustments: Optional Phase 4M+ adjustments
        
        Returns:
            ExitPlan with all parameters calculated
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
        
        # Step 2: Calculate adaptive TP/SL levels using AdaptiveLeverageEngine
        # This replaces the old hardcoded TP/SL calculations
        adaptive_levels = self.adaptive_engine.compute_levels(
            base_tp_pct=self.base_tp_pct,
            base_sl_pct=self.base_sl_pct,
            leverage=leverage_calc.leverage,
            volatility_factor=signal.atr_value / 100.0 if signal.atr_value > 0 else 0.0,  # Normalize ATR
            funding_delta=funding_rate,
            exchange_divergence=exch_divergence
        )
        
        # [MONITORING] Log adaptive level calculation
        logger.info(
            f"[ExitBrain-v3.5] Adaptive Levels | "
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
        
        # Step 4: Apply safety limits
        final_tp = max(self.min_tp_pct, min(self.max_tp_pct, base_tp))
        final_sl = max(self.min_sl_pct, min(self.max_sl_pct, base_sl))
        
        # Step 5: Calculate trailing callback if enabled
        trailing_callback = self.trailing_callback_pct if use_trailing else None
        
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
        
        # Step 7: Build calculation details
        calc_details = {
            "leverage": {
                "value": leverage_calc.leverage,
                "base": leverage_calc.base_leverage,
                "factors": leverage_calc.factors,
                "clamped": leverage_calc.clamped
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
                "base_pct": self.base_tp_pct,
                "tp_levels": tp_levels,
                "cross_exchange_multiplier": cross_exchange_adjustments.get("tp_multiplier", 1.0) if cross_exchange_adjustments else 1.0
            },
            "stop_loss": {
                "final_pct": final_sl,
                "base_pct": self.base_sl_pct,
                "cross_exchange_multiplier": cross_exchange_adjustments.get("sl_multiplier", 1.0) if cross_exchange_adjustments else 1.0
            },
            "inputs": {
                "confidence": signal.confidence,
                "atr_value": signal.atr_value,
                "pnl_trend": pnl_trend,
                "exch_divergence": exch_divergence,
                "funding_rate": funding_rate,
                "margin_util": margin_util
            }
        }
        
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
            f"[ExitBrain-v3.5] Plan Generated | "
            f"{signal.symbol} {signal.side.upper()} | "
            f"Leverage: {plan.leverage:.1f}x | "
            f"TP: {final_tp*100:.2f}% | "
            f"SL: {final_sl*100:.2f}%"
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
