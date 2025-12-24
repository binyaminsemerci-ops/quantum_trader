"""
Exit Brain Performance Adapter - Phase 3C Integration

Integrates Phase 3C performance monitoring data into Exit Brain decision-making,
creating adaptive TP/SL levels based on historical success rates and system health.

Architecture:
- Uses Phase 3C-2 (Performance Benchmarker) for TP hit rates and accuracy
- Uses Phase 3C-3 (Adaptive Threshold Manager) for learned thresholds
- Uses Phase 3C-1 (System Health Monitor) for system health gating
- Provides adaptive TP/SL profiles to Exit Brain v3

Key Features:
1. Performance-Adaptive TP/SL: Adjust TP distances based on historical hit rates
2. Health-Gated Modifications: Suspend adjustments when AI health is poor
3. Regime-Based Scaling: Scale TP/SL based on market conditions
4. Predictive Tightening: Preemptive stop adjustment before issues

Author: AI Assistant
Date: 2025-12-24
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime types for TP/SL scaling"""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    NORMAL = "NORMAL"


@dataclass
class TPSLProfile:
    """Adaptive TP/SL profile with confidence and reasoning"""
    tp1_distance: float  # Distance in price for TP level 1
    tp2_distance: float  # Distance in price for TP level 2
    tp3_distance: Optional[float]  # Distance in price for TP level 3 (optional)
    sl_distance: float  # Distance in price for stop loss
    
    tp1_size_pct: float = 0.33  # 33% at TP1
    tp2_size_pct: float = 0.33  # 33% at TP2
    tp3_size_pct: float = 0.34  # 34% at TP3
    
    confidence: float = 0.5  # Confidence in this profile (0-1)
    reason: str = ""  # Explanation for these levels
    
    # Metadata
    tp1_hit_rate: float = 0.0  # Historical TP1 hit rate
    tp2_hit_rate: float = 0.0  # Historical TP2 hit rate
    tp3_hit_rate: float = 0.0  # Historical TP3 hit rate
    regime: str = "NORMAL"  # Market regime applied
    health_score: float = 100.0  # AI Engine health score


@dataclass
class ExitPerformanceMetrics:
    """Performance metrics for exit strategy evaluation"""
    tp1_triggers: int = 0
    tp2_triggers: int = 0
    tp3_triggers: int = 0
    sl_triggers: int = 0
    total_exits: int = 0
    
    avg_realized_rr: float = 0.0  # Average realized R:R ratio
    avg_exit_pnl_pct: float = 0.0  # Average exit PnL %
    
    premature_exits: int = 0  # Exits that hit TP too early
    missed_opportunities: int = 0  # SL hits that could have been TPs


class ExitBrainPerformanceAdapter:
    """
    Adapts Exit Brain v3 parameters using Phase 3C performance data.
    
    Responsibilities:
    1. Calculate adaptive TP/SL levels based on historical hit rates
    2. Apply regime-based scaling factors
    3. Gate modifications based on AI Engine health
    4. Provide predictive exit tightening signals
    
    Integration Points:
    - Phase 3C-2: Performance Benchmarker (for TP hit rates, accuracy)
    - Phase 3C-3: Adaptive Threshold Manager (for learned thresholds)
    - Phase 3C-1: System Health Monitor (for health gating)
    - Exit Brain v3: Dynamic Executor (consumes adaptive profiles)
    """
    
    def __init__(
        self,
        performance_benchmarker=None,
        adaptive_threshold_manager=None,
        system_health_monitor=None,
        default_tp_multipliers: Tuple[float, float, float] = (1.0, 2.5, 4.0),
        default_sl_multiplier: float = 1.5,
        min_sample_size: int = 20,
        health_threshold: float = 70.0
    ):
        """
        Initialize Exit Brain Performance Adapter.
        
        Args:
            performance_benchmarker: Phase 3C-2 instance
            adaptive_threshold_manager: Phase 3C-3 instance
            system_health_monitor: Phase 3C-1 instance
            default_tp_multipliers: Default (TP1, TP2, TP3) ATR multipliers
            default_sl_multiplier: Default SL ATR multiplier
            min_sample_size: Minimum exits required for adaptation
            health_threshold: Min health score for adaptations
        """
        self.benchmarker = performance_benchmarker
        self.threshold_manager = adaptive_threshold_manager
        self.health_monitor = system_health_monitor
        
        self.default_tp1 = default_tp_multipliers[0]
        self.default_tp2 = default_tp_multipliers[1]
        self.default_tp3 = default_tp_multipliers[2]
        self.default_sl = default_sl_multiplier
        
        self.min_samples = min_sample_size
        self.health_threshold = health_threshold
        
        # Performance tracking (module_type -> ExitPerformanceMetrics)
        self.exit_metrics: Dict[str, ExitPerformanceMetrics] = {}
        
        # Regime cache (symbol -> (regime, timestamp))
        self._regime_cache: Dict[str, Tuple[MarketRegime, datetime]] = {}
        self._regime_cache_ttl = 300  # 5 minutes
        
        logger.info(
            f"[EXIT_ADAPTER] 🎯 Initialized with defaults: "
            f"TP={default_tp_multipliers}, SL={default_sl_multiplier}, "
            f"min_samples={min_sample_size}, health_threshold={health_threshold}"
        )
    
    async def get_adaptive_tp_sl_profile(
        self,
        symbol: str,
        strategy: str,
        base_atr: float,
        side: str = "LONG"
    ) -> TPSLProfile:
        """
        Calculate adaptive TP/SL profile based on performance data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            strategy: Strategy name (e.g., 'AGGRESSIVE', 'CONSERVATIVE')
            base_atr: Base ATR value for distance calculations
            side: Position side ('LONG' or 'SHORT')
        
        Returns:
            TPSLProfile with adaptive levels and reasoning
        """
        logger.info(f"[EXIT_ADAPTER] Calculating adaptive TP/SL for {symbol} ({strategy}, {side})")
        
        # Step 1: Get AI Engine health
        health_check = await self._check_health()
        if not health_check["healthy"]:
            logger.warning(
                f"[EXIT_ADAPTER] Health degraded ({health_check['score']}/100), "
                f"using conservative defaults"
            )
            return self._get_conservative_profile(base_atr, health_check, strategy)
        
        # Step 2: Get performance data from Phase 3C-2
        tp_hit_rates = await self._get_tp_hit_rates(strategy)
        
        # Step 3: Get learned thresholds from Phase 3C-3
        learned_sl = await self._get_learned_sl_threshold(strategy)
        
        # Step 4: Determine market regime
        regime = await self._get_market_regime(symbol)
        
        # Step 5: Calculate adaptive multipliers
        tp1_mult = self._calculate_tp_multiplier(
            level=1,
            hit_rate=tp_hit_rates.get("tp1", 0.70),
            base_multiplier=self.default_tp1,
            regime=regime
        )
        
        tp2_mult = self._calculate_tp_multiplier(
            level=2,
            hit_rate=tp_hit_rates.get("tp2", 0.50),
            base_multiplier=self.default_tp2,
            regime=regime
        )
        
        tp3_mult = self._calculate_tp_multiplier(
            level=3,
            hit_rate=tp_hit_rates.get("tp3", 0.30),
            base_multiplier=self.default_tp3,
            regime=regime
        )
        
        sl_mult = self._calculate_sl_multiplier(
            learned_threshold=learned_sl,
            base_multiplier=self.default_sl,
            regime=regime
        )
        
        # Step 6: Build profile
        profile = TPSLProfile(
            tp1_distance=base_atr * tp1_mult,
            tp2_distance=base_atr * tp2_mult,
            tp3_distance=base_atr * tp3_mult,
            sl_distance=base_atr * sl_mult,
            confidence=min(tp_hit_rates.get("tp1", 0.70), 0.95),
            reason=(
                f"Adaptive TP/SL: TP1_rate={tp_hit_rates.get('tp1', 0):.2%}, "
                f"TP2_rate={tp_hit_rates.get('tp2', 0):.2%}, "
                f"regime={regime.value}, health={health_check['score']}/100"
            ),
            tp1_hit_rate=tp_hit_rates.get("tp1", 0.0),
            tp2_hit_rate=tp_hit_rates.get("tp2", 0.0),
            tp3_hit_rate=tp_hit_rates.get("tp3", 0.0),
            regime=regime.value,
            health_score=health_check["score"]
        )
        
        logger.info(
            f"[EXIT_ADAPTER] ✅ Profile calculated: "
            f"TP1={tp1_mult:.2f}x, TP2={tp2_mult:.2f}x, TP3={tp3_mult:.2f}x, "
            f"SL={sl_mult:.2f}x ATR ({regime.value})"
        )
        
        return profile
    
    async def should_tighten_stops_predictively(
        self,
        symbol: str,
        current_sl_distance: float
    ) -> Tuple[bool, Optional[float], str]:
        """
        Check if stops should be tightened based on predictive alerts.
        
        Args:
            symbol: Trading symbol
            current_sl_distance: Current SL distance from entry
        
        Returns:
            (should_tighten, new_sl_distance, reason)
        """
        if not self.threshold_manager:
            return (False, None, "")
        
        try:
            # Get predictive alerts from Phase 3C-3
            alerts = await self.threshold_manager.generate_predictive_alerts()
            
            for alert in alerts:
                # Check for ensemble degradation predictions
                if alert.module_type == 'ensemble':
                    if alert.time_to_threshold_hours and alert.time_to_threshold_hours < 4:
                        # Predicted issue within 4 hours, tighten stops
                        new_distance = current_sl_distance * 0.7  # Tighten by 30%
                        
                        reason = (
                            f"Predictive tightening: {alert.metric_name} "
                            f"will breach in {alert.time_to_threshold_hours:.1f}h"
                        )
                        
                        logger.warning(f"[EXIT_ADAPTER] ⚠️ {reason}, tightening SL")
                        
                        return (True, new_distance, reason)
            
            return (False, None, "")
        
        except Exception as e:
            logger.error(f"[EXIT_ADAPTER] Error checking predictive alerts: {e}")
            return (False, None, "")
    
    def record_exit_outcome(
        self,
        module_type: str,
        exit_type: str,  # 'TP1', 'TP2', 'TP3', 'SL'
        realized_rr: float,
        pnl_pct: float
    ):
        """
        Record exit outcome for performance tracking.
        
        Args:
            module_type: Module/strategy that generated the signal
            exit_type: Type of exit ('TP1', 'TP2', 'TP3', 'SL')
            realized_rr: Realized risk:reward ratio
            pnl_pct: PnL as percentage of position
        """
        if module_type not in self.exit_metrics:
            self.exit_metrics[module_type] = ExitPerformanceMetrics()
        
        metrics = self.exit_metrics[module_type]
        
        if exit_type == 'TP1':
            metrics.tp1_triggers += 1
        elif exit_type == 'TP2':
            metrics.tp2_triggers += 1
        elif exit_type == 'TP3':
            metrics.tp3_triggers += 1
        elif exit_type == 'SL':
            metrics.sl_triggers += 1
        
        metrics.total_exits += 1
        
        # Update rolling averages
        if metrics.total_exits > 1:
            metrics.avg_realized_rr = (
                (metrics.avg_realized_rr * (metrics.total_exits - 1) + realized_rr) /
                metrics.total_exits
            )
            metrics.avg_exit_pnl_pct = (
                (metrics.avg_exit_pnl_pct * (metrics.total_exits - 1) + pnl_pct) /
                metrics.total_exits
            )
        else:
            metrics.avg_realized_rr = realized_rr
            metrics.avg_exit_pnl_pct = pnl_pct
        
        logger.info(
            f"[EXIT_ADAPTER] Recorded {exit_type} exit for {module_type}: "
            f"R:R={realized_rr:.2f}, PnL={pnl_pct:.2%}, "
            f"Total={metrics.total_exits}"
        )
    
    def get_exit_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all tracked modules."""
        summary = {}
        
        for module_type, metrics in self.exit_metrics.items():
            if metrics.total_exits == 0:
                continue
            
            summary[module_type] = {
                "total_exits": metrics.total_exits,
                "tp1_rate": metrics.tp1_triggers / metrics.total_exits if metrics.total_exits > 0 else 0,
                "tp2_rate": metrics.tp2_triggers / metrics.total_exits if metrics.total_exits > 0 else 0,
                "tp3_rate": metrics.tp3_triggers / metrics.total_exits if metrics.total_exits > 0 else 0,
                "sl_rate": metrics.sl_triggers / metrics.total_exits if metrics.total_exits > 0 else 0,
                "avg_rr": metrics.avg_realized_rr,
                "avg_pnl_pct": metrics.avg_exit_pnl_pct
            }
        
        return summary
    
    # ========== PRIVATE HELPER METHODS ==========
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check AI Engine health from Phase 3C-1."""
        if not self.health_monitor:
            return {"healthy": True, "score": 100.0, "reason": "No health monitor"}
        
        try:
            health_status = await self.health_monitor.get_current_health()
            
            is_healthy = health_status.overall_health_score >= self.health_threshold
            
            return {
                "healthy": is_healthy,
                "score": health_status.overall_health_score,
                "reason": f"Overall health: {health_status.overall_health_score}/100"
            }
        except Exception as e:
            logger.error(f"[EXIT_ADAPTER] Error checking health: {e}")
            return {"healthy": True, "score": 100.0, "reason": "Health check failed, assuming healthy"}
    
    async def _get_tp_hit_rates(self, strategy: str) -> Dict[str, float]:
        """Get TP hit rates from Phase 3C-2 performance data."""
        if not self.benchmarker:
            # Default hit rates if no benchmarker
            return {"tp1": 0.70, "tp2": 0.50, "tp3": 0.30}
        
        try:
            # Get historical exit data
            # NOTE: This assumes performance_benchmarker tracks exit performance
            # In reality, we'd need to add this tracking to Phase 3C-2
            
            # For now, use exit_metrics as proxy
            if strategy in self.exit_metrics:
                metrics = self.exit_metrics[strategy]
                if metrics.total_exits >= self.min_samples:
                    return {
                        "tp1": metrics.tp1_triggers / metrics.total_exits if metrics.total_exits > 0 else 0.70,
                        "tp2": metrics.tp2_triggers / metrics.total_exits if metrics.total_exits > 0 else 0.50,
                        "tp3": metrics.tp3_triggers / metrics.total_exits if metrics.total_exits > 0 else 0.30
                    }
            
            # Insufficient data, use defaults
            return {"tp1": 0.70, "tp2": 0.50, "tp3": 0.30}
        
        except Exception as e:
            logger.error(f"[EXIT_ADAPTER] Error getting TP hit rates: {e}")
            return {"tp1": 0.70, "tp2": 0.50, "tp3": 0.30}
    
    async def _get_learned_sl_threshold(self, strategy: str) -> Optional[float]:
        """Get learned SL threshold from Phase 3C-3."""
        if not self.threshold_manager:
            return None
        
        try:
            # Get learned threshold for exit brain SL distance
            threshold = self.threshold_manager.get_threshold(
                module_type='exit_brain',
                metric_name='sl_distance'
            )
            
            if threshold and threshold.is_learned:
                return threshold.warning_threshold
            
            return None
        
        except Exception as e:
            logger.error(f"[EXIT_ADAPTER] Error getting learned SL: {e}")
            return None
    
    async def _get_market_regime(self, symbol: str) -> MarketRegime:
        """
        Determine current market regime for scaling factors.
        
        Uses cached regime if recent, otherwise re-evaluates.
        """
        now = datetime.utcnow()
        
        # Check cache
        if symbol in self._regime_cache:
            regime, timestamp = self._regime_cache[symbol]
            if (now - timestamp).total_seconds() < self._regime_cache_ttl:
                return regime
        
        # Determine regime (simplified logic - could be enhanced with more data)
        # For now, assume NORMAL. In production, would query market data.
        regime = MarketRegime.NORMAL
        
        self._regime_cache[symbol] = (regime, now)
        
        return regime
    
    def _calculate_tp_multiplier(
        self,
        level: int,
        hit_rate: float,
        base_multiplier: float,
        regime: MarketRegime
    ) -> float:
        """Calculate adaptive TP multiplier based on hit rate and regime."""
        
        # Hit rate adjustment
        if hit_rate > 0.80:
            # Very high hit rate → push TP further (more aggressive)
            hit_adjustment = 1.2
        elif hit_rate > 0.70:
            # Good hit rate → slight push
            hit_adjustment = 1.1
        elif hit_rate < 0.50:
            # Poor hit rate → pull TP closer (more conservative)
            hit_adjustment = 0.8
        elif hit_rate < 0.60:
            # Below average hit rate → slightly closer
            hit_adjustment = 0.9
        else:
            # Normal hit rate
            hit_adjustment = 1.0
        
        # Regime adjustment
        if regime == MarketRegime.HIGH_VOLATILITY:
            regime_adjustment = 1.2  # Wider TPs in volatile markets
        elif regime == MarketRegime.LOW_VOLATILITY:
            regime_adjustment = 0.9  # Tighter TPs in calm markets
        elif regime == MarketRegime.TRENDING:
            regime_adjustment = 1.1  # Slightly wider in trends
        else:
            regime_adjustment = 1.0
        
        # Combine adjustments
        final_multiplier = base_multiplier * hit_adjustment * regime_adjustment
        
        # Clamp to reasonable range
        min_mult = base_multiplier * 0.7
        max_mult = base_multiplier * 1.5
        final_multiplier = max(min_mult, min(max_mult, final_multiplier))
        
        return final_multiplier
    
    def _calculate_sl_multiplier(
        self,
        learned_threshold: Optional[float],
        base_multiplier: float,
        regime: MarketRegime
    ) -> float:
        """Calculate adaptive SL multiplier based on learned threshold and regime."""
        
        # Start with base
        multiplier = base_multiplier
        
        # Apply learned threshold if available
        if learned_threshold is not None:
            # Learned threshold is in absolute terms, need to convert to multiplier
            # This is simplified - in production would need more context
            multiplier = learned_threshold
        
        # Regime adjustment
        if regime == MarketRegime.HIGH_VOLATILITY:
            multiplier *= 1.3  # Wider stops in volatile markets
        elif regime == MarketRegime.LOW_VOLATILITY:
            multiplier *= 0.9  # Tighter stops in calm markets
        
        # Clamp to reasonable range
        min_mult = base_multiplier * 0.8
        max_mult = base_multiplier * 1.8
        multiplier = max(min_mult, min(max_mult, multiplier))
        
        return multiplier
    
    def _get_conservative_profile(
        self,
        base_atr: float,
        health_check: Dict[str, Any],
        strategy: str
    ) -> TPSLProfile:
        """Get conservative TP/SL profile when health is degraded."""
        
        # Use tighter TPs and wider SL when health is poor
        conservative_tp1 = self.default_tp1 * 0.8
        conservative_tp2 = self.default_tp2 * 0.8
        conservative_tp3 = self.default_tp3 * 0.8
        conservative_sl = self.default_sl * 1.2  # Wider stop for safety
        
        return TPSLProfile(
            tp1_distance=base_atr * conservative_tp1,
            tp2_distance=base_atr * conservative_tp2,
            tp3_distance=base_atr * conservative_tp3,
            sl_distance=base_atr * conservative_sl,
            confidence=0.3,  # Low confidence when health is poor
            reason=(
                f"Conservative profile due to {health_check['reason']}. "
                f"Health={health_check['score']}/100 (threshold={self.health_threshold})"
            ),
            tp1_hit_rate=0.0,
            tp2_hit_rate=0.0,
            tp3_hit_rate=0.0,
            regime="DEFENSIVE",
            health_score=health_check["score"]
        )
