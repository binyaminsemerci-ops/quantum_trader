"""
Phase 3B: Improved Strategy Selection
ML-based intelligent strategy selector that adapts to market conditions.

Combines:
- Phase 2D: Volatility Structure Engine data
- Phase 2B: Orderbook Imbalance data
- Phase 3A: Risk Mode Predictor
- Historical strategy performance
- Market regime classification

Author: Quantum Trader AI Team
Version: 3B.1.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class TradingStrategy(Enum):
    """Available trading strategies."""
    MOMENTUM_AGGRESSIVE = "momentum_aggressive"      # High momentum + trending
    MOMENTUM_CONSERVATIVE = "momentum_conservative"  # Moderate momentum
    MEAN_REVERSION = "mean_reversion"               # Oversold/overbought bounces
    BREAKOUT = "breakout"                           # Volume + volatility breakouts
    SCALPING = "scalping"                           # Quick in/out, tight spreads
    SWING_TRADING = "swing_trading"                 # Multi-hour/day positions
    VOLATILITY_TRADING = "volatility_trading"       # High volatility exploitation
    RANGE_TRADING = "range_trading"                 # Sideways consolidation
    TREND_FOLLOWING = "trend_following"             # Strong directional moves


@dataclass
class StrategyCharacteristics:
    """Characteristics of a trading strategy."""
    name: str
    optimal_volatility_range: Tuple[float, float]  # (min, max) volatility score
    optimal_orderflow_range: Tuple[float, float]   # (min, max) orderflow imbalance
    optimal_risk_modes: List[str]                   # Preferred risk modes
    optimal_regimes: List[str]                      # Preferred market regimes
    min_confidence: float                           # Minimum confidence to use
    timeframe_preference: str                       # "short", "medium", "long"
    position_holding_time: str                      # "minutes", "hours", "days"
    
    # Performance metrics
    win_rate: float = 0.0
    avg_profit: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0


@dataclass
class StrategySelection:
    """Strategy selection output."""
    primary_strategy: TradingStrategy
    secondary_strategy: Optional[TradingStrategy]
    confidence: float                           # 0-1 confidence in selection
    reasoning: str                              # Human-readable explanation
    strategy_weights: Dict[str, float]          # All strategy scores
    market_alignment_score: float               # How well conditions match strategy
    expected_performance_score: float           # Historical performance score
    regime_compatibility: float                 # Regime match score
    
    # Supporting data
    volatility_score: float
    orderflow_score: float
    risk_mode: str
    market_regime: str


class StrategyPerformanceTracker:
    """Tracks historical performance of each strategy."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.strategy_history: Dict[str, deque] = {
            strategy.value: deque(maxlen=max_history) 
            for strategy in TradingStrategy
        }
        self.strategy_metrics: Dict[str, StrategyCharacteristics] = {}
        logger.info(f"[PHASE 3B] StrategyPerformanceTracker initialized (max_history={max_history})")
    
    def record_trade_outcome(
        self,
        strategy: TradingStrategy,
        profit_pct: float,
        duration_minutes: float,
        market_conditions: Dict
    ) -> None:
        """Record a trade outcome for a strategy."""
        outcome = {
            "timestamp": datetime.now(timezone.utc),
            "profit_pct": profit_pct,
            "duration_minutes": duration_minutes,
            "market_conditions": market_conditions,
            "win": profit_pct > 0
        }
        
        self.strategy_history[strategy.value].append(outcome)
        self._update_strategy_metrics(strategy)
    
    def _update_strategy_metrics(self, strategy: TradingStrategy) -> None:
        """Update performance metrics for a strategy."""
        history = list(self.strategy_history[strategy.value])
        if not history:
            return
        
        profits = [t["profit_pct"] for t in history]
        wins = [t for t in history if t["win"]]
        
        win_rate = len(wins) / len(history) if history else 0.0
        avg_profit = np.mean(profits) if profits else 0.0
        std_profit = np.std(profits) if len(profits) > 1 else 0.0
        sharpe = (avg_profit / std_profit) if std_profit > 0 else 0.0
        
        # Calculate max drawdown
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        if strategy.value not in self.strategy_metrics:
            # Initialize with default characteristics
            self.strategy_metrics[strategy.value] = self._get_default_characteristics(strategy)
        
        # Update metrics
        metrics = self.strategy_metrics[strategy.value]
        metrics.win_rate = win_rate
        metrics.avg_profit = avg_profit
        metrics.sharpe_ratio = sharpe
        metrics.max_drawdown = max_drawdown
        metrics.total_trades = len(history)
    
    def _get_default_characteristics(self, strategy: TradingStrategy) -> StrategyCharacteristics:
        """Get default characteristics for a strategy."""
        characteristics = {
            TradingStrategy.MOMENTUM_AGGRESSIVE: StrategyCharacteristics(
                name="Momentum Aggressive",
                optimal_volatility_range=(0.5, 0.9),
                optimal_orderflow_range=(0.4, 1.0),
                optimal_risk_modes=["aggressive", "ultra_aggressive"],
                optimal_regimes=["bull_strong", "bull_weak"],
                min_confidence=0.65,
                timeframe_preference="short",
                position_holding_time="minutes"
            ),
            TradingStrategy.MOMENTUM_CONSERVATIVE: StrategyCharacteristics(
                name="Momentum Conservative",
                optimal_volatility_range=(0.3, 0.6),
                optimal_orderflow_range=(0.2, 0.8),
                optimal_risk_modes=["normal", "conservative"],
                optimal_regimes=["bull_weak", "sideways_wide"],
                min_confidence=0.55,
                timeframe_preference="medium",
                position_holding_time="hours"
            ),
            TradingStrategy.MEAN_REVERSION: StrategyCharacteristics(
                name="Mean Reversion",
                optimal_volatility_range=(0.2, 0.5),
                optimal_orderflow_range=(-0.8, -0.2),  # Oversold/overbought
                optimal_risk_modes=["conservative", "normal"],
                optimal_regimes=["sideways_tight", "choppy"],
                min_confidence=0.60,
                timeframe_preference="short",
                position_holding_time="minutes"
            ),
            TradingStrategy.BREAKOUT: StrategyCharacteristics(
                name="Breakout",
                optimal_volatility_range=(0.6, 1.0),
                optimal_orderflow_range=(0.5, 1.0),
                optimal_risk_modes=["aggressive", "ultra_aggressive"],
                optimal_regimes=["volatile", "bull_strong"],
                min_confidence=0.70,
                timeframe_preference="short",
                position_holding_time="minutes"
            ),
            TradingStrategy.SCALPING: StrategyCharacteristics(
                name="Scalping",
                optimal_volatility_range=(0.1, 0.4),
                optimal_orderflow_range=(-0.3, 0.3),
                optimal_risk_modes=["normal", "aggressive"],
                optimal_regimes=["sideways_tight", "sideways_wide"],
                min_confidence=0.50,
                timeframe_preference="short",
                position_holding_time="minutes"
            ),
            TradingStrategy.SWING_TRADING: StrategyCharacteristics(
                name="Swing Trading",
                optimal_volatility_range=(0.3, 0.7),
                optimal_orderflow_range=(0.2, 0.8),
                optimal_risk_modes=["normal", "conservative"],
                optimal_regimes=["bull_weak", "sideways_wide"],
                min_confidence=0.55,
                timeframe_preference="long",
                position_holding_time="days"
            ),
            TradingStrategy.VOLATILITY_TRADING: StrategyCharacteristics(
                name="Volatility Trading",
                optimal_volatility_range=(0.7, 1.0),
                optimal_orderflow_range=(0.0, 1.0),
                optimal_risk_modes=["aggressive", "ultra_aggressive"],
                optimal_regimes=["volatile", "choppy"],
                min_confidence=0.65,
                timeframe_preference="short",
                position_holding_time="minutes"
            ),
            TradingStrategy.RANGE_TRADING: StrategyCharacteristics(
                name="Range Trading",
                optimal_volatility_range=(0.1, 0.4),
                optimal_orderflow_range=(-0.5, 0.5),
                optimal_risk_modes=["conservative", "normal"],
                optimal_regimes=["sideways_tight", "sideways_wide"],
                min_confidence=0.55,
                timeframe_preference="medium",
                position_holding_time="hours"
            ),
            TradingStrategy.TREND_FOLLOWING: StrategyCharacteristics(
                name="Trend Following",
                optimal_volatility_range=(0.4, 0.8),
                optimal_orderflow_range=(0.3, 1.0),
                optimal_risk_modes=["normal", "aggressive"],
                optimal_regimes=["bull_strong", "bull_weak"],
                min_confidence=0.60,
                timeframe_preference="long",
                position_holding_time="days"
            ),
        }
        
        return characteristics[strategy]
    
    def get_strategy_score(
        self,
        strategy: TradingStrategy,
        volatility_score: float,
        orderflow_score: float,
        risk_mode: str,
        market_regime: str
    ) -> Tuple[float, str]:
        """
        Calculate fitness score for a strategy given current conditions.
        
        Returns:
            Tuple of (score, reasoning)
        """
        if strategy.value not in self.strategy_metrics:
            self.strategy_metrics[strategy.value] = self._get_default_characteristics(strategy)
        
        char = self.strategy_metrics[strategy.value]
        
        # 1. Volatility alignment (30%)
        vol_min, vol_max = char.optimal_volatility_range
        if vol_min <= volatility_score <= vol_max:
            vol_score = 1.0
        else:
            # Penalize deviation
            if volatility_score < vol_min:
                vol_score = max(0.0, 1.0 - (vol_min - volatility_score) * 2)
            else:
                vol_score = max(0.0, 1.0 - (volatility_score - vol_max) * 2)
        
        # 2. Orderflow alignment (25%)
        flow_min, flow_max = char.optimal_orderflow_range
        if flow_min <= orderflow_score <= flow_max:
            flow_score = 1.0
        else:
            if orderflow_score < flow_min:
                flow_score = max(0.0, 1.0 - abs(flow_min - orderflow_score) * 1.5)
            else:
                flow_score = max(0.0, 1.0 - abs(orderflow_score - flow_max) * 1.5)
        
        # 3. Risk mode compatibility (20%)
        risk_score = 1.0 if risk_mode in char.optimal_risk_modes else 0.3
        
        # 4. Regime compatibility (15%)
        regime_score = 1.0 if market_regime in char.optimal_regimes else 0.2
        
        # 5. Historical performance (10%)
        if char.total_trades > 20:
            # Use historical data
            perf_score = min(1.0, max(0.0, 
                0.4 * char.win_rate +
                0.3 * min(1.0, char.sharpe_ratio / 2.0) +
                0.3 * min(1.0, char.avg_profit * 10)
            ))
        else:
            perf_score = 0.5  # Neutral for new strategies
        
        # Weighted total
        total_score = (
            0.30 * vol_score +
            0.25 * flow_score +
            0.20 * risk_score +
            0.15 * regime_score +
            0.10 * perf_score
        )
        
        # Generate reasoning
        reasons = []
        if vol_score > 0.7:
            reasons.append(f"optimal volatility ({volatility_score:.2f})")
        if flow_score > 0.7:
            reasons.append(f"favorable orderflow ({orderflow_score:.2f})")
        if risk_score > 0.7:
            reasons.append(f"matching risk mode ({risk_mode})")
        if regime_score > 0.7:
            reasons.append(f"compatible regime ({market_regime})")
        if perf_score > 0.6 and char.total_trades > 20:
            reasons.append(f"strong history (WR={char.win_rate:.1%})")
        
        reasoning = ", ".join(reasons) if reasons else "moderate conditions"
        
        return total_score, reasoning
    
    def get_best_strategies(
        self,
        volatility_score: float,
        orderflow_score: float,
        risk_mode: str,
        market_regime: str,
        top_n: int = 3
    ) -> List[Tuple[TradingStrategy, float, str]]:
        """Get top N best strategies for current conditions."""
        scores = []
        
        for strategy in TradingStrategy:
            score, reasoning = self.get_strategy_score(
                strategy, volatility_score, orderflow_score, risk_mode, market_regime
            )
            scores.append((strategy, score, reasoning))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_n]


class StrategySelector:
    """
    Phase 3B: Improved Strategy Selection
    
    Intelligently selects optimal trading strategy based on:
    - Phase 2D volatility analysis
    - Phase 2B orderbook data
    - Phase 3A risk mode
    - Historical strategy performance
    - Market regime
    """
    
    def __init__(
        self,
        volatility_engine=None,
        orderbook_module=None,
        risk_mode_predictor=None,
        confidence_threshold: float = 0.60
    ):
        """
        Initialize Strategy Selector.
        
        Args:
            volatility_engine: Phase 2D VolatilityStructureEngine
            orderbook_module: Phase 2B OrderbookImbalanceModule
            risk_mode_predictor: Phase 3A RiskModePredictor
            confidence_threshold: Minimum confidence to use strategy
        """
        self.volatility_engine = volatility_engine
        self.orderbook_module = orderbook_module
        self.risk_mode_predictor = risk_mode_predictor
        self.confidence_threshold = confidence_threshold
        
        self.performance_tracker = StrategyPerformanceTracker()
        
        logger.info(f"[PHASE 3B] StrategySelector initialized (confidence_threshold={confidence_threshold})")
    
    def select_strategy(
        self,
        symbol: str,
        current_price: float,
        ensemble_confidence: float,
        market_conditions: Optional[Dict] = None
    ) -> StrategySelection:
        """
        Select optimal trading strategy for current conditions.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            ensemble_confidence: Confidence from ML ensemble
            market_conditions: Optional external market data
            
        Returns:
            StrategySelection with recommended strategy
        """
        try:
            # 1. Get Phase 2D volatility analysis
            volatility_score = 0.5
            volatility_regime = "normal"
            if self.volatility_engine:
                vol_analysis = self.volatility_engine.get_complete_volatility_analysis(symbol)
                if vol_analysis:
                    volatility_score = vol_analysis.combined_volatility_score
                    volatility_regime = vol_analysis.volatility_regime.lower()
            
            # 2. Get Phase 2B orderflow analysis
            orderflow_score = 0.0
            if self.orderbook_module:
                ob_metrics = self.orderbook_module.get_metrics(symbol)
                if ob_metrics:
                    orderflow_score = ob_metrics.orderflow_imbalance
            
            # 3. Get Phase 3A risk mode
            risk_mode = "normal"
            market_regime = "sideways_wide"
            if self.risk_mode_predictor:
                risk_signal = self.risk_mode_predictor.predict_risk_mode(
                    symbol=symbol,
                    current_price=current_price,
                    market_conditions=market_conditions or {}
                )
                risk_mode = risk_signal.mode.value
                market_regime = risk_signal.regime.value
            
            # 4. Get best strategies from performance tracker
            top_strategies = self.performance_tracker.get_best_strategies(
                volatility_score=volatility_score,
                orderflow_score=orderflow_score,
                risk_mode=risk_mode,
                market_regime=market_regime,
                top_n=3
            )
            
            primary_strategy, primary_score, primary_reasoning = top_strategies[0]
            secondary_strategy, secondary_score, _ = top_strategies[1] if len(top_strategies) > 1 else (None, 0.0, "")
            
            # 5. Calculate confidence
            # Combine strategy score with ensemble confidence
            strategy_confidence = primary_score * 0.6 + ensemble_confidence * 0.4
            
            # 6. Calculate alignment scores
            market_alignment = primary_score
            
            # Historical performance score
            char = self.performance_tracker.strategy_metrics.get(primary_strategy.value)
            if char and char.total_trades > 20:
                expected_performance = min(1.0, 
                    0.5 * char.win_rate + 
                    0.3 * min(1.0, char.sharpe_ratio / 2.0) +
                    0.2 * min(1.0, char.avg_profit * 10)
                )
            else:
                expected_performance = 0.5
            
            # Regime compatibility
            char = self.performance_tracker.strategy_metrics.get(primary_strategy.value)
            regime_compat = 1.0 if char and market_regime in char.optimal_regimes else 0.3
            
            # 7. Build strategy weights dict
            strategy_weights = {
                strategy.value: score 
                for strategy, score, _ in top_strategies
            }
            
            # 8. Generate reasoning
            reasoning_parts = [
                f"{primary_strategy.value}",
                primary_reasoning,
                f"ensemble_conf={ensemble_confidence:.2f}",
                f"strategy_score={primary_score:.2f}"
            ]
            reasoning = " | ".join(reasoning_parts)
            
            selection = StrategySelection(
                primary_strategy=primary_strategy,
                secondary_strategy=secondary_strategy if secondary_score > 0.5 else None,
                confidence=strategy_confidence,
                reasoning=reasoning,
                strategy_weights=strategy_weights,
                market_alignment_score=market_alignment,
                expected_performance_score=expected_performance,
                regime_compatibility=regime_compat,
                volatility_score=volatility_score,
                orderflow_score=orderflow_score,
                risk_mode=risk_mode,
                market_regime=market_regime
            )
            
            logger.info(
                f"[PHASE 3B] {symbol} Strategy: {primary_strategy.value} "
                f"(conf={strategy_confidence:.1%}, align={market_alignment:.2f})"
            )
            
            return selection
            
        except Exception as e:
            logger.error(f"[PHASE 3B] Strategy selection failed for {symbol}: {e}", exc_info=True)
            
            # Return safe default
            return StrategySelection(
                primary_strategy=TradingStrategy.MOMENTUM_CONSERVATIVE,
                secondary_strategy=None,
                confidence=0.5,
                reasoning=f"fallback: {str(e)}",
                strategy_weights={},
                market_alignment_score=0.5,
                expected_performance_score=0.5,
                regime_compatibility=0.5,
                volatility_score=0.5,
                orderflow_score=0.0,
                risk_mode="normal",
                market_regime="sideways_wide"
            )
    
    def record_trade_result(
        self,
        strategy: TradingStrategy,
        profit_pct: float,
        duration_minutes: float,
        market_conditions: Dict
    ) -> None:
        """Record a completed trade for strategy performance tracking."""
        self.performance_tracker.record_trade_outcome(
            strategy=strategy,
            profit_pct=profit_pct,
            duration_minutes=duration_minutes,
            market_conditions=market_conditions
        )
        logger.info(f"[PHASE 3B] Recorded trade: {strategy.value} profit={profit_pct:.2%}")
    
    def get_strategy_statistics(self) -> Dict[str, Dict]:
        """Get performance statistics for all strategies."""
        stats = {}
        
        for strategy in TradingStrategy:
            char = self.performance_tracker.strategy_metrics.get(strategy.value)
            if char:
                stats[strategy.value] = {
                    "win_rate": char.win_rate,
                    "avg_profit": char.avg_profit,
                    "sharpe_ratio": char.sharpe_ratio,
                    "max_drawdown": char.max_drawdown,
                    "total_trades": char.total_trades,
                    "timeframe": char.timeframe_preference,
                    "holding_time": char.position_holding_time
                }
        
        return stats
