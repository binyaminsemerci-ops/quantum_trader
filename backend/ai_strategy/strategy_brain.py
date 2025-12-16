"""Strategy Brain - Strategy performance analysis and recommendation logic.

This module analyzes strategy and model performance to provide
recommendations for optimal strategy selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    
    strategy_name: str
    
    # Performance metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float                      # 0-1 scale
    
    total_pnl: float                     # Total profit/loss (USD)
    avg_profit: float                    # Average winning trade (USD)
    avg_loss: float                      # Average losing trade (USD)
    profit_factor: float                 # Gross profit / Gross loss
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float                  # Max drawdown (0-1)
    
    # Recent performance (last N trades)
    recent_win_rate: float               # Win rate for recent trades
    recent_pnl: float                    # Recent PnL
    
    # Regime compatibility
    best_regime: Optional[str] = None    # Best performing regime
    regime_score: float = 0.0            # Current regime compatibility (0-1)
    
    # Status
    is_active: bool = True
    last_update: datetime = None


@dataclass
class ModelPerformance:
    """Performance metrics for a single ML model."""
    
    model_name: str
    model_type: str                      # "ensemble", "timeseries", etc.
    
    # Prediction accuracy
    total_predictions: int
    correct_predictions: int
    accuracy: float                      # 0-1 scale
    
    # Signal quality
    avg_confidence: float                # Average confidence score
    high_confidence_accuracy: float      # Accuracy when confidence > 0.7
    
    # Economic value
    predictions_traded: int
    trades_profitable: int
    economic_win_rate: float             # Win rate of traded predictions
    
    # Recency
    last_prediction: datetime
    last_retrain: datetime
    
    # Status
    is_enabled: bool = True


@dataclass
class StrategyRecommendation:
    """Strategy recommendation output."""
    
    # Primary recommendation
    primary_strategy: str
    primary_reason: str
    
    # Fallback strategies
    fallback_strategies: list[str]
    
    # Strategies to disable
    disabled_strategies: list[str]
    disable_reasons: dict[str, str]
    
    # Model recommendations
    recommended_models: list[str]
    model_weights: dict[str, float]      # Model name → weight
    
    # Meta-strategy recommendation
    meta_strategy_mode: str              # "trend", "mean_reversion", "breakout", "range"
    meta_strategy_confidence: float      # 0-1 scale
    
    # Performance summary
    overall_win_rate: float
    overall_sharpe: float
    best_performing_strategy: str
    worst_performing_strategy: str
    
    # Alerts
    alerts: list[str]
    
    # Metadata
    recommendation_timestamp: datetime
    confidence: float                    # Overall confidence (0-1)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for event publishing."""
        return {
            "primary_strategy": self.primary_strategy,
            "primary_reason": self.primary_reason,
            "fallback_strategies": self.fallback_strategies,
            "disabled_strategies": self.disabled_strategies,
            "disable_reasons": self.disable_reasons,
            "recommended_models": self.recommended_models,
            "model_weights": self.model_weights,
            "meta_strategy_mode": self.meta_strategy_mode,
            "meta_strategy_confidence": self.meta_strategy_confidence,
            "overall_win_rate": self.overall_win_rate,
            "overall_sharpe": self.overall_sharpe,
            "best_performing_strategy": self.best_performing_strategy,
            "worst_performing_strategy": self.worst_performing_strategy,
            "alerts": self.alerts,
            "recommendation_timestamp": self.recommendation_timestamp.isoformat(),
            "confidence": self.confidence,
        }


class StrategyBrain:
    """
    Core strategy analysis and recommendation engine.
    
    Responsibilities:
    - Analyze strategy performance across multiple dimensions
    - Evaluate model prediction quality
    - Recommend primary and fallback strategies
    - Identify underperforming strategies
    - Suggest meta-strategy adjustments
    - Provide strategy state for AI CEO
    """
    
    # Performance thresholds
    MIN_ACCEPTABLE_WIN_RATE = 0.45
    GOOD_WIN_RATE = 0.55
    EXCELLENT_WIN_RATE = 0.60
    
    MIN_ACCEPTABLE_SHARPE = 0.5
    GOOD_SHARPE = 1.5
    EXCELLENT_SHARPE = 2.5
    
    MIN_TRADES_FOR_EVALUATION = 20
    
    def __init__(self):
        """Initialize Strategy Brain."""
        self._recommendation_history: list[StrategyRecommendation] = []
        self._max_history = 1000
        
        logger.info("StrategyBrain initialized")
    
    def analyze_and_recommend(
        self,
        strategies: list[StrategyPerformance],
        models: list[ModelPerformance],
        current_regime: str,
        current_primary_strategy: Optional[str] = None,
    ) -> StrategyRecommendation:
        """
        Analyze strategy and model performance and generate recommendations.
        
        Args:
            strategies: List of strategy performance data
            models: List of model performance data
            current_regime: Current market regime
            current_primary_strategy: Currently active primary strategy
        
        Returns:
            StrategyRecommendation with comprehensive recommendations
        """
        logger.debug(
            f"StrategyBrain analyzing {len(strategies)} strategies, {len(models)} models"
        )
        
        # Filter strategies with sufficient data
        evaluated_strategies = [
            s for s in strategies
            if s.total_trades >= self.MIN_TRADES_FOR_EVALUATION
        ]
        
        if not evaluated_strategies:
            # No strategies with enough data - use conservative defaults
            return self._default_recommendation(current_regime)
        
        # Rank strategies by performance
        ranked_strategies = self._rank_strategies(evaluated_strategies, current_regime)
        
        # Select primary strategy
        primary_strategy = self._select_primary_strategy(
            ranked_strategies,
            current_primary_strategy,
        )
        
        # Select fallback strategies
        fallback_strategies = self._select_fallback_strategies(ranked_strategies, primary_strategy)
        
        # Identify strategies to disable
        disabled_strategies, disable_reasons = self._identify_disabled_strategies(evaluated_strategies)
        
        # Analyze models
        recommended_models, model_weights = self._analyze_models(models)
        
        # Determine meta-strategy mode
        meta_strategy_mode, meta_confidence = self._determine_meta_strategy(
            ranked_strategies,
            current_regime,
        )
        
        # Calculate overall metrics
        overall_win_rate = np.mean([s.win_rate for s in evaluated_strategies])
        overall_sharpe = np.mean([s.sharpe_ratio for s in evaluated_strategies])
        
        best_strategy = ranked_strategies[0].strategy_name if ranked_strategies else "unknown"
        worst_strategy = ranked_strategies[-1].strategy_name if ranked_strategies else "unknown"
        
        # Generate alerts
        alerts = self._generate_alerts(evaluated_strategies, models)
        
        # Calculate confidence
        confidence = self._calculate_confidence(evaluated_strategies, models)
        
        # Create recommendation
        recommendation = StrategyRecommendation(
            primary_strategy=primary_strategy,
            primary_reason=f"Best performer in {current_regime} regime",
            fallback_strategies=fallback_strategies,
            disabled_strategies=disabled_strategies,
            disable_reasons=disable_reasons,
            recommended_models=recommended_models,
            model_weights=model_weights,
            meta_strategy_mode=meta_strategy_mode,
            meta_strategy_confidence=meta_confidence,
            overall_win_rate=overall_win_rate,
            overall_sharpe=overall_sharpe,
            best_performing_strategy=best_strategy,
            worst_performing_strategy=worst_strategy,
            alerts=alerts,
            recommendation_timestamp=datetime.utcnow(),
            confidence=confidence,
        )
        
        # Store in history
        self._recommendation_history.append(recommendation)
        if len(self._recommendation_history) > self._max_history:
            self._recommendation_history.pop(0)
        
        logger.info(
            f"StrategyBrain recommendation: primary={primary_strategy}, "
            f"meta={meta_strategy_mode}, "
            f"win_rate={overall_win_rate:.2%}, "
            f"confidence={confidence:.2f}"
        )
        
        return recommendation
    
    def _rank_strategies(
        self,
        strategies: list[StrategyPerformance],
        current_regime: str,
    ) -> list[StrategyPerformance]:
        """
        Rank strategies by combined performance score.
        
        Score considers:
        - Win rate
        - Sharpe ratio
        - Profit factor
        - Regime compatibility
        - Recent performance
        """
        scored_strategies = []
        
        for strategy in strategies:
            score = 0.0
            
            # Win rate contribution (0-40 points)
            win_rate_score = strategy.win_rate * 40
            score += win_rate_score
            
            # Sharpe ratio contribution (0-30 points)
            sharpe_score = min(strategy.sharpe_ratio / 3.0, 1.0) * 30
            score += sharpe_score
            
            # Profit factor contribution (0-20 points)
            if strategy.profit_factor > 0:
                pf_score = min(strategy.profit_factor / 2.0, 1.0) * 20
                score += pf_score
            
            # Regime compatibility (0-10 points)
            if strategy.best_regime == current_regime:
                score += 10 * strategy.regime_score
            
            # Recent performance bonus (up to 10 points)
            if strategy.recent_win_rate > strategy.win_rate:
                recent_bonus = (strategy.recent_win_rate - strategy.win_rate) * 20
                score += min(recent_bonus, 10)
            
            scored_strategies.append((strategy, score))
        
        # Sort by score descending
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in scored_strategies]
    
    def _select_primary_strategy(
        self,
        ranked_strategies: list[StrategyPerformance],
        current_primary: Optional[str],
    ) -> str:
        """Select primary strategy, considering stability."""
        if not ranked_strategies:
            return "ensemble_conservative"
        
        best_strategy = ranked_strategies[0]
        
        # If current primary is still performing well, stick with it (stability)
        if current_primary:
            current_perf = next(
                (s for s in ranked_strategies if s.strategy_name == current_primary),
                None,
            )
            
            if current_perf and current_perf.win_rate >= self.MIN_ACCEPTABLE_WIN_RATE:
                # Current strategy is acceptable - only switch if new one is significantly better
                if best_strategy.win_rate > current_perf.win_rate * 1.1:  # 10% better
                    return best_strategy.strategy_name
                return current_primary
        
        return best_strategy.strategy_name
    
    def _select_fallback_strategies(
        self,
        ranked_strategies: list[StrategyPerformance],
        primary_strategy: str,
    ) -> list[str]:
        """Select 2-3 fallback strategies."""
        fallbacks = []
        
        for strategy in ranked_strategies:
            if strategy.strategy_name == primary_strategy:
                continue
            
            if strategy.win_rate >= self.MIN_ACCEPTABLE_WIN_RATE:
                fallbacks.append(strategy.strategy_name)
            
            if len(fallbacks) >= 3:
                break
        
        return fallbacks
    
    def _identify_disabled_strategies(
        self,
        strategies: list[StrategyPerformance],
    ) -> tuple[list[str], dict[str, str]]:
        """Identify strategies that should be disabled."""
        disabled = []
        reasons = {}
        
        for strategy in strategies:
            # Disable if consistently underperforming
            if strategy.win_rate < self.MIN_ACCEPTABLE_WIN_RATE:
                disabled.append(strategy.strategy_name)
                reasons[strategy.strategy_name] = f"Low win rate: {strategy.win_rate:.2%}"
            
            # Disable if Sharpe is too low
            elif strategy.sharpe_ratio < 0:
                disabled.append(strategy.strategy_name)
                reasons[strategy.strategy_name] = f"Negative Sharpe: {strategy.sharpe_ratio:.2f}"
            
            # Disable if profit factor < 1
            elif strategy.profit_factor < 1.0 and strategy.total_trades > 50:
                disabled.append(strategy.strategy_name)
                reasons[strategy.strategy_name] = f"Profit factor < 1: {strategy.profit_factor:.2f}"
        
        return disabled, reasons
    
    def _analyze_models(
        self,
        models: list[ModelPerformance],
    ) -> tuple[list[str], dict[str, float]]:
        """Analyze model performance and assign weights."""
        recommended = []
        weights = {}
        
        for model in models:
            if not model.is_enabled:
                continue
            
            # Recommend if accuracy is decent
            if model.accuracy >= 0.52:  # 52%+ accuracy
                recommended.append(model.model_name)
                
                # Calculate weight based on performance
                # Base weight = accuracy above 50%
                base_weight = model.accuracy - 0.5
                
                # Bonus for high confidence accuracy
                if model.high_confidence_accuracy > model.accuracy:
                    base_weight *= 1.2
                
                # Bonus for economic win rate
                if model.economic_win_rate > 0.55:
                    base_weight *= 1.1
                
                weights[model.model_name] = base_weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return recommended, weights
    
    def _determine_meta_strategy(
        self,
        ranked_strategies: list[StrategyPerformance],
        current_regime: str,
    ) -> tuple[str, float]:
        """Determine optimal meta-strategy mode based on regime and performance."""
        # Map regimes to meta-strategies
        regime_to_meta = {
            "BULLISH": "trend",
            "BEARISH": "trend",
            "SIDEWAYS": "range",
            "HIGH_VOLATILITY": "breakout",
            "LOW_VOLATILITY": "mean_reversion",
            "UNKNOWN": "ensemble",
        }
        
        base_meta = regime_to_meta.get(current_regime, "ensemble")
        
        # Check if trend strategies are performing well
        if ranked_strategies:
            best_strategy = ranked_strategies[0]
            
            if "trend" in best_strategy.strategy_name.lower():
                return "trend", 0.85
            elif "mean_reversion" in best_strategy.strategy_name.lower():
                return "mean_reversion", 0.85
            elif "breakout" in best_strategy.strategy_name.lower():
                return "breakout", 0.85
            elif "range" in best_strategy.strategy_name.lower():
                return "range", 0.85
        
        # Default to regime-based
        return base_meta, 0.70
    
    def _generate_alerts(
        self,
        strategies: list[StrategyPerformance],
        models: list[ModelPerformance],
    ) -> list[str]:
        """Generate strategy-related alerts."""
        alerts = []
        
        # Check for universally poor performance
        if strategies:
            avg_win_rate = np.mean([s.win_rate for s in strategies])
            if avg_win_rate < self.MIN_ACCEPTABLE_WIN_RATE:
                alerts.append(
                    f"Low overall win rate: {avg_win_rate:.2%}"
                )
        
        # Check for declining performance
        recent_strategies = [s for s in strategies if s.recent_win_rate < s.win_rate * 0.8]
        if len(recent_strategies) >= len(strategies) * 0.5:
            alerts.append("Multiple strategies showing declining performance")
        
        # Check model accuracy
        if models:
            avg_accuracy = np.mean([m.accuracy for m in models if m.is_enabled])
            if avg_accuracy < 0.52:
                alerts.append(f"Low model accuracy: {avg_accuracy:.2%}")
        
        # Check for stale models
        stale_models = [
            m for m in models
            if (datetime.utcnow() - m.last_retrain).days > 7
        ]
        if stale_models:
            alerts.append(f"{len(stale_models)} models need retraining")
        
        return alerts
    
    def _calculate_confidence(
        self,
        strategies: list[StrategyPerformance],
        models: list[ModelPerformance],
    ) -> float:
        """Calculate confidence in recommendation."""
        confidence_factors = []
        
        # Number of strategies with sufficient data
        if len(strategies) >= 3:
            confidence_factors.append(0.9)
        elif len(strategies) >= 2:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Performance consistency
        if strategies:
            win_rates = [s.win_rate for s in strategies]
            win_rate_std = np.std(win_rates)
            if win_rate_std < 0.05:  # Low variation = high confidence
                confidence_factors.append(0.9)
            elif win_rate_std < 0.10:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
        
        # Model agreement
        if models:
            enabled_models = [m for m in models if m.is_enabled]
            if len(enabled_models) >= 3:
                confidence_factors.append(0.9)
            elif len(enabled_models) >= 2:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _default_recommendation(self, current_regime: str) -> StrategyRecommendation:
        """Provide default recommendation when insufficient data."""
        return StrategyRecommendation(
            primary_strategy="ensemble_conservative",
            primary_reason="Insufficient data - using conservative default",
            fallback_strategies=["trend_following", "mean_reversion"],
            disabled_strategies=[],
            disable_reasons={},
            recommended_models=["xgboost", "lightgbm"],
            model_weights={"xgboost": 0.5, "lightgbm": 0.5},
            meta_strategy_mode="ensemble",
            meta_strategy_confidence=0.5,
            overall_win_rate=0.50,
            overall_sharpe=1.0,
            best_performing_strategy="unknown",
            worst_performing_strategy="unknown",
            alerts=["Insufficient strategy performance data"],
            recommendation_timestamp=datetime.utcnow(),
            confidence=0.5,
        )
    
    def get_recommendation_history(self, limit: int = 100) -> list[StrategyRecommendation]:
        """Get recent recommendation history."""
        return self._recommendation_history[-limit:]
