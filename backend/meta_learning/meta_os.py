"""Meta-Learning OS - Learns How to Learn.

Operates at meta-level to optimize:
- RL hyperparameters (learning rate, exploration rate, etc.)
- Strategy weighting in ensemble
- Model weighting across AI agents
- SESA evolution parameters
- Risk tolerance based on performance
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from backend.meta_learning.meta_policy import (
    MetaDecision,
    MetaDecisionType,
    MetaPolicy,
    MetaPolicyRules,
)

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for Meta-Learning OS."""
    
    # Update intervals
    meta_update_interval_minutes: float = 60.0
    
    # Learning rates for meta-parameters
    meta_learning_rate: float = 0.01
    
    # Performance tracking
    performance_window_hours: int = 24
    min_samples_for_update: int = 20
    
    # Adaptation flags
    adapt_rl_hyperparameters: bool = True
    adapt_strategy_weights: bool = True
    adapt_model_weights: bool = True
    adapt_risk_tolerance: bool = True


@dataclass
class MetaLearningState:
    """
    Current state of meta-learning system.
    """
    
    # RL Hyperparameters
    rl_learning_rate: float = 0.001
    rl_exploration_rate: float = 0.15
    rl_discount_factor: float = 0.95
    
    # Strategy Weights (strategy_id → weight)
    strategy_weights: dict[str, float] = field(default_factory=dict)
    
    # Model Weights (model_name → weight)
    model_weights: dict[str, float] = field(default_factory=dict)
    
    # Risk Parameters
    risk_multiplier: float = 1.0
    max_position_size_pct: float = 0.02
    
    # SESA Parameters
    sesa_mutation_rate: float = 0.3
    sesa_mutation_magnitude: float = 0.2
    
    # Performance tracking
    recent_sharpe: float = 0.0
    recent_win_rate: float = 0.0
    recent_profit_factor: float = 1.0
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rl_learning_rate": self.rl_learning_rate,
            "rl_exploration_rate": self.rl_exploration_rate,
            "rl_discount_factor": self.rl_discount_factor,
            "strategy_weights": self.strategy_weights,
            "model_weights": self.model_weights,
            "risk_multiplier": self.risk_multiplier,
            "max_position_size_pct": self.max_position_size_pct,
            "sesa_mutation_rate": self.sesa_mutation_rate,
            "sesa_mutation_magnitude": self.sesa_mutation_magnitude,
            "recent_sharpe": self.recent_sharpe,
            "recent_win_rate": self.recent_win_rate,
            "recent_profit_factor": self.recent_profit_factor,
            "last_updated": self.last_updated.isoformat(),
            "update_count": self.update_count,
        }


class MetaLearningOS:
    """
    Meta-Learning Operating System.
    
    Learns at a meta-level to optimize:
    1. RL hyperparameters → Better learning
    2. Strategy weights → Better ensemble
    3. Model weights → Better predictions
    4. Risk parameters → Better risk management
    5. SESA parameters → Better evolution
    
    Uses performance feedback to adjust parameters that control
    the learning process itself.
    
    Integration:
    - Memory Engine: Retrieve performance history
    - SESA Engine: Update evolution parameters
    - RL Agent: Update hyperparameters
    - Federation: Aggregate meta-learnings
    
    Usage:
        meta_os = MetaLearningOS(
            config=MetaLearningConfig(),
            meta_policy=MetaPolicy(),
        )
        
        # Run meta-update cycle
        await meta_os.run_meta_update(
            recent_trades=[...],
            recent_evaluations=[...],
        )
        
        # Get current state
        state = meta_os.get_state()
        print(f"RL learning rate: {state.rl_learning_rate}")
        print(f"Exploration rate: {state.rl_exploration_rate}")
    """
    
    def __init__(
        self,
        config: Optional[MetaLearningConfig] = None,
        meta_policy: Optional[MetaPolicy] = None,
        memory_engine: Optional[Any] = None,
        event_bus: Optional[Any] = None,
    ):
        """
        Initialize Meta-Learning OS.
        
        Args:
            config: Configuration
            meta_policy: Meta-policy for decisions
            memory_engine: Memory Engine for history
            event_bus: EventBus for publishing updates
        """
        self.config = config or MetaLearningConfig()
        self.meta_policy = meta_policy or MetaPolicy()
        self.memory_engine = memory_engine
        self.event_bus = event_bus
        
        # Current state
        self.state = MetaLearningState()
        
        # Performance history
        self.performance_history: list[dict[str, Any]] = []
        
        logger.info("MetaLearningOS initialized")
    
    def get_state(self) -> MetaLearningState:
        """Get current meta-learning state."""
        return self.state
    
    async def run_meta_update(
        self,
        recent_trades: list[dict[str, Any]],
        recent_evaluations: Optional[list[dict[str, Any]]] = None,
    ) -> MetaLearningState:
        """
        Run meta-update cycle.
        
        Analyzes recent performance and adjusts meta-parameters.
        
        Args:
            recent_trades: Recent trade results
            recent_evaluations: Recent strategy evaluations (from SESA)
        
        Returns:
            Updated MetaLearningState
        """
        logger.info("Running meta-update cycle")
        
        # Calculate current performance metrics
        performance = self._calculate_performance_metrics(recent_trades)
        
        self.state.recent_sharpe = performance["sharpe"]
        self.state.recent_win_rate = performance["win_rate"]
        self.state.recent_profit_factor = performance["profit_factor"]
        
        # Store in history
        self.performance_history.append({
            "timestamp": datetime.now(),
            "performance": performance,
        })
        
        # Trim history to window
        self._trim_history()
        
        # Check if sufficient samples
        if len(self.performance_history) < self.config.min_samples_for_update:
            logger.info(f"Insufficient samples: {len(self.performance_history)}/{self.config.min_samples_for_update}")
            return self.state
        
        # Adapt RL hyperparameters
        if self.config.adapt_rl_hyperparameters:
            await self._adapt_rl_hyperparameters(performance)
        
        # Adapt strategy weights
        if self.config.adapt_strategy_weights and recent_evaluations:
            await self._adapt_strategy_weights(recent_evaluations)
        
        # Adapt model weights
        if self.config.adapt_model_weights:
            await self._adapt_model_weights(recent_trades)
        
        # Adapt risk parameters
        if self.config.adapt_risk_tolerance:
            await self._adapt_risk_parameters(performance, recent_trades)
        
        # Update metadata
        self.state.last_updated = datetime.now()
        self.state.update_count += 1
        
        # Publish update event
        if self.event_bus:
            await self._publish_meta_update()
        
        logger.info(
            f"Meta-update complete (#{self.state.update_count}): "
            f"LR={self.state.rl_learning_rate:.4f}, "
            f"ER={self.state.rl_exploration_rate:.2f}, "
            f"Risk={self.state.risk_multiplier:.2f}"
        )
        
        return self.state
    
    def _calculate_performance_metrics(
        self,
        trades: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                "sharpe": 0.0,
                "win_rate": 0.0,
                "profit_factor": 1.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }
        
        pnls = np.array([t.get("pnl", 0) for t in trades])
        
        # Sharpe
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252) if np.std(pnls) > 0 else 0
        
        # Win rate
        wins = pnls[pnls > 0]
        win_rate = len(wins) / len(pnls)
        
        # Profit factor
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        losses = pnls[pnls < 0]
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-6
        profit_factor = total_wins / total_losses
        
        return {
            "sharpe": float(sharpe),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_pnl": float(np.sum(pnls)),
            "avg_pnl": float(np.mean(pnls)),
        }
    
    def _trim_history(self) -> None:
        """Trim performance history to configured window."""
        max_age = datetime.now().timestamp() - (self.config.performance_window_hours * 3600)
        
        self.performance_history = [
            h for h in self.performance_history
            if h["timestamp"].timestamp() > max_age
        ]
    
    async def _adapt_rl_hyperparameters(
        self,
        performance: dict[str, float],
    ) -> None:
        """
        Adapt RL hyperparameters based on performance.
        
        Rules:
        - High Sharpe → Decrease exploration (exploit more)
        - Low Sharpe → Increase exploration (explore more)
        - Adjust learning rate based on stability
        """
        current_sharpe = performance["sharpe"]
        
        # Exploration rate adjustment
        if current_sharpe > 1.5:
            # Excellent performance → exploit more
            target_exploration = 0.10
        elif current_sharpe > 1.0:
            # Good performance → balanced
            target_exploration = 0.15
        elif current_sharpe > 0.5:
            # Moderate performance → explore slightly more
            target_exploration = 0.20
        else:
            # Poor performance → explore aggressively
            target_exploration = 0.30
        
        # Smooth adjustment
        self.state.rl_exploration_rate += (
            (target_exploration - self.state.rl_exploration_rate)
            * self.config.meta_learning_rate
        )
        self.state.rl_exploration_rate = np.clip(self.state.rl_exploration_rate, 0.05, 0.40)
        
        # Learning rate adjustment based on performance variance
        if len(self.performance_history) >= 5:
            recent_sharpes = [h["performance"]["sharpe"] for h in self.performance_history[-5:]]
            sharpe_std = np.std(recent_sharpes)
            
            if sharpe_std > 0.5:
                # High variance → decrease learning rate for stability
                self.state.rl_learning_rate *= 0.95
            else:
                # Low variance → can increase learning rate
                self.state.rl_learning_rate *= 1.02
            
            self.state.rl_learning_rate = np.clip(self.state.rl_learning_rate, 0.0001, 0.01)
        
        logger.info(
            f"Adapted RL hyperparameters: "
            f"exploration={self.state.rl_exploration_rate:.3f}, "
            f"learning_rate={self.state.rl_learning_rate:.5f}"
        )
    
    async def _adapt_strategy_weights(
        self,
        recent_evaluations: list[dict[str, Any]],
    ) -> None:
        """
        Adapt strategy weights based on evaluation results.
        
        Strategies with higher composite scores get higher weights.
        """
        if not recent_evaluations:
            return
        
        # Extract scores
        strategy_scores: dict[str, float] = {}
        
        for eval_data in recent_evaluations:
            strategy_id = eval_data.get("strategy_id")
            score = eval_data.get("composite_score", 0)
            
            if strategy_id:
                strategy_scores[strategy_id] = score
        
        if not strategy_scores:
            return
        
        # Convert scores to weights (softmax-like)
        scores = np.array(list(strategy_scores.values()))
        exp_scores = np.exp(scores / 20.0)  # Temperature = 20
        weights = exp_scores / np.sum(exp_scores)
        
        # Update state
        new_weights = dict(zip(strategy_scores.keys(), weights))
        
        # Smooth update
        for strategy_id, new_weight in new_weights.items():
            old_weight = self.state.strategy_weights.get(strategy_id, 0)
            self.state.strategy_weights[strategy_id] = (
                old_weight * 0.7 + new_weight * 0.3  # 30% new, 70% old
            )
        
        logger.info(f"Adapted strategy weights for {len(new_weights)} strategies")
    
    async def _adapt_model_weights(
        self,
        recent_trades: list[dict[str, Any]],
    ) -> None:
        """
        Adapt model weights based on prediction accuracy.
        
        Models with better predictions get higher weights.
        """
        # TODO: Implement model-specific accuracy tracking
        # For now, placeholder
        
        # Example: Track which model's predictions led to profitable trades
        # model_performance = {}
        # for trade in recent_trades:
        #     model = trade.get("model_used")
        #     pnl = trade.get("pnl", 0)
        #     if model:
        #         model_performance[model] = model_performance.get(model, 0) + pnl
        
        logger.info("Model weight adaptation (placeholder)")
    
    async def _adapt_risk_parameters(
        self,
        performance: dict[str, float],
        recent_trades: list[dict[str, Any]],
    ) -> None:
        """
        Adapt risk parameters based on performance and recent trades.
        """
        # Count recent wins/losses
        recent_wins = sum(1 for t in recent_trades[-10:] if t.get("pnl", 0) > 0)
        recent_losses = sum(1 for t in recent_trades[-10:] if t.get("pnl", 0) < 0)
        
        # Use meta-policy to decide risk adjustment
        decision = self.meta_policy.adjust_risk_limits(
            recent_wins=recent_wins,
            recent_losses=recent_losses,
            current_risk_multiplier=self.state.risk_multiplier,
        )
        
        if decision.decision_type == MetaDecisionType.INCREASE_RISK_LIMITS:
            self.state.risk_multiplier = decision.parameters["new_risk_multiplier"]
            logger.info(f"Increased risk multiplier: {self.state.risk_multiplier:.2f}")
        elif decision.decision_type == MetaDecisionType.DECREASE_RISK_LIMITS:
            self.state.risk_multiplier = decision.parameters["new_risk_multiplier"]
            logger.info(f"Decreased risk multiplier: {self.state.risk_multiplier:.2f}")
        
        # Adjust max position size
        self.state.max_position_size_pct = 0.02 * self.state.risk_multiplier
    
    async def _publish_meta_update(self) -> None:
        """Publish meta-update event."""
        if not self.event_bus:
            return
        
        try:
            await self.event_bus.publish(
                "meta_learning_update",
                {
                    "update_count": self.state.update_count,
                    "rl_learning_rate": self.state.rl_learning_rate,
                    "rl_exploration_rate": self.state.rl_exploration_rate,
                    "risk_multiplier": self.state.risk_multiplier,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish meta_update: {e}")
