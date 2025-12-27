"""
Meta-Strategy Selector with Reinforcement Learning

AI-powered strategy selection that learns which strategies work best
for different market regimes and symbols over time.

Uses contextual multi-armed bandit approach for robust, adaptive learning.

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import random
import numpy as np

from backend.services.ai.strategy_profiles import (
    StrategyID,
    StrategyProfile,
    get_strategy_profile,
    get_strategies_for_regime,
    get_strategies_for_symbol_tier,
    get_default_strategy,
)
from backend.services.ai.regime_detector import MarketRegime, MarketContext

logger = logging.getLogger(__name__)


@dataclass
class StrategyDecision:
    """
    Result of strategy selection.
    
    Attributes:
        strategy_id: Selected strategy ID
        strategy_profile: Complete strategy profile
        regime: Detected market regime
        is_exploration: Whether this was exploration (vs exploitation)
        confidence: Selection confidence (0-1)
        reasoning: Human-readable explanation
        q_values: Q-values for all candidate strategies
        timestamp: Decision timestamp
    """
    
    strategy_id: StrategyID
    strategy_profile: StrategyProfile
    regime: MarketRegime
    is_exploration: bool
    confidence: float
    reasoning: str
    q_values: Dict[str, float]
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "strategy_id": self.strategy_id.value,
            "strategy_name": self.strategy_profile.name,
            "regime": self.regime.value,
            "is_exploration": self.is_exploration,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "q_values": self.q_values,
            "timestamp": self.timestamp,
        }


@dataclass
class QStats:
    """
    Q-learning statistics for a (symbol, regime, strategy) tuple.
    
    Tracks reward history using exponential moving average for
    adaptive learning in non-stationary environments.
    """
    
    count: int = 0                    # Number of updates
    sum_reward: float = 0.0           # Sum of rewards (for simple avg)
    avg_reward: float = 0.0           # Current average reward
    ema_reward: float = 0.0           # Exponential moving average
    last_reward: float = 0.0          # Most recent reward
    last_update: Optional[str] = None  # Timestamp of last update
    
    # Performance tracking
    win_count: int = 0
    loss_count: int = 0
    total_r: float = 0.0              # Total R accumulated
    
    def update(self, reward: float, alpha: float = 0.2):
        """
        Update Q-statistics with new reward.
        
        Args:
            reward: New reward value (typically in R units)
            alpha: EMA smoothing factor (0-1, higher = more weight to recent)
        """
        self.count += 1
        self.sum_reward += reward
        self.last_reward = reward
        self.last_update = datetime.now(timezone.utc).isoformat()
        
        # Update simple average
        self.avg_reward = self.sum_reward / self.count
        
        # Update EMA (more weight to recent rewards)
        if self.count == 1:
            self.ema_reward = reward
        else:
            self.ema_reward = (1 - alpha) * self.ema_reward + alpha * reward
        
        # Track wins/losses
        if reward > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        self.total_r += reward
    
    def get_value(self, use_ema: bool = True) -> float:
        """
        Get current Q-value estimate.
        
        Args:
            use_ema: If True, use EMA (adapts faster). If False, use simple avg.
            
        Returns:
            Current Q-value estimate
        """
        if self.count == 0:
            return 0.0
        
        return self.ema_reward if use_ema else self.avg_reward
    
    def get_win_rate(self) -> float:
        """Get win rate (0-1)."""
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> QStats:
        """Create from dictionary."""
        return cls(**data)


class MetaStrategySelector:
    """
    AI-powered strategy selector using contextual multi-armed bandit.
    
    For each (symbol, regime) context, maintains Q-values for available strategies.
    Uses epsilon-greedy exploration/exploitation with adaptive learning.
    
    Key Features:
    - Learns which strategies perform best in different market conditions
    - Adapts to changing market dynamics via EMA
    - Explores new strategies to discover better approaches
    - Persists learning to survive restarts
    """
    
    def __init__(
        self,
        epsilon: float = 0.10,
        alpha: float = 0.20,
        use_ema: bool = True,
        min_confidence_for_exploit: float = 0.40,
        state_file: Optional[Path] = None,
        auto_save: bool = True,
    ):
        """
        Initialize Meta-Strategy Selector.
        
        Args:
            epsilon: Exploration probability (0-1). E.g., 0.10 = 10% random exploration
            alpha: EMA smoothing factor (0-1). Higher = more weight to recent rewards
            use_ema: Use EMA vs simple average for Q-values
            min_confidence_for_exploit: Minimum confidence to trust learned Q-values
            state_file: Path to state persistence file (JSON)
            auto_save: Automatically save state after updates
            
        Example:
            >>> selector = MetaStrategySelector(epsilon=0.10, alpha=0.20)
            >>> decision = selector.choose_strategy(
            ...     symbol="BTCUSDT",
            ...     regime=MarketRegime.TREND_UP,
            ...     context=market_context
            ... )
            >>> print(decision.strategy_id, decision.confidence)
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_ema = use_ema
        self.min_confidence_for_exploit = min_confidence_for_exploit
        self.auto_save = auto_save
        
        # Q-table: {(symbol, regime, strategy_id): QStats}
        self.q_table: Dict[Tuple[str, str, str], QStats] = {}
        
        # State persistence
        self.state_file = state_file or Path("data/meta_strategy_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing state
        self.load_state()
        
        # Metrics
        self.total_decisions = 0
        self.total_explorations = 0
        self.total_exploitations = 0
        self.total_updates = 0
        
        logger.info(
            f"[OK] MetaStrategySelector initialized: "
            f"epsilon={epsilon:.2%}, alpha={alpha:.2%}, "
            f"state_file={self.state_file}, "
            f"Q-entries={len(self.q_table)}"
        )
    
    def choose_strategy(
        self,
        symbol: str,
        regime: MarketRegime,
        context: MarketContext,
        force_strategy: Optional[StrategyID] = None,
    ) -> StrategyDecision:
        """
        Choose optimal strategy for given context.
        
        Uses epsilon-greedy: with probability epsilon, explore random strategy;
        otherwise, exploit strategy with highest Q-value.
        
        Args:
            symbol: Trading symbol
            regime: Detected market regime
            context: Full market context
            force_strategy: Force specific strategy (bypass RL)
            
        Returns:
            StrategyDecision with selected strategy and metadata
            
        Example:
            >>> selector = MetaStrategySelector()
            >>> decision = selector.choose_strategy(
            ...     symbol="BTCUSDT",
            ...     regime=MarketRegime.TREND_UP,
            ...     context=market_context
            ... )
            >>> print(f"Selected: {decision.strategy_profile.name}")
            >>> print(f"Exploration: {decision.is_exploration}")
        """
        self.total_decisions += 1
        
        # Get candidate strategies for this regime
        candidates = self._get_candidate_strategies(regime, context)
        
        if not candidates:
            logger.warning(f"No suitable strategies for {symbol} in {regime.value}")
            # Fallback to default
            default = get_default_strategy()
            return StrategyDecision(
                strategy_id=default.strategy_id,
                strategy_profile=default,
                regime=regime,
                is_exploration=False,
                confidence=0.50,
                reasoning="No suitable strategies found - using default",
                q_values={},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        
        # Force strategy if requested
        if force_strategy is not None:
            try:
                profile = get_strategy_profile(force_strategy)
                return StrategyDecision(
                    strategy_id=force_strategy,
                    strategy_profile=profile,
                    regime=regime,
                    is_exploration=False,
                    confidence=1.0,
                    reasoning=f"Forced strategy: {force_strategy.value}",
                    q_values={},
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            except ValueError:
                logger.warning(f"Invalid forced strategy: {force_strategy}")
        
        # Get Q-values for all candidates
        q_values = self._get_q_values(symbol, regime, candidates)
        
        # Epsilon-greedy selection
        is_exploration = random.random() < self.epsilon
        
        if is_exploration:
            # EXPLORE: random strategy
            self.total_explorations += 1
            selected_id = random.choice([c.strategy_id for c in candidates])
            confidence = 0.5
            reasoning = f"Exploration (epsilon={self.epsilon:.2%}): random selection"
        else:
            # EXPLOIT: best Q-value
            self.total_exploitations += 1
            
            # Check if we have enough data to trust Q-values
            max_q_value = max(q_values.values()) if q_values else 0.0
            total_samples = sum(
                self._get_q_stats(symbol, regime, sid.value).count
                for sid in [c.strategy_id for c in candidates]
            )
            
            if total_samples < 5 or max_q_value < self.min_confidence_for_exploit:
                # Not enough data - use heuristic
                selected_id = self._select_by_heuristic(candidates, context)
                confidence = 0.60
                reasoning = f"Heuristic selection (insufficient data: {total_samples} samples)"
            else:
                # Use learned Q-values
                selected_id = max(q_values, key=q_values.get)
                confidence = min(0.95, 0.60 + (max_q_value - self.min_confidence_for_exploit) * 0.5)
                reasoning = f"Exploitation: highest Q-value={q_values[selected_id]:.3f}"
        
        # Get selected strategy profile
        selected_profile = get_strategy_profile(selected_id)
        
        return StrategyDecision(
            strategy_id=selected_id,
            strategy_profile=selected_profile,
            regime=regime,
            is_exploration=is_exploration,
            confidence=confidence,
            reasoning=reasoning,
            q_values=q_values,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    def update_reward(
        self,
        symbol: str,
        regime: MarketRegime,
        strategy_id: StrategyID,
        reward: float,
        meta_info: Optional[Dict] = None,
    ):
        """
        Update Q-value with observed reward.
        
        Call this when a trade closes to update learning.
        
        Args:
            symbol: Trading symbol
            regime: Market regime during trade
            strategy_id: Strategy that was used
            reward: Observed reward (typically in R units)
            meta_info: Optional metadata (pnl, duration, etc.)
            
        Example:
            >>> selector.update_reward(
            ...     symbol="BTCUSDT",
            ...     regime=MarketRegime.TREND_UP,
            ...     strategy_id=StrategyID.ULTRA_AGGRESSIVE,
            ...     reward=3.5,  # +3.5R trade
            ...     meta_info={"pnl": 85.0, "duration_hours": 4.2}
            ... )
        """
        self.total_updates += 1
        
        key = (symbol, regime.value, strategy_id.value)
        
        # Get or create Q-stats
        if key not in self.q_table:
            self.q_table[key] = QStats()
        
        # Update stats
        self.q_table[key].update(reward, alpha=self.alpha)
        
        # Log update
        stats = self.q_table[key]
        logger.info(
            f"[RL UPDATE] {symbol} {regime.value} {strategy_id.value}: "
            f"R={reward:+.2f}, EMA={stats.ema_reward:.2f}, "
            f"count={stats.count}, WR={stats.get_win_rate():.1%}"
        )
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_state()
    
    def _get_candidate_strategies(
        self,
        regime: MarketRegime,
        context: MarketContext,
    ) -> List[StrategyProfile]:
        """Get list of suitable strategies for regime and context."""
        
        # Get strategies suitable for regime
        candidates = get_strategies_for_regime(regime.value)
        
        # Further filter by symbol tier if available
        if hasattr(context, 'symbol_tier') and context.symbol_tier:
            candidates = [
                s for s in candidates
                if s.is_suitable_for_symbol_tier(context.symbol_tier)
            ]
        
        # Filter by liquidity if poor
        if context.spread_bps > 10.0 or context.depth_5bps < 100_000:
            # Exclude aggressive strategies for illiquid symbols
            candidates = [
                s for s in candidates
                if s.aggressiveness.value not in ["very_high", "high"]
            ]
        
        return candidates
    
    def _get_q_values(
        self,
        symbol: str,
        regime: MarketRegime,
        candidates: List[StrategyProfile],
    ) -> Dict[StrategyID, float]:
        """Get Q-values for all candidate strategies."""
        q_values = {}
        
        for profile in candidates:
            stats = self._get_q_stats(symbol, regime, profile.strategy_id.value)
            q_values[profile.strategy_id] = stats.get_value(use_ema=self.use_ema)
        
        return q_values
    
    def _get_q_stats(self, symbol: str, regime: MarketRegime, strategy_id: str) -> QStats:
        """Get Q-stats for specific (symbol, regime, strategy) tuple."""
        key = (symbol, regime.value, strategy_id)
        if key not in self.q_table:
            self.q_table[key] = QStats()
        return self.q_table[key]
    
    def _select_by_heuristic(
        self,
        candidates: List[StrategyProfile],
        context: MarketContext,
    ) -> StrategyID:
        """
        Select strategy using heuristic rules when insufficient RL data.
        
        Uses domain knowledge to make reasonable defaults.
        """
        # Prefer strategies with higher expected R:R for the context
        scored = []
        
        for profile in candidates:
            score = profile.expected_risk_reward
            
            # Boost aggressive strategies in high conviction scenarios
            if context.recent_win_rate and context.recent_win_rate > 0.55:
                if profile.aggressiveness.value in ["high", "very_high"]:
                    score *= 1.2
            
            # Penalize aggressive strategies in low conviction
            if context.recent_win_rate and context.recent_win_rate < 0.45:
                if profile.aggressiveness.value in ["high", "very_high"]:
                    score *= 0.7
            
            # Prefer lower risk for illiquid markets
            if context.spread_bps > 5.0:
                if profile.aggressiveness.value in ["very_high"]:
                    score *= 0.8
            
            scored.append((profile.strategy_id, score))
        
        # Return highest scoring strategy
        return max(scored, key=lambda x: x[1])[0]
    
    def get_performance_summary(self, symbol: Optional[str] = None) -> Dict:
        """
        Get performance summary of learned strategies.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            Dictionary with performance metrics
        """
        if symbol:
            entries = {k: v for k, v in self.q_table.items() if k[0] == symbol}
        else:
            entries = self.q_table
        
        if not entries:
            return {"message": "No data yet"}
        
        summary = {
            "total_entries": len(entries),
            "total_decisions": self.total_decisions,
            "total_updates": self.total_updates,
            "exploration_rate": self.total_explorations / max(1, self.total_decisions),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "best_strategies": [],
        }
        
        # Find best performing strategies
        for (sym, reg, strat), stats in sorted(
            entries.items(),
            key=lambda x: x[1].ema_reward,
            reverse=True
        )[:10]:
            summary["best_strategies"].append({
                "symbol": sym,
                "regime": reg,
                "strategy": strat,
                "ema_reward": stats.ema_reward,
                "count": stats.count,
                "win_rate": stats.get_win_rate(),
                "total_r": stats.total_r,
            })
        
        return summary
    
    def save_state(self):
        """Save Q-table state to disk."""
        try:
            state = {
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epsilon": self.epsilon,
                    "alpha": self.alpha,
                    "total_decisions": self.total_decisions,
                    "total_updates": self.total_updates,
                },
                "q_table": {
                    f"{sym}|{reg}|{strat}": stats.to_dict()
                    for (sym, reg, strat), stats in self.q_table.items()
                },
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"[OK] Saved Q-table state: {len(self.q_table)} entries to {self.state_file}")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load Q-table state from disk."""
        if not self.state_file.exists():
            logger.info("[INFO] No existing state file found - starting fresh")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            metadata = state.get("metadata", {})
            q_table_data = state.get("q_table", {})
            
            # Restore Q-table
            for key_str, stats_dict in q_table_data.items():
                sym, reg, strat = key_str.split('|')
                self.q_table[(sym, reg, strat)] = QStats.from_dict(stats_dict)
            
            # Restore metrics
            self.total_decisions = metadata.get("total_decisions", 0)
            self.total_updates = metadata.get("total_updates", 0)
            
            logger.info(
                f"[OK] Loaded Q-table state: {len(self.q_table)} entries from {self.state_file}, "
                f"{self.total_updates} updates"
            )
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    def reset_learning(self):
        """Reset all learned Q-values (start fresh)."""
        self.q_table.clear()
        self.total_decisions = 0
        self.total_explorations = 0
        self.total_exploitations = 0
        self.total_updates = 0
        self.save_state()
        logger.info("[RESET] Cleared all Q-learning state")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_global_selector: Optional[MetaStrategySelector] = None


def get_meta_strategy_selector(
    epsilon: float = 0.10,
    alpha: float = 0.20,
    state_file: Optional[Path] = None,
) -> MetaStrategySelector:
    """
    Get or create global Meta-Strategy Selector instance.
    
    Args:
        epsilon: Exploration rate
        alpha: EMA smoothing factor
        state_file: State persistence file
        
    Returns:
        MetaStrategySelector instance
    """
    global _global_selector
    
    if _global_selector is None:
        _global_selector = MetaStrategySelector(
            epsilon=epsilon,
            alpha=alpha,
            state_file=state_file,
        )
    
    return _global_selector


if __name__ == "__main__":
    # Demo
    print("🧠 META-STRATEGY SELECTOR DEMO\n")
    
    selector = MetaStrategySelector(epsilon=0.15, alpha=0.25)
    
    # Simulate some trades
    from backend.services.ai.regime_detector import MarketContext
    
    context = MarketContext(
        symbol="BTCUSDT",
        atr_pct=0.03,
        trend_strength=0.7,
        adx=40.0,
        volume_24h=50_000_000,
        depth_5bps=500_000,
        spread_bps=2.5,
    )
    
    print("Making 5 strategy decisions:\n")
    
    for i in range(5):
        decision = selector.choose_strategy(
            symbol="BTCUSDT",
            regime=MarketRegime.TREND_UP,
            context=context,
        )
        
        print(f"Decision {i+1}:")
        print(f"  Strategy: {decision.strategy_profile.name}")
        print(f"  Exploration: {decision.is_exploration}")
        print(f"  Confidence: {decision.confidence:.1%}")
        print(f"  Reasoning: {decision.reasoning}")
        
        # Simulate reward
        reward = random.uniform(-1.0, 5.0)
        selector.update_reward(
            symbol="BTCUSDT",
            regime=MarketRegime.TREND_UP,
            strategy_id=decision.strategy_id,
            reward=reward,
        )
        print(f"  Simulated reward: {reward:+.2f}R\n")
    
    print("\nPerformance Summary:")
    summary = selector.get_performance_summary()
    print(json.dumps(summary, indent=2))
