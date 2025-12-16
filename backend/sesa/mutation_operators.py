"""Mutation Operators - Strategy Variation Generation.

Generates mutated versions of trading strategies by applying
controlled modifications to parameters, thresholds, and logic.

Mutation Types:
- Parameter mutations (small threshold adjustments)
- Time window mutations (timeframe variations)
- Volatility filter mutations (vol threshold tweaks)
- Risk/Reward mutations (TP/SL adjustments)
- RL parameter mutations (exploration/exploitation balance)
"""

from __future__ import annotations

import copy
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MutationType(str, Enum):
    """Types of strategy mutations."""
    
    PARAMETER_TWEAK = "parameter_tweak"
    TIME_WINDOW_ADJUST = "time_window_adjust"
    VOLATILITY_FILTER = "volatility_filter"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    RL_PARAMETERS = "rl_parameters"
    DYNAMIC_TPSL = "dynamic_tpsl"
    ENTRY_THRESHOLD = "entry_threshold"
    EXIT_THRESHOLD = "exit_threshold"
    HYBRID_LOGIC = "hybrid_logic"


@dataclass
class StrategyConfig:
    """
    Complete strategy configuration.
    
    Represents a trading strategy with all parameters that can be mutated.
    """
    
    strategy_id: str
    strategy_name: str
    
    # Entry/Exit logic parameters
    entry_confidence_threshold: float = 0.65
    exit_confidence_threshold: float = 0.40
    min_signal_strength: float = 0.50
    
    # Time windows
    timeframe_minutes: int = 5
    lookback_periods: int = 24
    cooldown_periods: int = 3
    
    # Volatility filters
    min_volatility: float = 0.01
    max_volatility: float = 0.50
    volatility_multiplier: float = 1.0
    
    # Risk/Reward
    risk_reward_ratio: float = 2.0
    min_rr_ratio: float = 1.5
    max_rr_ratio: float = 4.0
    
    # TP/SL settings
    use_dynamic_tpsl: bool = True
    base_tp_pct: float = 0.02
    base_sl_pct: float = 0.01
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.015
    
    # RL parameters (if using RL-based strategy)
    rl_exploration_rate: float = 0.15
    rl_learning_rate: float = 0.001
    rl_discount_factor: float = 0.95
    
    # Risk settings (within policy boundaries)
    position_size_pct: float = 0.02
    max_concurrent_positions: int = 3
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    parent_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = self.__dict__.copy()
        d["created_at"] = self.created_at.isoformat()
        return d
    
    def clone(self) -> StrategyConfig:
        """Create a deep copy."""
        return copy.deepcopy(self)


@dataclass
class MutatedStrategy:
    """
    A mutated strategy with metadata about the mutation.
    """
    
    mutation_id: str
    parent_id: str
    config: StrategyConfig
    mutation_type: MutationType
    mutation_magnitude: float  # 0.0 to 1.0 (how aggressive)
    changed_parameters: list[str]
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mutation_id": self.mutation_id,
            "parent_id": self.parent_id,
            "config": self.config.to_dict(),
            "mutation_type": self.mutation_type.value,
            "mutation_magnitude": self.mutation_magnitude,
            "changed_parameters": self.changed_parameters,
            "created_at": self.created_at.isoformat(),
        }


class MutationOperators:
    """
    Mutation operators for strategy evolution.
    
    Features:
    - Generate N mutations from a parent strategy
    - Control mutation magnitude (small vs large changes)
    - Respect policy boundaries (no unsafe mutations)
    - Track mutation history
    
    Usage:
        operators = MutationOperators(seed=42)
        
        parent = StrategyConfig(
            strategy_id="base_strategy_v1",
            strategy_name="Base Scalping Strategy",
            entry_confidence_threshold=0.65,
            risk_reward_ratio=2.0,
        )
        
        mutations = operators.generate_mutations(
            parent=parent,
            num_mutations=10,
            mutation_rate=0.3,
        )
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        policy_constraints: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """
        Initialize Mutation Operators.
        
        Args:
            seed: Random seed for reproducibility
            policy_constraints: Dict of parameter: (min, max) constraints
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Default policy constraints
        self.constraints = policy_constraints or {
            "entry_confidence_threshold": (0.50, 0.90),
            "exit_confidence_threshold": (0.30, 0.70),
            "risk_reward_ratio": (1.2, 5.0),
            "position_size_pct": (0.005, 0.05),
            "max_concurrent_positions": (1, 10),
            "rl_exploration_rate": (0.05, 0.40),
            "volatility_multiplier": (0.5, 2.0),
        }
        
        logger.info("MutationOperators initialized")
    
    def generate_mutations(
        self,
        parent: StrategyConfig,
        num_mutations: int = 10,
        mutation_rate: float = 0.3,
        mutation_magnitude: float = 0.5,
    ) -> list[MutatedStrategy]:
        """
        Generate mutations from parent strategy.
        
        Args:
            parent: Parent strategy configuration
            num_mutations: Number of mutations to generate
            mutation_rate: Probability of mutating each parameter (0-1)
            mutation_magnitude: Size of mutations (0-1, small to large)
        
        Returns:
            List of mutated strategies
        """
        mutations: list[MutatedStrategy] = []
        
        for i in range(num_mutations):
            # Select mutation type
            mutation_type = random.choice(list(MutationType))
            
            # Apply mutation
            mutated_config = self._apply_mutation(
                parent,
                mutation_type,
                mutation_rate,
                mutation_magnitude,
            )
            
            # Create mutation record
            mutation = MutatedStrategy(
                mutation_id=f"{parent.strategy_id}_mut_{i+1}_{uuid.uuid4().hex[:6]}",
                parent_id=parent.strategy_id,
                config=mutated_config,
                mutation_type=mutation_type,
                mutation_magnitude=mutation_magnitude,
                changed_parameters=self._get_changed_parameters(parent, mutated_config),
            )
            
            mutations.append(mutation)
        
        logger.info(
            f"Generated {len(mutations)} mutations from {parent.strategy_id}"
        )
        
        return mutations
    
    def _apply_mutation(
        self,
        parent: StrategyConfig,
        mutation_type: MutationType,
        mutation_rate: float,
        magnitude: float,
    ) -> StrategyConfig:
        """Apply specific mutation type to parent."""
        mutated = parent.clone()
        mutated.strategy_id = f"{parent.strategy_id}_mutated"
        mutated.parent_id = parent.strategy_id
        mutated.created_at = datetime.now()
        
        if mutation_type == MutationType.PARAMETER_TWEAK:
            self._mutate_parameters(mutated, mutation_rate, magnitude)
        
        elif mutation_type == MutationType.TIME_WINDOW_ADJUST:
            self._mutate_time_windows(mutated, magnitude)
        
        elif mutation_type == MutationType.VOLATILITY_FILTER:
            self._mutate_volatility_filters(mutated, magnitude)
        
        elif mutation_type == MutationType.RISK_REWARD_RATIO:
            self._mutate_risk_reward(mutated, magnitude)
        
        elif mutation_type == MutationType.RL_PARAMETERS:
            self._mutate_rl_parameters(mutated, magnitude)
        
        elif mutation_type == MutationType.DYNAMIC_TPSL:
            self._mutate_dynamic_tpsl(mutated, magnitude)
        
        elif mutation_type == MutationType.ENTRY_THRESHOLD:
            self._mutate_entry_threshold(mutated, magnitude)
        
        elif mutation_type == MutationType.EXIT_THRESHOLD:
            self._mutate_exit_threshold(mutated, magnitude)
        
        elif mutation_type == MutationType.HYBRID_LOGIC:
            self._mutate_hybrid(mutated, mutation_rate, magnitude)
        
        return mutated
    
    def _mutate_parameters(
        self,
        config: StrategyConfig,
        mutation_rate: float,
        magnitude: float,
    ) -> None:
        """Mutate general parameters with controlled magnitude."""
        # Entry confidence
        if random.random() < mutation_rate:
            delta = np.random.normal(0, 0.05 * magnitude)
            config.entry_confidence_threshold = self._clamp(
                config.entry_confidence_threshold + delta,
                *self.constraints["entry_confidence_threshold"],
            )
        
        # Min signal strength
        if random.random() < mutation_rate:
            delta = np.random.normal(0, 0.05 * magnitude)
            config.min_signal_strength = np.clip(
                config.min_signal_strength + delta,
                0.30,
                0.80,
            )
        
        # Position size
        if random.random() < mutation_rate:
            delta = np.random.normal(0, 0.005 * magnitude)
            config.position_size_pct = self._clamp(
                config.position_size_pct + delta,
                *self.constraints["position_size_pct"],
            )
    
    def _mutate_time_windows(self, config: StrategyConfig, magnitude: float) -> None:
        """Mutate time window parameters."""
        # Timeframe
        if random.random() < 0.5:
            timeframes = [1, 5, 15, 30, 60]
            current_idx = timeframes.index(config.timeframe_minutes) if config.timeframe_minutes in timeframes else 1
            
            if magnitude > 0.7:
                # Large mutation: jump 2 steps
                new_idx = current_idx + random.choice([-2, -1, 1, 2])
            else:
                # Small mutation: jump 1 step
                new_idx = current_idx + random.choice([-1, 1])
            
            new_idx = np.clip(new_idx, 0, len(timeframes) - 1)
            config.timeframe_minutes = timeframes[new_idx]
        
        # Lookback periods
        if random.random() < 0.5:
            delta = int(np.random.normal(0, 5 * magnitude))
            config.lookback_periods = int(np.clip(
                config.lookback_periods + delta,
                6,
                100,
            ))
        
        # Cooldown periods
        if random.random() < 0.5:
            delta = int(np.random.normal(0, 2 * magnitude))
            config.cooldown_periods = int(np.clip(
                config.cooldown_periods + delta,
                1,
                10,
            ))
    
    def _mutate_volatility_filters(
        self,
        config: StrategyConfig,
        magnitude: float,
    ) -> None:
        """Mutate volatility filter parameters."""
        # Min volatility
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.005 * magnitude)
            config.min_volatility = np.clip(
                config.min_volatility + delta,
                0.001,
                0.10,
            )
        
        # Max volatility
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.05 * magnitude)
            config.max_volatility = np.clip(
                config.max_volatility + delta,
                0.20,
                1.0,
            )
        
        # Volatility multiplier
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.2 * magnitude)
            config.volatility_multiplier = self._clamp(
                config.volatility_multiplier + delta,
                *self.constraints["volatility_multiplier"],
            )
    
    def _mutate_risk_reward(self, config: StrategyConfig, magnitude: float) -> None:
        """Mutate risk/reward ratio."""
        delta = np.random.normal(0, 0.5 * magnitude)
        config.risk_reward_ratio = self._clamp(
            config.risk_reward_ratio + delta,
            *self.constraints["risk_reward_ratio"],
        )
        
        # Adjust min/max bounds accordingly
        config.min_rr_ratio = max(1.2, config.risk_reward_ratio - 0.5)
        config.max_rr_ratio = min(5.0, config.risk_reward_ratio + 1.0)
    
    def _mutate_rl_parameters(self, config: StrategyConfig, magnitude: float) -> None:
        """Mutate RL hyperparameters."""
        # Exploration rate
        if random.random() < 0.6:
            delta = np.random.normal(0, 0.05 * magnitude)
            config.rl_exploration_rate = self._clamp(
                config.rl_exploration_rate + delta,
                *self.constraints["rl_exploration_rate"],
            )
        
        # Learning rate
        if random.random() < 0.6:
            # Use log scale for learning rate
            log_lr = np.log10(config.rl_learning_rate)
            delta = np.random.normal(0, 0.3 * magnitude)
            new_log_lr = np.clip(log_lr + delta, -4, -2)
            config.rl_learning_rate = 10 ** new_log_lr
        
        # Discount factor
        if random.random() < 0.6:
            delta = np.random.normal(0, 0.02 * magnitude)
            config.rl_discount_factor = np.clip(
                config.rl_discount_factor + delta,
                0.90,
                0.99,
            )
    
    def _mutate_dynamic_tpsl(self, config: StrategyConfig, magnitude: float) -> None:
        """Mutate dynamic TP/SL parameters."""
        # Base TP
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.005 * magnitude)
            config.base_tp_pct = np.clip(
                config.base_tp_pct + delta,
                0.005,
                0.05,
            )
        
        # Base SL
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.003 * magnitude)
            config.base_sl_pct = np.clip(
                config.base_sl_pct + delta,
                0.003,
                0.03,
            )
        
        # Trailing stop activation
        if random.random() < 0.5:
            delta = np.random.normal(0, 0.003 * magnitude)
            config.trailing_stop_activation_pct = np.clip(
                config.trailing_stop_activation_pct + delta,
                0.005,
                0.03,
            )
        
        # Toggle trailing stop
        if random.random() < 0.2 * magnitude:
            config.trailing_stop_enabled = not config.trailing_stop_enabled
    
    def _mutate_entry_threshold(
        self,
        config: StrategyConfig,
        magnitude: float,
    ) -> None:
        """Mutate entry threshold specifically."""
        delta = np.random.normal(0, 0.10 * magnitude)
        config.entry_confidence_threshold = self._clamp(
            config.entry_confidence_threshold + delta,
            *self.constraints["entry_confidence_threshold"],
        )
    
    def _mutate_exit_threshold(
        self,
        config: StrategyConfig,
        magnitude: float,
    ) -> None:
        """Mutate exit threshold specifically."""
        delta = np.random.normal(0, 0.10 * magnitude)
        config.exit_confidence_threshold = self._clamp(
            config.exit_confidence_threshold + delta,
            *self.constraints["exit_confidence_threshold"],
        )
    
    def _mutate_hybrid(
        self,
        config: StrategyConfig,
        mutation_rate: float,
        magnitude: float,
    ) -> None:
        """Apply multiple mutations (hybrid approach)."""
        self._mutate_parameters(config, mutation_rate * 0.5, magnitude)
        self._mutate_time_windows(config, magnitude * 0.7)
        self._mutate_risk_reward(config, magnitude * 0.7)
    
    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))
    
    def _get_changed_parameters(
        self,
        original: StrategyConfig,
        mutated: StrategyConfig,
    ) -> list[str]:
        """Identify which parameters changed."""
        changed = []
        
        for key in original.__dict__:
            if key in ["strategy_id", "parent_id", "created_at", "strategy_name"]:
                continue
            
            orig_val = getattr(original, key)
            mut_val = getattr(mutated, key)
            
            if orig_val != mut_val:
                changed.append(key)
        
        return changed
