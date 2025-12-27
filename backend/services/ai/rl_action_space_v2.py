"""
RL Action Space v2 - Expanded Action Definitions
=================================================

Defines expanded action spaces for:
- Meta Strategy RL (strategy, model, weight actions)
- Position Sizing RL (size multiplier, leverage actions)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class MetaStrategyType(str, Enum):
    """Meta strategy types."""
    TREND = "TREND"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"


class ModelType(str, Enum):
    """Model types for ensemble."""
    XGB = "MODEL_XGB"
    LGBM = "MODEL_LGBM"
    NHITS = "MODEL_NHITS"
    PATCHTST = "MODEL_PATCHTST"


class WeightAction(str, Enum):
    """Weight adjustment actions."""
    WEIGHT_UP = "WEIGHT_UP"
    WEIGHT_DOWN = "WEIGHT_DOWN"
    WEIGHT_HOLD = "WEIGHT_HOLD"


class ActionSpaceV2:
    """
    Expanded action space manager for RL v2.
    
    Handles:
    - Meta strategy selection
    - Model selection
    - Weight adjustments
    - Position size multipliers
    - Leverage levels
    """
    
    # Meta Strategy Actions
    META_STRATEGIES = [
        MetaStrategyType.TREND,
        MetaStrategyType.RANGE,
        MetaStrategyType.BREAKOUT,
        MetaStrategyType.MEAN_REVERSION
    ]
    
    # Model Selection Actions
    MODELS = [
        ModelType.XGB,
        ModelType.LGBM,
        ModelType.NHITS,
        ModelType.PATCHTST
    ]
    
    # Weight Adjustment Actions
    WEIGHT_ACTIONS = [
        WeightAction.WEIGHT_UP,
        WeightAction.WEIGHT_DOWN,
        WeightAction.WEIGHT_HOLD
    ]
    
    # Position Size Multipliers
    SIZE_MULTIPLIERS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 1.8]
    
    # Leverage Levels
    LEVERAGE_LEVELS = [1, 2, 3, 4, 5, 6, 7]
    
    def __init__(self):
        """Initialize Action Space v2."""
        logger.info(
            "[Action Space v2] Initialized",
            meta_strategies=len(self.META_STRATEGIES),
            models=len(self.MODELS),
            size_multipliers=len(self.SIZE_MULTIPLIERS),
            leverage_levels=len(self.LEVERAGE_LEVELS)
        )
    
    def select_meta_strategy_action(
        self,
        q_values: Dict[str, float],
        epsilon: float = 0.1
    ) -> str:
        """
        Select meta strategy action using epsilon-greedy policy.
        
        Args:
            q_values: Q-values for each strategy
            epsilon: Exploration rate
            
        Returns:
            Selected strategy
        """
        # Epsilon-greedy selection
        if np.random.random() < epsilon:
            # Explore: random strategy
            action = np.random.choice(self.META_STRATEGIES)
            logger.debug(
                "[Action Space v2] Meta strategy exploration",
                action=action
            )
        else:
            # Exploit: best strategy
            action = max(q_values, key=q_values.get)
            logger.debug(
                "[Action Space v2] Meta strategy exploitation",
                action=action,
                q_value=q_values[action]
            )
        
        return action
    
    def select_model_action(
        self,
        q_values: Dict[str, float],
        epsilon: float = 0.1
    ) -> str:
        """
        Select model action using epsilon-greedy policy.
        
        Args:
            q_values: Q-values for each model
            epsilon: Exploration rate
            
        Returns:
            Selected model
        """
        if np.random.random() < epsilon:
            action = np.random.choice(self.MODELS)
            logger.debug(
                "[Action Space v2] Model exploration",
                action=action
            )
        else:
            action = max(q_values, key=q_values.get)
            logger.debug(
                "[Action Space v2] Model exploitation",
                action=action,
                q_value=q_values[action]
            )
        
        return action
    
    def select_weight_action(
        self,
        q_values: Dict[str, float],
        epsilon: float = 0.1
    ) -> str:
        """
        Select weight adjustment action.
        
        Args:
            q_values: Q-values for each weight action
            epsilon: Exploration rate
            
        Returns:
            Selected weight action
        """
        if np.random.random() < epsilon:
            action = np.random.choice(self.WEIGHT_ACTIONS)
            logger.debug(
                "[Action Space v2] Weight action exploration",
                action=action
            )
        else:
            action = max(q_values, key=q_values.get)
            logger.debug(
                "[Action Space v2] Weight action exploitation",
                action=action,
                q_value=q_values[action]
            )
        
        return action
    
    def select_size_multiplier(
        self,
        q_values: List[float],
        epsilon: float = 0.1
    ) -> float:
        """
        Select position size multiplier.
        
        Args:
            q_values: Q-values for each multiplier
            epsilon: Exploration rate
            
        Returns:
            Selected size multiplier
        """
        if np.random.random() < epsilon:
            idx = np.random.randint(len(self.SIZE_MULTIPLIERS))
            multiplier = self.SIZE_MULTIPLIERS[idx]
            logger.debug(
                "[Action Space v2] Size multiplier exploration",
                multiplier=multiplier
            )
        else:
            idx = np.argmax(q_values)
            multiplier = self.SIZE_MULTIPLIERS[idx]
            logger.debug(
                "[Action Space v2] Size multiplier exploitation",
                multiplier=multiplier,
                q_value=q_values[idx]
            )
        
        return multiplier
    
    def select_leverage(
        self,
        q_values: List[float],
        epsilon: float = 0.1
    ) -> int:
        """
        Select leverage level.
        
        Args:
            q_values: Q-values for each leverage level
            epsilon: Exploration rate
            
        Returns:
            Selected leverage
        """
        if np.random.random() < epsilon:
            idx = np.random.randint(len(self.LEVERAGE_LEVELS))
            leverage = self.LEVERAGE_LEVELS[idx]
            logger.debug(
                "[Action Space v2] Leverage exploration",
                leverage=leverage
            )
        else:
            idx = np.argmax(q_values)
            leverage = self.LEVERAGE_LEVELS[idx]
            logger.debug(
                "[Action Space v2] Leverage exploitation",
                leverage=leverage,
                q_value=q_values[idx]
            )
        
        return leverage
    
    def get_meta_action_space_size(self) -> int:
        """Get total meta action space size."""
        return len(self.META_STRATEGIES) * len(self.MODELS) * len(self.WEIGHT_ACTIONS)
    
    def get_size_action_space_size(self) -> int:
        """Get total size action space size."""
        return len(self.SIZE_MULTIPLIERS) * len(self.LEVERAGE_LEVELS)
    
    def decode_meta_action(self, action_index: int) -> Tuple[str, str, str]:
        """
        Decode meta action index to components.
        
        Args:
            action_index: Flattened action index
            
        Returns:
            (strategy, model, weight_action)
        """
        n_strategies = len(self.META_STRATEGIES)
        n_models = len(self.MODELS)
        n_weights = len(self.WEIGHT_ACTIONS)
        
        weight_idx = action_index % n_weights
        model_idx = (action_index // n_weights) % n_models
        strategy_idx = (action_index // (n_weights * n_models)) % n_strategies
        
        strategy = self.META_STRATEGIES[strategy_idx]
        model = self.MODELS[model_idx]
        weight_action = self.WEIGHT_ACTIONS[weight_idx]
        
        return strategy, model, weight_action
    
    def encode_meta_action(
        self,
        strategy: str,
        model: str,
        weight_action: str
    ) -> int:
        """
        Encode meta action components to index.
        
        Args:
            strategy: Strategy type
            model: Model type
            weight_action: Weight action
            
        Returns:
            Flattened action index
        """
        strategy_idx = self.META_STRATEGIES.index(strategy)
        model_idx = self.MODELS.index(model)
        weight_idx = self.WEIGHT_ACTIONS.index(weight_action)
        
        n_weights = len(self.WEIGHT_ACTIONS)
        n_models = len(self.MODELS)
        
        action_index = (
            strategy_idx * n_models * n_weights +
            model_idx * n_weights +
            weight_idx
        )
        
        return action_index
    
    def decode_size_action(self, action_index: int) -> Tuple[float, int]:
        """
        Decode size action index to components.
        
        Args:
            action_index: Flattened action index
            
        Returns:
            (size_multiplier, leverage)
        """
        n_multipliers = len(self.SIZE_MULTIPLIERS)
        n_leverage = len(self.LEVERAGE_LEVELS)
        
        leverage_idx = action_index % n_leverage
        multiplier_idx = (action_index // n_leverage) % n_multipliers
        
        multiplier = self.SIZE_MULTIPLIERS[multiplier_idx]
        leverage = self.LEVERAGE_LEVELS[leverage_idx]
        
        return multiplier, leverage
    
    def encode_size_action(
        self,
        size_multiplier: float,
        leverage: int
    ) -> int:
        """
        Encode size action components to index.
        
        Args:
            size_multiplier: Size multiplier
            leverage: Leverage level
            
        Returns:
            Flattened action index
        """
        multiplier_idx = self.SIZE_MULTIPLIERS.index(size_multiplier)
        leverage_idx = self.LEVERAGE_LEVELS.index(leverage)
        
        n_leverage = len(self.LEVERAGE_LEVELS)
        
        action_index = multiplier_idx * n_leverage + leverage_idx
        
        return action_index


# Global singleton instance
_action_space_instance: Optional[ActionSpaceV2] = None


def get_action_space() -> ActionSpaceV2:
    """Get or create global ActionSpaceV2 instance."""
    global _action_space_instance
    if _action_space_instance is None:
        _action_space_instance = ActionSpaceV2()
    return _action_space_instance
