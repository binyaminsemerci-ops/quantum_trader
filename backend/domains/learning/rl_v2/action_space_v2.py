"""
Action Space v2 - Advanced Action Space
========================================

Defines action spaces for:
- Meta Strategy Agent (strategy + model + weight)
- Position Sizing Agent (size multiplier + leverage)

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 2.0
"""

from typing import Dict, List, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetaStrategyAction:
    """Meta strategy action."""
    strategy: str  # "dual_momentum" | "mean_reversion" | "momentum_flip"
    model: str  # "lstm" | "gru" | "transformer" | "ensemble"
    weight: float  # 0.5 - 1.5


@dataclass
class PositionSizingAction:
    """Position sizing action."""
    size_multiplier: float  # 0.5 - 2.0
    leverage: int  # 5 - 50


class ActionSpaceV2:
    """
    Advanced action space for RL v2.
    
    Provides:
    - Meta strategy action generation
    - Position sizing action generation
    - Action validation
    """
    
    # Action space definitions
    STRATEGIES = ["dual_momentum", "mean_reversion", "momentum_flip"]
    MODELS = ["lstm", "gru", "transformer", "ensemble"]
    WEIGHTS = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    SIZE_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]
    LEVERAGES = [5, 10, 15, 20, 25, 30, 40, 50]
    
    def __init__(self):
        """Initialize Action Space v2."""
        logger.info("[Action Space v2] Initialized")
    
    def get_meta_strategy_actions(self) -> List[MetaStrategyAction]:
        """
        Get all possible meta strategy actions.
        
        Returns:
            List of meta strategy actions
        """
        actions = []
        for strategy in self.STRATEGIES:
            for model in self.MODELS:
                for weight in self.WEIGHTS:
                    actions.append(
                        MetaStrategyAction(
                            strategy=strategy,
                            model=model,
                            weight=weight
                        )
                    )
        return actions
    
    def get_position_sizing_actions(self) -> List[PositionSizingAction]:
        """
        Get all possible position sizing actions.
        
        Returns:
            List of position sizing actions
        """
        actions = []
        for multiplier in self.SIZE_MULTIPLIERS:
            for leverage in self.LEVERAGES:
                actions.append(
                    PositionSizingAction(
                        size_multiplier=multiplier,
                        leverage=leverage
                    )
                )
        return actions
    
    def validate_meta_strategy_action(self, action: MetaStrategyAction) -> bool:
        """
        Validate meta strategy action.
        
        Args:
            action: Meta strategy action
            
        Returns:
            True if valid
        """
        return (
            action.strategy in self.STRATEGIES and
            action.model in self.MODELS and
            0.5 <= action.weight <= 1.5
        )
    
    def validate_position_sizing_action(self, action: PositionSizingAction) -> bool:
        """
        Validate position sizing action.
        
        Args:
            action: Position sizing action
            
        Returns:
            True if valid
        """
        return (
            0.5 <= action.size_multiplier <= 2.0 and
            5 <= action.leverage <= 50
        )
    
    def get_meta_strategy_action_count(self) -> int:
        """Get total number of meta strategy actions."""
        return len(self.STRATEGIES) * len(self.MODELS) * len(self.WEIGHTS)
    
    def get_position_sizing_action_count(self) -> int:
        """Get total number of position sizing actions."""
        return len(self.SIZE_MULTIPLIERS) * len(self.LEVERAGES)
    
    def action_to_dict(self, action: Any) -> Dict[str, Any]:
        """
        Convert action to dictionary.
        
        Args:
            action: Meta or position sizing action
            
        Returns:
            Action dictionary
        """
        if isinstance(action, MetaStrategyAction):
            return {
                "strategy": action.strategy,
                "model": action.model,
                "weight": action.weight
            }
        elif isinstance(action, PositionSizingAction):
            return {
                "size_multiplier": action.size_multiplier,
                "leverage": action.leverage
            }
        else:
            return {}
    
    def dict_to_meta_action(self, data: Dict[str, Any]) -> MetaStrategyAction:
        """
        Convert dictionary to meta strategy action.
        
        Args:
            data: Action dictionary
            
        Returns:
            Meta strategy action
        """
        return MetaStrategyAction(
            strategy=data.get("strategy", "dual_momentum"),
            model=data.get("model", "ensemble"),
            weight=data.get("weight", 1.0)
        )
    
    def dict_to_sizing_action(self, data: Dict[str, Any]) -> PositionSizingAction:
        """
        Convert dictionary to position sizing action.
        
        Args:
            data: Action dictionary
            
        Returns:
            Position sizing action
        """
        return PositionSizingAction(
            size_multiplier=data.get("size_multiplier", 1.0),
            leverage=data.get("leverage", 20)
        )
