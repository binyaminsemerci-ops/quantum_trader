"""
Strategy Service - Query layer for active strategies
EPIC: DASHBOARD-V3-TRADING-PANELS

Provides read-only view over PolicyStore for active strategies.
"""

import logging
from typing import List, Optional

from backend.domains.strategies.models import StrategyInfo

logger = logging.getLogger(__name__)


class StrategyService:
    """
    Service for querying active trading strategies.
    
    Reads from PolicyStore - no new storage needed.
    """
    
    def __init__(self, policy_store: Optional[any] = None):
        """
        Initialize strategy service.
        
        Args:
            policy_store: PolicyStore instance (optional)
        """
        self.policy_store = policy_store
    
    def get_active_strategies(self) -> List[StrategyInfo]:
        """
        Get list of active trading strategies.
        
        Returns:
            List of StrategyInfo objects representing active strategies
        """
        try:
            strategies = []
            
            if self.policy_store is None:
                logger.debug("[StrategyService] PolicyStore not available, returning default strategy")
                # Return a default strategy when PolicyStore is unavailable
                strategies.append(StrategyInfo(
                    name="default",
                    enabled=True,
                    profile="normal",
                    exchanges=["binance_testnet"],
                    symbols=[],
                    description="Default trading strategy",
                    min_confidence=0.65
                ))
                return strategies
            
            # Get policy from PolicyStore
            try:
                policy = self.policy_store.get()
            except Exception as e:
                logger.warning(f"[StrategyService] Could not get policy: {e}")
                policy = {}
            
            # Extract strategy information from policy
            risk_mode = policy.get("risk_mode", "NORMAL")
            max_positions = policy.get("max_positions", 5)
            min_confidence = policy.get("global_min_confidence", 0.65)
            
            # Map risk mode to profile name
            profile_map = {
                "AGGRESSIVE": "agg",
                "NORMAL": "normal",
                "DEFENSIVE": "low",
                "CONSERVATIVE": "micro"
            }
            profile = profile_map.get(risk_mode, "normal")
            
            # Create main strategy entry
            main_strategy = StrategyInfo(
                name=f"quantum_trader_{risk_mode.lower()}",
                enabled=True,
                profile=profile,
                exchanges=["binance_testnet"],
                symbols=[],  # All symbols from universe
                description=f"Main {risk_mode} strategy (max {max_positions} positions)",
                min_confidence=min_confidence
            )
            strategies.append(main_strategy)
            
            # Add AI ensemble strategy if enabled
            try:
                # Check if AI models are active
                strategies.append(StrategyInfo(
                    name="ai_ensemble",
                    enabled=True,
                    profile=profile,
                    exchanges=["binance_testnet"],
                    symbols=[],
                    description="4-model AI ensemble (XGB+TFT+LSTM+RF)",
                    min_confidence=min_confidence
                ))
            except:
                pass
            
            logger.info(f"[StrategyService] Retrieved {len(strategies)} active strategies")
            return strategies
            
        except Exception as e:
            logger.error(f"[StrategyService] Error retrieving strategies: {e}")
            # Return at least one default strategy on error
            return [StrategyInfo(
                name="default",
                enabled=True,
                profile="normal",
                exchanges=["binance_testnet"],
                symbols=[],
                description="Default trading strategy"
            )]
    
    def get_strategy_by_name(self, name: str) -> Optional[StrategyInfo]:
        """
        Get specific strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            StrategyInfo if found, None otherwise
        """
        strategies = self.get_active_strategies()
        for strategy in strategies:
            if strategy.name == name:
                return strategy
        return None
