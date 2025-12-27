"""
Policy Store implementation for Quantum Trader.

Provides async interface for storing and retrieving global trading policies.
"""

import json
import logging
from typing import Protocol, Optional
from datetime import datetime

import redis.asyncio as redis

from .models import GlobalPolicy, RiskMode

logger = logging.getLogger(__name__)


class PolicyStore(Protocol):
    """Protocol defining the PolicyStore interface."""
    
    async def get_policy(self) -> GlobalPolicy:
        """Get the current global policy."""
        ...
    
    async def set_policy(self, policy: GlobalPolicy) -> None:
        """Set the global policy."""
        ...
    
    async def update_risk_mode(self, mode: RiskMode, updated_by: str = "system") -> None:
        """Update just the risk mode."""
        ...
    
    async def add_allowed_strategy(self, strategy_id: str) -> None:
        """Add a strategy to the allowed list."""
        ...
    
    async def remove_allowed_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the allowed list."""
        ...
    
    async def get_allowed_strategies(self) -> list[str]:
        """Get list of allowed strategies."""
        ...


class RedisPolicyStore(PolicyStore):
    """
    Redis-backed implementation of PolicyStore.
    
    Stores the global policy in Redis for fast access and persistence.
    Uses atomic operations to ensure consistency.
    """
    
    POLICY_KEY = "quantum_trader:policy:global"
    POLICY_HISTORY_KEY = "quantum_trader:policy:history"
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize the PolicyStore.
        
        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self._default_policy = GlobalPolicy()
    
    async def get_policy(self) -> GlobalPolicy:
        """
        Get the current global policy.
        
        Returns:
            Current GlobalPolicy, or default if not set
        """
        try:
            data = await self.redis.get(self.POLICY_KEY)
            
            if data is None:
                logger.info("No policy found in store, using default")
                return self._default_policy
            
            policy_dict = json.loads(data)
            return GlobalPolicy.from_dict(policy_dict)
            
        except Exception as e:
            logger.error(f"Error reading policy from store: {e}", exc_info=True)
            return self._default_policy
    
    async def set_policy(self, policy: GlobalPolicy) -> None:
        """
        Set the global policy.
        
        Args:
            policy: New GlobalPolicy to store
        """
        try:
            # Update timestamp and increment version
            policy.updated_at = datetime.utcnow()
            policy.version += 1
            
            # Store current policy
            policy_json = json.dumps(policy.to_dict())
            await self.redis.set(self.POLICY_KEY, policy_json)
            
            # Store in history (keep last 100)
            await self.redis.lpush(self.POLICY_HISTORY_KEY, policy_json)
            await self.redis.ltrim(self.POLICY_HISTORY_KEY, 0, 99)
            
            logger.info(f"Policy updated (version {policy.version})")
            
        except Exception as e:
            logger.error(f"Error storing policy: {e}", exc_info=True)
            raise
    
    async def update_risk_mode(self, mode: RiskMode, updated_by: str = "system") -> None:
        """
        Update just the risk mode.
        
        Args:
            mode: New RiskMode
            updated_by: Who/what triggered the update
        """
        policy = await self.get_policy()
        policy.risk_mode = mode
        policy.updated_by = updated_by
        
        # Adjust other parameters based on risk mode
        if mode == RiskMode.AGGRESSIVE:
            policy.global_min_confidence = 0.60
            policy.max_risk_per_trade = 0.03
            policy.max_positions = 8
            policy.max_daily_trades = 30
        elif mode == RiskMode.DEFENSIVE:
            policy.global_min_confidence = 0.75
            policy.max_risk_per_trade = 0.01
            policy.max_positions = 3
            policy.max_daily_trades = 10
        else:  # NORMAL
            policy.global_min_confidence = 0.65
            policy.max_risk_per_trade = 0.02
            policy.max_positions = 5
            policy.max_daily_trades = 20
        
        await self.set_policy(policy)
        logger.info(f"Risk mode updated to {mode.value} by {updated_by}")
    
    async def add_allowed_strategy(self, strategy_id: str) -> None:
        """Add a strategy to the allowed list."""
        policy = await self.get_policy()
        
        if strategy_id not in policy.allowed_strategies:
            policy.allowed_strategies.append(strategy_id)
            await self.set_policy(policy)
            logger.info(f"Strategy {strategy_id} added to allowed list")
    
    async def remove_allowed_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the allowed list."""
        policy = await self.get_policy()
        
        if strategy_id in policy.allowed_strategies:
            policy.allowed_strategies.remove(strategy_id)
            await self.set_policy(policy)
            logger.info(f"Strategy {strategy_id} removed from allowed list")
    
    async def get_allowed_strategies(self) -> list[str]:
        """Get list of allowed strategies."""
        policy = await self.get_policy()
        return policy.allowed_strategies
    
    async def get_policy_history(self, limit: int = 10) -> list[GlobalPolicy]:
        """
        Get policy history.
        
        Args:
            limit: Maximum number of historical policies to return
            
        Returns:
            List of historical GlobalPolicy objects
        """
        try:
            history_data = await self.redis.lrange(self.POLICY_HISTORY_KEY, 0, limit - 1)
            
            policies = []
            for data in history_data:
                policy_dict = json.loads(data)
                policies.append(GlobalPolicy.from_dict(policy_dict))
            
            return policies
            
        except Exception as e:
            logger.error(f"Error reading policy history: {e}", exc_info=True)
            return []
    
    async def initialize_default_policy(self) -> None:
        """Initialize the policy store with default policy if empty."""
        existing = await self.redis.get(self.POLICY_KEY)
        
        if existing is None:
            logger.info("Initializing policy store with default policy")
            await self.set_policy(self._default_policy)


class PolicyDefaults:
    """Factory for default policy configurations."""

    @staticmethod
    def create_default() -> GlobalPolicy:
        """Create a default empty policy with safe initial values."""
        return GlobalPolicy(
            risk_mode=RiskMode.NORMAL,
            allowed_strategies=[],
            max_risk_per_trade=0.02,
            max_positions=5,
            global_min_confidence=0.65,
            max_daily_trades=20,
            updated_by="system",
        )

    @staticmethod
    def create_conservative() -> GlobalPolicy:
        """Create a conservative policy for defensive trading."""
        return GlobalPolicy(
            risk_mode=RiskMode.DEFENSIVE,
            allowed_strategies=[],
            max_risk_per_trade=0.01,
            max_positions=3,
            global_min_confidence=0.75,
            max_daily_trades=10,
            updated_by="system",
        )

    @staticmethod
    def create_aggressive() -> GlobalPolicy:
        """Create an aggressive policy for high-risk trading."""
        return GlobalPolicy(
            risk_mode=RiskMode.AGGRESSIVE,
            allowed_strategies=[],
            max_risk_per_trade=0.03,
            max_positions=8,
            global_min_confidence=0.60,
            max_daily_trades=30,
            updated_by="system",
        )


class InMemoryPolicyStore(PolicyStore):
    """
    In-memory implementation for testing.
    
    Not suitable for production use.
    """
    
    def __init__(self, initial_policy: Optional[GlobalPolicy] = None):
        self._policy = initial_policy if initial_policy else GlobalPolicy()
        self._history: list[GlobalPolicy] = []
    
    async def get_policy(self) -> GlobalPolicy:
        return self._policy
    
    async def set_policy(self, policy: GlobalPolicy) -> None:
        policy.updated_at = datetime.utcnow()
        policy.version += 1
        self._history.append(self._policy)
        self._policy = policy
    
    async def update_risk_mode(self, mode: RiskMode, updated_by: str = "system") -> None:
        policy = await self.get_policy()
        policy.risk_mode = mode
        policy.updated_by = updated_by
        await self.set_policy(policy)
    
    async def add_allowed_strategy(self, strategy_id: str) -> None:
        if strategy_id not in self._policy.allowed_strategies:
            self._policy.allowed_strategies.append(strategy_id)
    
    async def remove_allowed_strategy(self, strategy_id: str) -> None:
        if strategy_id in self._policy.allowed_strategies:
            self._policy.allowed_strategies.remove(strategy_id)
    
    async def get_allowed_strategies(self) -> list[str]:
        return self._policy.allowed_strategies
