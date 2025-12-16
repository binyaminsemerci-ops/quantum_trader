"""
Unit tests for Policy Store.
"""

import pytest
from datetime import datetime

from backend.services.policy_store import (
    InMemoryPolicyStore,
    GlobalPolicy,
    RiskMode,
)


@pytest.fixture
def policy_store():
    return InMemoryPolicyStore()


class TestPolicyStore:
    """Test PolicyStore functionality."""
    
    @pytest.mark.asyncio
    async def test_get_default_policy(self, policy_store):
        """Test getting default policy."""
        policy = await policy_store.get_policy()
        
        assert policy.risk_mode == RiskMode.NORMAL
        assert policy.global_min_confidence == 0.65
        assert policy.max_positions == 5
    
    @pytest.mark.asyncio
    async def test_set_policy(self, policy_store):
        """Test setting a new policy."""
        new_policy = GlobalPolicy(
            risk_mode=RiskMode.AGGRESSIVE,
            global_min_confidence=0.70,
            max_positions=10,
        )
        
        await policy_store.set_policy(new_policy)
        
        retrieved = await policy_store.get_policy()
        assert retrieved.risk_mode == RiskMode.AGGRESSIVE
        assert retrieved.global_min_confidence == 0.70
        assert retrieved.max_positions == 10
    
    @pytest.mark.asyncio
    async def test_update_risk_mode(self, policy_store):
        """Test updating risk mode."""
        await policy_store.update_risk_mode(RiskMode.DEFENSIVE, "test")
        
        policy = await policy_store.get_policy()
        assert policy.risk_mode == RiskMode.DEFENSIVE
        assert policy.updated_by == "test"
    
    @pytest.mark.asyncio
    async def test_allowed_strategies(self, policy_store):
        """Test managing allowed strategies."""
        # Add strategies
        await policy_store.add_allowed_strategy("strat1")
        await policy_store.add_allowed_strategy("strat2")
        
        strategies = await policy_store.get_allowed_strategies()
        assert "strat1" in strategies
        assert "strat2" in strategies
        
        # Remove strategy
        await policy_store.remove_allowed_strategy("strat1")
        
        strategies = await policy_store.get_allowed_strategies()
        assert "strat1" not in strategies
        assert "strat2" in strategies
    
    @pytest.mark.asyncio
    async def test_policy_serialization(self):
        """Test policy to_dict and from_dict."""
        original = GlobalPolicy(
            risk_mode=RiskMode.AGGRESSIVE,
            allowed_strategies=["s1", "s2"],
            max_positions=8,
        )
        
        # Serialize
        data = original.to_dict()
        assert data["risk_mode"] == "AGGRESSIVE"
        assert data["allowed_strategies"] == ["s1", "s2"]
        
        # Deserialize
        restored = GlobalPolicy.from_dict(data)
        assert restored.risk_mode == RiskMode.AGGRESSIVE
        assert restored.allowed_strategies == ["s1", "s2"]
        assert restored.max_positions == 8
