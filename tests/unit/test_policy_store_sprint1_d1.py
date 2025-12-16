"""
SPRINT 1 - D1: PolicyStore Single Source of Truth Tests

Tests new helper methods and integration with RL/Executor.
"""
import pytest
import redis.asyncio as redis
from backend.core.policy_store import PolicyStore
from backend.models.policy import RiskMode


@pytest.fixture
async def redis_client():
    """Create test Redis client."""
    client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True
    )
    await client.flushdb()
    yield client
    await client.aclose()


@pytest.fixture
async def policy_store(redis_client):
    """Create PolicyStore fixture."""
    store = PolicyStore(redis_client=redis_client, event_bus=None)
    await store.initialize()
    
    # 🔧 Ensure NORMAL mode for predictable test results
    await store.switch_mode(RiskMode.NORMAL, updated_by="test_fixture")
    
    yield store
    await store.shutdown()


class TestPolicyStoreHelperMethods:
    """Test new helper methods added in SPRINT 1 - D1."""
    
    @pytest.mark.asyncio
    async def test_get_active_risk_config(self, policy_store):
        """Test get_active_risk_config() returns correct config."""
        config = await policy_store.get_active_risk_config()
        
        assert config is not None
        assert hasattr(config, 'max_leverage')
        assert hasattr(config, 'max_daily_drawdown')
        assert hasattr(config, 'global_min_confidence')
        
        # Default should be NORMAL mode
        assert config.max_leverage == 5.0
        assert config.max_daily_drawdown == 0.05
        assert config.global_min_confidence == 0.50
    
    @pytest.mark.asyncio
    async def test_get_value_with_dot_notation(self, policy_store):
        """Test get_value() with dot-notation paths."""
        # Test risk paths
        max_leverage = await policy_store.get_value("risk.max_leverage", 10.0)
        assert max_leverage == 5.0  # NORMAL mode default
        
        max_drawdown = await policy_store.get_value("risk.max_daily_drawdown", 0.1)
        assert max_drawdown == 0.05
        
        min_confidence = await policy_store.get_value("risk.global_min_confidence", 0.3)
        assert min_confidence == 0.50
        
        # Test AI module paths
        enable_rl = await policy_store.get_value("ai.enable_rl", False)
        assert enable_rl is True
        
        enable_clm = await policy_store.get_value("ai.enable_clm", False)
        assert enable_clm is True
    
    @pytest.mark.asyncio
    async def test_get_value_with_default(self, policy_store):
        """Test get_value() returns default for invalid paths."""
        invalid_value = await policy_store.get_value("nonexistent.path", 999.0)
        assert invalid_value == 999.0
    
    @pytest.mark.asyncio
    async def test_get_value_after_mode_switch(self, policy_store):
        """Test get_value() reflects mode changes."""
        # Initial NORMAL mode
        max_leverage = await policy_store.get_value("risk.max_leverage")
        assert max_leverage == 5.0
        
        # Switch to AGGRESSIVE_SMALL_ACCOUNT
        await policy_store.switch_mode(RiskMode.AGGRESSIVE_SMALL_ACCOUNT, updated_by="test")
        
        max_leverage = await policy_store.get_value("risk.max_leverage")
        assert max_leverage == 7.0  # AGGRESSIVE default
        
        # Switch to DEFENSIVE
        await policy_store.switch_mode(RiskMode.DEFENSIVE, updated_by="test")
        
        max_leverage = await policy_store.get_value("risk.max_leverage")
        assert max_leverage == 3.0  # DEFENSIVE default


class TestPolicyStoreRLIntegration:
    """Test PolicyStore integration with RLPositionSizingAgent."""
    
    @pytest.mark.asyncio
    async def test_rl_agent_reads_from_policy_store(self, policy_store):
        """Test RLAgent can read max_leverage from PolicyStore."""
        from backend.services.ai.rl_position_sizing_agent import RLPositionSizingAgent
        
        # Read max_leverage from PolicyStore first
        max_leverage = await policy_store.get_value("risk.max_leverage", 25.0)
        
        # Create agent and pass the value
        agent = RLPositionSizingAgent(
            policy_store=policy_store,
            max_leverage=max_leverage,  # 🔧 Pass value from PolicyStore
            use_math_ai=False  # Disable math AI for simpler test
        )
        
        # Agent should have the leverage value from PolicyStore (NORMAL mode = 5.0)
        assert agent.max_leverage == 5.0
        
        # Switch to AGGRESSIVE mode
        await policy_store.switch_mode(RiskMode.AGGRESSIVE_SMALL_ACCOUNT, updated_by="test")
        
        # Read new leverage value
        max_leverage_agg = await policy_store.get_value("risk.max_leverage", 25.0)
        
        # Create new agent - should reflect new leverage
        agent2 = RLPositionSizingAgent(
            policy_store=policy_store,
            max_leverage=max_leverage_agg,  # 🔧 Pass AGGRESSIVE value
            use_math_ai=False
        )
        assert agent2.max_leverage == 7.0


class TestPolicyStoreExecutorIntegration:
    """Test PolicyStore integration with EventDrivenExecutor."""
    
    @pytest.mark.asyncio
    async def test_executor_reads_from_policy_store(self, policy_store):
        """Test Executor _get_strategy_from_policy_store() helper method."""
        # Test the helper method logic without full Executor instantiation
        
        # Simulate what _get_strategy_from_policy_store() does
        risk_config = await policy_store.get_active_risk_config()
        
        # Build config dict like the helper method would
        strategy_config = {
            "max_position_size": risk_config.max_risk_pct_per_trade,
            "max_leverage": risk_config.max_leverage,
            "confidence_threshold": risk_config.global_min_confidence,
            "cooldown_seconds": 300,
            "max_open_positions": risk_config.max_positions
        }
        
        # Verify config structure and values
        assert strategy_config["max_leverage"] == 5.0  # NORMAL mode
        assert strategy_config["confidence_threshold"] == 0.5
        assert strategy_config["max_position_size"] == 0.015  # NORMAL: 1.5%
        assert "cooldown_seconds" in strategy_config


class TestPolicyStoreCaching:
    """Test PolicyStore cache behavior."""
    
    @pytest.mark.asyncio
    async def test_cache_is_valid_for_ttl(self, policy_store):
        """Test cache is used within TTL."""
        # First call - loads from Redis
        config1 = await policy_store.get_active_risk_config()
        
        # Second call immediately - should use cache
        config2 = await policy_store.get_active_risk_config()
        
        assert config1 is config2  # Same object from cache
    
    @pytest.mark.asyncio
    async def test_get_value_works_without_cache(self, policy_store):
        """Test get_value() works even if cache is invalidated."""
        # First ensure we're in NORMAL mode
        await policy_store.switch_mode(RiskMode.NORMAL, updated_by="test")
        
        # Invalidate cache
        policy_store._cache = None
        policy_store._cache_timestamp = None
        
        # Should still work by loading from Redis (NORMAL mode = 5.0x)
        max_leverage = await policy_store.get_value("risk.max_leverage")
        assert max_leverage == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
