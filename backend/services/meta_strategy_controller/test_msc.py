"""
Unit tests for Meta Strategy Controller.
"""

import pytest
import asyncio
from datetime import datetime

from backend.services.meta_strategy_controller import (
    MetaStrategyController,
    MarketAnalysis,
    MarketRegime,
)
from backend.services.eventbus import InMemoryEventBus, HealthStatusChangedEvent, HealthStatus
from backend.services.policy_store import InMemoryPolicyStore, RiskMode


@pytest.fixture
def event_bus():
    return InMemoryEventBus()


@pytest.fixture
def policy_store():
    return InMemoryPolicyStore()


@pytest.fixture
def msc_ai(event_bus, policy_store):
    return MetaStrategyController(
        event_bus=event_bus,
        policy_store=policy_store,
        update_interval=1,  # Fast for testing
        auto_adjust=True,
    )


class TestMarketAnalysis:
    """Test MarketAnalysis model."""
    
    def test_is_favorable_for_aggressive(self):
        """Test aggressive condition detection."""
        analysis = MarketAnalysis(
            regime=MarketRegime.TRENDING_UP,
            volatility=0.6,
            trend_strength=0.8,
            correlation=0.3,
            liquidity_score=0.9,
            risk_score=0.2,
            recent_win_rate=0.70,
            recent_sharpe=2.0,
            recent_drawdown=-1.5,
            timestamp=datetime.utcnow(),
        )
        
        assert analysis.is_favorable_for_aggressive()
    
    def test_is_unfavorable_requires_defensive(self):
        """Test defensive condition detection."""
        analysis = MarketAnalysis(
            regime=MarketRegime.CHOPPY,
            volatility=0.9,
            trend_strength=0.3,
            correlation=0.8,
            liquidity_score=0.5,
            risk_score=0.8,
            recent_win_rate=0.35,
            recent_sharpe=0.5,
            recent_drawdown=-6.0,
            timestamp=datetime.utcnow(),
        )
        
        assert analysis.is_unfavorable_requires_defensive()


class TestMetaStrategyController:
    """Test MetaStrategyController functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, msc_ai, event_bus):
        """Test MSC AI initializes correctly."""
        assert msc_ai.auto_adjust is True
        assert msc_ai.update_interval == 1
        assert not msc_ai._emergency_mode
    
    @pytest.mark.asyncio
    async def test_analyze_market_conditions(self, msc_ai):
        """Test market analysis."""
        analysis = await msc_ai.analyze_market_conditions()
        
        assert isinstance(analysis, MarketAnalysis)
        assert isinstance(analysis.regime, MarketRegime)
        assert 0 <= analysis.volatility <= 1
        assert 0 <= analysis.trend_strength <= 1
    
    @pytest.mark.asyncio
    async def test_determine_optimal_risk_mode_normal(self, msc_ai):
        """Test risk mode determination for normal conditions."""
        analysis = MarketAnalysis(
            regime=MarketRegime.RANGING,
            volatility=0.5,
            trend_strength=0.6,
            correlation=0.4,
            liquidity_score=0.8,
            risk_score=0.3,
            recent_win_rate=0.60,
            recent_sharpe=1.5,
            recent_drawdown=-2.5,
            timestamp=datetime.utcnow(),
        )
        
        mode = msc_ai.determine_optimal_risk_mode(analysis)
        assert mode == RiskMode.NORMAL
    
    @pytest.mark.asyncio
    async def test_determine_optimal_risk_mode_defensive(self, msc_ai):
        """Test risk mode switches to defensive in bad conditions."""
        analysis = MarketAnalysis(
            regime=MarketRegime.CHOPPY,
            volatility=0.9,
            trend_strength=0.2,
            correlation=0.8,
            liquidity_score=0.5,
            risk_score=0.8,
            recent_win_rate=0.35,
            recent_sharpe=0.5,
            recent_drawdown=-6.5,
            timestamp=datetime.utcnow(),
        )
        
        mode = msc_ai.determine_optimal_risk_mode(analysis)
        assert mode == RiskMode.DEFENSIVE
    
    @pytest.mark.asyncio
    async def test_update_policy(self, msc_ai, policy_store, event_bus, running_bus):
        """Test policy update flow."""
        events_received = []
        
        async def capture_event(event):
            events_received.append(event)
        
        event_bus.subscribe("policy.updated", capture_event)
        
        # Trigger update
        await msc_ai.update_policy("test")
        await asyncio.sleep(0.1)
        
        # Check policy was considered (may not change if already optimal)
        policy = await policy_store.get_policy()
        assert policy is not None
    
    @pytest.mark.asyncio
    async def test_health_alert_critical(self, msc_ai, event_bus, policy_store):
        """Test MSC AI reacts to CRITICAL health alert."""
        # Start bus
        task = asyncio.create_task(event_bus.run_forever())
        
        # Send critical health alert
        event = HealthStatusChangedEvent.create(
            old_status=HealthStatus.HEALTHY,
            new_status=HealthStatus.CRITICAL,
            component="DrawdownGuard",
            reason="DD exceeded",
        )
        
        await event_bus.publish(event)
        await asyncio.sleep(0.2)
        
        # Check emergency mode activated
        assert msc_ai._emergency_mode is True
        
        # Check policy switched to defensive
        policy = await policy_store.get_policy()
        assert policy.risk_mode == RiskMode.DEFENSIVE
        
        # Cleanup
        event_bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_force_defensive_mode(self, msc_ai, policy_store):
        """Test manual defensive mode override."""
        await msc_ai.force_defensive_mode("Manual override for testing")
        
        assert msc_ai._emergency_mode is True
        
        policy = await policy_store.get_policy()
        assert policy.risk_mode == RiskMode.DEFENSIVE
    
    @pytest.mark.asyncio
    async def test_clear_emergency_mode(self, msc_ai):
        """Test clearing emergency mode."""
        # Set emergency mode
        msc_ai._emergency_mode = True
        
        # Clear it
        await msc_ai.clear_emergency_mode()
        
        assert msc_ai._emergency_mode is False


@pytest.fixture
async def running_bus(event_bus):
    """Fixture that starts and stops event bus."""
    task = asyncio.create_task(event_bus.run_forever())
    yield event_bus
    event_bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
