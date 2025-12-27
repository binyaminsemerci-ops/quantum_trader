"""
Integration tests for RL v3 Live Orchestrator.
Tests all modes: OFF, SHADOW, PRIMARY, HYBRID.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
import uuid


class MockEventBus:
    """Mock EventBus for testing."""
    
    def __init__(self):
        self.published_events = []
        self.subscribers = {}
    
    async def publish(self, event_type: str, payload: Dict[str, Any], trace_id: str = None):
        """Mock publish."""
        self.published_events.append({
            "event_type": event_type,
            "payload": payload,
            "trace_id": trace_id,
        })
    
    async def subscribe(self, event_type: str, handler):
        """Mock subscribe."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def trigger_event(self, event_type: str, payload: Dict[str, Any]):
        """Trigger event for testing."""
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                await handler(payload)


class MockRLv3Manager:
    """Mock RLv3Manager for testing."""
    
    def __init__(self, action: int = 1, confidence: float = 0.8, value: float = 1.0):
        self.action = action
        self.confidence = confidence
        self.value = value
        self.predict_calls = []
    
    def predict(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Mock predict."""
        self.predict_calls.append(obs_dict)
        return {
            "action": self.action,
            "confidence": self.confidence,
            "value": self.value,
        }


class MockFeatureAdapter:
    """Mock RLv3LiveFeatureAdapter for testing."""
    
    async def build_observation(self, symbol: str, trace_id: str) -> Dict[str, Any]:
        """Mock build_observation."""
        return {f"feature_{i}": 0.0 for i in range(64)}


class MockPolicyStore:
    """Mock PolicyStore for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "enabled": True,
            "mode": "SHADOW",
            "min_confidence": 0.6,
            "max_trades_per_hour": 10,
        }
        self.risk_profile = Mock()
        self.risk_profile.max_leverage = 10
    
    async def get_policy(self):
        """Mock get_policy."""
        policy = Mock()
        policy.rl_v3_live = self.config
        return policy
    
    async def get_active_risk_profile(self):
        """Mock get_active_risk_profile."""
        return self.risk_profile


class MockRiskGuard:
    """Mock RiskGuard for testing."""
    
    def __init__(self, allow: bool = True, reason: str = ""):
        self.allow = allow
        self.reason = reason
        self.check_calls = []
    
    async def can_execute(self, symbol, notional, leverage, trade_risk_pct, position_size_usd, trace_id):
        """Mock can_execute."""
        self.check_calls.append({
            "symbol": symbol,
            "notional": notional,
            "leverage": leverage,
            "trade_risk_pct": trade_risk_pct,
            "position_size_usd": position_size_usd,
            "trace_id": trace_id,
        })
        return self.allow, self.reason


@pytest.fixture
def mock_metrics_store():
    """Mock metrics store."""
    with patch("backend.services.rl_v3_live_orchestrator.RLv3MetricsStore") as mock_cls:
        mock_instance = Mock()
        mock_instance.record_live_decision = Mock()
        mock_instance.record_trade_intent = Mock()
        mock_instance.get_live_status = Mock(return_value={
            "total_live_decisions": 0,
            "shadow_decisions": 0,
            "trade_intents_total": 0,
            "trade_intents_executed": 0,
            "last_decision_at": None,
            "last_trade_intent_at": None,
        })
        mock_cls.instance.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_orchestrator_mode_off(mock_metrics_store):
    """Test OFF mode - should not process events."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager()
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "OFF"})
    risk_guard = MockRiskGuard()
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.8,
        "trace_id": str(uuid.uuid4()),
    })
    
    # Should not make any predictions in OFF mode
    assert len(rl_manager.predict_calls) == 0
    
    # Should not publish any trade.intent events
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 0


@pytest.mark.asyncio
async def test_orchestrator_mode_shadow(mock_metrics_store):
    """Test SHADOW mode - should predict but not publish trade.intent."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=1, confidence=0.8)  # LONG_SMALL
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "SHADOW", "min_confidence": 0.6})
    risk_guard = MockRiskGuard()
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.8,
        "trace_id": str(uuid.uuid4()),
    })
    
    # Should make prediction
    await asyncio.sleep(0.1)  # Allow async processing
    assert len(rl_manager.predict_calls) == 1
    
    # Should record decision
    assert mock_metrics_store.record_live_decision.call_count == 1
    
    # Should NOT publish trade.intent
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 0


@pytest.mark.asyncio
async def test_orchestrator_mode_primary(mock_metrics_store):
    """Test PRIMARY mode - should predict and publish trade.intent."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=1, confidence=0.8)  # LONG_SMALL
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "PRIMARY", "min_confidence": 0.6})
    risk_guard = MockRiskGuard(allow=True)
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event
    trace_id = str(uuid.uuid4())
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.7,
        "trace_id": trace_id,
    })
    
    # Should make prediction
    await asyncio.sleep(0.1)
    assert len(rl_manager.predict_calls) == 1
    
    # Should record decision and trade intent
    assert mock_metrics_store.record_live_decision.call_count == 1
    assert mock_metrics_store.record_trade_intent.call_count == 1
    
    # Should publish trade.intent
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 1
    
    intent = trade_intents[0]["payload"]
    assert intent["symbol"] == "BTCUSDT"
    assert intent["side"] == "LONG"
    assert intent["source"] == "RL_V3_PRIMARY"
    assert intent["confidence"] == 0.8


@pytest.mark.asyncio
async def test_orchestrator_mode_hybrid(mock_metrics_store):
    """Test HYBRID mode - should combine RL and signal."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=1, confidence=0.9)  # LONG_SMALL, high confidence
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "HYBRID", "min_confidence": 0.6})
    risk_guard = MockRiskGuard(allow=True)
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event with lower confidence
    trace_id = str(uuid.uuid4())
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.7,
        "trace_id": trace_id,
    })
    
    await asyncio.sleep(0.1)
    
    # Should publish trade.intent
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 1
    
    intent = trade_intents[0]["payload"]
    assert intent["symbol"] == "BTCUSDT"
    assert intent["side"] == "LONG"
    # Should use RL since it has higher confidence
    assert "HYBRID" in intent["source"]
    assert intent["confidence"] == 0.9  # RL confidence


@pytest.mark.asyncio
async def test_orchestrator_confidence_threshold(mock_metrics_store):
    """Test confidence threshold filtering."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=1, confidence=0.5)  # Below threshold
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "PRIMARY", "min_confidence": 0.6})
    risk_guard = MockRiskGuard(allow=True)
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.8,
        "trace_id": str(uuid.uuid4()),
    })
    
    await asyncio.sleep(0.1)
    
    # Should make prediction but not publish (below threshold)
    assert len(rl_manager.predict_calls) == 1
    
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 0


@pytest.mark.asyncio
async def test_orchestrator_risk_guard_denial(mock_metrics_store):
    """Test RiskGuard denial prevents trade.intent."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=1, confidence=0.8)
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "PRIMARY", "min_confidence": 0.6})
    risk_guard = MockRiskGuard(allow=False, reason="Max leverage exceeded")
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.8,
        "trace_id": str(uuid.uuid4()),
    })
    
    await asyncio.sleep(0.1)
    
    # Should check with RiskGuard
    assert len(risk_guard.check_calls) == 1
    
    # Should NOT publish trade.intent (denied)
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 0


@pytest.mark.asyncio
async def test_orchestrator_rate_limiting(mock_metrics_store):
    """Test rate limiting prevents excessive trades."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=1, confidence=0.8)
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={
        "enabled": True,
        "mode": "PRIMARY",
        "min_confidence": 0.6,
        "max_trades_per_hour": 2,
    })
    risk_guard = MockRiskGuard(allow=True)
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger 3 events (should only execute 2 due to rate limit)
    for i in range(3):
        await event_bus.trigger_event("signal.generated", {
            "symbol": "BTCUSDT",
            "action": "LONG",
            "confidence": 0.8,
            "trace_id": str(uuid.uuid4()),
        })
        await asyncio.sleep(0.1)
    
    # Should only publish 2 trade.intent events
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 2


@pytest.mark.asyncio
async def test_orchestrator_hold_action(mock_metrics_store):
    """Test HOLD action does not publish trade.intent."""
    from backend.services.ai.rl_v3_live_orchestrator import RLv3LiveOrchestrator
    
    event_bus = MockEventBus()
    rl_manager = MockRLv3Manager(action=5, confidence=0.8)  # HOLD action
    feature_adapter = MockFeatureAdapter()
    policy_store = MockPolicyStore(config={"enabled": True, "mode": "PRIMARY", "min_confidence": 0.6})
    risk_guard = MockRiskGuard(allow=True)
    
    orchestrator = RLv3LiveOrchestrator(
        event_bus=event_bus,
        rl_manager=rl_manager,
        feature_adapter=feature_adapter,
        policy_store=policy_store,
        risk_guard=risk_guard,
    )
    
    await orchestrator.start()
    
    # Trigger signal event
    await event_bus.trigger_event("signal.generated", {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.8,
        "trace_id": str(uuid.uuid4()),
    })
    
    await asyncio.sleep(0.1)
    
    # Should make prediction but not publish (HOLD action)
    assert len(rl_manager.predict_calls) == 1
    
    trade_intents = [e for e in event_bus.published_events if e["event_type"] == "trade.intent"]
    assert len(trade_intents) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
