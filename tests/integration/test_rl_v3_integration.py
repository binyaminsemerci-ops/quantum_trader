"""
Integration tests for RL v3 with EventBus and API.
"""

import pytest
import asyncio
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.domains.learning.rl_v3.config_v3 import RLv3Config
from backend.core.event_bus import EventBus
from backend.events.event_types import EventType
from backend.events.subscribers.rl_subscriber_v3 import RLSubscriberV3


@pytest.mark.asyncio
async def test_rl_v3_subscriber_signal_handling():
    """Test RL v3 subscriber handles SIGNAL_GENERATED events."""
    # Create event bus
    event_bus = EventBusV2()
    
    # Create RL v3 subscriber
    config = RLv3Config()
    subscriber = RLSubscriberV3(event_bus, config, shadow_mode=True)
    
    # Test signal
    signal_data = {
        "symbol": "BTCUSDT",
        "price_change_1m": 0.001,
        "price_change_5m": 0.005,
        "price_change_15m": 0.01,
        "volatility": 0.02,
        "rsi": 55.0,
        "macd": 0.5,
        "position_size": 0.0,
        "position_side": 0.0,
        "balance": 10000.0,
        "equity": 10000.0,
        "regime": "TREND",
        "trend_strength": 0.7,
        "volume_ratio": 1.2,
        "spread": 0.001,
        "time_of_day": 0.5
    }
    
    # Collect published events
    published_events = []
    
    async def capture_event(event):
        published_events.append(event)
    
    event_bus.subscribe(EventType.RL_V3_DECISION, capture_event)
    
    # Publish signal
    await event_bus.publish(EventType.SIGNAL_GENERATED, signal_data)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify RL v3 decision was published
    assert len(published_events) > 0
    
    decision = published_events[0]['data']
    assert 'action' in decision
    assert 'confidence' in decision
    assert 'value' in decision
    assert 'shadow_mode' in decision
    assert decision['shadow_mode'] == True
    
    print(f"✅ RL v3 signal handling test passed: action={decision['action']}, confidence={decision['confidence']:.3f}")


@pytest.mark.asyncio
async def test_rl_v3_subscriber_experience_collection():
    """Test RL v3 subscriber collects experiences from position closures."""
    # Create event bus
    event_bus = EventBusV2()
    
    # Create RL v3 subscriber
    config = RLv3Config()
    subscriber = RLSubscriberV3(event_bus, config, shadow_mode=True)
    
    # Generate signal first
    signal_data = {
        "symbol": "BTCUSDT",
        "price_change_1m": 0.001,
        "volatility": 0.02,
        "rsi": 55.0,
        "position_size": 0.0,
        "balance": 10000.0,
        "equity": 10000.0
    }
    
    await event_bus.publish(EventType.SIGNAL_GENERATED, signal_data)
    await asyncio.sleep(0.1)
    
    # Simulate position closed
    position_data = {
        "symbol": "BTCUSDT",
        "pnl": 150.0,
        "roi": 0.015
    }
    
    initial_exp_count = len(subscriber.experiences)
    
    await event_bus.publish(EventType.POSITION_CLOSED, position_data)
    await asyncio.sleep(0.1)
    
    # Verify experience was collected
    assert len(subscriber.experiences) > initial_exp_count
    
    exp = subscriber.experiences[-1]
    assert 'observation' in exp
    assert 'action' in exp
    assert 'reward' in exp
    assert exp['pnl'] == 150.0
    
    print(f"✅ RL v3 experience collection test passed: {len(subscriber.experiences)} experiences")


@pytest.mark.asyncio
async def test_rl_v3_integration_full_flow():
    """Test complete RL v3 integration flow."""
    # Create event bus
    event_bus = EventBusV2()
    
    # Create RL v3 subscriber
    config = RLv3Config()
    subscriber = RLSubscriberV3(event_bus, config, shadow_mode=True)
    
    # Track decisions
    decisions = []
    
    async def capture_decision(event):
        decisions.append(event['data'])
    
    event_bus.subscribe(EventType.RL_V3_DECISION, capture_decision)
    
    # Simulate trading flow
    for i in range(5):
        # Generate signal
        signal = {
            "symbol": "BTCUSDT",
            "price_change_1m": 0.001 * (i + 1),
            "volatility": 0.02 + i * 0.001,
            "rsi": 50.0 + i * 2,
            "position_size": 0.1 if i > 0 else 0.0,
            "balance": 10000.0,
            "equity": 10000.0 + i * 50
        }
        
        await event_bus.publish(EventType.SIGNAL_GENERATED, signal)
        await asyncio.sleep(0.05)
        
        # Simulate position closed
        if i > 0:
            position = {
                "symbol": "BTCUSDT",
                "pnl": 50.0 + i * 10,
                "roi": 0.005 + i * 0.001
            }
            
            await event_bus.publish(EventType.POSITION_CLOSED, position)
            await asyncio.sleep(0.05)
    
    # Verify
    assert len(decisions) == 5
    assert len(subscriber.experiences) >= 4
    
    print(f"✅ RL v3 full flow test passed: {len(decisions)} decisions, {len(subscriber.experiences)} experiences")


def test_rl_v3_manager_standalone():
    """Test RL v3 manager works standalone without EventBus."""
    config = RLv3Config()
    manager = RLv3Manager(config)
    
    # Test prediction
    obs = {
        "price_change_1m": 0.001,
        "volatility": 0.02,
        "rsi": 55.0,
        "position_size": 0.0,
        "balance": 10000.0,
        "equity": 10000.0
    }
    
    result = manager.predict(obs)
    
    assert 'action' in result
    assert 'confidence' in result
    assert 'value' in result
    assert 0 <= result['action'] <= 5
    assert 0.0 <= result['confidence'] <= 1.0
    
    print(f"✅ RL v3 standalone test passed: action={result['action']}, confidence={result['confidence']:.3f}")


if __name__ == '__main__':
    print("Running RL v3 integration tests...\n")
    
    # Run standalone test
    test_rl_v3_manager_standalone()
    
    # Run async tests
    asyncio.run(test_rl_v3_subscriber_signal_handling())
    asyncio.run(test_rl_v3_subscriber_experience_collection())
    asyncio.run(test_rl_v3_integration_full_flow())
    
    print("\n✅ All RL v3 integration tests passed!")
