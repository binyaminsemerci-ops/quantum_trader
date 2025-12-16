"""
Integration test demonstrating XGBAgent signal generation with metadata.
Shows that the ML model integration is working end-to-end.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_xgb_agent_generates_signals_with_metadata():
    """Test that XGBAgent generates signals with proper source/model metadata"""
    from backend.routes.live_ai_signals import _agent_signals
    
    # Mock the agent to return actionable signals
    mock_agent = MagicMock()
    mock_agent.scan_top_by_volume_from_api = AsyncMock(return_value={
        "BTCUSDT": {
            "action": "BUY",
            "score": 0.75,
            "confidence": 0.75,
            "model": "ensemble"
        },
        "ETHUSDT": {
            "action": "SELL",
            "score": 0.65,
            "confidence": 0.65,
            "model": "xgboost"
        },
        "SOLUSDT": {
            "action": "BUY",
            "score": 0.55,
            "confidence": 0.55,
            "model": "ensemble"
        }
    })
    
    # Mock price fetching
    with patch("backend.routes.live_ai_signals._get_agent", AsyncMock(return_value=mock_agent)), \
         patch("backend.routes.live_ai_signals._fetch_latest_prices", AsyncMock(return_value={
             "BTCUSDT": 42000.0,
             "ETHUSDT": 2200.0,
             "SOLUSDT": 95.0
         })):
        
        signals = await _agent_signals(["BTCUSDT", "ETHUSDT", "SOLUSDT"], limit=10)
        
        # Verify we got signals
        assert len(signals) == 3
        
        # Verify metadata is present
        for sig in signals:
            assert "source" in sig
            assert sig["source"] == "XGBAgent"
            assert "model" in sig
            assert sig["model"] in ["ensemble", "xgboost"]
            assert "symbol" in sig
            assert sig["symbol"] in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            assert "side" in sig
            assert sig["side"] in ["buy", "sell"]
            assert "confidence" in sig
            assert sig["confidence"] >= 0.55


@pytest.mark.asyncio
async def test_heuristic_signals_have_metadata():
    """Test that heuristic fallback signals have proper metadata"""
    from backend.routes.live_ai_signals import SimpleAITrader
    import pandas as pd
    
    trader = SimpleAITrader()
    
    # Mock Binance data with strong buy signal
    mock_df = pd.DataFrame({
        "close": [100] * 45 + list(range(101, 111)),  # Strong uptrend
        "volume": [1000] * 55
    })
    
    with patch.object(trader, "get_binance_data", AsyncMock(return_value=mock_df)):
        signals = await trader.generate_signals(["TESTUSDT"], limit=5)
        
        assert len(signals) > 0
        
        for sig in signals:
            assert "source" in sig
            assert sig["source"] == "LiveAIHeuristic"
            assert "model" in sig
            assert sig["model"] == "technical"
            assert "details" in sig
            assert "source" in sig["details"]


@pytest.mark.asyncio
async def test_get_live_ai_signals_prioritizes_agent():
    """Test that get_live_ai_signals prioritizes XGBAgent over heuristics"""
    from backend.routes.live_ai_signals import get_live_ai_signals
    
    # Mock agent returning strong signals
    mock_agent = MagicMock()
    mock_agent.scan_top_by_volume_from_api = AsyncMock(return_value={
        "BTCUSDT": {"action": "BUY", "score": 0.8, "confidence": 0.8, "model": "ensemble"},
        "ETHUSDT": {"action": "SELL", "score": 0.7, "confidence": 0.7, "model": "ensemble"}
    })
    
    with patch("backend.routes.live_ai_signals._get_agent", AsyncMock(return_value=mock_agent)), \
         patch("backend.routes.live_ai_signals._fetch_latest_prices", AsyncMock(return_value={
             "BTCUSDT": 42000.0, "ETHUSDT": 2200.0
         })):
        
        signals = await get_live_ai_signals(limit=5, profile="mixed")
        
        # Should get agent signals
        assert len(signals) >= 2
        
        # Verify they're from the agent
        agent_signals = [s for s in signals if s.get("source") == "XGBAgent"]
        assert len(agent_signals) >= 2


@pytest.mark.asyncio
async def test_metadata_propagation_through_api():
    """Test that metadata flows correctly through the API normalization"""
    from backend.main import _normalise_signals
    
    # Simulate raw signals from XGBAgent
    raw_signals = [
        {
            "id": "xgb_BTCUSDT_123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "BTCUSDT",
            "side": "buy",
            "score": 0.75,
            "confidence": 0.75,
            "price": 42000.0,
            "source": "XGBAgent",
            "model": "ensemble",
            "details": {
                "source": "XGBAgent",
                "note": "ensemble"
            }
        },
        {
            "id": "ai_ETHUSDT_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": "ETHUSDT",
            "side": "sell",
            "score": 0.35,
            "confidence": 0.35,
            "price": 2200.0,
            "source": "LiveAIHeuristic",
            "model": "technical",
            "details": {
                "source": "Live AI Analysis",
                "note": "RSI: 68.5, Overbought"
            }
        }
    ]
    
    normalized = _normalise_signals(raw_signals, 10)
    
    assert len(normalized) == 2
    
    # Check XGBAgent signal
    xgb_sig = normalized[0]
    assert xgb_sig["symbol"] == "BTCUSDT"
    assert xgb_sig["type"] == "BUY"
    assert xgb_sig["source"] == "XGBAgent"
    assert xgb_sig["model"] == "ensemble"
    assert xgb_sig["confidence"] == 0.75
    
    # Check heuristic signal
    heur_sig = normalized[1]
    assert heur_sig["symbol"] == "ETHUSDT"
    assert heur_sig["type"] == "SELL"
    assert heur_sig["source"] == "Live AI Analysis"  # From details.source
    assert heur_sig["model"] == "technical"
    assert heur_sig["confidence"] == 0.35


@pytest.mark.asyncio
async def test_trading_bot_prioritizes_agent_signals():
    """Test that trading bot processes XGBAgent signals before heuristics"""
    from backend.trading_bot.autonomous_trader import AutonomousTradingBot
    
    bot = AutonomousTradingBot(
        balance=10000.0,
        dry_run=True
    )
    
    # Mix of agent and heuristic signals
    mixed_signals = [
        {"symbol": "BTCUSDT", "side": "buy", "confidence": 0.3, "source": "LiveAIHeuristic"},
        {"symbol": "ETHUSDT", "side": "sell", "confidence": 0.7, "source": "XGBAgent"},
        {"symbol": "SOLUSDT", "side": "buy", "confidence": 0.5, "source": "LiveAIHeuristic"},
        {"symbol": "ADAUSDT", "side": "sell", "confidence": 0.6, "source": "XGBAgent"},
    ]
    
    prioritized = bot._prioritize_signals(mixed_signals)
    
    # Agent signals should come first
    assert prioritized[0]["source"] == "XGBAgent"
    assert prioritized[1]["source"] == "XGBAgent"
    assert prioritized[2]["source"] == "LiveAIHeuristic"
    assert prioritized[3]["source"] == "LiveAIHeuristic"


def test_xgb_model_loads_successfully():
    """Test that XGBoost model loads without errors"""
    from ai_engine.agents.xgb_agent import make_default_agent
    
    agent = make_default_agent()
    
    assert agent is not None
    assert hasattr(agent, "scan_top_by_volume_from_api")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
