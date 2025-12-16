"""
Tests for Market Opportunity Ranker.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict

from backend.services.eventbus import InMemoryEventBus, OpportunitiesUpdatedEvent
from backend.services.opportunity_ranker import (
    MarketOpportunityRanker,
    SymbolScore,
    RankingCriteria,
)


@pytest.fixture
async def running_bus(eventbus):
    """Create and start an EventBus, cleanup after test."""
    task = asyncio.create_task(eventbus.run_forever())
    await asyncio.sleep(0.05)  # Let it start
    yield eventbus
    eventbus.stop()
    await asyncio.sleep(0.05)  # Let it stop


@pytest.fixture
def eventbus():
    return InMemoryEventBus()


@pytest.fixture
def ranker(eventbus):
    criteria = RankingCriteria(
        min_volume=1e9,
        min_liquidity_score=0.5,
        trend_weight=0.35,
        volatility_weight=0.25,
        liquidity_weight=0.20,
        performance_weight=0.20,
    )
    return MarketOpportunityRanker(eventbus, criteria)


def create_market_data(
    volume_24h: float = 2e9,
    atr: float = 2.0,
    trend_strength: float = 0.8,
    volatility: float = 2.0,
    spread: float = 0.001,
    recent_return: float = 0.05,
) -> Dict:
    """Helper to create market data dict."""
    return {
        "volume_24h": volume_24h,
        "atr": atr,
        "trend_strength": trend_strength,
        "volatility": volatility,
        "spread": spread,
        "recent_return": recent_return,
    }


@pytest.mark.asyncio
async def test_score_symbol_strong_opportunity(ranker):
    """Test scoring a symbol with strong opportunity."""
    data = create_market_data(
        volume_24h=5e9,
        atr=2.0,
        trend_strength=0.8,
        spread=0.001,
        recent_return=0.1,
    )
    
    score = await ranker.score_symbol("BTCUSDT", data)
    
    assert score is not None
    assert score.symbol == "BTCUSDT"
    assert score.total_score > 0.7  # Should be high
    assert score.trend_score == 1.0  # Strong trend
    assert score.volatility_score == 1.0  # Ideal volatility
    assert score.liquidity_score >= 0.5  # Good liquidity
    assert score.performance_score >= 0.8  # Strong performance


@pytest.mark.asyncio
async def test_score_symbol_low_volume_rejected(ranker):
    """Test that low volume symbols are rejected."""
    data = create_market_data(volume_24h=5e8)  # 500M < 1B minimum
    
    score = await ranker.score_symbol("LOWVOL", data)
    
    assert score is None


@pytest.mark.asyncio
async def test_score_symbol_low_liquidity_rejected(ranker):
    """Test that low liquidity symbols are rejected."""
    data = create_market_data(
        volume_24h=5e8,  # Below minimum volume
        spread=0.1,  # High spread = low liquidity
    )
    
    score = await ranker.score_symbol("ILLIQUID", data)
    
    assert score is None


@pytest.mark.asyncio
async def test_calculate_trend_score(ranker):
    """Test trend score calculation."""
    assert ranker._calculate_trend_score(0.8) == 1.0
    assert ranker._calculate_trend_score(-0.7) == 1.0
    assert ranker._calculate_trend_score(0.5) == 0.7
    assert ranker._calculate_trend_score(0.3) == 0.4
    assert ranker._calculate_trend_score(0.1) == 0.1


@pytest.mark.asyncio
async def test_calculate_volatility_score(ranker):
    """Test volatility score calculation."""
    assert ranker._calculate_volatility_score(2.0, 2.0) == 1.0  # Ideal
    assert ranker._calculate_volatility_score(3.0, 3.0) == 1.0  # Still ideal
    assert ranker._calculate_volatility_score(0.7, 0.7) == 0.7  # Low but ok
    assert ranker._calculate_volatility_score(0.3, 0.3) == 0.3  # Too low
    assert ranker._calculate_volatility_score(10.0, 10.0) == 0.3  # Too high


@pytest.mark.asyncio
async def test_calculate_liquidity_score(ranker):
    """Test liquidity score calculation."""
    # High volume + tight spread
    assert ranker._calculate_liquidity_score(1e10, 0.0005) == 1.0
    
    # Good volume + tight spread (0.6 + 0.3 = 0.9)
    score = ranker._calculate_liquidity_score(5e9, 0.001)
    assert abs(score - 0.9) < 0.01  # Allow floating point tolerance
    
    # Minimum volume + ok spread
    score = ranker._calculate_liquidity_score(1e9, 0.003)
    assert 0.5 <= score <= 0.7


@pytest.mark.asyncio
async def test_calculate_performance_score(ranker):
    """Test performance score calculation."""
    assert ranker._calculate_performance_score(0.15) == 1.0  # 15% gain
    assert ranker._calculate_performance_score(0.08) == 0.8  # 8% gain
    assert ranker._calculate_performance_score(0.03) == 0.6  # 3% gain
    assert ranker._calculate_performance_score(0.01) == 0.4  # 1% gain
    assert ranker._calculate_performance_score(-0.05) == 0.1  # 5% loss


@pytest.mark.asyncio
async def test_rank_all_symbols(ranker):
    """Test ranking multiple symbols."""
    symbols_data = {
        "BTCUSDT": create_market_data(
            volume_24h=1e10,
            trend_strength=0.9,
            recent_return=0.12,
        ),
        "ETHUSDT": create_market_data(
            volume_24h=8e9,
            trend_strength=0.7,
            recent_return=0.08,
        ),
        "SOLUSDT": create_market_data(
            volume_24h=3e9,
            trend_strength=0.5,
            recent_return=0.04,
        ),
        "LOWVOL": create_market_data(
            volume_24h=5e8,  # Too low
        ),
    }
    
    scores = await ranker.rank_all_symbols(symbols_data)
    
    # Should exclude LOWVOL
    assert len(scores) == 3
    
    # Should be sorted by total_score
    assert scores[0].symbol == "BTCUSDT"  # Highest score
    assert scores[0].total_score > scores[1].total_score
    assert scores[1].total_score > scores[2].total_score


@pytest.mark.asyncio
async def test_get_top_n_opportunities(ranker):
    """Test getting top N opportunities."""
    symbols_data = {
        f"SYMBOL{i}": create_market_data(
            volume_24h=1e9 + i * 1e9,
            trend_strength=0.5 + i * 0.05,
            recent_return=0.02 + i * 0.01,
        )
        for i in range(20)
    }
    
    await ranker.rank_all_symbols(symbols_data)
    
    top_5 = ranker.get_top_n_opportunities(5)
    assert len(top_5) == 5
    
    top_10 = ranker.get_top_n_opportunities(10)
    assert len(top_10) == 10


@pytest.mark.asyncio
async def test_publish_rankings(running_bus):
    """Test publishing rankings to EventBus."""
    # Create ranker with running bus
    criteria = RankingCriteria()
    ranker = MarketOpportunityRanker(running_bus, criteria)
    
    published_events = []
    
    async def event_handler(event):
        published_events.append(event)
    
    # Subscribe by event type string, not class
    running_bus.subscribe("opportunities.updated", event_handler)
    
    # Create rankings
    symbols_data = {
        "BTCUSDT": create_market_data(volume_24h=1e10),
        "ETHUSDT": create_market_data(volume_24h=8e9),
    }
    await ranker.rank_all_symbols(symbols_data)
    
    # Publish
    await ranker.publish_rankings()
    
    # Wait for event to be processed
    await asyncio.sleep(0.1)
    
    assert len(published_events) == 1
    event = published_events[0]
    assert event.type == "opportunities.updated"
    assert len(event.payload["top_symbols"]) >= 2
    assert "criteria" in event.payload


@pytest.mark.asyncio
async def test_symbol_score_calculate():
    """Test SymbolScore.calculate factory method."""
    criteria = RankingCriteria(
        trend_weight=0.4,
        volatility_weight=0.3,
        liquidity_weight=0.2,
        performance_weight=0.1,
    )
    
    score = SymbolScore.calculate(
        symbol="BTCUSDT",
        trend_score=1.0,
        volatility_score=0.8,
        liquidity_score=0.9,
        performance_score=0.7,
        criteria=criteria,
        volume_24h=1e10,
        atr=2.0,
        trend_strength=0.85,
    )
    
    expected_total = 1.0 * 0.4 + 0.8 * 0.3 + 0.9 * 0.2 + 0.7 * 0.1
    
    assert score.symbol == "BTCUSDT"
    assert abs(score.total_score - expected_total) < 0.01
    assert score.trend_score == 1.0
    assert score.volatility_score == 0.8
    assert score.liquidity_score == 0.9
    assert score.performance_score == 0.7


@pytest.mark.asyncio
async def test_ranker_with_custom_criteria():
    """Test ranker with custom criteria weights."""
    eventbus = InMemoryEventBus()
    criteria = RankingCriteria(
        min_volume=5e9,  # Higher minimum
        trend_weight=0.5,  # Prioritize trend
        volatility_weight=0.1,
        liquidity_weight=0.3,
        performance_weight=0.1,
    )
    ranker = MarketOpportunityRanker(eventbus, criteria)
    
    symbols_data = {
        "HIGHVOL": create_market_data(
            volume_24h=6e9,
            trend_strength=0.9,
        ),
        "LOWVOL": create_market_data(
            volume_24h=2e9,  # Below minimum
            trend_strength=0.9,
        ),
    }
    
    scores = await ranker.rank_all_symbols(symbols_data)
    
    # Should only include HIGHVOL
    assert len(scores) == 1
    assert scores[0].symbol == "HIGHVOL"
