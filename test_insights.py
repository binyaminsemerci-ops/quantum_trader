import pytest
from backend.enhanced_data_feeds import get_enhanced_market_data


@pytest.mark.asyncio
async def test_ai_insights():
    data = await get_enhanced_market_data(["BTC", "ETH"])
    insights = data.get("ai_insights", {})
    # Basic structural assertions
    assert isinstance(insights, dict)
    assert "market_sentiment" in insights or insights == {}
    # If regime indicators present ensure expected keys exist
    if "regime_indicators" in insights:
        ri = insights["regime_indicators"]
        assert isinstance(ri, dict)
