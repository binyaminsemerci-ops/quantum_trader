
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

import pytest

from backend.enhanced_data_feeds import get_enhanced_market_data


@pytest.mark.asyncio()
async def test_ai_insights() -> None:
    data = await get_enhanced_market_data(["BTC", "ETH"])
    insights = data.get("ai_insights", {})
    # Basic structural assertions
    assert isinstance(insights, dict)
    assert "market_sentiment" in insights or insights == {}
    # If regime indicators present ensure expected keys exist
    if "regime_indicators" in insights:
        ri = insights["regime_indicators"]
        assert isinstance(ri, dict)
