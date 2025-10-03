#!/usr/bin/env python3
"""Enhanced Multi-Source Data Integration - Final Demo
Shows the complete system in action.
"""

import asyncio

from backend.enhanced_data_feeds import get_enhanced_market_data


async def final_demo() -> None:

    # Test the complete enhanced system
    data = await get_enhanced_market_data(["BTC", "ETH", "ADA"])
    insights = data.get("ai_insights", {})

    # Show active data sources
    sources = [k for k, v in data.items() if isinstance(v, dict) and v.get("source")]
    for _i, _source in enumerate(sources, 1):
        pass

    # Show AI intelligence

    # Show market regime
    regime = insights.get("regime_indicators", {})
    if regime:
        pass

    # Show Fear & Greed
    fear_greed = data.get("fear_greed", {}).get("current", {})
    if fear_greed:
        pass

    # Show Reddit sentiment
    reddit_data = data.get("reddit", {}).get("symbols", {})
    if reddit_data:
        for sentiment_info in reddit_data.values():
            sentiment_info.get("sentiment_score", 0)
            sentiment_info.get("total_posts", 0)

    # System status


if __name__ == "__main__":
    asyncio.run(final_demo())
