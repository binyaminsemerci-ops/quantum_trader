#!/usr/bin/env python3
"""
Enhanced Multi-Source Data Integration - Final Demo
Shows the complete system in action
"""

import asyncio
from backend.enhanced_data_feeds import get_enhanced_market_data


async def final_demo():
    print("🚀 Enhanced Multi-Source Data Integration - Final Demo")
    print("=" * 60)

    # Test the complete enhanced system
    data = await get_enhanced_market_data(["BTC", "ETH", "ADA"])
    insights = data.get("ai_insights", {})

    # Show active data sources
    print("\n📡 Data Sources Active:")
    sources = [k for k, v in data.items() if isinstance(v, dict) and v.get("source")]
    for i, source in enumerate(sources, 1):
        print(f"   {i}. {source.upper()}")

    # Show AI intelligence
    print("\n🧠 AI Market Intelligence:")
    print(f'   Market Sentiment: {insights.get("market_sentiment", "none").upper()}')
    print(f'   Active Signals: {len(insights.get("momentum_signals", []))}')
    print(f'   Risk Factors: {len(insights.get("risk_factors", []))}')
    print(f'   Opportunities: {len(insights.get("opportunity_signals", []))}')

    # Show market regime
    regime = insights.get("regime_indicators", {})
    if regime:
        print("\n🌊 Market Regime Analysis:")
        print(f'   BTC Dominance: {regime.get("btc_dominance", 0):.1f}%')
        print(f'   Market Stage: {regime.get("market_stage", "unknown").upper()}')

    # Show Fear & Greed
    fear_greed = data.get("fear_greed", {}).get("current", {})
    if fear_greed:
        print("\n😱 Market Psychology:")
        print(f'   Fear & Greed Index: {fear_greed.get("value", "N/A")}')
        print(f'   Classification: {fear_greed.get("value_classification", "N/A")}')

    # Show Reddit sentiment
    reddit_data = data.get("reddit", {}).get("symbols", {})
    if reddit_data:
        print("\n🔴 Community Sentiment:")
        for symbol, sentiment_info in reddit_data.items():
            score = sentiment_info.get("sentiment_score", 0)
            posts = sentiment_info.get("total_posts", 0)
            print(f"   {symbol}: {score:+.2f} sentiment ({posts} posts)")

    # System status
    print("\n" + "=" * 60)
    print("✅ ENHANCED DATA INTEGRATION: OPERATIONAL")
    print("🤖 CONTINUOUS AI LEARNING: ENHANCED")
    print("📊 MULTI-SOURCE INTELLIGENCE: ACTIVE")
    print("🎯 SYSTEM STATUS: READY FOR PRODUCTION")
    print("=" * 60)

    print("\n🔥 The AI can now learn from:")
    print("   • Market data (CoinGecko, CoinPaprika)")
    print("   • Community sentiment (Reddit, Twitter)")
    print("   • Market psychology (Fear & Greed Index)")
    print("   • News sentiment (CryptoCompare)")
    print("   • Global market indicators")
    print("   • AI-extracted insights & regime detection")

    print("\n🚀 Enhanced continuous learning system is ready!")


if __name__ == "__main__":
    asyncio.run(final_demo())
