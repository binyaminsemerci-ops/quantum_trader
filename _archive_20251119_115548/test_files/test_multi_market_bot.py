"""
Test script for Multi-Market Autonomous Trading Bot
Tests SPOT, FUTURES, MARGIN trading across Layer 1/2 tokens
"""

import asyncio
import logging
from backend.trading_bot.autonomous_trader import AutonomousTradingBot
from backend.trading_bot.market_config import (
    get_trading_pairs,
    get_volume_weighted_pairs,
    is_layer1_layer2_token,
    get_optimal_market_for_token,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_multi_market_bot():
    """Test the multi-market trading bot functionality"""

    logger.info("🔬 TESTING MULTI-MARKET AUTONOMOUS TRADING BOT")
    logger.info("=" * 60)

    # Test market configurations
    logger.info("[CHART] Testing market configurations...")

    for market in ["SPOT", "FUTURES", "MARGIN", "CROSS_MARGIN"]:
        pairs = get_volume_weighted_pairs(market, 10)
        logger.info(f"{market}: {len(pairs)} trading pairs - {pairs[:5]}...")

    # Test token classification
    logger.info("\n🏷️  Testing Layer 1/2 token classification...")
    test_tokens = ["BTC", "ETH", "MATIC", "DOGE", "SHIB", "RANDOMTOKEN"]

    for token in test_tokens:
        is_l1l2 = is_layer1_layer2_token(token)
        optimal_market = get_optimal_market_for_token(token)
        logger.info(f"{token}: Layer1/2={is_l1l2}, Optimal Market={optimal_market}")

    # Initialize multi-market bot
    logger.info("\n🤖 Initializing Multi-Market Trading Bot...")

    bot = AutonomousTradingBot(
        balance=50000.0,  # $50k split across markets
        risk_per_trade=0.001,  # 0.1% risk per trade (smaller for testing)
        min_confidence=0.5,  # Higher confidence threshold
        dry_run=True,  # SAFE: Dry run mode
        enabled_markets=["SPOT", "FUTURES"],  # Test with 2 markets
    )

    # Display bot configuration
    logger.info(f"[MONEY] Total Balance: ${bot.balance:,.2f}")
    logger.info(f"[CHART_UP] Enabled Markets: {bot.enabled_markets}")
    logger.info(f"💵 Market Balances:")
    for market, balance in bot.market_balances.items():
        logger.info(f"   {market}: ${balance:,.2f}")

    # Test trading pairs for each market
    logger.info(f"\n[TARGET] Trading Pairs Configuration:")
    for market, pairs in bot.trading_pairs.items():
        logger.info(f"   {market}: {len(pairs)} pairs")

    # Create mock AI signals for testing
    mock_signals = [
        {
            "symbol": "BTCUSDC",
            "side": "buy",
            "confidence": 0.85,
            "timestamp": "2025-10-04T19:30:00",
            "reason": "Strong bullish momentum detected",
        },
        {
            "symbol": "ETHUSDC",
            "side": "buy",
            "confidence": 0.72,
            "timestamp": "2025-10-04T19:30:00",
            "reason": "Layer 2 scaling adoption increasing",
        },
        {
            "symbol": "MATICUSDC",
            "side": "sell",
            "confidence": 0.68,
            "timestamp": "2025-10-04T19:30:00",
            "reason": "Overbought conditions on technical indicators",
        },
    ]

    # Process mock signals
    logger.info(f"\n⚡ Processing {len(mock_signals)} mock AI signals...")

    for signal in mock_signals:
        logger.info(
            f"\n[SIGNAL] Processing signal: {signal['symbol']} {signal['side']} (confidence: {signal['confidence']})"
        )
        await bot._process_signal(signal)

    # Display final positions
    logger.info(f"\n[CHART] Final Positions Summary:")
    total_positions = 0

    for market_type in bot.enabled_markets:
        positions = bot.positions[market_type]
        logger.info(f"   {market_type}: {len(positions)} positions")

        for symbol, position in positions.items():
            logger.info(
                f"     • {symbol}: {position['side']} {position['qty']} @ ${position['entry_price']:.2f}"
            )
            total_positions += 1

    logger.info(f"\n[OK] Test completed - Total positions opened: {total_positions}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_multi_market_bot())
