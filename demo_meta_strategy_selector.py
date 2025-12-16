"""
Demo: Meta-Strategy Selector System

Demonstrates the complete Meta-Strategy Selector workflow:
1. Market regime detection
2. Strategy selection via RL
3. TP/SL configuration
4. Reward updates and learning

Author: Quantum Trader Team
Date: 2025-11-26
"""

import asyncio
import json
import random
from pathlib import Path

# Import meta-strategy components
from backend.services.ai.strategy_profiles import (
    list_all_strategies,
    get_strategy_profile,
    StrategyID,
)
from backend.services.ai.regime_detector import (
    RegimeDetector,
    MarketContext,
    MarketRegime,
)
from backend.services.ai.meta_strategy_selector import (
    MetaStrategySelector,
)
from backend.services.meta_strategy_integration import (
    MetaStrategyIntegration,
)


def print_header(title: str):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_strategy_profiles():
    """Demo: View all available strategies."""
    print_header("📊 AVAILABLE TRADING STRATEGIES")
    
    strategies = list_all_strategies()
    
    print(f"{'Strategy':<25} {'SL':<6} {'TP1':<6} {'TP2':<6} {'TP3':<6} {'R:R':<6} {'WR%':<6}")
    print("-" * 70)
    
    for sid, info in strategies.items():
        if sid == "none":
            continue
        
        print(
            f"{info['name']:<25} "
            f"{info['sl']:<6} "
            f"{info['tp1']:<6} "
            f"{info['tp2']:<6} "
            f"{info['tp3']:<6} "
            f"{info['risk_reward']:<6.1f} "
            f"{info['win_rate']*100:<6.0f}"
        )
    
    print(f"\n✅ Currently using: {get_strategy_profile(StrategyID.ULTRA_AGGRESSIVE).name}")


def demo_regime_detection():
    """Demo: Detect market regimes."""
    print_header("🔍 MARKET REGIME DETECTION")
    
    detector = RegimeDetector()
    
    # Test cases
    test_cases = [
        {
            "name": "BTC Strong Uptrend",
            "context": MarketContext(
                symbol="BTCUSDT",
                atr_pct=0.025,
                trend_strength=0.7,
                adx=45.0,
                volume_24h=80_000_000,
                depth_5bps=1_000_000,
                spread_bps=2.0,
            )
        },
        {
            "name": "ETH Range-Bound Low Vol",
            "context": MarketContext(
                symbol="ETHUSDT",
                atr_pct=0.012,
                trend_strength=0.1,
                adx=18.0,
                volume_24h=40_000_000,
                depth_5bps=500_000,
                spread_bps=2.5,
            )
        },
        {
            "name": "ALT High Volatility",
            "context": MarketContext(
                symbol="ALTUSDT",
                atr_pct=0.08,
                trend_strength=0.3,
                adx=28.0,
                volume_24h=5_000_000,
                depth_5bps=100_000,
                spread_bps=8.0,
            )
        },
        {
            "name": "SHITCOIN Illiquid",
            "context": MarketContext(
                symbol="SHITUSDT",
                atr_pct=0.05,
                trend_strength=0.2,
                adx=22.0,
                volume_24h=500_000,
                depth_5bps=20_000,
                spread_bps=25.0,
            )
        },
    ]
    
    for case in test_cases:
        result = detector.detect_regime(case["context"])
        
        print(f"📍 {case['name']}:")
        print(f"   Regime: {result.regime.value.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Reasoning: {result.reasoning}")
        print()


async def demo_strategy_selection():
    """Demo: Select strategies via RL."""
    print_header("🧠 META-STRATEGY SELECTOR (RL)")
    
    selector = MetaStrategySelector(epsilon=0.15, alpha=0.25)
    
    # Simulate 20 trades to show learning
    print("Simulating 20 trades to demonstrate learning...\n")
    
    for i in range(20):
        # Vary between trending and ranging
        if i % 3 == 0:
            regime = MarketRegime.RANGE_LOW_VOL
            context = MarketContext(
                symbol="BTCUSDT",
                atr_pct=0.015,
                trend_strength=0.1,
                adx=20.0,
            )
        else:
            regime = MarketRegime.TREND_UP
            context = MarketContext(
                symbol="BTCUSDT",
                atr_pct=0.03,
                trend_strength=0.6,
                adx=35.0,
            )
        
        # Choose strategy
        decision = selector.choose_strategy(
            symbol="BTCUSDT",
            regime=regime,
            context=context,
        )
        
        # Simulate reward (ultra_aggressive works better in trends)
        if regime == MarketRegime.TREND_UP:
            if decision.strategy_id == StrategyID.ULTRA_AGGRESSIVE:
                reward = random.uniform(2.0, 5.0)  # Better performance
            else:
                reward = random.uniform(0.5, 2.5)
        else:  # Range
            if decision.strategy_id == StrategyID.SCALP:
                reward = random.uniform(0.8, 2.0)  # Better in range
            elif decision.strategy_id == StrategyID.ULTRA_AGGRESSIVE:
                reward = random.uniform(-1.0, 1.5)  # Worse in range
            else:
                reward = random.uniform(0.5, 2.0)
        
        # Update RL
        selector.update_reward(
            symbol="BTCUSDT",
            regime=regime,
            strategy_id=decision.strategy_id,
            reward=reward,
        )
        
        if (i + 1) % 5 == 0:
            print(f"✅ Trade {i+1}: {decision.strategy_profile.name} → R={reward:+.2f}")
    
    print("\n📊 LEARNING RESULTS:")
    summary = selector.get_performance_summary()
    
    print(f"   Total decisions: {summary['total_decisions']}")
    print(f"   Total updates: {summary['total_updates']}")
    print(f"   Exploration rate: {summary['exploration_rate']:.1%}")
    
    print(f"\n   Top Performing Strategies:")
    for strat in summary['best_strategies'][:5]:
        print(
            f"      {strat['strategy']:<20} (regime={strat['regime']:<15}) "
            f"EMA={strat['ema_reward']:+.2f}R  WR={strat['win_rate']:.0%}  "
            f"N={strat['count']}"
        )


async def demo_full_integration():
    """Demo: Complete integration workflow."""
    print_header("🚀 FULL META-STRATEGY INTEGRATION")
    
    integration = MetaStrategyIntegration(
        enabled=True,
        epsilon=0.10,
        alpha=0.20,
    )
    
    # Simulate AI signal with market data
    signal = {
        "symbol": "BTCUSDT",
        "action": "BUY",
        "confidence": 0.75,
        "price": 100000.0,
        "model": "ensemble",
    }
    
    market_data = {
        "atr": 500.0,
        "atr_pct": 0.005,
        "sma_20": 99500.0,
        "sma_50": 98000.0,
        "ema_12": 99800.0,
        "ema_26": 99200.0,
        "adx": 38.0,
        "volume_24h": 75_000_000,
        "avg_volume_24h": 60_000_000,
        "depth_5bps": 800_000,
        "spread_bps": 2.3,
        "funding_rate": 0.0001,
        "open_interest": 5_000_000_000,
    }
    
    print("📥 AI Signal Received:")
    print(f"   Symbol: {signal['symbol']}")
    print(f"   Action: {signal['action']}")
    print(f"   Confidence: {signal['confidence']:.1%}")
    print(f"   Price: ${signal['price']:,.0f}")
    
    # Select strategy
    result = await integration.select_strategy_for_signal(
        symbol=signal["symbol"],
        signal=signal,
        market_data=market_data,
    )
    
    print(f"\n🔍 Detected Regime: {result.regime.value.upper()}")
    print(f"   ATR: {result.context.atr_pct:.2%}")
    print(f"   Trend Strength: {result.context.trend_strength:+.2f}")
    print(f"   ADX: {result.context.adx:.1f}")
    
    print(f"\n🎯 Selected Strategy: {result.strategy.name}")
    print(f"   SL: {result.strategy.r_sl:.1f}R")
    print(f"   TP1: {result.strategy.r_tp1:.1f}R (close {result.strategy.partial_close_tp1:.0%})")
    print(f"   TP2: {result.strategy.r_tp2:.1f}R (close {result.strategy.partial_close_tp2:.0%})")
    print(f"   TP3: {result.strategy.r_tp3:.1f}R (close {result.strategy.partial_close_tp3:.0%})")
    print(f"   Expected R:R: {result.strategy.expected_risk_reward:.1f}")
    print(f"   Expected WR: {result.strategy.expected_win_rate:.0%}")
    
    print(f"\n📋 TP/SL Configuration:")
    for key, value in result.tpsl_config.items():
        print(f"   {key}: {value}")
    
    print(f"\n💡 Decision Reasoning:")
    print(f"   Exploration: {result.decision.is_exploration}")
    print(f"   Confidence: {result.decision.confidence:.1%}")
    print(f"   {result.decision.reasoning}")
    
    # Simulate trade execution and close
    print(f"\n⏳ Simulating trade execution...")
    await asyncio.sleep(1)
    
    # Simulate profit (hit TP2 at 5R)
    realized_r = 5.0
    pnl = 125.0
    duration_hours = 3.5
    
    print(f"\n✅ Trade Closed:")
    print(f"   Realized R: {realized_r:+.2f}R")
    print(f"   PnL: ${pnl:+.2f}")
    print(f"   Duration: {duration_hours:.1f} hours")
    
    # Update RL reward
    await integration.update_strategy_reward(
        symbol=signal["symbol"],
        realized_r=realized_r,
        trade_meta={
            "pnl": pnl,
            "duration_hours": duration_hours,
            "exit_reason": "TP2",
        }
    )
    
    print(f"\n📊 Integration Metrics:")
    metrics = integration.get_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("  🧠 META-STRATEGY SELECTOR SYSTEM DEMO")
    print("  Quantum Trader - AI Hedge Fund OS")
    print("="*70)
    
    # Demo 1: Strategy profiles
    demo_strategy_profiles()
    
    input("\n[Press Enter to continue...]")
    
    # Demo 2: Regime detection
    demo_regime_detection()
    
    input("\n[Press Enter to continue...]")
    
    # Demo 3: Strategy selection with RL
    await demo_strategy_selection()
    
    input("\n[Press Enter to continue...]")
    
    # Demo 4: Full integration
    await demo_full_integration()
    
    print_header("✅ DEMO COMPLETE")
    
    print("Key Takeaways:")
    print("  1. ✅ 7 strategy profiles available (defensive → ultra aggressive)")
    print("  2. ✅ Market regime detection (trending, ranging, volatile, illiquid)")
    print("  3. ✅ RL-based strategy selection (epsilon-greedy with Q-learning)")
    print("  4. ✅ Full integration ready for EventDrivenExecutor")
    print("  5. ✅ Persistent learning (survives restarts)")
    print("\n🚀 Ready to maximize profits with AI-driven strategies!")


if __name__ == "__main__":
    asyncio.run(main())
