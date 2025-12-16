"""
Meta-Strategy Q-Learning Performance Monitor

Monitors Q-learning performance and provides recommendations for parameter tuning.
Run this script weekly to track learning progress and adjust epsilon/alpha.
"""
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List
import sys

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.services.meta_strategy_integration import get_meta_strategy_integration


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}\n")


def analyze_exploration_rate(metrics: Dict) -> str:
    """Analyze exploration rate and provide recommendations"""
    total_decisions = metrics.get("total_selections", 0)
    if total_decisions == 0:
        return "❌ No decisions made yet - system needs more time"
    
    # Calculate exploration rate from Q-table
    integration = get_meta_strategy_integration()
    q_table = integration.meta_selector.q_table
    
    if not q_table:
        return "❌ Q-table empty - system in cold start phase"
    
    # Count decisions per strategy
    strategy_counts = {}
    for key, stats in q_table.items():
        symbol, regime, strategy = key
        count = stats.count
        if strategy not in strategy_counts:
            strategy_counts[strategy] = 0
        strategy_counts[strategy] += count
    
    total_updates = sum(strategy_counts.values())
    if total_updates < 20:
        return "⚠️  Too few updates (<20) - continue with current epsilon (10%)"
    
    # Check if strategies are converging
    max_count = max(strategy_counts.values())
    min_count = min(strategy_counts.values())
    convergence_ratio = min_count / max_count if max_count > 0 else 0
    
    if convergence_ratio < 0.3:
        return "✅ Good convergence - strategies emerging. Consider reducing epsilon to 0.05"
    else:
        return "⚠️  High convergence ratio - may need more exploration. Keep epsilon at 0.10"


def analyze_learning_stability(q_table: Dict) -> str:
    """Analyze EMA stability and provide alpha recommendations"""
    if not q_table:
        return "❌ No Q-table data"
    
    # Calculate variance in EMA rewards
    ema_rewards = [stats.ema_reward for stats in q_table.values() if stats.count >= 3]
    
    if len(ema_rewards) < 5:
        return "⚠️  Too few samples (<5) - keep alpha at 0.20"
    
    import statistics
    mean_reward = statistics.mean(ema_rewards)
    stdev_reward = statistics.stdev(ema_rewards) if len(ema_rewards) > 1 else 0
    
    cv = stdev_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
    
    if cv < 0.3:
        return "✅ Stable learning - alpha=0.20 working well"
    elif cv < 0.5:
        return "⚠️  Moderate volatility - consider lowering alpha to 0.15"
    else:
        return "❌ High volatility - lower alpha to 0.10 for more stability"


def get_regime_distribution(q_table: Dict) -> Dict[str, int]:
    """Get distribution of decisions per regime"""
    regime_counts = {}
    for key, stats in q_table.items():
        symbol, regime, strategy = key
        if regime not in regime_counts:
            regime_counts[regime] = 0
        regime_counts[regime] += stats.count
    return regime_counts


def get_strategy_distribution(q_table: Dict) -> Dict[str, int]:
    """Get distribution of decisions per strategy"""
    strategy_counts = {}
    for key, stats in q_table.items():
        symbol, regime, strategy = key
        if strategy not in strategy_counts:
            strategy_counts[strategy] = 0
        strategy_counts[strategy] += stats.count
    return strategy_counts


async def main():
    """Main monitoring function"""
    print_header("📊 META-STRATEGY Q-LEARNING PERFORMANCE MONITOR")
    
    # Load Meta-Strategy Integration
    try:
        integration = get_meta_strategy_integration()
    except Exception as e:
        print(f"❌ Error loading Meta-Strategy Integration: {e}")
        return
    
    # Get metrics
    metrics = integration.get_metrics()
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: SYSTEM STATUS
    # ═══════════════════════════════════════════════════════════════════
    print_section("1. System Status")
    
    print(f"Enabled: {'✅ YES' if metrics['enabled'] else '❌ NO'}")
    print(f"Epsilon (Exploration Rate): {metrics['epsilon']:.0%}")
    print(f"Alpha (EMA Smoothing): {metrics['alpha']:.0%}")
    print(f"\nTotal Selections: {metrics['total_selections']}")
    print(f"Total Reward Updates: {metrics['total_reward_updates']}")
    print(f"Active Strategies: {metrics['active_strategies']}")
    
    if metrics['total_selections'] == 0:
        print("\n⚠️  No trading activity yet - system waiting for signals")
        return
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: Q-TABLE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print_section("2. Q-Table Analysis")
    
    q_table = integration.meta_selector.q_table
    print(f"Q-Table Entries: {len(q_table)}")
    
    if len(q_table) == 0:
        print("❌ Q-table empty - no learning has occurred yet")
        print("   Wait for at least 5-10 trades to complete")
        return
    
    # Regime distribution
    regime_dist = get_regime_distribution(q_table)
    print(f"\n📍 Regime Distribution:")
    for regime, count in sorted(regime_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   {regime:20s}: {count:3d} decisions")
    
    # Strategy distribution
    strategy_dist = get_strategy_distribution(q_table)
    print(f"\n🎯 Strategy Distribution:")
    for strategy, count in sorted(strategy_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"   {strategy:25s}: {count:3d} decisions")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: TOP PERFORMING STRATEGIES
    # ═══════════════════════════════════════════════════════════════════
    print_section("3. Top Performing Strategies (by EMA Reward)")
    
    summary = integration.get_performance_summary()
    best_strategies = summary.get("best_strategies", [])
    
    if not best_strategies:
        print("❌ No performance data yet")
    else:
        print(f"{'Rank':<5} {'Symbol':<12} {'Regime':<15} {'Strategy':<25} {'EMA R':<8} {'WR':<6} {'N':<4} {'Total R':<8}")
        print("─" * 120)
        
        for i, strat in enumerate(best_strategies[:20], 1):
            symbol = strat['symbol']
            regime = strat['regime']
            strategy = strat['strategy']
            ema_r = strat['ema_reward']
            wr = strat['win_rate']
            n = strat['count']
            total_r = strat['total_r']
            
            # Color coding
            if ema_r > 2.0:
                emoji = "🏆"
            elif ema_r > 1.0:
                emoji = "✅"
            elif ema_r > 0:
                emoji = "⚠️ "
            else:
                emoji = "❌"
            
            print(f"{emoji} {i:<3} {symbol:<12} {regime:<15} {strategy:<25} {ema_r:>+6.2f}R {wr:>5.0%} {n:>3} {total_r:>+7.1f}R")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: LEARNING METRICS
    # ═══════════════════════════════════════════════════════════════════
    print_section("4. Learning Metrics")
    
    # Calculate actual exploration rate
    if metrics['total_selections'] > 0:
        # Note: We don't track exploration/exploitation separately yet, so estimate
        print(f"Configured Epsilon: {metrics['epsilon']:.0%}")
        print(f"Expected Exploration Rate: ~{metrics['epsilon']:.0%}")
        print(f"Expected Exploitation Rate: ~{1-metrics['epsilon']:.0%}")
    
    # Count strategies with sufficient data
    strategies_with_data = sum(1 for stats in q_table.values() if stats.count >= 5)
    print(f"\nStrategies with ≥5 samples: {strategies_with_data}/{len(q_table)}")
    
    # Calculate average EMA reward
    ema_rewards = [stats.ema_reward for stats in q_table.values() if stats.count >= 3]
    if ema_rewards:
        import statistics
        mean_ema = statistics.mean(ema_rewards)
        median_ema = statistics.median(ema_rewards)
        stdev_ema = statistics.stdev(ema_rewards) if len(ema_rewards) > 1 else 0
        
        print(f"\nEMA Reward Statistics (N={len(ema_rewards)}):")
        print(f"   Mean:   {mean_ema:+.2f}R")
        print(f"   Median: {median_ema:+.2f}R")
        print(f"   StdDev: {stdev_ema:.2f}R")
        print(f"   CV:     {stdev_ema/abs(mean_ema):.2f}" if mean_ema != 0 else "   CV:     N/A")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════
    print_section("5. Parameter Tuning Recommendations")
    
    # Exploration rate recommendation
    print("🔍 Exploration Rate (Epsilon):")
    exploration_rec = analyze_exploration_rate(metrics)
    print(f"   {exploration_rec}")
    
    # Learning stability recommendation
    print("\n📊 Learning Stability (Alpha):")
    stability_rec = analyze_learning_stability(q_table)
    print(f"   {stability_rec}")
    
    # Convergence analysis
    print("\n📈 Convergence Status:")
    total_updates = metrics['total_reward_updates']
    
    if total_updates < 20:
        print("   ⏳ Cold Start Phase (< 20 updates)")
        print("      → Keep current parameters")
        print("      → Wait for 20+ trades to complete")
    elif total_updates < 50:
        print("   🔄 Learning Phase (20-50 updates)")
        print("      → Monitor Q-values weekly")
        print("      → Watch for strategy patterns")
    elif total_updates < 100:
        print("   📊 Convergence Phase (50-100 updates)")
        print("      → Strategies should be emerging")
        print("      → Consider reducing epsilon to 0.05")
    else:
        print("   ✅ Mature Phase (100+ updates)")
        print("      → System has learned optimal strategies")
        print("      → Fine-tune epsilon (0.03-0.05) for exploitation")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 6: REGIME-SPECIFIC ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print_section("6. Regime-Specific Best Strategies")
    
    # Group by regime and find best strategy per regime
    regime_best = {}
    for key, stats in q_table.items():
        symbol, regime, strategy = key
        if stats.count < 3:  # Skip insufficient data
            continue
        
        if regime not in regime_best:
            regime_best[regime] = []
        regime_best[regime].append({
            "strategy": strategy,
            "ema_reward": stats.ema_reward,
            "count": stats.count,
            "win_rate": stats.get_win_rate()
        })
    
    for regime in sorted(regime_best.keys()):
        strategies = sorted(regime_best[regime], key=lambda x: x["ema_reward"], reverse=True)
        best = strategies[0] if strategies else None
        
        if best:
            print(f"\n{regime}:")
            print(f"   Best: {best['strategy']:25s} | EMA R={best['ema_reward']:+.2f} | WR={best['win_rate']:.0%} | N={best['count']}")
            
            if len(strategies) > 1:
                print(f"   Alternatives:")
                for alt in strategies[1:4]:  # Show top 3 alternatives
                    print(f"      • {alt['strategy']:25s} | EMA R={alt['ema_reward']:+.2f} | WR={alt['win_rate']:.0%} | N={alt['count']}")
    
    # ═══════════════════════════════════════════════════════════════════
    # SECTION 7: ACTION ITEMS
    # ═══════════════════════════════════════════════════════════════════
    print_section("7. Recommended Actions")
    
    actions = []
    
    # Check if need to adjust epsilon
    if total_updates >= 50 and metrics['epsilon'] > 0.05:
        actions.append("✓ Reduce epsilon to 0.05 in .env (more exploitation)")
    
    # Check if need to adjust alpha
    if len(ema_rewards) >= 10:
        cv = stdev_ema / abs(mean_ema) if mean_ema != 0 else 0
        if cv > 0.5 and metrics['alpha'] > 0.15:
            actions.append("✓ Reduce alpha to 0.15 in .env (more stability)")
    
    # Check if ready for production
    if total_updates >= 100:
        actions.append("✓ System mature - ready for full production use")
    
    if not actions:
        actions.append("✓ No actions needed - continue monitoring")
    
    for action in actions:
        print(f"   {action}")
    
    # ═══════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"  Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Next Review: Run this script weekly for 4 weeks")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
