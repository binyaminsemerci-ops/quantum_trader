"""
Meta-Strategy Parameter Tuner

Automatically tunes epsilon and alpha based on performance metrics.
This script analyzes Q-learning performance and updates .env with optimal parameters.
"""
import os
from pathlib import Path
import sys
from typing import Tuple

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.services.meta_strategy_integration import get_meta_strategy_integration


def calculate_optimal_epsilon(
    total_updates: int,
    q_table_size: int,
    convergence_ratio: float
) -> float:
    """
    Calculate optimal epsilon based on learning progress.
    
    Args:
        total_updates: Total number of reward updates
        q_table_size: Number of Q-table entries
        convergence_ratio: Ratio of min/max strategy counts (0-1)
    
    Returns:
        Optimal epsilon value
    """
    # Phase 1: Cold Start (< 20 updates) - High exploration
    if total_updates < 20:
        return 0.20  # 20% exploration
    
    # Phase 2: Learning (20-50 updates) - Moderate exploration
    elif total_updates < 50:
        return 0.10  # 10% exploration
    
    # Phase 3: Convergence (50-100 updates) - Lower exploration
    elif total_updates < 100:
        # Check convergence - if strategies converging well, reduce epsilon
        if convergence_ratio < 0.3:  # Good convergence
            return 0.05  # 5% exploration
        else:
            return 0.08  # 8% exploration (still learning)
    
    # Phase 4: Mature (100+ updates) - Minimal exploration
    else:
        # Fine-tune based on convergence
        if convergence_ratio < 0.2:  # Excellent convergence
            return 0.03  # 3% exploration (mostly exploit)
        elif convergence_ratio < 0.4:  # Good convergence
            return 0.05  # 5% exploration
        else:
            return 0.07  # 7% exploration (some patterns unclear)


def calculate_optimal_alpha(
    ema_rewards: list,
    market_volatility: str = "NORMAL"
) -> float:
    """
    Calculate optimal alpha based on reward stability.
    
    Args:
        ema_rewards: List of EMA rewards from Q-table
        market_volatility: Market volatility level (LOW/NORMAL/HIGH)
    
    Returns:
        Optimal alpha value
    """
    if len(ema_rewards) < 5:
        return 0.20  # Default for insufficient data
    
    import statistics
    mean_reward = statistics.mean(ema_rewards)
    stdev_reward = statistics.stdev(ema_rewards) if len(ema_rewards) > 1 else 0
    
    # Calculate coefficient of variation (CV)
    cv = stdev_reward / abs(mean_reward) if mean_reward != 0 else float('inf')
    
    # Adjust alpha based on stability
    if cv < 0.3:
        # Stable rewards - can use higher alpha (adapt faster)
        base_alpha = 0.25
    elif cv < 0.5:
        # Moderate stability - balanced alpha
        base_alpha = 0.20
    else:
        # High volatility - use lower alpha (more stability)
        base_alpha = 0.15
    
    # Adjust for market volatility
    if market_volatility == "HIGH":
        base_alpha *= 0.8  # More conservative in volatile markets
    elif market_volatility == "LOW":
        base_alpha *= 1.1  # Slightly more aggressive in stable markets
    
    # Clamp to reasonable range [0.10, 0.30]
    return max(0.10, min(0.30, base_alpha))


def update_env_file(epsilon: float, alpha: float, dry_run: bool = False) -> bool:
    """
    Update .env file with new epsilon and alpha values.
    
    Args:
        epsilon: New epsilon value
        alpha: New alpha value
        dry_run: If True, only show changes without applying
    
    Returns:
        True if successful
    """
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print(f"❌ .env file not found at {env_path}")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Find and update lines
    updated_lines = []
    epsilon_found = False
    alpha_found = False
    
    for line in lines:
        if line.startswith("META_STRATEGY_EPSILON="):
            old_value = line.split('=')[1].strip()
            updated_lines.append(f"META_STRATEGY_EPSILON={epsilon:.2f}\n")
            epsilon_found = True
            print(f"   EPSILON: {old_value} → {epsilon:.2f}")
        elif line.startswith("META_STRATEGY_ALPHA="):
            old_value = line.split('=')[1].strip()
            updated_lines.append(f"META_STRATEGY_ALPHA={alpha:.2f}\n")
            alpha_found = True
            print(f"   ALPHA:   {old_value} → {alpha:.2f}")
        else:
            updated_lines.append(line)
    
    if not epsilon_found or not alpha_found:
        print("❌ META_STRATEGY_EPSILON or META_STRATEGY_ALPHA not found in .env")
        return False
    
    if dry_run:
        print("\n⚠️  DRY RUN - No changes applied")
        return True
    
    # Write updated .env
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    print("\n✅ .env file updated successfully")
    print("⚠️  Restart backend to apply changes: docker-compose --profile dev restart backend")
    return True


def main():
    """Main tuning function"""
    print("=" * 80)
    print("  META-STRATEGY PARAMETER TUNER")
    print("=" * 80 + "\n")
    
    # Load integration
    try:
        integration = get_meta_strategy_integration()
    except Exception as e:
        print(f"❌ Error loading Meta-Strategy Integration: {e}")
        return
    
    # Get current metrics
    metrics = integration.get_metrics()
    
    print("Current Configuration:")
    print(f"   Epsilon: {metrics['epsilon']:.2f}")
    print(f"   Alpha:   {metrics['alpha']:.2f}")
    print(f"   Total Updates: {metrics['total_reward_updates']}")
    print(f"   Q-Table Size: {len(integration.meta_selector.q_table)}")
    
    if metrics['total_reward_updates'] < 10:
        print("\n⚠️  Too few updates (< 10) - wait for more trading activity")
        print("   Run this script after at least 10 trades complete")
        return
    
    # Calculate convergence ratio
    q_table = integration.meta_selector.q_table
    strategy_counts = {}
    for key, stats in q_table.items():
        symbol, regime, strategy = key
        if strategy not in strategy_counts:
            strategy_counts[strategy] = 0
        strategy_counts[strategy] += stats.count
    
    convergence_ratio = 0
    if strategy_counts:
        max_count = max(strategy_counts.values())
        min_count = min(strategy_counts.values())
        convergence_ratio = min_count / max_count if max_count > 0 else 0
    
    # Get EMA rewards
    ema_rewards = [stats.ema_reward for stats in q_table.values() if stats.count >= 3]
    
    # Calculate optimal parameters
    print("\n" + "─" * 80)
    print("Calculating Optimal Parameters...")
    print("─" * 80 + "\n")
    
    optimal_epsilon = calculate_optimal_epsilon(
        total_updates=metrics['total_reward_updates'],
        q_table_size=len(q_table),
        convergence_ratio=convergence_ratio
    )
    
    optimal_alpha = calculate_optimal_alpha(
        ema_rewards=ema_rewards,
        market_volatility="NORMAL"  # TODO: Detect from market data
    )
    
    print("Analysis:")
    print(f"   Convergence Ratio: {convergence_ratio:.2f}")
    print(f"   EMA Samples: {len(ema_rewards)}")
    
    if ema_rewards:
        import statistics
        cv = statistics.stdev(ema_rewards) / abs(statistics.mean(ema_rewards)) if statistics.mean(ema_rewards) != 0 else 0
        print(f"   Reward CV: {cv:.2f}")
    
    print("\nRecommended Configuration:")
    print(f"   Epsilon: {optimal_epsilon:.2f} (Exploration Rate)")
    print(f"   Alpha:   {optimal_alpha:.2f} (EMA Smoothing)")
    
    # Check if changes needed
    epsilon_change = abs(optimal_epsilon - metrics['epsilon']) > 0.01
    alpha_change = abs(optimal_alpha - metrics['alpha']) > 0.01
    
    if not epsilon_change and not alpha_change:
        print("\n✅ Current parameters are optimal - no changes needed")
        return
    
    print("\n" + "─" * 80)
    print("Proposed Changes:")
    print("─" * 80 + "\n")
    
    # Ask for confirmation
    response = input("Apply these changes to .env? (y/n/dry): ").strip().lower()
    
    if response == 'y':
        success = update_env_file(optimal_epsilon, optimal_alpha, dry_run=False)
        if success:
            print("\n📊 Next Steps:")
            print("   1. Restart backend: docker-compose --profile dev restart backend")
            print("   2. Monitor performance for 1 week")
            print("   3. Run this script again to fine-tune")
    elif response == 'dry':
        print("\n" + "─" * 80)
        print("DRY RUN - Showing proposed changes:")
        print("─" * 80 + "\n")
        update_env_file(optimal_epsilon, optimal_alpha, dry_run=True)
    else:
        print("\n❌ Changes cancelled")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
