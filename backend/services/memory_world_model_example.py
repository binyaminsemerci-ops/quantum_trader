"""Usage Example: Memory + World Model Integration.

This example demonstrates:
1. Storing trade episodes in memory
2. Generating memory summaries
3. Running scenario simulations with candidate policies
4. Using memory insights to inform decisions
"""

import asyncio
import logging
from datetime import datetime

import redis.asyncio as redis

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.memory.episodic_memory import EpisodeType
from backend.memory.memory_engine import MemoryEngine
from backend.memory.memory_retrieval import MemoryRetrieval
from backend.world_model.scenario_simulator import ScenarioSimulator, SimulationConfig
from backend.world_model.world_model import MarketRegime, MarketState, WorldModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_1_store_trade_episode():
    """Example 1: Store a trade episode in memory."""
    logger.info("=== Example 1: Store Trade Episode ===")
    
    # Initialize components
    redis_client = redis.Redis.from_url("redis://localhost:6379")
    event_bus = EventBus(redis_client, service_name="example")
    policy_store = PolicyStore(redis_client, event_bus)
    
    await event_bus.initialize()
    await policy_store.initialize()
    
    # Initialize memory engine
    memory = MemoryEngine(redis_client, event_bus, policy_store)
    await memory.initialize()
    await memory.start()
    
    # Store a trade episode
    episode = await memory.store_event(
        event_type=EpisodeType.TRADE,
        data={
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 43000,
            "exit_price": 43500,
            "size": 100,
            "duration_minutes": 45,
        },
        regime="TRENDING_UP",
        risk_mode="AGGRESSIVE_SMALL_ACCOUNT",
        global_mode="GROWTH",
        pnl=50.0,
        tags=["profitable", "btc", "scalp"],
    )
    
    logger.info(f"Stored episode: {episode.episode_id}")
    logger.info(f"  Type: {episode.episode_type.value}")
    logger.info(f"  PnL: ${episode.pnl:.2f}")
    logger.info(f"  Regime: {episode.regime}")
    logger.info(f"  Tags: {episode.tags}")
    
    # Query similar episodes
    similar = await memory.query_memory(
        episode_type=EpisodeType.TRADE,
        context={"regime": "TRENDING_UP"},
        days=7,
        limit=10,
    )
    
    logger.info(f"\nFound {len(similar)} similar trades in TRENDING_UP regime")
    
    await memory.stop()
    await event_bus.stop()
    await redis_client.close()


async def example_2_generate_memory_summary():
    """Example 2: Generate a memory summary."""
    logger.info("\n=== Example 2: Generate Memory Summary ===")
    
    # Initialize components
    redis_client = redis.Redis.from_url("redis://localhost:6379")
    event_bus = EventBus(redis_client, service_name="example")
    policy_store = PolicyStore(redis_client, event_bus)
    
    await event_bus.initialize()
    await policy_store.initialize()
    
    # Initialize memory
    memory = MemoryEngine(redis_client, event_bus, policy_store)
    await memory.initialize()
    await memory.start()
    
    # Initialize retrieval
    retrieval = MemoryRetrieval(
        redis_client,
        event_bus,
        memory.episodic,
        memory.semantic,
        memory.policy,
    )
    await retrieval.initialize()
    await retrieval.start()
    
    # Generate 7-day summary
    summary = await retrieval.generate_summary(
        days=7,
        include_patterns=True,
        include_policy_insights=True,
    )
    
    logger.info(f"Memory Summary ({summary.period_start.date()} to {summary.period_end.date()}):")
    logger.info(f"  Total trades: {summary.total_trades}")
    logger.info(f"  Profitable: {summary.profitable_trades}")
    logger.info(f"  Losses: {summary.loss_trades}")
    logger.info(f"  Total PnL: ${summary.total_pnl:.2f}")
    logger.info(f"  Avg PnL: ${summary.avg_pnl:.2f}")
    logger.info(f"  Win rate: {summary.profitable_trades / summary.total_trades * 100:.1f}%")
    logger.info(f"  Risk events: {summary.total_risk_events}")
    logger.info(f"  CEO decisions: {summary.total_ceo_decisions}")
    logger.info(f"  Mode switches: {summary.mode_switches}")
    
    logger.info("\n  Regime distribution:")
    for regime, count in summary.regime_distribution.items():
        logger.info(f"    {regime}: {count} trades")
    
    if summary.key_patterns:
        logger.info("\n  Key patterns:")
        for pattern in summary.key_patterns[:3]:
            logger.info(f"    - {pattern['description']} (confidence: {pattern['confidence']:.2f})")
    
    await retrieval.stop()
    await memory.stop()
    await event_bus.stop()
    await redis_client.close()


async def example_3_scenario_simulation():
    """Example 3: Run scenario simulation with candidate policies."""
    logger.info("\n=== Example 3: Scenario Simulation ===")
    
    # Define current market state
    current_state = MarketState(
        current_price=43000,
        current_regime=MarketRegime.TRENDING_UP,
        volatility=0.35,
        trend_strength=0.75,
        volume_ratio=1.2,
        sentiment_score=0.6,
    )
    
    logger.info("Current Market State:")
    logger.info(f"  Price: ${current_state.current_price:,.0f}")
    logger.info(f"  Regime: {current_state.current_regime.value}")
    logger.info(f"  Volatility: {current_state.volatility:.2f}")
    logger.info(f"  Trend Strength: {current_state.trend_strength:.2f}")
    
    # Define candidate policies
    current_policy = SimulationConfig(
        policy_id="CURRENT_GROWTH",
        global_mode="GROWTH",
        leverage=3.0,
        max_positions=5,
        position_size_pct=0.02,
        risk_per_trade_pct=0.015,
        max_drawdown_pct=0.15,
    )
    
    aggressive_policy = SimulationConfig(
        policy_id="AGGRESSIVE_EXPANSION",
        global_mode="EXPANSION",
        leverage=5.0,
        max_positions=8,
        position_size_pct=0.03,
        risk_per_trade_pct=0.02,
        max_drawdown_pct=0.20,
    )
    
    conservative_policy = SimulationConfig(
        policy_id="CONSERVATIVE_DEFENSIVE",
        global_mode="DEFENSIVE",
        leverage=2.0,
        max_positions=3,
        position_size_pct=0.015,
        risk_per_trade_pct=0.01,
        max_drawdown_pct=0.10,
    )
    
    # Initialize simulator
    simulator = ScenarioSimulator(seed=42)
    
    # Run simulations
    logger.info("\nRunning simulations (1000 paths)...")
    results = simulator.run_scenarios(
        current_state=current_state,
        candidate_policies=[current_policy, aggressive_policy, conservative_policy],
        num_paths=1000,
    )
    
    # Display results
    logger.info("\nSimulation Results:\n")
    
    for result in results:
        logger.info(f"Policy: {result.policy_id}")
        logger.info(f"  Expected PnL: ${result.expected_pnl:.2f}")
        logger.info(f"  Expected Return: {result.expected_return_pct:.2%}")
        logger.info(f"  PnL Std Dev: ${result.pnl_std:.2f}")
        logger.info(f"  Median PnL: ${result.pnl_median:.2f}")
        logger.info(f"  5th Percentile: ${result.pnl_p5:.2f}")
        logger.info(f"  95th Percentile: ${result.pnl_p95:.2f}")
        logger.info(f"  Worst Case PnL: ${result.worst_case_pnl:.2f}")
        logger.info(f"  Expected Drawdown: {result.expected_drawdown_pct:.2%}")
        logger.info(f"  Worst Case Drawdown: {result.worst_case_drawdown_pct:.2%}")
        logger.info(f"  P(Loss): {result.probability_of_loss:.2%}")
        logger.info(f"  P(Large Loss >5%): {result.probability_of_large_loss:.2%}")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
        logger.info("")
    
    # Compare policies
    logger.info("Policy Comparison:\n")
    
    for risk_tolerance in ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]:
        comparison = simulator.compare_policies(results, risk_tolerance=risk_tolerance)
        logger.info(f"{risk_tolerance} Risk Tolerance:")
        logger.info(f"  Recommendation: {comparison['recommendation']}")
        logger.info(f"  Reason: {comparison['reason']}")
        logger.info("")


async def example_4_ai_ceo_with_memory_and_scenarios():
    """Example 4: AI CEO using memory and scenarios for decision-making."""
    logger.info("\n=== Example 4: AI CEO with Memory + Scenarios ===")
    
    # Initialize components
    redis_client = redis.Redis.from_url("redis://localhost:6379")
    event_bus = EventBus(redis_client, service_name="example")
    policy_store = PolicyStore(redis_client, event_bus)
    
    await event_bus.initialize()
    await policy_store.initialize()
    
    # Initialize memory
    memory = MemoryEngine(redis_client, event_bus, policy_store)
    await memory.initialize()
    await memory.start()
    
    # Initialize retrieval
    retrieval = MemoryRetrieval(
        redis_client,
        event_bus,
        memory.episodic,
        memory.semantic,
        memory.policy,
    )
    await retrieval.initialize()
    
    logger.info("AI CEO Decision-Making Process:")
    
    # Step 1: Query memory for recent performance
    logger.info("\n1. Querying memory for recent performance...")
    
    recent_trades = await memory.query_memory(
        episode_type=EpisodeType.TRADE,
        days=7,
        limit=100,
    )
    
    if recent_trades:
        total_pnl = sum(t.pnl for t in recent_trades if t.pnl)
        win_rate = sum(1 for t in recent_trades if t.pnl and t.pnl > 0) / len(recent_trades)
        
        logger.info(f"  Recent 7 days: {len(recent_trades)} trades")
        logger.info(f"  Total PnL: ${total_pnl:.2f}")
        logger.info(f"  Win Rate: {win_rate:.1%}")
    
    # Step 2: Check for similar historical states
    logger.info("\n2. Checking policy memory for similar states...")
    
    similar_policies = await memory.policy.lookup_similar_policy_states(
        context={
            "regime": "TRENDING_UP",
            "volatility": 0.35,
            "trend_strength": 0.75,
        },
        days=30,
        limit=5,
    )
    
    logger.info(f"  Found {len(similar_policies)} similar historical policy states")
    
    if similar_policies:
        with_outcomes = [p for p in similar_policies if p.outcomes]
        if with_outcomes:
            best = max(with_outcomes, key=lambda p: p.outcomes.get("total_pnl", 0))
            logger.info(f"  Best performing: {best.global_mode} (leverage={best.leverage})")
    
    # Step 3: Get policy suggestions
    logger.info("\n3. Getting policy adjustment suggestions...")
    
    suggestions = await memory.policy.suggest_policy_adjustments(
        current_context={
            "regime": "TRENDING_UP",
            "volatility": 0.35,
        },
        lookback_days=30,
    )
    
    logger.info(f"  Confidence: {suggestions['confidence']:.2f}")
    logger.info(f"  Sample size: {suggestions['sample_size']}")
    
    for suggestion in suggestions.get("suggestions", []):
        logger.info(f"  - {suggestion['parameter']}: {suggestion['suggested_value']}")
        logger.info(f"    Reason: {suggestion['reason']}")
    
    # Step 4: Run scenario simulations
    logger.info("\n4. Running scenario simulations for candidate policies...")
    
    current_state = MarketState(
        current_price=43000,
        current_regime=MarketRegime.TRENDING_UP,
        volatility=0.35,
        trend_strength=0.75,
        volume_ratio=1.2,
    )
    
    current_policy = SimulationConfig(
        policy_id="CURRENT",
        global_mode="GROWTH",
        leverage=3.0,
        max_positions=5,
        position_size_pct=0.02,
        risk_per_trade_pct=0.015,
    )
    
    suggested_policy = SimulationConfig(
        policy_id="SUGGESTED",
        global_mode="EXPANSION",
        leverage=4.0,  # Suggested increase
        max_positions=6,
        position_size_pct=0.025,
        risk_per_trade_pct=0.018,
    )
    
    simulator = ScenarioSimulator(seed=42)
    
    results = simulator.run_scenarios(
        current_state=current_state,
        candidate_policies=[current_policy, suggested_policy],
        num_paths=500,
    )
    
    for result in results:
        logger.info(f"\n  Policy: {result.policy_id}")
        logger.info(f"    Expected PnL: ${result.expected_pnl:.2f}")
        logger.info(f"    Worst Case: ${result.worst_case_pnl:.2f}")
        logger.info(f"    P(Loss): {result.probability_of_loss:.2%}")
        logger.info(f"    Sharpe: {result.sharpe_ratio:.2f}")
    
    # Step 5: Make decision
    logger.info("\n5. AI CEO Decision:")
    
    comparison = simulator.compare_policies(results, risk_tolerance="MODERATE")
    
    logger.info(f"  Recommended Policy: {comparison['recommendation']}")
    logger.info(f"  Reason: {comparison['reason']}")
    logger.info(f"\n  Decision: Switch to {comparison['recommendation']} policy")
    logger.info("  - Memory shows similar states performed well with higher leverage")
    logger.info("  - Scenario simulation confirms positive expected value")
    logger.info("  - Risk metrics within acceptable bounds")
    
    await retrieval.stop()
    await memory.stop()
    await event_bus.stop()
    await redis_client.close()


async def main():
    """Run all examples."""
    try:
        await example_1_store_trade_episode()
        await asyncio.sleep(1)
        
        await example_2_generate_memory_summary()
        await asyncio.sleep(1)
        
        await example_3_scenario_simulation()
        await asyncio.sleep(1)
        
        await example_4_ai_ceo_with_memory_and_scenarios()
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
