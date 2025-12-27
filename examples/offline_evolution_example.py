"""
SESA + Meta-Learning + Federation v2 - Offline Evolution Example

This example demonstrates:
1. SESA Evolution: Generate and evaluate strategy mutations
2. Meta-Learning OS: Adjust hyperparameters based on performance
3. Federated Intelligence v2: Create Global Action Plan
4. Full integration: Memory Engine, Scenario Simulator, Event Bus

Usage:
    python examples/offline_evolution_example.py
"""

import asyncio
import logging
from datetime import datetime

# SESA Components
from backend.sesa.mutation_operators import MutationOperators, StrategyConfig
from backend.sesa.evaluation_engine import EvaluationEngine
from backend.sesa.selection_engine import SelectionEngine, SelectionCriteria
from backend.sesa.sesa_engine import SESAEngine, EvolutionConfig

# Meta-Learning Components
from backend.meta_learning.meta_os import MetaLearningOS, MetaLearningConfig
from backend.meta_learning.meta_policy import MetaPolicy, MetaPolicyRules

# Federation Components
from backend.federation.federated_engine_v2 import FederatedEngineV2, FederationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """
    Run complete offline evolution demonstration.
    """
    logger.info("=" * 80)
    logger.info("QUANTUM TRADER V5 - PROMPT 9C DEMONSTRATION")
    logger.info("Self-Evolving Strategy Architect + Meta-Learning OS + Federation v2")
    logger.info("=" * 80)
    
    # ============================================================================
    # STEP 1: Initialize Components
    # ============================================================================
    logger.info("\n[STEP 1] Initializing components...")
    
    # SESA Components
    mutation_ops = MutationOperators()
    evaluation_engine = EvaluationEngine()
    selection_engine = SelectionEngine(
        criteria=SelectionCriteria(
            top_k=5,
            min_composite_score=60.0,
            shadow_score_threshold=65.0,
            production_score_threshold=80.0,
        )
    )
    
    sesa_engine = SESAEngine(
        mutation_operators=mutation_ops,
        evaluation_engine=evaluation_engine,
        selection_engine=selection_engine,
    )
    
    logger.info("✓ SESA Engine initialized")
    
    # Meta-Learning OS
    meta_policy = MetaPolicy(
        rules=MetaPolicyRules(
            sesa_trigger_interval_hours=24.0,
            rl_retrain_interval_hours=12.0,
            shadow_min_test_days=7,
        )
    )
    
    meta_os = MetaLearningOS(
        config=MetaLearningConfig(
            meta_update_interval_minutes=60.0,
            adapt_rl_hyperparameters=True,
            adapt_strategy_weights=True,
        ),
        meta_policy=meta_policy,
    )
    
    logger.info("✓ Meta-Learning OS initialized")
    
    # Federation V2
    federation = FederatedEngineV2(
        config=FederationConfig(
            gap_update_interval_seconds=30.0,
            enable_sesa_integration=True,
            enable_meta_learning_integration=True,
        ),
        meta_learning_os=meta_os,
        sesa_engine=sesa_engine,
    )
    
    logger.info("✓ Federated Engine V2 initialized")
    
    # ============================================================================
    # STEP 2: Create Baseline Strategy
    # ============================================================================
    logger.info("\n[STEP 2] Creating baseline strategy...")
    
    baseline_strategy = StrategyConfig(
        strategy_id="baseline_v1",
        entry_confidence_threshold=0.70,
        exit_confidence_threshold=0.50,
        min_signal_strength=0.65,
        timeframe_minutes=15,
        lookback_periods=20,
        cooldown_periods=5,
        min_volatility=0.01,
        max_volatility=0.50,
        volatility_multiplier=1.0,
        risk_reward_ratio=2.5,
        min_rr_ratio=1.5,
        max_rr_ratio=5.0,
        use_dynamic_tpsl=True,
        base_tp_pct=0.02,
        base_sl_pct=0.01,
        trailing_stop_enabled=True,
        trailing_stop_activation_pct=0.015,
        rl_exploration_rate=0.15,
        rl_learning_rate=0.001,
        rl_discount_factor=0.95,
        position_size_pct=0.02,
        max_concurrent_positions=3,
    )
    
    logger.info(f"✓ Baseline strategy created: {baseline_strategy.strategy_id}")
    logger.info(f"  - Entry threshold: {baseline_strategy.entry_confidence_threshold}")
    logger.info(f"  - Risk/Reward: {baseline_strategy.risk_reward_ratio}")
    logger.info(f"  - Timeframe: {baseline_strategy.timeframe_minutes}m")
    
    # ============================================================================
    # STEP 3: Run SESA Evolution (Multi-Generation)
    # ============================================================================
    logger.info("\n[STEP 3] Running SESA evolution (3 generations)...")
    
    evolution_config = EvolutionConfig(
        num_mutations_per_parent=10,
        mutation_rate=0.30,
        mutation_magnitude=0.20,
        evaluation_type="backtest",
        evaluation_lookback_days=30,
        max_concurrent_evaluations=5,
        num_generations=3,
        elite_carry_forward=2,
        use_memory_engine=False,  # Offline mode
        publish_events=False,  # Offline mode
    )
    
    evolution_result = await sesa_engine.run_evolution(
        initial_strategies=[baseline_strategy],
        config=evolution_config,
    )
    
    logger.info(f"\n✓ Evolution complete:")
    logger.info(f"  - Evolution ID: {evolution_result.evolution_id}")
    logger.info(f"  - Generations: {len(evolution_result.generations)}")
    logger.info(f"  - Total mutations evaluated: {evolution_result.total_mutations_evaluated}")
    logger.info(f"  - Best strategy: {evolution_result.best_strategy_id}")
    logger.info(f"  - Best score: {evolution_result.best_score:.2f}")
    logger.info(f"  - Production candidates: {len(evolution_result.production_candidates)}")
    logger.info(f"  - Shadow candidates: {len(evolution_result.shadow_candidates)}")
    logger.info(f"  - Duration: {evolution_result.total_duration_seconds:.1f}s")
    
    # Show generation breakdown
    logger.info("\n  Generation Details:")
    for gen in evolution_result.generations:
        logger.info(
            f"    Gen {gen.generation_number}: "
            f"{gen.mutations_generated} mutations → "
            f"{gen.evaluations_completed} evaluated → "
            f"Best score: {gen.best_score:.2f} "
            f"({gen.duration_seconds:.1f}s)"
        )
    
    # ============================================================================
    # STEP 4: Simulate Trading Performance (for Meta-Learning)
    # ============================================================================
    logger.info("\n[STEP 4] Simulating trading performance...")
    
    # Simulate 100 trades with varying performance
    simulated_trades = []
    for i in range(100):
        # 55% win rate with some variance
        is_win = (i % 10 < 6)  # ~60% wins
        pnl = 150 if is_win else -80
        
        simulated_trades.append({
            "trade_id": f"trade_{i+1}",
            "pnl": pnl,
            "is_win": is_win,
            "timestamp": datetime.now(),
        })
    
    total_pnl = sum(t["pnl"] for t in simulated_trades)
    win_rate = sum(1 for t in simulated_trades if t["is_win"]) / len(simulated_trades)
    
    logger.info(f"✓ Simulated {len(simulated_trades)} trades:")
    logger.info(f"  - Total PnL: ${total_pnl:.2f}")
    logger.info(f"  - Win Rate: {win_rate:.1%}")
    
    # ============================================================================
    # STEP 5: Run Meta-Learning Update
    # ============================================================================
    logger.info("\n[STEP 5] Running meta-learning update...")
    
    meta_state = await meta_os.run_meta_update(
        recent_trades=simulated_trades,
        recent_evaluations=[
            {
                "strategy_id": f"strategy_{i}",
                "composite_score": 70 + (i * 5),
            }
            for i in range(5)
        ],
    )
    
    logger.info(f"\n✓ Meta-learning update complete:")
    logger.info(f"  - Update count: {meta_state.update_count}")
    logger.info(f"  - RL Learning Rate: {meta_state.rl_learning_rate:.5f}")
    logger.info(f"  - RL Exploration Rate: {meta_state.rl_exploration_rate:.3f}")
    logger.info(f"  - Risk Multiplier: {meta_state.risk_multiplier:.2f}")
    logger.info(f"  - Recent Sharpe: {meta_state.recent_sharpe:.2f}")
    logger.info(f"  - Recent Win Rate: {meta_state.recent_win_rate:.1%}")
    
    # ============================================================================
    # STEP 6: Test Meta-Policy Decisions
    # ============================================================================
    logger.info("\n[STEP 6] Testing meta-policy decisions...")
    
    # Should we run SESA evolution?
    sesa_decision = meta_policy.should_run_sesa_evolution(
        hours_since_last_run=25,
        recent_performance=0.70,
        recent_performance_baseline=0.85,
    )
    
    logger.info(f"\n  SESA Evolution Decision:")
    logger.info(f"    - Decision: {sesa_decision.decision_type.value}")
    logger.info(f"    - Confidence: {sesa_decision.confidence:.2f}")
    logger.info(f"    - Reasoning: {sesa_decision.reasoning}")
    
    # Should we promote shadow strategy?
    promotion_decision = meta_policy.should_promote_shadow_strategy(
        shadow_test_days=8,
        shadow_trades=75,
        shadow_sharpe=1.25,
        shadow_drawdown=0.08,
        shadow_win_rate=0.58,
        production_sharpe=1.05,
    )
    
    logger.info(f"\n  Shadow Promotion Decision:")
    logger.info(f"    - Decision: {promotion_decision.decision_type.value}")
    logger.info(f"    - Confidence: {promotion_decision.confidence:.2f}")
    logger.info(f"    - Reasoning: {promotion_decision.reasoning}")
    
    # Explore vs Exploit?
    mode_decision = meta_policy.determine_explore_exploit_mode(
        current_performance=0.75,
        target_performance=1.0,
    )
    
    logger.info(f"\n  Explore/Exploit Decision:")
    logger.info(f"    - Decision: {mode_decision.decision_type.value}")
    logger.info(f"    - Confidence: {mode_decision.confidence:.2f}")
    logger.info(f"    - Reasoning: {mode_decision.reasoning}")
    logger.info(f"    - Exploration Rate: {mode_decision.parameters.get('exploration_rate', 0):.2%}")
    
    # ============================================================================
    # STEP 7: Generate Global Action Plan
    # ============================================================================
    logger.info("\n[STEP 7] Generating Global Action Plan...")
    
    gap = await federation.generate_global_action_plan()
    
    logger.info(f"\n✓ Global Action Plan generated:")
    logger.info(f"  - Plan ID: {gap.plan_id}")
    logger.info(f"  - Total Actions: {len(gap.get_all_actions())}")
    logger.info(f"  - Critical: {len(gap.critical_actions)}")
    logger.info(f"  - High Priority: {len(gap.high_priority_actions)}")
    logger.info(f"  - Medium Priority: {len(gap.medium_priority_actions)}")
    logger.info(f"  - Low Priority: {len(gap.low_priority_actions)}")
    logger.info(f"  - System Health: {gap.system_health:.2f}")
    logger.info(f"  - Risk Level: {gap.risk_level}")
    logger.info(f"  - Recommended Mode: {gap.recommended_mode}")
    logger.info(f"  - Plan Confidence: {gap.plan_confidence:.2f}")
    logger.info(f"  - Consensus Score: {gap.consensus_score:.2f}")
    
    # Show actions
    if gap.get_all_actions():
        logger.info("\n  Actions in Plan:")
        for i, action in enumerate(gap.get_all_actions()[:5], 1):  # Show first 5
            logger.info(
                f"    {i}. [{action.priority.value.upper()}] {action.action_type.value}"
            )
            logger.info(f"       Source: {action.source_agent}")
            logger.info(f"       Reasoning: {action.reasoning}")
            logger.info(f"       Confidence: {action.confidence:.2f}")
    
    # ============================================================================
    # STEP 8: Summary
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info("\nCapabilities Demonstrated:")
    logger.info("  ✓ SESA: Self-evolving strategy generation & evaluation")
    logger.info("  ✓ Meta-Learning: Hyperparameter adaptation based on performance")
    logger.info("  ✓ Meta-Policy: Strategic decision making (SESA, RL, promotions)")
    logger.info("  ✓ Federation v2: Global Action Plan aggregation")
    logger.info("  ✓ Full Integration: All components working together")
    
    logger.info("\nKey Results:")
    logger.info(f"  - {evolution_result.total_mutations_evaluated} strategies evaluated")
    logger.info(f"  - Best strategy score: {evolution_result.best_score:.2f}")
    logger.info(f"  - {len(evolution_result.production_candidates)} production-ready candidates")
    logger.info(f"  - Meta-learning optimized {meta_state.update_count} parameters")
    logger.info(f"  - Global Action Plan with {len(gap.get_all_actions())} coordinated actions")
    
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
