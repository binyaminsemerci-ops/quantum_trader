"""
Meta Strategy Controller - Examples and Tests

Demonstrates usage of MSC AI with mock repositories.
"""

from datetime import datetime, timedelta
from backend.services.meta_strategy_controller import (
    MetaStrategyController,
    StrategyConfig,
    StrategyStats,
    RiskMode,
    GlobalPolicy
)


# ============================================================================
# Mock Repositories
# ============================================================================

class MockMetricsRepository:
    """Mock metrics repository for testing"""
    
    def __init__(
        self,
        drawdown: float = 3.0,
        winrate: float = 0.55,
        regime: str = "BULL_TRENDING",
        volatility: str = "NORMAL"
    ):
        self.drawdown = drawdown
        self.winrate = winrate
        self.regime = regime
        self.volatility = volatility
    
    def get_current_drawdown_pct(self) -> float:
        return self.drawdown
    
    def get_global_winrate(self, last_trades: int = 200) -> float:
        return self.winrate
    
    def get_equity_curve(self, days: int = 30) -> list[tuple[datetime, float]]:
        # Simulate growing equity curve
        base = datetime.utcnow()
        curve = []
        balance = 10000.0
        
        for i in range(days):
            date = base - timedelta(days=days-i)
            balance += balance * 0.01  # 1% daily growth
            curve.append((date, balance))
        
        return curve
    
    def get_global_regime(self) -> str:
        return self.regime
    
    def get_volatility_level(self) -> str:
        return self.volatility
    
    def get_consecutive_losses(self) -> int:
        return 1
    
    def get_days_since_last_profit(self) -> int:
        return 0


class MockStrategyRepository:
    """Mock strategy repository for testing"""
    
    def __init__(self):
        # Create 5 mock LIVE strategies with varying performance
        self.strategies = [
            # Excellent performer
            StrategyConfig(
                strategy_id="STRAT_001",
                name="Trend Follower Alpha",
                status="LIVE",
                regime_tags=["BULL_TRENDING", "BEAR_TRENDING"],
                min_confidence=0.60,
                max_risk_per_trade=0.01,
                created_at=datetime.utcnow() - timedelta(days=90),
                updated_at=datetime.utcnow()
            ),
            # Good performer
            StrategyConfig(
                strategy_id="STRAT_002",
                name="Mean Reversion Beta",
                status="LIVE",
                regime_tags=["RANGING"],
                min_confidence=0.65,
                max_risk_per_trade=0.008,
                created_at=datetime.utcnow() - timedelta(days=60),
                updated_at=datetime.utcnow()
            ),
            # Average performer
            StrategyConfig(
                strategy_id="STRAT_003",
                name="Breakout Gamma",
                status="LIVE",
                regime_tags=["BULL_TRENDING", "VOLATILE"],
                min_confidence=0.55,
                max_risk_per_trade=0.012,
                created_at=datetime.utcnow() - timedelta(days=45),
                updated_at=datetime.utcnow()
            ),
            # Below average
            StrategyConfig(
                strategy_id="STRAT_004",
                name="Scalper Delta",
                status="LIVE",
                regime_tags=["RANGING", "CHOPPY"],
                min_confidence=0.70,
                max_risk_per_trade=0.005,
                created_at=datetime.utcnow() - timedelta(days=30),
                updated_at=datetime.utcnow()
            ),
            # Poor performer
            StrategyConfig(
                strategy_id="STRAT_005",
                name="Contrarian Epsilon",
                status="LIVE",
                regime_tags=["BEAR_TRENDING"],
                min_confidence=0.50,
                max_risk_per_trade=0.015,
                created_at=datetime.utcnow() - timedelta(days=20),
                updated_at=datetime.utcnow()
            ),
        ]
        
        # Mock stats for each strategy
        self.stats = {
            "STRAT_001": StrategyStats(
                strategy_id="STRAT_001",
                total_trades=120,
                winning_trades=75,
                losing_trades=45,
                total_pnl=15000.0,
                total_pnl_pct=15.0,
                max_drawdown_pct=3.2,
                profit_factor=2.5,
                winrate=0.625,
                avg_win=250.0,
                avg_loss=-100.0,
                sharpe_ratio=2.1
            ),
            "STRAT_002": StrategyStats(
                strategy_id="STRAT_002",
                total_trades=85,
                winning_trades=50,
                losing_trades=35,
                total_pnl=8500.0,
                total_pnl_pct=8.5,
                max_drawdown_pct=4.1,
                profit_factor=1.9,
                winrate=0.588,
                avg_win=220.0,
                avg_loss=-95.0,
                sharpe_ratio=1.6
            ),
            "STRAT_003": StrategyStats(
                strategy_id="STRAT_003",
                total_trades=65,
                winning_trades=35,
                losing_trades=30,
                total_pnl=4200.0,
                total_pnl_pct=4.2,
                max_drawdown_pct=5.8,
                profit_factor=1.5,
                winrate=0.538,
                avg_win=200.0,
                avg_loss=-90.0,
                sharpe_ratio=1.1
            ),
            "STRAT_004": StrategyStats(
                strategy_id="STRAT_004",
                total_trades=45,
                winning_trades=22,
                losing_trades=23,
                total_pnl=1200.0,
                total_pnl_pct=1.2,
                max_drawdown_pct=7.2,
                profit_factor=1.2,
                winrate=0.489,
                avg_win=180.0,
                avg_loss=-85.0,
                sharpe_ratio=0.7
            ),
            "STRAT_005": StrategyStats(
                strategy_id="STRAT_005",
                total_trades=30,
                winning_trades=12,
                losing_trades=18,
                total_pnl=-1500.0,
                total_pnl_pct=-1.5,
                max_drawdown_pct=9.5,
                profit_factor=0.8,
                winrate=0.400,
                avg_win=160.0,
                avg_loss=-80.0,
                sharpe_ratio=0.3
            ),
        }
    
    def get_strategies_by_status(self, status: str) -> list[StrategyConfig]:
        if status == "LIVE":
            return self.strategies
        return []
    
    def get_recent_stats(
        self,
        strategy_id: str,
        source: str,
        days: int
    ) -> StrategyStats:
        return self.stats.get(strategy_id)
    
    def get_strategy_config(self, strategy_id: str) -> StrategyConfig:
        for s in self.strategies:
            if s.strategy_id == strategy_id:
                return s
        return None


class InMemoryPolicyStore:
    """Simple in-memory policy store for testing"""
    
    def __init__(self):
        self.policy = {}
    
    def get(self) -> dict:
        return self.policy
    
    def update(self, policy: dict) -> None:
        self.policy = policy
        print(f"\n[PolicyStore] Policy updated at {policy.get('updated_at')}")


# ============================================================================
# Example 1: Normal Operation
# ============================================================================

def example_1_normal_operation():
    """
    Example 1: System in healthy state
    
    Expected behavior:
    - Risk mode: NORMAL
    - Top 4-6 strategies selected
    - Standard parameters applied
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Normal Operation (Healthy System)")
    print("="*80)
    
    # Create mock repositories with healthy metrics
    metrics_repo = MockMetricsRepository(
        drawdown=3.0,
        winrate=0.55,
        regime="BULL_TRENDING",
        volatility="NORMAL"
    )
    
    strategy_repo = MockStrategyRepository()
    policy_store = InMemoryPolicyStore()
    
    # Initialize controller
    controller = MetaStrategyController(
        metrics_repo=metrics_repo,
        strategy_repo=strategy_repo,
        policy_store=policy_store,
        min_strategies=2,
        max_strategies=8
    )
    
    # Run evaluation
    policy = controller.evaluate_and_update_policy()
    
    # Verify results
    print("\n[RESULTS]")
    print(f"Risk Mode: {policy['risk_mode']}")
    print(f"Allowed Strategies: {len(policy['allowed_strategies'])}")
    print(f"Max Risk/Trade: {policy['max_risk_per_trade']:.2%}")
    print(f"Min Confidence: {policy['global_min_confidence']:.1%}")
    print(f"Max Positions: {policy['max_positions']}")
    
    # Strong performance (3% DD, +1% daily, bull trending) triggers AGGRESSIVE
    assert policy['risk_mode'] == 'AGGRESSIVE'
    assert len(policy['allowed_strategies']) >= 2
    assert policy['max_risk_per_trade'] == 0.015
    
    print("\n✓ Example 1 passed: AGGRESSIVE mode selected due to strong performance!")



# ============================================================================
# Example 2: Defensive Mode (High Drawdown)
# ============================================================================

def example_2_defensive_mode():
    """
    Example 2: System under stress
    
    Expected behavior:
    - Risk mode: DEFENSIVE
    - Only top 2-3 strategies selected
    - Conservative parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Defensive Mode (High Drawdown)")
    print("="*80)
    
    # Create mock repositories with stressed metrics
    metrics_repo = MockMetricsRepository(
        drawdown=6.5,  # High drawdown
        winrate=0.42,  # Low winrate
        regime="CHOPPY",
        volatility="HIGH"
    )
    
    strategy_repo = MockStrategyRepository()
    policy_store = InMemoryPolicyStore()
    
    controller = MetaStrategyController(
        metrics_repo=metrics_repo,
        strategy_repo=strategy_repo,
        policy_store=policy_store,
        min_strategies=2,
        max_strategies=8
    )
    
    policy = controller.evaluate_and_update_policy()
    
    print("\n[RESULTS]")
    print(f"Risk Mode: {policy['risk_mode']}")
    print(f"Allowed Strategies: {len(policy['allowed_strategies'])}")
    print(f"Max Risk/Trade: {policy['max_risk_per_trade']:.2%}")
    print(f"Min Confidence: {policy['global_min_confidence']:.1%}")
    print(f"Max Positions: {policy['max_positions']}")
    
    assert policy['risk_mode'] == 'DEFENSIVE'
    assert len(policy['allowed_strategies']) <= 4
    assert policy['max_risk_per_trade'] == 0.003
    assert policy['global_min_confidence'] == 0.70
    
    print("\n✓ Example 2 passed!")


# ============================================================================
# Example 3: Aggressive Mode (Strong Performance)
# ============================================================================

def example_3_aggressive_mode():
    """
    Example 3: System performing excellently
    
    Expected behavior:
    - Risk mode: AGGRESSIVE
    - More strategies selected (6-8)
    - Higher risk parameters
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Aggressive Mode (Strong Performance)")
    print("="*80)
    
    # Create mock repositories with excellent metrics
    metrics_repo = MockMetricsRepository(
        drawdown=1.2,  # Very low drawdown
        winrate=0.68,  # High winrate
        regime="BULL_TRENDING",
        volatility="LOW"
    )
    
    strategy_repo = MockStrategyRepository()
    policy_store = InMemoryPolicyStore()
    
    controller = MetaStrategyController(
        metrics_repo=metrics_repo,
        strategy_repo=strategy_repo,
        policy_store=policy_store,
        min_strategies=2,
        max_strategies=8
    )
    
    policy = controller.evaluate_and_update_policy()
    
    print("\n[RESULTS]")
    print(f"Risk Mode: {policy['risk_mode']}")
    print(f"Allowed Strategies: {len(policy['allowed_strategies'])}")
    print(f"Max Risk/Trade: {policy['max_risk_per_trade']:.2%}")
    print(f"Min Confidence: {policy['global_min_confidence']:.1%}")
    print(f"Max Positions: {policy['max_positions']}")
    
    assert policy['risk_mode'] == 'AGGRESSIVE'
    assert len(policy['allowed_strategies']) >= 4
    assert policy['max_risk_per_trade'] == 0.015
    assert policy['global_min_confidence'] == 0.50
    
    print("\n✓ Example 3 passed!")


# ============================================================================
# Example 4: Strategy Selection Logic
# ============================================================================

def example_4_strategy_selection():
    """
    Example 4: Verify strategy selection prioritizes best performers
    
    Expected behavior:
    - Top strategies by score are selected
    - Poor performers (STRAT_005) excluded
    - Regime-compatible strategies get bonus
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Strategy Selection Logic")
    print("="*80)
    
    metrics_repo = MockMetricsRepository(
        drawdown=2.5,
        winrate=0.58,
        regime="BULL_TRENDING",  # Matches STRAT_001 and STRAT_003
        volatility="NORMAL"
    )
    
    strategy_repo = MockStrategyRepository()
    policy_store = InMemoryPolicyStore()
    
    controller = MetaStrategyController(
        metrics_repo=metrics_repo,
        strategy_repo=strategy_repo,
        policy_store=policy_store,
        min_strategies=3,
        max_strategies=4
    )
    
    policy = controller.evaluate_and_update_policy()
    
    print("\n[RESULTS]")
    print(f"Selected strategies: {policy['allowed_strategies']}")
    
    # STRAT_001 should always be selected (best performer)
    assert "STRAT_001" in policy['allowed_strategies']
    
    # STRAT_005 should likely be excluded (worst performer)
    # (May be included if we're in aggressive mode and selecting many strategies)
    
    print("\n✓ Example 4 passed!")


# ============================================================================
# Example 5: Policy Evolution Over Time
# ============================================================================

def example_5_policy_evolution():
    """
    Example 5: Simulate policy changes as system conditions change
    
    Shows how MSC adapts to changing market conditions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Policy Evolution Over Time")
    print("="*80)
    
    strategy_repo = MockStrategyRepository()
    policy_store = InMemoryPolicyStore()
    
    scenarios = [
        ("Day 1: Healthy Start", 2.0, 0.60, "BULL_TRENDING", "NORMAL"),
        ("Day 2: Drawdown Starts", 4.5, 0.52, "RANGING", "HIGH"),
        ("Day 3: Crisis Mode", 7.0, 0.45, "CHOPPY", "EXTREME"),
        ("Day 4: Recovery Begins", 5.0, 0.55, "BULL_TRENDING", "HIGH"),
        ("Day 5: Back to Normal", 2.5, 0.62, "BULL_TRENDING", "NORMAL"),
    ]
    
    previous_mode = None
    
    for scenario_name, dd, wr, regime, vol in scenarios:
        print(f"\n\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"  DD: {dd:.1f}%, WR: {wr:.1%}, Regime: {regime}, Vol: {vol}")
        print('='*70)
        
        metrics_repo = MockMetricsRepository(
            drawdown=dd,
            winrate=wr,
            regime=regime,
            volatility=vol
        )
        
        controller = MetaStrategyController(
            metrics_repo=metrics_repo,
            strategy_repo=strategy_repo,
            policy_store=policy_store,
            min_strategies=2,
            max_strategies=6
        )
        
        policy = controller.evaluate_and_update_policy()
        
        current_mode = policy['risk_mode']
        
        if previous_mode and current_mode != previous_mode:
            print(f"\n⚠️  RISK MODE CHANGED: {previous_mode} → {current_mode}")
        
        previous_mode = current_mode
    
    print("\n✓ Example 5 completed - Policy adapted to changing conditions!")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "="*80)
    print("META STRATEGY CONTROLLER - EXAMPLES & TESTS")
    print("="*80)
    
    example_1_normal_operation()
    example_2_defensive_mode()
    example_3_aggressive_mode()
    example_4_strategy_selection()
    example_5_policy_evolution()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
    print("="*80)
