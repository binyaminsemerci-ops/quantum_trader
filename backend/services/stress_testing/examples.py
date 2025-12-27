"""
Example usage of the Stress Testing System

Demonstrates:
- Creating custom scenarios
- Running single scenario
- Batch testing
- Using scenario library
- Analyzing results
"""

import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from backend.services.stress_testing import (
    Scenario,
    ScenarioType,
    ScenarioLoader,
    ScenarioTransformer,
    ExchangeSimulator,
    ScenarioExecutor,
    StressTestRunner,
    ScenarioLibrary
)


# ============================================================================
# MOCK COMPONENTS (replace with real implementations)
# ============================================================================

class MockStrategyEngine:
    """Mock strategy signal generator"""
    def generate_signals(self, bar, context):
        # Simple mock: generate buy signal randomly
        import random
        if random.random() < 0.05:  # 5% chance
            return [{
                "symbol": bar.get("symbol", "BTCUSDT"),
                "side": "BUY",
                "confidence": 0.7,
                "strategy": "mock_strategy"
            }]
        return []


class MockOrchestrator:
    """Mock orchestrator"""
    def evaluate_signal(self, signal, context):
        # Create trade decision
        class Decision:
            def __init__(self):
                self.symbol = signal["symbol"]
                self.side = signal["side"]
                self.quantity = 0.01
                self.price = 0.0
                self.stop_loss = None
                self.take_profit = None
                self.strategy = signal["strategy"]
        
        return Decision()


class MockRiskGuard:
    """Mock risk guard"""
    def validate_trade(self, decision, context):
        # Allow trades unless equity is too low
        if context.get("equity", 0) < 10000:
            return False, "Equity too low"
        return True, ""


class MockPortfolioBalancer:
    """Mock portfolio balancer"""
    def check_constraints(self, decision, positions):
        # Allow if < 10 positions
        return len(positions) < 10


class MockSafetyGovernor:
    """Mock safety governor"""
    def check_safety(self, equity, drawdown):
        # Stop if drawdown > 30%
        if drawdown > 30.0:
            return False, "Maximum drawdown exceeded"
        return True, ""


class MockMSC:
    """Mock meta strategy controller"""
    def get_current_policy(self):
        return {
            "risk_mode": "NORMAL",
            "max_positions": 10,
            "global_min_confidence": 0.6
        }
    
    def update_policy(self, metrics):
        # In real implementation, adjust policy based on metrics
        pass


# ============================================================================
# EXAMPLE 1: Single Scenario Test
# ============================================================================

def example_single_scenario():
    """Run a single stress test scenario"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Scenario Test")
    print("="*70 + "\n")
    
    # Create scenario
    scenario = Scenario(
        name="Test Flash Crash",
        type=ScenarioType.FLASH_CRASH,
        parameters={
            "drop_pct": 0.20,
            "duration_bars": 3,
            "recovery_bars": 10
        },
        symbols=["BTCUSDT"],
        description="20% flash crash test"
    )
    
    # Setup components
    loader = ScenarioLoader()
    transformer = ScenarioTransformer()
    exchange = ExchangeSimulator()
    
    executor = ScenarioExecutor(
        runtime_engine=MockStrategyEngine(),
        orchestrator=MockOrchestrator(),
        risk_guard=MockRiskGuard(),
        portfolio_balancer=MockPortfolioBalancer(),
        safety_governor=MockSafetyGovernor(),
        msc=MockMSC(),
        exchange_simulator=exchange,
        initial_capital=100000.0
    )
    
    # Load and transform data
    print("Loading data...")
    df = loader.load_data(scenario)
    print(f"Loaded {len(df)} bars")
    
    print("Applying stress transformation...")
    df = transformer.apply(df, scenario)
    print(f"Stressed bars: {df['stressed'].sum()}")
    
    # Run simulation
    print("Running simulation...")
    result = executor.run(df, scenario)
    
    # Print results
    print("\nRESULTS:")
    print("-" * 70)
    print(f"Scenario: {result.scenario_name}")
    print(f"Success: {result.success}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.winrate:.1f}%")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Emergency Stops: {result.emergency_stops}")
    print(f"Execution Failures: {result.execution_failures}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    
    if result.notes:
        print(f"\nNotes:")
        for note in result.notes[:5]:  # Show first 5
            print(f"  - {note}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# EXAMPLE 2: Batch Testing with Scenario Library
# ============================================================================

def example_batch_testing():
    """Run multiple scenarios from library"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Testing with Scenario Library")
    print("="*70 + "\n")
    
    # Get scenarios from library
    scenarios = [
        ScenarioLibrary.btc_flash_crash_2021(),
        ScenarioLibrary.eth_liquidity_collapse(),
        ScenarioLibrary.sol_volatility_explosion()
    ]
    
    print(f"Selected {len(scenarios)} scenarios from library:")
    for s in scenarios:
        print(f"  - {s.name}: {s.description}")
    
    # Setup runner
    executor = ScenarioExecutor(
        runtime_engine=MockStrategyEngine(),
        orchestrator=MockOrchestrator(),
        risk_guard=MockRiskGuard(),
        portfolio_balancer=MockPortfolioBalancer(),
        safety_governor=MockSafetyGovernor(),
        msc=MockMSC(),
        exchange_simulator=ExchangeSimulator(),
        initial_capital=100000.0
    )
    
    runner = StressTestRunner(
        executor=executor,
        max_workers=2  # Run 2 scenarios in parallel
    )
    
    # Run batch
    print("\nRunning batch tests...")
    results = runner.run_batch(scenarios, parallel=True)
    
    # Print summary
    runner.print_summary(results)


# ============================================================================
# EXAMPLE 3: Custom Mixed Scenario
# ============================================================================

def example_custom_scenario():
    """Create and run custom complex scenario"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Mixed Scenario")
    print("="*70 + "\n")
    
    # Create complex custom scenario
    scenario = Scenario(
        name="Custom Chaos Scenario",
        type=ScenarioType.MIXED_CUSTOM,
        parameters={
            "flash_crash_drop_pct": 0.25,
            "volatility_multiplier": 6.0,
            "spread_mult": 8.0,
            "liquidity_drop_pct": 0.85
        },
        symbols=["BTCUSDT", "ETHUSDT"],
        description="Custom multi-stress scenario combining crash, vol, spread, and liquidity"
    )
    
    # Setup and run
    executor = ScenarioExecutor(
        runtime_engine=MockStrategyEngine(),
        orchestrator=MockOrchestrator(),
        risk_guard=MockRiskGuard(),
        portfolio_balancer=MockPortfolioBalancer(),
        safety_governor=MockSafetyGovernor(),
        msc=MockMSC(),
        exchange_simulator=ExchangeSimulator(),
        initial_capital=100000.0
    )
    
    runner = StressTestRunner(executor=executor)
    
    print("Running custom scenario...")
    results = runner.run_batch([scenario], parallel=False)
    
    # Detailed result analysis
    result = results[scenario.name]
    
    print("\nDETAILED ANALYSIS:")
    print("-" * 70)
    print(f"Final Equity: ${result.equity_curve[-1]:.2f}")
    print(f"Total PnL: ${result.pnl_curve[-1]:.2f}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Max DD Duration: {result.max_dd_duration} bars")
    print(f"Data Quality Issues: {result.data_quality_issues}")
    print(f"Policy Transitions: {len(result.policy_transitions)}")
    
    if result.trades:
        print(f"\nSample Trades:")
        for trade in result.trades[:3]:
            print(f"  {trade.symbol} {trade.side}: Entry=${trade.entry_price:.2f}, "
                  f"Exit=${trade.exit_price:.2f}, PnL=${trade.pnl:.2f}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# EXAMPLE 4: Historical Replay
# ============================================================================

def example_historical_replay():
    """Replay a historical period"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Historical Replay")
    print("="*70 + "\n")
    
    # Create historical replay scenario
    scenario = Scenario(
        name="BTC 2021 May Crash Replay",
        type=ScenarioType.HISTORIC_REPLAY,
        start=datetime(2021, 5, 1),
        end=datetime(2021, 5, 31),
        symbols=["BTCUSDT"]
    )
    
    print(f"Replaying: {scenario.start} to {scenario.end}")
    
    # Note: This would require real market data client
    # For demo, we'll use synthetic data
    executor = ScenarioExecutor(
        runtime_engine=MockStrategyEngine(),
        orchestrator=MockOrchestrator(),
        risk_guard=MockRiskGuard(),
        portfolio_balancer=MockPortfolioBalancer(),
        safety_governor=MockSafetyGovernor(),
        msc=MockMSC(),
        exchange_simulator=ExchangeSimulator(),
        initial_capital=100000.0
    )
    
    runner = StressTestRunner(executor=executor)
    results = runner.run_batch([scenario], parallel=False)
    
    result = results[scenario.name]
    print(f"\nReplay Results:")
    print(f"  Trades: {result.total_trades}")
    print(f"  Max DD: {result.max_drawdown:.2f}%")
    print(f"  ESS Activations: {result.emergency_stops}")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# EXAMPLE 5: Comprehensive Suite
# ============================================================================

def example_comprehensive_suite():
    """Run all library scenarios"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Comprehensive Test Suite")
    print("="*70 + "\n")
    
    # Get all library scenarios
    scenarios = ScenarioLibrary.get_all()
    
    print(f"Running all {len(scenarios)} library scenarios...\n")
    
    # Setup
    executor = ScenarioExecutor(
        runtime_engine=MockStrategyEngine(),
        orchestrator=MockOrchestrator(),
        risk_guard=MockRiskGuard(),
        portfolio_balancer=MockPortfolioBalancer(),
        safety_governor=MockSafetyGovernor(),
        msc=MockMSC(),
        exchange_simulator=ExchangeSimulator(),
        initial_capital=100000.0
    )
    
    runner = StressTestRunner(executor=executor, max_workers=4)
    
    # Run all
    results = runner.run_batch(scenarios, parallel=True)
    
    # Print comprehensive summary
    runner.print_summary(results)
    
    # Identify weakest scenarios
    print("\nWEAKEST SCENARIOS (Highest Drawdown):")
    print("-" * 70)
    sorted_by_dd = sorted(
        results.items(),
        key=lambda x: x[1].max_drawdown,
        reverse=True
    )
    
    for name, result in sorted_by_dd[:5]:
        print(f"{name}: {result.max_drawdown:.2f}% drawdown")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM TRADER - STRESS TESTING SYSTEM EXAMPLES")
    print("="*70)
    
    # Run examples
    try:
        example_single_scenario()
        example_batch_testing()
        example_custom_scenario()
        example_historical_replay()
        example_comprehensive_suite()
        
        print("\n✅ All examples completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}\n")
        import traceback
        traceback.print_exc()
