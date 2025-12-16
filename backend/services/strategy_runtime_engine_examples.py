"""
Example usage and tests for Strategy Runtime Engine

Demonstrates how to:
1. Set up the engine with mock dependencies
2. Create and evaluate strategies
3. Generate trading signals
4. Integrate with the existing execution pipeline
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from backend.services.strategy_runtime_engine import (
    StrategyRuntimeEngine,
    StrategyEvaluator,
    StrategyConfig,
    TradeDecision,
    StrategySignal,
    SignalType
)


# ============================================================================
# Mock Implementations for Testing
# ============================================================================

class MockStrategyRepository:
    """Mock repository for testing"""
    
    def __init__(self):
        self.strategies = {}
        self.execution_times = {}
    
    def add_strategy(self, strategy: StrategyConfig):
        """Add a strategy to the mock repository"""
        self.strategies[strategy.strategy_id] = strategy
    
    def get_by_status(self, status: str) -> List[StrategyConfig]:
        return [s for s in self.strategies.values() if s.status == status]
    
    def get_by_id(self, strategy_id: str) -> Optional[StrategyConfig]:
        return self.strategies.get(strategy_id)
    
    def update_last_execution(self, strategy_id: str, timestamp: datetime) -> None:
        self.execution_times[strategy_id] = timestamp


class MockMarketDataClient:
    """Mock market data client for testing"""
    
    def __init__(self):
        self.prices = {"BTCUSDT": 50000.0, "ETHUSDT": 3000.0}
        self.indicators = {}
    
    def set_indicators(self, symbol: str, indicators: Dict[str, float]):
        """Set indicator values for testing"""
        self.indicators[symbol] = indicators
    
    def get_current_price(self, symbol: str) -> float:
        return self.prices.get(symbol, 50000.0)
    
    def get_latest_bars(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int
    ) -> pd.DataFrame:
        """Generate mock OHLCV data"""
        dates = pd.date_range(end=datetime.utcnow(), periods=limit, freq='1h')
        
        # Generate realistic price data
        base_price = self.prices.get(symbol, 50000.0)
        prices = base_price + np.random.randn(limit).cumsum() * 100
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, limit)
        })
        
        return df
    
    def get_indicators(
        self, 
        symbol: str, 
        indicators: List[str]
    ) -> Dict[str, float]:
        """Return pre-set indicator values"""
        if symbol in self.indicators:
            return self.indicators[symbol]
        
        # Default values
        return {
            "RSI": 50.0,
            "MACD": 0.0,
            "MACD_SIGNAL": 0.0,
            "SMA_50": 50000.0,
            "SMA_200": 49000.0
        }


class MockPolicyStore:
    """Mock policy store for testing"""
    
    def __init__(self):
        self.risk_mode = "NORMAL"
        self.global_min_confidence = 0.5
        self.allowed_strategies = set()
    
    def get_risk_mode(self) -> str:
        return self.risk_mode
    
    def get_global_min_confidence(self) -> float:
        return self.global_min_confidence
    
    def is_strategy_allowed(self, strategy_id: str) -> bool:
        # If no strategies specified, allow all
        if not self.allowed_strategies:
            return True
        return strategy_id in self.allowed_strategies


# ============================================================================
# Example 1: Basic Strategy Evaluation
# ============================================================================

def example_1_basic_strategy_evaluation():
    """
    Example 1: Create a simple RSI strategy and evaluate it
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Strategy Evaluation")
    print("="*70)
    
    # Create a simple RSI oversold strategy
    strategy = StrategyConfig(
        strategy_id="rsi_oversold_001",
        name="RSI Oversold Long",
        status="LIVE",
        entry_indicators=[
            {"name": "RSI", "operator": "<", "value": 30}
        ],
        entry_logic="ALL",
        base_size_usd=1000.0,
        confidence_scaling=True,
        stop_loss_pct=0.02,  # 2%
        take_profit_pct=0.05,  # 5%
        allowed_regimes=["TRENDING", "NORMAL"],
        min_confidence=0.5,
        max_positions=1,
        fitness_score=0.75
    )
    
    # Set up mock market data with oversold RSI
    market_data_client = MockMarketDataClient()
    market_data_client.set_indicators("BTCUSDT", {
        "RSI": 25.0,  # Oversold!
        "MACD": 50.0,
        "MACD_SIGNAL": 30.0  # Bullish crossover
    })
    
    # Evaluate
    evaluator = StrategyEvaluator()
    
    market_data = market_data_client.get_latest_bars("BTCUSDT", "1h", 100)
    indicators = market_data_client.get_indicators("BTCUSDT", ["RSI", "MACD", "MACD_SIGNAL"])
    
    signal = evaluator.evaluate(
        strategy=strategy,
        symbol="BTCUSDT",
        market_data=market_data,
        indicators=indicators,
        current_regime="TRENDING"
    )
    
    if signal:
        print(f"✅ Signal generated!")
        print(f"   Direction: {signal.direction}")
        print(f"   Strength: {signal.strength:.2f}")
        print(f"   Conditions met: {signal.conditions_met}")
    else:
        print("❌ No signal generated")
    
    return signal


# ============================================================================
# Example 2: Multiple Strategies and Signal Generation
# ============================================================================

def example_2_multiple_strategies():
    """
    Example 2: Set up multiple strategies and generate signals
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Strategies")
    print("="*70)
    
    # Set up repository with multiple strategies
    repo = MockStrategyRepository()
    
    # Strategy 1: RSI Oversold
    repo.add_strategy(StrategyConfig(
        strategy_id="rsi_oversold_001",
        name="RSI Oversold Long",
        status="LIVE",
        entry_indicators=[{"name": "RSI", "operator": "<", "value": 30}],
        entry_logic="ALL",
        base_size_usd=1000.0,
        confidence_scaling=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        allowed_regimes=["TRENDING", "NORMAL"],
        min_confidence=0.5,
        max_positions=1,
        fitness_score=0.75
    ))
    
    # Strategy 2: RSI Overbought (Short)
    repo.add_strategy(StrategyConfig(
        strategy_id="rsi_overbought_001",
        name="RSI Overbought Short",
        status="LIVE",
        entry_indicators=[{"name": "RSI", "operator": ">", "value": 70}],
        entry_logic="ALL",
        base_size_usd=1500.0,
        confidence_scaling=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        allowed_regimes=["TRENDING"],
        min_confidence=0.6,
        max_positions=1,
        fitness_score=0.82
    ))
    
    # Strategy 3: MACD Crossover
    repo.add_strategy(StrategyConfig(
        strategy_id="macd_cross_001",
        name="MACD Bullish Crossover",
        status="LIVE",
        entry_indicators=[
            {"name": "MACD", "operator": ">", "value": 0},
            {"name": "RSI", "operator": ">", "value": 40}
        ],
        entry_logic="ALL",
        base_size_usd=2000.0,
        confidence_scaling=True,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        allowed_regimes=["TRENDING", "NORMAL", "RANGING"],
        min_confidence=0.55,
        max_positions=2,
        fitness_score=0.68
    ))
    
    # Set up market data client
    market_data_client = MockMarketDataClient()
    
    # BTCUSDT: Oversold (should trigger strategy 1)
    market_data_client.set_indicators("BTCUSDT", {
        "RSI": 28.0,
        "MACD": -50.0,
        "MACD_SIGNAL": -30.0
    })
    
    # ETHUSDT: Overbought (should trigger strategy 2)
    market_data_client.set_indicators("ETHUSDT", {
        "RSI": 75.0,
        "MACD": 100.0,
        "MACD_SIGNAL": 80.0
    })
    
    # Set up policy store
    policy_store = MockPolicyStore()
    policy_store.risk_mode = "AGGRESSIVE"
    policy_store.global_min_confidence = 0.4
    
    # Create engine
    engine = StrategyRuntimeEngine(
        strategy_repository=repo,
        market_data_client=market_data_client,
        policy_store=policy_store
    )
    
    # Generate signals
    decisions = engine.generate_signals(
        symbols=["BTCUSDT", "ETHUSDT"],
        current_regime="TRENDING"
    )
    
    print(f"\n📊 Generated {len(decisions)} trade decisions:")
    for decision in decisions:
        print(f"\n   Strategy: {decision.strategy_id}")
        print(f"   Symbol: {decision.symbol}")
        print(f"   Side: {decision.side}")
        print(f"   Size: ${decision.size_usd:.2f}")
        print(f"   Confidence: {decision.confidence:.2%}")
        print(f"   Entry: ${decision.entry_price:.2f}")
        print(f"   TP: ${decision.take_profit:.2f}")
        print(f"   SL: ${decision.stop_loss:.2f}")
        print(f"   Reasoning: {decision.reasoning}")
    
    return decisions


# ============================================================================
# Example 3: Integration with Execution Pipeline
# ============================================================================

def example_3_integration_with_pipeline():
    """
    Example 3: Show how Strategy Runtime Engine integrates with the
    existing Quantum Trader execution pipeline
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Integration with Execution Pipeline")
    print("="*70)
    
    # Set up engine (same as Example 2)
    repo = MockStrategyRepository()
    repo.add_strategy(StrategyConfig(
        strategy_id="test_strategy_001",
        name="Test Strategy",
        status="LIVE",
        entry_indicators=[{"name": "RSI", "operator": "<", "value": 35}],
        entry_logic="ALL",
        base_size_usd=1000.0,
        confidence_scaling=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        allowed_regimes=["TRENDING"],
        min_confidence=0.5,
        max_positions=1,
        fitness_score=0.7
    ))
    
    market_data_client = MockMarketDataClient()
    market_data_client.set_indicators("BTCUSDT", {"RSI": 30.0, "MACD": 0.0, "MACD_SIGNAL": 0.0})
    
    policy_store = MockPolicyStore()
    
    engine = StrategyRuntimeEngine(
        strategy_repository=repo,
        market_data_client=market_data_client,
        policy_store=policy_store
    )
    
    # Generate signals (this would be called from event-driven loop)
    decisions = engine.generate_signals(
        symbols=["BTCUSDT"],
        current_regime="TRENDING"
    )
    
    print("\n🔄 Simulating Execution Pipeline:")
    
    for decision in decisions:
        print(f"\n1️⃣  Strategy Runtime Engine → TradeDecision generated")
        print(f"   Strategy ID: {decision.strategy_id}")
        print(f"   Symbol: {decision.symbol}")
        print(f"   Side: {decision.side}")
        print(f"   Confidence: {decision.confidence:.2%}")
        
        # 2. Orchestrator Policy (checks if trade is allowed)
        print(f"\n2️⃣  Orchestrator Policy → Checking trade...")
        orchestrator_approved = decision.confidence >= 0.5
        if orchestrator_approved:
            print(f"   ✅ APPROVED (confidence {decision.confidence:.2%} >= 50%)")
        else:
            print(f"   ❌ REJECTED (confidence too low)")
            continue
        
        # 3. Risk Guard (validates position size, stop loss, etc.)
        print(f"\n3️⃣  Risk Guard → Validating risk parameters...")
        risk_approved = True
        if decision.stop_loss and decision.entry_price:
            risk_pct = abs(decision.entry_price - decision.stop_loss) / decision.entry_price
            print(f"   Risk per trade: {risk_pct:.2%}")
            risk_approved = risk_pct <= 0.03  # Max 3%
        
        if risk_approved:
            print(f"   ✅ APPROVED")
        else:
            print(f"   ❌ REJECTED (risk too high)")
            continue
        
        # 4. Portfolio Balancer (checks exposure, correlation, etc.)
        print(f"\n4️⃣  Portfolio Balancer → Checking portfolio constraints...")
        portfolio_approved = True  # Simplified
        print(f"   ✅ APPROVED")
        
        # 5. Safety Governor (circuit breaker, drawdown limits)
        print(f"\n5️⃣  Safety Governor → Checking system health...")
        safety_approved = True  # Simplified
        print(f"   ✅ APPROVED")
        
        # 6. Executor (places the actual trade)
        print(f"\n6️⃣  Executor → Placing order...")
        print(f"   📈 MARKET {decision.side} {decision.symbol}")
        print(f"   💰 Size: ${decision.size_usd:.2f}")
        print(f"   🎯 TP: ${decision.take_profit:.2f}")
        print(f"   🛡️  SL: ${decision.stop_loss:.2f}")
        print(f"   ✅ ORDER PLACED")
        
        # 7. Position tracking (tagged with strategy_id)
        print(f"\n7️⃣  Position Monitor → Tracking position...")
        print(f"   Strategy ID: {decision.strategy_id}")
        print(f"   Position tagged for performance tracking")
    
    print("\n" + "="*70)
    print("✅ Complete execution pipeline flow demonstrated!")
    print("="*70)


# ============================================================================
# Example 4: Strategy Performance Tracking
# ============================================================================

def example_4_strategy_performance_tracking():
    """
    Example 4: Demonstrate how to track strategy performance
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Strategy Performance Tracking")
    print("="*70)
    
    # Simulate multiple signals from a strategy over time
    repo = MockStrategyRepository()
    
    strategy = StrategyConfig(
        strategy_id="performance_test_001",
        name="Performance Test Strategy",
        status="LIVE",
        entry_indicators=[{"name": "RSI", "operator": "<", "value": 35}],
        entry_logic="ALL",
        base_size_usd=1000.0,
        confidence_scaling=True,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        allowed_regimes=["TRENDING"],
        min_confidence=0.5,
        max_positions=1,
        fitness_score=0.72
    )
    repo.add_strategy(strategy)
    
    market_data_client = MockMarketDataClient()
    policy_store = MockPolicyStore()
    
    engine = StrategyRuntimeEngine(
        strategy_repository=repo,
        market_data_client=market_data_client,
        policy_store=policy_store
    )
    
    print("\n📈 Simulating 5 trading sessions:\n")
    
    # Track all trades
    all_trades = []
    
    for session in range(1, 6):
        print(f"Session {session}:")
        
        # Set different RSI values
        rsi_value = 25 + (session * 2)  # 27, 29, 31, 33, 35
        market_data_client.set_indicators("BTCUSDT", {
            "RSI": rsi_value,
            "MACD": 0.0,
            "MACD_SIGNAL": 0.0
        })
        
        # Generate signal
        decisions = engine.generate_signals(
            symbols=["BTCUSDT"],
            current_regime="TRENDING"
        )
        
        if decisions:
            decision = decisions[0]
            print(f"  ✅ Signal generated (RSI={rsi_value:.1f})")
            print(f"     Confidence: {decision.confidence:.2%}")
            print(f"     Size: ${decision.size_usd:.2f}")
            
            # Simulate trade outcome (for demonstration)
            outcome = "WIN" if np.random.random() > 0.4 else "LOSS"
            pnl = decision.size_usd * 0.05 if outcome == "WIN" else -decision.size_usd * 0.02
            
            all_trades.append({
                "session": session,
                "strategy_id": decision.strategy_id,
                "symbol": decision.symbol,
                "confidence": decision.confidence,
                "size": decision.size_usd,
                "outcome": outcome,
                "pnl": pnl
            })
            
            print(f"     Outcome: {outcome} (PnL: ${pnl:.2f})")
        else:
            print(f"  ❌ No signal (RSI={rsi_value:.1f})")
        
        print()
    
    # Summary
    total_pnl = sum(t["pnl"] for t in all_trades)
    win_rate = sum(1 for t in all_trades if t["outcome"] == "WIN") / len(all_trades) if all_trades else 0
    
    print("="*70)
    print("📊 Performance Summary:")
    print(f"   Strategy ID: {strategy.strategy_id}")
    print(f"   Total Trades: {len(all_trades)}")
    print(f"   Win Rate: {win_rate:.1%}")
    print(f"   Total PnL: ${total_pnl:.2f}")
    print(f"   Avg PnL per Trade: ${total_pnl / len(all_trades):.2f}" if all_trades else "   N/A")
    print("="*70)
    
    print("\n💡 Note: Each trade is tagged with strategy_id, allowing:")
    print("   - Per-strategy performance tracking")
    print("   - Real-time fitness score updates")
    print("   - Automatic promotion/demotion by SG AI")
    print("   - Portfolio attribution analysis")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STRATEGY RUNTIME ENGINE - EXAMPLES & TESTS")
    print("="*70)
    
    # Run examples
    example_1_basic_strategy_evaluation()
    example_2_multiple_strategies()
    example_3_integration_with_pipeline()
    example_4_strategy_performance_tracking()
    
    print("\n" + "="*70)
    print("✅ All examples completed successfully!")
    print("="*70)
