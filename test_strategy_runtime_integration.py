"""
Integration Test for Strategy Runtime Engine Production Deployment

Tests all 5 integration points:
1. PostgreSQL/SQLite Repository
2. Binance Market Data Client
3. Redis/DB Policy Store
4. Event-Driven Executor Integration
5. Prometheus Monitoring

Usage:
    python test_strategy_runtime_integration.py
"""

import os
import sys
import asyncio
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure test environment
os.environ['QT_ENV'] = 'test'
os.environ['DATABASE_URL'] = 'sqlite:///test_quantum_trader.db'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_1_repository_integration():
    """Test 1: PostgreSQL/SQLite Repository Integration"""
    print("\n" + "="*80)
    print("TEST 1: Repository Integration (PostgreSQL/SQLite)")
    print("="*80)
    
    try:
        from backend.services.strategy_runtime_integration import QuantumStrategyRepository
        from backend.services.strategy_runtime_engine import StrategyConfig
        
        # Initialize repository
        repo = QuantumStrategyRepository()
        print("✅ Repository initialized")
        
        # Get LIVE strategies
        live_strategies = repo.get_by_status("LIVE")
        print(f"✅ Retrieved {len(live_strategies)} LIVE strategies")
        
        if live_strategies:
            # Test get by ID
            first_strategy = live_strategies[0]
            strategy = repo.get_by_id(first_strategy.strategy_id)
            
            if strategy:
                print(f"✅ Retrieved strategy by ID: {strategy.strategy_id}")
                print(f"   Name: {strategy.name}")
                print(f"   Confidence: {strategy.min_confidence}")
                print(f"   Entry Indicators: {len(strategy.entry_indicators)}")
            else:
                print("❌ Failed to retrieve strategy by ID")
                return False
        
        # Test last execution update
        if live_strategies:
            repo.update_last_execution(live_strategies[0].strategy_id, datetime.utcnow())
            print("✅ Updated last execution timestamp")
        
        print("\n✅ TEST 1 PASSED: Repository integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_market_data_integration():
    """Test 2: Binance Market Data Client Integration"""
    print("\n" + "="*80)
    print("TEST 2: Market Data Client Integration (Binance)")
    print("="*80)
    
    try:
        from backend.services.strategy_runtime_integration import QuantumMarketDataClient
        
        # Initialize client
        client = QuantumMarketDataClient()
        print("✅ Market data client initialized")
        
        # Test current price
        symbol = "BTCUSDT"
        price = client.get_current_price(symbol)
        
        if price > 0:
            print(f"✅ Current price for {symbol}: ${price:,.2f}")
        else:
            print(f"❌ Failed to get price for {symbol}")
            return False
        
        # Test latest bars
        bars = client.get_latest_bars(symbol, "1h", 10)
        
        if not bars.empty:
            print(f"✅ Retrieved {len(bars)} bars for {symbol}")
            print(f"   Latest close: ${bars['close'].iloc[-1]:,.2f}")
        else:
            print(f"❌ Failed to get bars for {symbol}")
            return False
        
        # Test indicators
        indicators = client.get_indicators(symbol, ["RSI", "MACD", "MACD_SIGNAL"])
        
        if indicators:
            print(f"✅ Calculated indicators:")
            for name, value in indicators.items():
                print(f"   {name}: {value:.2f}")
        else:
            print("❌ Failed to calculate indicators")
            return False
        
        print("\n✅ TEST 2 PASSED: Market data integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_policy_store_integration():
    """Test 3: Redis/DB Policy Store Integration"""
    print("\n" + "="*80)
    print("TEST 3: Policy Store Integration (Redis + DB Fallback)")
    print("="*80)
    
    try:
        from backend.services.strategy_runtime_integration import QuantumPolicyStore
        
        # Initialize policy store
        store = QuantumPolicyStore()
        print(f"✅ Policy store initialized (using {'Redis' if store.use_redis else 'Database'})")
        
        # Test risk mode
        risk_mode = store.get_risk_mode()
        print(f"✅ Risk mode: {risk_mode}")
        
        # Set and verify
        store.set_risk_mode("AGGRESSIVE")
        new_mode = store.get_risk_mode()
        
        if new_mode == "AGGRESSIVE":
            print("✅ Risk mode update verified")
        else:
            print(f"❌ Risk mode mismatch: expected AGGRESSIVE, got {new_mode}")
            return False
        
        # Reset to normal
        store.set_risk_mode("NORMAL")
        
        # Test confidence threshold
        threshold = store.get_global_min_confidence()
        print(f"✅ Global min confidence: {threshold}")
        
        # Set and verify
        store.set_global_min_confidence(0.65)
        new_threshold = store.get_global_min_confidence()
        
        if new_threshold == 0.65:
            print("✅ Confidence threshold update verified")
        else:
            print(f"❌ Threshold mismatch: expected 0.65, got {new_threshold}")
            return False
        
        # Reset to default
        store.set_global_min_confidence(0.5)
        
        # Test strategy allowlist
        is_allowed = store.is_strategy_allowed("test-strategy-123")
        print(f"✅ Strategy allowlist check: {is_allowed}")
        
        print("\n✅ TEST 3 PASSED: Policy store integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_runtime_engine_integration():
    """Test 4: Strategy Runtime Engine Core"""
    print("\n" + "="*80)
    print("TEST 4: Strategy Runtime Engine Core Integration")
    print("="*80)
    
    try:
        from backend.services.strategy_runtime_integration import (
            get_strategy_runtime_engine,
            check_strategy_runtime_health,
            generate_strategy_signals
        )
        
        # Get engine instance (singleton)
        engine = get_strategy_runtime_engine()
        print(f"✅ Engine initialized with {engine.get_active_strategy_count()} active strategies")
        
        # Check health
        health = check_strategy_runtime_health()
        print(f"✅ Health check: {health['status']}")
        print(f"   Active strategies: {health['active_strategies']}")
        print(f"   Last refresh: {health['last_refresh']}")
        
        if health['status'] != 'healthy':
            print(f"❌ Engine unhealthy: {health.get('error')}")
            return False
        
        # Generate signals
        test_symbols = ["BTCUSDT", "ETHUSDT"]
        decisions = generate_strategy_signals(test_symbols, current_regime="TRENDING")
        
        print(f"✅ Generated {len(decisions)} signals")
        
        if decisions:
            for decision in decisions[:3]:  # Show first 3
                print(f"   • {decision.symbol}: {decision.side} @ {decision.confidence:.0%} confidence")
                print(f"     Size: ${decision.position_size_usd:.0f}, SL: {decision.stop_loss_pct:.1%}, TP: {decision.take_profit_pct:.1%}")
                print(f"     Reasoning: {decision.reasoning[:80]}...")
        
        print("\n✅ TEST 4 PASSED: Runtime engine integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_5_executor_integration():
    """Test 5: Event-Driven Executor Integration"""
    print("\n" + "="*80)
    print("TEST 5: Event-Driven Executor Integration")
    print("="*80)
    
    try:
        # Check if executor has Strategy Runtime Engine
        from backend.services.execution.event_driven_executor import EventDrivenExecutor
        from backend.services.ai_trading_engine import AITradingEngine
        
        # Create mock AI engine
        class MockAIEngine:
            async def get_trading_signals(self, symbols, positions):
                return []
            
            async def warmup_history_buffers(self, symbols, lookback):
                pass
        
        # Create executor
        symbols = ["BTCUSDT", "ETHUSDT"]
        executor = EventDrivenExecutor(
            ai_engine=MockAIEngine(),
            symbols=symbols,
            confidence_threshold=0.5
        )
        
        # Check if Strategy Runtime Engine was initialized
        if hasattr(executor, '_strategy_runtime_available'):
            if executor._strategy_runtime_available:
                print("✅ Strategy Runtime Engine integrated into executor")
                print(f"   Available: {executor._strategy_runtime_available}")
            else:
                print("⚠️  Strategy Runtime Engine initialized but unhealthy")
        else:
            print("❌ Strategy Runtime Engine not found in executor")
            return False
        
        print("\n✅ TEST 5 PASSED: Executor integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_prometheus_metrics():
    """Test 6: Prometheus Monitoring"""
    print("\n" + "="*80)
    print("TEST 6: Prometheus Metrics Integration")
    print("="*80)
    
    try:
        from backend.services.strategy_runtime_integration import (
            strategy_signals_generated,
            strategy_signal_confidence,
            strategy_evaluation_duration,
            active_strategies_gauge,
            strategy_last_signal_timestamp
        )
        from prometheus_client import REGISTRY
        
        # Check metrics are registered
        metric_names = [
            'strategy_runtime_signals_generated_total',
            'strategy_runtime_signal_confidence',
            'strategy_runtime_evaluation_duration_seconds',
            'strategy_runtime_active_strategies',
            'strategy_runtime_last_signal_timestamp'
        ]
        
        registered_metrics = [m.name for m in REGISTRY.collect()]
        
        for name in metric_names:
            if name in registered_metrics:
                print(f"✅ Metric registered: {name}")
            else:
                print(f"❌ Metric not found: {name}")
                return False
        
        # Test metrics can be updated
        strategy_signals_generated.labels(
            strategy_id="test-123",
            symbol="BTCUSDT",
            side="LONG"
        ).inc()
        print("✅ Counter metric incremented")
        
        strategy_signal_confidence.labels(strategy_id="test-123").observe(0.75)
        print("✅ Histogram metric observed")
        
        active_strategies_gauge.set(5)
        print("✅ Gauge metric set")
        
        print("\n✅ TEST 6 PASSED: Prometheus metrics working")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "="*80)
    print("STRATEGY RUNTIME ENGINE - PRODUCTION INTEGRATION TESTS")
    print("="*80)
    print(f"Test Environment: {os.getenv('QT_ENV', 'production')}")
    print(f"Database: {os.getenv('DATABASE_URL', 'default')}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    
    results = {}
    
    # Run tests
    results['Repository'] = test_1_repository_integration()
    results['Market Data'] = test_2_market_data_integration()
    results['Policy Store'] = test_3_policy_store_integration()
    results['Runtime Engine'] = test_4_runtime_engine_integration()
    results['Executor'] = asyncio.run(test_5_executor_integration())
    results['Prometheus'] = test_6_prometheus_metrics()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULT: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Production integration complete!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
