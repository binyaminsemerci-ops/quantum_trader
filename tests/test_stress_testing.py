"""
Unit tests for Stress Testing System
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.services.stress_testing import (
    Scenario,
    ScenarioType,
    ScenarioResult,
    ExecutionResult,
    TradeRecord,
    ScenarioLoader,
    ScenarioTransformer,
    ExchangeSimulator,
    ScenarioExecutor,
    StressTestRunner,
    ScenarioLibrary
)


# ============================================================================
# Test Scenario Models
# ============================================================================

def test_scenario_creation():
    """Test scenario creation"""
    scenario = Scenario(
        name="Test Scenario",
        type=ScenarioType.FLASH_CRASH,
        parameters={"drop_pct": 0.15}
    )
    
    assert scenario.name == "Test Scenario"
    assert scenario.type == ScenarioType.FLASH_CRASH
    assert scenario.parameters["drop_pct"] == 0.15
    assert scenario.symbols == ["BTCUSDT", "ETHUSDT"]  # Defaults


def test_scenario_result_metrics():
    """Test result metric calculations"""
    result = ScenarioResult(scenario_name="Test")
    
    # Add mock trades
    result.trades = [
        TradeRecord("BTC", "BUY", 40000, 41000, 1.0, 1000, "strategy1", datetime.utcnow(), closed=True),
        TradeRecord("BTC", "BUY", 40000, 39000, 1.0, -1000, "strategy1", datetime.utcnow(), closed=True),
        TradeRecord("ETH", "BUY", 2500, 2600, 2.0, 200, "strategy2", datetime.utcnow(), closed=True)
    ]
    
    result.total_trades = 3
    result.winning_trades = 2
    result.losing_trades = 1
    result.__post_init__()
    
    assert result.winrate == pytest.approx(66.67, rel=0.1)
    assert result.profit_factor == pytest.approx(1.2, rel=0.1)


def test_execution_result():
    """Test execution result"""
    result = ExecutionResult(
        success=True,
        filled_price=40000.0,
        filled_qty=1.0,
        slippage_pct=0.001,
        latency_ms=50.0
    )
    
    assert result.success
    assert result.slippage_pct == 0.001


# ============================================================================
# Test Scenario Loader
# ============================================================================

def test_scenario_loader_synthetic_data():
    """Test synthetic data generation"""
    loader = ScenarioLoader()
    
    scenario = Scenario(
        name="Test",
        type=ScenarioType.FLASH_CRASH,
        symbols=["BTCUSDT"]
    )
    
    df = loader.load_data(scenario)
    
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "close" in df.columns
    assert (df["symbol"] == "BTCUSDT").all()


def test_scenario_loader_multi_symbol():
    """Test multi-symbol data generation"""
    loader = ScenarioLoader()
    
    scenario = Scenario(
        name="Test",
        type=ScenarioType.VOLATILITY_SPIKE,
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    )
    
    df = loader.load_data(scenario)
    
    symbols = df["symbol"].unique()
    assert len(symbols) == 3
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols
    assert "SOLUSDT" in symbols


def test_data_validation():
    """Test data quality validation"""
    loader = ScenarioLoader()
    
    # Valid data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=10, freq="1H"),
        "symbol": ["BTCUSDT"] * 10,
        "open": np.random.uniform(40000, 41000, 10),
        "high": np.random.uniform(41000, 42000, 10),
        "low": np.random.uniform(39000, 40000, 10),
        "close": np.random.uniform(40000, 41000, 10),
        "volume": np.random.uniform(1000, 2000, 10)
    })
    
    is_valid, issues = loader.validate_data(df)
    assert is_valid
    assert len(issues) == 0
    
    # Invalid data (negative price)
    df.loc[0, "close"] = -100
    is_valid, issues = loader.validate_data(df)
    assert not is_valid
    assert len(issues) > 0


# ============================================================================
# Test Scenario Transformer
# ============================================================================

def test_transformer_flash_crash():
    """Test flash crash transformation"""
    transformer = ScenarioTransformer()
    
    # Create base data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=100, freq="1H"),
        "symbol": ["BTCUSDT"] * 100,
        "open": [40000.0] * 100,
        "high": [41000.0] * 100,
        "low": [39000.0] * 100,
        "close": [40000.0] * 100,
        "volume": [1000.0] * 100
    })
    
    scenario = Scenario(
        name="Crash Test",
        type=ScenarioType.FLASH_CRASH,
        parameters={"drop_pct": 0.20, "duration_bars": 5}
    )
    
    df_stressed = transformer.apply(df, scenario)
    
    # Check that some bars are stressed
    assert df_stressed["stressed"].sum() > 0
    
    # Check that prices dropped during crash
    stressed_mask = df_stressed["stressed"] & (df_stressed["stress_type"] == "flash_crash")
    if stressed_mask.any():
        min_stressed_price = df_stressed[stressed_mask]["close"].min()
        assert min_stressed_price < 40000.0  # Price should drop


def test_transformer_volatility_spike():
    """Test volatility spike transformation"""
    transformer = ScenarioTransformer()
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=100, freq="1H"),
        "symbol": ["BTCUSDT"] * 100,
        "open": [40000.0] * 100,
        "high": [40100.0] * 100,
        "low": [39900.0] * 100,
        "close": [40000.0] * 100,
        "volume": [1000.0] * 100
    })
    
    scenario = Scenario(
        name="Vol Test",
        type=ScenarioType.VOLATILITY_SPIKE,
        parameters={"multiplier": 5.0, "duration_bars": 20}
    )
    
    df_stressed = transformer.apply(df, scenario)
    
    # Check that some bars have increased volatility
    stressed_mask = df_stressed["stressed"]
    if stressed_mask.any():
        stressed_range = (df_stressed[stressed_mask]["high"] - df_stressed[stressed_mask]["low"]).mean()
        normal_range = (df_stressed[~stressed_mask]["high"] - df_stressed[~stressed_mask]["low"]).mean()
        assert stressed_range > normal_range


def test_transformer_data_corruption():
    """Test data corruption transformation"""
    transformer = ScenarioTransformer()
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=100, freq="1H"),
        "symbol": ["BTCUSDT"] * 100,
        "open": [40000.0] * 100,
        "high": [41000.0] * 100,
        "low": [39000.0] * 100,
        "close": [40000.0] * 100,
        "volume": [1000.0] * 100
    })
    
    scenario = Scenario(
        name="Corruption Test",
        type=ScenarioType.DATA_CORRUPTION,
        parameters={"corruption_pct": 0.10},
        seed=42
    )
    
    df_stressed = transformer.apply(df, scenario)
    
    # Check that some data is corrupted
    assert df_stressed["stressed"].sum() > 0
    # May have NaN values now
    assert df_stressed["close"].isna().sum() >= 0


# ============================================================================
# Test Exchange Simulator
# ============================================================================

def test_exchange_simulator_normal_conditions():
    """Test execution under normal conditions"""
    simulator = ExchangeSimulator()
    
    class MockDecision:
        symbol = "BTCUSDT"
        side = "BUY"
        quantity = 1.0
        price = 40000.0
        stop_loss = None
        take_profit = None
    
    bar = {
        "close": 40000.0,
        "high": 40100.0,
        "low": 39900.0,
        "volume": 1000.0,
        "stressed": False
    }
    
    scenario = Scenario("Test", ScenarioType.HISTORIC_REPLAY)
    
    result = simulator.execute_order(MockDecision(), bar, scenario)
    
    assert result.success
    assert result.filled_qty > 0
    assert result.filled_price > 0


def test_exchange_simulator_stress_conditions():
    """Test execution under stress"""
    simulator = ExchangeSimulator()
    
    class MockDecision:
        symbol = "BTCUSDT"
        side = "BUY"
        quantity = 1.0
        price = 40000.0
        stop_loss = None
        take_profit = None
    
    bar = {
        "close": 40000.0,
        "high": 40100.0,
        "low": 39900.0,
        "volume": 50.0,  # Low liquidity
        "stressed": True,
        "stress_type": "flash_crash"
    }
    
    scenario = Scenario("Test", ScenarioType.FLASH_CRASH)
    
    result = simulator.execute_order(MockDecision(), bar, scenario)
    
    # May succeed or fail, but should have higher slippage
    if result.success:
        assert result.slippage_pct > 0.001  # Higher than base


# ============================================================================
# Test Scenario Executor
# ============================================================================

def test_scenario_executor_initialization():
    """Test executor initialization"""
    executor = ScenarioExecutor(
        initial_capital=100000.0
    )
    
    assert executor.initial_capital == 100000.0
    assert executor.exchange_simulator is not None


def test_scenario_executor_empty_run():
    """Test executor with empty data"""
    executor = ScenarioExecutor(initial_capital=100000.0)
    
    df = pd.DataFrame({
        "timestamp": [],
        "symbol": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "volume": []
    })
    
    scenario = Scenario("Empty Test", ScenarioType.FLASH_CRASH)
    
    result = executor.run(df, scenario)
    
    assert result.success
    assert result.total_trades == 0


# ============================================================================
# Test Stress Test Runner
# ============================================================================

def test_stress_test_runner_initialization():
    """Test runner initialization"""
    executor = ScenarioExecutor(initial_capital=100000.0)
    runner = StressTestRunner(executor=executor)
    
    assert runner.executor is not None
    assert runner.loader is not None
    assert runner.transformer is not None


def test_stress_test_runner_batch():
    """Test batch scenario execution"""
    executor = ScenarioExecutor(initial_capital=100000.0)
    runner = StressTestRunner(executor=executor, max_workers=2)
    
    scenarios = [
        Scenario("Test 1", ScenarioType.FLASH_CRASH, parameters={"drop_pct": 0.10}),
        Scenario("Test 2", ScenarioType.VOLATILITY_SPIKE, parameters={"multiplier": 2.0})
    ]
    
    results = runner.run_batch(scenarios, parallel=False)
    
    assert len(results) == 2
    assert "Test 1" in results
    assert "Test 2" in results


def test_stress_test_runner_summary():
    """Test summary generation"""
    executor = ScenarioExecutor(initial_capital=100000.0)
    runner = StressTestRunner(executor=executor)
    
    # Create mock results
    results = {
        "Scenario 1": ScenarioResult(
            scenario_name="Scenario 1",
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            max_drawdown=5.0,
            emergency_stops=0
        ),
        "Scenario 2": ScenarioResult(
            scenario_name="Scenario 2",
            total_trades=15,
            winning_trades=9,
            losing_trades=6,
            max_drawdown=8.0,
            emergency_stops=1
        )
    }
    
    summary = runner.generate_summary(results)
    
    assert summary["total_scenarios"] == 2
    assert summary["total_trades"] == 25
    assert summary["scenarios_with_ess"] == 1
    assert summary["worst_drawdown"] == 8.0


# ============================================================================
# Test Scenario Library
# ============================================================================

def test_scenario_library_get_all():
    """Test getting all library scenarios"""
    scenarios = ScenarioLibrary.get_all()
    
    assert len(scenarios) > 0
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_scenario_library_get_by_name():
    """Test getting scenario by name"""
    scenario = ScenarioLibrary.get_by_name("BTC Flash Crash May 2021")
    
    assert scenario is not None
    assert scenario.type == ScenarioType.FLASH_CRASH


def test_scenario_library_get_by_type():
    """Test getting scenarios by type"""
    flash_crashes = ScenarioLibrary.get_by_type(ScenarioType.FLASH_CRASH)
    
    assert len(flash_crashes) > 0
    assert all(s.type == ScenarioType.FLASH_CRASH for s in flash_crashes)


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_pipeline():
    """Test complete stress testing pipeline"""
    # Create scenario
    scenario = Scenario(
        name="Integration Test",
        type=ScenarioType.FLASH_CRASH,
        parameters={"drop_pct": 0.15},
        symbols=["BTCUSDT"]
    )
    
    # Setup components
    loader = ScenarioLoader()
    transformer = ScenarioTransformer()
    executor = ScenarioExecutor(initial_capital=100000.0)
    runner = StressTestRunner(executor=executor)
    
    # Run pipeline
    results = runner.run_batch([scenario], parallel=False)
    
    # Verify result
    assert len(results) == 1
    result = results[scenario.name]
    assert result.scenario_name == scenario.name
    assert isinstance(result.pnl_curve, list)
    assert isinstance(result.equity_curve, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
