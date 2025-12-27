"""
Tests for System Health Monitor

Validates:
- Individual health check execution
- Result aggregation logic
- PolicyStore integration
- Status determination rules
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timedelta
from system_health_monitor import (
    HealthStatus,
    HealthCheckResult,
    SystemHealthSummary,
    SystemHealthMonitor,
    BaseHealthMonitor,
    FakeMarketDataHealthMonitor,
    FakePolicyStoreHealthMonitor,
    FakeExecutionHealthMonitor,
    FakeStrategyRuntimeHealthMonitor,
    FakeMSCHealthMonitor,
    FakeCLMHealthMonitor,
    FakeOpportunityRankerHealthMonitor,
)


# ============================================================================
# Mock PolicyStore
# ============================================================================

class MockPolicyStore:
    """Simple in-memory PolicyStore for testing."""
    
    def __init__(self):
        self.data = {}
    
    def patch(self, updates: dict) -> None:
        """Update policy data."""
        self.data.update(updates)
    
    def get(self) -> dict:
        """Get all policy data."""
        return self.data.copy()


# ============================================================================
# Test Data Models
# ============================================================================

def test_health_check_result_creation():
    """Test HealthCheckResult dataclass creation and serialization."""
    result = HealthCheckResult(
        module="test_module",
        status=HealthStatus.HEALTHY,
        details={"latency": 50},
        message="All good"
    )
    
    assert result.module == "test_module"
    assert result.status == HealthStatus.HEALTHY
    assert result.details["latency"] == 50
    assert result.message == "All good"
    
    # Test serialization
    data = result.to_dict()
    assert data["module"] == "test_module"
    assert data["status"] == "HEALTHY"
    assert "timestamp" in data


def test_system_health_summary_creation():
    """Test SystemHealthSummary dataclass creation and methods."""
    summary = SystemHealthSummary(
        status=HealthStatus.HEALTHY,
        healthy_modules=["mod1", "mod2"],
        warning_modules=[],
        failed_modules=[],
        total_checks=2
    )
    
    assert summary.is_healthy()
    assert not summary.is_degraded()
    assert not summary.is_critical()
    
    # Test serialization
    data = summary.to_dict()
    assert data["status"] == "HEALTHY"
    assert len(data["healthy_modules"]) == 2


def test_system_health_summary_warning_state():
    """Test SystemHealthSummary with WARNING status."""
    summary = SystemHealthSummary(
        status=HealthStatus.WARNING,
        healthy_modules=["mod1"],
        warning_modules=["mod2", "mod3"],
        failed_modules=[],
        total_checks=3
    )
    
    assert not summary.is_healthy()
    assert summary.is_degraded()
    assert not summary.is_critical()


def test_system_health_summary_critical_state():
    """Test SystemHealthSummary with CRITICAL status."""
    summary = SystemHealthSummary(
        status=HealthStatus.CRITICAL,
        healthy_modules=["mod1"],
        warning_modules=["mod2"],
        failed_modules=["mod3"],
        total_checks=3
    )
    
    assert not summary.is_healthy()
    assert not summary.is_degraded()
    assert summary.is_critical()


# ============================================================================
# Test Individual Monitors
# ============================================================================

def test_market_data_monitor_healthy():
    """Test market data monitor in healthy state."""
    monitor = FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0)
    result = monitor.run_check()
    
    assert result.module == "market_data"
    assert result.status == HealthStatus.HEALTHY
    assert result.details["latency_ms"] == 50
    assert "normally" in result.message.lower()


def test_market_data_monitor_warning():
    """Test market data monitor in warning state."""
    monitor = FakeMarketDataHealthMonitor(latency_ms=250, missing_candles=5)
    result = monitor.run_check()
    
    assert result.status == HealthStatus.WARNING
    assert "latency" in result.message.lower()


def test_market_data_monitor_critical():
    """Test market data monitor in critical state."""
    monitor = FakeMarketDataHealthMonitor(latency_ms=600, missing_candles=15)
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL


def test_policy_store_monitor_healthy():
    """Test PolicyStore monitor in healthy state."""
    monitor = FakePolicyStoreHealthMonitor(last_update_age_seconds=30)
    result = monitor.run_check()
    
    assert result.module == "policy_store"
    assert result.status == HealthStatus.HEALTHY


def test_policy_store_monitor_critical():
    """Test PolicyStore monitor in critical state."""
    monitor = FakePolicyStoreHealthMonitor(last_update_age_seconds=700)
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL
    assert "stale" in result.message.lower()


def test_execution_monitor_healthy():
    """Test execution monitor in healthy state."""
    monitor = FakeExecutionHealthMonitor(
        execution_latency_ms=100,
        failed_orders_rate=0.005
    )
    result = monitor.run_check()
    
    assert result.module == "execution"
    assert result.status == HealthStatus.HEALTHY


def test_execution_monitor_critical():
    """Test execution monitor in critical state."""
    monitor = FakeExecutionHealthMonitor(
        execution_latency_ms=1500,
        failed_orders_rate=0.15
    )
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL


def test_strategy_runtime_monitor_healthy():
    """Test strategy runtime monitor in healthy state."""
    monitor = FakeStrategyRuntimeHealthMonitor(
        active_strategies=8,
        signals_per_hour=15
    )
    result = monitor.run_check()
    
    assert result.module == "strategy_runtime"
    assert result.status == HealthStatus.HEALTHY


def test_strategy_runtime_monitor_critical():
    """Test strategy runtime monitor in critical state."""
    monitor = FakeStrategyRuntimeHealthMonitor(
        active_strategies=0,
        signals_per_hour=0
    )
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL
    assert "no active" in result.message.lower()


def test_msc_monitor_healthy():
    """Test MSC AI monitor in healthy state."""
    monitor = FakeMSCHealthMonitor(last_update_minutes_ago=20)
    result = monitor.run_check()
    
    assert result.module == "msc_ai"
    assert result.status == HealthStatus.HEALTHY


def test_msc_monitor_critical():
    """Test MSC AI monitor in critical state."""
    monitor = FakeMSCHealthMonitor(last_update_minutes_ago=90)
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL


def test_clm_monitor_healthy():
    """Test CLM monitor in healthy state."""
    monitor = FakeCLMHealthMonitor(last_retrain_hours_ago=24)
    result = monitor.run_check()
    
    assert result.module == "continuous_learning"
    assert result.status == HealthStatus.HEALTHY


def test_clm_monitor_critical():
    """Test CLM monitor in critical state."""
    monitor = FakeCLMHealthMonitor(last_retrain_hours_ago=200)
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL
    assert "week" in result.message.lower()


def test_opportunity_ranker_monitor_healthy():
    """Test OpportunityRanker monitor in healthy state."""
    monitor = FakeOpportunityRankerHealthMonitor(
        ranking_age_minutes=3,
        symbols_ranked=20
    )
    result = monitor.run_check()
    
    assert result.module == "opportunity_ranker"
    assert result.status == HealthStatus.HEALTHY


def test_opportunity_ranker_monitor_critical():
    """Test OpportunityRanker monitor in critical state."""
    monitor = FakeOpportunityRankerHealthMonitor(
        ranking_age_minutes=20,
        symbols_ranked=2
    )
    result = monitor.run_check()
    
    assert result.status == HealthStatus.CRITICAL


# ============================================================================
# Test System Health Monitor
# ============================================================================

def test_system_health_monitor_all_healthy():
    """Test SystemHealthMonitor with all modules healthy."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),
        FakeExecutionHealthMonitor(execution_latency_ms=100, failed_orders_rate=0.01),
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    
    summary = shm.run()
    
    assert summary.status == HealthStatus.HEALTHY
    assert len(summary.healthy_modules) == 3
    assert len(summary.warning_modules) == 0
    assert len(summary.failed_modules) == 0
    assert summary.total_checks == 3


def test_system_health_monitor_with_warnings():
    """Test SystemHealthMonitor with some warnings."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=250, missing_candles=5),  # WARNING
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),         # HEALTHY
        FakeExecutionHealthMonitor(execution_latency_ms=100, failed_orders_rate=0.01),  # HEALTHY
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store, warning_threshold=1)
    
    summary = shm.run()
    
    assert summary.status == HealthStatus.WARNING
    assert len(summary.warning_modules) == 1
    assert "market_data" in summary.warning_modules


def test_system_health_monitor_with_critical():
    """Test SystemHealthMonitor with critical failures."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=600, missing_candles=15),  # CRITICAL
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),          # HEALTHY
        FakeExecutionHealthMonitor(execution_latency_ms=1500, failed_orders_rate=0.15),  # CRITICAL
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store, critical_threshold=1)
    
    summary = shm.run()
    
    assert summary.status == HealthStatus.CRITICAL
    assert len(summary.failed_modules) == 2
    assert "market_data" in summary.failed_modules
    assert "execution" in summary.failed_modules


def test_system_health_monitor_policy_store_write():
    """Test that SystemHealthMonitor writes to PolicyStore."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store, enable_auto_write=True)
    
    summary = shm.run()
    
    # Check PolicyStore was updated
    data = policy_store.get()
    assert "system_health" in data
    assert data["system_health"]["status"] == "HEALTHY"
    assert data["system_health"]["total_checks"] == 1


def test_system_health_monitor_no_auto_write():
    """Test SystemHealthMonitor with auto-write disabled."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store, enable_auto_write=False)
    
    summary = shm.run()
    
    # PolicyStore should be empty
    data = policy_store.get()
    assert "system_health" not in data


def test_system_health_monitor_threshold_configuration():
    """Test SystemHealthMonitor with custom thresholds."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=250, missing_candles=5),  # WARNING
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),         # HEALTHY
        FakeExecutionHealthMonitor(execution_latency_ms=600, failed_orders_rate=0.06),  # WARNING
    ]
    
    policy_store = MockPolicyStore()
    
    # With high warning threshold, should stay HEALTHY
    shm = SystemHealthMonitor(monitors, policy_store, warning_threshold=5)
    summary = shm.run()
    assert summary.status == HealthStatus.HEALTHY
    
    # With low warning threshold, should be WARNING
    shm = SystemHealthMonitor(monitors, policy_store, warning_threshold=1)
    summary = shm.run()
    assert summary.status == HealthStatus.WARNING


def test_system_health_monitor_get_last_summary():
    """Test retrieving last summary."""
    monitors = [FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0)]
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    
    # Initially None
    assert shm.get_last_summary() is None
    
    # After run, should have summary
    summary = shm.run()
    last = shm.get_last_summary()
    assert last is not None
    assert last.status == summary.status


def test_system_health_monitor_get_module_status():
    """Test querying status of specific modules."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),    # HEALTHY
        FakePolicyStoreHealthMonitor(last_update_age_seconds=400),         # WARNING
        FakeExecutionHealthMonitor(execution_latency_ms=1500, failed_orders_rate=0.15),  # CRITICAL
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    shm.run()
    
    assert shm.get_module_status("market_data") == HealthStatus.HEALTHY
    assert shm.get_module_status("policy_store") == HealthStatus.WARNING
    assert shm.get_module_status("execution") == HealthStatus.CRITICAL
    assert shm.get_module_status("nonexistent") is None


def test_system_health_monitor_history():
    """Test health check history tracking."""
    monitors = [FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0)]
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    
    # Run multiple checks
    for _ in range(5):
        shm.run()
    
    history = shm.get_history(limit=3)
    assert len(history) == 3
    
    full_history = shm.get_history(limit=10)
    assert len(full_history) == 5


def test_system_health_monitor_monitor_exception_handling():
    """Test handling of monitor exceptions."""
    
    class FaultyMonitor:
        def run_check(self):
            raise RuntimeError("Monitor crashed!")
    
    monitors = [
        FaultyMonitor(),
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store, critical_threshold=1)
    
    summary = shm.run()
    
    # Should mark system as CRITICAL due to crashed monitor
    assert summary.status == HealthStatus.CRITICAL
    assert summary.total_checks == 2


def test_system_health_monitor_empty_monitors():
    """Test SystemHealthMonitor with no monitors."""
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor([], policy_store)
    
    summary = shm.run()
    
    assert summary.status == HealthStatus.HEALTHY
    assert summary.total_checks == 0
    assert len(summary.healthy_modules) == 0


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_system_integration_healthy():
    """Integration test with all monitors in healthy state."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),
        FakeExecutionHealthMonitor(execution_latency_ms=100, failed_orders_rate=0.01),
        FakeStrategyRuntimeHealthMonitor(active_strategies=8, signals_per_hour=15),
        FakeMSCHealthMonitor(last_update_minutes_ago=20),
        FakeCLMHealthMonitor(last_retrain_hours_ago=24),
        FakeOpportunityRankerHealthMonitor(ranking_age_minutes=3, symbols_ranked=20),
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    
    summary = shm.run()
    
    assert summary.status == HealthStatus.HEALTHY
    assert summary.total_checks == 7
    assert len(summary.healthy_modules) == 7
    
    # Check PolicyStore
    data = policy_store.get()
    assert data["system_health"]["status"] == "HEALTHY"
    assert len(data["system_health"]["details"]["check_results"]) == 7


def test_full_system_integration_mixed_states():
    """Integration test with mixed health states."""
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=250, missing_candles=5),     # WARNING
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),            # HEALTHY
        FakeExecutionHealthMonitor(execution_latency_ms=1500, failed_orders_rate=0.15),  # CRITICAL
        FakeStrategyRuntimeHealthMonitor(active_strategies=2, signals_per_hour=0.5),     # WARNING
        FakeMSCHealthMonitor(last_update_minutes_ago=20),                    # HEALTHY
        FakeCLMHealthMonitor(last_retrain_hours_ago=200),                    # CRITICAL
        FakeOpportunityRankerHealthMonitor(ranking_age_minutes=3, symbols_ranked=20),    # HEALTHY
    ]
    
    policy_store = MockPolicyStore()
    shm = SystemHealthMonitor(
        monitors,
        policy_store,
        critical_threshold=1,
        warning_threshold=1
    )
    
    summary = shm.run()
    
    assert summary.status == HealthStatus.CRITICAL
    assert len(summary.failed_modules) == 2
    assert len(summary.warning_modules) == 2
    assert len(summary.healthy_modules) == 3
    
    assert "execution" in summary.failed_modules
    assert "continuous_learning" in summary.failed_modules
    assert "market_data" in summary.warning_modules
    assert "strategy_runtime" in summary.warning_modules


def test_base_health_monitor_heartbeat_check():
    """Test BaseHealthMonitor heartbeat checking logic."""
    
    class TestMonitor(BaseHealthMonitor):
        def __init__(self, last_heartbeat: datetime | None):
            super().__init__("test")
            self.test_heartbeat = last_heartbeat
        
        def _perform_check(self):
            status, message = self._check_heartbeat(self.test_heartbeat)
            return HealthCheckResult(
                module=self.module_name,
                status=status,
                message=message
            )
    
    # Fresh heartbeat
    monitor = TestMonitor(datetime.utcnow() - timedelta(seconds=30))
    result = monitor.run_check()
    assert result.status == HealthStatus.HEALTHY
    
    # Aging heartbeat
    monitor = TestMonitor(datetime.utcnow() - timedelta(minutes=3))
    result = monitor.run_check()
    assert result.status == HealthStatus.WARNING
    
    # Stale heartbeat
    monitor = TestMonitor(datetime.utcnow() - timedelta(minutes=10))
    result = monitor.run_check()
    assert result.status == HealthStatus.CRITICAL
    
    # No heartbeat
    monitor = TestMonitor(None)
    result = monitor.run_check()
    assert result.status == HealthStatus.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
