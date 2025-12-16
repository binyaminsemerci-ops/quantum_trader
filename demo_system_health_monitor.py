"""
System Health Monitor - Complete Demo

Demonstrates full SystemHealthMonitor capabilities with realistic scenarios:
1. All systems healthy
2. Some warnings
3. Critical failures
4. PolicyStore integration
5. History tracking
"""

from datetime import datetime
from backend.services.monitoring.system_health_monitor import (
    SystemHealthMonitor,
    FakeMarketDataHealthMonitor,
    FakePolicyStoreHealthMonitor,
    FakeExecutionHealthMonitor,
    FakeStrategyRuntimeHealthMonitor,
    FakeMSCHealthMonitor,
    FakeCLMHealthMonitor,
    FakeOpportunityRankerHealthMonitor,
)


# ============================================================================
# Mock PolicyStore (minimal implementation)
# ============================================================================

class InMemoryPolicyStore:
    """Simple in-memory PolicyStore for demo."""
    
    def __init__(self):
        self._data = {
            "risk_mode": "NORMAL",
            "max_risk_per_trade": 0.01,
            "max_positions": 5,
        }
    
    def patch(self, updates: dict) -> None:
        """Update policy with partial changes."""
        self._data.update(updates)
        print(f"✅ PolicyStore updated: {list(updates.keys())}")
    
    def get(self) -> dict:
        """Get full policy state."""
        return self._data.copy()


# ============================================================================
# Demo Scenarios
# ============================================================================

def demo_scenario_1_all_healthy():
    """Scenario 1: All systems operating normally."""
    print("\n" + "="*70)
    print("📊 SCENARIO 1: All Systems Healthy")
    print("="*70)
    
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),
        FakeExecutionHealthMonitor(execution_latency_ms=100, failed_orders_rate=0.01),
        FakeStrategyRuntimeHealthMonitor(active_strategies=8, signals_per_hour=15),
        FakeMSCHealthMonitor(last_update_minutes_ago=20),
        FakeCLMHealthMonitor(last_retrain_hours_ago=24),
        FakeOpportunityRankerHealthMonitor(ranking_age_minutes=3, symbols_ranked=20),
    ]
    
    policy_store = InMemoryPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    
    summary = shm.run()
    
    print(f"\n📈 System Status: {summary.status.value}")
    print(f"   Total Checks: {summary.total_checks}")
    print(f"   ✅ Healthy: {len(summary.healthy_modules)}")
    print(f"   ⚠️  Warnings: {len(summary.warning_modules)}")
    print(f"   🚨 Critical: {len(summary.failed_modules)}")
    
    print("\n📋 Healthy Modules:")
    for module in summary.healthy_modules:
        print(f"   • {module}")
    
    print("\n📊 PolicyStore State:")
    health_data = policy_store.get()["system_health"]
    print(f"   Status: {health_data['status']}")
    print(f"   Timestamp: {health_data['timestamp']}")


def demo_scenario_2_warnings():
    """Scenario 2: Some systems showing warnings."""
    print("\n" + "="*70)
    print("⚠️  SCENARIO 2: System Degraded (Warnings)")
    print("="*70)
    
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=250, missing_candles=5),     # WARNING
        FakePolicyStoreHealthMonitor(last_update_age_seconds=400),          # WARNING
        FakeExecutionHealthMonitor(execution_latency_ms=100, failed_orders_rate=0.01),  # HEALTHY
        FakeStrategyRuntimeHealthMonitor(active_strategies=2, signals_per_hour=0.8),    # WARNING
        FakeMSCHealthMonitor(last_update_minutes_ago=20),                   # HEALTHY
        FakeCLMHealthMonitor(last_retrain_hours_ago=80),                    # WARNING
        FakeOpportunityRankerHealthMonitor(ranking_age_minutes=3, symbols_ranked=20),   # HEALTHY
    ]
    
    policy_store = InMemoryPolicyStore()
    shm = SystemHealthMonitor(
        monitors,
        policy_store,
        warning_threshold=2  # 2+ warnings → system WARNING
    )
    
    summary = shm.run()
    
    print(f"\n📈 System Status: {summary.status.value}")
    print(f"   Total Checks: {summary.total_checks}")
    print(f"   ✅ Healthy: {len(summary.healthy_modules)}")
    print(f"   ⚠️  Warnings: {len(summary.warning_modules)}")
    print(f"   🚨 Critical: {len(summary.failed_modules)}")
    
    print("\n⚠️  Warning Modules:")
    for module in summary.warning_modules:
        print(f"   • {module}")
    
    print("\n✅ Healthy Modules:")
    for module in summary.healthy_modules:
        print(f"   • {module}")


def demo_scenario_3_critical():
    """Scenario 3: Critical system failures."""
    print("\n" + "="*70)
    print("🚨 SCENARIO 3: Critical System Failures")
    print("="*70)
    
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=600, missing_candles=20),    # CRITICAL
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),           # HEALTHY
        FakeExecutionHealthMonitor(execution_latency_ms=1500, failed_orders_rate=0.15),  # CRITICAL
        FakeStrategyRuntimeHealthMonitor(active_strategies=0, signals_per_hour=0),       # CRITICAL
        FakeMSCHealthMonitor(last_update_minutes_ago=20),                   # HEALTHY
        FakeCLMHealthMonitor(last_retrain_hours_ago=200),                   # CRITICAL
        FakeOpportunityRankerHealthMonitor(ranking_age_minutes=25, symbols_ranked=2),    # CRITICAL
    ]
    
    policy_store = InMemoryPolicyStore()
    shm = SystemHealthMonitor(
        monitors,
        policy_store,
        critical_threshold=1  # 1+ critical → system CRITICAL
    )
    
    summary = shm.run()
    
    print(f"\n📈 System Status: {summary.status.value}")
    print(f"   Total Checks: {summary.total_checks}")
    print(f"   ✅ Healthy: {len(summary.healthy_modules)}")
    print(f"   ⚠️  Warnings: {len(summary.warning_modules)}")
    print(f"   🚨 Critical: {len(summary.failed_modules)}")
    
    print("\n🚨 CRITICAL FAILURES:")
    for module in summary.failed_modules:
        print(f"   • {module}")
        # Find details
        for result in summary.details["check_results"]:
            if result["module"] == module:
                print(f"     → {result['message']}")
    
    print("\n✅ Still Healthy:")
    for module in summary.healthy_modules:
        print(f"   • {module}")


def demo_scenario_4_policy_store_integration():
    """Scenario 4: Demonstrate PolicyStore integration."""
    print("\n" + "="*70)
    print("🔄 SCENARIO 4: PolicyStore Integration")
    print("="*70)
    
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),
        FakePolicyStoreHealthMonitor(last_update_age_seconds=30),
        FakeExecutionHealthMonitor(execution_latency_ms=100, failed_orders_rate=0.01),
    ]
    
    policy_store = InMemoryPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store, enable_auto_write=True)
    
    print("\n1️⃣ Running first health check...")
    summary1 = shm.run()
    
    print("\n📊 PolicyStore Contents:")
    data = policy_store.get()
    print(f"   system_health.status: {data['system_health']['status']}")
    print(f"   system_health.total_checks: {data['system_health']['total_checks']}")
    print(f"   system_health.timestamp: {data['system_health']['timestamp']}")
    
    print("\n2️⃣ Running second health check...")
    summary2 = shm.run()
    
    print("\n📊 Updated PolicyStore:")
    data = policy_store.get()
    print(f"   system_health.status: {data['system_health']['status']}")
    print(f"   system_health.timestamp: {data['system_health']['timestamp']}")
    
    print("\n📜 Check History:")
    history = shm.get_history(limit=5)
    for i, h in enumerate(history, 1):
        print(f"   {i}. {h.status.value} @ {h.timestamp.strftime('%H:%M:%S')}")


def demo_scenario_5_query_module_status():
    """Scenario 5: Query individual module status."""
    print("\n" + "="*70)
    print("🔍 SCENARIO 5: Query Module Status")
    print("="*70)
    
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=50, missing_candles=0),      # HEALTHY
        FakePolicyStoreHealthMonitor(last_update_age_seconds=400),          # WARNING
        FakeExecutionHealthMonitor(execution_latency_ms=1500, failed_orders_rate=0.15),  # CRITICAL
        FakeMSCHealthMonitor(last_update_minutes_ago=20),                   # HEALTHY
    ]
    
    policy_store = InMemoryPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    shm.run()
    
    print("\n🔍 Querying Individual Module Status:\n")
    
    modules = ["market_data", "policy_store", "execution", "msc_ai", "nonexistent"]
    
    for module in modules:
        status = shm.get_module_status(module)
        if status:
            emoji = {"HEALTHY": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}[status.value]
            print(f"   {emoji} {module:20s} → {status.value}")
        else:
            print(f"   ❓ {module:20s} → NOT FOUND")


def demo_scenario_6_detailed_diagnostics():
    """Scenario 6: View detailed diagnostics from health checks."""
    print("\n" + "="*70)
    print("🔬 SCENARIO 6: Detailed Diagnostics")
    print("="*70)
    
    monitors = [
        FakeMarketDataHealthMonitor(latency_ms=250, missing_candles=5),
        FakeExecutionHealthMonitor(execution_latency_ms=600, failed_orders_rate=0.08),
        FakeStrategyRuntimeHealthMonitor(active_strategies=3, signals_per_hour=8),
    ]
    
    policy_store = InMemoryPolicyStore()
    shm = SystemHealthMonitor(monitors, policy_store)
    summary = shm.run()
    
    print("\n🔬 Detailed Health Check Results:\n")
    
    for result in summary.details["check_results"]:
        status_emoji = {
            "HEALTHY": "✅",
            "WARNING": "⚠️",
            "CRITICAL": "🚨"
        }[result["status"]]
        
        print(f"{status_emoji} {result['module'].upper()}")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        print(f"   Details:")
        for key, value in result["details"].items():
            print(f"      • {key}: {value}")
        print()


# ============================================================================
# Main Demo Runner
# ============================================================================

def main():
    """Run all demo scenarios."""
    print("\n" + "="*70)
    print("🚀 SYSTEM HEALTH MONITOR - COMPLETE DEMO")
    print("="*70)
    print("\nThis demo showcases the System Health Monitor in action:")
    print("  • Individual module health checks")
    print("  • System-wide status aggregation")
    print("  • PolicyStore integration")
    print("  • History tracking")
    print("  • Diagnostic queries")
    
    # Run all scenarios
    demo_scenario_1_all_healthy()
    demo_scenario_2_warnings()
    demo_scenario_3_critical()
    demo_scenario_4_policy_store_integration()
    demo_scenario_5_query_module_status()
    demo_scenario_6_detailed_diagnostics()
    
    print("\n" + "="*70)
    print("✅ Demo Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. SHM monitors 7+ critical subsystems continuously")
    print("  2. Aggregates health into HEALTHY/WARNING/CRITICAL states")
    print("  3. Writes status to PolicyStore for system-wide coordination")
    print("  4. Enables emergency shutdown and self-healing behavior")
    print("  5. Provides detailed diagnostics for debugging")
    print("\n🎯 The System Health Monitor is the 'nervous system' of Quantum Trader!")
    print()


if __name__ == "__main__":
    main()
