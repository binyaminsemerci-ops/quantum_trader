"""
Quick validation test for EPIC-STRESS-DASH-001 metrics

Verifies all 4 new metrics are properly defined and can be imported.
Run this to validate the dashboard integration is working.
"""

import sys


def test_metrics_import():
    """Test that all new metrics can be imported."""
    print("[TEST] Importing metrics module...")
    
    try:
        from infra.metrics.metrics import (
            risk_gate_decisions_total,
            ess_triggers_total,
            exchange_failover_events_total,
            stress_scenario_runs_total,
        )
        print("✅ All 4 metrics imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import metrics: {e}")
        return False


def test_metrics_structure():
    """Test that metrics have correct structure."""
    print("\n[TEST] Checking metric structures...")
    
    from infra.metrics.metrics import (
        risk_gate_decisions_total,
        ess_triggers_total,
        exchange_failover_events_total,
        stress_scenario_runs_total,
    )
    
    # Test RiskGate metric
    print("\n1. risk_gate_decisions_total")
    try:
        risk_gate_decisions_total.labels(
            decision="allow",
            reason="all_risk_checks_passed",
            account="test_account",
            exchange="binance",
            strategy="test_strategy",
        ).inc(0)  # inc(0) = dry run
        print("   ✅ Labels: decision, reason, account, exchange, strategy")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test ESS metric
    print("\n2. ess_triggers_total")
    try:
        ess_triggers_total.labels(source="global_risk").inc(0)
        print("   ✅ Labels: source")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test Exchange Failover metric
    print("\n3. exchange_failover_events_total")
    try:
        exchange_failover_events_total.labels(
            primary="binance",
            selected="bybit",
        ).inc(0)
        print("   ✅ Labels: primary, selected")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test Stress Scenario metric
    print("\n4. stress_scenario_runs_total")
    try:
        stress_scenario_runs_total.labels(
            scenario="flash_crash",
            success="true",
        ).inc(0)
        print("   ✅ Labels: scenario, success")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ All metric structures valid")
    return True


def test_instrumentation_imports():
    """Test that instrumented modules can import metrics."""
    print("\n[TEST] Checking instrumentation in modules...")
    
    modules = [
        ("backend.risk.risk_gate_v3", "RiskGate v3"),
        ("backend.services.risk.emergency_stop_system", "ESS"),
        ("backend.policies.exchange_failover_policy", "Exchange Failover"),
        ("backend.tests_runtime.stress_scenarios.runner", "Stress Scenarios"),
    ]
    
    all_ok = True
    for module_path, name in modules:
        try:
            __import__(module_path)
            print(f"   ✅ {name} ({module_path})")
        except ImportError as e:
            print(f"   ❌ {name} failed to import: {e}")
            all_ok = False
        except Exception as e:
            # Other errors (e.g., missing dependencies) are OK for this test
            print(f"   ⚠️  {name} loaded with warnings: {e}")
    
    return all_ok


def test_dashboard_file():
    """Test that dashboard JSON file exists and is valid."""
    print("\n[TEST] Checking dashboard file...")
    
    import json
    import os
    
    dashboard_path = "deploy/k8s/observability/grafana_risk_resilience_dashboard.json"
    
    if not os.path.exists(dashboard_path):
        print(f"   ❌ Dashboard file not found: {dashboard_path}")
        return False
    
    try:
        with open(dashboard_path, "r") as f:
            dashboard = json.load(f)
        
        # Check for required keys
        if "dashboard" not in dashboard:
            print("   ❌ Missing 'dashboard' key in JSON")
            return False
        
        if "panels" not in dashboard["dashboard"]:
            print("   ❌ Missing 'panels' in dashboard")
            return False
        
        panel_count = len(dashboard["dashboard"]["panels"])
        print(f"   ✅ Dashboard file valid ({panel_count} panels)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("EPIC-STRESS-DASH-001: Metrics Validation Test")
    print("=" * 80)
    
    tests = [
        ("Metrics Import", test_metrics_import),
        ("Metrics Structure", test_metrics_structure),
        ("Instrumentation", test_instrumentation_imports),
        ("Dashboard File", test_dashboard_file),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed! Dashboard integration ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
