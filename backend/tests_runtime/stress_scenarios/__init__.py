"""
Stress Scenarios Module

EPIC-STRESS-001: Runtime Stress & Failure Scenarios (System Hardening)

This module provides a framework for running scripted, repeatable stress scenarios
to validate system resilience under adverse conditions.

Key components:
- Scenario Registry: Register and discover stress scenarios
- Scenario Runner: Execute scenarios individually or in batch
- Result Types: Structured results with metrics

Tested Systems:
- Global Risk v3 + RiskGate (EPIC-RISK3-EXEC-001)
- ESS (Emergency Stop System)
- Multi-exchange + failover (EPIC-EXCH-ROUTING-001, EPIC-EXCH-FAIL-001)
- EventBus / microservices
- Redis/Postgres robustness (simulated)
- Observability (metrics + logs) under stress

Usage:
    from backend.tests_runtime.stress_scenarios.runner import run_all_scenarios
    
    results = await run_all_scenarios()
    for result in results:
        print(f"{result.name}: {result.success}")
"""

from .types import ScenarioResult
from .runner import run_scenario, run_all_scenarios
from .scenarios import SCENARIOS, register_scenario

__all__ = [
    "ScenarioResult",
    "run_scenario",
    "run_all_scenarios",
    "SCENARIOS",
    "register_scenario",
]
