"""
Stress Scenario Runner

EPIC-STRESS-001: Execution engine for stress test scenarios.

Provides run_scenario() and run_all_scenarios() functions to execute
registered scenarios and collect results.

Instrumented with Prometheus metrics for dashboard visibility (EPIC-STRESS-DASH-001).
"""

import asyncio
import logging
import time
from typing import List, Optional
from .types import ScenarioResult
from .scenarios import SCENARIOS

# EPIC-STRESS-DASH-001: Prometheus metrics for dashboard visibility
try:
    from infra.metrics.metrics import stress_scenario_runs_total
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


async def run_scenario(name: str) -> ScenarioResult:
    """
    Execute a single stress scenario by name.
    
    Args:
        name: Scenario identifier (must be registered in SCENARIOS)
        
    Returns:
        ScenarioResult with outcome and metrics
        
    Example:
        result = await run_scenario("flash_crash")
        print(result)  # Prints formatted result with ✅/❌
    """
    logger.info(f"[STRESS] Executing scenario: {name}")
    
    # Check if scenario exists
    scenario_fn = SCENARIOS.get(name)
    if not scenario_fn:
        logger.error(f"[STRESS] Scenario '{name}' not found in registry")
        return ScenarioResult(
            name=name,
            success=False,
            reason="scenario_not_found",
            metrics={},
            duration_sec=0.0,
        )
    
    # Execute scenario with exception handling
    start_time = time.perf_counter()
    try:
        result = await scenario_fn()
        duration = time.perf_counter() - start_time
        result.duration_sec = duration
        
        logger.info(
            f"[STRESS] Scenario '{name}' completed in {duration:.2f}s: "
            f"{'SUCCESS' if result.success else 'FAILED'} - {result.reason}"
        )
        
        # EPIC-STRESS-DASH-001: Record metric
        if METRICS_AVAILABLE and stress_scenario_runs_total:
            try:
                stress_scenario_runs_total.labels(
                    scenario=result.name,
                    success=str(result.success).lower(),
                ).inc()
            except Exception as e:
                logger.warning(f"[STRESS] Failed to record metric: {e}")
        
        return result
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.exception(f"[STRESS] Scenario '{name}' raised exception: {e}")
        
        result = ScenarioResult(
            name=name,
            success=False,
            reason=f"exception: {type(e).__name__}: {str(e)}",
            metrics={},
            duration_sec=duration,
        )
        
        # EPIC-STRESS-DASH-001: Record metric
        if METRICS_AVAILABLE and stress_scenario_runs_total:
            try:
                stress_scenario_runs_total.labels(
                    scenario=result.name,
                    success="false",
                ).inc()
            except Exception as metric_err:
                logger.warning(f"[STRESS] Failed to record metric: {metric_err}")
        
        return result


async def run_all_scenarios(
    only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[ScenarioResult]:
    """
    Execute all registered stress scenarios.
    
    Args:
        only: If provided, only run scenarios in this list
        exclude: If provided, skip scenarios in this list
        
    Returns:
        List of ScenarioResult for each executed scenario
        
    Example:
        results = await run_all_scenarios()
        for result in results:
            print(result)
            
        # Run only flash crash and outage scenarios
        results = await run_all_scenarios(only=["flash_crash", "binance_outage"])
        
        # Run all except slow scenarios
        results = await run_all_scenarios(exclude=["signal_flood_30"])
    """
    logger.info("[STRESS] Starting stress test suite")
    
    # Determine which scenarios to run
    scenario_names = list(SCENARIOS.keys())
    
    if only:
        scenario_names = [name for name in scenario_names if name in only]
        logger.info(f"[STRESS] Running only: {scenario_names}")
    
    if exclude:
        scenario_names = [name for name in scenario_names if name not in exclude]
        logger.info(f"[STRESS] Excluding: {exclude}")
    
    if not scenario_names:
        logger.warning("[STRESS] No scenarios to run")
        return []
    
    # Execute all scenarios sequentially (not parallel to avoid race conditions)
    results = []
    for name in sorted(scenario_names):
        try:
            result = await run_scenario(name)
            results.append(result)
        except Exception as e:
            # This should be caught by run_scenario, but double-check
            logger.exception(f"[STRESS] Unexpected error running '{name}': {e}")
            results.append(
                ScenarioResult(
                    name=name,
                    success=False,
                    reason=f"runner_exception: {type(e).__name__}: {str(e)}",
                    metrics={},
                    duration_sec=0.0,
                )
            )
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    
    logger.info(
        f"[STRESS] Stress test suite completed: "
        f"{passed}/{total} passed, {failed} failed"
    )
    
    return results


def print_results_summary(results: List[ScenarioResult]) -> None:
    """
    Print a formatted summary of scenario results.
    
    Args:
        results: List of scenario results to summarize
    """
    if not results:
        print("[STRESS] No results to display")
        return
    
    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS")
    print("=" * 80)
    
    for result in results:
        print(result)
    
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    
    print("=" * 80)
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    print("=" * 80 + "\n")
