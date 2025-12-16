"""
Stress Test Runner - Executes batch scenario testing
"""

import logging
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .scenario_models import Scenario, ScenarioResult
from .scenario_loader import ScenarioLoader
from .scenario_transformer import ScenarioTransformer
from .scenario_executor import ScenarioExecutor

logger = logging.getLogger(__name__)


class StressTestRunner:
    """
    Orchestrates batch stress testing.
    
    Runs multiple scenarios in parallel, aggregates results,
    and provides statistical summaries.
    """
    
    def __init__(
        self,
        executor: ScenarioExecutor,
        loader: ScenarioLoader | None = None,
        transformer: ScenarioTransformer | None = None,
        max_workers: int = 4
    ):
        """
        Initialize runner.
        
        Args:
            executor: Configured scenario executor
            loader: Scenario data loader
            transformer: Scenario transformer
            max_workers: Max parallel scenarios
        """
        self.executor = executor
        self.loader = loader or ScenarioLoader()
        self.transformer = transformer or ScenarioTransformer()
        self.max_workers = max_workers
        
        logger.info("[SST] StressTestRunner initialized")
    
    def run_batch(
        self,
        scenarios: list[Scenario],
        parallel: bool = True
    ) -> dict[str, ScenarioResult]:
        """
        Run multiple scenarios.
        
        Args:
            scenarios: List of scenario definitions
            parallel: Whether to run in parallel
            
        Returns:
            Dictionary mapping scenario name to result
        """
        logger.info(f"[SST] Running batch of {len(scenarios)} scenarios")
        
        results = {}
        
        if parallel and len(scenarios) > 1:
            # Run in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_scenario = {
                    executor.submit(self._run_single, scenario): scenario
                    for scenario in scenarios
                }
                
                for future in as_completed(future_to_scenario):
                    scenario = future_to_scenario[future]
                    try:
                        result = future.result()
                        results[scenario.name] = result
                        logger.info(f"[SST] Completed: {scenario.name}")
                    except Exception as e:
                        logger.error(f"[SST] Failed: {scenario.name} - {e}")
                        results[scenario.name] = ScenarioResult(
                            scenario_name=scenario.name,
                            success=False,
                            notes=[f"Execution failed: {e}"]
                        )
        else:
            # Run sequentially
            for scenario in scenarios:
                try:
                    result = self._run_single(scenario)
                    results[scenario.name] = result
                    logger.info(f"[SST] Completed: {scenario.name}")
                except Exception as e:
                    logger.error(f"[SST] Failed: {scenario.name} - {e}")
                    results[scenario.name] = ScenarioResult(
                        scenario_name=scenario.name,
                        success=False,
                        notes=[f"Execution failed: {e}"]
                    )
        
        logger.info(f"[SST] Batch complete: {len(results)}/{len(scenarios)} successful")
        return results
    
    def _run_single(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario"""
        # Load data
        df = self.loader.load_data(scenario)
        
        # Validate data
        is_valid, issues = self.loader.validate_data(df)
        if not is_valid and scenario.type != "data_corruption":
            logger.warning(f"[SST] Data quality issues in {scenario.name}: {issues}")
        
        # Apply transformations
        df = self.transformer.apply(df, scenario)
        
        # Execute simulation
        result = self.executor.run(df, scenario)
        
        return result
    
    def generate_summary(self, results: dict[str, ScenarioResult]) -> dict[str, Any]:
        """
        Generate aggregate statistics across all scenarios.
        
        Args:
            results: Dictionary of scenario results
            
        Returns:
            Summary statistics
        """
        if not results:
            return {"error": "No results to summarize"}
        
        successful = [r for r in results.values() if r.success]
        
        summary = {
            "total_scenarios": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "total_trades": sum(r.total_trades for r in successful),
            "total_emergency_stops": sum(r.emergency_stops for r in successful),
            "total_execution_failures": sum(r.execution_failures for r in successful),
            "avg_winrate": sum(r.winrate for r in successful) / len(successful) if successful else 0,
            "avg_max_drawdown": sum(r.max_drawdown for r in successful) / len(successful) if successful else 0,
            "avg_profit_factor": sum(r.profit_factor for r in successful) / len(successful) if successful else 0,
            "worst_drawdown": max((r.max_drawdown for r in successful), default=0),
            "best_winrate": max((r.winrate for r in successful), default=0),
            "worst_winrate": min((r.winrate for r in successful), default=0),
            "scenarios_with_ess": sum(1 for r in successful if r.emergency_stops > 0),
            "unique_failed_models": len(set(
                model
                for r in successful
                for model in r.failed_models
            )),
            "unique_failed_strategies": len(set(
                strategy
                for r in successful
                for strategy in r.failed_strategies
            ))
        }
        
        return summary
    
    def print_summary(self, results: dict[str, ScenarioResult]):
        """Print human-readable summary"""
        summary = self.generate_summary(results)
        
        print("\n" + "="*70)
        print("STRESS TEST SUMMARY")
        print("="*70)
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"\nTotal Trades: {summary['total_trades']}")
        print(f"Avg Win Rate: {summary['avg_winrate']:.1f}%")
        print(f"Avg Max Drawdown: {summary['avg_max_drawdown']:.2f}%")
        print(f"Worst Drawdown: {summary['worst_drawdown']:.2f}%")
        print(f"Avg Profit Factor: {summary['avg_profit_factor']:.2f}")
        print(f"\nEmergency Stops: {summary['total_emergency_stops']}")
        print(f"Execution Failures: {summary['total_execution_failures']}")
        print(f"Scenarios with ESS: {summary['scenarios_with_ess']}")
        print(f"Unique Failed Models: {summary['unique_failed_models']}")
        print(f"Unique Failed Strategies: {summary['unique_failed_strategies']}")
        print("="*70)
        
        # Individual scenario results
        print("\nINDIVIDUAL RESULTS:")
        print("-"*70)
        for name, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"{status} {name}")
            print(f"   Trades: {result.total_trades}, Win Rate: {result.winrate:.1f}%, Max DD: {result.max_drawdown:.2f}%")
            if result.emergency_stops > 0:
                print(f"   ⚠ ESS Activations: {result.emergency_stops}")
            if result.execution_failures > 0:
                print(f"   ⚠ Execution Failures: {result.execution_failures}")
        print("="*70 + "\n")
