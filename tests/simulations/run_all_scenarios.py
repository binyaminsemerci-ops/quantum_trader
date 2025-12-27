"""
Failure Simulation Test Suite - Master Runner
Sprint 3 Part 3

Runs all 5 failure scenarios and generates comprehensive report.

Usage:
    python tests/simulations/run_all_scenarios.py
    
    Or with pytest:
    pytest tests/simulations/ -v -s --tb=short
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

from harness import (
    FailureSimulationHarness,
    FlashCrashConfig,
    RedisDownConfig,
    BinanceDownConfig,
    SignalFloodConfig,
    ESSTriggeredConfig,
    ScenarioStatus
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulations.log')
    ]
)

logger = logging.getLogger(__name__)


class SimulationRunner:
    """Orchestrates all failure simulation scenarios"""
    
    def __init__(self):
        self.harness = FailureSimulationHarness()
        self.results = []
    
    async def run_all_scenarios(self):
        """Run all 5 failure scenarios in sequence"""
        
        logger.info("="*70)
        logger.info("FAILURE SIMULATION TEST SUITE - Sprint 3 Part 3")
        logger.info("="*70)
        logger.info(f"Start time: {datetime.now(timezone.utc).isoformat()}")
        logger.info("")
        
        # Scenario 1: Flash Crash
        logger.info(">>> SCENARIO 1: FLASH CRASH")
        logger.info("-" * 70)
        result1 = await self.harness.run_flash_crash_scenario(
            FlashCrashConfig(
                price_drop_percent=15.0,
                duration_seconds=60.0,
                ess_drawdown_threshold=10.0
            )
        )
        self._print_scenario_result(result1)
        self.results.append(result1)
        
        await asyncio.sleep(2)  # Brief pause between scenarios
        
        # Scenario 2: Redis Down
        logger.info("\n>>> SCENARIO 2: REDIS DOWN")
        logger.info("-" * 70)
        result2 = await self.harness.run_redis_down_scenario(
            RedisDownConfig(
                downtime_seconds=60.0,
                publish_attempts_during_downtime=10
            )
        )
        self._print_scenario_result(result2)
        self.results.append(result2)
        
        await asyncio.sleep(2)
        
        # Scenario 3: Binance Down
        logger.info("\n>>> SCENARIO 3: BINANCE API DOWN")
        logger.info("-" * 70)
        result3 = await self.harness.run_binance_down_scenario(
            BinanceDownConfig(
                error_codes=[-1003, -1015],
                trade_attempts=5,
                expected_retries_per_attempt=3
            )
        )
        self._print_scenario_result(result3)
        self.results.append(result3)
        
        await asyncio.sleep(2)
        
        # Scenario 4: Signal Flood
        logger.info("\n>>> SCENARIO 4: SIGNAL FLOOD")
        logger.info("-" * 70)
        result4 = await self.harness.run_signal_flood_scenario(
            SignalFloodConfig(
                signal_count=50,
                publish_interval_ms=100.0
            )
        )
        self._print_scenario_result(result4)
        self.results.append(result4)
        
        await asyncio.sleep(2)
        
        # Scenario 5: ESS Trigger
        logger.info("\n>>> SCENARIO 5: ESS TRIGGER & RECOVERY")
        logger.info("-" * 70)
        result5 = await self.harness.run_ess_trigger_scenario(
            ESSTriggeredConfig(
                initial_balance=10000.0,
                loss_amount=1200.0,
                ess_threshold_percent=10.0
            )
        )
        self._print_scenario_result(result5)
        self.results.append(result5)
        
        # Generate final report
        logger.info("\n" + "="*70)
        logger.info("FINAL REPORT")
        logger.info("="*70)
        
        summary = self.harness.get_summary_report()
        self._print_summary_report(summary)
        
        # Save report to file
        self._save_report(summary)
        
        logger.info(f"\nEnd time: {datetime.now(timezone.utc).isoformat()}")
        logger.info("="*70)
        
        return summary
    
    def _print_scenario_result(self, result):
        """Print detailed scenario result"""
        status_emoji = {
            ScenarioStatus.PASSED: "✅",
            ScenarioStatus.FAILED: "❌",
            ScenarioStatus.DEGRADED: "⚠️",
            ScenarioStatus.NOT_STARTED: "⏳"
        }
        
        logger.info(f"Status: {status_emoji.get(result.status, '?')} {result.status.value.upper()}")
        logger.info(f"Duration: {result.duration_seconds:.2f}s")
        logger.info(f"Checks: {result.checks_passed} passed, {result.checks_failed} failed")
        
        if result.observations:
            logger.info("\nKey observations:")
            for obs in result.observations[:10]:  # First 10 observations
                logger.info(f"  • {obs}")
            if len(result.observations) > 10:
                logger.info(f"  ... and {len(result.observations) - 10} more")
        
        if result.errors:
            logger.error("\nErrors:")
            for err in result.errors:
                logger.error(f"  ✗ {err}")
        
        if result.metrics:
            logger.info("\nMetrics:")
            for key, value in result.metrics.items():
                logger.info(f"  {key}: {value}")
    
    def _print_summary_report(self, summary):
        """Print summary report"""
        logger.info(f"\nTotal scenarios: {summary['summary']['total_scenarios']}")
        logger.info(f"✅ Passed: {summary['summary']['passed']}")
        logger.info(f"❌ Failed: {summary['summary']['failed']}")
        logger.info(f"⚠️  Degraded: {summary['summary']['degraded']}")
        logger.info(f"Pass rate: {summary['summary']['pass_rate']:.1f}%")
        
        logger.info(f"\nTotal checks: {summary['checks']['total_passed'] + summary['checks']['total_failed']}")
        logger.info(f"✓ Passed: {summary['checks']['total_passed']}")
        logger.info(f"✗ Failed: {summary['checks']['total_failed']}")
        logger.info(f"Success rate: {summary['checks']['success_rate']:.1f}%")
        
        logger.info("\nGlobal metrics:")
        for key, value in summary['metrics'].items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\nScenario breakdown:")
        for scenario in summary['scenarios']:
            status_emoji = "✅" if scenario['status'] == "passed" else "❌"
            logger.info(f"  {status_emoji} {scenario['name']}: {scenario['status']} "
                       f"({scenario['duration']}, {scenario['checks']} checks)")
    
    def _save_report(self, summary):
        """Save report to JSON file"""
        report_file = Path("tests/simulations/simulation_report.json")
        
        # Add timestamp
        summary['report_generated'] = datetime.now(timezone.utc).isoformat()
        summary['detailed_results'] = [
            {
                "scenario": r.scenario_name,
                "status": r.status.value,
                "duration_seconds": r.duration_seconds,
                "checks_passed": r.checks_passed,
                "checks_failed": r.checks_failed,
                "observations": r.observations,
                "errors": r.errors,
                "metrics": r.metrics
            }
            for r in self.results
        ]
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n📄 Full report saved to: {report_file}")


async def main():
    """Main entry point"""
    runner = SimulationRunner()
    
    try:
        summary = await runner.run_all_scenarios()
        
        # Exit code based on results
        if summary['summary']['failed'] > 0:
            logger.error("\n❌ Some scenarios FAILED")
            return 1
        elif summary['summary']['degraded'] > 0:
            logger.warning("\n⚠️  Some scenarios DEGRADED")
            return 0
        else:
            logger.info("\n✅ All scenarios PASSED")
            return 0
    
    except Exception as e:
        logger.error(f"\n❌ Runner failed with exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
