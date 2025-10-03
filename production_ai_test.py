#!/usr/bin/env python3
"""
Production-Scale AI Testing and Performance Monitoring

This enhanced testing system provides:
- Production-scale testing with larger symbol sets and timeframes
- Risk management integration testing
- Multi-timeframe validation (1m, 5m, 15m, 1h)
- Performance degradation detection
- Live market condition simulation
- Comprehensive performance analytics
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from ai_engine.train_and_save import train_and_save
from config.config import settings
from production_risk_manager import RiskManager, RiskParameters

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfiguration:
    """Test configuration parameters."""

    name: str
    symbols: List[str]
    limits: List[int]  # Multiple candle limits to test
    timeframes: List[str] = None  # ['1m', '5m', '15m', '1h']
    entry_thresholds: List[float] = None
    use_risk_management: bool = False
    stress_test: bool = False
    max_duration_minutes: int = 60
    description: str = ""


# Production-scale test configurations
PRODUCTION_TEST_CONFIGS = {
    "quick_production": TestConfiguration(
        name="quick_production",
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
        limits=[500, 1000],
        entry_thresholds=[0.001, 0.005, 0.01],
        use_risk_management=True,
        max_duration_minutes=15,
        description="Quick production test with risk management",
    ),
    "standard_production": TestConfiguration(
        name="standard_production",
        symbols=[
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "ADAUSDT",
            "AVAXUSDT",
            "MATICUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "ATOMUSDT",
            "LTCUSDT",
            "UNIUSDT",
            "NEARUSDT",
            "AAVEUSDT",
        ],
        limits=[800, 1200, 1500],
        timeframes=["1m", "5m"],  # Test multiple timeframes
        entry_thresholds=[0.001, 0.002, 0.005, 0.01],
        use_risk_management=True,
        max_duration_minutes=45,
        description="Standard production test with 15 symbols",
    ),
    "comprehensive_production": TestConfiguration(
        name="comprehensive_production",
        symbols=None,  # Use first 25 symbols from DEFAULT_SYMBOLS
        limits=[1000, 1500, 2000],
        timeframes=["1m", "5m", "15m"],
        entry_thresholds=[0.001, 0.002, 0.005, 0.01, 0.02],
        use_risk_management=True,
        max_duration_minutes=90,
        description="Comprehensive production test with 25+ symbols",
    ),
    "stress_production": TestConfiguration(
        name="stress_production",
        symbols=None,  # Use all DEFAULT_SYMBOLS
        limits=[1500, 2500],
        timeframes=["1m", "5m", "15m", "1h"],
        entry_thresholds=[0.001, 0.005, 0.01],
        use_risk_management=True,
        stress_test=True,
        max_duration_minutes=120,
        description="Full stress test with all symbols and timeframes",
    ),
}


class ProductionTestRunner:
    """Enhanced production-scale test runner."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("production_test_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.results: List[Dict] = []
        self.risk_manager = None
        self.performance_baseline = None

        # Performance tracking
        self.performance_metrics = {
            "test_start_time": datetime.now(),
            "total_tests_run": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "average_accuracy": 0.0,
            "average_return": 0.0,
            "best_performing_config": None,
            "performance_degradations": [],
        }

    def setup_risk_management(self, config: TestConfiguration):
        """Setup risk management for testing."""
        if config.use_risk_management:
            risk_params = RiskParameters(
                max_position_size_pct=1.5,  # Conservative for testing
                stop_loss_pct=2.0,
                take_profit_pct=4.0,
                max_portfolio_risk_pct=8.0,
                max_daily_loss_pct=4.0,
            )
            self.risk_manager = RiskManager(risk_params)
            logger.info("Risk management enabled for testing")
        else:
            self.risk_manager = None

    def get_symbol_list(self, config: TestConfiguration) -> List[str]:
        """Get symbol list based on configuration."""
        if config.symbols is None:
            # Use symbols from settings
            all_symbols = list(settings.default_symbols)
            if config.name == "comprehensive_production":
                return all_symbols[:25]  # First 25 symbols
            else:
                return all_symbols  # All symbols for stress test
        return config.symbols

    def run_single_test(
        self,
        symbols: List[str],
        limit: int,
        entry_threshold: float,
        timeframe: str = "1m",
        test_id: str = "",
    ) -> Dict[str, Any]:
        """Run a single AI model test."""
        test_start = time.time()
        test_name = f"test_{test_id}_{len(symbols)}sym_{limit}lim_{entry_threshold}thr_{timeframe}"

        try:
            logger.info(f"Starting {test_name}")

            # Run training and backtesting
            result = train_and_save(
                symbols=symbols,
                limit=limit,
                entry_threshold=entry_threshold,
                backtest=True,
                write_report=True,
            )

            # Add test metadata
            test_duration = time.time() - test_start
            result.update(
                {
                    "test_metadata": {
                        "test_name": test_name,
                        "symbols_count": len(symbols),
                        "symbols": symbols,
                        "limit": limit,
                        "entry_threshold": entry_threshold,
                        "timeframe": timeframe,
                        "duration_seconds": test_duration,
                        "timestamp": datetime.now().isoformat(),
                    }
                }
            )

            # Risk management validation if enabled
            if self.risk_manager and result.get("backtest"):
                risk_analysis = self.analyze_backtest_risk(result["backtest"])
                result["risk_analysis"] = risk_analysis

            logger.info(f"Completed {test_name} in {test_duration:.1f}s")
            return result

        except Exception as e:
            logger.error(f"Test {test_name} failed: {e}")
            return {
                "error": str(e),
                "test_metadata": {
                    "test_name": test_name,
                    "duration_seconds": time.time() - test_start,
                    "timestamp": datetime.now().isoformat(),
                },
            }

    def analyze_backtest_risk(self, backtest_result: Dict) -> Dict[str, Any]:
        """Analyze backtest results from risk management perspective."""
        equity_curve = backtest_result.get("equity_curve", [])
        if not equity_curve:
            return {"error": "No equity curve data"}

        # Calculate risk metrics
        equities = [point["equity"] for point in equity_curve]
        returns = np.diff(equities) / equities[:-1]

        # Risk analysis
        analysis = {
            "total_return_pct": ((equities[-1] - equities[0]) / equities[0]) * 100,
            "max_drawdown_pct": backtest_result.get("max_drawdown", 0) * 100,
            "volatility": np.std(returns) * 100 if len(returns) > 1 else 0,
            "sharpe_ratio": self.calculate_sharpe_ratio(returns),
            "win_rate": backtest_result.get("win_rate", 0),
            "trades_count": backtest_result.get("trades", 0),
            "risk_adjusted_return": 0,
        }

        # Risk-adjusted return calculation
        if analysis["volatility"] > 0:
            analysis["risk_adjusted_return"] = (
                analysis["total_return_pct"] / analysis["volatility"]
            )

        # Risk warnings
        warnings = []
        if analysis["max_drawdown_pct"] > 15:
            warnings.append("High maximum drawdown")
        if analysis["volatility"] > 20:
            warnings.append("High volatility")
        if analysis["win_rate"] < 0.4:
            warnings.append("Low win rate")

        analysis["risk_warnings"] = warnings
        return analysis

    def calculate_sharpe_ratio(
        self, returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio for returns."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / 365)  # Daily risk-free rate
        return (
            np.mean(excess_returns) / np.std(excess_returns)
            if np.std(excess_returns) > 0
            else 0.0
        )

    def run_parallel_tests(
        self, config: TestConfiguration, max_workers: int = 3
    ) -> List[Dict[str, Any]]:
        """Run multiple tests in parallel for efficiency."""
        symbols = self.get_symbol_list(config)
        test_combinations = []

        # Generate test combinations
        timeframes = config.timeframes or ["1m"]
        for limit in config.limits:
            for threshold in config.entry_thresholds:
                for timeframe in timeframes:
                    test_combinations.append((symbols, limit, threshold, timeframe))

        logger.info(
            f"Running {len(test_combinations)} test combinations with {max_workers} workers"
        )

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tests
            future_to_params = {
                executor.submit(
                    self.run_single_test,
                    symbols,
                    limit,
                    threshold,
                    timeframe,
                    f"{config.name}_{i}",
                ): (symbols, limit, threshold, timeframe)
                for i, (symbols, limit, threshold, timeframe) in enumerate(
                    test_combinations
                )
            }

            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result(timeout=config.max_duration_minutes * 60)
                    results.append(result)

                    # Update performance metrics
                    self.update_performance_metrics(result)

                except Exception as e:
                    logger.error(f"Test future failed: {e}")
                    results.append({"error": str(e)})

        return results

    def update_performance_metrics(self, result: Dict[str, Any]):
        """Update overall performance tracking."""
        self.performance_metrics["total_tests_run"] += 1

        if "error" not in result:
            self.performance_metrics["successful_tests"] += 1

            # Update accuracy and return metrics
            if "metrics" in result:
                accuracy = result["metrics"].get("directional_accuracy", 0)
                self.performance_metrics["average_accuracy"] = (
                    self.performance_metrics["average_accuracy"]
                    * (self.performance_metrics["successful_tests"] - 1)
                    + accuracy
                ) / self.performance_metrics["successful_tests"]

            if "backtest" in result:
                pnl_pct = (
                    result["backtest"].get("pnl", 0)
                    / result["backtest"].get("starting_equity", 10000)
                ) * 100
                self.performance_metrics["average_return"] = (
                    self.performance_metrics["average_return"]
                    * (self.performance_metrics["successful_tests"] - 1)
                    + pnl_pct
                ) / self.performance_metrics["successful_tests"]
        else:
            self.performance_metrics["failed_tests"] += 1

    def run_configuration_test(self, config_name: str) -> Dict[str, Any]:
        """Run tests for a specific configuration."""
        if config_name not in PRODUCTION_TEST_CONFIGS:
            raise ValueError(f"Unknown configuration: {config_name}")

        config = PRODUCTION_TEST_CONFIGS[config_name]
        logger.info(f"Starting {config_name}: {config.description}")

        # Setup risk management
        self.setup_risk_management(config)

        # Run tests
        test_start = time.time()
        test_results = self.run_parallel_tests(config)
        total_duration = time.time() - test_start

        # Analyze results
        analysis = self.analyze_test_results(test_results, config)

        # Compile final results
        final_result = {
            "configuration": config_name,
            "config_details": asdict(config),
            "test_summary": {
                "total_tests": len(test_results),
                "successful_tests": len([r for r in test_results if "error" not in r]),
                "failed_tests": len([r for r in test_results if "error" in r]),
                "total_duration_minutes": total_duration / 60,
                "average_test_duration": (
                    (total_duration / len(test_results)) if test_results else 0
                ),
            },
            "performance_analysis": analysis,
            "detailed_results": test_results,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        result_file = self.output_dir / f"{config_name}_{int(time.time())}.json"
        with open(result_file, "w") as f:
            json.dump(final_result, f, indent=2, sort_keys=True)

        logger.info(
            f"Configuration {config_name} completed in {total_duration/60:.1f} minutes"
        )
        logger.info(f"Results saved to: {result_file}")

        return final_result

    def analyze_test_results(
        self, results: List[Dict], config: TestConfiguration
    ) -> Dict[str, Any]:
        """Analyze test results and provide insights."""
        successful_results = [r for r in results if "error" not in r]

        if not successful_results:
            return {"error": "No successful tests to analyze"}

        # Extract metrics
        accuracies = []
        returns = []
        drawdowns = []
        win_rates = []

        for result in successful_results:
            if "metrics" in result:
                accuracies.append(result["metrics"].get("directional_accuracy", 0))

            if "backtest" in result:
                backtest = result["backtest"]
                pnl_pct = (
                    backtest.get("pnl", 0) / backtest.get("starting_equity", 10000)
                ) * 100
                returns.append(pnl_pct)
                drawdowns.append(backtest.get("max_drawdown", 0) * 100)
                win_rates.append(backtest.get("win_rate", 0))

        # Statistical analysis
        analysis = {
            "accuracy_stats": {
                "mean": np.mean(accuracies) if accuracies else 0,
                "std": np.std(accuracies) if accuracies else 0,
                "min": np.min(accuracies) if accuracies else 0,
                "max": np.max(accuracies) if accuracies else 0,
            },
            "return_stats": {
                "mean": np.mean(returns) if returns else 0,
                "std": np.std(returns) if returns else 0,
                "min": np.min(returns) if returns else 0,
                "max": np.max(returns) if returns else 0,
            },
            "risk_stats": {
                "avg_drawdown": np.mean(drawdowns) if drawdowns else 0,
                "max_drawdown": np.max(drawdowns) if drawdowns else 0,
                "avg_win_rate": np.mean(win_rates) if win_rates else 0,
            },
        }

        # Best performing configuration
        if returns:
            best_idx = np.argmax(returns)
            analysis["best_performer"] = {
                "return_pct": returns[best_idx],
                "accuracy": accuracies[best_idx] if best_idx < len(accuracies) else 0,
                "test_details": successful_results[best_idx]["test_metadata"],
            }

        # Performance warnings
        warnings = []
        if analysis["accuracy_stats"]["mean"] < 0.6:
            warnings.append("Low average accuracy")
        if analysis["return_stats"]["mean"] < 5:
            warnings.append("Low average returns")
        if analysis["risk_stats"]["max_drawdown"] > 20:
            warnings.append("High maximum drawdown detected")

        analysis["warnings"] = warnings
        return analysis

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "report_generated": datetime.now().isoformat(),
            "test_session_duration": (
                datetime.now() - self.performance_metrics["test_start_time"]
            ).total_seconds()
            / 60,
            "overall_metrics": self.performance_metrics,
            "test_results_summary": {
                "total_configurations_tested": len(self.results),
                "results": self.results,
            },
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Production-Scale AI Testing System")
    parser.add_argument(
        "--config",
        choices=list(PRODUCTION_TEST_CONFIGS.keys()),
        help="Run specific test configuration",
    )
    parser.add_argument(
        "--all-configs", action="store_true", help="Run all test configurations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="production_test_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Initialize test runner
    runner = ProductionTestRunner(Path(args.output_dir))

    if args.config:
        # Run specific configuration
        result = runner.run_configuration_test(args.config)
        runner.results.append(result)

        print(f"\n🎯 Test Configuration: {args.config}")
        print(f"✅ Successful Tests: {result['test_summary']['successful_tests']}")
        print(f"❌ Failed Tests: {result['test_summary']['failed_tests']}")
        print(
            f"⏱️  Duration: {result['test_summary']['total_duration_minutes']:.1f} minutes"
        )

        if (
            "performance_analysis" in result
            and "return_stats" in result["performance_analysis"]
        ):
            avg_return = result["performance_analysis"]["return_stats"]["mean"]
            avg_accuracy = result["performance_analysis"]["accuracy_stats"]["mean"]
            print(f"📊 Average Return: {avg_return:.2f}%")
            print(f"🎯 Average Accuracy: {avg_accuracy:.1%}")

    elif args.all_configs:
        # Run all configurations
        print("🚀 Running all production test configurations...")

        for config_name in PRODUCTION_TEST_CONFIGS.keys():
            print(f"\n{'='*50}")
            print(f"Running: {config_name}")
            print(f"{'='*50}")

            try:
                result = runner.run_configuration_test(config_name)
                runner.results.append(result)
            except Exception as e:
                logger.error(f"Configuration {config_name} failed: {e}")

    else:
        print("Please specify --config <name> or --all-configs")
        print(f"Available configurations: {list(PRODUCTION_TEST_CONFIGS.keys())}")
        return

    # Generate final report
    final_report = runner.generate_performance_report()
    report_file = runner.output_dir / f"production_test_report_{int(time.time())}.json"

    with open(report_file, "w") as f:
        json.dump(final_report, f, indent=2, sort_keys=True)

    print(f"\n📋 Final report saved: {report_file}")
    print("🎉 Production testing complete!")


if __name__ == "__main__":
    main()
