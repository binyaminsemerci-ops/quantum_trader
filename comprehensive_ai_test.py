#!/usr/bin/env python3
"""
Comprehensive AI Model Test Script for Quantum Trader
====================================================

Dette skriptet kjører en fullstendig test av AI-systemet inkludert:
- Modelltrening på flere tidsrammer og symbol-sett
- Backtesting med forskjellige entry thresholds
- Performance metrics og validering
- Stress testing med live data (hvis tilgjengelig)
- Sammenligning av forskjellige konfigurasjoner

Usage:
    python comprehensive_ai_test.py --full-test
    python comprehensive_ai_test.py --quick-test
    python comprehensive_ai_test.py --stress-test
    python comprehensive_ai_test.py --compare-configs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai_engine.train_and_save import (
    load_report,
    run_backtest_only,
    train_and_save,
)
from config.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test configurations
TEST_CONFIGS = {
    "quick": {
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "limit": 300,
        "entry_thresholds": [0.001, 0.005],
        "description": "Quick test with 3 major symbols",
    },
    "standard": {
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"],
        "limit": 600,
        "entry_thresholds": [0.001, 0.005, 0.01],
        "description": "Standard test with 6 major symbols",
    },
    "comprehensive": {
        "symbols": [
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
        ],
        "limit": 1000,
        "entry_thresholds": [0.001, 0.002, 0.005, 0.01, 0.02],
        "description": "Comprehensive test with 12 symbols",
    },
    "stress": {
        "symbols": None,  # Will use full DEFAULT_SYMBOLS from config
        "limit": 2000,
        "entry_thresholds": [0.001, 0.005, 0.01],
        "description": "Stress test with all available symbols",
    },
}


class AITestRunner:
    """Comprehensive AI model test runner"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def log_result(self, test_name: str, result: Dict[str, Any], duration: float):
        """Log test result with metadata"""
        test_result = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "result": result,
            "success": result is not None and "error" not in result,
        }
        self.test_results.append(test_result)

        # Save individual result
        result_file = self.output_dir / f"{test_name}_{int(time.time())}.json"
        with open(result_file, "w") as f:
            json.dump(test_result, f, indent=2, sort_keys=True)

        logger.info(f"Test '{test_name}' completed in {duration:.2f}s")

    def run_training_test(
        self, config_name: str, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run training test with given configuration"""
        logger.info(f"Starting training test: {config_name}")
        logger.info(f"Description: {config['description']}")

        symbols = config["symbols"]
        if symbols is None:
            # Use full symbol set from config
            symbols = settings.default_symbols
            logger.info(f"Using full symbol set: {len(symbols)} symbols")

        start_time = time.time()

        try:
            # Create model directory for this test
            model_dir = self.output_dir / f"model_{config_name}_{int(start_time)}"

            result = train_and_save(
                symbols=symbols,
                limit=config["limit"],
                model_dir=model_dir,
                backtest=True,
                write_report=True,
                entry_threshold=config["entry_thresholds"][
                    0
                ],  # Use first threshold for training
            )

            duration = time.time() - start_time
            self.log_result(f"training_{config_name}", result, duration)

            return result

        except Exception as e:
            duration = time.time() - start_time
            error_result = {"error": str(e), "type": type(e).__name__}
            self.log_result(f"training_{config_name}_error", error_result, duration)
            logger.error(f"Training test {config_name} failed: {e}")
            return None

    def run_backtest_variations(
        self, config_name: str, config: Dict[str, Any], model_dir: Path = None
    ) -> List[Dict[str, Any]]:
        """Run backtest with different entry thresholds"""
        logger.info(f"Running backtest variations for {config_name}")

        symbols = config["symbols"]
        if symbols is None:
            symbols = settings.default_symbols

        # Use default model directory if none provided
        if model_dir is None:
            model_dir = Path("ai_engine/models")

        results = []

        for threshold in config["entry_thresholds"]:
            logger.info(f"Testing entry threshold: {threshold}")
            start_time = time.time()

            try:
                result = run_backtest_only(
                    symbols=symbols,
                    limit=config["limit"],
                    model_dir=model_dir,
                    entry_threshold=threshold,
                )

                duration = time.time() - start_time
                test_name = f"backtest_{config_name}_threshold_{threshold}"
                self.log_result(test_name, result, duration)
                results.append(result)

            except Exception as e:
                duration = time.time() - start_time
                error_result = {
                    "error": str(e),
                    "type": type(e).__name__,
                    "threshold": threshold,
                }
                test_name = f"backtest_{config_name}_threshold_{threshold}_error"
                self.log_result(test_name, error_result, duration)
                logger.error(f"Backtest with threshold {threshold} failed: {e}")

        return results

    def run_model_validation(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """Validate trained model artifacts and report"""
        logger.info("Validating model artifacts")

        validation_result = {
            "model_dir": str(model_dir),
            "artifacts_found": {},
            "report_data": None,
            "validation_errors": [],
        }

        # Check for expected files
        expected_files = ["model.pkl", "scaler.pkl", "training_report.json"]
        for file_name in expected_files:
            file_path = model_dir / file_name
            validation_result["artifacts_found"][file_name] = file_path.exists()
            if not file_path.exists():
                validation_result["validation_errors"].append(
                    f"Missing artifact: {file_name}"
                )

        # Load and validate report
        try:
            report = load_report(model_dir=model_dir)
            if report:
                validation_result["report_data"] = report

                # Validate report structure
                required_fields = ["symbols", "num_samples", "metrics"]
                for field in required_fields:
                    if field not in report:
                        validation_result["validation_errors"].append(
                            f"Missing report field: {field}"
                        )

                # Validate metrics
                if "metrics" in report:
                    metrics = report["metrics"]
                    if "mse" not in metrics:
                        validation_result["validation_errors"].append(
                            "Missing MSE metric"
                        )
                    if "r2" not in metrics:
                        validation_result["validation_errors"].append(
                            "Missing R² metric"
                        )
            else:
                validation_result["validation_errors"].append(
                    "Could not load training report"
                )

        except Exception as e:
            validation_result["validation_errors"].append(
                f"Report validation error: {e}"
            )

        self.log_result("model_validation", validation_result, 0)
        return validation_result

    def run_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance across different test configurations"""
        logger.info("Running performance comparison")

        comparison = {
            "test_count": len(self.test_results),
            "successful_tests": len([r for r in self.test_results if r["success"]]),
            "failed_tests": len([r for r in self.test_results if not r["success"]]),
            "total_duration": (datetime.now() - self.start_time).total_seconds(),
            "test_breakdown": {},
            "performance_metrics": {},
        }

        # Analyze results by test type
        for result in self.test_results:
            test_type = result["test_name"].split("_")[0]
            if test_type not in comparison["test_breakdown"]:
                comparison["test_breakdown"][test_type] = {
                    "count": 0,
                    "successful": 0,
                    "avg_duration": 0,
                }

            comparison["test_breakdown"][test_type]["count"] += 1
            if result["success"]:
                comparison["test_breakdown"][test_type]["successful"] += 1
            comparison["test_breakdown"][test_type]["avg_duration"] += result[
                "duration_seconds"
            ]

        # Calculate averages
        for test_type, data in comparison["test_breakdown"].items():
            if data["count"] > 0:
                data["avg_duration"] /= data["count"]
                data["success_rate"] = data["successful"] / data["count"]

        # Extract performance metrics from successful backtests
        backtest_results = [
            r
            for r in self.test_results
            if r["test_name"].startswith("backtest_") and r["success"]
        ]

        if backtest_results:
            returns = []
            sharpe_ratios = []
            win_rates = []

            for result in backtest_results:
                if "result" in result and result["result"]:
                    res = result["result"]
                    if "total_return" in res:
                        returns.append(res["total_return"])
                    if "sharpe_ratio" in res:
                        sharpe_ratios.append(res["sharpe_ratio"])
                    if "win_rate" in res:
                        win_rates.append(res["win_rate"])

            if returns:
                comparison["performance_metrics"]["avg_return"] = sum(returns) / len(
                    returns
                )
                comparison["performance_metrics"]["best_return"] = max(returns)
                comparison["performance_metrics"]["worst_return"] = min(returns)

            if sharpe_ratios:
                comparison["performance_metrics"]["avg_sharpe"] = sum(
                    sharpe_ratios
                ) / len(sharpe_ratios)
                comparison["performance_metrics"]["best_sharpe"] = max(sharpe_ratios)

            if win_rates:
                comparison["performance_metrics"]["avg_win_rate"] = sum(
                    win_rates
                ) / len(win_rates)

        self.log_result("performance_comparison", comparison, 0)
        return comparison

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        logger.info("Generating summary report")

        summary_file = self.output_dir / f"test_summary_{int(time.time())}.json"

        summary = {
            "test_session": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": (datetime.now() - self.start_time).total_seconds(),
                "output_directory": str(self.output_dir),
            },
            "results": self.test_results,
            "comparison": self.run_performance_comparison(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        # Print summary to console
        print("\n" + "=" * 80)
        print("AI MODEL COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        print(f"Test Duration: {summary['test_session']['total_duration']:.1f} seconds")
        print(f"Total Tests: {summary['comparison']['test_count']}")
        print(f"Successful: {summary['comparison']['successful_tests']}")
        print(f"Failed: {summary['comparison']['failed_tests']}")

        if "performance_metrics" in summary["comparison"]:
            metrics = summary["comparison"]["performance_metrics"]
            print("\nPerformance Metrics:")
            if "avg_return" in metrics:
                print(f"  Average Return: {metrics['avg_return']:.4f}")
                print(f"  Best Return: {metrics['best_return']:.4f}")
                print(f"  Worst Return: {metrics['worst_return']:.4f}")
            if "avg_sharpe" in metrics:
                print(f"  Average Sharpe: {metrics['avg_sharpe']:.4f}")
            if "avg_win_rate" in metrics:
                print(f"  Average Win Rate: {metrics['avg_win_rate']:.4f}")

        print(f"\nDetailed results saved to: {summary_file}")
        print("=" * 80)


def run_quick_test(runner: AITestRunner):
    """Run quick test configuration"""
    config = TEST_CONFIGS["quick"]
    result = runner.run_training_test("quick", config)
    if result:
        # Use the default model directory where artifacts are saved
        runner.run_backtest_variations("quick", config)


def run_standard_test(runner: AITestRunner):
    """Run standard test configuration"""
    config = TEST_CONFIGS["standard"]
    result = runner.run_training_test("standard", config)
    if result:
        runner.run_backtest_variations("standard", config)


def run_comprehensive_test(runner: AITestRunner):
    """Run comprehensive test configuration"""
    config = TEST_CONFIGS["comprehensive"]
    result = runner.run_training_test("comprehensive", config)
    if result:
        runner.run_backtest_variations("comprehensive", config)


def run_stress_test(runner: AITestRunner):
    """Run stress test configuration"""
    config = TEST_CONFIGS["stress"]
    result = runner.run_training_test("stress", config)
    if result:
        runner.run_backtest_variations("stress", config)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive AI Model Testing for Quantum Trader"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick test with 3 symbols"
    )
    parser.add_argument(
        "--standard-test", action="store_true", help="Run standard test with 6 symbols"
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run comprehensive test with 12 symbols",
    )
    parser.add_argument(
        "--stress-test", action="store_true", help="Run stress test with all symbols"
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Run all configurations and compare",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for test results"
    )

    args = parser.parse_args()

    if not any(
        [
            args.quick_test,
            args.standard_test,
            args.full_test,
            args.stress_test,
            args.compare_configs,
        ]
    ):
        print("No test specified. Use --help to see options.")
        return 1

    runner = AITestRunner(output_dir=args.output_dir)

    try:
        if args.quick_test:
            run_quick_test(runner)
        elif args.standard_test:
            run_standard_test(runner)
        elif args.full_test:
            run_comprehensive_test(runner)
        elif args.stress_test:
            run_stress_test(runner)
        elif args.compare_configs:
            # Run all tests for comparison
            logger.info("Running all test configurations for comparison")
            run_quick_test(runner)
            run_standard_test(runner)
            run_comprehensive_test(runner)
            # Skip stress test in comparison mode unless explicitly requested

        # Always generate summary
        runner.generate_summary_report()

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        runner.generate_summary_report()
        return 130
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        runner.generate_summary_report()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
