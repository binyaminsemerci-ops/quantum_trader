#!/usr/bin/env python3
"""Quick AI Test Script - Kjøre rask AI modell test.
=============================================

Dette skriptet kjører en rask test av AI-systemet med færre symboler for rask validering.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


def run_quick_ai_test() -> Optional[bool]:
    """Run a quick AI test and show results."""
    # Import here to avoid startup delays
    from ai_engine.train_and_save import run_backtest_only, train_and_save

    # Test configuration - small set for quick results
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    limit = 200  # Fewer candles for speed


    try:
        # Step 1: Train the model
        start_time = datetime.now()

        training_result = train_and_save(
            symbols=symbols,
            limit=limit,
            backtest=True,
            write_report=True,
            entry_threshold=0.001,
        )

        (datetime.now() - start_time).total_seconds()

        if training_result:
            for value in training_result.values():
                if isinstance(value, dict):
                    for _k, _v in value.items():
                        pass
                else:
                    pass

        # Step 2: Run backtest with different thresholds
        thresholds = [0.001, 0.005, 0.01]

        for threshold in thresholds:
            backtest_start = datetime.now()

            backtest_result = run_backtest_only(
                symbols=symbols, limit=limit, entry_threshold=threshold,
            )

            (datetime.now() - backtest_start).total_seconds()

            if backtest_result:
                # Show key metrics
                if "total_return" in backtest_result:
                    backtest_result["total_return"]

                if "sharpe_ratio" in backtest_result:
                    backtest_result["sharpe_ratio"]

                if "win_rate" in backtest_result:
                    backtest_result["win_rate"]

                if "num_trades" in backtest_result:
                    backtest_result["num_trades"]

        (datetime.now() - start_time).total_seconds()

        return True

    except Exception:
        return False

    finally:
        pass


def show_model_info() -> None:
    """Show information about saved model."""
    try:
        from ai_engine.train_and_save import load_report


        # Check default model directory
        model_dir = Path("ai_engine/models")
        if model_dir.exists():
            files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.json"))
            if files:
                for file in sorted(files):
                    file.stat().st_size / 1024  # KB
            else:
                pass

        # Try to load report
        report = load_report()
        if report:

            metrics = report.get("metrics", {})
            if metrics:
                pass
        else:
            pass

    except Exception:
        pass


if __name__ == "__main__":

    # Show current model status first
    show_model_info()

    # Ask user if they want to run the test
    try:
        response = input().lower().strip()
        if response in ["y", "yes", "ja", "j"]:
            success = run_quick_ai_test()
            if success:
                show_model_info()  # Show updated info after test
        else:
            pass
    except KeyboardInterrupt:
        pass
