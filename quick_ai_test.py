#!/usr/bin/env python3
"""
Quick AI Test Script - Kjøre rask AI modell test
=============================================

Dette skriptet kjører en rask test av AI-systemet med færre symboler for rask validering.
"""

from datetime import datetime
from pathlib import Path


def run_quick_ai_test():
    """Run a quick AI test and show results"""
    print("🚀 Starter rask AI modell test...")
    print("=" * 60)

    # Import here to avoid startup delays
    from ai_engine.train_and_save import run_backtest_only, train_and_save

    # Test configuration - small set for quick results
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    limit = 200  # Fewer candles for speed

    print(f"📊 Testing symbols: {', '.join(symbols)}")
    print(f"📈 Candles per symbol: {limit}")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}\n")

    try:
        # Step 1: Train the model
        print("🔄 Step 1: Training AI model...")
        start_time = datetime.now()

        training_result = train_and_save(
            symbols=symbols,
            limit=limit,
            backtest=True,
            write_report=True,
            entry_threshold=0.001,
        )

        training_duration = (datetime.now() - start_time).total_seconds()
        print(f"✅ Training completed in {training_duration:.1f} seconds")

        if training_result:
            print("\n📋 Training Results:")
            for key, value in training_result.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

        # Step 2: Run backtest with different thresholds
        print("\n🔄 Step 2: Running backtest variations...")
        thresholds = [0.001, 0.005, 0.01]

        for threshold in thresholds:
            print(f"\n  🧪 Testing threshold: {threshold}")
            backtest_start = datetime.now()

            backtest_result = run_backtest_only(
                symbols=symbols, limit=limit, entry_threshold=threshold
            )

            backtest_duration = (datetime.now() - backtest_start).total_seconds()
            print(f"     ⏱️ Completed in {backtest_duration:.1f} seconds")

            if backtest_result:
                # Show key metrics
                if "total_return" in backtest_result:
                    ret = backtest_result["total_return"]
                    print(f"     💰 Total Return: {ret:.4f}")

                if "sharpe_ratio" in backtest_result:
                    sharpe = backtest_result["sharpe_ratio"]
                    print(f"     📊 Sharpe Ratio: {sharpe:.4f}")

                if "win_rate" in backtest_result:
                    win_rate = backtest_result["win_rate"]
                    print(f"     🎯 Win Rate: {win_rate:.4f}")

                if "num_trades" in backtest_result:
                    trades = backtest_result["num_trades"]
                    print(f"     🔄 Number of Trades: {trades}")

        total_duration = (datetime.now() - start_time).total_seconds()
        print(f"\n🏁 Total test duration: {total_duration:.1f} seconds")
        print("✅ AI model test completed successfully!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

    finally:
        print("\n" + "=" * 60)
        print(f"🕐 Test finished: {datetime.now().strftime('%H:%M:%S')}")


def show_model_info():
    """Show information about saved model"""
    try:
        from ai_engine.train_and_save import load_report

        print("\n📁 Checking for saved model artifacts...")

        # Check default model directory
        model_dir = Path("ai_engine/models")
        if model_dir.exists():
            files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.json"))
            if files:
                print(f"   Found {len(files)} artifact files:")
                for file in sorted(files):
                    size = file.stat().st_size / 1024  # KB
                    print(f"   - {file.name} ({size:.1f} KB)")
            else:
                print("   No model artifacts found")

        # Try to load report
        report = load_report()
        if report:
            print("\n📊 Latest Training Report:")
            print(f"   Symbols: {report.get('symbols', 'N/A')}")
            print(f"   Samples: {report.get('num_samples', 'N/A')}")

            metrics = report.get("metrics", {})
            if metrics:
                print(f"   MSE: {metrics.get('mse', 'N/A')}")
                print(f"   R²: {metrics.get('r2', 'N/A')}")
        else:
            print("   No training report found")

    except Exception as e:
        print(f"   Error checking model info: {e}")


if __name__ == "__main__":
    print("🤖 Quantum Trader - AI Model Quick Test")
    print("=" * 60)

    # Show current model status first
    show_model_info()

    # Ask user if they want to run the test
    print("\nVil du kjøre en rask AI modell test? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ["y", "yes", "ja", "j"]:
            success = run_quick_ai_test()
            if success:
                show_model_info()  # Show updated info after test
        else:
            print("Test avbrutt.")
    except KeyboardInterrupt:
        print("\n\nTest avbrutt av bruker.")
