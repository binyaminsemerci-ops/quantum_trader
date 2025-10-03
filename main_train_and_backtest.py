from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_engine.train_and_save import (
    load_report,
    run_backtest_only,
    train_and_save,
)


def _print_json(payload: dict | None) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _handle_train(args: argparse.Namespace) -> None:
    summary = train_and_save(
        symbols=args.symbols,
        limit=args.limit,
        model_dir=args.model_dir,
        backtest=not args.skip_backtest,
        write_report=not args.no_report,
        entry_threshold=args.entry_threshold,
    )
    _print_json(summary)


def _handle_backtest(args: argparse.Namespace) -> None:
    result = run_backtest_only(
        symbols=args.symbols,
        limit=args.limit,
        model_dir=args.model_dir,
        entry_threshold=args.entry_threshold,
    )
    _print_json(result)


def _handle_report(args: argparse.Namespace) -> None:
    report = load_report(model_dir=args.model_dir)
    if report is None:
        print("No report found. Run `python main_train_and_backtest.py train` first.")
    else:
        if args.json:
            _print_json(report)
        else:
            print("Symbols:", ", ".join(report.get("symbols", [])))
            print("Samples:", report.get("num_samples"))
            metrics = report.get("metrics", {})
            print("RMSE:", metrics.get("rmse"))
            print("MAE:", metrics.get("mae"))
            print("Directional accuracy:", metrics.get("directional_accuracy"))
            backtest = report.get("backtest")
            if backtest:
                print("Final equity:", backtest.get("final_equity"))
                print("PnL:", backtest.get("pnl"))
                print("Trades:", backtest.get("trades"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Quantum Trader model and run quick backtests.",
    )
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser(
        "train", help="Train a model and optionally run a backtest."
    )
    train_parser.add_argument(
        "--symbols", nargs="+", help="Symbols to pull data for (defaults to config)"
    )
    train_parser.add_argument(
        "--limit", type=int, default=600, help="Number of candles per symbol"
    )
    train_parser.add_argument(
        "--model-dir", type=Path, help="Directory where artifacts are stored"
    )
    train_parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.001,
        help="Minimum predicted return required before taking a trade during the backtest",
    )
    train_parser.add_argument(
        "--skip-backtest", action="store_true", help="Skip the evaluation step"
    )
    train_parser.add_argument(
        "--no-report", action="store_true", help="Do not write training_report.json"
    )
    train_parser.set_defaults(func=_handle_train)

    backtest_parser = sub.add_parser(
        "backtest", help="Run a backtest using existing artifacts."
    )
    backtest_parser.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols to evaluate"
    )
    backtest_parser.add_argument(
        "--limit", type=int, default=600, help="Number of candles per symbol"
    )
    backtest_parser.add_argument(
        "--model-dir", type=Path, help="Directory with previously saved artifacts"
    )
    backtest_parser.add_argument(
        "--entry-threshold",
        type=float,
        default=0.001,
        help="Minimum predicted return required before taking a trade",
    )
    backtest_parser.set_defaults(func=_handle_backtest)

    report_parser = sub.add_parser("report", help="Print the last training report.")
    report_parser.add_argument(
        "--model-dir", type=Path, help="Directory containing training_report.json"
    )
    report_parser.add_argument(
        "--json", action="store_true", help="Emit the raw JSON instead of a summary"
    )
    report_parser.set_defaults(func=_handle_report)

    parser.set_defaults(func=_handle_train)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    func = getattr(args, "func", _handle_train)
    func(args)


if __name__ == "__main__":
    main()
