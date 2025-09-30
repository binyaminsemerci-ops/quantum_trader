"""Backtest endpoint combining trained model metrics with legacy fallbacks."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, cast

from fastapi import APIRouter
from sqlalchemy import select

from ai_engine.train_and_save import load_report, run_backtest_only, train_and_save
from backend.database import session_scope, Trade, Candle

router = APIRouter()
logger = logging.getLogger(__name__)


def _normalise_curve(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    curve: List[Dict[str, Any]] = []
    for idx, point in enumerate(points):
        timestamp = (
            point.get("timestamp")
            or point.get("time")
            or point.get("date")
            or f"t{idx}"
        )
        equity_val = point.get("equity", 0.0)
        try:
            equity = float(equity_val)
        except Exception:
            equity = 0.0
        curve.append({"timestamp": str(timestamp), "equity": equity})
    return curve


def _model_backtest(symbol: str, limit: int, entry_threshold: float) -> Dict[str, Any]:
    try:
        summary = run_backtest_only(symbols=[symbol], limit=limit, entry_threshold=entry_threshold)
    except FileNotFoundError:
        logger.debug("Model artifacts missing; running training step for %s", symbol)
        train_and_save(symbols=[symbol], limit=max(limit, 300), backtest=True)
        summary = run_backtest_only(symbols=[symbol], limit=limit, entry_threshold=entry_threshold)

    metrics = summary.get("metrics", {})
    backtest = summary.get("backtest") or {}
    curve = _normalise_curve(backtest.get("equity_curve", []))
    backtest_payload = dict(backtest)
    backtest_payload["equity_curve"] = curve

    response: Dict[str, Any] = {
        "symbol": symbol,
        "mode": "model",
        "source": "model_artifacts",
        "num_samples": summary.get("num_samples"),
        "metrics": metrics,
        "backtest": backtest_payload,
        "equity_curve": curve,
        "pnl": backtest.get("pnl"),
        "final_equity": backtest.get("final_equity"),
        "trades": backtest.get("trades"),
        "win_rate": backtest.get("win_rate"),
        "max_drawdown": backtest.get("max_drawdown"),
    }
    report = load_report()
    if report:
        response["report"] = report
    return response


def _legacy_backtest(symbol: str, days: int) -> Dict[str, Any]:
    with session_scope() as session:
        trade_rows = session.execute(
            select(Trade).where(Trade.symbol == symbol).order_by(cast(Any, Trade.timestamp).asc())
        ).scalars().all()

        if trade_rows:
            start_equity = 10_000.0
            equity = start_equity
            curve: List[Dict[str, Any]] = []
            wins = 0
            for trade in trade_rows:
                if trade.side.upper() == "BUY":
                    equity -= trade.qty * trade.price
                else:
                    equity += trade.qty * trade.price
                    wins += 1
                curve.append(
                    {
                        "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
                        "equity": round(equity, 2),
                    }
                )
            pnl = round(equity - start_equity, 2)
            return {
                "symbol": symbol,
                "mode": "legacy-trades",
                "source": "database",
                "equity_curve": curve,
                "pnl": pnl,
                "win_rate": wins / max(1, len(trade_rows)),
                "trades": len(trade_rows),
                "max_drawdown": None,
            }

        candle_rows = session.execute(
            select(Candle)
            .where(Candle.symbol == symbol)
            .order_by(cast(Any, Candle.timestamp).asc())
            .limit(days)
        ).scalars().all()

    if not candle_rows:
        return {
            "symbol": symbol,
            "mode": "legacy-empty",
            "source": "database",
            "equity_curve": [],
            "pnl": 0.0,
            "error": f"No trades or candles found for {symbol}",
        }

    start_equity = 10_000.0
    equity = start_equity
    curve = []
    wins = 0
    for candle in candle_rows:
        if candle.close > candle.open:
            change = (candle.close - candle.open) / candle.open
            equity *= 1 + change
            wins += 1
        curve.append(
            {
                "timestamp": candle.timestamp.isoformat() if candle.timestamp else None,
                "equity": round(equity, 2),
            }
        )

    pnl = round(equity - start_equity, 2)
    return {
        "symbol": symbol,
        "mode": "legacy-candles",
        "source": "database",
        "equity_curve": curve,
        "pnl": pnl,
        "win_rate": wins / max(1, len(candle_rows)),
        "trades": 0,
        "max_drawdown": None,
    }


@router.get("/backtest")
def run_backtest(
    symbol: str = "BTCUSDT",
    days: int = 30,
    limit: int | None = None,
    entry_threshold: float = 0.0,
):
    """Return backtest metrics derived from the latest trained model."""
    candle_limit = limit or max(days * 24, 120)
    try:
        return _model_backtest(symbol, candle_limit, entry_threshold)
    except Exception as exc:
        logger.debug("Model backtest failed for %s: %s", symbol, exc, exc_info=True)
        return _legacy_backtest(symbol, days)
