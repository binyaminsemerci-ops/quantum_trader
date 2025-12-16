"""Background scheduler utilities for refreshing external data caches."""

from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .telemetry import (
    record_provider_failure,
    record_provider_success,
    record_scheduler_run,
    set_provider_circuit,
)

logger = logging.getLogger(__name__)

_SCHEDULER: Optional[AsyncIOScheduler] = None
# Extended to top coins - will be filtered by liquidity selection
_DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "TONUSDT", "LINKUSDT",
    "AVAXUSDT", "DOTUSDT", "MATICUSDT", "ATOMUSDT", "ARBUSDT",
    "OPUSDT", "UNIUSDT", "PEPEUSDT", "SHIBUSDT", "APTUSDT"
]

_SCHEDULER_CONFIG: Dict[str, Any] = {
    "symbols": _DEFAULT_SYMBOLS.copy(),
    "refresh_seconds": 180,
    "execution_seconds": 0,
}

_RUN_STATE: Dict[str, Any] = {
    "status": "idle",
    "runs": 0,
    "last_started": None,
    "last_completed": None,
    "last_duration": None,
    "symbols": [],
    "successful_symbols": [],
    "errors": {},
    "providers": {},
}

_LIQUIDITY_STATE: Dict[str, Any] = {
    "status": "idle",
    "runs": 0,
    "last_started": None,
    "last_completed": None,
    "last_duration": None,
    "provider_primary": None,
    "universe_size": 0,
    "selection_size": 0,
    "error": None,
    "run_id": None,
    "analytics": None,
}

_EXECUTION_STATE: Dict[str, Any] = {
    "status": "idle",
    "runs": 0,
    "last_started": None,
    "last_completed": None,
    "last_duration": None,
    "orders_planned": 0,
    "orders_submitted": 0,
    "orders_skipped": 0,
    "orders_failed": 0,
    "error": None,
    "run_id": None,
    "gross_exposure": 0.0,
    "positions_synced": False,
}

_PROVIDER_STATE: Dict[str, Any] = {}
_CIRCUIT_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SECONDS = int(os.getenv("QUANTUM_TRADER_PROVIDER_COOLDOWN", "120"))
_PROVIDER_PRIORITY = ("binance", "coingecko")


def _to_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _reset_internal_state_for_tests() -> None:
    """Reset module state. Intended for test isolation only."""

    global _SCHEDULER
    _SCHEDULER = None
    _RUN_STATE.update(
        {
            "status": "idle",
            "runs": 0,
            "last_started": None,
            "last_completed": None,
            "last_duration": None,
            "symbols": [],
            "successful_symbols": [],
            "errors": {},
            "providers": {},
        }
    )
    _PROVIDER_STATE.clear()
    _LIQUIDITY_STATE.update(
        {
            "status": "idle",
            "runs": 0,
            "last_started": None,
            "last_completed": None,
            "last_duration": None,
            "provider_primary": None,
            "universe_size": 0,
            "selection_size": 0,
            "error": None,
            "run_id": None,
            "analytics": None,
        }
    )
    _EXECUTION_STATE.update(
        {
            "status": "idle",
            "runs": 0,
            "last_started": None,
            "last_completed": None,
            "last_duration": None,
            "orders_planned": 0,
            "orders_submitted": 0,
            "orders_skipped": 0,
            "orders_failed": 0,
            "error": None,
            "run_id": None,
            "gross_exposure": 0.0,
            "positions_synced": False,
        }
    )


def _resolve_symbols() -> List[str]:
    raw = os.getenv("QUANTUM_TRADER_SYMBOLS")
    if raw:
        symbols = [item.strip().upper() for item in raw.split(",") if item.strip()]
        symbols = symbols or _DEFAULT_SYMBOLS
        _SCHEDULER_CONFIG["symbols"] = symbols
        return symbols

    # Try to use QT_UNIVERSE setting (same as event-driven mode)
    universe_profile = os.getenv("QT_UNIVERSE", "l1l2-top")
    try:
        if universe_profile.lower() in ("l1l2-top", "l1l2volume", "l1l2-vol"):
            try:
                from config.config import DEFAULT_QUOTE  # type: ignore
                quote = DEFAULT_QUOTE
            except Exception:
                quote = "USDC"
            from backend.utils.universe import get_l1l2_top_by_volume, get_l1l2_top_by_volume_multi
            max_symbols = int(os.getenv("QT_MAX_SYMBOLS", "100"))
            quotes_env = os.getenv("QT_QUOTES", "").strip()
            
            # Default to cross-margin with USDC + USDT if not specified
            if not quotes_env:
                quotes_env = "USDC,USDT"
                logger.info("Using cross-margin mode (USDC + USDT futures)")
            
            if quotes_env:
                quotes = [q.strip().upper() for q in quotes_env.split(",") if q.strip()]
                symbols = get_l1l2_top_by_volume_multi(quotes=quotes, max_symbols=max_symbols)
                logger.info(
                    "Selected %d L1/L2+base pairs by 24h volume (quotes=%s, cross-margin)",
                    len(symbols), ",".join(quotes)
                )
            else:
                symbols = get_l1l2_top_by_volume(quote=quote, max_symbols=max_symbols)
                logger.info(
                    "Selected %d L1/L2+base symbols by 24h volume (quote=%s)",
                    len(symbols), quote
                )
        else:
            from config.trading_universe import get_trading_universe
            symbols = get_trading_universe(universe_profile)
            # Apply max symbols limit if set
            max_symbols_env = int(os.getenv("QT_MAX_SYMBOLS", "0"))
            if max_symbols_env > 0 and len(symbols) > max_symbols_env:
                symbols = symbols[:max_symbols_env]
                logger.info(f"Limited to {max_symbols_env} symbols (QT_MAX_SYMBOLS)")
    except Exception as e:
        logger.warning(
            f"Failed to load trading universe '{universe_profile}': {e}, using defaults"
        )
        symbols = _DEFAULT_SYMBOLS

    _SCHEDULER_CONFIG["symbols"] = symbols
    return symbols


def _liquidity_interval_seconds() -> int:
    raw = os.getenv("QUANTUM_TRADER_LIQUIDITY_REFRESH_SECONDS")
    if not raw:
        return 900
    try:
        value = int(raw)
        return max(value, 0)
    except ValueError:
        logger.warning(
            "Invalid QUANTUM_TRADER_LIQUIDITY_REFRESH_SECONDS=%s; using default 900",
            raw,
        )
        return 900


def _execution_interval_seconds() -> int:
    raw = os.getenv("QUANTUM_TRADER_EXECUTION_SECONDS")
    if not raw:
        return 300  # 5 minutes (shortened for faster testing)
    try:
        value = int(raw)
        return max(value, 0)
    except ValueError:
        logger.warning(
            "Invalid QUANTUM_TRADER_EXECUTION_SECONDS=%s; using default 300",
            raw,
        )
        return 900


def _scheduler_disabled() -> bool:
    if os.getenv("QUANTUM_TRADER_DISABLE_SCHEDULER") == "1":
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    return False


async def warm_market_caches(symbols: Iterable[str]) -> None:
    """Refresh price and sentiment caches for the provided symbols.

    Failures are logged and do not propagate to keep the scheduler resilient.
    """

    from backend.routes.external_data import binance_ohlcv, twitter_sentiment
    try:
        from backend.routes.coingecko_data import (  # type: ignore
            get_coin_price_data,
            symbol_to_coingecko_id,
        )
    except ModuleNotFoundError:  # pragma: no cover
        from routes.coingecko_data import get_coin_price_data, symbol_to_coingecko_id  # type: ignore

    started_at = datetime.now(timezone.utc)
    _RUN_STATE["last_started"] = started_at
    _RUN_STATE["status"] = "running"

    errors: Dict[str, List[str]] = defaultdict(list)
    successes: Dict[str, set[str]] = defaultdict(set)

    async def _refresh(raw_symbol: str) -> None:
        symbol = raw_symbol.upper()

        provider_used = await _fetch_price_with_failover(
            symbol,
            price_fetchers={
                "binance": lambda: binance_ohlcv(symbol, limit=600),
                "coingecko": lambda: _fetch_coingecko_price(
                    symbol, get_coin_price_data, symbol_to_coingecko_id
                ),
            },
            errors=errors,
        )
        if provider_used:
            successes[symbol].add(provider_used)

        try:
            await twitter_sentiment(symbol)
            successes[symbol].add("sentiment")
        except Exception as exc:  # noqa: BLE001
            message = f"sentiment: {exc}"
            errors[symbol].append(message)
            logger.warning("Sentiment refresh failed for %s: %s", symbol, exc)

    tasks = [_refresh(symbol) for symbol in symbols]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    completed_at = datetime.now(timezone.utc)
    duration_seconds = (completed_at - started_at).total_seconds()
    _RUN_STATE["last_completed"] = completed_at
    _RUN_STATE["last_duration"] = duration_seconds
    all_symbols = sorted({symbol.upper() for symbol in symbols})
    _RUN_STATE["symbols"] = all_symbols
    _RUN_STATE["successful_symbols"] = sorted(successes.keys())
    _RUN_STATE["errors"] = {symbol: messages for symbol, messages in errors.items()}
    _RUN_STATE["providers"] = _provider_snapshot()
    _RUN_STATE["runs"] = int(_RUN_STATE.get("runs", 0)) + 1
    provider_circuit = any(
        state.get("circuit_open") for state in _PROVIDER_STATE.values()
    )
    status = "degraded" if errors or provider_circuit else "ok"
    _RUN_STATE["status"] = status
    record_scheduler_run("market-cache-refresh", status, duration_seconds)


def _get_provider_state(name: str) -> Dict[str, Any]:
    state = _PROVIDER_STATE.setdefault(
        name,
        {
            "success": 0,
            "failures": 0,
            "failure_streak": 0,
            "circuit_open": False,
            "circuit_open_at": None,
            "last_success": None,
            "last_failure": None,
            "last_error": None,
        },
    )
    return state


def _record_provider_success(name: str) -> None:
    state = _get_provider_state(name)
    state["success"] = int(state.get("success", 0)) + 1
    state["failure_streak"] = 0
    state["circuit_open"] = False
    state["circuit_open_at"] = None
    state["last_success"] = datetime.now(timezone.utc)
    state["last_error"] = None
    record_provider_success(name)
    set_provider_circuit(name, False)


def _record_provider_failure(name: str, exc: Exception) -> None:
    state = _get_provider_state(name)
    state["failures"] = int(state.get("failures", 0)) + 1
    state["failure_streak"] = int(state.get("failure_streak", 0)) + 1
    state["last_error"] = str(exc)
    state["last_failure"] = datetime.now(timezone.utc)

    if state["failure_streak"] >= _CIRCUIT_THRESHOLD:
        if not state.get("circuit_open"):
            logger.warning("Opening circuit for provider %s after repeated failures", name)
        state["circuit_open"] = True
        state["circuit_open_at"] = state["last_failure"]

    record_provider_failure(name, bool(state.get("circuit_open")))


def _should_skip_provider(name: str) -> bool:
    state = _PROVIDER_STATE.get(name)
    if not state or not state.get("circuit_open"):
        return False

    opened_at = state.get("circuit_open_at")
    if opened_at is None:
        return False

    elapsed = (datetime.now(timezone.utc) - opened_at).total_seconds()
    if elapsed >= _CIRCUIT_COOLDOWN_SECONDS:
        state["circuit_open"] = False
        state["failure_streak"] = 0
        state["circuit_open_at"] = None
        logger.info("Reclosing circuit for provider %s after cooldown", name)
        set_provider_circuit(name, False)
        return False

    set_provider_circuit(name, True)
    return True


async def _fetch_coingecko_price(
    symbol: str,
    get_coin_price_data,
    symbol_to_coingecko_id,
) -> Dict[str, Any]:
    coin_id = symbol_to_coingecko_id(symbol)
    return await get_coin_price_data(coin_id, days=1)


async def _fetch_price_with_failover(
    symbol: str,
    *,
    price_fetchers: Dict[str, Any],
    errors: Dict[str, List[str]],
) -> Optional[str]:
    provider_used: Optional[str] = None
    for provider in _PROVIDER_PRIORITY:
        fetcher = price_fetchers.get(provider)
        if fetcher is None:
            continue
        if _should_skip_provider(provider):
            errors[symbol].append(f"{provider}: circuit_open")
            continue
        try:
            await fetcher()
            _record_provider_success(provider)
            provider_used = provider
            break
        except Exception as exc:  # noqa: BLE001
            _record_provider_failure(provider, exc)
            errors[symbol].append(f"{provider}: {exc}")
            logger.warning("Provider %s failed for %s: %s", provider, symbol, exc)
    return provider_used


def _provider_snapshot() -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for name, state in _PROVIDER_STATE.items():
        snapshot[name] = {
            "success": int(state.get("success", 0)),
            "failures": int(state.get("failures", 0)),
            "failure_streak": int(state.get("failure_streak", 0)),
            "circuit_open": bool(state.get("circuit_open", False)),
            "last_success": _to_iso(state.get("last_success")),
            "last_failure": _to_iso(state.get("last_failure")),
            "last_error": state.get("last_error"),
        }
    return snapshot


async def start_scheduler(app) -> None:
    """Initialise and start the AsyncIO scheduler if enabled."""

    global _SCHEDULER

    if _scheduler_disabled():
        logger.info("Background scheduler disabled via environment")
        return

    if _SCHEDULER is not None:
        return

    symbols = _resolve_symbols()
    refresh_seconds = int(os.getenv("QUANTUM_TRADER_REFRESH_SECONDS", "180"))
    _SCHEDULER_CONFIG["refresh_seconds"] = refresh_seconds
    execution_interval = _execution_interval_seconds()
    _SCHEDULER_CONFIG["execution_seconds"] = execution_interval

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        warm_market_caches,
        "interval",
        seconds=max(refresh_seconds, 60),
        args=[symbols],
        id="market-cache-refresh",
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now(timezone.utc),
    )

    liquidity_interval = _liquidity_interval_seconds()
    if liquidity_interval > 0:
        scheduler.add_job(
            _run_liquidity_refresh,
            "interval",
            seconds=max(liquidity_interval, 300),
            id="liquidity-refresh",
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(timezone.utc),
        )

    if execution_interval > 0:
        scheduler.add_job(
            _run_execution_cycle,
            "interval",
            seconds=max(execution_interval, 300),
            id="execution-rebalance",
            coalesce=True,
            max_instances=1,
            next_run_time=datetime.now(timezone.utc),
        )

    # Add AI model retraining job (daily at 3 AM UTC)
    ai_retraining_enabled = os.getenv("QT_AI_RETRAINING_ENABLED", "1") == "1"
    if ai_retraining_enabled:
        scheduler.add_job(
            _run_ai_retraining,
            "cron",
            hour=3,
            minute=0,
            id="ai-retraining",
            coalesce=True,
            max_instances=1,
        )
        logger.info("Scheduled AI retraining job: daily at 03:00 UTC")

    scheduler.start()
    _SCHEDULER = scheduler
    setattr(app.state, "scheduler", scheduler)

    logger.info(
        "Started background scheduler for symbols %s (interval %ss)",
        symbols,
        refresh_seconds,
    )

    # Kick off an initial warm-up without blocking app startup too long.
    asyncio.create_task(_initial_warm(symbols))


async def _initial_warm(symbols: Iterable[str]) -> None:
    try:
        await warm_market_caches(symbols)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Initial market cache warm-up failed: %s", exc)


async def shutdown_scheduler(app) -> None:
    """Tear down the scheduler if it was started."""

    global _SCHEDULER

    scheduler: Optional[AsyncIOScheduler] = getattr(app.state, "scheduler", None)
    if scheduler is None:
        return

    scheduler.shutdown(wait=False)
    logger.info("Background scheduler stopped")
    _SCHEDULER = None
    if hasattr(app.state, "scheduler"):
        delattr(app.state, "scheduler")


def get_scheduler_snapshot() -> Dict[str, Any]:
    """Return a serialisable snapshot of scheduler state."""

    scheduler = _SCHEDULER
    job_info: Optional[Dict[str, Any]] = None
    liquidity_job_info: Optional[Dict[str, Any]] = None
    execution_job_info: Optional[Dict[str, Any]] = None
    if scheduler is not None:
        job = scheduler.get_job("market-cache-refresh")
        if job is not None:
            job_info = {
                "id": job.id,
                "next_run_time": _to_iso(job.next_run_time),
                "trigger": str(job.trigger),
            }
        liquidity_job = scheduler.get_job("liquidity-refresh")
        if liquidity_job is not None:
            liquidity_job_info = {
                "id": liquidity_job.id,
                "next_run_time": _to_iso(liquidity_job.next_run_time),
                "trigger": str(liquidity_job.trigger),
            }
        execution_job = scheduler.get_job("execution-rebalance")
        if execution_job is not None:
            execution_job_info = {
                "id": execution_job.id,
                "next_run_time": _to_iso(execution_job.next_run_time),
                "trigger": str(execution_job.trigger),
            }

    last_run = {
        "status": _RUN_STATE.get("status", "unknown"),
        "started_at": _to_iso(_RUN_STATE.get("last_started")),
        "completed_at": _to_iso(_RUN_STATE.get("last_completed")),
        "duration_seconds": _RUN_STATE.get("last_duration"),
        "symbols": list(_RUN_STATE.get("symbols", [])),
        "successful_symbols": list(_RUN_STATE.get("successful_symbols", [])),
        "errors": dict(_RUN_STATE.get("errors", {})),
        "runs": int(_RUN_STATE.get("runs", 0)),
        "providers": dict(_RUN_STATE.get("providers", {})),
    }

    return {
        "enabled": not _scheduler_disabled(),
        "running": bool(scheduler and scheduler.running),
        "config": {
            "symbols": list(_SCHEDULER_CONFIG.get("symbols", _DEFAULT_SYMBOLS)),
            "refresh_seconds": int(_SCHEDULER_CONFIG.get("refresh_seconds", 180)),
            "liquidity_refresh_seconds": _liquidity_interval_seconds(),
            "execution_seconds": int(_SCHEDULER_CONFIG.get("execution_seconds", 0)),
        },
        "job": job_info,
        "liquidity_job": liquidity_job_info,
        "execution_job": execution_job_info,
        "last_run": last_run,
        "providers": _provider_snapshot(),
        "liquidity": {
            "status": _LIQUIDITY_STATE.get("status", "unknown"),
            "last_started": _to_iso(_LIQUIDITY_STATE.get("last_started")),
            "last_completed": _to_iso(_LIQUIDITY_STATE.get("last_completed")),
            "duration_seconds": _LIQUIDITY_STATE.get("last_duration"),
            "provider_primary": _LIQUIDITY_STATE.get("provider_primary"),
            "universe_size": _LIQUIDITY_STATE.get("universe_size"),
            "selection_size": _LIQUIDITY_STATE.get("selection_size"),
            "runs": int(_LIQUIDITY_STATE.get("runs", 0)),
            "error": _LIQUIDITY_STATE.get("error"),
            "run_id": _LIQUIDITY_STATE.get("run_id"),
            "analytics": _LIQUIDITY_STATE.get("analytics"),
        },
        "execution": {
            "status": _EXECUTION_STATE.get("status", "unknown"),
            "last_started": _to_iso(_EXECUTION_STATE.get("last_started")),
            "last_completed": _to_iso(_EXECUTION_STATE.get("last_completed")),
            "duration_seconds": _EXECUTION_STATE.get("last_duration"),
            "orders_planned": _EXECUTION_STATE.get("orders_planned"),
            "orders_submitted": _EXECUTION_STATE.get("orders_submitted"),
            "orders_skipped": _EXECUTION_STATE.get("orders_skipped"),
            "orders_failed": _EXECUTION_STATE.get("orders_failed"),
            "runs": int(_EXECUTION_STATE.get("runs", 0)),
            "error": _EXECUTION_STATE.get("error"),
            "run_id": _EXECUTION_STATE.get("run_id"),
            "gross_exposure": _EXECUTION_STATE.get("gross_exposure", 0.0),
            "positions_synced": bool(_EXECUTION_STATE.get("positions_synced", False)),
        },
        "provider_priority": list(_PROVIDER_PRIORITY),
    }


def get_scheduler_config() -> Dict[str, Any]:
    """Return the current scheduler configuration without run state."""

    snapshot = get_scheduler_snapshot()
    return snapshot.get("config", {}).copy()


def get_configured_symbols() -> List[str]:
    """Return the configured scheduler symbol universe."""

    return list(_SCHEDULER_CONFIG.get("symbols", _DEFAULT_SYMBOLS))


async def run_market_cache_refresh_now(symbols: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Execute a market cache warm-up immediately and return summary state."""

    target_symbols = list(symbols) if symbols else get_configured_symbols()
    await warm_market_caches(target_symbols)
    snapshot = get_scheduler_snapshot()
    last_run = snapshot.get("last_run", {})
    return {"symbols": target_symbols, "last_run": last_run}


async def run_liquidity_refresh_now() -> Dict[str, Any]:
    """Run the liquidity refresh job immediately and return the latest state."""

    await _run_liquidity_refresh()
    snapshot = get_scheduler_snapshot()
    return snapshot.get("liquidity", {}).copy()


async def run_execution_cycle_now() -> Dict[str, Any]:
    """Run the execution rebalance cycle immediately and return the latest state."""

    await _run_execution_cycle()
    snapshot = get_scheduler_snapshot()
    return snapshot.get("execution", {}).copy()


async def _run_liquidity_refresh() -> None:
    if os.getenv("QUANTUM_TRADER_DISABLE_LIQUIDITY") == "1":
        logger.info("Liquidity refresh disabled via environment")
        return

    started_at = datetime.now(timezone.utc)
    _LIQUIDITY_STATE["last_started"] = started_at
    _LIQUIDITY_STATE["status"] = "running"
    _LIQUIDITY_STATE["error"] = None

    try:
        from backend.database import SessionLocal

        try:
            from backend.config import load_liquidity_config  # type: ignore
        except ImportError:  # pragma: no cover - package fallback
            from config import load_liquidity_config  # type: ignore

        try:
            from backend.services.liquidity import refresh_liquidity  # type: ignore
        except ImportError:  # pragma: no cover - package fallback
            from services.liquidity import refresh_liquidity  # type: ignore

        config = load_liquidity_config()
        db = SessionLocal()
        try:
            # Ensure liquidity refresh cannot hang the scheduler indefinitely
            result = await asyncio.wait_for(refresh_liquidity(db, config=config), timeout=45)
        finally:
            db.close()

        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        _LIQUIDITY_STATE.update(
            {
                "status": "ok",
                "last_completed": completed_at,
                "last_duration": duration_seconds,
                "provider_primary": result.get("provider_primary"),
                "universe_size": result.get("universe_size", 0),
                "selection_size": result.get("selection_size", 0),
                "run_id": result.get("run_id"),
                "runs": int(_LIQUIDITY_STATE.get("runs", 0)) + 1,
                "error": None,
                "analytics": result.get("analytics"),
            }
        )
        record_scheduler_run("liquidity-refresh", "ok", duration_seconds)
        logger.info(
            "Liquidity universe refresh completed: run_id=%s primary=%s universe=%s selection=%s",
            result.get("run_id"),
            result.get("provider_primary"),
            result.get("universe_size"),
            result.get("selection_size"),
        )
    except asyncio.TimeoutError as exc:  # noqa: BLE001
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        _LIQUIDITY_STATE.update(
            {
                "status": "failed",
                "last_completed": completed_at,
                "last_duration": duration_seconds,
                "error": "Liquidity refresh timed out",
                "run_id": None,
                "runs": int(_LIQUIDITY_STATE.get("runs", 0)) + 1,
                "analytics": None,
            }
        )
        record_scheduler_run("liquidity-refresh", "failed", duration_seconds)
        logger.warning("Liquidity refresh timed out after %ss", duration_seconds)
    except Exception as exc:  # noqa: BLE001
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        _LIQUIDITY_STATE.update(
            {
                "status": "failed",
                "last_completed": completed_at,
                "last_duration": duration_seconds,
                "error": str(exc),
                "run_id": None,
                "runs": int(_LIQUIDITY_STATE.get("runs", 0)) + 1,
                "analytics": None,
            }
        )
        record_scheduler_run("liquidity-refresh", "failed", duration_seconds)
        logger.exception("Automated liquidity refresh failed")


async def _run_execution_cycle() -> None:
    if os.getenv("QUANTUM_TRADER_DISABLE_EXECUTION") == "1":
        if _EXECUTION_STATE.get("status") != "disabled":
            logger.info("Execution loop disabled via environment")
        _EXECUTION_STATE.update({"status": "disabled", "error": None, "positions_synced": False})
        return

    started_at = datetime.now(timezone.utc)
    _EXECUTION_STATE.update(
        {
            "status": "running",
            "last_started": started_at,
            "error": None,
            "orders_planned": 0,
            "orders_submitted": 0,
            "orders_skipped": 0,
            "orders_failed": 0,
            "gross_exposure": 0.0,
            "positions_synced": False,
        }
    )

    try:
        try:
            from backend.database import SessionLocal
        except ImportError:  # pragma: no cover - package fallback
            from database import SessionLocal  # type: ignore

        try:
            from backend.services.execution.execution import run_portfolio_rebalance  # type: ignore
        except ImportError:  # pragma: no cover - package fallback
            from services.execution import run_portfolio_rebalance  # type: ignore

        db = SessionLocal()
        try:
            result = await run_portfolio_rebalance(db)
        finally:
            db.close()

        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        _EXECUTION_STATE.update(
            {
                "last_completed": completed_at,
                "last_duration": duration_seconds,
                "runs": int(_EXECUTION_STATE.get("runs", 0)) + 1,
                "orders_planned": int(result.get("orders_planned", 0)),
                "orders_submitted": int(result.get("orders_submitted", 0)),
                "orders_skipped": int(result.get("orders_skipped", 0)),
                "orders_failed": int(result.get("orders_failed", 0)),
                "run_id": result.get("run_id"),
                "gross_exposure": float(result.get("gross_exposure", 0.0)),
                "positions_synced": bool(result.get("positions_synced", False)),
            }
        )

        status = result.get("status", "ok")
        if status in {"ok", "no_portfolio"}:
            _EXECUTION_STATE["status"] = status
            _EXECUTION_STATE["error"] = None
        else:
            _EXECUTION_STATE["status"] = status
            _EXECUTION_STATE["error"] = result.get("error")

        record_scheduler_run("execution-rebalance", status, duration_seconds)
    except Exception as exc:  # noqa: BLE001
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        _EXECUTION_STATE.update(
            {
                "status": "error",
                "error": str(exc),
                "last_completed": completed_at,
                "last_duration": duration_seconds,
                "positions_synced": False,
            }
        )
        logger.exception("Execution cycle failed: %s", exc)
        record_scheduler_run("execution-rebalance", "error", duration_seconds)


async def _run_ai_retraining() -> None:
    """
    Scheduled job for automatic AI model retraining.
    
    Runs daily to retrain model on accumulated trading outcomes.
    """
    from backend.database import SessionLocal
    from backend.services.ai_trading_engine import create_ai_trading_engine
    from ai_engine.ensemble_manager import EnsembleManager
    
    logger.info("Starting scheduled AI retraining...")
    started_at = datetime.now(timezone.utc)
    
    db = SessionLocal()
    try:
        agent = EnsembleManager()
        ai_engine = create_ai_trading_engine(agent=agent, db_session=db)
        
        # Run retraining (requires at least 100 samples by default)
        result = await ai_engine._retrain_model(min_samples=100)
        
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        
        status = result.get("status", "unknown")
        
        if status == "success":
            logger.info(
                f"AI retraining completed successfully: "
                f"version={result.get('version_id')} "
                f"train_acc={result.get('train_accuracy', 0):.2%} "
                f"val_acc={result.get('validation_accuracy', 0):.2%} "
                f"duration={duration_seconds:.1f}s"
            )
        elif status == "skipped":
            logger.info(
                f"AI retraining skipped: {result.get('reason')} "
                f"(samples: {result.get('samples_count', 0)}/{result.get('required', 0)})"
            )
        else:
            logger.warning(
                f"AI retraining failed: {result.get('reason', 'unknown error')}"
            )
        
        record_scheduler_run("ai-retraining", status, duration_seconds)
        
    except Exception as exc:
        completed_at = datetime.now(timezone.utc)
        duration_seconds = (completed_at - started_at).total_seconds()
        logger.exception("AI retraining failed: %s", exc)
        record_scheduler_run("ai-retraining", "error", duration_seconds)
    finally:
        db.close()
