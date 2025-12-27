import pytest

from backend.utils import scheduler
from backend.utils.telemetry import (  # type: ignore[import-error]
    PROVIDER_FAILURES,
    PROVIDER_SUCCESSES,
    SCHEDULER_RUN_TOTAL,
)


@pytest.fixture(autouse=True)
def reset_scheduler_state():
    scheduler._reset_internal_state_for_tests()
    yield
    scheduler._reset_internal_state_for_tests()


@pytest.mark.asyncio
async def test_warm_market_caches_invokes_providers(monkeypatch):
    calls = []

    async def fake_binance(symbol: str, limit: int = 600):
        calls.append(("binance", symbol, limit))
        return {"candles": []}

    async def fake_sentiment(symbol: str):
        calls.append(("sentiment", symbol))
        return {"score": 0.5}

    monkeypatch.setattr(
        "backend.routes.external_data.binance_ohlcv", fake_binance
    )
    monkeypatch.setattr(
        "backend.routes.external_data.twitter_sentiment", fake_sentiment
    )

    symbols = ["BTCUSDT", "ETHUSDT"]
    before_success = PROVIDER_SUCCESSES.labels(provider="binance")._value.get()
    before_runs = SCHEDULER_RUN_TOTAL.labels(job_id="market-cache-refresh", status="ok")._value.get()
    await scheduler.warm_market_caches(symbols)

    assert ("binance", "BTCUSDT", 600) in calls
    assert ("binance", "ETHUSDT", 600) in calls
    assert ("sentiment", "BTCUSDT") in calls
    assert ("sentiment", "ETHUSDT") in calls

    snapshot = scheduler.get_scheduler_snapshot()
    assert snapshot["last_run"]["status"] == "ok"
    assert "BTCUSDT" in snapshot["last_run"]["successful_symbols"]
    assert snapshot["last_run"]["errors"] == {}
    providers = snapshot["providers"]
    assert providers["binance"]["success"] >= 2
    assert providers["binance"]["failures"] == 0
    after_success = PROVIDER_SUCCESSES.labels(provider="binance")._value.get()
    after_runs = SCHEDULER_RUN_TOTAL.labels(job_id="market-cache-refresh", status="ok")._value.get()
    assert after_success == pytest.approx(before_success + len(symbols))
    assert after_runs == pytest.approx(before_runs + 1)


def test_scheduler_disabled_when_env_set(monkeypatch):
    monkeypatch.setenv("QUANTUM_TRADER_DISABLE_SCHEDULER", "1")
    assert scheduler._scheduler_disabled() is True
    monkeypatch.delenv("QUANTUM_TRADER_DISABLE_SCHEDULER", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "marker")
    assert scheduler._scheduler_disabled() is True
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "")
    assert scheduler._scheduler_disabled() is False


@pytest.mark.asyncio
async def test_snapshot_records_errors(monkeypatch):
    async def failing_binance(symbol: str, limit: int = 600):  # noqa: ARG001
        raise RuntimeError("boom")

    async def failing_sentiment(symbol: str):  # noqa: ARG001
        raise RuntimeError("oops")

    monkeypatch.setattr(
        "backend.routes.external_data.binance_ohlcv", failing_binance
    )
    monkeypatch.setattr(
        "backend.routes.external_data.twitter_sentiment", failing_sentiment
    )

    before_degraded = SCHEDULER_RUN_TOTAL.labels(job_id="market-cache-refresh", status="degraded")._value.get()
    await scheduler.warm_market_caches(["BTCUSDT"])

    snapshot = scheduler.get_scheduler_snapshot()
    assert snapshot["last_run"]["status"] == "degraded"
    assert "BTCUSDT" in snapshot["last_run"]["errors"]
    after_degraded = SCHEDULER_RUN_TOTAL.labels(job_id="market-cache-refresh", status="degraded")._value.get()
    assert after_degraded == pytest.approx(before_degraded + 1)


@pytest.mark.asyncio
async def test_price_provider_failover(monkeypatch):
    async def failing_binance(symbol: str, limit: int = 600):  # noqa: ARG001
        raise RuntimeError("binance down")

    async def fake_coingecko(coin_id: str, days: int = 1):  # noqa: ARG001
        return {"prices": [[0, 100.0]]}

    def fake_symbol_to_id(symbol: str) -> str:  # noqa: ARG001
        return "bitcoin"

    async def fake_sentiment(symbol: str):  # noqa: ARG001
        return {"score": 0.5}

    monkeypatch.setattr(
        "backend.routes.external_data.binance_ohlcv", failing_binance
    )
    monkeypatch.setattr(
        "backend.routes.coingecko_data.get_coin_price_data", fake_coingecko
    )
    monkeypatch.setattr(
        "backend.routes.coingecko_data.symbol_to_coingecko_id", fake_symbol_to_id
    )
    monkeypatch.setattr(
        "backend.routes.external_data.twitter_sentiment", fake_sentiment
    )

    before_failures = PROVIDER_FAILURES.labels(provider="binance")._value.get()
    await scheduler.warm_market_caches(["BTCUSDT"])

    snapshot = scheduler.get_scheduler_snapshot()
    providers = snapshot["providers"]
    assert providers["binance"]["failures"] >= 1
    assert providers["coingecko"]["success"] >= 1
    assert "binance" in snapshot["last_run"]["errors"]["BTCUSDT"][0]
    after_failures = PROVIDER_FAILURES.labels(provider="binance")._value.get()
    assert after_failures == pytest.approx(before_failures + 1)


@pytest.mark.asyncio
async def test_run_liquidity_refresh_updates_state(monkeypatch):
    monkeypatch.delenv("QUANTUM_TRADER_DISABLE_LIQUIDITY", raising=False)

    class DummySession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = DummySession()

    monkeypatch.setattr("backend.database.SessionLocal", lambda: session, raising=False)
    monkeypatch.setattr("backend.config.load_liquidity_config", lambda: object(), raising=False)

    analytics_payload = {
        "selection_size": 12,
        "top_allocations": [{"symbol": "BTCUSDT", "allocation_score": 0.42}],
    }

    async def fake_refresh(db, *, config=None):  # noqa: ARG001
        assert db is session
        return {
            "run_id": 42,
            "universe_size": 123,
            "selection_size": 12,
            "provider_primary": "binance",
            "analytics": analytics_payload,
        }

    monkeypatch.setattr("backend.services.liquidity.refresh_liquidity", fake_refresh, raising=False)

    before_ok = SCHEDULER_RUN_TOTAL.labels(job_id="liquidity-refresh", status="ok")._value.get()
    await scheduler._run_liquidity_refresh()
    after_ok = SCHEDULER_RUN_TOTAL.labels(job_id="liquidity-refresh", status="ok")._value.get()

    assert session.closed is True
    assert after_ok == pytest.approx(before_ok + 1)

    snapshot = scheduler.get_scheduler_snapshot()
    liquidity = snapshot["liquidity"]
    assert liquidity["status"] == "ok"
    assert liquidity["run_id"] == 42
    assert liquidity["universe_size"] == 123
    assert liquidity["selection_size"] == 12
    assert liquidity["provider_primary"] == "binance"
    assert liquidity["error"] is None
    assert liquidity["runs"] == 1
    assert liquidity["analytics"] == analytics_payload


@pytest.mark.asyncio
async def test_run_liquidity_refresh_failure_records_error(monkeypatch):
    monkeypatch.delenv("QUANTUM_TRADER_DISABLE_LIQUIDITY", raising=False)

    class DummySession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = DummySession()

    monkeypatch.setattr("backend.database.SessionLocal", lambda: session, raising=False)
    monkeypatch.setattr("backend.config.load_liquidity_config", lambda: object(), raising=False)

    async def failing_refresh(db, *, config=None):  # noqa: ARG001
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("backend.services.liquidity.refresh_liquidity", failing_refresh, raising=False)

    before_failed = SCHEDULER_RUN_TOTAL.labels(job_id="liquidity-refresh", status="failed")._value.get()
    await scheduler._run_liquidity_refresh()
    after_failed = SCHEDULER_RUN_TOTAL.labels(job_id="liquidity-refresh", status="failed")._value.get()

    assert session.closed is True
    assert after_failed == pytest.approx(before_failed + 1)

    snapshot = scheduler.get_scheduler_snapshot()
    liquidity = snapshot["liquidity"]
    assert liquidity["status"] == "failed"
    assert liquidity["run_id"] is None
    assert liquidity["error"] and "provider unavailable" in liquidity["error"]
    assert liquidity["runs"] == 1


@pytest.mark.asyncio
async def test_run_execution_cycle_disabled(monkeypatch):
    monkeypatch.setenv("QUANTUM_TRADER_DISABLE_EXECUTION", "1")

    await scheduler._run_execution_cycle()

    snapshot = scheduler.get_scheduler_snapshot()
    execution = snapshot["execution"]
    assert execution["status"] == "disabled"
    assert execution["error"] is None
    assert execution["runs"] == 0
    assert execution["positions_synced"] is False


@pytest.mark.asyncio
async def test_run_execution_cycle_records_success(monkeypatch):
    monkeypatch.delenv("QUANTUM_TRADER_DISABLE_EXECUTION", raising=False)

    class DummySession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = DummySession()

    monkeypatch.setattr("backend.database.SessionLocal", lambda: session, raising=False)

    async def fake_rebalance(db):  # noqa: ARG001
        return {
            "status": "no_portfolio",
            "orders_planned": 0,
            "orders_submitted": 0,
            "orders_skipped": 0,
            "orders_failed": 0,
            "run_id": None,
            "gross_exposure": 0.0,
            "positions_synced": True,
        }

    monkeypatch.setattr(
        "backend.services.execution.run_portfolio_rebalance",
        fake_rebalance,
        raising=False,
    )

    before = SCHEDULER_RUN_TOTAL.labels(job_id="execution-rebalance", status="no_portfolio")._value.get()
    await scheduler._run_execution_cycle()
    after = SCHEDULER_RUN_TOTAL.labels(job_id="execution-rebalance", status="no_portfolio")._value.get()

    assert session.closed is True
    assert after == pytest.approx(before + 1)

    execution = scheduler.get_scheduler_snapshot()["execution"]
    assert execution["status"] == "no_portfolio"
    assert execution["runs"] == 1
    assert execution["orders_planned"] == 0
    assert execution["error"] is None
    assert execution["positions_synced"] is True
    assert execution["gross_exposure"] == 0.0


@pytest.mark.asyncio
async def test_run_execution_cycle_failure_records_error(monkeypatch):
    monkeypatch.delenv("QUANTUM_TRADER_DISABLE_EXECUTION", raising=False)

    class DummySession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = DummySession()
    monkeypatch.setattr("backend.database.SessionLocal", lambda: session, raising=False)

    async def failing_rebalance(db):  # noqa: ARG001
        raise RuntimeError("exchange offline")

    monkeypatch.setattr(
        "backend.services.execution.run_portfolio_rebalance",
        failing_rebalance,
        raising=False,
    )

    before = SCHEDULER_RUN_TOTAL.labels(job_id="execution-rebalance", status="error")._value.get()
    await scheduler._run_execution_cycle()
    after = SCHEDULER_RUN_TOTAL.labels(job_id="execution-rebalance", status="error")._value.get()

    assert session.closed is True
    assert after == pytest.approx(before + 1)

    execution = scheduler.get_scheduler_snapshot()["execution"]
    assert execution["status"] == "error"
    assert execution["error"] and "exchange offline" in execution["error"]
    assert execution["runs"] == 0
    assert execution["positions_synced"] is False
