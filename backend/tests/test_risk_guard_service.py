import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from backend.config.risk import RiskConfig
from backend.services.risk.risk_guard import (
    InMemoryRiskStateStore,
    KillSwitchState,
    RiskGuardService,
    SqliteRiskStateStore,
)
from backend.utils.telemetry import RISK_DAILY_LOSS, RISK_DENIALS  # type: ignore[import-error]


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        staging_mode=False,
        kill_switch=False,
        max_notional_per_trade=1000.0,
        max_daily_loss=500.0,
        allowed_symbols=["BTCUSDT", "ETHUSDT"],
        failsafe_reset_minutes=60,
        risk_state_db_path=None,
        admin_api_token=None,
        max_position_per_symbol=None,
        max_gross_exposure=None,
        min_unit_price=None,
        max_unit_price=None,
        max_price_staleness_seconds=None,
    )


@pytest.mark.asyncio
async def test_risk_guard_allows_trade_within_limits(risk_config):
    guard = RiskGuardService(risk_config)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=250.0)
    assert allowed is True
    assert reason == ""


@pytest.mark.asyncio
async def test_risk_guard_denies_on_kill_switch(risk_config):
    guard = RiskGuardService(risk_config)
    baseline = RISK_DENIALS.labels(reason="kill_switch")._value.get()
    await guard.set_kill_switch(True)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=250.0)
    assert allowed is False
    assert reason == "kill_switch"
    updated = RISK_DENIALS.labels(reason="kill_switch")._value.get()
    assert updated == pytest.approx(baseline + 1)


@pytest.mark.asyncio
async def test_risk_guard_denies_on_notional_limit(risk_config):
    guard = RiskGuardService(risk_config)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=5000.0)
    assert allowed is False
    assert reason == "notional_limit"


@pytest.mark.asyncio
async def test_risk_guard_denies_on_invalid_price(risk_config):
    guard = RiskGuardService(risk_config)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=100.0, price=0.0)
    assert allowed is False
    assert reason == "price_invalid"


@pytest.mark.asyncio
async def test_risk_guard_denies_on_price_floor(risk_config):
    config = replace(risk_config, min_unit_price=100.0)
    guard = RiskGuardService(config)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=200.0, price=50.0)
    assert allowed is False
    assert reason == "price_floor"


@pytest.mark.asyncio
async def test_risk_guard_denies_on_price_ceiling(risk_config):
    config = replace(risk_config, max_unit_price=10_000.0)
    guard = RiskGuardService(config)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=200.0, price=20_000.0)
    assert allowed is False
    assert reason == "price_ceiling"


@pytest.mark.asyncio
async def test_risk_guard_denies_on_stale_price(risk_config):
    config = replace(risk_config, max_price_staleness_seconds=60)
    guard = RiskGuardService(config)
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    allowed, reason = await guard.can_execute(
        symbol="BTCUSDT",
        notional=200.0,
        price=25_000.0,
        price_as_of=stale_time,
    )
    assert allowed is False
    assert reason == "price_stale"


@pytest.mark.asyncio
async def test_risk_guard_denies_on_daily_loss(risk_config):
    store = InMemoryRiskStateStore(window=timedelta(hours=24))
    guard = RiskGuardService(risk_config, store=store)
    await guard.record_execution(symbol="BTCUSDT", notional=200.0, pnl=-300.0)
    await guard.record_execution(symbol="BTCUSDT", notional=150.0, pnl=-250.0)
    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=100.0)
    assert allowed is False
    assert reason == "daily_loss_limit"


@pytest.mark.asyncio
async def test_risk_guard_snapshot_reflects_state(risk_config):
    guard = RiskGuardService(risk_config)
    await guard.record_execution(symbol="ETHUSDT", notional=50.0, pnl=-10.0)
    snapshot = await guard.snapshot()
    assert snapshot["config"]["max_notional_per_trade"] == pytest.approx(1000.0)
    assert snapshot["state"]["trade_count"] == 1
    assert snapshot["state"]["daily_loss"] == pytest.approx(10.0)
    assert snapshot["state"]["kill_switch_state"] is None
    assert snapshot["positions"]["total_notional"] >= 0.0
    assert RISK_DAILY_LOSS._value.get() >= 10.0


@pytest.mark.asyncio
async def test_risk_guard_denies_on_position_exposure(risk_config):
    config = replace(risk_config, max_position_per_symbol=1000.0)
    guard = RiskGuardService(config)
    allowed, reason = await guard.can_execute(
        symbol="BTCUSDT",
        notional=200.0,
        projected_notional=1500.0,
    )
    assert allowed is False
    assert reason == "position_limit"


@pytest.mark.asyncio
async def test_risk_guard_denies_on_gross_exposure(risk_config):
    config = replace(risk_config, max_gross_exposure=5000.0)
    guard = RiskGuardService(config)
    allowed, reason = await guard.can_execute(
        symbol="ETHUSDT",
        notional=200.0,
        projected_notional=1200.0,
        total_exposure=6000.0,
    )
    assert allowed is False
    assert reason == "gross_exposure_limit"


@pytest.mark.asyncio
async def test_kill_switch_override_persists_with_sqlite(tmp_path, risk_config):
    db_path = tmp_path / "risk_state.db"
    store = SqliteRiskStateStore(db_path)
    guard = RiskGuardService(risk_config, store=store)
    await guard.set_kill_switch(True, reason="test_persist")

    allowed, _ = await guard.can_execute(symbol="BTCUSDT", notional=100.0)
    assert allowed is False

    # New guard instance should load persisted override from SQLite
    new_store = SqliteRiskStateStore(db_path)
    new_guard = RiskGuardService(risk_config, store=new_store)
    allowed2, reason2 = await new_guard.can_execute(symbol="BTCUSDT", notional=100.0)
    assert allowed2 is False
    assert reason2 == "kill_switch"
    snapshot = await new_guard.snapshot()
    assert snapshot["state"]["kill_switch_state"]["reason"] == "test_persist"


@pytest.mark.asyncio
async def test_reset_clears_persistent_state(tmp_path, risk_config):
    db_path = tmp_path / "risk_state.db"
    store = SqliteRiskStateStore(db_path)
    guard = RiskGuardService(risk_config, store=store)
    await guard.record_execution(symbol="BTCUSDT", notional=100.0, pnl=-50.0)
    await guard.set_kill_switch(True)

    snapshot_before = await guard.snapshot()
    assert snapshot_before["state"]["trade_count"] == 1
    assert snapshot_before["state"]["kill_switch_override"] is True

    await guard.reset()

    snapshot_after = await guard.snapshot()
    assert snapshot_after["state"]["trade_count"] == 0
    assert snapshot_after["state"]["kill_switch_override"] is None
    assert snapshot_after["state"]["kill_switch_state"] is None
    assert RISK_DAILY_LOSS._value.get() >= 0.0


@pytest.mark.asyncio
async def test_kill_switch_state_serialization(risk_config):
    guard = RiskGuardService(risk_config)
    await guard.set_kill_switch(True, reason="manual_test")
    snapshot = await guard.snapshot()
    state = snapshot["state"]["kill_switch_state"]
    assert state["enabled"] is True
    assert state["reason"] == "manual_test"
    assert "updated_at" in state


@pytest.mark.asyncio
async def test_failsafe_auto_reset_clears_kill_switch(risk_config):
    config = replace(risk_config, failsafe_reset_minutes=1)
    store = InMemoryRiskStateStore()
    past = datetime.now(timezone.utc) - timedelta(minutes=5)
    await store.set_kill_switch_state(
        KillSwitchState(enabled=True, reason="breach", updated_at=past)
    )
    guard = RiskGuardService(config, store=store)

    allowed, reason = await guard.can_execute(symbol="BTCUSDT", notional=100.0)
    assert allowed is True
    assert reason == ""
    state = await guard.kill_switch_state()
    assert state is None
