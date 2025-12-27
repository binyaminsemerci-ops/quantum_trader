from datetime import datetime, timezone

import pytest

from backend.services.execution.execution import TradeStateStore, _evaluate_forced_exits, _exit_config_from_env


@pytest.fixture(autouse=True)
def _clear_force_exit_env(monkeypatch):
    for name in (
        "QT_FORCE_EXITS_ENABLED",
        "QT_SL_PCT",
        "QT_TP_PCT",
        "QT_TRAIL_PCT",
        "QT_PARTIAL_TP",
        "QT_FORCE_EXIT_SL_DEFAULT",
        "QT_FORCE_EXIT_TP_DEFAULT",
        "QT_FORCE_EXIT_TRAIL_DEFAULT",
        "QT_FORCE_EXIT_PARTIAL_DEFAULT",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def test_exit_config_uses_defaults_when_unset():
    cfg = _exit_config_from_env()
    assert pytest.approx(0.03, rel=1e-6) == cfg["sl_pct"]
    assert pytest.approx(0.05, rel=1e-6) == cfg["tp_pct"]
    assert pytest.approx(0.02, rel=1e-6) == cfg["trail_pct"]
    assert cfg["partial_tp"] == 0.0


def test_exit_config_can_be_disabled(monkeypatch):
    monkeypatch.setenv("QT_FORCE_EXITS_ENABLED", "false")
    cfg = _exit_config_from_env()
    assert cfg == {"sl_pct": None, "tp_pct": None, "trail_pct": None, "partial_tp": None}


def test_forced_exit_triggers_with_default_thresholds(tmp_path):
    store = TradeStateStore(tmp_path / "trade_state.json")
    state = {
        "side": "LONG",
        "qty": 1.0,
        "avg_entry": 100.0,
        "peak": 110.0,
        "trough": None,
        "opened_at": datetime.now(timezone.utc).isoformat(),
    }
    store.set("BTCUSDT", state)

    prices = {"BTCUSDT": 90.0}
    positions = {"BTCUSDT": 1.0}

    intents = _evaluate_forced_exits(
        prices=prices,
        positions=positions,
        min_notional=10.0,
        store=store,
    )

    assert len(intents) == 1
    intent = intents[0]
    assert intent.symbol == "BTCUSDT"
    assert intent.quantity == pytest.approx(1.0)
    assert "SL" in intent.reason
