"""Tests for PATCH-8B: OutcomeTracker position closure detection.

Coverage goals
--------------
* No outcomes generated on first tick (baseline establishment).
* Symbol disappearance triggers outcome event(s).
* Multiple pending decision_ids for the same symbol are each processed.
* Missing snapshot hash (expired / never written) does not crash tracker.
* Processed decision_ids are removed from the symbol pending set.
* Symbols present in both ticks produce no outcome.
* New symbols appearing produce no outcome (not a closure).
* Outcome event written to the correct stream with all required fields.
* hold_duration_sec computed correctly from ts_epoch in snapshot.
* closed_by logic: FULL_CLOSE/PARTIAL_CLOSE_25/TIME_STOP_EXIT → "exit_management_agent";
  HOLD / UNKNOWN → "unknown".
* SREM is NOT called if the xadd fails (pending set preserved for retry).
* enabled=False makes tracker a no-op on all ticks.
* smembers returning empty set → no outcome written.

Fakes
-----
_TrackerFake    — full fake RedisClient for OutcomeTracker; records calls.
_XaddFail       — xadd raises; tests that pending set is not cleaned up.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.outcome_tracker import (
    OutcomeTracker,
    _OUTCOMES_STREAM_DEFAULT,
    _PENDING_SET_PREFIX,
    _SNAPSHOT_HASH_PREFIX,
)


# ── Fakes ──────────────────────────────────────────────────────────────────────

_OUTCOMES = _OUTCOMES_STREAM_DEFAULT


class _TrackerFake:
    """
    Duck-typed fake RedisClient for OutcomeTracker tests.
    Supports all methods called by OutcomeTracker.
    """

    def __init__(
        self,
        smembers_results: dict | None = None,
        hgetall_results: dict | None = None,
        ticker_prices: dict | None = None,
    ) -> None:
        self.xadd_calls: list = []           # [(stream, fields)]
        self.srem_calls: list = []           # [(key, decision_id)]
        self._smembers: dict = smembers_results or {}
        self._hgetall: dict = hgetall_results or {}
        self._tickers: dict = ticker_prices or {}

    async def smembers_pending_decisions(self, key: str) -> set:
        return set(self._smembers.get(key, set()))

    async def hgetall_snapshot(self, key: str) -> dict:
        return dict(self._hgetall.get(key, {}))

    async def get_mark_price_from_ticker(self, symbol: str):
        return self._tickers.get(symbol)

    async def xadd(self, stream: str, fields: dict) -> None:
        self.xadd_calls.append((stream, dict(fields)))

    async def srem_pending_decision(self, key: str, decision_id: str) -> None:
        self.srem_calls.append((key, decision_id))


class _XaddFailFake(_TrackerFake):
    """xadd always raises; tests that srem is NOT called."""

    async def xadd(self, stream: str, fields: dict) -> None:
        raise ConnectionError("Redis write failed")


def _make_snapshot(
    symbol: str = "BTCUSDT",
    decision_id: str = "dec-001",
    ts_epoch: int = 1_000_000,
    live_action: str = "HOLD",
    exit_score: str = "0.3500",
) -> dict:
    return {
        "decision_id": decision_id,
        "ts_epoch": str(ts_epoch),
        "symbol": symbol,
        "side": "LONG",
        "entry_price": "30000.00000000",
        "mark_price": "31500.00000000",
        "quantity": "0.01000000",
        "unrealized_pnl": "15.0000",
        "formula_action": live_action,
        "formula_conf": "0.7500",
        "qwen3_action": "",
        "qwen3_conf": "0.0000",
        "qwen3_reason": "",
        "qwen3_fallback": "false",
        "live_action": live_action,
        "live_conf": "0.7500",
        "diverged": "false",
        "exit_score": exit_score,
    }


# ── First-tick baseline ────────────────────────────────────────────────────────


class TestFirstTickBaseline:
    @pytest.mark.asyncio
    async def test_no_outcome_on_first_tick(self):
        """First call establishes baseline — must not generate any outcomes."""
        fake = _TrackerFake()
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT", "ETHUSDT"})

        assert len(fake.xadd_calls) == 0
        assert len(fake.srem_calls) == 0

    @pytest.mark.asyncio
    async def test_baseline_used_on_second_tick(self):
        """Symbol present on tick-1 but absent on tick-2 should be detected."""
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-001"
        snap = _make_snapshot(symbol="BTCUSDT", decision_id="dec-001")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-001"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)

        await tracker.update({"BTCUSDT", "ETHUSDT"})  # tick 1 — baseline
        await tracker.update({"ETHUSDT"})              # tick 2 — BTC gone

        assert len(fake.xadd_calls) == 1


# ── Symbol closure detection ───────────────────────────────────────────────────


class TestSymbolClosureDetection:
    @pytest.mark.asyncio
    async def test_outcome_written_to_correct_stream(self):
        set_key = f"{_PENDING_SET_PREFIX}SOLUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-sol"
        snap = _make_snapshot(symbol="SOLUSDT", decision_id="dec-sol")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-sol"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"SOLUSDT"})
        await tracker.update(set())

        assert len(fake.xadd_calls) == 1
        stream, _ = fake.xadd_calls[0]
        assert stream == _OUTCOMES

    @pytest.mark.asyncio
    async def test_outcome_required_fields_present(self):
        set_key = f"{_PENDING_SET_PREFIX}ETHUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-eth"
        snap = _make_snapshot(symbol="ETHUSDT", decision_id="dec-eth")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-eth"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"ETHUSDT"})
        await tracker.update(set())

        required = {
            "decision_id", "symbol", "close_time_epoch", "hold_duration_sec",
            "outcome_action", "close_price", "close_pnl_usdt", "closed_by",
            "mae_pct", "mfe_pct", "entry_price", "quantity", "side",
            "exit_score", "source", "patch",
        }
        _, fields = fake.xadd_calls[0]
        assert required.issubset(fields.keys())

    @pytest.mark.asyncio
    async def test_outcome_decision_id_matches(self):
        sid = "dec-xyz-123"
        set_key = f"{_PENDING_SET_PREFIX}XRPUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}{sid}"
        snap = _make_snapshot(symbol="XRPUSDT", decision_id=sid)
        fake = _TrackerFake(
            smembers_results={set_key: {sid}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"XRPUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["decision_id"] == sid
        assert fields["symbol"] == "XRPUSDT"

    @pytest.mark.asyncio
    async def test_outcome_patch_tag(self):
        set_key = f"{_PENDING_SET_PREFIX}BNBUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-bnb"
        snap = _make_snapshot(symbol="BNBUSDT", decision_id="dec-bnb")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-bnb"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BNBUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["patch"] == "PATCH-8B"
        assert fields["source"] == "exit_management_agent"


# ── Multiple decision_ids ──────────────────────────────────────────────────────


class TestMultipleDecisionIds:
    @pytest.mark.asyncio
    async def test_all_decision_ids_generate_outcome(self):
        symbol = "ADAUSDT"
        ids = {"dec-1", "dec-2", "dec-3"}
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"
        snaps = {
            f"{_SNAPSHOT_HASH_PREFIX}{d}": _make_snapshot(symbol=symbol, decision_id=d)
            for d in ids
        }
        fake = _TrackerFake(smembers_results={set_key: ids}, hgetall_results=snaps)
        tracker = OutcomeTracker(fake)
        await tracker.update({symbol})
        await tracker.update(set())

        assert len(fake.xadd_calls) == 3
        written_ids = {f["decision_id"] for _, f in fake.xadd_calls}
        assert written_ids == ids

    @pytest.mark.asyncio
    async def test_all_processed_ids_removed_from_set(self):
        symbol = "SUIUSDT"
        ids = {"dec-a", "dec-b"}
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"
        snaps = {
            f"{_SNAPSHOT_HASH_PREFIX}{d}": _make_snapshot(symbol=symbol, decision_id=d)
            for d in ids
        }
        fake = _TrackerFake(smembers_results={set_key: ids}, hgetall_results=snaps)
        tracker = OutcomeTracker(fake)
        await tracker.update({symbol})
        await tracker.update(set())

        removed = {did for _, did in fake.srem_calls}
        assert removed == ids


# ── hold_duration_sec ─────────────────────────────────────────────────────────


class TestHoldDuration:
    @pytest.mark.asyncio
    async def test_hold_duration_computed_from_ts_epoch(self, monkeypatch):
        """hold_duration_sec = close_time - ts_epoch from snapshot."""
        import microservices.exit_management_agent.outcome_tracker as ot_module

        monkeypatch.setattr(ot_module.time, "time", lambda: 1_001_800.0)

        set_key = f"{_PENDING_SET_PREFIX}LTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-ltc"
        snap = _make_snapshot(
            symbol="LTCUSDT", decision_id="dec-ltc", ts_epoch=1_000_000
        )
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-ltc"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"LTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["hold_duration_sec"] == "1800"

    @pytest.mark.asyncio
    async def test_hold_duration_null_when_ts_missing(self):
        set_key = f"{_PENDING_SET_PREFIX}DOTUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-dot"
        snap = {"live_action": "HOLD", "side": "LONG"}  # no ts_epoch
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-dot"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"DOTUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["hold_duration_sec"] == "null"


# ── closed_by logic ────────────────────────────────────────────────────────────


class TestClosedByInference:
    @pytest.mark.asyncio
    async def test_full_close_closed_by_ema(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-fc"
        snap = _make_snapshot(decision_id="dec-fc", live_action="FULL_CLOSE")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-fc"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["closed_by"] == "exit_management_agent"

    @pytest.mark.asyncio
    async def test_partial_close_closed_by_ema(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-pc"
        snap = _make_snapshot(decision_id="dec-pc", live_action="PARTIAL_CLOSE_25")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-pc"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["closed_by"] == "exit_management_agent"

    @pytest.mark.asyncio
    async def test_time_stop_closed_by_ema(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-ts"
        snap = _make_snapshot(decision_id="dec-ts", live_action="TIME_STOP_EXIT")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-ts"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["closed_by"] == "exit_management_agent"

    @pytest.mark.asyncio
    async def test_hold_closed_by_unknown(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-hold"
        snap = _make_snapshot(decision_id="dec-hold", live_action="HOLD")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-hold"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["closed_by"] == "unknown"

    @pytest.mark.asyncio
    async def test_unknown_action_closed_by_unknown(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-unk"
        # Snapshot missing live_action entirely
        snap = {"ts_epoch": "1000000", "side": "LONG"}
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-unk"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["closed_by"] == "unknown"
        assert fields["outcome_action"] == "UNKNOWN"


# ── Missing snapshot hash ──────────────────────────────────────────────────────


class TestMissingSnapshotHash:
    @pytest.mark.asyncio
    async def test_empty_snapshot_does_not_crash(self):
        """Expired / missing snapshot — best-effort outcome with nulls."""
        set_key = f"{_PENDING_SET_PREFIX}AVAXUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-avax"
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-avax"}},
            hgetall_results={hash_key: {}},  # empty = expired
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"AVAXUSDT"})
        await tracker.update(set())

        # Should still emit an outcome event (best-effort)
        assert len(fake.xadd_calls) == 1
        _, fields = fake.xadd_calls[0]
        assert fields["hold_duration_sec"] == "null"
        assert fields["outcome_action"] == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_missing_snapshot_still_removes_from_pending_set(self):
        """Even with empty snapshot, processed id must be cleaned up."""
        set_key = f"{_PENDING_SET_PREFIX}MATICUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-matic"
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-matic"}},
            hgetall_results={hash_key: {}},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"MATICUSDT"})
        await tracker.update(set())

        assert len(fake.srem_calls) == 1
        _, removed_id = fake.srem_calls[0]
        assert removed_id == "dec-matic"


# ── Persistent id protection ───────────────────────────────────────────────────


class TestXaddFailurePreventsCleanup:
    @pytest.mark.asyncio
    async def test_srem_not_called_when_xadd_fails(self):
        """If outcome write fails, decision_id must NOT be removed from pending set."""
        set_key = f"{_PENDING_SET_PREFIX}LINKUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-link"
        snap = _make_snapshot(symbol="LINKUSDT", decision_id="dec-link")
        fake = _XaddFailFake(
            smembers_results={set_key: {"dec-link"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"LINKUSDT"})
        await tracker.update(set())

        assert len(fake.srem_calls) == 0


# ── No false positives ─────────────────────────────────────────────────────────


class TestNoFalsePositives:
    @pytest.mark.asyncio
    async def test_present_symbol_no_outcome(self):
        """Symbol present in both ticks must produce no outcome event."""
        fake = _TrackerFake()
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT", "ETHUSDT"})
        await tracker.update({"BTCUSDT", "ETHUSDT"})

        assert len(fake.xadd_calls) == 0

    @pytest.mark.asyncio
    async def test_new_symbol_no_outcome(self):
        """Newly appearing symbol (not present on tick-1) must produce no outcome."""
        fake = _TrackerFake()
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update({"BTCUSDT", "SOLUSDT"})  # SOL is new

        assert len(fake.xadd_calls) == 0

    @pytest.mark.asyncio
    async def test_empty_smembers_no_outcome(self):
        """Symbol that disappeared but has no pending decisions → no outcome."""
        set_key = f"{_PENDING_SET_PREFIX}DOGEUSDT"
        fake = _TrackerFake(smembers_results={set_key: set()})
        tracker = OutcomeTracker(fake)
        await tracker.update({"DOGEUSDT"})
        await tracker.update(set())

        assert len(fake.xadd_calls) == 0

    @pytest.mark.asyncio
    async def test_multiple_disappearances_across_ticks(self):
        """Each tick independently handles its own closed symbols."""
        set_a = f"{_PENDING_SET_PREFIX}AAVEUSDT"
        set_b = f"{_PENDING_SET_PREFIX}UNIUSDT"
        snap_a = _make_snapshot(symbol="AAVEUSDT", decision_id="dec-aave")
        snap_b = _make_snapshot(symbol="UNIUSDT", decision_id="dec-uni")
        fake = _TrackerFake(
            smembers_results={set_a: {"dec-aave"}, set_b: {"dec-uni"}},
            hgetall_results={
                f"{_SNAPSHOT_HASH_PREFIX}dec-aave": snap_a,
                f"{_SNAPSHOT_HASH_PREFIX}dec-uni": snap_b,
            },
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"AAVEUSDT", "UNIUSDT"})  # baseline
        await tracker.update({"UNIUSDT"})              # AAVE closes
        await tracker.update(set())                    # UNI closes

        assert len(fake.xadd_calls) == 2
        symbols = {f["symbol"] for _, f in fake.xadd_calls}
        assert symbols == {"AAVEUSDT", "UNIUSDT"}


# ── Close price from ticker ────────────────────────────────────────────────────


class TestClosePriceFromTicker:
    @pytest.mark.asyncio
    async def test_close_price_populated_when_ticker_available(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-btc"
        snap = _make_snapshot(decision_id="dec-btc")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-btc"}},
            hgetall_results={hash_key: snap},
            ticker_prices={"BTCUSDT": 45000.12345678},
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["close_price"] == "45000.12345678"

    @pytest.mark.asyncio
    async def test_close_price_null_when_ticker_unavailable(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-btc"
        snap = _make_snapshot(decision_id="dec-btc")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-btc"}},
            hgetall_results={hash_key: snap},
            ticker_prices={},  # no ticker
        )
        tracker = OutcomeTracker(fake)
        await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        _, fields = fake.xadd_calls[0]
        assert fields["close_price"] == "null"


# ── Disabled tracker ──────────────────────────────────────────────────────────


class TestDisabledTracker:
    @pytest.mark.asyncio
    async def test_disabled_tracker_never_writes(self):
        set_key = f"{_PENDING_SET_PREFIX}BTCUSDT"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}dec-dis"
        snap = _make_snapshot(decision_id="dec-dis")
        fake = _TrackerFake(
            smembers_results={set_key: {"dec-dis"}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(fake, enabled=False)
        await tracker.update({"BTCUSDT"})  # would normally be baseline
        await tracker.update(set())        # would normally detect closure

        assert len(fake.xadd_calls) == 0
        assert len(fake.srem_calls) == 0

    @pytest.mark.asyncio
    async def test_disabled_tracker_ignores_all_ticks(self):
        fake = _TrackerFake()
        tracker = OutcomeTracker(fake, enabled=False)
        for _ in range(10):
            await tracker.update({"BTCUSDT"})
        await tracker.update(set())

        assert len(fake.xadd_calls) == 0

    @pytest.mark.asyncio
    async def test_re_enabled_tracker_establishes_baseline_on_first_real_tick(self):
        """enabled=False then enabled=True: first update sets baseline, no false positiveson that tick."""
        fake = _TrackerFake()
        tracker = OutcomeTracker(fake, enabled=False)
        await tracker.update({"BTCUSDT"})  # disabled — no-op, _initialized stays False

        # Now enable externally (simulating config change or test override).
        tracker._enabled = True
        # First enabled tick establishes baseline — no outcomes generated.
        await tracker.update({"BTCUSDT"})

        # BTC still present — no outcome.
        await tracker.update({"BTCUSDT"})
        assert len(fake.xadd_calls) == 0
