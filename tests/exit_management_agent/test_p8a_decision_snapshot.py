"""Tests for PATCH-8A: decision snapshot capture.

Coverage goals
--------------
* decision_id is generated and present in every audit stream record.
* HSET written to quantum:hash:exit.decision:{decision_id} with all required
  snapshot fields and the configured TTL.
* SADD written to quantum:set:exit.pending_decisions:{symbol}.
* Snapshot payload is fully populated: position fields, formula fields, live
  action fields, and diverged/exit_score.
* Qwen3 fields default to empty / zero / false when qwen3_result is None.
* diverged=True when formula_action != qwen3_action (and Qwen3 was called).
* diverged=False when Qwen3 was not called.
* Snapshot write errors are swallowed — audit stream write is unaffected.
* Existing audit stream behaviour is unchanged (dry_run guard still fires).

Fakes
-----
_FullFake        — records xadd, hset_snapshot, sadd_pending_decision calls.
_SnapshotFail    — xadd succeeds; hset_snapshot raises; tests error swallowing.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.audit import AuditWriter
from microservices.exit_management_agent.models import (
    ExitDecision,
    ExitScoreState,
    PositionSnapshot,
    Qwen3LayerResult,
)


# ── Fakes ──────────────────────────────────────────────────────────────────────


class _FullFake:
    """Records xadd, hset_snapshot, and sadd_pending_decision calls."""

    def __init__(self) -> None:
        self.xadd_calls: list = []
        self.hset_calls: list = []   # [(key, mapping, ttl_sec)]
        self.sadd_calls: list = []   # [(key, decision_id)]

    async def xadd(self, stream: str, fields: dict) -> None:
        self.xadd_calls.append((stream, dict(fields)))

    async def hset_snapshot(self, key: str, mapping: dict, ttl_sec: int) -> None:
        self.hset_calls.append((key, dict(mapping), ttl_sec))

    async def sadd_pending_decision(self, key: str, decision_id: str) -> None:
        self.sadd_calls.append((key, decision_id))


class _SnapshotFail:
    """xadd succeeds; hset_snapshot raises to test error swallowing."""

    def __init__(self) -> None:
        self.xadd_calls: list = []
        self.hset_calls: list = []
        self.sadd_calls: list = []

    async def xadd(self, stream: str, fields: dict) -> None:
        self.xadd_calls.append((stream, dict(fields)))

    async def hset_snapshot(self, key: str, mapping: dict, ttl_sec: int) -> None:
        raise ConnectionError("Redis unavailable")

    async def sadd_pending_decision(self, key: str, decision_id: str) -> None:
        self.sadd_calls.append((key, decision_id))


# ── Factories ──────────────────────────────────────────────────────────────────

_AUDIT = "quantum:stream:exit.audit"
_METRICS = "quantum:stream:exit.metrics"


def _snap(**kw) -> PositionSnapshot:
    base = dict(
        symbol="ETHUSDT",
        side="LONG",
        quantity=0.5,
        entry_price=2_000.0,
        mark_price=2_200.0,
        leverage=5.0,
        stop_loss=1_900.0,
        take_profit=0.0,
        unrealized_pnl=100.0,
        entry_risk_usdt=50.0,
        sync_timestamp=1_700_000_000.0,
    )
    base.update(kw)
    return PositionSnapshot(**base)


def _score_state(snap: PositionSnapshot, formula_action: str = "HOLD") -> ExitScoreState:
    return ExitScoreState(
        symbol=snap.symbol,
        side=snap.side,
        R_net=0.5,
        age_sec=300.0,
        age_fraction=0.02,
        giveback_pct=0.0,
        distance_to_sl_pct=5.0,
        peak_price=snap.mark_price,
        mark_price=snap.mark_price,
        entry_price=snap.entry_price,
        leverage=snap.leverage,
        r_effective_t1=0.2,
        r_effective_lock=0.1,
        d_r_loss=0.0,
        d_r_gain=0.3,
        d_giveback=0.0,
        d_time=0.1,
        d_sl_proximity=0.0,
        exit_score=0.15,
        formula_action=formula_action,
        formula_urgency="LOW",
        formula_confidence=0.75,
        formula_reason="test formula reason",
    )


def _qwen3_result(action: str = "HOLD", fallback: bool = False) -> Qwen3LayerResult:
    return Qwen3LayerResult(
        action=action,
        confidence=0.80,
        reason="test qwen3 reason",
        fallback=fallback,
        latency_ms=142.0,
        raw="{}",
    )


def _decision(snap: PositionSnapshot, **kw) -> ExitDecision:
    base = dict(
        snapshot=snap,
        action="HOLD",
        reason="test",
        urgency="LOW",
        R_net=0.5,
        confidence=0.75,
        suggested_sl=None,
        suggested_qty_fraction=None,
        dry_run=True,
        score_state=None,
        qwen3_result=None,
    )
    base.update(kw)
    return ExitDecision(**base)


def _writer(fake, ttl: int = 14400) -> AuditWriter:
    return AuditWriter(fake, _AUDIT, _METRICS, decision_ttl_sec=ttl)


# ── decision_id presence ───────────────────────────────────────────────────────


class TestDecisionIdPresence:
    @pytest.mark.asyncio
    async def test_decision_id_in_stream_record(self):
        """Every audit stream record must contain a decision_id field."""
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        assert len(fake.xadd_calls) == 1
        _, fields = fake.xadd_calls[0]
        assert "decision_id" in fields
        assert len(fields["decision_id"]) == 36  # uuid4 format

    @pytest.mark.asyncio
    async def test_decision_id_unique_per_call(self):
        """Each write_decision call must produce a unique decision_id."""
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        writer = _writer(fake)
        await writer.write_decision(dec, "loop-001")
        await writer.write_decision(dec, "loop-002")

        assert len(fake.xadd_calls) == 2
        id1 = fake.xadd_calls[0][1]["decision_id"]
        id2 = fake.xadd_calls[1][1]["decision_id"]
        assert id1 != id2


# ── hash write ─────────────────────────────────────────────────────────────────


class TestHashWrite:
    @pytest.mark.asyncio
    async def test_hset_called_with_correct_key_prefix(self):
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        assert len(fake.hset_calls) == 1
        key, mapping, ttl = fake.hset_calls[0]
        assert key.startswith("quantum:hash:exit.decision:")

    @pytest.mark.asyncio
    async def test_hset_key_matches_stream_decision_id(self):
        """Hash key must embed the same decision_id as the stream record."""
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        decision_id_from_stream = fake.xadd_calls[0][1]["decision_id"]
        hash_key = fake.hset_calls[0][0]
        assert hash_key == f"quantum:hash:exit.decision:{decision_id_from_stream}"

    @pytest.mark.asyncio
    async def test_hset_ttl_matches_config(self):
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake, ttl=7200).write_decision(dec, "loop-001")

        _, _, ttl = fake.hset_calls[0]
        assert ttl == 7200

    @pytest.mark.asyncio
    async def test_hset_mapping_has_required_fields(self):
        """Snapshot mapping must contain all required keys."""
        required = {
            "decision_id", "ts_epoch", "symbol", "side",
            "entry_price", "mark_price", "quantity", "unrealized_pnl",
            "formula_action", "formula_conf",
            "qwen3_action", "qwen3_conf", "qwen3_reason", "qwen3_fallback",
            "live_action", "live_conf", "diverged", "exit_score",
        }
        fake = _FullFake()
        snap = _snap()
        ss = _score_state(snap)
        dec = _decision(snap, score_state=ss)
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert required.issubset(mapping.keys())

    @pytest.mark.asyncio
    async def test_hset_symbol_correct(self):
        fake = _FullFake()
        snap = _snap(symbol="SOLUSDT")
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert mapping["symbol"] == "SOLUSDT"

    @pytest.mark.asyncio
    async def test_hset_exit_score_from_score_state(self):
        fake = _FullFake()
        snap = _snap()
        ss = _score_state(snap)
        dec = _decision(snap, score_state=ss)
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert float(mapping["exit_score"]) == pytest.approx(ss.exit_score, abs=1e-4)

    @pytest.mark.asyncio
    async def test_hset_exit_score_zero_when_no_score_state(self):
        """Hard-guard decisions (score_state=None) must have exit_score=0.0."""
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap, score_state=None)
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert float(mapping["exit_score"]) == pytest.approx(0.0, abs=1e-4)


# ── symbol pending set ─────────────────────────────────────────────────────────


class TestSymbolPendingSet:
    @pytest.mark.asyncio
    async def test_sadd_called_with_correct_key(self):
        fake = _FullFake()
        snap = _snap(symbol="BTCUSDT")
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        assert len(fake.sadd_calls) == 1
        key, decision_id = fake.sadd_calls[0]
        assert key == "quantum:set:exit.pending_decisions:BTCUSDT"

    @pytest.mark.asyncio
    async def test_sadd_decision_id_matches_stream(self):
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        decision_id_from_stream = fake.xadd_calls[0][1]["decision_id"]
        _, sadd_id = fake.sadd_calls[0]
        assert sadd_id == decision_id_from_stream


# ── Qwen3 fields ───────────────────────────────────────────────────────────────


class TestQwen3Fields:
    @pytest.mark.asyncio
    async def test_qwen3_fields_present_when_called(self):
        fake = _FullFake()
        snap = _snap()
        ss = _score_state(snap, formula_action="HOLD")
        qr = _qwen3_result(action="FULL_CLOSE", fallback=False)
        dec = _decision(snap, score_state=ss, qwen3_result=qr, action="FULL_CLOSE")
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert mapping["qwen3_action"] == "FULL_CLOSE"
        assert float(mapping["qwen3_conf"]) == pytest.approx(0.80, abs=1e-4)
        assert mapping["qwen3_reason"] == "test qwen3 reason"
        assert mapping["qwen3_fallback"] == "false"

    @pytest.mark.asyncio
    async def test_qwen3_fields_empty_when_not_called(self):
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap, qwen3_result=None)
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert mapping["qwen3_action"] == ""
        assert float(mapping["qwen3_conf"]) == pytest.approx(0.0, abs=1e-4)
        assert mapping["qwen3_reason"] == ""
        assert mapping["qwen3_fallback"] == "false"

    @pytest.mark.asyncio
    async def test_diverged_true_when_formula_and_qwen3_differ(self):
        fake = _FullFake()
        snap = _snap()
        ss = _score_state(snap, formula_action="HOLD")
        qr = _qwen3_result(action="FULL_CLOSE")
        dec = _decision(snap, score_state=ss, qwen3_result=qr, action="FULL_CLOSE")
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert mapping["diverged"] == "true"

    @pytest.mark.asyncio
    async def test_diverged_false_when_formula_and_qwen3_agree(self):
        fake = _FullFake()
        snap = _snap()
        ss = _score_state(snap, formula_action="HOLD")
        qr = _qwen3_result(action="HOLD")
        dec = _decision(snap, score_state=ss, qwen3_result=qr, action="HOLD")
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert mapping["diverged"] == "false"

    @pytest.mark.asyncio
    async def test_diverged_false_when_qwen3_not_called(self):
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap, qwen3_result=None)
        await _writer(fake).write_decision(dec, "loop-001")

        _, mapping, _ = fake.hset_calls[0]
        assert mapping["diverged"] == "false"


# ── Error swallowing ───────────────────────────────────────────────────────────


class TestSnapshotErrorSwallowing:
    @pytest.mark.asyncio
    async def test_snapshot_error_does_not_propagate(self):
        """hset_snapshot failure must be logged and swallowed — not raised."""
        fake = _SnapshotFail()
        snap = _snap()
        dec = _decision(snap)
        # Should not raise even though hset_snapshot raises ConnectionError
        await _writer(fake).write_decision(dec, "loop-001")

        # Audit stream write still occurred
        assert len(fake.xadd_calls) == 1

    @pytest.mark.asyncio
    async def test_audit_stream_record_written_even_on_snapshot_error(self):
        """Audit stream record must be committed before snapshot attempt."""
        fake = _SnapshotFail()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-001")

        assert len(fake.xadd_calls) == 1
        _, fields = fake.xadd_calls[0]
        assert fields["symbol"] == "ETHUSDT"


# ── Existing audit behaviour unchanged ────────────────────────────────────────


class TestExistingBehaviourUnchanged:
    @pytest.mark.asyncio
    async def test_dry_run_guard_still_fires(self):
        """write_decision() must still raise RuntimeError for dry_run=False."""
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap, dry_run=False)
        with pytest.raises(RuntimeError, match="AUDIT_WRITE_GUARD"):
            await _writer(fake).write_decision(dec, "loop-001")
        # No Redis call should have been made
        assert len(fake.xadd_calls) == 0
        assert len(fake.hset_calls) == 0
        assert len(fake.sadd_calls) == 0

    @pytest.mark.asyncio
    async def test_mandatory_stream_fields_present(self):
        """Core stream fields must remain intact after PATCH-8A changes."""
        fake = _FullFake()
        snap = _snap()
        dec = _decision(snap)
        await _writer(fake).write_decision(dec, "loop-P8A")

        _, fields = fake.xadd_calls[0]
        for required in ("ts", "symbol", "side", "action", "dry_run", "source", "loop_id"):
            assert required in fields, f"Missing required stream field: {required!r}"
        assert fields["dry_run"] == "true"
        assert fields["source"] == "exit_management_agent"
        assert fields["loop_id"] == "loop-P8A"

    @pytest.mark.asyncio
    async def test_constructor_forbidden_stream_still_rejected(self):
        """Construction with a forbidden stream must still raise ValueError."""
        with pytest.raises(ValueError):
            AuditWriter(
                _FullFake(),
                "quantum:stream:apply.plan",   # categorically forbidden
                _METRICS,
            )
