"""Tests for audit.py — AuditWriter safety contracts and nominal shadow write path.

Coverage goals
--------------
* Constructor stream validation: forbidden / unknown streams must be rejected at
  construction time before any network call.
* dry_run guard: write_decision() must raise RuntimeError("AUDIT_WRITE_GUARD")
  when dec.dry_run is False, and must NOT reach Redis.
* Nominal write_decision path: correct stream, mandatory shadow-contract fields
  (dry_run="true", source, patch), suggested_sl inclusion logic, Redis error
  swallowed.
* write_metrics path: correct stream, correct payload fields, Redis error
  swallowed.

Fakes
-----
_RecordingFake  — duck-typed fake with only xadd(); records calls, no guard.
                  Isolates AuditWriter logic from redis_io layer.
_RaisingFake    — raises ConnectionError on xadd(); tests error-swallowing paths.
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.audit import AuditWriter
from microservices.exit_management_agent.models import ExitDecision, PositionSnapshot


# ── Fakes ──────────────────────────────────────────────────────────────────────


class _RecordingFake:
    """
    Duck-typed fake Redis client for audit tests.
    Implements ONLY xadd() — no write-guard, no network.
    Any method other than xadd() would raise AttributeError, which would
    immediately fail the test (useful implicit contract).
    """

    def __init__(self) -> None:
        self.xadd_calls: list = []

    async def xadd(self, stream: str, fields: dict) -> None:
        self.xadd_calls.append((stream, dict(fields)))


class _RaisingFake:
    """Fake that always raises on xadd(); tests error-swallowing paths."""

    async def xadd(self, stream: str, fields: dict) -> None:
        raise ConnectionError("Redis unavailable")


# ── Factories ──────────────────────────────────────────────────────────────────


def _snapshot(**kw) -> PositionSnapshot:
    base = dict(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.01,
        entry_price=30_000.0,
        mark_price=31_500.0,
        leverage=10.0,
        stop_loss=29_000.0,
        take_profit=0.0,
        unrealized_pnl=15.0,
        entry_risk_usdt=100.0,
        sync_timestamp=1_700_000_000.0,
    )
    base.update(kw)
    return PositionSnapshot(**base)


def _decision(snap: PositionSnapshot, action: str = "HOLD", dry_run: bool = True, **kw) -> ExitDecision:
    base = dict(
        snapshot=snap,
        action=action,
        reason="test",
        urgency="LOW",
        R_net=0.2,
        confidence=1.0,
        suggested_sl=None,
        suggested_qty_fraction=None,
        dry_run=dry_run,
    )
    base.update(kw)
    return ExitDecision(**base)


# ── AuditWriter constructor stream validation ─────────────────────────────────


class TestAuditWriterConstructor:
    def test_valid_streams_accepted(self):
        """Construction with both allowed streams must not raise."""
        aw = AuditWriter(
            _RecordingFake(),
            "quantum:stream:exit.audit",
            "quantum:stream:exit.metrics",
        )
        assert aw is not None

    def test_forbidden_audit_stream_raises_value_error(self):
        """trade.intent is on the categorical forbidden list."""
        with pytest.raises(ValueError, match="forbidden stream"):
            AuditWriter(
                _RecordingFake(),
                "quantum:stream:trade.intent",
                "quantum:stream:exit.metrics",
            )

    def test_forbidden_metrics_stream_raises_value_error(self):
        """apply.plan is on the categorical forbidden list."""
        with pytest.raises(ValueError, match="forbidden stream"):
            AuditWriter(
                _RecordingFake(),
                "quantum:stream:exit.audit",
                "quantum:stream:apply.plan",
            )

    def test_unknown_audit_stream_raises_value_error(self):
        """A stream that is neither forbidden nor allowed must be rejected."""
        with pytest.raises(ValueError, match="not on allowlist"):
            AuditWriter(
                _RecordingFake(),
                "quantum:stream:some.new.stream",
                "quantum:stream:exit.metrics",
            )


# ── dry_run guard ─────────────────────────────────────────────────────────────


class TestDryRunGuard:
    @pytest.mark.asyncio
    async def test_dry_run_false_raises_runtime_error(self):
        """This is the second shadow safety guard. Must raise AUDIT_WRITE_GUARD."""
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        dec = _decision(_snapshot(), dry_run=False)
        with pytest.raises(RuntimeError, match="AUDIT_WRITE_GUARD"):
            await aw.write_decision(dec, "testloop")

    @pytest.mark.asyncio
    async def test_dry_run_false_does_not_reach_redis(self):
        """Guard fires before the xadd call — fake must receive zero calls."""
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        dec = _decision(_snapshot(), dry_run=False)
        try:
            await aw.write_decision(dec, "testloop")
        except RuntimeError:
            pass
        assert len(fake.xadd_calls) == 0


# ── Nominal write_decision path ───────────────────────────────────────────────


class TestWriteDecisionNominal:
    @pytest.mark.asyncio
    async def test_write_goes_to_audit_stream_only(self):
        """write_decision must write to audit stream and nowhere else."""
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_decision(_decision(_snapshot()), "loop001")
        assert len(fake.xadd_calls) == 1
        stream, _ = fake.xadd_calls[0]
        assert stream == "quantum:stream:exit.audit"

    @pytest.mark.asyncio
    async def test_payload_shadow_contract_fields(self):
        """Every audit record must carry the three shadow-contract markers."""
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_decision(_decision(_snapshot()), "loop001")
        _, fields = fake.xadd_calls[0]
        assert fields["dry_run"] == "true"
        assert fields["source"] == "exit_management_agent"
        assert fields["patch"] == "PATCH-1"

    @pytest.mark.asyncio
    async def test_payload_contains_position_fields(self):
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_decision(_decision(_snapshot()), "loop001")
        _, fields = fake.xadd_calls[0]
        assert fields["symbol"] == "BTCUSDT"
        assert fields["side"] == "LONG"
        assert fields["action"] == "HOLD"

    @pytest.mark.asyncio
    async def test_suggested_sl_included_when_set(self):
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        dec = _decision(_snapshot(), action="MOVE_TO_BREAKEVEN", suggested_sl=30_060.0)
        await aw.write_decision(dec, "loop002")
        _, fields = fake.xadd_calls[0]
        assert "suggested_sl" in fields

    @pytest.mark.asyncio
    async def test_suggested_sl_absent_when_none(self):
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_decision(_decision(_snapshot(), suggested_sl=None), "loop003")
        _, fields = fake.xadd_calls[0]
        assert "suggested_sl" not in fields

    @pytest.mark.asyncio
    async def test_does_not_write_to_metrics_stream(self):
        """write_decision must NOT touch the metrics stream."""
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_decision(_decision(_snapshot()), "loop001")
        streams_written = [s for s, _ in fake.xadd_calls]
        assert "quantum:stream:exit.metrics" not in streams_written

    @pytest.mark.asyncio
    async def test_redis_error_is_swallowed(self):
        """A Redis failure after the guard must not propagate — agent stays alive."""
        aw = AuditWriter(
            _RaisingFake(),
            "quantum:stream:exit.audit",
            "quantum:stream:exit.metrics",
        )
        # Must NOT raise despite underlying ConnectionError
        await aw.write_decision(_decision(_snapshot()), "loop001")


# ── write_metrics path ────────────────────────────────────────────────────────


class TestWriteMetrics:
    @pytest.mark.asyncio
    async def test_write_goes_to_metrics_stream(self):
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_metrics("loopX", 5, 2, 3, 123.4, 0)
        assert len(fake.xadd_calls) == 1
        stream, _ = fake.xadd_calls[0]
        assert stream == "quantum:stream:exit.metrics"

    @pytest.mark.asyncio
    async def test_payload_fields(self):
        fake = _RecordingFake()
        aw = AuditWriter(fake, "quantum:stream:exit.audit", "quantum:stream:exit.metrics")
        await aw.write_metrics("loopX", 5, 2, 3, 123.4, 0)
        _, fields = fake.xadd_calls[0]
        assert fields["n_positions"] == "5"
        assert fields["n_actionable"] == "2"
        assert fields["n_hold"] == "3"
        assert fields["source"] == "exit_management_agent"
        assert fields["patch"] == "PATCH-1"

    @pytest.mark.asyncio
    async def test_redis_error_is_swallowed(self):
        """write_metrics must not raise on Redis failure — same swallow policy as write_decision."""
        aw = AuditWriter(
            _RaisingFake(),
            "quantum:stream:exit.audit",
            "quantum:stream:exit.metrics",
        )
        await aw.write_metrics("loopX", 0, 0, 0, 10.0, 0)  # must not raise
