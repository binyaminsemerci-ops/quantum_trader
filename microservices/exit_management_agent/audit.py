"""audit: write shadow decisions and metrics to exit.audit / exit.metrics streams.

Safety contract
---------------
* write_decision() raises RuntimeError if dec.dry_run is False.
* AuditWriter.__init__() raises ValueError if either stream is in the
  categorically forbidden set (applied.plan, trade.intent, etc.).
* All writes are routed through RedisClient.xadd() which has its own
  independent allowlist guard.

This means three independent checks block any accidental write to an
execution stream:
  1. AuditWriter constructor validates stream names.
  2. write_decision() checks dry_run flag.
  3. RedisClient.xadd() enforces its own allowlist.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .models import ExitDecision
from .redis_io import RedisClient, _FORBIDDEN_STREAMS

_log = logging.getLogger("exit_management_agent.audit")


class AuditWriter:
    """
    Serialise ExitDecision and PerceptionResult metrics into Redis streams.

    Writes ONLY to:
      • quantum:stream:exit.audit    (per-decision records)
      • quantum:stream:exit.metrics  (per-loop aggregate counters)
    """

    def __init__(
        self,
        redis: RedisClient,
        audit_stream: str,
        metrics_stream: str,
    ) -> None:
        self._redis = redis
        self._audit_stream = audit_stream
        self._metrics_stream = metrics_stream
        self._validate_streams()

    def _validate_streams(self) -> None:
        from .redis_io import _ALLOWED_WRITE_STREAMS

        for stream in (self._audit_stream, self._metrics_stream):
            if stream in _FORBIDDEN_STREAMS:
                raise ValueError(
                    f"AuditWriter constructed with forbidden stream: {stream!r}"
                )
            if stream not in _ALLOWED_WRITE_STREAMS:
                raise ValueError(
                    f"AuditWriter stream not on allowlist: {stream!r}. "
                    f"Allowed: {sorted(_ALLOWED_WRITE_STREAMS)}"
                )

    async def write_decision(self, dec: ExitDecision, loop_id: str) -> None:
        """
        Write a single ExitDecision to the audit stream.

        Raises:
            RuntimeError: if dec.dry_run is False (PATCH-1 invariant).
        """
        if not dec.dry_run:
            raise RuntimeError(
                "[AUDIT_WRITE_GUARD] dry_run=False on ExitDecision — "
                "PATCH-1 is shadow-only. This is a code bug; see decision_engine.py."
            )

        snap = dec.snapshot
        fields: dict = {
            "loop_id": loop_id,
            "ts": str(int(time.time())),
            "symbol": snap.symbol,
            "side": snap.side,
            "action": dec.action,
            "urgency": dec.urgency,
            "reason": dec.reason,
            "R_net": f"{dec.R_net:.4f}",
            "confidence": f"{dec.confidence:.2f}",
            "mark_price": f"{snap.mark_price:.8f}",
            "entry_price": f"{snap.entry_price:.8f}",
            "quantity": f"{snap.quantity:.8f}",
            "leverage": f"{snap.leverage:.1f}",
            "stop_loss": f"{snap.stop_loss:.8f}",
            "take_profit": f"{snap.take_profit:.8f}",
            "unrealized_pnl": f"{snap.unrealized_pnl:.4f}",
            "entry_risk_usdt": f"{snap.entry_risk_usdt:.4f}",
            "dry_run": "true",
            "source": "exit_management_agent",
            "patch": "PATCH-7A",
        }

        if dec.suggested_sl is not None:
            fields["suggested_sl"] = f"{dec.suggested_sl:.8f}"
        if dec.suggested_qty_fraction is not None:
            fields["suggested_qty_fraction"] = f"{dec.suggested_qty_fraction:.4f}"

        # PATCH-7A: include formula scoring fields when ScoringEngine ran.
        # score_state is None for hard-guard decisions.
        if dec.score_state is not None:
            ss = dec.score_state
            fields["exit_score"] = f"{ss.exit_score:.4f}"
            fields["d_r_loss"] = f"{ss.d_r_loss:.4f}"
            fields["d_r_gain"] = f"{ss.d_r_gain:.4f}"
            fields["d_giveback"] = f"{ss.d_giveback:.4f}"
            fields["d_time"] = f"{ss.d_time:.4f}"
            fields["d_sl_proximity"] = f"{ss.d_sl_proximity:.4f}"
            fields["formula_action"] = ss.formula_action
            fields["formula_urgency"] = ss.formula_urgency
            fields["formula_confidence"] = f"{ss.formula_confidence:.4f}"
            fields["formula_reason"] = ss.formula_reason  # C-2 fix

        # PATCH-7B: write Qwen3 layer fields when model was called.
        # Present only in scoring_mode="ai" for non-skipped actions.
        if dec.qwen3_result is not None:
            qr = dec.qwen3_result
            fields["qwen3_action"] = qr.action
            fields["qwen3_confidence"] = f"{qr.confidence:.4f}"
            fields["qwen3_reason"] = qr.reason
            fields["qwen3_fallback"] = "true" if qr.fallback else "false"
            fields["qwen3_latency_ms"] = f"{qr.latency_ms:.1f}"
            fields["patch"] = "PATCH-7B"

        try:
            await self._redis.xadd(self._audit_stream, fields)
        except Exception as exc:
            _log.error("Failed to write audit for %s: %s", snap.symbol, exc)
            return

        if dec.is_actionable:
            _log.info(
                "AUDIT %-22s %-12s R=%+.2f urgency=%-9s loop=%s",
                dec.action,
                snap.symbol,
                dec.R_net,
                dec.urgency,
                loop_id,
            )
        else:
            _log.debug(
                "AUDIT HOLD %-12s R=%+.2f loop=%s",
                snap.symbol,
                dec.R_net,
                loop_id,
            )

    async def write_metrics(
        self,
        loop_id: str,
        n_positions: int,
        n_actionable: int,
        n_hold: int,
        loop_ms: float,
        errors: int,
    ) -> None:
        """Write per-loop aggregate metrics to the metrics stream."""
        fields: dict = {
            "loop_id": loop_id,
            "ts": str(int(time.time())),
            "n_positions": str(n_positions),
            "n_actionable": str(n_actionable),
            "n_hold": str(n_hold),
            "loop_ms": f"{loop_ms:.1f}",
            "errors": str(errors),
            "source": "exit_management_agent",
            "patch": "PATCH-1",
        }
        try:
            await self._redis.xadd(self._metrics_stream, fields)
        except Exception as exc:
            _log.error("Failed to write metrics: %s", exc)
