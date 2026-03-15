"""intent_writer: publishes validated exit intents to exit.intent (PATCH-5A).

IntentWriter is the sole code path that may write to quantum:stream:exit.intent.
It is constructed with live_writes_enabled from AgentConfig; when that flag is
False the writer is a strict no-op and the stream is never touched.

Schema written to quantum:stream:exit.intent
--------------------------------------------
All field values are strings (Redis stream requirement).

  intent_id         — hex UUID, unique per publish call
  symbol            — upper-case symbol, e.g. "BTCUSDT"
  action            — one of FULL_CLOSE | PARTIAL_CLOSE_25 | TIME_STOP_EXIT
  urgency           — LOW | MEDIUM | HIGH | EMERGENCY
  side              — LONG | SHORT (position side, not order side)
  qty_fraction      — "1.0" for full close, "0.25" for PARTIAL_CLOSE_25
  quantity          — total position quantity (float as string)
  entry_price       — position entry price (float as string)
  mark_price        — mark price at decision time (float as string)
  R_net             — risk-adjusted net return (float as string)
  confidence        — decision confidence 0.0–1.0 (float as string)
  reason            — human-readable decision reason text
  loop_id           — hex loop identifier from main tick
  source            — "exit_management_agent"
  patch             — "PATCH-5A"
  ts_epoch          — unix epoch at publish time (float as string)

Downstream consumer: exit_intent_gateway (PATCH-5B, not yet implemented).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.contracts.validation import validate_xadd

from .models import ExitDecision
from .redis_io import RedisClient
from .validator import ExitIntentValidator, ValidationResult

_log = logging.getLogger("exit_management_agent.intent_writer")

_INTENT_STREAM_DEFAULT = "quantum:stream:exit.intent"


class IntentWriter:
    """
    Publishes validated ExitDecision objects to the exit.intent stream.

    When live_writes_enabled=False (PATCH-1 default), every call to
    maybe_publish() is a no-op — nothing is written, not even the audit log.

    When live_writes_enabled=True (PATCH-5A active), maybe_publish() will:
      1. Run ExitIntentValidator.validate() — all 7 synchronous checks.
      2. If validation passes: serialise and XADD to intent_stream.
      3. If validation fails: log a warning and return False (no write).
    """

    def __init__(
        self,
        redis: RedisClient,
        live_writes_enabled: bool,
        intent_stream: str = _INTENT_STREAM_DEFAULT,
    ) -> None:
        self._redis = redis
        self._live_writes_enabled = live_writes_enabled
        self._intent_stream = intent_stream
        self._validator = ExitIntentValidator()

        if live_writes_enabled:
            _log.warning(
                "IntentWriter initialised with live_writes_enabled=True. "
                "Validated exits will be published to %s. "
                "PATCH-5B gateway is required before any order is executed.",
                intent_stream,
            )
        else:
            _log.debug(
                "IntentWriter initialised with live_writes_enabled=False "
                "(shadow-only, PATCH-1 behaviour preserved)."
            )

    async def maybe_publish(self, dec: ExitDecision, loop_id: str) -> bool:
        """
        Attempt to publish *dec* to exit.intent.

        Returns True  if the intent was published.
        Returns False if live_writes_enabled=False OR validation failed.

        Never raises — validation failures and write errors are logged and
        swallowed so that a single bad decision cannot break the tick loop.
        """
        if not self._live_writes_enabled:
            return False

        # Only actionable decisions can ever be published.
        if not dec.is_actionable:
            return False

        result: ValidationResult = self._validator.validate(dec)

        if result.failed:
            _log.warning(
                "INTENT_BLOCKED loop=%s symbol=%s action=%s rule=%s reason=%s",
                loop_id,
                dec.snapshot.symbol,
                dec.action,
                result.rule,
                result.reason,
            )
            return False

        fields = self._serialise(dec, loop_id)

        try:
            _v = validate_xadd("exit.intent", fields, _log)
            if _v is None:
                _log.error("INTENT_VALIDATION_BLOCKED symbol=%s", dec.snapshot.symbol)
                return False
            await self._redis.xadd(self._intent_stream, _v)
            _log.info(
                "INTENT_PUBLISHED loop=%s symbol=%s action=%s urgency=%s "
                "confidence=%.3f R_net=%.4f intent_id=%s",
                loop_id,
                dec.snapshot.symbol,
                dec.action,
                dec.urgency,
                dec.confidence,
                dec.R_net,
                fields["intent_id"],
            )
            return True
        except Exception as exc:
            _log.error(
                "INTENT_WRITE_FAILED loop=%s symbol=%s action=%s error=%s",
                loop_id,
                dec.snapshot.symbol,
                dec.action,
                exc,
                exc_info=True,
            )
            return False

    @staticmethod
    def _serialise(dec: ExitDecision, loop_id: str) -> dict:
        """
        Convert an ExitDecision into a flat dict of strings for Redis XADD.

        qty_fraction is set to:
          0.25  for PARTIAL_CLOSE_25 (or dec.suggested_qty_fraction if present)
          1.0   for FULL_CLOSE and TIME_STOP_EXIT
        """
        snap = dec.snapshot

        if dec.action == "PARTIAL_CLOSE_25":
            qty_fraction = (
                dec.suggested_qty_fraction
                if dec.suggested_qty_fraction is not None
                else 0.25
            )
        else:
            qty_fraction = (
                dec.suggested_qty_fraction
                if dec.suggested_qty_fraction is not None
                else 1.0
            )

        return {
            "intent_id": uuid.uuid4().hex,
            "symbol": snap.symbol,
            "action": dec.action,
            "urgency": dec.urgency,
            "side": snap.side,
            "qty_fraction": str(qty_fraction),
            "quantity": str(snap.quantity),
            "entry_price": str(snap.entry_price),
            "mark_price": str(snap.mark_price),
            "R_net": str(round(dec.R_net, 6)),
            "confidence": str(round(dec.confidence, 6)),
            "reason": dec.reason,
            "loop_id": loop_id,
            "source": "exit_management_agent",
            "patch": "PATCH-5A",
            "ts_epoch": str(time.time()),
        }
