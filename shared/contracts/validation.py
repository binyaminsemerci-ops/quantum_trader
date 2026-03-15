"""Runtime validation helpers for Redis stream boundaries (OP 7D Phase 2).

Usage at XADD (write) sites:
    from shared.contracts.validation import validate_xadd
    fields = {"symbol": "BTCUSDT", "side": "BUY", ...}
    validated = validate_xadd("trade.intent", fields, logger)
    if validated is not None:
        redis.xadd(stream_key, validated)

Usage at XREAD (read) sites:
    from shared.contracts.validation import validate_xread
    event = validate_xread("apply.result", raw_fields, logger)
    # event is the Pydantic model if valid, None if invalid
    # Processing continues either way (fail-open on read side)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

from pydantic import ValidationError

from shared.contracts.base import StreamEvent
from shared.contracts.trade_intent import TradeIntentEvent
from shared.contracts.apply_plan import ApplyPlanEvent
from shared.contracts.apply_result import ApplyResultEvent
from shared.contracts.exit_intent import ExitIntentEvent
from shared.contracts.harvest_intent import HarvestIntentEvent
from shared.contracts.trade_closed import TradeClosedEvent

# Stream short-name → Pydantic contract
STREAM_CONTRACTS: Dict[str, Type[StreamEvent]] = {
    "trade.intent":   TradeIntentEvent,
    "apply.plan":     ApplyPlanEvent,
    "apply.result":   ApplyResultEvent,
    "exit.intent":    ExitIntentEvent,
    "harvest.intent": HarvestIntentEvent,
    "trade.closed":   TradeClosedEvent,
}

_fallback_logger = logging.getLogger("stream_validation")


def _decode_fields(raw: Dict) -> Dict[str, str]:
    """Decode bytes keys/values to str (Redis returns bytes by default)."""
    decoded = {}
    for k, v in raw.items():
        key = k.decode() if isinstance(k, bytes) else str(k)
        val = v.decode() if isinstance(v, bytes) else str(v)
        decoded[key] = val
    return decoded


def validate_xadd(
    stream_short: str,
    fields: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[str, str]]:
    """Validate fields BEFORE an XADD write.

    Args:
        stream_short: Short stream name, e.g. "trade.intent"
        fields: Dict to be written to the stream
        logger: Logger for validation errors

    Returns:
        Dict[str, str] ready for xadd if valid, None if invalid.
        On validation failure, logs the error with field details.
    """
    log = logger or _fallback_logger
    contract = STREAM_CONTRACTS.get(stream_short)
    if contract is None:
        # Unknown stream — pass through without validation
        return {k: str(v) for k, v in fields.items() if v is not None}

    decoded = _decode_fields(fields)

    try:
        event = contract.model_validate(decoded)
        return event.to_redis()
    except ValidationError as exc:
        log.error(
            "XADD_VALIDATION_FAIL stream=%s errors=%s fields=%s",
            stream_short,
            exc.error_count(),
            {k: v[:80] if isinstance(v, str) and len(v) > 80 else v
             for k, v in decoded.items()},
        )
        return None


def validate_xread(
    stream_short: str,
    fields: Dict,
    logger: Optional[logging.Logger] = None,
) -> Optional[StreamEvent]:
    """Validate fields AFTER an XREAD/XREADGROUP parse.

    Args:
        stream_short: Short stream name, e.g. "apply.result"
        fields: Raw dict from Redis (may have bytes keys/values)
        logger: Logger for validation warnings

    Returns:
        Validated Pydantic model if valid, None if invalid.
        FAIL-OPEN: Callers should continue processing even if None
        (using the raw dict as fallback). This prevents old messages
        from crashing readers.
    """
    log = logger or _fallback_logger
    contract = STREAM_CONTRACTS.get(stream_short)
    if contract is None:
        return None

    decoded = _decode_fields(fields)

    try:
        return contract.model_validate(decoded)
    except ValidationError as exc:
        log.warning(
            "XREAD_VALIDATION_WARN stream=%s errors=%s fields=%s",
            stream_short,
            exc.error_count(),
            {k: v[:80] if isinstance(v, str) and len(v) > 80 else v
             for k, v in decoded.items()},
        )
        return None
