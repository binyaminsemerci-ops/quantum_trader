"""models: pure value objects for exit_intent_gateway (PATCH-5B).

No Redis I/O, no side effects.  Tests can construct all objects freely.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Action whitelist ─────────────────────────────────────────────────────────
# Only these actions from exit_management_agent may be forwarded by the gateway.
GATEWAY_ACTION_WHITELIST: frozenset = frozenset(
    {"FULL_CLOSE", "PARTIAL_CLOSE_25", "TIME_STOP_EXIT"}
)

# Required source field value in every exit.intent message.
EXPECTED_SOURCE: str = "exit_management_agent"

# Required patch field value in every exit.intent message.
EXPECTED_PATCH: str = "PATCH-5A"


@dataclass(frozen=True)
class IntentMessage:
    """
    Parsed representation of a single Redis stream entry from exit.intent.

    All fields arrive as strings from Redis XREADGROUP; this class
    coerces them into their native types and exposes them for validation
    and serialisation.

    Raises ValueError if mandatory fields are missing or unparsable.
    """

    # Stream metadata (not from the message body).
    stream_id: str  # Redis stream entry ID, e.g. "1234567890000-0"

    # Message body fields — all sourced from exit.intent stream entries.
    intent_id: str
    symbol: str
    action: str       # e.g. FULL_CLOSE
    urgency: str      # MEDIUM | HIGH | EMERGENCY
    side: str         # LONG | SHORT  (position side — NOT order side)
    qty_fraction: float
    quantity: float
    entry_price: float
    mark_price: float
    confidence: float
    reason: str
    loop_id: str
    source: str       # should be "exit_management_agent"
    patch: str        # should be "PATCH-5A"
    ts_epoch: float   # unix epoch when the intent was emitted

    # Optional fields.
    R_net: float = 0.0

    @classmethod
    def from_redis_fields(cls, stream_id: str, fields: dict) -> "IntentMessage":
        """
        Parse raw Redis stream fields (dict of str→str) into IntentMessage.

        Raises ValueError if any mandatory numeric field fails to parse.
        """
        _req = cls._require
        return cls(
            stream_id=stream_id,
            intent_id=_req(fields, "intent_id"),
            symbol=_req(fields, "symbol").upper(),
            action=_req(fields, "action").upper(),
            urgency=_req(fields, "urgency").upper(),
            side=_req(fields, "side").upper(),
            qty_fraction=float(_req(fields, "qty_fraction")),
            quantity=float(_req(fields, "quantity")),
            entry_price=float(_req(fields, "entry_price")),
            mark_price=float(_req(fields, "mark_price")),
            confidence=float(_req(fields, "confidence")),
            reason=fields.get("reason", ""),
            loop_id=fields.get("loop_id", ""),
            source=fields.get("source", ""),
            patch=fields.get("patch", ""),
            ts_epoch=float(_req(fields, "ts_epoch")),
            R_net=float(fields.get("R_net", "0") or "0"),
        )

    @staticmethod
    def _require(fields: dict, key: str) -> str:
        val = fields.get(key)
        if val is None or str(val).strip() == "":
            raise ValueError(f"IntentMessage: required field '{key}' is missing or empty")
        return str(val).strip()

    @property
    def order_side(self) -> str:
        """
        Map position side to order side.
        Closing a LONG position requires a SELL order.
        Closing a SHORT position requires a BUY order.
        """
        if self.side == "LONG":
            return "SELL"
        if self.side == "SHORT":
            return "BUY"
        raise ValueError(f"Unknown position side: {self.side!r}")

    @property
    def computed_qty(self) -> float:
        """
        Actual order quantity: qty_fraction × quantity.
        Clamped to 8 decimal places (Binance precision).
        """
        return round(self.qty_fraction * self.quantity, 8)

    def to_trade_intent_payload(self, patch: str = "PATCH-5B") -> str:
        """
        Serialise to the trade.intent payload JSON string expected by intent_bridge.

        Format: FORMAT1_QTY — direct qty float, BUY/SELL side, reduceOnly=true.
        """
        payload = {
            "symbol": self.symbol,
            "side": self.order_side,  # BUY or SELL
            "qty": self.computed_qty,
            "type": "MARKET",
            "reduceOnly": True,
            "source": EXPECTED_SOURCE,
            "patch": patch,
            "intent_id": self.intent_id,
            "confidence": self.confidence,
        }
        return json.dumps(payload)


@dataclass(frozen=True)
class GatewayValidationResult:
    """
    Result of a single gateway validation run against one IntentMessage.
    """

    passed: bool
    rule: str        # e.g. "V5_DEDUP" or "PASS"
    reason: str      # human-readable explanation
    intent_id: str
    symbol: str


@dataclass
class RateLimitState:
    """
    In-process rolling window rate-limit counter.
    Tracks how many intents were published in the last 60 seconds.
    """

    window_sec: float = 60.0
    _timestamps: list = field(default_factory=list)

    def check_and_record(self, limit: int) -> bool:
        """
        Return True if the publish is allowed (below limit), False if at/over limit.
        Always records the current timestamp if allowed.
        """
        now = time.monotonic()
        cutoff = now - self.window_sec
        self._timestamps = [t for t in self._timestamps if t >= cutoff]
        if len(self._timestamps) >= limit:
            return False
        self._timestamps.append(now)
        return True
