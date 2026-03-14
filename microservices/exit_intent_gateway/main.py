"""main: exit_intent_gateway entry point (PATCH-5B).

Gateway flow (per message)
--------------------------
1.  XREADGROUP from quantum:stream:exit.intent (consumer group: exit-intent-gateway)
2.  Parse raw stream fields into IntentMessage
3.  Run GatewayValidator (9 checks: testnet, enabled, lockdown, stale, dedup,
    cooldown, action, source, rate-limit)
4a. If PASS: XADD to quantum:stream:trade.intent + XACK + log INTENT_FORWARDED
4b. If FAIL: XADD to quantum:stream:exit.intent.rejected + XACK + log INTENT_REJECTED

Safety constraints
------------------
- Hard startup abort if TESTNET_MODE != "true"
- apply.plan is UNCONDITIONALLY FORBIDDEN (write-guard in redis_io)
- Service starts in inert standby when EXIT_GATEWAY_ENABLED=false;
  messages are consumed and ACK'd to keep the stream moving, but
  nothing is forwarded to trade.intent.
- SIGTERM / SIGINT → graceful shutdown after current message.

PATCH-5B: this gateway is the ONLY path between exit.intent and trade.intent.
AutonomousTrader, harvest patches, and the panic-close path are NOT modified.
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time

from .config import GatewayConfig
from .models import IntentMessage
from .redis_io import GatewayRedisClient
from .validator import GatewayValidator

_log = logging.getLogger("exit_intent_gateway.main")

BUILD_TAG = "PATCH-5B"


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


class ExitIntentGateway:
    """
    Consumes exit.intent, validates each message, and routes approved
    intents to trade.intent (testnet pipeline only).
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._cfg = config
        self._redis = GatewayRedisClient(config.redis_host, config.redis_port)
        self._validator = GatewayValidator(config, self._redis)
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Counters for metrics logging.
        self._total_received: int = 0
        self._total_forwarded: int = 0
        self._total_rejected: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Redis, validate startup preconditions, and run."""
        await self._redis.connect()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        _log.warning(
            "GATEWAY_START version=0.1.0 patch=%s "
            "testnet_mode=%s enabled=%s "
            "intent_stream=%s trade_stream=%s rejected_stream=%s "
            "group=%s consumer=%s stale_sec=%d dedup_ttl=%d cooldown_sec=%d",
            BUILD_TAG,
            self._cfg.testnet_mode,
            self._cfg.enabled,
            self._cfg.intent_stream,
            self._cfg.trade_stream,
            self._cfg.rejected_stream,
            self._cfg.group,
            self._cfg.consumer,
            self._cfg.stale_sec,
            self._cfg.dedup_ttl_sec,
            self._cfg.cooldown_sec,
        )

        # Create consumer group (idempotent via BusyGroupError swallow).
        await self._redis.ensure_consumer_group(
            self._cfg.intent_stream,
            self._cfg.group,
            start_id="$",
        )

        # Re-process any messages that were delivered but not ACK'd in a
        # previous run (e.g. Redis XADD to trade.intent failed mid-flight).
        await self._drain_pending()

        self._running = True
        try:
            await self._loop()
        finally:
            await self._stop()

    def _handle_shutdown(self) -> None:
        _log.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()

    async def _stop(self) -> None:
        _log.warning(
            "GATEWAY_STOP forwarded=%d rejected=%d total=%d",
            self._total_forwarded,
            self._total_rejected,
            self._total_received,
        )
        await self._redis.close()

    # ── Main loop ────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:  # pragma: no cover
                _log.error("Unhandled loop error: %s", exc, exc_info=True)
                await asyncio.sleep(1.0)

            # Check for shutdown between polls.
            if self._shutdown_event.is_set():
                break

    async def _poll_once(self) -> None:
        """
        Read up to 1 message from exit.intent and process it.
        Blocks for up to 2 seconds if no messages are available.
        """
        results = await self._redis.xreadgroup(
            group=self._cfg.group,
            consumer=self._cfg.consumer,
            stream=self._cfg.intent_stream,
            count=1,
            block_ms=2000,
        )
        if not results:
            return

        for _stream_name, messages in results:
            for msg_id, fields in messages:
                await self._process_message(msg_id, fields)

    # ── Message processing ───────────────────────────────────────────────────

    async def _process_message(self, msg_id: str, fields: dict) -> None:
        self._total_received += 1

        # Always ACK even if parse fails — malformed messages must not block.
        try:
            msg = IntentMessage.from_redis_fields(msg_id, fields)
        except (ValueError, KeyError) as exc:
            _log.error(
                "INTENT_PARSE_ERROR msg_id=%s error=%s fields=%r",
                msg_id,
                exc,
                fields,
            )
            await self._ack_safe(msg_id)
            self._total_rejected += 1
            await self._write_rejected_raw(msg_id, fields, rule="PARSE_ERROR", reason=str(exc))
            return

        result = await self._validator.validate(msg)

        if result.passed:
            await self._forward_intent(msg)
        else:
            await self._reject_intent(msg, result.rule, result.reason)

        # ACK after processing so intent is not re-delivered on restart.
        await self._ack_safe(msg_id)

    async def _drain_pending(self) -> None:
        """
        Re-process any PEL (pending entry list) messages left unACK'd from a
        previous run.  Reads with id="0" until the PEL is empty, running the
        full _process_message path (parse → validate → forward/reject → ACK)
        on each recovered message.
        """
        _log.info(
            "GATEWAY_PEL_DRAIN_START group=%s consumer=%s stream=%s",
            self._cfg.group,
            self._cfg.consumer,
            self._cfg.intent_stream,
        )
        drained = 0
        while True:
            results = await self._redis.xreadgroup(
                group=self._cfg.group,
                consumer=self._cfg.consumer,
                stream=self._cfg.intent_stream,
                count=10,
                pending=True,
            )
            if not results:
                break
            had_messages = False
            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    if not fields:  # already ACK'd by another consumer
                        continue
                    had_messages = True
                    await self._process_message(msg_id, fields)
                    drained += 1
            if not had_messages:
                break
        _log.info("GATEWAY_PEL_DRAIN_DONE count=%d", drained)

    async def _forward_intent(self, msg: IntentMessage) -> None:
        """Publish approved intent to harvest.intent stream (PATCH-5C)."""
        _ACTION_MAP = {
            "FULL_CLOSE": "CLOSE", "PARTIAL_CLOSE_25": "PARTIAL_CLOSE",
            "TIME_STOP_EXIT": "CLOSE", "REDUCE_25": "PARTIAL_CLOSE",
            "REDUCE_50": "PARTIAL_CLOSE", "HARVEST_70_KEEP_30": "PARTIAL_CLOSE",
            "TOXICITY_UNWIND": "CLOSE",
        }
        harvest_action = _ACTION_MAP.get(msg.action, "CLOSE")
        harvest_fields = {
            "symbol": msg.symbol,
            "action": harvest_action,
            "percentage": str(msg.qty_fraction),
            "reason": f"{msg.action} {msg.reason}",
            "R_net": str(msg.R_net),
            "pnl_usd": "0",
            "entry_price": str(msg.entry_price),
            "exit_price": str(msg.mark_price),
        }
        await self._redis.xadd(self._cfg.trade_stream, harvest_fields)
        _log.warning(
            "INTENT_FORWARDED intent_id=%s symbol=%s action=%s side=%s "
            "qty=%.8f confidence=%.4f patch=%s",
            msg.intent_id,
            msg.symbol,
            msg.action,
            msg.order_side,
            msg.computed_qty,
            msg.confidence,
            BUILD_TAG,
        )

    async def _reject_intent(self, msg: IntentMessage, rule: str, reason: str) -> None:
        """Write rejected intent to the audit stream and log."""
        reject_fields = {
            "intent_id": msg.intent_id,
            "symbol": msg.symbol,
            "action": msg.action,
            "rule": rule,
            "reason": reason,
            "source": msg.source,
            "patch": BUILD_TAG,
            "ts_epoch": str(time.time()),
        }
        try:
            await self._redis.xadd(self._cfg.rejected_stream, reject_fields)
        except Exception as exc:  # pragma: no cover
            _log.error("Failed to write to rejected stream: %s", exc)

        self._total_rejected += 1
        _log.warning(
            "INTENT_REJECTED intent_id=%s symbol=%s action=%s rule=%s reason=%r",
            msg.intent_id,
            msg.symbol,
            msg.action,
            rule,
            reason,
        )

    async def _write_rejected_raw(
        self, msg_id: str, fields: dict, rule: str, reason: str
    ) -> None:
        """Write a parse-error rejection to the audit stream (fields may be partial)."""
        reject_fields = {
            "intent_id": fields.get("intent_id", "UNKNOWN"),
            "symbol": fields.get("symbol", "UNKNOWN"),
            "action": fields.get("action", "UNKNOWN"),
            "rule": rule,
            "reason": reason,
            "source": fields.get("source", "UNKNOWN"),
            "patch": BUILD_TAG,
            "ts_epoch": str(time.time()),
        }
        try:
            await self._redis.xadd(self._cfg.rejected_stream, reject_fields)
        except Exception as exc:  # pragma: no cover
            _log.error("Failed to write raw rejection to rejected stream: %s", exc)

    async def _ack_safe(self, msg_id: str) -> None:
        try:
            await self._redis.xack(self._cfg.intent_stream, self._cfg.group, msg_id)
        except Exception as exc:  # pragma: no cover
            _log.error("Failed to XACK msg_id=%s: %s", msg_id, exc)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:  # pragma: no cover
    try:
        config = GatewayConfig.from_env()
    except RuntimeError as exc:
        # from_env already logged the GATEWAY_TESTNET_REQUIRED_ABORT message.
        sys.exit(f"GATEWAY_ABORT: {exc}")

    _setup_logging(config.log_level)
    gateway = ExitIntentGateway(config)
    asyncio.run(gateway.start())


if __name__ == "__main__":
    main()
