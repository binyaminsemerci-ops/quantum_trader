"""main: exit_management_agent entry point (PATCH-1, shadow-only).

Per-tick flow
-------------
1.  Write heartbeat (quantum:exit_agent:heartbeat)
2.  SCAN quantum:position:*  →  parse PositionSnapshot list
3.  For each snapshot:
        a. compute PerceptionResult  (mark price, peak, R_net, giveback …)
        b. decide ExitDecision       (deterministic formula, no LLM)
        c. write to quantum:stream:exit.audit
4.  Write per-loop metrics to quantum:stream:exit.metrics
5.  Sleep(loop_sec); repeat

Writes NEVER go to: apply.plan, trade.intent, harvest.intent, exit.intent live.
active_flag is NOT written in PATCH-1.

Graceful shutdown
-----------------
SIGTERM / SIGINT → _running = False → exits after current tick finishes.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid

from .audit import AuditWriter
from .config import AgentConfig
from .decision_engine import DecisionEngine
from .heartbeat import HeartbeatWriter
from .logging_utils import setup_logging
from .perception import PerceptionEngine
from .position_source import PositionSource
from .redis_io import RedisClient

_log = logging.getLogger("exit_management_agent.main")


class ExitManagementAgent:
    """
    Shadow-only exit evaluation loop.

    Instantiate via ExitManagementAgent(cfg) and call await agent.start().
    """

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._redis = RedisClient(config.redis_host, config.redis_port)
        self._position_source = PositionSource(
            self._redis, config.max_positions_per_loop
        )
        self._perception = PerceptionEngine(self._redis)
        self._decision = DecisionEngine(max_hold_sec=config.max_hold_sec)
        self._audit = AuditWriter(
            self._redis, config.audit_stream, config.metrics_stream
        )
        self._heartbeat = HeartbeatWriter(
            self._redis, config.heartbeat_key, config.heartbeat_ttl_sec
        )
        # Per-symbol first-observation clock (lower-bound age tracker).
        # Cleared when a symbol disappears from active positions.
        self._first_observed: dict = {}
        self._running: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to Redis, install signal handlers, and run the main loop."""
        await self._redis.connect()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        _log.warning(
            "EXIT_AGENT_START version=0.1.0 patch=PATCH-1 "
            "dry_run=%s loop_sec=%.1f "
            "audit_stream=%s metrics_stream=%s heartbeat_key=%s",
            self._cfg.dry_run,
            self._cfg.loop_sec,
            self._cfg.audit_stream,
            self._cfg.metrics_stream,
            self._cfg.heartbeat_key,
        )

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
        _log.warning("EXIT_AGENT_STOP")
        await self._redis.close()

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            tick_start = time.monotonic()

            try:
                await self._tick()
            except Exception as exc:  # pragma: no cover
                _log.error("Unhandled tick error: %s", exc, exc_info=True)

            elapsed_sec = time.monotonic() - tick_start
            sleep_remaining = self._cfg.loop_sec - elapsed_sec

            if sleep_remaining > 0.0:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=sleep_remaining
                    )
                    # Shutdown event fired — exit loop.
                    break
                except asyncio.TimeoutError:
                    pass  # Normal path: keep running.

    async def _tick(self) -> None:
        loop_id = uuid.uuid4().hex[:12]
        tick_start = time.monotonic()

        # 1. Heartbeat.
        await self._heartbeat.beat()

        # 2. Fetch open positions.
        allowlist = self._cfg.symbol_allowlist if self._cfg.symbol_allowlist else None
        positions = await self._position_source.get_open_positions(allowlist=allowlist)

        n_actionable = 0
        n_hold = 0
        errors = 0

        # 3. Evaluate each position.
        for snap in positions:
            sym = snap.symbol
            try:
                # Track lower-bound age since first observation in this process.
                if sym not in self._first_observed:
                    self._first_observed[sym] = time.time()
                age_sec = time.time() - self._first_observed[sym]

                # a. Perceive.
                p = await self._perception.compute(snap, age_sec)

                # b. Decide.
                dec = self._decision.decide(p, dry_run=self._cfg.dry_run)

                # c. Audit (shadow write).
                await self._audit.write_decision(dec, loop_id)

                if dec.is_actionable:
                    n_actionable += 1
                else:
                    n_hold += 1

            except Exception as exc:
                _log.error("Error evaluating %s: %s", sym, exc, exc_info=True)
                errors += 1

        # 4. Prune first_observed for symbols no longer present.
        active = {s.symbol for s in positions}
        stale = [s for s in list(self._first_observed) if s not in active]
        for sym in stale:
            del self._first_observed[sym]
            self._perception.forget(sym)

        # 5. Write loop metrics.
        elapsed_ms = (time.monotonic() - tick_start) * 1000.0
        await self._audit.write_metrics(
            loop_id=loop_id,
            n_positions=len(positions),
            n_actionable=n_actionable,
            n_hold=n_hold,
            loop_ms=elapsed_ms,
            errors=errors,
        )

        _log.info(
            "TICK loop=%s positions=%d actionable=%d hold=%d errors=%d ms=%.0f",
            loop_id,
            len(positions),
            n_actionable,
            n_hold,
            errors,
            elapsed_ms,
        )


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    cfg = AgentConfig.from_env()
    setup_logging(cfg.log_level)

    if not cfg.enabled:
        logging.getLogger("exit_management_agent").warning(
            "EXIT_AGENT_ENABLED=false — agent disabled; exiting immediately"
        )
        return

    asyncio.run(ExitManagementAgent(cfg).start())


if __name__ == "__main__":
    main()
