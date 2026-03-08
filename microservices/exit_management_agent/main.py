"""main: exit_management_agent entry point (PATCH-1 / PATCH-5A / PATCH-6 / PATCH-7B).

Per-tick flow
-------------
0.  [PATCH-6] Write ownership flag (quantum:exit_agent:active_flag) — no-op
    when EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=false (default).
1.  Write heartbeat (quantum:exit_agent:heartbeat)
2.  SCAN quantum:position:*  →  parse PositionSnapshot list
3.  For each snapshot:
        a. compute PerceptionResult  (mark price, peak, R_net, giveback …)
        b. [PATCH-7A] HardGuards.evaluate() — emergency drawdown / SL breach /
           time stop bypass scoring and return immediately.
        c. If no guard fired:
               [PATCH-7A shadow] ScoringEngine.score() runs for audit data.
               DecisionEngine.decide() still drives the live action.
               score_state is attached to the ExitDecision for audit.
        c. write to quantum:stream:exit.audit
        d. [PATCH-5A] if live_writes_enabled and decision passes validation:
               publish to quantum:stream:exit.intent
4.  Write per-loop metrics to quantum:stream:exit.metrics
5.  Sleep(loop_sec); repeat

Writes NEVER go to: apply.plan, trade.intent, harvest.intent.
exit.intent is only written to when EXIT_AGENT_LIVE_WRITES_ENABLED=true.
active_flag is written when EXIT_AGENT_OWNERSHIP_TRANSFER_ENABLED=true
    AND EXIT_AGENT_TESTNET_MODE=true.

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
from .intent_writer import IntentWriter
from .logging_utils import setup_logging
from .models import ExitDecision
from .ownership_flag import OwnershipFlagWriter
from .perception import PerceptionEngine
from .position_source import PositionSource
from .redis_io import RedisClient
from .qwen3_layer import Qwen3Layer
from .scoring_engine import FORMULA_QTY_MAP, ScoringEngine
from .scoring_guards import HardGuards

_log = logging.getLogger("exit_management_agent.main")


class ExitManagementAgent:
    """
    Shadow-only exit evaluation loop.

    Instantiate via ExitManagementAgent(cfg) and call await agent.start().
    """

    def __init__(self, config: AgentConfig) -> None:
        self._cfg = config
        self._redis = RedisClient(
            config.redis_host,
            config.redis_port,
            live_writes_enabled=config.live_writes_enabled,
        )
        self._position_source = PositionSource(
            self._redis, config.max_positions_per_loop
        )
        self._perception = PerceptionEngine(self._redis)
        self._decision = DecisionEngine(max_hold_sec=config.max_hold_sec)
        # PATCH-7A: formula scoring engine — always instantiated; hot path
        # determined by config.scoring_mode (default "shadow").
        self._scoring_engine = ScoringEngine(max_hold_sec=config.max_hold_sec)
        # PATCH-7B: Qwen3 constrained decision layer — instantiated always;
        # only called when scoring_mode="ai" and formula_action is in _ALLOWED_ACTIONS.
        self._qwen3 = Qwen3Layer(
            endpoint=config.qwen3_endpoint,
            timeout_ms=config.qwen3_timeout_ms,
            shadow=config.qwen3_shadow,
            model=config.qwen3_model,
            api_key=config.qwen3_api_key,
            min_interval_sec=config.qwen3_min_interval_sec,
        )
        self._audit = AuditWriter(
            self._redis, config.audit_stream, config.metrics_stream,
            decision_ttl_sec=config.decision_ttl_sec,
        )
        self._heartbeat = HeartbeatWriter(
            self._redis, config.heartbeat_key, config.heartbeat_ttl_sec
        )
        # PATCH-5A: intent writer — no-op when live_writes_enabled=False.
        self._intent_writer = IntentWriter(
            self._redis,
            live_writes_enabled=config.live_writes_enabled,
            intent_stream=config.intent_stream,
        )
        # PATCH-6: ownership flag writer — no-op when ownership_transfer_enabled=False
        # or testnet_mode != "true".  When active, writes quantum:exit_agent:active_flag
        # each tick to keep the PATCH-2 kill-switch engaged in AutonomousTrader.
        self._ownership_flag = OwnershipFlagWriter(
            self._redis,
            enabled=config.ownership_transfer_enabled,
            flag_key=config.active_flag_key,
            ttl_sec=config.active_flag_ttl_sec,
            testnet_mode=config.testnet_mode,
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

        if self._cfg.scoring_mode == "ai":
            _log.warning(
                "EXIT_AGENT_START version=0.1.0 patch=PATCH-7B "
                "dry_run=%s live_writes=%s ownership_transfer=%s scoring_mode=%s "
                "qwen3_shadow=%s qwen3_endpoint=%s qwen3_model=%s loop_sec=%.1f "
                "audit_stream=%s metrics_stream=%s intent_stream=%s heartbeat_key=%s",
                self._cfg.dry_run,
                self._cfg.live_writes_enabled,
                self._cfg.ownership_transfer_enabled,
                self._cfg.scoring_mode,
                self._cfg.qwen3_shadow,
                self._cfg.qwen3_endpoint,
                self._cfg.qwen3_model,
                self._cfg.loop_sec,
                self._cfg.audit_stream,
                self._cfg.metrics_stream,
                self._cfg.intent_stream,
                self._cfg.heartbeat_key,
            )
        else:
            _log.warning(
                "EXIT_AGENT_START version=0.1.0 patch=PATCH-7A "
                "dry_run=%s live_writes=%s ownership_transfer=%s scoring_mode=%s loop_sec=%.1f "
                "audit_stream=%s metrics_stream=%s intent_stream=%s heartbeat_key=%s",
                self._cfg.dry_run,
                self._cfg.live_writes_enabled,
                self._cfg.ownership_transfer_enabled,
                self._cfg.scoring_mode,
                self._cfg.loop_sec,
                self._cfg.audit_stream,
                self._cfg.metrics_stream,
                self._cfg.intent_stream,
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

        # 0. [PATCH-6] Refresh exit-ownership flag (no-op when disabled).
        await self._ownership_flag.write()

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

                # b. [PATCH-7A] Hard guards first — bypass scoring on emergency.
                dec = HardGuards.evaluate(
                    p,
                    max_hold_sec=self._cfg.max_hold_sec,
                    dry_run=self._cfg.dry_run,
                )

                if dec is None:
                    # No hard guard fired — run scoring engine for audit data.
                    score_state = self._scoring_engine.score(p)

                    if self._cfg.scoring_mode == "ai":
                        # PATCH-7B: formula engine always runs first; Qwen3
                        # refines only within the 4 allowed actions.
                        # TIGHTEN_TRAIL and MOVE_TO_BREAKEVEN bypass Qwen3 —
                        # those actions require exact SL price computation.
                        _skip_qwen3 = score_state.formula_action in (
                            "TIGHTEN_TRAIL", "MOVE_TO_BREAKEVEN"
                        )
                        if _skip_qwen3:
                            # Use formula action directly; no model call.
                            dec = ExitDecision(
                                snapshot=snap,
                                action=score_state.formula_action,
                                reason=score_state.formula_reason,
                                urgency=score_state.formula_urgency,
                                R_net=p.R_net,
                                confidence=score_state.formula_confidence,
                                suggested_sl=None,
                                suggested_qty_fraction=FORMULA_QTY_MAP.get(
                                    score_state.formula_action
                                ),
                                dry_run=self._cfg.dry_run,
                                score_state=score_state,
                                qwen3_result=None,
                            )
                        else:
                            qr = await self._qwen3.evaluate(score_state)
                            # shadow=True → formula drives the live path;
                            # qwen3 output is audit-only.
                            # shadow=False (or fallback) → qwen3 action drives live.
                            # Fallback (qr.fallback=True) always uses formula regardless.
                            if self._cfg.qwen3_shadow or qr.fallback:
                                live_action = score_state.formula_action
                                live_confidence = score_state.formula_confidence
                                live_reason = score_state.formula_reason
                                live_urgency = score_state.formula_urgency
                            else:
                                live_action = qr.action
                                live_confidence = qr.confidence
                                live_reason = qr.reason
                                # urgency stays formula-derived (Qwen3 does not score urgency)
                                live_urgency = score_state.formula_urgency
                            dec = ExitDecision(
                                snapshot=snap,
                                action=live_action,
                                reason=live_reason,
                                urgency=live_urgency,
                                R_net=p.R_net,
                                confidence=live_confidence,
                                suggested_sl=None,
                                suggested_qty_fraction=FORMULA_QTY_MAP.get(live_action),
                                dry_run=self._cfg.dry_run,
                                score_state=score_state,
                                qwen3_result=qr,
                            )
                    elif self._cfg.scoring_mode == "formula":
                        # PATCH-7A formula mode: scoring engine drives live path.
                        # [C-1 fix] qty_fraction and suggested_sl are derived from
                        # FORMULA_QTY_MAP keyed on the formula action — they are
                        # NOT inherited from the legacy DecisionEngine.  This
                        # prevents FULL_CLOSE inheriting qty_fraction=0.25 from a
                        # legacy PARTIAL_CLOSE_25, or PARTIAL_CLOSE_25 inheriting
                        # qty_fraction=None from a legacy HOLD.
                        dec = ExitDecision(
                            snapshot=snap,
                            action=score_state.formula_action,
                            reason=score_state.formula_reason,
                            urgency=score_state.formula_urgency,
                            R_net=p.R_net,
                            confidence=score_state.formula_confidence,
                            suggested_sl=None,
                            suggested_qty_fraction=FORMULA_QTY_MAP.get(
                                score_state.formula_action
                            ),
                            dry_run=self._cfg.dry_run,
                            score_state=score_state,
                        )
                    else:
                        # Shadow mode (default): legacy decision drives live path;
                        # score_state is attached for audit comparison only.
                        dec = self._decision.decide(p, dry_run=self._cfg.dry_run)
                        dec.score_state = score_state

                # c. Audit (shadow write — always).
                await self._audit.write_decision(dec, loop_id)

                # d. [PATCH-5A] Publish to exit.intent if live writes enabled.
                if self._cfg.live_writes_enabled and dec.is_actionable:
                    await self._intent_writer.maybe_publish(dec, loop_id)

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
