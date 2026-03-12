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
from typing import Optional

from .audit import AuditWriter
from .config import AgentConfig
from .decision_engine import DecisionEngine
from .ensemble_bridge import EnsembleBridge, ENSEMBLE_QTY_MAP
from .heartbeat import HeartbeatWriter
from .intent_writer import IntentWriter
from .logging_utils import setup_logging
from .models import ExitDecision
from .outcome_tracker import OutcomeTracker
from .ownership_flag import OwnershipFlagWriter
from .perception import PerceptionEngine
from .position_source import PositionSource
from .redis_io import RedisClient
from .replay_writer import ReplayWriter
from .reward_engine import RewardEngine
from .qwen3_layer import Qwen3Layer
from .scoring_engine import FORMULA_QTY_MAP, ScoringEngine
from .scoring_guards import HardGuards
# PATCH-11: LLM judge layer.
from .llm.groq_client import GroqModelClient
from .llm.judge_orchestrator import JudgeOrchestrator
from .patch11_actions import PATCH11_QTY_MAP

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
        # Ensemble bridge: wraps Exit Brain v1 pipeline (6 ML models + policy).
        # Instantiated always but only called when scoring_mode="ensemble".
        self._ensemble: Optional[EnsembleBridge] = None
        if config.scoring_mode == "ensemble":
            self._ensemble = EnsembleBridge(config.redis_host, config.redis_port)
        # PATCH-11: LLM judge orchestrator — only when mode != "off" and ensemble active.
        self._judge: Optional[JudgeOrchestrator] = None
        if config.patch11_mode != "off" and self._ensemble is not None:
            _primary_client = GroqModelClient(
                model=config.groq_primary_model,
                endpoint=config.groq_endpoint,
                api_key=config.groq_api_key,
                timeout_ms=config.groq_primary_timeout_ms,
                min_interval_sec=config.groq_primary_min_interval_sec,
            )
            _fallback_client = GroqModelClient(
                model=config.groq_fallback_model,
                endpoint=config.groq_endpoint,
                api_key=config.groq_api_key,
                timeout_ms=config.groq_fallback_timeout_ms,
                min_interval_sec=config.groq_fallback_min_interval_sec,
            )
            self._judge = JudgeOrchestrator(
                primary=_primary_client,
                fallback=_fallback_client,
                confidence_threshold=config.groq_confidence_threshold,
                conflict_threshold=config.groq_conflict_threshold,
                large_position_usdt=config.groq_large_position_usdt,
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
        # PATCH-8B: outcome tracker — detects position closures and writes outcome
        # events to quantum:stream:exit.outcomes.  No-op when disabled.
        # PATCH-8C: inject RewardEngine + ReplayWriter for learning record generation.
        _reward_engine: Optional[RewardEngine] = None
        _replay_writer: Optional[ReplayWriter] = None
        if config.reward_engine_enabled:
            _reward_engine = RewardEngine(
                late_hold_threshold_sec=config.reward_late_hold_threshold_sec,
                premature_close_threshold_sec=config.reward_premature_close_threshold_sec,
            )
            _replay_writer = ReplayWriter(
                redis=self._redis,
                replay_stream=config.replay_stream,
                enabled=True,
            )
        self._outcome_tracker = OutcomeTracker(
            redis=self._redis,
            outcomes_stream=config.outcomes_stream,
            enabled=config.outcome_tracker_enabled,
            reward_engine=_reward_engine,
            replay_writer=_replay_writer,
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

        if self._cfg.scoring_mode == "ensemble":
            models_n = self._ensemble.n_models_loaded if self._ensemble else 0
            _log.warning(
                "EXIT_AGENT_START version=0.2.0 scoring_mode=ensemble "
                "models_loaded=%d dry_run=%s live_writes=%s loop_sec=%.1f "
                "audit_stream=%s metrics_stream=%s intent_stream=%s heartbeat_key=%s",
                models_n,
                self._cfg.dry_run,
                self._cfg.live_writes_enabled,
                self._cfg.loop_sec,
                self._cfg.audit_stream,
                self._cfg.metrics_stream,
                self._cfg.intent_stream,
                self._cfg.heartbeat_key,
            )
        elif self._cfg.scoring_mode == "ai":
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

        n_actionable = 0
        n_hold = 0
        errors = 0
        n_positions = 0

        # ── Ensemble mode: use Exit Brain v1 pipeline ────────────────────────
        if self._cfg.scoring_mode == "ensemble" and self._ensemble is not None:
            symbols = await self._ensemble.discover_positions()
            if allowlist:
                symbols = [s for s in symbols if s in allowlist]
            n_positions = len(symbols)

            llm_budget = self._cfg.patch11_max_llm_per_cycle
            llm_used = 0
            for symbol in symbols:
                try:
                    if self._cfg.patch11_mode != "off" and self._judge is not None and llm_used < llm_budget:
                        # PATCH-11: LLM judge pipeline on top of ensemble
                        pipeline_result = await self._ensemble.evaluate_for_judge(symbol)
                        if pipeline_result is None:
                            n_hold += 1
                            continue
                        bridge_result, pipeline_ctx = pipeline_result
                        judge_result = await self._judge.evaluate(ctx=pipeline_ctx)
                        llm_used += 1

                        if self._cfg.patch11_mode == "shadow":
                            # Ensemble drives live; LLM logged as audit-only
                            dec = ExitDecision(
                                snapshot=bridge_result.snap,
                                action=bridge_result.action,
                                reason=bridge_result.reason,
                                urgency=bridge_result.urgency,
                                R_net=bridge_result.snap.unrealized_pnl,
                                confidence=bridge_result.confidence,
                                suggested_sl=None,
                                suggested_qty_fraction=ENSEMBLE_QTY_MAP.get(bridge_result.action),
                                dry_run=self._cfg.dry_run,
                                score_state=None,
                            )
                            _log.info(
                                "PATCH11_SHADOW %s ensemble=%s llm=%s(%s) conf=%.2f",
                                symbol, bridge_result.action,
                                judge_result.action, judge_result.source,
                                judge_result.confidence,
                            )
                        elif self._cfg.patch11_mode == "hybrid":
                            # LLM influences soft actions only
                            _HYBRID_SOFT = frozenset({"HOLD", "DEFENSIVE_HOLD", "REDUCE_25"})
                            if judge_result.action in _HYBRID_SOFT:
                                dec = ExitDecision(
                                    snapshot=bridge_result.snap,
                                    action=judge_result.action,
                                    reason=f"LLM:{judge_result.source}",
                                    urgency=bridge_result.urgency,
                                    R_net=bridge_result.snap.unrealized_pnl,
                                    confidence=judge_result.confidence,
                                    suggested_sl=None,
                                    suggested_qty_fraction=PATCH11_QTY_MAP.get(judge_result.action),
                                    dry_run=self._cfg.dry_run,
                                    score_state=None,
                                )
                            else:
                                dec = ExitDecision(
                                    snapshot=bridge_result.snap,
                                    action=bridge_result.action,
                                    reason=bridge_result.reason,
                                    urgency=bridge_result.urgency,
                                    R_net=bridge_result.snap.unrealized_pnl,
                                    confidence=bridge_result.confidence,
                                    suggested_sl=None,
                                    suggested_qty_fraction=ENSEMBLE_QTY_MAP.get(bridge_result.action),
                                    dry_run=self._cfg.dry_run,
                                    score_state=None,
                                )
                        else:  # live
                            if bridge_result.urgency == "EMERGENCY":
                                _urgency = "EMERGENCY"
                            elif judge_result.action in ("FULL_CLOSE", "TOXICITY_UNWIND"):
                                _urgency = "HIGH"
                            elif judge_result.action in ("REDUCE_50", "HARVEST_70_KEEP_30"):
                                _urgency = "MEDIUM"
                            else:
                                _urgency = "LOW"
                            dec = ExitDecision(
                                snapshot=bridge_result.snap,
                                action=judge_result.action,
                                reason=f"LLM:{judge_result.source}",
                                urgency=_urgency,
                                R_net=bridge_result.snap.unrealized_pnl,
                                confidence=judge_result.confidence,
                                suggested_sl=None,
                                suggested_qty_fraction=judge_result.qty_fraction,
                                dry_run=self._cfg.dry_run,
                                score_state=None,
                            )
                    else:
                        result = await self._ensemble.evaluate(symbol)
                        if result is None:
                            n_hold += 1
                            continue

                        dec = ExitDecision(
                            snapshot=result.snap,
                            action=result.action,
                            reason=result.reason,
                            urgency=result.urgency,
                            R_net=result.snap.unrealized_pnl,
                            confidence=result.confidence,
                            suggested_sl=None,
                            suggested_qty_fraction=ENSEMBLE_QTY_MAP.get(result.action),
                            dry_run=self._cfg.dry_run,
                            score_state=None,
                        )

                    await self._audit.write_decision(dec, loop_id)

                    if self._cfg.live_writes_enabled and dec.is_actionable:
                        await self._intent_writer.maybe_publish(dec, loop_id)

                    if dec.is_actionable:
                        n_actionable += 1
                    else:
                        n_hold += 1

                except Exception as exc:
                    _log.error("Error evaluating %s: %s", symbol, exc, exc_info=True)
                    errors += 1

            active = set(symbols)

        # ── Standard modes: shadow / formula / ai ────────────────────────────
        else:
            positions = await self._position_source.get_open_positions(
                allowlist=allowlist,
            )
            n_positions = len(positions)

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
                            _skip_qwen3 = score_state.formula_action in (
                                "TIGHTEN_TRAIL", "MOVE_TO_BREAKEVEN"
                            )
                            if _skip_qwen3:
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
                                if self._cfg.qwen3_shadow or qr.fallback:
                                    live_action = score_state.formula_action
                                    live_confidence = score_state.formula_confidence
                                    live_reason = score_state.formula_reason
                                    live_urgency = score_state.formula_urgency
                                else:
                                    live_action = qr.action
                                    live_confidence = qr.confidence
                                    live_reason = qr.reason
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
                            # Shadow mode (default).
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

            # Prune first_observed for closed positions.
            active = {s.symbol for s in positions}
            stale = [s for s in list(self._first_observed) if s not in active]
            for sym in stale:
                del self._first_observed[sym]
                self._perception.forget(sym)

        # 4b. [PATCH-8B] Detect closed symbols and emit outcome events.
        await self._outcome_tracker.update(active)

        # 5. Write loop metrics.
        elapsed_ms = (time.monotonic() - tick_start) * 1000.0
        await self._audit.write_metrics(
            loop_id=loop_id,
            n_positions=n_positions,
            n_actionable=n_actionable,
            n_hold=n_hold,
            loop_ms=elapsed_ms,
            errors=errors,
        )

        _log.info(
            "TICK loop=%s positions=%d actionable=%d hold=%d errors=%d ms=%.0f",
            loop_id,
            n_positions,
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
