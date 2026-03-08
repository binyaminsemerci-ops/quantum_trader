"""outcome_tracker: detect closed positions and write outcome events (PATCH-8B).

On each agent tick, OutcomeTracker.update() is called with the current set
of active symbol names.  Symbols that were active on the *previous* tick but
are absent now are treated as "closed".

For each closed symbol the tracker:
  1. Reads pending decision_ids from quantum:set:exit.pending_decisions:{symbol}
  2. For each id, reads the snapshot hash at quantum:hash:exit.decision:{id}
  3. Writes one outcome event per id to quantum:stream:exit.outcomes
  4. Removes the processed id from the pending set (only on successful write)
  5. [PATCH-8C] Computes reward via RewardEngine and writes a replay record via
     ReplayWriter — best-effort; failure here never blocks outcome or cleanup.

Writes NEVER go to execution streams.  If any data is missing or a Redis
call fails, the best-effort outcome is written with null placeholders and
processing continues without raising.

PATCH-8B scope: detection + outcome event capture.
PATCH-8C scope: reward computation + replay record writing.
NOT in scope: online learning, MAE/MFE tracking.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .reward_engine import RewardEngine
    from .replay_writer import ReplayWriter

_log = logging.getLogger("exit_management_agent.outcome_tracker")

_OUTCOMES_STREAM_DEFAULT: str = "quantum:stream:exit.outcomes"
_PENDING_SET_PREFIX: str = "quantum:set:exit.pending_decisions:"
_SNAPSHOT_HASH_PREFIX: str = "quantum:hash:exit.decision:"


class OutcomeTracker:
    """
    Detect position closures and write outcome events tied to PATCH-8A
    decision snapshots.

    Usage (per-tick inside the agent loop):
        await tracker.update(active_symbols={s.symbol for s in positions})

    First call establishes the baseline — no outcomes generated (avoids false
    positives on agent startup when we cannot distinguish "just closed" from
    "was never open").
    """

    def __init__(
        self,
        redis,
        outcomes_stream: str = _OUTCOMES_STREAM_DEFAULT,
        enabled: bool = True,
        reward_engine: Optional["RewardEngine"] = None,  # PATCH-8C
        replay_writer: Optional["ReplayWriter"] = None,   # PATCH-8C
    ) -> None:
        self._redis = redis
        self._outcomes_stream = outcomes_stream
        self._enabled = enabled
        self._reward_engine = reward_engine
        self._replay_writer = replay_writer
        self._prev_symbols: set = set()
        self._initialized: bool = False

    async def update(self, active_symbols: set) -> None:
        """
        Compare active_symbols against the previous tick's snapshot.
        For every symbol that disappeared, generate outcome event(s).

        Returns immediately (no-op) when enabled=False.
        On the very first call, sets the baseline and returns.
        """
        if not self._enabled:
            return

        if not self._initialized:
            # First tick: store baseline; never generate false closure outcomes.
            self._prev_symbols = set(active_symbols)
            self._initialized = True
            return

        closed_symbols = self._prev_symbols - active_symbols
        for symbol in closed_symbols:
            await self._process_closed_symbol(symbol)

        self._prev_symbols = set(active_symbols)

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _process_closed_symbol(self, symbol: str) -> None:
        """Handle a symbol that disappeared from the active position list."""
        close_time = int(time.time())
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"

        try:
            decision_ids = await self._redis.smembers_pending_decisions(set_key)
        except Exception as exc:
            _log.error(
                "PATCH-8B: Failed to read pending decisions for %s: %s", symbol, exc
            )
            return

        if not decision_ids:
            _log.debug(
                "PATCH-8B: Symbol %s closed — no pending decisions found", symbol
            )
            return

        # Best-effort close price from ticker (position is already gone).
        close_price: Optional[str] = None
        try:
            price = await self._redis.get_mark_price_from_ticker(symbol)
            if price is not None:
                close_price = f"{price:.8f}"
        except Exception:
            pass  # Non-critical; outcome written with null if unavailable.

        _log.info(
            "PATCH-8B: Symbol %s closed — processing %d pending decision(s)",
            symbol,
            len(decision_ids),
        )

        for decision_id in decision_ids:
            await self._write_outcome_for_decision(
                symbol=symbol,
                decision_id=decision_id,
                close_time=close_time,
                close_price=close_price,
                set_key=set_key,
            )

    async def _write_outcome_for_decision(
        self,
        symbol: str,
        decision_id: str,
        close_time: int,
        close_price: Optional[str],
        set_key: str,
    ) -> None:
        """Write a single outcome event and clean up the pending set entry."""
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}{decision_id}"

        # Read snapshot hash — tolerate missing / expired entry.
        snapshot: dict = {}
        try:
            snapshot = await self._redis.hgetall_snapshot(hash_key) or {}
        except Exception as exc:
            _log.warning(
                "PATCH-8B: Could not read snapshot for id=%s: %s", decision_id, exc
            )
            snapshot = {}

        # ── Derive outcome fields (best-effort; null if data missing) ──────────

        ts_epoch_raw = snapshot.get("ts_epoch", "")
        hold_duration_sec = "null"
        if ts_epoch_raw:
            try:
                hold_duration_sec = str(close_time - int(ts_epoch_raw))
            except (ValueError, TypeError):
                pass

        live_action = snapshot.get("live_action", "UNKNOWN")

        # Infer who closed: if the last decision was actionable, attribute to EMA;
        # otherwise label as unknown (closed by someone else, e.g. stop-loss hit).
        if live_action in ("FULL_CLOSE", "PARTIAL_CLOSE_25", "TIME_STOP_EXIT"):
            closed_by = "exit_management_agent"
        else:
            closed_by = "unknown"

        event: dict = {
            "decision_id": decision_id,
            "symbol": symbol,
            "close_time_epoch": str(close_time),
            "hold_duration_sec": hold_duration_sec,
            "outcome_action": live_action,
            "close_price": close_price if close_price is not None else "null",
            # close_pnl_usdt is not derivable without the actual fill price.
            "close_pnl_usdt": "null",
            "closed_by": closed_by,
            # MAE/MFE tracking is not in scope for PATCH-8B.
            "mae_pct": "null",
            "mfe_pct": "null",
            # Pass-through position context from snapshot.
            "entry_price": snapshot.get("entry_price", "null"),
            "quantity": snapshot.get("quantity", "null"),
            "side": snapshot.get("side", "UNKNOWN"),
            "exit_score": snapshot.get("exit_score", "null"),
            "source": "exit_management_agent",
            "patch": "PATCH-8B",
        }

        try:
            await self._redis.xadd(self._outcomes_stream, event)
            _log.info(
                "PATCH-8B: OUTCOME decision_id=%s symbol=%s action=%s hold_sec=%s",
                decision_id,
                symbol,
                live_action,
                hold_duration_sec,
            )
        except Exception as exc:
            _log.error(
                "PATCH-8B: Failed to write outcome for %s id=%s: %s",
                symbol,
                decision_id,
                exc,
            )
            # Do NOT remove from pending set — outcome not committed.
            return

        # Only clean up from pending set after the outcome event is safely written.
        try:
            await self._redis.srem_pending_decision(set_key, decision_id)
        except Exception as exc:
            _log.warning(
                "PATCH-8B: Failed to remove id=%s from pending set for %s: %s",
                decision_id,
                symbol,
                exc,
            )

        # [PATCH-8C] Compute reward and write replay record (best-effort).
        # Failure here does NOT affect the already-committed outcome or srem.
        if self._reward_engine is not None and self._replay_writer is not None:
            try:
                result = self._reward_engine.compute(snapshot, event)
                await self._replay_writer.write(
                    decision_id=decision_id,
                    symbol=symbol,
                    snapshot=snapshot,
                    outcome=event,
                    result=result,
                )
            except Exception as exc:
                _log.error(
                    "PATCH-8C: Failed to compute/write replay for %s id=%s: %s",
                    symbol,
                    decision_id,
                    exc,
                )
