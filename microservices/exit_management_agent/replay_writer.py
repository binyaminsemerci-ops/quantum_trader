"""replay_writer: write per-decision replay/training records to exit.replay
(PATCH-8C).

ReplayWriter is a thin async adapter that combines data from three sources:
  - PATCH-8A snapshot (agent decision context)
  - PATCH-8B outcome event (position closure facts)
  - PATCH-8C RewardResult (computed reward + regret label)

and writes a single, self-contained record to quantum:stream:exit.replay.

The stream is an append-only learning feed — it is NEVER read by the
execution path and has no effect on live trading decisions.

Failures
--------
A failed write is logged as ERROR and silently swallowed.  The replay stream
is best-effort; a missed record does not warrant crashing the agent loop.
"""
from __future__ import annotations

import logging
import time

from .reward_engine import RewardResult

_log = logging.getLogger("exit_management_agent.replay_writer")

_REPLAY_STREAM_DEFAULT: str = "quantum:stream:exit.replay"


class ReplayWriter:
    """
    Write one replay record per closed decision to quantum:stream:exit.replay.

    Usage:
        await writer.write(
            decision_id=..., symbol=...,
            snapshot=snapshot_dict,   # PATCH-8A keys
            outcome=outcome_dict,     # PATCH-8B keys
            result=reward_result,     # PATCH-8C RewardResult
        )

    When enabled=False every write call is a no-op (returns immediately).
    """

    def __init__(
        self,
        redis,
        replay_stream: str = _REPLAY_STREAM_DEFAULT,
        enabled: bool = True,
    ) -> None:
        self._redis = redis
        self._replay_stream = replay_stream
        self._enabled = enabled

    async def write(
        self,
        decision_id: str,
        symbol: str,
        snapshot: dict,
        outcome: dict,
        result: RewardResult,
    ) -> None:
        """Build and append a replay record.  Never raises."""
        if not self._enabled:
            return

        record = self._build_record(decision_id, symbol, snapshot, outcome, result)

        try:
            await self._redis.xadd(self._replay_stream, record)
            _log.info(
                "PATCH-8C: REPLAY decision_id=%s symbol=%s "
                "reward=%.4f regret=%s preferred=%s",
                decision_id,
                symbol,
                result.reward,
                result.regret_label,
                result.preferred_action,
            )
        except Exception as exc:
            _log.error(
                "PATCH-8C: Failed to write replay for %s id=%s: %s",
                symbol,
                decision_id,
                exc,
            )

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_record(
        decision_id: str,
        symbol: str,
        snapshot: dict,
        outcome: dict,
        result: RewardResult,
    ) -> dict:
        return {
            # Identity / metadata
            "decision_id": decision_id,
            "symbol": symbol,
            "record_time_epoch": str(int(time.time())),
            "patch": "PATCH-8C",
            "source": "exit_management_agent",
            # ── From PATCH-8A snapshot ──────────────────────────────────────
            "live_action": snapshot.get("live_action") or "UNKNOWN",
            "formula_action": snapshot.get("formula_action") or "null",
            "qwen3_action": snapshot.get("qwen3_action") or "null",
            "diverged": snapshot.get("diverged") or "false",
            "exit_score": snapshot.get("exit_score") or "null",
            "entry_price": snapshot.get("entry_price") or "null",
            "side": snapshot.get("side") or "UNKNOWN",
            "quantity": snapshot.get("quantity") or "null",
            # ── From PATCH-8B outcome ───────────────────────────────────────
            "hold_duration_sec": outcome.get("hold_duration_sec") or "null",
            "close_price": outcome.get("close_price") or "null",
            "closed_by": outcome.get("closed_by") or "unknown",
            "outcome_action": outcome.get("outcome_action") or "UNKNOWN",
            # ── Computed by PATCH-8C RewardEngine ──────────────────────────
            "reward": f"{result.reward:.6f}",
            "regret_label": result.regret_label,
            "regret_score": f"{result.regret_score:.4f}",
            "preferred_action": result.preferred_action,
        }
