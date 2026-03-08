"""scoring_guards: Hard-stop guards that bypass the formula scoring engine (PATCH-7A).

These three conditions are extracted from decision_engine.py Rules 1–3.  When
any guard fires the resulting ExitDecision is returned directly to the caller
and ScoringEngine.score() is NOT called.

Rules
-----
Guard 1 — Drawdown stop  : R_net <= -1.5  → FULL_CLOSE EMERGENCY  confidence=0.95
Guard 2 — SL breach      : distance_to_sl_pct < 0  → FULL_CLOSE EMERGENCY  confidence=0.99
Guard 3 — Time stop      : age_sec >= max_hold_sec  → TIME_STOP_EXIT MEDIUM  confidence=0.80

All returned ExitDecision objects have score_state=None to make it explicit that
deterministic hard-stop rules — not formula scoring — fired.

The legacy DecisionEngine still contains its own copies of these rules.  During
shadow mode both may fire for the same tick; that is expected and both events
are audited.  When scoring_mode switches to "formula", the hard guards here
fully replace Rules 1–3 in the call path.
"""
from __future__ import annotations

import logging
from typing import Optional

from .models import ExitDecision, PerceptionResult, PositionSnapshot

_log = logging.getLogger("exit_management_agent.scoring_guards")

_DRAWDOWN_STOP_R: float = -1.5   # must match decision_engine._DRAWDOWN_STOP_R


class HardGuards:
    """Stateless hard-stop guard battery."""

    @staticmethod
    def evaluate(
        p: PerceptionResult,
        max_hold_sec: float,
        dry_run: bool,
    ) -> Optional[ExitDecision]:
        """
        Evaluate all three hard guards in priority order.

        Returns an ExitDecision when any guard fires, or None if the position
        should proceed to normal scoring.

        Never raises — errors are logged and None is returned so the scoring
        path continues safely.
        """
        try:
            # Guard 1: drawdown stop
            if p.R_net <= _DRAWDOWN_STOP_R:
                _log.warning(
                    "[HardGuard-1-Drawdown] R_net=%.4f <= %.4f symbol=%s side=%s dry_run=%s",
                    p.R_net, _DRAWDOWN_STOP_R, p.snapshot.symbol, p.snapshot.side, dry_run,
                )
                return ExitDecision(
                    snapshot=p.snapshot,
                    action="FULL_CLOSE",
                    reason=f"[HardGuard] Drawdown stop: R_net={p.R_net:.4f} <= {_DRAWDOWN_STOP_R}",
                    urgency="EMERGENCY",
                    R_net=p.R_net,
                    confidence=0.95,
                    suggested_sl=None,
                    suggested_qty_fraction=None,
                    dry_run=dry_run,
                    score_state=None,
                )

            # Guard 2: SL breach (price has crossed stop-loss)
            if p.snapshot.stop_loss > 0.0 and p.distance_to_sl_pct < 0.0:
                _log.warning(
                    "[HardGuard-2-SLBreach] dist_sl=%.4f symbol=%s side=%s dry_run=%s",
                    p.distance_to_sl_pct, p.snapshot.symbol, p.snapshot.side, dry_run,
                )
                return ExitDecision(
                    snapshot=p.snapshot,
                    action="FULL_CLOSE",
                    reason=f"[HardGuard] SL breach: dist_to_sl={p.distance_to_sl_pct:.4f} < 0",
                    urgency="EMERGENCY",
                    R_net=p.R_net,
                    confidence=0.99,
                    suggested_sl=None,
                    suggested_qty_fraction=None,
                    dry_run=dry_run,
                    score_state=None,
                )

            # Guard 3: time stop (max hold exceeded)
            if p.age_sec >= max_hold_sec:
                _log.info(
                    "[HardGuard-3-TimeStop] age_sec=%.1f >= max_hold_sec=%.1f symbol=%s dry_run=%s",
                    p.age_sec, max_hold_sec, p.snapshot.symbol, dry_run,
                )
                return ExitDecision(
                    snapshot=p.snapshot,
                    action="TIME_STOP_EXIT",
                    reason=f"[HardGuard] Time stop: age={p.age_sec:.1f}s >= max={max_hold_sec:.1f}s",
                    urgency="MEDIUM",
                    R_net=p.R_net,
                    confidence=0.80,
                    suggested_sl=None,
                    suggested_qty_fraction=None,
                    dry_run=dry_run,
                    score_state=None,
                )

        except Exception:  # pylint: disable=broad-except
            _log.exception("[HardGuards] Unexpected error; skipping guards")

        # All guards passed — proceed to scoring
        return None
