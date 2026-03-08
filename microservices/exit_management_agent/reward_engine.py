"""reward_engine: compute reward, regret label, and preferred action from
closed-position data (PATCH-8C).

Design
------
Since actual PnL is not available at closure time (close_pnl_usdt is always
"null" in PATCH-8B), the reward formula uses two proxy signals:

    exit_score       — agent's composite urgency signal at decision time [0, 1]
    live_action      — last action the agent emitted before the position closed
    hold_duration_sec — elapsed time from snapshot capture to position closure
    closed_by        — "exit_management_agent" | "unknown"
    diverged         — formula and Qwen3 disagreed at decision time

Reward mapping (clipped to [-1.0, 1.0])
    FULL_CLOSE / PARTIAL_CLOSE_25 / TIME_STOP_EXIT
         →  exit_score  (high confidence close = positive)
         →  minus hold_penalty if close happened very early (premature)
    HOLD (position then closed by external force)
         → -exit_score  (high exit signal but held = missed the exit)
    TIME_STOP_EXIT → exit_score * 0.5 (time-chased exit; partial credit)
    UNKNOWN / other → 0.0 (no signal)

Regret labels (minimum set specified in PATCH-8C)
    late_hold         — HOLD + external close + exit_score >= 0.5
    premature_close   — EMA-initiated close with hold_duration < threshold
    divergence_regret — formula and Qwen3 action disagreed
    none              — no negative pattern detected

Scope: PATCH-8C.
NOT in scope: online learning, model weight updates, MAE/MFE tracking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger("exit_management_agent.reward_engine")

# Actions initiated by EMA that earn a reward based on exit confidence.
_EMA_CLOSE_ACTIONS: frozenset = frozenset(
    {"FULL_CLOSE", "PARTIAL_CLOSE_25", "TIME_STOP_EXIT"}
)

# Threshold below which a reward-neutral hold is considered correct (no regret).
_LATE_HOLD_EXIT_SCORE_MIN: float = 0.5

# Reward penalty applied when the agent closed prematurely (max at hold=0).
_PREMATURE_PENALTY_MAX: float = 0.4

# Reward multiplier for TIME_STOP_EXIT (mechanically correct but not always
# the optimal exit point; graded lower than a full/partial confidence exit).
_TIME_STOP_REWARD_FACTOR: float = 0.5


# ── Helper ─────────────────────────────────────────────────────────────────────

def _safe_float(raw, *, default: float = 0.0) -> float:
    """Parse a string or numeric value to float; return default on failure."""
    if raw is None or raw in ("null", "", "None"):
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


def _safe_int(raw, *, default: Optional[int] = None) -> Optional[int]:
    """Parse a string or numeric value to int; return default on failure."""
    if raw is None or raw in ("null", "", "None"):
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RewardResult:
    """
    Outcome of a single reward computation.

    reward          — scalar in [-1.0, 1.0]; higher = agent decision was better
    regret_label    — categorical label for the observed error pattern
    regret_score    — continuous severity of regret in [0.0, 1.0]
    preferred_action — action the agent should have taken (or did take correctly)
    """

    reward: float           # clipped to [-1.0, 1.0]
    regret_label: str       # "late_hold" | "premature_close" | "divergence_regret" | "none"
    regret_score: float     # [0.0, 1.0]
    preferred_action: str   # suggested corrective or validated action


# ── Engine ─────────────────────────────────────────────────────────────────────

class RewardEngine:
    """
    Pure synchronous reward computation — no Redis, no async, no side effects.

    Parameters
    ----------
    late_hold_threshold_sec
        Minimum hold duration (seconds) above which a HOLD + external close is
        classified as a "late_hold" with maximum regret.  Below this the regret
        is partial, scaling with exit_score.  Default: 3600 (1 hour).

    premature_close_threshold_sec
        Hold durations less than this value (seconds) after an EMA-initiated
        FULL_CLOSE or PARTIAL_CLOSE_25 are labelled "premature_close".
        Default: 300 (5 minutes).  Set to 0 to disable the check.
    """

    def __init__(
        self,
        late_hold_threshold_sec: int = 3600,
        premature_close_threshold_sec: int = 300,
    ) -> None:
        self._late_hold_threshold_sec = max(1, late_hold_threshold_sec)
        self._premature_close_threshold_sec = max(0, premature_close_threshold_sec)

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(self, snapshot: dict, outcome: dict) -> RewardResult:
        """
        Compute a RewardResult from a PATCH-8A snapshot dict and a PATCH-8B
        outcome event dict.

        Both dicts are read-only; missing / null fields fall back to safe defaults.
        Never raises.
        """
        try:
            return self._compute_safe(snapshot, outcome)
        except Exception as exc:
            _log.error("PATCH-8C: Unexpected error in RewardEngine.compute: %s", exc)
            return RewardResult(
                reward=0.0,
                regret_label="none",
                regret_score=0.0,
                preferred_action="HOLD",
            )

    # ── Private implementation ─────────────────────────────────────────────────

    def _compute_safe(self, snapshot: dict, outcome: dict) -> RewardResult:
        # ── Parse scalar inputs ───────────────────────────────────────────────
        exit_score = _safe_float(snapshot.get("exit_score"), default=0.5)
        exit_score = max(0.0, min(1.0, exit_score))  # clamp to [0, 1]

        # live_action is stored in snapshot; outcome_action mirrors it.
        live_action = (
            snapshot.get("live_action")
            or outcome.get("outcome_action")
            or "UNKNOWN"
        ).upper().strip()

        closed_by = outcome.get("closed_by", "unknown") or "unknown"

        hold_duration_sec: Optional[int] = _safe_int(
            outcome.get("hold_duration_sec"), default=None
        )

        diverged = (snapshot.get("diverged") or "false").lower() == "true"

        # ── Reward ────────────────────────────────────────────────────────────
        raw_reward = self._raw_reward(live_action, exit_score, hold_duration_sec)
        reward = max(-1.0, min(1.0, raw_reward))

        # ── Regret ────────────────────────────────────────────────────────────
        regret_label, regret_score = self._classify_regret(
            live_action=live_action,
            exit_score=exit_score,
            hold_duration_sec=hold_duration_sec,
            closed_by=closed_by,
            diverged=diverged,
        )

        # ── Preferred action ──────────────────────────────────────────────────
        preferred_action = self._preferred_action(live_action, reward, exit_score)

        return RewardResult(
            reward=round(reward, 6),
            regret_label=regret_label,
            regret_score=round(regret_score, 4),
            preferred_action=preferred_action,
        )

    def _raw_reward(
        self,
        live_action: str,
        exit_score: float,
        hold_duration_sec: Optional[int],
    ) -> float:
        if live_action == "TIME_STOP_EXIT":
            # Time-triggered close: mechanically correct but less informative
            # about whether the agent *chose* the right moment.
            return exit_score * _TIME_STOP_REWARD_FACTOR

        if live_action in ("FULL_CLOSE", "PARTIAL_CLOSE_25"):
            reward = exit_score  # [0.0, 1.0]
            # Premature penalty: exit happened very quickly after entry.
            if (
                self._premature_close_threshold_sec > 0
                and hold_duration_sec is not None
                and hold_duration_sec < self._premature_close_threshold_sec
            ):
                # fraction in (0, 1): 0 = instantly closed, 1 = at threshold
                fraction = hold_duration_sec / self._premature_close_threshold_sec
                penalty = _PREMATURE_PENALTY_MAX * (1.0 - fraction)
                reward -= penalty
            return reward

        if live_action == "HOLD":
            # Agent held; position was closed by external force.
            # Higher exit_score means the agent ignored a strong exit signal → bad.
            return -(exit_score)

        return 0.0  # TIGHTEN_TRAIL, MOVE_TO_BREAKEVEN, UNKNOWN, etc.

    def _classify_regret(
        self,
        live_action: str,
        exit_score: float,
        hold_duration_sec: Optional[int],
        closed_by: str,
        diverged: bool,
    ) -> tuple:
        """Return (regret_label: str, regret_score: float)."""

        # ── premature_close: EMA closed very early ────────────────────────────
        if (
            live_action in ("FULL_CLOSE", "PARTIAL_CLOSE_25")
            and self._premature_close_threshold_sec > 0
            and hold_duration_sec is not None
            and hold_duration_sec < self._premature_close_threshold_sec
        ):
            raw = 1.0 - (hold_duration_sec / self._premature_close_threshold_sec)
            return "premature_close", round(max(0.0, min(1.0, raw)), 4)

        # ── late_hold: agent held a strong exit signal; external force closed ─
        if (
            live_action == "HOLD"
            and closed_by == "unknown"
            and exit_score >= _LATE_HOLD_EXIT_SCORE_MIN
        ):
            if (
                hold_duration_sec is not None
                and hold_duration_sec >= self._late_hold_threshold_sec
            ):
                # Scale: 2× threshold → regret_score = 1.0
                raw = hold_duration_sec / (self._late_hold_threshold_sec * 2.0)
                regret_score = round(min(1.0, raw), 4)
            else:
                # Below the hard threshold but still missed exit signal.
                regret_score = round(min(1.0, exit_score * 0.6), 4)
            return "late_hold", regret_score

        # ── divergence_regret: formula and Qwen3 disagreed ────────────────────
        if diverged:
            return "divergence_regret", 0.4

        return "none", 0.0

    def _preferred_action(
        self,
        live_action: str,
        reward: float,
        exit_score: float,
    ) -> str:
        """
        Infer the action the agent should have taken.

        If the agent held with large negative reward (missed a strong exit),
        suggest a close action based on exit_score magnitude.
        Otherwise the live_action was the correct (or best-available) choice.
        """
        if live_action == "HOLD" and reward < -0.3:
            # Strong exit signal was present but ignored — prescribe a close.
            if exit_score >= 0.7:
                return "FULL_CLOSE"
            return "PARTIAL_CLOSE_25"

        # Validated or neutral: stay with what was decided.
        return live_action if live_action and live_action != "UNKNOWN" else "HOLD"
