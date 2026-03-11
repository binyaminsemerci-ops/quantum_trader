"""Tests for PATCH-8C: RewardEngine, ReplayWriter, and OutcomeTracker integration.

Test matrix
-----------
TestRewardEngineBasicRewards
  - FULL_CLOSE, high exit_score, long hold  → positive reward
  - FULL_CLOSE, low exit_score              → small positive reward
  - PARTIAL_CLOSE_25, mid exit_score        → positive reward
  - TIME_STOP_EXIT                          → partial reward (factor 0.5)
  - HOLD, high exit_score                   → strong negative reward
  - HOLD, low exit_score                    → mild negative reward
  - unknown action                          → reward 0.0
  - reward always clipped to [-1.0, 1.0]

TestPrematureClosePenalty
  - FULL_CLOSE with hold < threshold        → reduced reward
  - FULL_CLOSE with hold at threshold       → no penalty
  - PARTIAL_CLOSE_25 with hold < threshold  → reduced reward
  - threshold=0 disables premature check

TestRegretLabels
  - HOLD + external close + exit_score >= 0.5  → late_hold
  - HOLD + external close + high exit_score + very long hold
    → late_hold with regret_score > 0.5
  - HOLD + external close + exit_score < 0.5   → none (below threshold)
  - FULL_CLOSE, short hold                     → premature_close
  - premature_close takes priority over divergence_regret
  - diverged=true, no other regret             → divergence_regret
  - diverged=true, good FULL_CLOSE             → premature takes priority
  - nothing special                            → none

TestPreferredAction
  - HOLD + reward < -0.3, exit_score >= 0.7   → FULL_CLOSE
  - HOLD + reward < -0.3, exit_score < 0.7    → PARTIAL_CLOSE_25
  - HOLD + reward = -0.25 (mild)              → HOLD (reward >= -0.3)
  - FULL_CLOSE correct                        → FULL_CLOSE
  - PARTIAL_CLOSE_25 correct                  → PARTIAL_CLOSE_25
  - UNKNOWN action                            → HOLD (fallback)

TestMissingAndNullFields
  - empty snapshot + empty outcome            → no crash; defaults apply
  - exit_score = "null"                       → defaults to 0.5
  - hold_duration_sec = "null"                → no premature penalty applied
  - diverged = "" / missing                   → treated as False
  - exception inside _compute_safe            → returns zero RewardResult

TestReplayWriter
  - write() calls xadd with correct stream
  - record contains all required fields
  - reward/regret_score formatted as strings
  - enabled=False → no xadd
  - xadd raises → no crash (error logged silently)
  - snapshot + outcome fields pass-through correctly

TestOutcomeTrackerReplayIntegration
  - replay record written after outcome committed
  - replay NOT written if outcome xadd fails (return early)
  - replay write failure does not crash tracker (outcome already committed)
  - reward_engine=None, replay_writer=None → no replay, no crash
  - convergence: replay stream is different from outcomes stream
"""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.reward_engine import (
    RewardEngine,
    RewardResult,
    _safe_float,
    _safe_int,
)
from microservices.exit_management_agent.replay_writer import (
    ReplayWriter,
    _REPLAY_STREAM_DEFAULT,
)
from microservices.exit_management_agent.outcome_tracker import (
    OutcomeTracker,
    _OUTCOMES_STREAM_DEFAULT,
    _PENDING_SET_PREFIX,
    _SNAPSHOT_HASH_PREFIX,
)

_REPLAY = _REPLAY_STREAM_DEFAULT
_OUTCOMES = _OUTCOMES_STREAM_DEFAULT


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_engine(
    late_hold_threshold_sec: int = 3600,
    premature_close_threshold_sec: int = 300,
) -> RewardEngine:
    return RewardEngine(
        late_hold_threshold_sec=late_hold_threshold_sec,
        premature_close_threshold_sec=premature_close_threshold_sec,
    )


def make_snapshot(
    live_action: str = "HOLD",
    exit_score: str = "0.5000",
    formula_action: str = "HOLD",
    qwen3_action: str = "HOLD",
    diverged: str = "false",
    side: str = "LONG",
    entry_price: str = "30000.00000000",
    quantity: str = "0.01000000",
) -> dict:
    return {
        "live_action": live_action,
        "exit_score": exit_score,
        "formula_action": formula_action,
        "qwen3_action": qwen3_action,
        "diverged": diverged,
        "side": side,
        "entry_price": entry_price,
        "quantity": quantity,
    }


def make_outcome(
    hold_duration_sec: str = "3600",
    close_price: str = "31000.00000000",
    closed_by: str = "unknown",
    outcome_action: str = "HOLD",
) -> dict:
    return {
        "hold_duration_sec": hold_duration_sec,
        "close_price": close_price,
        "closed_by": closed_by,
        "outcome_action": outcome_action,
    }


# ── Fakes ──────────────────────────────────────────────────────────────────────

class _ReplayFake:
    """Minimal fake Redis client for ReplayWriter tests."""

    def __init__(self) -> None:
        self.xadd_calls: list = []  # [(stream, fields)]

    async def xadd(self, stream: str, fields: dict) -> None:
        self.xadd_calls.append((stream, dict(fields)))


class _XaddRaiseFake(_ReplayFake):
    async def xadd(self, stream: str, fields: dict) -> None:
        raise ConnectionError("Redis unavailable")


class _TrackerFake:
    """Fake for OutcomeTracker integration with PATCH-8C replay."""

    def __init__(
        self,
        smembers_results: dict | None = None,
        hgetall_results: dict | None = None,
        ticker_prices: dict | None = None,
        fail_on_stream: str | None = None,
    ) -> None:
        self.xadd_by_stream: dict = {}  # stream -> [fields]
        self.srem_calls: list = []
        self._smembers: dict = smembers_results or {}
        self._hgetall: dict = hgetall_results or {}
        self._tickers: dict = ticker_prices or {}
        self._fail_on_stream = fail_on_stream

    async def smembers_pending_decisions(self, key: str) -> set:
        return set(self._smembers.get(key, set()))

    async def hgetall_snapshot(self, key: str) -> dict:
        return dict(self._hgetall.get(key, {}))

    async def get_mark_price_from_ticker(self, symbol: str):
        return self._tickers.get(symbol)

    async def xadd(self, stream: str, fields: dict) -> None:
        if self._fail_on_stream and stream == self._fail_on_stream:
            raise ConnectionError(f"forced failure on {stream}")
        self.xadd_by_stream.setdefault(stream, []).append(dict(fields))

    async def srem_pending_decision(self, key: str, decision_id: str) -> None:
        self.srem_calls.append((key, decision_id))


def _make_full_snapshot(
    symbol: str = "BTCUSDT",
    decision_id: str = "dec-001",
    ts_epoch: int = 1_000_000,
    live_action: str = "HOLD",
    exit_score: str = "0.6000",
    diverged: str = "false",
) -> dict:
    return {
        "decision_id": decision_id,
        "ts_epoch": str(ts_epoch),
        "symbol": symbol,
        "side": "LONG",
        "entry_price": "30000.00000000",
        "mark_price": "31500.00000000",
        "quantity": "0.01000000",
        "formula_action": live_action,
        "formula_conf": "0.7500",
        "qwen3_action": "",
        "qwen3_conf": "0.0000",
        "live_action": live_action,
        "live_conf": "0.7500",
        "diverged": diverged,
        "exit_score": exit_score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TestRewardEngineBasicRewards
# ═══════════════════════════════════════════════════════════════════════════════

class TestRewardEngineBasicRewards:
    def test_full_close_high_exit_score(self):
        eng = make_engine()
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent",
                           outcome_action="FULL_CLOSE")
        r = eng.compute(snap, out)
        assert r.reward == pytest.approx(0.8, abs=1e-5)

    def test_full_close_low_exit_score(self):
        eng = make_engine()
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.2000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert 0.0 < r.reward < 0.5

    def test_partial_close_mid_exit_score(self):
        eng = make_engine()
        snap = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.5000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.reward > 0.0

    def test_time_stop_exit_reward(self):
        """TIME_STOP_EXIT earns exit_score * 0.5."""
        eng = make_engine()
        snap = make_snapshot(live_action="TIME_STOP_EXIT", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="14400", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.reward == pytest.approx(0.8 * 0.5, abs=1e-5)

    def test_hold_high_exit_score_negative_reward(self):
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="0.9000")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward == pytest.approx(-0.9, abs=1e-5)

    def test_hold_low_exit_score_mild_negative(self):
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="0.1000")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward == pytest.approx(-0.1, abs=1e-5)

    def test_unknown_action_reward_zero(self):
        eng = make_engine()
        snap = make_snapshot(live_action="TIGHTEN_TRAIL", exit_score="0.5000")
        out = make_outcome(closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward == pytest.approx(0.0, abs=1e-5)

    def test_reward_clipped_upper(self):
        """exit_score=1.0 should stay at 1.0, not exceed it."""
        eng = make_engine()
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="1.0000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.reward <= 1.0

    def test_reward_clipped_lower(self):
        """exit_score=1.0 with HOLD should give -1.0, not below."""
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="1.0000")
        out = make_outcome(closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward >= -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestPrematureClosePenalty
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrematureClosePenalty:
    def test_full_close_short_hold_penalty_applied(self):
        """Hold at 50% of threshold → penalty reduces reward."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        # 150s = 50% of 300s threshold
        out = make_outcome(hold_duration_sec="150", closed_by="exit_management_agent")
        r_short = eng.compute(snap, out)

        out_long = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r_long = eng.compute(snap, out_long)

        assert r_short.reward < r_long.reward

    def test_full_close_at_threshold_no_penalty(self):
        """Hold exactly at threshold → no penalty applied."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="300", closed_by="exit_management_agent")
        r_at = eng.compute(snap, out)

        out_long = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r_long = eng.compute(snap, out_long)

        assert r_at.reward == pytest.approx(r_long.reward, abs=1e-5)

    def test_partial_close_short_hold_penalty(self):
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.6000")
        out_short = make_outcome(hold_duration_sec="60", closed_by="exit_management_agent")
        out_long = make_outcome(hold_duration_sec="3600", closed_by="exit_management_agent")
        r_short = eng.compute(snap, out_short)
        r_long = eng.compute(snap, out_long)
        assert r_short.reward < r_long.reward

    def test_threshold_zero_disables_premature_penalty(self):
        """Setting threshold=0 must not raise and must not apply penalty."""
        eng = make_engine(premature_close_threshold_sec=0)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="1", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        # No penalty: reward should equal exit_score
        assert r.reward == pytest.approx(0.8, abs=1e-5)

    def test_time_stop_exit_never_premature_penalized(self):
        """TIME_STOP_EXIT uses its own factor; premature check does not apply."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="TIME_STOP_EXIT", exit_score="0.8000")
        out_short = make_outcome(hold_duration_sec="1", closed_by="exit_management_agent")
        out_long = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r_short = eng.compute(snap, out_short)
        r_long = eng.compute(snap, out_long)
        # Both receive exit_score * 0.5; hold duration does not affect TIME_STOP
        assert r_short.reward == pytest.approx(r_long.reward, abs=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# TestRegretLabels
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegretLabels:
    def test_late_hold_basic(self):
        eng = make_engine(late_hold_threshold_sec=3600)
        snap = make_snapshot(live_action="HOLD", exit_score="0.7000")
        out = make_outcome(hold_duration_sec="3600", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.regret_label == "late_hold"

    def test_late_hold_long_duration_high_regret_score(self):
        """Hold >> threshold → regret_score approaches 1.0."""
        eng = make_engine(late_hold_threshold_sec=3600)
        snap = make_snapshot(live_action="HOLD", exit_score="0.8000")
        # 2× threshold = max regret_score (1.0)
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.regret_label == "late_hold"
        assert r.regret_score >= 0.9

    def test_late_hold_short_duration_partial_regret(self):
        """Hold < threshold but exit_score high → late_hold with partial regret."""
        eng = make_engine(late_hold_threshold_sec=3600)
        snap = make_snapshot(live_action="HOLD", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="600", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.regret_label == "late_hold"
        assert 0.0 < r.regret_score < 1.0

    def test_hold_low_exit_score_no_late_hold(self):
        """exit_score < 0.5 with HOLD → no late_hold regret."""
        eng = make_engine(late_hold_threshold_sec=3600)
        snap = make_snapshot(live_action="HOLD", exit_score="0.3000")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.regret_label == "none"

    def test_premature_close_label(self):
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="60", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "premature_close"

    def test_premature_close_regret_score_scales_with_hold(self):
        """Shorter hold = higher regret_score for premature_close."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        r_very_short = eng.compute(snap, make_outcome(hold_duration_sec="10"))
        r_medium = eng.compute(snap, make_outcome(hold_duration_sec="200"))
        assert r_very_short.regret_score > r_medium.regret_score

    def test_premature_close_priority_over_divergence(self):
        """premature_close is detected before divergence_regret."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000",
                             diverged="true")
        out = make_outcome(hold_duration_sec="30", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "premature_close"

    def test_divergence_regret(self):
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="HOLD", exit_score="0.3000", diverged="true")
        # exit_score < 0.5 → no late_hold; hold > threshold → no premature
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "divergence_regret"
        assert r.regret_score == pytest.approx(0.4, abs=1e-5)

    def test_no_regret(self):
        eng = make_engine()
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "none"
        assert r.regret_score == pytest.approx(0.0, abs=1e-5)

    def test_none_regret_score_zero(self):
        eng = make_engine()
        snap = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.6000")
        out = make_outcome(hold_duration_sec="1800", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "none"
        assert r.regret_score == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreferredAction
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreferredAction:
    def test_hold_strong_negative_high_exit_score_prefers_full_close(self):
        """HOLD + reward < -0.3 + exit_score >= 0.7 → FULL_CLOSE."""
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="0.9000")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward < -0.3
        assert r.preferred_action == "FULL_CLOSE"

    def test_hold_strong_negative_low_exit_score_prefers_partial_close(self):
        """HOLD + reward < -0.3 + exit_score < 0.7 → PARTIAL_CLOSE_25."""
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="0.5500")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward < -0.3
        assert r.preferred_action == "PARTIAL_CLOSE_25"

    def test_hold_mild_negative_prefers_hold(self):
        """reward = -0.1 (>= -0.3) → preferred_action stays HOLD."""
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="0.1000")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        # reward = -0.1 (>= -0.3 by a margin)
        assert r.reward > -0.3
        assert r.preferred_action == "HOLD"

    def test_full_close_preferred_action_is_full_close(self):
        eng = make_engine()
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.preferred_action == "FULL_CLOSE"

    def test_partial_close_preferred_action_is_partial_close(self):
        eng = make_engine()
        snap = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.5000")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.preferred_action == "PARTIAL_CLOSE_25"

    def test_unknown_action_preferred_is_hold(self):
        eng = make_engine()
        snap = make_snapshot(live_action="UNKNOWN", exit_score="0.5000")
        out = make_outcome(closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.preferred_action == "HOLD"


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreferredActionCounterfactual  (PATCH-10A)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreferredActionCounterfactual:
    """Rule 1: diverged+bad → formula_action; Rule 2: premature_close → HOLD."""

    # ── Rule 1: diverged + negative reward ────────────────────────────────────

    def test_diverged_negative_reward_returns_formula_action(self):
        """BNBUSDT live case: Qwen3 chose HOLD but formula said PARTIAL_CLOSE_25;
        position later closed with reward=-0.25 → preferred should be
        formula_action=PARTIAL_CLOSE_25."""
        eng = make_engine()
        snap = make_snapshot(
            live_action="HOLD",
            formula_action="PARTIAL_CLOSE_25",
            diverged="true",
            exit_score="0.2500",  # exit_score<0.5 so no late_hold
        )
        # closed_by="unknown" but exit_score<0.5 → no late_hold, no premature;
        # just divergence_regret → reward = -(0.25) = -0.25
        out = make_outcome(hold_duration_sec="600", closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward < 0.0
        assert r.preferred_action == "PARTIAL_CLOSE_25"

    def test_diverged_positive_reward_keeps_live_action(self):
        """When Qwen3 diverged but the outcome was good (reward > 0),
        Rule 1 must NOT fire; live_action is correct."""
        eng = make_engine()
        snap = make_snapshot(
            live_action="FULL_CLOSE",
            formula_action="HOLD",
            diverged="true",
            exit_score="0.8000",
        )
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.reward > 0.0
        assert r.preferred_action == "FULL_CLOSE"

    def test_diverged_reward_zero_keeps_live_action(self):
        """reward == 0.0 is not < 0 → Rule 1 does not apply."""
        eng = make_engine()
        snap = make_snapshot(
            live_action="TIGHTEN_TRAIL",  # unknown action → reward=0
            formula_action="HOLD",
            diverged="true",
            exit_score="0.5000",
        )
        out = make_outcome(closed_by="unknown")
        r = eng.compute(snap, out)
        assert r.reward == pytest.approx(0.0, abs=1e-5)
        assert r.preferred_action == "TIGHTEN_TRAIL"

    def test_diverged_negative_reward_empty_formula_action_falls_through(self):
        """If formula_action is missing, Rule 1 guard (formula_action truthy)
        prevents returning an empty string; falls through to existing logic."""
        eng = make_engine()
        snap = make_snapshot(
            live_action="HOLD",
            formula_action="",   # explicitly empty
            diverged="true",
            exit_score="0.2000",
        )
        out = make_outcome(hold_duration_sec="600", closed_by="unknown")
        r = eng.compute(snap, out)
        # exit_score<0.5 → no late_hold; reward=-0.2 (>-0.3) → Rule 3 won't fire
        assert r.preferred_action == "HOLD"

    # ── Rule 2: premature_close PARTIAL_CLOSE_25 ──────────────────────────────

    def test_premature_close_partial_close_25_below_threshold_returns_hold(self):
        """premature_close + live=PARTIAL_CLOSE_25 + reward < -0.05 → HOLD."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.5000")
        # hold=60 < 300 → premature_close; exit_score=0.5 → base reward=0.5,
        # penalty = 0.4*(1-60/300) = 0.4*0.8 = 0.32 → reward ≈ 0.5-0.32 = 0.18
        # That's positive, not < -0.05 → need a low exit_score to make reward negative
        snap_low = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.1000")
        # reward = 0.1 - 0.4*(1-60/300) = 0.1 - 0.32 = -0.22 → < -0.05 ✓
        out = make_outcome(hold_duration_sec="60", closed_by="exit_management_agent")
        r = eng.compute(snap_low, out)
        assert r.regret_label == "premature_close"
        assert r.reward < -0.05
        assert r.preferred_action == "HOLD"

    def test_premature_close_above_reward_threshold_keeps_partial_close(self):
        """reward is between 0 and -0.05 (shallow) → Rule 2 does NOT fire;
        preferred_action stays PARTIAL_CLOSE_25."""
        eng = make_engine(premature_close_threshold_sec=300)
        # exit_score=0.5, hold=290 (just under threshold)
        # penalty = 0.4*(1-290/300) = 0.4*(10/300) ≈ 0.0133
        # reward ≈ 0.5 - 0.0133 = 0.487 → positive, NOT < -0.05
        snap = make_snapshot(live_action="PARTIAL_CLOSE_25", exit_score="0.5000")
        out = make_outcome(hold_duration_sec="290", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "premature_close"
        assert r.reward > -0.05
        assert r.preferred_action == "PARTIAL_CLOSE_25"

    def test_premature_close_full_close_rule_2_does_not_apply(self):
        """Rule 2 condition requires live_action == PARTIAL_CLOSE_25 specifically;
        a premature FULL_CLOSE is unaffected and keeps its live_action."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.1000")
        out = make_outcome(hold_duration_sec="60", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.regret_label == "premature_close"
        # reward = 0.1 - 0.32 = -0.22 → < -0.05 but Rule 2 MUST NOT fire
        assert r.preferred_action == "FULL_CLOSE"

    # ── Interaction: Rule 1 takes priority over Rule 2 ────────────────────────

    def test_rule1_fires_before_rule2_when_both_conditions_met(self):
        """If both diverged+bad AND premature_close conditions are satisfied,
        Rule 1 (formula_action) wins because it is evaluated first."""
        eng = make_engine(premature_close_threshold_sec=300)
        # premature_close: live=PARTIAL_CLOSE_25, hold=60s
        # diverged: formula_action=HOLD (formula said hold, Qwen3 closed early)
        snap = make_snapshot(
            live_action="PARTIAL_CLOSE_25",
            formula_action="HOLD",
            diverged="true",
            exit_score="0.1000",  # reward will be negative
        )
        out = make_outcome(hold_duration_sec="60", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.reward < 0.0
        # Rule 1: formula_action="HOLD" → preferred="HOLD"
        # Rule 2: would also give "HOLD" — they agree here, both produce HOLD
        assert r.preferred_action == "HOLD"

    # ── Neutral: neither rule should alter existing behaviour ──────────────────

    def test_no_divergence_no_premature_neutral_hold(self):
        """Standard HOLD with mild negative reward — no counterfactual rules fire."""
        eng = make_engine()
        snap = make_snapshot(live_action="HOLD", exit_score="0.1000", diverged="false")
        out = make_outcome(hold_duration_sec="7200", closed_by="unknown")
        r = eng.compute(snap, out)
        # reward = -0.1 → > -0.3 so Rule 3 also doesn't fire
        assert r.preferred_action == "HOLD"

    def test_no_divergence_positive_reward_full_close_unchanged(self):
        eng = make_engine()
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000", diverged="false")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        r = eng.compute(snap, out)
        assert r.reward > 0.0
        assert r.preferred_action == "FULL_CLOSE"


# ═══════════════════════════════════════════════════════════════════════════════
# TestMissingAndNullFields
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingAndNullFields:
    def test_empty_snapshot_and_outcome_no_crash(self):
        eng = make_engine()
        r = eng.compute({}, {})
        # Should return a valid RewardResult with defaults
        assert isinstance(r, RewardResult)
        assert -1.0 <= r.reward <= 1.0
        assert r.regret_label in ("none", "late_hold", "premature_close", "divergence_regret")

    def test_null_exit_score_defaults_to_0_5(self):
        """exit_score='null' must default to 0.5, not 0.0."""
        eng = make_engine()
        # HOLD with exit_score=0.5 should give reward=-0.5
        r = eng.compute({"live_action": "HOLD", "exit_score": "null"}, {})
        assert r.reward == pytest.approx(-0.5, abs=1e-5)

    def test_null_hold_duration_no_premature_penalty(self):
        """hold_duration_sec='null' → premature check skipped."""
        eng = make_engine(premature_close_threshold_sec=300)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000")
        out = make_outcome(hold_duration_sec="null")
        r = eng.compute(snap, out)
        # No penalty should be applied since hold_duration is unknown
        assert r.reward == pytest.approx(0.8, abs=1e-5)
        assert r.regret_label == "none"

    def test_missing_diverged_treated_as_false(self):
        """Snapshot without 'diverged' key → no divergence_regret."""
        eng = make_engine()
        snap = {"live_action": "HOLD", "exit_score": "0.3"}  # no diverged key
        out = {"hold_duration_sec": "7200", "closed_by": "exit_management_agent"}
        r = eng.compute(snap, out)
        assert r.regret_label != "divergence_regret"

    def test_exception_inside_compute_returns_zero(self):
        """If the internal compute raises (e.g. corrupted instance state),
        the outer try-except must return a neutral RewardResult rather than re-raising."""
        eng = make_engine()

        class _BrokenEngine(RewardEngine):
            def _compute_safe(self, s, o):
                raise RuntimeError("simulated internal error")

        broken = _BrokenEngine()
        r = broken.compute({}, {})
        assert r.reward == 0.0
        assert r.regret_label == "none"

    def test_safe_float_null_returns_default(self):
        assert _safe_float("null") == pytest.approx(0.0)
        assert _safe_float("null", default=0.5) == pytest.approx(0.5)

    def test_safe_float_none_returns_default(self):
        assert _safe_float(None, default=0.75) == pytest.approx(0.75)

    def test_safe_int_null_returns_none(self):
        assert _safe_int("null") is None

    def test_safe_int_valid_parses(self):
        assert _safe_int("1800") == 1800


# ═══════════════════════════════════════════════════════════════════════════════
# TestReplayWriter
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayWriter:
    _RESULT = RewardResult(
        reward=-0.6,
        regret_label="late_hold",
        regret_score=0.45,
        preferred_action="FULL_CLOSE",
    )

    @pytest.mark.asyncio
    async def test_write_calls_correct_stream(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake)
        await writer.write(
            "dec-001", "BTCUSDT",
            snapshot=make_snapshot(),
            outcome=make_outcome(),
            result=self._RESULT,
        )
        assert len(fake.xadd_calls) == 1
        stream, _ = fake.xadd_calls[0]
        assert stream == _REPLAY

    @pytest.mark.asyncio
    async def test_record_required_fields(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake)
        snap = make_snapshot(live_action="HOLD", exit_score="0.7000")
        out = make_outcome(hold_duration_sec="3600", closed_by="unknown")
        await writer.write("dec-abc", "ETHUSDT", snapshot=snap, outcome=out, result=self._RESULT)

        required = {
            "decision_id", "symbol", "record_time_epoch", "patch", "source",
            "live_action", "formula_action", "qwen3_action", "diverged",
            "exit_score", "entry_price", "side", "quantity",
            "hold_duration_sec", "close_price", "closed_by", "outcome_action",
            "reward", "regret_label", "regret_score", "preferred_action",
        }
        _, fields = fake.xadd_calls[0]
        assert required.issubset(fields.keys())

    @pytest.mark.asyncio
    async def test_record_patch_tag(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake)
        await writer.write("dec-x", "SOLUSDT",
                           snapshot={}, outcome={}, result=self._RESULT)
        _, fields = fake.xadd_calls[0]
        assert fields["patch"] == "PATCH-8C"
        assert fields["source"] == "exit_management_agent"

    @pytest.mark.asyncio
    async def test_reward_formatted_as_string(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake)
        result = RewardResult(reward=-0.123456, regret_label="none",
                              regret_score=0.0, preferred_action="HOLD")
        await writer.write("dec-f", "DOTUSDT", snapshot={}, outcome={}, result=result)
        _, fields = fake.xadd_calls[0]
        assert fields["reward"] == "-0.123456"
        assert fields["regret_score"] == "0.0000"

    @pytest.mark.asyncio
    async def test_snapshot_fields_pass_through(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake)
        snap = make_snapshot(live_action="FULL_CLOSE", exit_score="0.8000",
                             side="SHORT", entry_price="50000.0")
        out = make_outcome(hold_duration_sec="7200", closed_by="exit_management_agent")
        await writer.write("dec-pt", "BTCUSDT", snapshot=snap, outcome=out, result=self._RESULT)
        _, fields = fake.xadd_calls[0]
        assert fields["live_action"] == "FULL_CLOSE"
        assert fields["exit_score"] == "0.8000"
        assert fields["side"] == "SHORT"

    @pytest.mark.asyncio
    async def test_enabled_false_no_write(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake, enabled=False)
        await writer.write("dec-dis", "BTCUSDT",
                           snapshot={}, outcome={}, result=self._RESULT)
        assert len(fake.xadd_calls) == 0

    @pytest.mark.asyncio
    async def test_xadd_raises_no_crash(self):
        writer = ReplayWriter(_XaddRaiseFake())
        # Must not raise; error is swallowed
        await writer.write("dec-err", "ADAUSDT",
                           snapshot={}, outcome={}, result=self._RESULT)

    @pytest.mark.asyncio
    async def test_custom_stream_used(self):
        fake = _ReplayFake()
        writer = ReplayWriter(fake, replay_stream="custom:stream:my.replay")
        await writer.write("dec-cs", "BNBUSDT",
                           snapshot={}, outcome={}, result=self._RESULT)
        stream, _ = fake.xadd_calls[0]
        assert stream == "custom:stream:my.replay"


# ═══════════════════════════════════════════════════════════════════════════════
# TestOutcomeTrackerReplayIntegration
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutcomeTrackerReplayIntegration:
    def _make_tracker(self, symbol: str, decision_id: str, live_action: str = "HOLD",
                      exit_score: str = "0.70", fail_outcomes: bool = False):
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}{decision_id}"
        snap = _make_full_snapshot(symbol=symbol, decision_id=decision_id,
                                   live_action=live_action, exit_score=exit_score)
        fake = _TrackerFake(
            smembers_results={set_key: {decision_id}},
            hgetall_results={hash_key: snap},
            fail_on_stream=_OUTCOMES if fail_outcomes else None,
        )
        engine = RewardEngine()
        writer = ReplayWriter(fake, replay_stream=_REPLAY)
        tracker = OutcomeTracker(
            redis=fake,
            outcomes_stream=_OUTCOMES,
            enabled=True,
            reward_engine=engine,
            replay_writer=writer,
        )
        return tracker, fake

    @pytest.mark.asyncio
    async def test_replay_written_after_outcome_committed(self):
        tracker, fake = self._make_tracker("BTCUSDT", "dec-001")
        await tracker.update({"BTCUSDT"})  # baseline
        await tracker.update(set())        # BTC closed

        outcomes = fake.xadd_by_stream.get(_OUTCOMES, [])
        replays = fake.xadd_by_stream.get(_REPLAY, [])
        assert len(outcomes) == 1
        assert len(replays) == 1

    @pytest.mark.asyncio
    async def test_replay_stream_separate_from_outcomes_stream(self):
        tracker, fake = self._make_tracker("ETHUSDT", "dec-eth")
        await tracker.update({"ETHUSDT"})
        await tracker.update(set())

        streams_written = set(fake.xadd_by_stream.keys())
        assert _OUTCOMES in streams_written
        assert _REPLAY in streams_written
        assert _OUTCOMES != _REPLAY

    @pytest.mark.asyncio
    async def test_replay_contains_decision_id(self):
        tracker, fake = self._make_tracker("SOLUSDT", "dec-sol-99")
        await tracker.update({"SOLUSDT"})
        await tracker.update(set())

        replay_record = fake.xadd_by_stream[_REPLAY][0]
        assert replay_record["decision_id"] == "dec-sol-99"
        assert replay_record["symbol"] == "SOLUSDT"

    @pytest.mark.asyncio
    async def test_replay_not_written_if_outcome_fails(self):
        """If outcome xadd raises, no srem and no replay must be written."""
        tracker, fake = self._make_tracker("LTCUSDT", "dec-ltc", fail_outcomes=True)
        await tracker.update({"LTCUSDT"})
        await tracker.update(set())

        assert _OUTCOMES not in fake.xadd_by_stream
        assert _REPLAY not in fake.xadd_by_stream
        assert len(fake.srem_calls) == 0

    @pytest.mark.asyncio
    async def test_replay_write_failure_does_not_crash_tracker(self):
        """ReplayWriter.write failure must not propagate to the agent loop."""
        symbol, did = "LINKUSDT", "dec-link"
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}{did}"
        snap = _make_full_snapshot(symbol=symbol, decision_id=did)

        fake = _TrackerFake(
            smembers_results={set_key: {did}},
            hgetall_results={hash_key: snap},
        )
        # The xadd_raises fake as the inner redis of ReplayWriter
        _raise_fake = _XaddRaiseFake()

        engine = RewardEngine()
        writer = ReplayWriter(_raise_fake, replay_stream=_REPLAY)
        tracker = OutcomeTracker(
            redis=fake,
            outcomes_stream=_OUTCOMES,
            enabled=True,
            reward_engine=engine,
            replay_writer=writer,
        )
        # Should not raise
        await tracker.update({symbol})
        await tracker.update(set())

        # Outcome must still be written via the normal fake
        assert len(fake.xadd_by_stream.get(_OUTCOMES, [])) == 1

    @pytest.mark.asyncio
    async def test_no_engine_no_replay(self):
        """OutcomeTracker without reward_engine/replay_writer = PATCH-8B behaviour."""
        symbol, did = "XRPUSDT", "dec-xrp"
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"
        hash_key = f"{_SNAPSHOT_HASH_PREFIX}{did}"
        snap = _make_full_snapshot(symbol=symbol, decision_id=did)
        fake = _TrackerFake(
            smembers_results={set_key: {did}},
            hgetall_results={hash_key: snap},
        )
        tracker = OutcomeTracker(
            redis=fake,
            outcomes_stream=_OUTCOMES,
            enabled=True,
            reward_engine=None,
            replay_writer=None,
        )
        await tracker.update({symbol})
        await tracker.update(set())

        # Outcome written, replay not written
        assert len(fake.xadd_by_stream.get(_OUTCOMES, [])) == 1
        assert _REPLAY not in fake.xadd_by_stream

    @pytest.mark.asyncio
    async def test_replay_has_reward_and_regret_fields(self):
        tracker, fake = self._make_tracker("AVAXUSDT", "dec-avax",
                                           live_action="HOLD", exit_score="0.80")
        await tracker.update({"AVAXUSDT"})
        await tracker.update(set())

        record = fake.xadd_by_stream[_REPLAY][0]
        assert "reward" in record
        assert "regret_label" in record
        assert "regret_score" in record
        assert "preferred_action" in record
        # HOLD with high exit_score → negative reward string
        assert float(record["reward"]) < 0.0

    @pytest.mark.asyncio
    async def test_multiple_decisions_all_get_replay_records(self):
        symbol = "MATICUSDT"
        ids = {"dec-m1", "dec-m2"}
        set_key = f"{_PENDING_SET_PREFIX}{symbol}"
        snaps = {
            f"{_SNAPSHOT_HASH_PREFIX}{d}": _make_full_snapshot(symbol=symbol, decision_id=d)
            for d in ids
        }
        fake = _TrackerFake(smembers_results={set_key: ids}, hgetall_results=snaps)
        engine = RewardEngine()
        writer = ReplayWriter(fake, replay_stream=_REPLAY)
        tracker = OutcomeTracker(
            redis=fake,
            outcomes_stream=_OUTCOMES,
            enabled=True,
            reward_engine=engine,
            replay_writer=writer,
        )
        await tracker.update({symbol})
        await tracker.update(set())

        assert len(fake.xadd_by_stream.get(_OUTCOMES, [])) == 2
        assert len(fake.xadd_by_stream.get(_REPLAY, [])) == 2
