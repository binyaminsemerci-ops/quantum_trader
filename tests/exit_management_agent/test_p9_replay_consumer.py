"""Tests for PATCH-9: replay consumer / evaluation engine.

Covers:
    - ops/offline/replay_metrics.py  — load_jsonl, load_jsonl_dir, aggregate(),
                                       RewardStats, RegretDistribution, signals
    - ops/offline/dpo_export.py      — build_dpo_dataset, write_dpo_jsonl,
                                       DpoPair, prompt builder

No I/O calls touch real files except where tmp_path is used explicitly.
All aggregation tests use synthetic ReplayRecord objects.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import pytest

from ops.offline.replay_schema import ReplayRecord
from ops.offline.replay_metrics import (
    RewardStats,
    RegretDistribution,
    ReplayMetrics,
    aggregate,
    load_jsonl,
    load_jsonl_dir,
    _bucket_hold,
    _MIN_SAMPLES,
    _DIVERGENCE_SIGNAL_THRESHOLD,
    _HOLD_NEGATIVE_MEAN,
    _HIGH_REGRET_THRESHOLD,
)
from ops.offline.dpo_export import (
    DpoPair,
    build_dpo_dataset,
    write_dpo_jsonl,
    _build_prompt,
    _estimate_reward_gap,
    _DEFAULT_MIN_REWARD_GAP,
)


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def _make_record(
    *,
    stream_id: str = "1700000000001-0",
    decision_id: str = "test-dec-001",
    symbol: str = "BTCUSDT",
    side: str = "LONG",
    live_action: str = "HOLD",
    formula_action: str = "HOLD",
    qwen3_action: str = "null",
    diverged: bool = False,
    exit_score: Optional[float] = 0.4,
    entry_price: Optional[float] = 30000.0,
    hold_duration_sec: Optional[int] = 120,
    reward: Optional[float] = None,
    regret_label: str = "none",
    regret_score: Optional[float] = None,
    preferred_action: str = "HOLD",
    close_price: Optional[float] = None,
    quantity: Optional[float] = 1.0,
) -> ReplayRecord:
    return ReplayRecord(
        stream_id=stream_id,
        decision_id=decision_id,
        symbol=symbol,
        record_time_epoch=1700000000,
        patch="PATCH-9-test",
        source="test",
        live_action=live_action,
        formula_action=formula_action,
        qwen3_action=qwen3_action,
        diverged=diverged,
        exit_score=exit_score,
        entry_price=entry_price,
        side=side,
        quantity=quantity,
        hold_duration_sec=hold_duration_sec,
        close_price=close_price,
        closed_by="exit_management_agent",
        outcome_action=live_action,
        reward=reward,
        regret_label=regret_label,
        regret_score=regret_score,
        preferred_action=preferred_action,
    )


def _write_jsonl(path: Path, records: list[ReplayRecord]) -> None:
    """Write records to a JSONL file for test I/O cases."""
    with open(path, "w") as fh:
        for rec in records:
            fh.write(rec.to_json_line() + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  _bucket_hold
# ══════════════════════════════════════════════════════════════════════════════

class TestBucketHold:
    def test_none_is_unknown(self):
        assert _bucket_hold(None) == "unknown"

    def test_zero_is_lt5m(self):
        assert _bucket_hold(0) == "<5m"

    def test_299_is_lt5m(self):
        assert _bucket_hold(299) == "<5m"

    def test_300_is_5m_1h(self):
        assert _bucket_hold(300) == "5m-1h"

    def test_3599_is_5m_1h(self):
        assert _bucket_hold(3599) == "5m-1h"

    def test_3600_is_1h_4h(self):
        assert _bucket_hold(3600) == "1h-4h"

    def test_14399_is_1h_4h(self):
        assert _bucket_hold(14399) == "1h-4h"

    def test_14400_is_gt4h(self):
        assert _bucket_hold(14400) == ">4h"

    def test_large_value_is_gt4h(self):
        assert _bucket_hold(999999) == ">4h"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RewardStats.from_values
# ══════════════════════════════════════════════════════════════════════════════

class TestRewardStatsFromValues:
    def test_empty_list(self):
        rs = RewardStats.from_values([])
        assert rs.count == 0
        assert rs.mean is None
        assert rs.stdev is None
        assert rs.min is None
        assert rs.max is None
        assert rs.win_count == 0
        assert rs.loss_count == 0
        assert rs.neutral_count == 0

    def test_all_none(self):
        rs = RewardStats.from_values([None, None, None])
        assert rs.count == 3
        assert rs.mean is None
        assert rs.win_count == 0
        assert rs.neutral_count == 3

    def test_single_positive(self):
        rs = RewardStats.from_values([0.5])
        assert rs.count == 1
        assert rs.mean == pytest.approx(0.5, abs=1e-5)
        assert rs.win_count == 1
        assert rs.loss_count == 0
        assert rs.neutral_count == 0
        assert rs.stdev is None       # can't compute stdev for n=1

    def test_win_loss_neutral_counts(self):
        rs = RewardStats.from_values([0.5, -0.3, None, 0.0, 0.8, -0.1])
        # 0.5 > 0 → win ; -0.3 < 0 → loss ; None → neutral ; 0.0 → neutral ; 0.8 → win ; -0.1 → loss
        assert rs.win_count == 2
        assert rs.loss_count == 2
        assert rs.neutral_count == 2

    def test_win_rate_property(self):
        rs = RewardStats.from_values([0.5, 0.3, -0.1])
        assert rs.win_rate == pytest.approx(100.0 * 2 / 3, abs=0.1)

    def test_loss_rate_property(self):
        rs = RewardStats.from_values([0.5, 0.3, -0.1])
        assert rs.loss_rate == pytest.approx(100.0 * 1 / 3, abs=0.1)

    def test_win_rate_zero_count(self):
        rs = RewardStats.from_values([])
        assert rs.win_rate == 0.0
        assert rs.loss_rate == 0.0

    def test_stdev_with_multiple_values(self):
        rs = RewardStats.from_values([1.0, -1.0])
        assert rs.stdev is not None
        assert rs.stdev > 0

    def test_min_max_correct(self):
        rs = RewardStats.from_values([0.5, -1.0, 0.2])
        assert rs.min == pytest.approx(-1.0, abs=1e-5)
        assert rs.max == pytest.approx(0.5, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RegretDistribution
# ══════════════════════════════════════════════════════════════════════════════

class TestRegretDistribution:
    def test_defaults_zero(self):
        rd = RegretDistribution()
        assert rd.none == 0
        assert rd.late_hold == 0
        assert rd.premature_close == 0
        assert rd.divergence_regret == 0
        assert rd.total == 0
        assert rd.weighted_severity == 0.0

    def test_actionable_rate_all_none(self):
        rd = RegretDistribution(none=10, total=10)
        assert rd.actionable_rate == 0.0

    def test_actionable_rate_partial(self):
        rd = RegretDistribution(none=5, late_hold=3, premature_close=2, total=10)
        assert rd.actionable_rate == pytest.approx(50.0, abs=0.1)

    def test_actionable_rate_zero_total(self):
        rd = RegretDistribution()
        assert rd.actionable_rate == 0.0

    def test_to_dict_keys(self):
        rd = RegretDistribution(none=2, late_hold=1, total=3)
        d = rd.to_dict()
        assert set(d.keys()) == {
            "none", "late_hold", "premature_close", "divergence_regret",
            "total", "actionable_rate_pct", "weighted_severity"
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  aggregate() — core logic
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregateEmpty:
    def test_empty_returns_defaults(self):
        m = aggregate([])
        assert m.total_records == 0
        assert m.divergence_count == 0
        assert m.overall_reward is None

    def test_empty_signals_empty(self):
        m = aggregate([])
        assert m.signals == []


class TestAggregateCounts:
    def test_total_records(self):
        recs = [_make_record() for _ in range(7)]
        m = aggregate(recs)
        assert m.total_records == 7

    def test_divergence_count(self):
        recs = [
            _make_record(diverged=True),
            _make_record(diverged=True),
            _make_record(diverged=False),
        ]
        m = aggregate(recs)
        assert m.divergence_count == 2

    def test_divergence_rate_pct(self):
        recs = [_make_record(diverged=True)] * 3 + [_make_record(diverged=False)] * 7
        m = aggregate(recs)
        assert m.divergence_rate_pct == pytest.approx(30.0, abs=0.1)

    def test_records_with_reward(self):
        recs = [
            _make_record(reward=0.5),
            _make_record(reward=None),
            _make_record(reward=-0.1),
        ]
        m = aggregate(recs)
        assert m.records_with_reward == 2


class TestAggregateQwen3:
    def test_qwen3_available_count(self):
        recs = [
            _make_record(qwen3_action="FULL_CLOSE"),
            _make_record(qwen3_action="null"),
            _make_record(qwen3_action="HOLD"),
        ]
        m = aggregate(recs)
        assert m.qwen3_available == 2

    def test_qwen3_agrees_formula(self):
        recs = [
            _make_record(qwen3_action="HOLD", formula_action="HOLD"),
            _make_record(qwen3_action="FULL_CLOSE", formula_action="HOLD"),
        ]
        m = aggregate(recs)
        assert m.qwen3_agrees_formula == 1

    def test_qwen3_overrode_formula(self):
        recs = [
            _make_record(qwen3_action="HOLD", formula_action="HOLD"),
            _make_record(qwen3_action="FULL_CLOSE", formula_action="HOLD"),
            _make_record(qwen3_action="PARTIAL_CLOSE_25", formula_action="HOLD"),
        ]
        m = aggregate(recs)
        assert m.qwen3_overrode_formula == 2

    def test_qwen3_agrees_live(self):
        recs = [
            _make_record(qwen3_action="HOLD", live_action="HOLD"),
            _make_record(qwen3_action="FULL_CLOSE", live_action="HOLD"),
        ]
        m = aggregate(recs)
        assert m.qwen3_agrees_live == 1

    def test_qwen3_null_not_counted(self):
        recs = [_make_record(qwen3_action="null") for _ in range(5)]
        m = aggregate(recs)
        assert m.qwen3_available == 0
        assert m.qwen3_agrees_formula == 0
        assert m.qwen3_overrode_formula == 0


class TestAggregateRewardSlices:
    def test_reward_by_action_grouping(self):
        recs = [
            _make_record(live_action="HOLD", reward=0.5),
            _make_record(live_action="HOLD", reward=-0.1),
            _make_record(live_action="FULL_CLOSE", reward=0.8),
        ]
        m = aggregate(recs)
        assert "HOLD" in m.reward_by_action
        assert "FULL_CLOSE" in m.reward_by_action
        assert m.reward_by_action["HOLD"].count == 2
        assert m.reward_by_action["FULL_CLOSE"].count == 1

    def test_reward_by_symbol_grouping(self):
        recs = [
            _make_record(symbol="BTCUSDT", reward=0.3),
            _make_record(symbol="ETHUSDT", reward=-0.2),
            _make_record(symbol="BTCUSDT", reward=0.1),
        ]
        m = aggregate(recs)
        assert m.reward_by_symbol["BTCUSDT"].count == 2
        assert m.reward_by_symbol["ETHUSDT"].count == 1

    def test_reward_by_side_grouping(self):
        recs = [
            _make_record(side="LONG", reward=0.5),
            _make_record(side="SHORT", reward=-0.3),
        ]
        m = aggregate(recs)
        assert "LONG" in m.reward_by_side
        assert "SHORT" in m.reward_by_side

    def test_overall_reward_mean(self):
        recs = [_make_record(reward=r) for r in [0.2, 0.4, 0.6]]
        m = aggregate(recs)
        assert m.overall_reward.mean == pytest.approx(0.4, abs=1e-4)


class TestAggregateRegret:
    def test_regret_label_counts(self):
        recs = [
            _make_record(regret_label="late_hold", regret_score=0.5),
            _make_record(regret_label="premature_close", regret_score=0.3),
            _make_record(regret_label="divergence_regret", regret_score=0.6),
            _make_record(regret_label="none"),
        ]
        m = aggregate(recs)
        assert m.regret.late_hold == 1
        assert m.regret.premature_close == 1
        assert m.regret.divergence_regret == 1
        assert m.regret.none == 1
        assert m.regret.total == 4

    def test_regret_weighted_severity(self):
        recs = [
            _make_record(regret_label="late_hold", regret_score=0.4),
            _make_record(regret_label="premature_close", regret_score=0.6),
            _make_record(regret_label="none"),  # no score contribution
        ]
        m = aggregate(recs)
        assert m.regret.weighted_severity == pytest.approx(1.0, abs=1e-4)

    def test_regret_none_ignored_for_severity(self):
        recs = [_make_record(regret_label="none", regret_score=0.9)]
        m = aggregate(recs)
        assert m.regret.weighted_severity == 0.0


class TestAggregatePreferredAction:
    def test_preferred_matches_live(self):
        recs = [
            _make_record(live_action="HOLD", preferred_action="HOLD"),
            _make_record(live_action="HOLD", preferred_action="FULL_CLOSE"),
        ]
        m = aggregate(recs)
        assert m.preferred_matches_live == 1
        assert m.preferred_differs == 1

    def test_preferred_action_freq(self):
        recs = [
            _make_record(preferred_action="HOLD"),
            _make_record(preferred_action="HOLD"),
            _make_record(preferred_action="FULL_CLOSE"),
        ]
        m = aggregate(recs)
        assert m.preferred_action_freq["HOLD"] == 2
        assert m.preferred_action_freq["FULL_CLOSE"] == 1


class TestAggregateHoldBands:
    def test_hold_bands_populated(self):
        recs = [
            _make_record(hold_duration_sec=100),    # <5m
            _make_record(hold_duration_sec=600),    # 5m-1h
            _make_record(hold_duration_sec=7200),   # 1h-4h
            _make_record(hold_duration_sec=20000),  # >4h
            _make_record(hold_duration_sec=None),   # unknown
        ]
        m = aggregate(recs)
        assert m.hold_duration_bands.get("<5m") == 1
        assert m.hold_duration_bands.get("5m-1h") == 1
        assert m.hold_duration_bands.get("1h-4h") == 1
        assert m.hold_duration_bands.get(">4h") == 1
        assert m.hold_duration_bands.get("unknown") == 1


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Signal derivation
# ══════════════════════════════════════════════════════════════════════════════

class TestDeriveSignals:
    """Test that _derive_signals fires (or does not fire) at threshold boundaries."""

    def _recs_with_divergence(self, rate: float, n: int = 20) -> list[ReplayRecord]:
        k = int(n * rate)
        return (
            [_make_record(diverged=True)] * k
            + [_make_record(diverged=False)] * (n - k)
        )

    def test_divergence_signal_fires_above_threshold(self):
        recs = self._recs_with_divergence(0.30, n=20)  # 30% > 25%
        m = aggregate(recs)
        assert any("DIVERGENCE" in s for s in m.signals)

    def test_divergence_signal_no_fire_below_threshold(self):
        recs = self._recs_with_divergence(0.20, n=20)  # 20% < 25%
        m = aggregate(recs)
        assert not any("DIVERGENCE" in s for s in m.signals)

    def test_hold_drag_signal_fires(self):
        # HOLD with mean reward below _HOLD_NEGATIVE_MEAN = -0.2
        recs = [_make_record(live_action="HOLD", reward=-0.5) for _ in range(_MIN_SAMPLES)]
        m = aggregate(recs)
        assert any("HOLD DRAG" in s for s in m.signals)

    def test_hold_drag_no_fire_positive_mean(self):
        recs = [_make_record(live_action="HOLD", reward=0.3) for _ in range(_MIN_SAMPLES)]
        m = aggregate(recs)
        assert not any("HOLD DRAG" in s for s in m.signals)

    def test_hold_drag_no_fire_below_min_samples(self):
        recs = [_make_record(live_action="HOLD", reward=-0.5) for _ in range(_MIN_SAMPLES - 1)]
        m = aggregate(recs)
        assert not any("HOLD DRAG" in s for s in m.signals)

    def test_premature_close_signal_fires(self):
        # 5/10 = 50% > 20%
        recs = (
            [_make_record(regret_label="premature_close")] * 5
            + [_make_record(regret_label="none")] * 5
        )
        m = aggregate(recs)
        assert any("PREMATURE CLOSE" in s for s in m.signals)

    def test_premature_close_no_fire_below_threshold(self):
        # 1/10 = 10% < 20%
        recs = (
            [_make_record(regret_label="premature_close")] * 1
            + [_make_record(regret_label="none")] * 9
        )
        m = aggregate(recs)
        assert not any("PREMATURE CLOSE" in s for s in m.signals)

    def test_late_hold_signal_fires(self):
        recs = (
            [_make_record(regret_label="late_hold")] * 5
            + [_make_record(regret_label="none")] * 5
        )
        m = aggregate(recs)
        assert any("LATE HOLD" in s for s in m.signals)

    def test_missed_signal_fires_at_high_rate(self):
        # 9/10 = 90% preferred != live → > 40%
        recs = (
            [_make_record(preferred_action="FULL_CLOSE", live_action="HOLD")] * 9
            + [_make_record(preferred_action="HOLD", live_action="HOLD")] * 1
        )
        m = aggregate(recs)
        assert any("MISSED SIGNALS" in s for s in m.signals)

    def test_missed_signal_no_fire_low_rate(self):
        # 2/10 = 20% < 40%
        recs = (
            [_make_record(preferred_action="FULL_CLOSE", live_action="HOLD")] * 2
            + [_make_record(preferred_action="HOLD", live_action="HOLD")] * 8
        )
        m = aggregate(recs)
        assert not any("MISSED SIGNALS" in s for s in m.signals)

    def test_no_signals_on_clean_data(self):
        # Good behaviour: moderate divergence, positive rewards, matching preferred
        recs = [
            _make_record(
                diverged=False,
                live_action="FULL_CLOSE",
                preferred_action="FULL_CLOSE",
                reward=0.5,
                regret_label="none",
            )
            for _ in range(10)
        ]
        m = aggregate(recs)
        # Should have little or no signals
        hold_drag   = any("HOLD DRAG" in s for s in m.signals)
        divergence  = any("DIVERGENCE" in s for s in m.signals)
        premature   = any("PREMATURE CLOSE" in s for s in m.signals)
        assert not hold_drag
        assert not divergence
        assert not premature


# ══════════════════════════════════════════════════════════════════════════════
# 6.  load_jsonl (file I/O via tmp_path)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadJsonl:
    def test_round_trip_single_record(self, tmp_path):
        rec = _make_record()
        p = tmp_path / "replay_2024-01-01.jsonl"
        _write_jsonl(p, [rec])

        loaded, skipped = load_jsonl(p)
        assert skipped == 0
        assert len(loaded) == 1
        assert loaded[0].decision_id == rec.decision_id
        assert loaded[0].symbol == rec.symbol

    def test_round_trip_multiple_records(self, tmp_path):
        recs = [_make_record(decision_id=f"dec-{i}", symbol="ETHUSDT") for i in range(5)]
        p = tmp_path / "replay_2024-01-02.jsonl"
        _write_jsonl(p, recs)

        loaded, skipped = load_jsonl(p)
        assert skipped == 0
        assert len(loaded) == 5

    def test_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "replay_bad.jsonl"
        rec = _make_record()
        with open(p, "w") as fh:
            fh.write(rec.to_json_line() + "\n")
            fh.write("NOT_JSON{{{\n")
            fh.write(rec.to_json_line() + "\n")

        loaded, skipped = load_jsonl(p)
        assert skipped == 1
        assert len(loaded) == 2

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "replay_empty.jsonl"
        p.write_text("")
        loaded, skipped = load_jsonl(p)
        assert loaded == []
        assert skipped == 0

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "replay_blanks.jsonl"
        rec = _make_record()
        with open(p, "w") as fh:
            fh.write("\n")
            fh.write(rec.to_json_line() + "\n")
            fh.write("   \n")
        loaded, skipped = load_jsonl(p)
        assert len(loaded) == 1
        assert skipped == 0

    def test_preserves_reward_none(self, tmp_path):
        rec = _make_record(reward=None)
        p = tmp_path / "replay_norew.jsonl"
        _write_jsonl(p, [rec])
        loaded, _ = load_jsonl(p)
        assert loaded[0].reward is None

    def test_preserves_numeric_reward(self, tmp_path):
        rec = _make_record(reward=0.73)
        p = tmp_path / "replay_rew.jsonl"
        _write_jsonl(p, [rec])
        loaded, _ = load_jsonl(p)
        assert loaded[0].reward == pytest.approx(0.73, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  load_jsonl_dir (directory scanning)
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadJsonlDir:
    def test_loads_matching_files(self, tmp_path):
        for i in range(3):
            p = tmp_path / f"replay_2024-01-0{i+1}.jsonl"
            _write_jsonl(p, [_make_record(decision_id=f"d{i}")])

        records, skipped = load_jsonl_dir(tmp_path)
        assert len(records) == 3
        assert skipped == 0

    def test_ignores_non_matching_files(self, tmp_path):
        p_good = tmp_path / "replay_2024-01-01.jsonl"
        p_bad  = tmp_path / "other_file.jsonl"
        _write_jsonl(p_good, [_make_record()])
        _write_jsonl(p_bad, [_make_record(decision_id="should-not-load")])

        records, _ = load_jsonl_dir(tmp_path)
        assert len(records) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        records, skipped = load_jsonl_dir(tmp_path)
        assert records == []
        assert skipped == 0

    def test_accumulates_skipped_from_multiple_files(self, tmp_path):
        for i in range(2):
            p = tmp_path / f"replay_2024-01-0{i+1}.jsonl"
            with open(p, "w") as fh:
                fh.write("BADJSON\n")

        _, skipped = load_jsonl_dir(tmp_path)
        assert skipped == 2

    def test_total_record_count_across_files(self, tmp_path):
        for i in range(4):
            p = tmp_path / f"replay_2024-01-0{i+1}.jsonl"
            _write_jsonl(p, [_make_record() for _ in range(3)])

        records, _ = load_jsonl_dir(tmp_path)
        assert len(records) == 12


# ══════════════════════════════════════════════════════════════════════════════
# 8.  _estimate_reward_gap
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateRewardGap:
    def test_no_reward_uses_regret_score(self):
        rec = _make_record(reward=None, regret_score=0.6)
        gap = _estimate_reward_gap(rec)
        assert gap == pytest.approx(0.6, abs=1e-5)

    def test_no_reward_no_regret_score_is_zero(self):
        rec = _make_record(reward=None, regret_score=None)
        gap = _estimate_reward_gap(rec)
        assert gap == 0.0

    def test_with_reward_adds_abs_reward(self):
        rec = _make_record(reward=-0.5, regret_score=0.4)
        gap = _estimate_reward_gap(rec)
        # abs(-0.5) + 0.4 * 0.5 = 0.5 + 0.2 = 0.7
        assert gap == pytest.approx(0.7, abs=1e-5)

    def test_positive_reward_adds_reward(self):
        rec = _make_record(reward=0.8, regret_score=0.0)
        gap = _estimate_reward_gap(rec)
        assert gap == pytest.approx(0.8, abs=1e-5)

    def test_reward_with_no_regret_score(self):
        rec = _make_record(reward=0.5, regret_score=None)
        gap = _estimate_reward_gap(rec)
        # abs(0.5) + 0.0 * 0.5 = 0.5
        assert gap == pytest.approx(0.5, abs=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  build_dpo_dataset
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildDpoDataset:
    def test_empty_input(self):
        pairs = build_dpo_dataset([])
        assert pairs == []

    def test_matching_preferred_and_live_skipped(self):
        rec = _make_record(preferred_action="HOLD", live_action="HOLD", reward=0.5)
        pairs = build_dpo_dataset([rec])
        assert pairs == []

    def test_unknown_preferred_skipped(self):
        rec = _make_record(preferred_action="UNKNOWN", live_action="HOLD", reward=0.5)
        pairs = build_dpo_dataset([rec])
        assert pairs == []

    def test_unknown_live_skipped(self):
        rec = _make_record(preferred_action="FULL_CLOSE", live_action="UNKNOWN", reward=0.5)
        pairs = build_dpo_dataset([rec])
        assert pairs == []

    def test_empty_preferred_skipped(self):
        rec = _make_record(preferred_action="", live_action="HOLD", reward=0.5)
        pairs = build_dpo_dataset([rec])
        assert pairs == []

    def test_gap_below_threshold_skipped(self):
        # reward=None, regret_score=None → gap=0.0 < 0.2 default
        rec = _make_record(
            preferred_action="FULL_CLOSE",
            live_action="HOLD",
            reward=None,
            regret_score=None,
        )
        pairs = build_dpo_dataset([rec], min_reward_gap=0.2)
        assert pairs == []

    def test_valid_pair_produced(self):
        rec = _make_record(
            preferred_action="FULL_CLOSE",
            live_action="HOLD",
            reward=-0.4,
            regret_score=0.3,
        )
        pairs = build_dpo_dataset([rec])
        assert len(pairs) == 1
        p = pairs[0]
        assert p.chosen_action == "FULL_CLOSE"
        assert p.rejected_action == "HOLD"
        assert p.decision_id == rec.decision_id
        assert p.symbol == rec.symbol

    def test_pair_gap_meets_threshold(self):
        # Exactly at threshold — should be included
        rec = _make_record(
            preferred_action="FULL_CLOSE",
            live_action="HOLD",
            reward=None,
            regret_score=0.2,
        )
        # gap = 0.2 = threshold → included (>= check)
        pairs = build_dpo_dataset([rec], min_reward_gap=0.2)
        assert len(pairs) == 1

    def test_multiple_records_filtered_correctly(self):
        good = _make_record(preferred_action="FULL_CLOSE", live_action="HOLD", reward=-0.5)
        bad_match = _make_record(preferred_action="HOLD", live_action="HOLD", reward=0.5)
        bad_gap = _make_record(
            preferred_action="FULL_CLOSE", live_action="HOLD",
            reward=None, regret_score=None
        )
        pairs = build_dpo_dataset([good, bad_match, bad_gap])
        assert len(pairs) == 1

    def test_chosen_rejected_format(self):
        rec = _make_record(preferred_action="TIGHTEN_TRAIL", live_action="HOLD", reward=-0.6)
        pairs = build_dpo_dataset([rec])
        p = pairs[0]
        assert p.chosen.startswith("<|assistant|>TIGHTEN_TRAIL")
        assert p.rejected.startswith("<|assistant|>HOLD")

    def test_prompt_contains_system_header(self):
        rec = _make_record(preferred_action="FULL_CLOSE", live_action="HOLD", reward=-0.5)
        pairs = build_dpo_dataset([rec])
        assert "<|system|>" in pairs[0].prompt

    def test_custom_min_reward_gap(self):
        rec = _make_record(
            preferred_action="FULL_CLOSE",
            live_action="HOLD",
            reward=-0.3,
            regret_score=None,
        )
        # gap = 0.3, default threshold 0.2 → passes; threshold 0.5 → rejects
        assert len(build_dpo_dataset([rec], min_reward_gap=0.2)) == 1
        assert len(build_dpo_dataset([rec], min_reward_gap=0.5)) == 0

    def test_reward_gap_stored_on_pair(self):
        rec = _make_record(
            preferred_action="FULL_CLOSE",
            live_action="HOLD",
            reward=-0.5,
            regret_score=0.4,
        )
        pairs = build_dpo_dataset([rec])
        assert pairs[0].reward_gap > 0.0

    def test_regret_label_stored_on_pair(self):
        rec = _make_record(
            preferred_action="FULL_CLOSE",
            live_action="HOLD",
            reward=-0.5,
            regret_label="premature_close",
        )
        pairs = build_dpo_dataset([rec])
        assert pairs[0].regret_label == "premature_close"


# ══════════════════════════════════════════════════════════════════════════════
# 10.  _build_prompt
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:
    def test_contains_symbol(self):
        rec = _make_record(symbol="SOLUSDT")
        prompt = _build_prompt(rec)
        assert "SOLUSDT" in prompt

    def test_contains_side(self):
        rec = _make_record(side="SHORT")
        prompt = _build_prompt(rec)
        assert "SHORT" in prompt

    def test_contains_formula_action(self):
        rec = _make_record(formula_action="FULL_CLOSE")
        prompt = _build_prompt(rec)
        assert "FULL_CLOSE" in prompt

    def test_contains_qwen3_action(self):
        rec = _make_record(qwen3_action="TIGHTEN_TRAIL")
        prompt = _build_prompt(rec)
        assert "TIGHTEN_TRAIL" in prompt

    def test_system_and_user_markers(self):
        rec = _make_record()
        prompt = _build_prompt(rec)
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt

    def test_none_exit_score_rendered_as_unknown(self):
        rec = _make_record(exit_score=None)
        prompt = _build_prompt(rec)
        assert "unknown" in prompt

    def test_numeric_exit_score_rendered(self):
        rec = _make_record(exit_score=0.87654)
        prompt = _build_prompt(rec)
        assert "0.8765" in prompt  # 4 decimal places

    def test_none_hold_duration_rendered_as_unknown(self):
        rec = _make_record(hold_duration_sec=None)
        prompt = _build_prompt(rec)
        assert "unknown" in prompt

    def test_hold_duration_rendered_with_sec_suffix(self):
        rec = _make_record(hold_duration_sec=1800)
        prompt = _build_prompt(rec)
        assert "1800 sec" in prompt


# ══════════════════════════════════════════════════════════════════════════════
# 11.  DpoPair conversions
# ══════════════════════════════════════════════════════════════════════════════

class TestDpoPair:
    def _make_pair(self) -> DpoPair:
        return DpoPair(
            decision_id="dec-001",
            symbol="BTCUSDT",
            prompt="<|system|>sys\n<|user|>q",
            chosen="<|assistant|>FULL_CLOSE — Close the entire position immediately.",
            rejected="<|assistant|>HOLD — Keep the position open and continue monitoring.",
            chosen_action="FULL_CLOSE",
            rejected_action="HOLD",
            reward=-0.4,
            regret_label="late_hold",
            reward_gap=0.6,
        )

    def test_to_training_dict_keys(self):
        p = self._make_pair()
        d = p.to_training_dict()
        assert set(d.keys()) == {"prompt", "chosen", "rejected"}

    def test_to_full_dict_contains_metadata(self):
        p = self._make_pair()
        d = p.to_full_dict()
        assert "decision_id" in d
        assert "symbol" in d
        assert "chosen_action" in d
        assert "rejected_action" in d
        assert "reward" in d
        assert "regret_label" in d
        assert "reward_gap" in d
        assert "source" in d

    def test_to_full_dict_reward_gap_rounded(self):
        p = self._make_pair()
        p.reward_gap = 0.66666
        d = p.to_full_dict()
        assert d["reward_gap"] == pytest.approx(0.6667, abs=1e-4)

    def test_to_training_dict_values_match(self):
        p = self._make_pair()
        d = p.to_training_dict()
        assert d["prompt"] == p.prompt
        assert d["chosen"] == p.chosen
        assert d["rejected"] == p.rejected


# ══════════════════════════════════════════════════════════════════════════════
# 12.  write_dpo_jsonl
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteDpoJsonl:
    def _make_pairs(self, n: int = 3) -> list[DpoPair]:
        pairs = []
        for i in range(n):
            rec = _make_record(
                decision_id=f"d-{i}",
                preferred_action="FULL_CLOSE",
                live_action="HOLD",
                reward=-0.5,
            )
            pairs.extend(build_dpo_dataset([rec]))
        return pairs

    def test_creates_file(self, tmp_path):
        p = tmp_path / "dpo_output.jsonl"
        pairs = self._make_pairs(2)
        write_dpo_jsonl(p, pairs)
        assert p.exists()

    def test_line_count_matches_pairs(self, tmp_path):
        n = 4
        p = tmp_path / "dpo_output.jsonl"
        pairs = self._make_pairs(n)
        write_dpo_jsonl(p, pairs)
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        assert len(lines) == n

    def test_each_line_is_valid_json(self, tmp_path):
        p = tmp_path / "dpo_output.jsonl"
        pairs = self._make_pairs(3)
        write_dpo_jsonl(p, pairs)
        for line in p.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_full_mode_contains_metadata(self, tmp_path):
        p = tmp_path / "dpo_full.jsonl"
        pairs = self._make_pairs(1)
        write_dpo_jsonl(p, pairs, full=True)
        obj = json.loads(p.read_text().strip())
        assert "decision_id" in obj
        assert "source" in obj

    def test_training_mode_only_has_three_keys(self, tmp_path):
        p = tmp_path / "dpo_training.jsonl"
        pairs = self._make_pairs(1)
        write_dpo_jsonl(p, pairs, full=False)
        obj = json.loads(p.read_text().strip())
        assert set(obj.keys()) == {"prompt", "chosen", "rejected"}

    def test_empty_pairs_creates_empty_file(self, tmp_path):
        p = tmp_path / "dpo_empty.jsonl"
        write_dpo_jsonl(p, [])
        assert p.read_text() == ""
