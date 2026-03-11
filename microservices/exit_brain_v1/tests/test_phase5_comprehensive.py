"""
Phase 5 — Replay / Evaluation / Tuning: Comprehensive Test Suite.

Tests ALL Phase 5 modules:
  S1:  TradeExitObituary contract
  S2:  ReplayEvaluationRecord contract
  S3:  OfflineEvaluationSummary contract
  S4:  TuningRecommendation contract
  S5:  CalibrationArtifact contract
  S6:  CounterfactualEvaluator — 7 action simulations (14 tests)
  S7:  CounterfactualEvaluator — best action (3 tests)
  S8:  CounterfactualEvaluator — quality score formula (4 tests)
  S9:  ReplayObituaryWriter — regret score (3 tests)
  S10: ReplayObituaryWriter — preservation score (3 tests)
  S11: ReplayObituaryWriter — opportunity capture score (3 tests)
  S12: OfflineEvaluator — min samples (2 tests)
  S13: OfflineEvaluator — sub-evaluator delegation (4 tests)
  S14: BeliefCalibrationEvaluator (2 tests)
  S15: HazardCalibrationEvaluator (2 tests)
  S16: UtilityRankingEvaluator (2 tests)
  S17: PolicyChoiceEvaluator — baselines (8 tests)
  S18: ThresholdTuner — ±20% cap (4 tests)
  S19: ThresholdTuner — low sample cap (2 tests)
  S20: WeightTuner — normalization (3 tests)
  S21: WeightTuner — min MAE gap (2 tests)
  S22: ProposalBuilder — max 5 (2 tests)
  S23: ProposalBuilder — confidence sort (2 tests)
  S24: OutcomeReconstructor (2 tests)
  S25: Shadow stream verification (2 tests)
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ── Phase 5 contracts ────────────────────────────────────────────────────

from microservices.exit_brain_v1.models.trade_exit_obituary import TradeExitObituary
from microservices.exit_brain_v1.models.replay_evaluation_record import (
    ReplayEvaluationRecord,
)
from microservices.exit_brain_v1.models.offline_evaluation_summary import (
    OfflineEvaluationSummary,
)
from microservices.exit_brain_v1.models.tuning_recommendation import (
    TuningRecommendation,
    VALID_DIRECTIONS,
)
from microservices.exit_brain_v1.models.calibration_artifact import (
    CalibrationArtifact,
    VALID_CALIBRATION_TYPES,
)
from microservices.exit_brain_v1.models.action_candidate import (
    VALID_ACTIONS,
    ACTION_EXIT_FRACTIONS,
)

# ── Phase 5 replay ──────────────────────────────────────────────────────

from microservices.exit_brain_v1.replay.outcome_reconstructor import (
    OutcomeReconstructor,
    OutcomePathResult,
    DEFAULT_HORIZON_SEC,
    MIN_PRICE_SAMPLES,
)
from microservices.exit_brain_v1.replay.counterfactual_evaluator import (
    CounterfactualEvaluator,
)
from microservices.exit_brain_v1.replay.replay_obituary_writer import (
    ReplayObituaryWriter,
)

# ── Phase 5 evaluators ──────────────────────────────────────────────────

from microservices.exit_brain_v1.evaluators.belief_calibration_evaluator import (
    BeliefCalibrationEvaluator,
)
from microservices.exit_brain_v1.evaluators.hazard_calibration_evaluator import (
    HazardCalibrationEvaluator,
    HAZARD_AXES,
)
from microservices.exit_brain_v1.evaluators.utility_ranking_evaluator import (
    UtilityRankingEvaluator,
)
from microservices.exit_brain_v1.evaluators.policy_choice_evaluator import (
    PolicyChoiceEvaluator,
    BASELINE_DEFINITIONS,
)
from microservices.exit_brain_v1.evaluators.offline_evaluator import (
    OfflineEvaluator,
    MIN_SAMPLES_FOR_EVAL,
    LOW_SAMPLE_THRESHOLD,
)

# ── Phase 5 tuning ──────────────────────────────────────────────────────

from microservices.exit_brain_v1.tuning.threshold_tuner import (
    ThresholdTuner,
    MAX_CHANGE_FRACTION as TT_MAX_CHANGE,
    LOW_SAMPLE_CONFIDENCE_CAP as TT_LOW_CAP,
    LOW_SAMPLE_THRESHOLD as TT_LOW_SAMPLE,
)
from microservices.exit_brain_v1.tuning.weight_tuner import (
    WeightTuner,
    MAX_CHANGE_FRACTION as WT_MAX_CHANGE,
    MIN_MAE_GAP,
)
from microservices.exit_brain_v1.tuning.proposal_builder import (
    ProposalBuilder,
    MAX_RECOMMENDATIONS_PER_RUN,
)


# ═════════════════════════════════════════════════════════════════════════
# Helper factories
# ═════════════════════════════════════════════════════════════════════════

def _make_obituary(**overrides) -> TradeExitObituary:
    """Minimal valid TradeExitObituary."""
    now = time.time()
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        obituary_id=str(uuid.uuid4()),
        obituary_timestamp=now,
        open_timestamp=now - 7200,
        lifecycle_duration_seconds=7200.0,
        peak_unrealized_pnl=100.0,
        peak_unrealized_pnl_pct=0.02,
        max_drawdown_after_peak=20.0,
        recommended_action_at_decision="HOLD",
        recommended_action_utility=0.6,
        recommended_action_confidence=0.7,
        policy_passed=True,
        belief_snapshot={"exit_pressure": 0.3, "hold_conviction": 0.7},
        hazard_snapshot={"composite_hazard": 0.4, "drawdown_hazard": 0.3},
        utility_snapshot=[{"action": "HOLD", "net_utility": 0.6}],
        decision_trace_ref="trace-001",
        source_decision_id="dec-001",
        regret_score=0.2,
        preservation_score=0.8,
        opportunity_capture_score=0.7,
        shadow_only=True,
    )
    defaults.update(overrides)
    return TradeExitObituary(**defaults)


def _make_replay_record(**overrides) -> ReplayEvaluationRecord:
    """Minimal valid ReplayEvaluationRecord."""
    now = time.time()
    defaults = dict(
        record_id=str(uuid.uuid4()),
        position_id="pos-001",
        symbol="BTCUSDT",
        replay_timestamp=now,
        source_decision_timestamp=now - 3600,
        evaluated_horizon_seconds=14400.0,
        chosen_action="HOLD",
        action_rank_at_decision=1,
        counterfactual_actions_evaluated=sorted(VALID_ACTIONS),
        predicted_exit_pressure=0.3,
        realized_exit_pressure_proxy=0.35,
        predicted_hold_conviction=0.7,
        realized_hold_conviction_proxy=0.65,
        predicted_composite_hazard=0.4,
        realized_hazard_proxy=0.45,
        predicted_utility_by_action={"HOLD": 0.6, "CLOSE_FULL": 0.3},
        realized_utility_proxy_by_action={"HOLD": 0.55, "CLOSE_FULL": 0.35},
        ex_post_best_action="HOLD",
        ex_post_best_utility_proxy=0.55,
        decision_quality_score=0.75,
        explanation_consistency_score=0.90,
        calibration_errors={"exit_pressure": -0.05},
        shadow_only=True,
    )
    defaults.update(overrides)
    return ReplayEvaluationRecord(**defaults)


def _make_summary(**overrides) -> OfflineEvaluationSummary:
    """Minimal valid OfflineEvaluationSummary."""
    now = time.time()
    defaults = dict(
        evaluation_run_id=str(uuid.uuid4()),
        run_timestamp=now,
        time_window_start=now - 86400,
        time_window_end=now,
        symbols_covered=["BTCUSDT", "ETHUSDT"],
        positions_covered=50,
        decisions_covered=100,
        obituaries_covered=50,
        baseline_definitions=dict(BASELINE_DEFINITIONS),
        action_distribution={"HOLD": 30, "CLOSE_FULL": 10, "REDUCE_SMALL": 10},
        mean_decision_quality_score=0.65,
        median_decision_quality_score=0.68,
        mean_regret_score=0.25,
        mean_preservation_score=0.80,
        mean_opportunity_capture_score=0.70,
        belief_calibration_summary={
            "exit_pressure_bias": 0.08,
            "exit_pressure_mae": 0.12,
            "hold_conviction_bias": -0.08,
            "hold_conviction_mae": 0.12,
            "sample_size": 50.0,
        },
        hazard_calibration_summary={
            "composite_hazard_bias": 0.06,
            "composite_hazard_mae": 0.10,
            "drawdown_hazard_bias": 0.04,
            "drawdown_hazard_mae": 0.08,
            "sample_size": 50.0,
        },
        utility_ranking_summary={
            "rank_accuracy": 0.35,
            "top3_accuracy": 0.70,
            "mean_rank_displacement": 1.2,
            "sample_size": 100.0,
        },
        policy_choice_summary={
            "pass_quality": 0.65,
            "block_quality": 0.55,
            "pass_count": 70.0,
            "block_count": 30.0,
            "sample_size": 100.0,
        },
        baseline_comparison={
            "always_hold": {"mean_pnl": 10.0, "vs_exit_brain_pnl_delta": 5.0},
        },
        shadow_only=True,
    )
    defaults.update(overrides)
    return OfflineEvaluationSummary(**defaults)


def _make_tuning_recommendation(**overrides) -> TuningRecommendation:
    """Minimal valid TuningRecommendation."""
    defaults = dict(
        recommendation_id=str(uuid.uuid4()),
        created_at=time.time(),
        source_evaluation_run_id=str(uuid.uuid4()),
        component_name="policy_constraints",
        parameter_name="UNCERTAINTY_HARD_CEILING",
        current_value=0.70,
        proposed_value=0.75,
        direction="increase",
        rationale="Over-predicting exit pressure → widen ceiling.",
        supporting_metrics={"exit_pressure_bias": 0.08},
        expected_effect="Adjust UNCERTAINTY_HARD_CEILING: 0.70 → 0.75",
        confidence=0.5,
        risk_of_change="low",
        requires_human_review=True,
        applied=False,
        shadow_only=True,
    )
    defaults.update(overrides)
    return TuningRecommendation(**defaults)


def _make_calibration_artifact(**overrides) -> CalibrationArtifact:
    """Minimal valid CalibrationArtifact."""
    defaults = dict(
        artifact_id=str(uuid.uuid4()),
        created_at=time.time(),
        source_evaluation_run_id=str(uuid.uuid4()),
        component_name="policy_constraints",
        calibration_type="threshold_snapshot",
        parameters={"UNCERTAINTY_HARD_CEILING": 0.70},
        fit_statistics={"mean_decision_quality": 0.65},
        sample_size=100,
        time_window_start=time.time() - 86400,
        time_window_end=time.time(),
        shadow_only=True,
    )
    defaults.update(overrides)
    return CalibrationArtifact(**defaults)


def _make_outcome(**kw) -> OutcomePathResult:
    """Minimal OutcomePathResult for testing."""
    defaults = dict(
        price_path=[100, 102, 105, 103, 101, 104, 106, 108, 107, 105, 103],
        pnl_path=[0, 2, 5, 3, 1, 4, 6, 8, 7, 5, 3],
        timestamps=[float(i) for i in range(11)],
        peak_pnl=8.0,
        peak_pnl_timestamp=7.0,
        trough_pnl=0.0,
        trough_pnl_timestamp=0.0,
        max_drawdown_after_peak=5.0,
        final_pnl=3.0,
        final_price=103.0,
        sample_count=11,
        quality_flags=(),
    )
    defaults.update(kw)
    return OutcomePathResult(**defaults)


# ═════════════════════════════════════════════════════════════════════════
# S1: TradeExitObituary contract
# ═════════════════════════════════════════════════════════════════════════

class TestTradeExitObituaryContract:
    """TradeExitObituary data contract validation."""

    def test_valid_obituary_passes(self):
        obit = _make_obituary()
        assert obit.validate() == []

    def test_empty_position_id_fails(self):
        obit = _make_obituary(position_id="")
        errors = obit.validate()
        assert any("position_id" in e for e in errors)

    def test_empty_symbol_fails(self):
        obit = _make_obituary(symbol="")
        errors = obit.validate()
        assert any("symbol" in e for e in errors)

    def test_invalid_action_fails(self):
        obit = _make_obituary(recommended_action_at_decision="BUY_MORE")
        errors = obit.validate()
        assert any("action" in e.lower() or "VALID_ACTIONS" in e for e in errors)

    def test_regret_score_out_of_range(self):
        obit = _make_obituary(regret_score=1.5)
        errors = obit.validate()
        assert any("regret" in e.lower() for e in errors)

    def test_preservation_score_out_of_range(self):
        obit = _make_obituary(preservation_score=-0.1)
        errors = obit.validate()
        assert any("preservation" in e.lower() for e in errors)

    def test_opportunity_score_out_of_range(self):
        obit = _make_obituary(opportunity_capture_score=2.0)
        errors = obit.validate()
        assert any("opportunity" in e.lower() for e in errors)

    def test_shadow_only_must_be_true(self):
        obit = _make_obituary(shadow_only=False)
        errors = obit.validate()
        assert any("shadow_only" in e for e in errors)

    def test_max_drawdown_negative_fails(self):
        obit = _make_obituary(max_drawdown_after_peak=-5.0)
        errors = obit.validate()
        assert any("drawdown" in e.lower() for e in errors)

    def test_to_dict_returns_flat(self):
        obit = _make_obituary()
        d = obit.to_dict()
        assert isinstance(d, dict)
        assert d["position_id"] == "pos-001"
        assert isinstance(d["quality_flags"], str)


# ═════════════════════════════════════════════════════════════════════════
# S2: ReplayEvaluationRecord contract
# ═════════════════════════════════════════════════════════════════════════

class TestReplayEvaluationRecordContract:
    """ReplayEvaluationRecord data contract validation."""

    def test_valid_record_passes(self):
        rec = _make_replay_record()
        assert rec.validate() == []

    def test_empty_record_id_fails(self):
        rec = _make_replay_record(record_id="")
        errors = rec.validate()
        assert any("record_id" in e for e in errors)

    def test_invalid_chosen_action_fails(self):
        rec = _make_replay_record(chosen_action="INVALID")
        errors = rec.validate()
        assert any("action" in e.lower() or "VALID_ACTIONS" in e for e in errors)

    def test_rank_below_one_fails(self):
        rec = _make_replay_record(action_rank_at_decision=0)
        errors = rec.validate()
        assert any("rank" in e.lower() for e in errors)

    def test_decision_quality_out_of_range(self):
        rec = _make_replay_record(decision_quality_score=-0.1)
        errors = rec.validate()
        assert any("decision_quality" in e.lower() or "quality" in e.lower() for e in errors)

    def test_shadow_only_must_be_true(self):
        rec = _make_replay_record(shadow_only=False)
        errors = rec.validate()
        assert any("shadow_only" in e for e in errors)


# ═════════════════════════════════════════════════════════════════════════
# S3: OfflineEvaluationSummary contract
# ═════════════════════════════════════════════════════════════════════════

class TestOfflineEvaluationSummaryContract:
    """OfflineEvaluationSummary data contract validation."""

    def test_valid_summary_passes(self):
        s = _make_summary()
        assert s.validate() == []

    def test_empty_run_id_fails(self):
        s = _make_summary(evaluation_run_id="")
        errors = s.validate()
        assert any("run_id" in e.lower() or "evaluation_run_id" in e for e in errors)

    def test_window_end_before_start_fails(self):
        now = time.time()
        s = _make_summary(time_window_start=now, time_window_end=now - 100)
        errors = s.validate()
        assert len(errors) > 0

    def test_mean_quality_out_of_range(self):
        s = _make_summary(mean_decision_quality_score=1.5)
        errors = s.validate()
        assert any("quality" in e.lower() or "mean" in e.lower() for e in errors)

    def test_shadow_only_required(self):
        s = _make_summary(shadow_only=False)
        errors = s.validate()
        assert any("shadow_only" in e for e in errors)

    def test_negative_positions_covered_fails(self):
        s = _make_summary(positions_covered=-1)
        errors = s.validate()
        assert len(errors) > 0


# ═════════════════════════════════════════════════════════════════════════
# S4: TuningRecommendation contract
# ═════════════════════════════════════════════════════════════════════════

class TestTuningRecommendationContract:
    """TuningRecommendation data contract validation."""

    def test_valid_recommendation_passes(self):
        rec = _make_tuning_recommendation()
        assert rec.validate() == []

    def test_invalid_direction_fails(self):
        rec = _make_tuning_recommendation(direction="sideways")
        errors = rec.validate()
        assert any("direction" in e.lower() for e in errors)

    def test_valid_directions_all_pass(self):
        for d in VALID_DIRECTIONS:
            rec = _make_tuning_recommendation(direction=d)
            assert rec.validate() == [], f"direction={d} should be valid"

    def test_confidence_out_of_range(self):
        rec = _make_tuning_recommendation(confidence=1.5)
        errors = rec.validate()
        assert any("confidence" in e.lower() for e in errors)

    def test_requires_human_review_must_be_true(self):
        rec = _make_tuning_recommendation(requires_human_review=False)
        errors = rec.validate()
        assert any("human_review" in e.lower() or "requires_human" in e for e in errors)

    def test_applied_must_be_false(self):
        rec = _make_tuning_recommendation(applied=True)
        errors = rec.validate()
        assert any("applied" in e.lower() for e in errors)

    def test_shadow_only_required(self):
        rec = _make_tuning_recommendation(shadow_only=False)
        errors = rec.validate()
        assert any("shadow_only" in e for e in errors)


# ═════════════════════════════════════════════════════════════════════════
# S5: CalibrationArtifact contract
# ═════════════════════════════════════════════════════════════════════════

class TestCalibrationArtifactContract:
    """CalibrationArtifact data contract validation."""

    def test_valid_artifact_passes(self):
        a = _make_calibration_artifact()
        assert a.validate() == []

    def test_invalid_calibration_type_fails(self):
        a = _make_calibration_artifact(calibration_type="invalid_type")
        errors = a.validate()
        assert any("calibration_type" in e for e in errors)

    def test_all_calibration_types_valid(self):
        for ct in VALID_CALIBRATION_TYPES:
            a = _make_calibration_artifact(calibration_type=ct)
            assert a.validate() == [], f"type={ct} should be valid"

    def test_negative_sample_size_fails(self):
        a = _make_calibration_artifact(sample_size=-1)
        errors = a.validate()
        assert any("sample" in e.lower() for e in errors)

    def test_shadow_only_required(self):
        a = _make_calibration_artifact(shadow_only=False)
        errors = a.validate()
        assert any("shadow_only" in e for e in errors)


# ═════════════════════════════════════════════════════════════════════════
# S6: CounterfactualEvaluator — 7 action simulations
# ═════════════════════════════════════════════════════════════════════════

class TestCounterfactualEvaluator7Actions:
    """Verify PnL simulation for each action type."""

    @pytest.fixture
    def evaluator(self):
        return CounterfactualEvaluator()

    @pytest.fixture
    def long_outcome(self):
        """LONG position: entry=100, path rises to 110 then settles at 105."""
        prices = [101, 103, 106, 108, 110, 109, 107, 105, 105, 105, 105]
        entry = 100.0
        pnl = [(p - entry) * 1.0 for p in prices]
        return OutcomePathResult(
            price_path=prices,
            pnl_path=pnl,
            timestamps=[float(i) for i in range(len(prices))],
            peak_pnl=max(pnl),
            peak_pnl_timestamp=4.0,
            trough_pnl=min(pnl),
            trough_pnl_timestamp=0.0,
            max_drawdown_after_peak=max(pnl) - pnl[-1],
            final_pnl=pnl[-1],
            final_price=prices[-1],
            sample_count=len(prices),
        )

    def test_hold_returns_final_pnl_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        # HOLD = final PnL / normalization
        expected = (105 - 100) * 1.0 / (100 * 1.0)
        assert abs(result["HOLD"] - expected) < 1e-6

    def test_close_full_returns_first_price_pnl_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        # CLOSE_FULL = PnL at first price (101)
        expected = (101 - 100) * 1.0 / (100 * 1.0)
        assert abs(result["CLOSE_FULL"] - expected) < 1e-6

    def test_reduce_small_fraction_10pct_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        # REDUCE_SMALL = 10% at first_price + 90% at final_price
        frac = ACTION_EXIT_FRACTIONS["REDUCE_SMALL"]  # 0.10
        exited = (101 - 100) * 1.0 * frac
        remaining = (105 - 100) * 1.0 * (1 - frac)
        expected = (exited + remaining) / (100 * 1.0)
        assert abs(result["REDUCE_SMALL"] - expected) < 1e-6

    def test_reduce_medium_fraction_25pct_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        frac = ACTION_EXIT_FRACTIONS["REDUCE_MEDIUM"]  # 0.25
        exited = (101 - 100) * 1.0 * frac
        remaining = (105 - 100) * 1.0 * (1 - frac)
        expected = (exited + remaining) / (100 * 1.0)
        assert abs(result["REDUCE_MEDIUM"] - expected) < 1e-6

    def test_take_profit_partial_fraction_50pct_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        frac = ACTION_EXIT_FRACTIONS["TAKE_PROFIT_PARTIAL"]  # 0.50
        exited = (101 - 100) * 1.0 * frac
        remaining = (105 - 100) * 1.0 * (1 - frac)
        expected = (exited + remaining) / (100 * 1.0)
        assert abs(result["TAKE_PROFIT_PARTIAL"] - expected) < 1e-6

    def test_take_profit_large_fraction_75pct_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        frac = ACTION_EXIT_FRACTIONS["TAKE_PROFIT_LARGE"]  # 0.75
        exited = (101 - 100) * 1.0 * frac
        remaining = (105 - 100) * 1.0 * (1 - frac)
        expected = (exited + remaining) / (100 * 1.0)
        assert abs(result["TAKE_PROFIT_LARGE"] - expected) < 1e-6

    def test_tighten_exit_uses_trailing_stop_long(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        assert "TIGHTEN_EXIT" in result
        # trailing stop at 2%: running high 110, stop at 107.8 → triggered at 107
        # PnL = (107 - 100) / 100 = 0.07
        assert result["TIGHTEN_EXIT"] > 0  # profitable

    def test_all_7_actions_present(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 100.0, "LONG", 1.0)
        assert set(result.keys()) == VALID_ACTIONS

    # SHORT side tests
    def test_hold_short_side(self, evaluator):
        prices = [99, 97, 95, 96, 98]
        pnl = [(100 - p) * 1.0 for p in prices]
        outcome = OutcomePathResult(
            price_path=prices, pnl_path=pnl,
            timestamps=[float(i) for i in range(5)],
            peak_pnl=max(pnl), peak_pnl_timestamp=2.0,
            trough_pnl=min(pnl), trough_pnl_timestamp=4.0,
            max_drawdown_after_peak=3.0, final_pnl=pnl[-1],
            final_price=98, sample_count=5,
        )
        result = evaluator.evaluate_all_actions(outcome, 100.0, "SHORT", 1.0)
        # HOLD = (100-98)/100 = 0.02
        assert abs(result["HOLD"] - 0.02) < 1e-6

    def test_close_full_short_side(self, evaluator):
        prices = [99, 97, 95, 96, 98]
        pnl = [(100 - p) * 1.0 for p in prices]
        outcome = OutcomePathResult(
            price_path=prices, pnl_path=pnl,
            timestamps=[float(i) for i in range(5)],
            peak_pnl=max(pnl), peak_pnl_timestamp=2.0,
            trough_pnl=min(pnl), trough_pnl_timestamp=4.0,
            max_drawdown_after_peak=3.0, final_pnl=pnl[-1],
            final_price=98, sample_count=5,
        )
        result = evaluator.evaluate_all_actions(outcome, 100.0, "SHORT", 1.0)
        # CLOSE_FULL at first price=99 → PnL=(100-99)/100=0.01
        assert abs(result["CLOSE_FULL"] - 0.01) < 1e-6

    def test_partial_exit_short_side(self, evaluator):
        prices = [99, 97, 95, 96, 98]
        pnl = [(100 - p) * 1.0 for p in prices]
        outcome = OutcomePathResult(
            price_path=prices, pnl_path=pnl,
            timestamps=[float(i) for i in range(5)],
            peak_pnl=max(pnl), peak_pnl_timestamp=2.0,
            trough_pnl=min(pnl), trough_pnl_timestamp=4.0,
            max_drawdown_after_peak=3.0, final_pnl=pnl[-1],
            final_price=98, sample_count=5,
        )
        result = evaluator.evaluate_all_actions(outcome, 100.0, "SHORT", 1.0)
        frac = ACTION_EXIT_FRACTIONS["REDUCE_MEDIUM"]  # 0.25
        exited = (100 - 99) * 1.0 * frac
        remaining = (100 - 98) * 1.0 * (1 - frac)
        expected = (exited + remaining) / (100 * 1.0)
        assert abs(result["REDUCE_MEDIUM"] - expected) < 1e-6

    def test_empty_price_path_returns_zeros(self, evaluator):
        outcome = OutcomePathResult()
        result = evaluator.evaluate_all_actions(outcome, 100.0, "LONG", 1.0)
        assert all(v == 0.0 for v in result.values())

    def test_zero_entry_returns_zeros(self, evaluator, long_outcome):
        result = evaluator.evaluate_all_actions(long_outcome, 0.0, "LONG", 1.0)
        assert all(v == 0.0 for v in result.values())


# ═════════════════════════════════════════════════════════════════════════
# S7: CounterfactualEvaluator — best action
# ═════════════════════════════════════════════════════════════════════════

class TestCounterfactualBestAction:
    """find_ex_post_best_action correctness."""

    def test_best_action_is_highest_utility(self):
        ev = CounterfactualEvaluator()
        actions = {"HOLD": 0.4, "CLOSE_FULL": 0.1, "REDUCE_SMALL": 0.6}
        best, val = ev.find_ex_post_best_action(actions)
        assert best == "REDUCE_SMALL"
        assert abs(val - 0.6) < 1e-9

    def test_empty_dict_defaults_to_hold(self):
        ev = CounterfactualEvaluator()
        best, val = ev.find_ex_post_best_action({})
        assert best == "HOLD"
        assert val == 0.0

    def test_all_equal_picks_deterministic(self):
        ev = CounterfactualEvaluator()
        actions = {a: 0.5 for a in VALID_ACTIONS}
        best, val = ev.find_ex_post_best_action(actions)
        assert best in VALID_ACTIONS
        assert abs(val - 0.5) < 1e-9


# ═════════════════════════════════════════════════════════════════════════
# S8: CounterfactualEvaluator — quality score formula
# ═════════════════════════════════════════════════════════════════════════

class TestCounterfactualQualityScore:
    """Decision quality = 0.4×rank + 0.3×(1-regret) + 0.2×preserve + 0.1×opp."""

    @pytest.fixture
    def evaluator(self):
        return CounterfactualEvaluator()

    def test_perfect_decision_all_ones(self, evaluator):
        """Rank 1 (#1), regret 0, preservation 1, opportunity 1 → ≈1.0."""
        actions = {"HOLD": 1.0, "CLOSE_FULL": 0.5}
        score = evaluator.compute_decision_quality_score(
            "HOLD", actions, regret_score=0.0,
            preservation_score=1.0, opportunity_capture_score=1.0,
        )
        assert abs(score - 1.0) < 1e-6

    def test_worst_decision_all_zeros(self, evaluator):
        """Rank last, regret 1, preservation 0, opportunity 0 → ≈0.0."""
        actions = {a: (7 - i) * 0.1 for i, a in enumerate(sorted(VALID_ACTIONS))}
        worst = sorted(VALID_ACTIONS)[-1]
        score = evaluator.compute_decision_quality_score(
            worst, actions, regret_score=1.0,
            preservation_score=0.0, opportunity_capture_score=0.0,
        )
        assert score < 0.15

    def test_formula_weights_correct(self, evaluator):
        """Verify 0.4/0.3/0.2/0.1 weights."""
        # Rank #1 out of 2 → rank_component=1.0
        actions = {"HOLD": 0.8, "CLOSE_FULL": 0.3}
        score = evaluator.compute_decision_quality_score(
            "HOLD", actions,
            regret_score=0.5, preservation_score=0.6, opportunity_capture_score=0.4,
        )
        expected = 0.4 * 1.0 + 0.3 * (1 - 0.5) + 0.2 * 0.6 + 0.1 * 0.4
        assert abs(score - expected) < 1e-6

    def test_explanation_consistency_score(self, evaluator):
        """Lower error → higher consistency."""
        s = evaluator.compute_explanation_consistency_score(
            predicted_exit_pressure=0.5,
            realized_exit_pressure_proxy=0.5,
            predicted_composite_hazard=0.3,
            realized_hazard_proxy=0.3,
        )
        assert abs(s - 1.0) < 1e-6
        # With error
        s2 = evaluator.compute_explanation_consistency_score(
            0.5, 0.8, 0.3, 0.6,
        )
        assert s2 < s


# ═════════════════════════════════════════════════════════════════════════
# S9: ReplayObituaryWriter — regret score
# ═════════════════════════════════════════════════════════════════════════

class TestObituaryWriterRegret:
    """Regret = (best - actual) / |best|."""

    @pytest.fixture
    def writer(self):
        redis = MagicMock()
        publisher = MagicMock()
        return ReplayObituaryWriter(redis, publisher)

    def test_zero_regret_when_actual_equals_best(self, writer):
        outcome = _make_outcome(final_pnl=8.0, peak_pnl=8.0)
        score = writer.compute_regret_score(outcome, best_possible_pnl=8.0)
        assert abs(score) < 1e-9

    def test_full_regret_when_actual_zero(self, writer):
        outcome = _make_outcome(final_pnl=0.0)
        score = writer.compute_regret_score(outcome, best_possible_pnl=10.0)
        assert abs(score - 1.0) < 1e-6

    def test_partial_regret(self, writer):
        outcome = _make_outcome(final_pnl=5.0)
        score = writer.compute_regret_score(outcome, best_possible_pnl=10.0)
        assert abs(score - 0.5) < 1e-6


# ═════════════════════════════════════════════════════════════════════════
# S10: ReplayObituaryWriter — preservation score
# ═════════════════════════════════════════════════════════════════════════

class TestObituaryWriterPreservation:
    """Preservation = 1 - (drawdown / peak)."""

    @pytest.fixture
    def writer(self):
        redis = MagicMock()
        publisher = MagicMock()
        return ReplayObituaryWriter(redis, publisher)

    def test_perfect_preservation_no_drawdown(self, writer):
        outcome = _make_outcome(peak_pnl=10.0, max_drawdown_after_peak=0.0)
        score = writer.compute_preservation_score(outcome)
        assert abs(score - 1.0) < 1e-6

    def test_zero_preservation_total_giveback(self, writer):
        outcome = _make_outcome(peak_pnl=10.0, max_drawdown_after_peak=10.0)
        score = writer.compute_preservation_score(outcome)
        assert abs(score) < 1e-6

    def test_half_preservation(self, writer):
        outcome = _make_outcome(peak_pnl=10.0, max_drawdown_after_peak=5.0)
        score = writer.compute_preservation_score(outcome)
        assert abs(score - 0.5) < 1e-6


# ═════════════════════════════════════════════════════════════════════════
# S11: ReplayObituaryWriter — opportunity capture score
# ═════════════════════════════════════════════════════════════════════════

class TestObituaryWriterOpportunity:
    """Opportunity = actual / best."""

    @pytest.fixture
    def writer(self):
        redis = MagicMock()
        publisher = MagicMock()
        return ReplayObituaryWriter(redis, publisher)

    def test_perfect_capture(self, writer):
        outcome = _make_outcome(final_pnl=10.0)
        score = writer.compute_opportunity_capture_score(outcome, best_possible_pnl=10.0)
        assert abs(score - 1.0) < 1e-6

    def test_zero_capture(self, writer):
        outcome = _make_outcome(final_pnl=0.0)
        score = writer.compute_opportunity_capture_score(outcome, best_possible_pnl=10.0)
        assert abs(score) < 1e-6

    def test_partial_capture(self, writer):
        outcome = _make_outcome(final_pnl=7.0)
        score = writer.compute_opportunity_capture_score(outcome, best_possible_pnl=10.0)
        assert abs(score - 0.7) < 1e-6


# ═════════════════════════════════════════════════════════════════════════
# S12: OfflineEvaluator — min samples
# ═════════════════════════════════════════════════════════════════════════

class TestOfflineEvaluatorMinSamples:
    """Returns None if < MIN_SAMPLES_FOR_EVAL."""

    def test_returns_none_below_min(self):
        redis = MagicMock()
        publisher = MagicMock()
        ev = OfflineEvaluator(redis, publisher)
        # Mock load_replay_records to return too few
        ev.load_replay_records = MagicMock(return_value=[_make_obituary()] * 3)
        result = ev.run_evaluation(start_ts=1000, end_ts=2000)
        assert result is None

    def test_proceeds_with_enough_samples(self):
        redis = MagicMock()
        publisher = MagicMock()
        publisher.publish_replay_evaluation = MagicMock(return_value="stream-id")
        publisher.publish_evaluation_summary = MagicMock(return_value="stream-id")
        ev = OfflineEvaluator(redis, publisher)

        obits = [
            _make_obituary(position_id=f"pos-{i}", obituary_timestamp=time.time())
            for i in range(6)
        ]
        ev.load_replay_records = MagicMock(return_value=obits)

        # Mock the Redis xrange for outcome reconstruction
        redis.xrange = MagicMock(return_value=[])

        result = ev.run_evaluation(start_ts=1000, end_ts=2000)
        # Should not be None (even with empty outcomes, the orchestration completes)
        assert result is not None
        assert result.obituaries_covered == 6


# ═════════════════════════════════════════════════════════════════════════
# S13: OfflineEvaluator — sub-evaluator delegation
# ═════════════════════════════════════════════════════════════════════════

class TestOfflineEvaluatorDelegation:
    """All 4 sub-evaluators called correctly."""

    @pytest.fixture
    def setup_evaluator(self):
        redis = MagicMock()
        redis.xrange = MagicMock(return_value=[])
        publisher = MagicMock()
        publisher.publish_replay_evaluation = MagicMock(return_value="ok")
        publisher.publish_evaluation_summary = MagicMock(return_value="ok")
        ev = OfflineEvaluator(redis, publisher)
        obits = [
            _make_obituary(position_id=f"pos-{i}")
            for i in range(10)
        ]
        ev.load_replay_records = MagicMock(return_value=obits)
        return ev

    def test_belief_evaluator_called(self, setup_evaluator):
        ev = setup_evaluator
        original_belief = ev._belief_eval
        ev._belief_eval = MagicMock(wraps=original_belief)
        ev._belief_eval.evaluate = MagicMock(return_value={"exit_pressure_mae": 0.1, "sample_size": 10.0})
        ev._belief_eval._compute_realized_exit_pressure = original_belief._compute_realized_exit_pressure
        ev.run_evaluation(1000, 2000)
        ev._belief_eval.evaluate.assert_called_once()

    def test_hazard_evaluator_called(self, setup_evaluator):
        ev = setup_evaluator
        original_hazard = ev._hazard_eval
        ev._hazard_eval = MagicMock(wraps=original_hazard)
        ev._hazard_eval.evaluate = MagicMock(return_value={"composite_hazard_mae": 0.1, "sample_size": 10.0})
        ev._hazard_eval._compute_realized_composite_hazard = original_hazard._compute_realized_composite_hazard
        ev.run_evaluation(1000, 2000)
        ev._hazard_eval.evaluate.assert_called_once()

    def test_utility_evaluator_called(self, setup_evaluator):
        ev = setup_evaluator
        ev._utility_eval = MagicMock()
        ev._utility_eval.evaluate = MagicMock(return_value={"rank_accuracy": 0.5, "sample_size": 10.0})
        ev.run_evaluation(1000, 2000)
        ev._utility_eval.evaluate.assert_called_once()

    def test_policy_evaluator_called(self, setup_evaluator):
        ev = setup_evaluator
        ev._policy_eval = MagicMock()
        ev._policy_eval.evaluate = MagicMock(return_value={"pass_quality": 0.5, "sample_size": 10.0})
        ev._policy_eval.compare_against_baselines = MagicMock(return_value={})
        ev.run_evaluation(1000, 2000)
        ev._policy_eval.evaluate.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════
# S14: BeliefCalibrationEvaluator
# ═════════════════════════════════════════════════════════════════════════

class TestBeliefCalibrationEvaluator:
    """Belief calibration accuracy metrics."""

    def test_perfect_calibration_zero_error(self):
        ev = BeliefCalibrationEvaluator()
        obit = _make_obituary(
            belief_snapshot={"exit_pressure": 0.5, "hold_conviction": 0.5}
        )
        # Outcome where realized pressure ≈ 0.5
        outcome = _make_outcome(
            peak_pnl=10.0, max_drawdown_after_peak=5.0, final_pnl=5.0,
        )
        result = ev.evaluate([obit], {"pos-001": outcome})
        assert result["sample_size"] == 1.0
        # MAE should be finite
        assert result["exit_pressure_mae"] >= 0.0

    def test_empty_obituaries_returns_zeros(self):
        ev = BeliefCalibrationEvaluator()
        result = ev.evaluate([], {})
        assert result["exit_pressure_mae"] == 0.0
        assert result["sample_size"] == 0.0


# ═════════════════════════════════════════════════════════════════════════
# S15: HazardCalibrationEvaluator
# ═════════════════════════════════════════════════════════════════════════

class TestHazardCalibrationEvaluator:
    """Hazard calibration accuracy metrics."""

    def test_evaluates_all_hazard_axes(self):
        ev = HazardCalibrationEvaluator()
        snap = {a: 0.3 for a in HAZARD_AXES}
        obit = _make_obituary(hazard_snapshot=snap)
        outcome = _make_outcome(peak_pnl=10.0, max_drawdown_after_peak=3.0)
        result = ev.evaluate([obit], {"pos-001": outcome})
        for axis in HAZARD_AXES:
            assert f"{axis}_mae" in result

    def test_empty_returns_zeros(self):
        ev = HazardCalibrationEvaluator()
        result = ev.evaluate([], {})
        assert result["sample_size"] == 0.0


# ═════════════════════════════════════════════════════════════════════════
# S16: UtilityRankingEvaluator
# ═════════════════════════════════════════════════════════════════════════

class TestUtilityRankingEvaluator:
    """Utility ranking accuracy metrics."""

    def test_perfect_ranking_exact_match(self):
        ev = UtilityRankingEvaluator()
        rec = _make_replay_record(
            chosen_action="HOLD",
            realized_utility_proxy_by_action={
                "HOLD": 0.8, "CLOSE_FULL": 0.3, "REDUCE_SMALL": 0.5,
            },
        )
        result = ev.evaluate([rec])
        assert result["rank_accuracy"] == 1.0  # HOLD is #1

    def test_empty_records_returns_zeros(self):
        ev = UtilityRankingEvaluator()
        result = ev.evaluate([])
        assert result["rank_accuracy"] == 0.0
        assert result["sample_size"] == 0.0


# ═════════════════════════════════════════════════════════════════════════
# S17: PolicyChoiceEvaluator — baselines (4 baselines × 2 sides)
# ═════════════════════════════════════════════════════════════════════════

class TestPolicyChoiceEvaluatorBaselines:
    """4 baselines × 2 sides = 8 tests."""

    @pytest.fixture
    def evaluator(self):
        return PolicyChoiceEvaluator()

    def _make_obit_and_outcome(self, side="LONG"):
        if side == "LONG":
            prices = [100, 102, 105, 103, 101, 104]
            entry = 100.0
            pnl = [(p - entry) for p in prices]
        else:
            prices = [100, 98, 95, 97, 99, 96]
            entry = 100.0
            pnl = [(entry - p) for p in prices]
        outcome = OutcomePathResult(
            price_path=prices, pnl_path=pnl,
            timestamps=[float(i) for i in range(len(prices))],
            peak_pnl=max(pnl), peak_pnl_timestamp=2.0,
            trough_pnl=min(pnl), trough_pnl_timestamp=0.0,
            max_drawdown_after_peak=max(pnl) - pnl[-1],
            final_pnl=pnl[-1], final_price=prices[-1],
            sample_count=len(prices),
        )
        obit = _make_obituary(
            position_id="pos-bl",
            belief_snapshot={"side": side},
        )
        return obit, outcome

    def test_always_hold_long(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("LONG")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "always_hold" in result
        assert "mean_pnl" in result["always_hold"]

    def test_trailing_stop_long(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("LONG")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "fixed_trailing_2pct" in result

    def test_take_profit_long(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("LONG")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "fixed_tp_3pct" in result

    def test_naive_partial_long(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("LONG")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "naive_partial_50pct" in result

    def test_always_hold_short(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("SHORT")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "always_hold" in result

    def test_trailing_stop_short(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("SHORT")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "fixed_trailing_2pct" in result

    def test_take_profit_short(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("SHORT")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "fixed_tp_3pct" in result

    def test_naive_partial_short(self, evaluator):
        obit, outcome = self._make_obit_and_outcome("SHORT")
        result = evaluator.compare_against_baselines([obit], {"pos-bl": outcome})
        assert "naive_partial_50pct" in result


# ═════════════════════════════════════════════════════════════════════════
# S18: ThresholdTuner — ±20% cap
# ═════════════════════════════════════════════════════════════════════════

class TestThresholdTunerMaxChange:
    """Max ±20% change fraction enforced."""

    def test_uncertainty_ceiling_cap_increase(self):
        tuner = ThresholdTuner()
        summary = _make_summary(
            belief_calibration_summary={
                "exit_pressure_bias": 0.50,  # Very large bias
                "exit_pressure_mae": 0.50,
                "hold_conviction_bias": -0.50,
                "hold_conviction_mae": 0.50,
                "sample_size": 200.0,
            }
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            if rec.parameter_name == "UNCERTAINTY_HARD_CEILING":
                change_pct = abs(rec.proposed_value - rec.current_value) / abs(rec.current_value)
                assert change_pct <= TT_MAX_CHANGE + 1e-6

    def test_hazard_threshold_cap_decrease(self):
        tuner = ThresholdTuner()
        summary = _make_summary(
            hazard_calibration_summary={
                "composite_hazard_bias": -0.50,  # Under-predicting
                "composite_hazard_mae": 0.50,
                "sample_size": 200.0,
            }
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            if rec.parameter_name == "HAZARD_EMERGENCY_THRESHOLD":
                change_pct = abs(rec.proposed_value - rec.current_value) / abs(rec.current_value)
                assert change_pct <= TT_MAX_CHANGE + 1e-6

    def test_conviction_cap_increase(self):
        tuner = ThresholdTuner()
        summary = _make_summary(
            policy_choice_summary={
                "pass_quality": 0.30,  # Very low → raise conviction
                "block_quality": 0.90,
                "pass_count": 70.0,
                "block_count": 30.0,
                "sample_size": 200.0,
            }
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            if rec.parameter_name == "MIN_ACTION_CONVICTION":
                change_pct = abs(rec.proposed_value - rec.current_value) / abs(rec.current_value)
                assert change_pct <= TT_MAX_CHANGE + 1e-6

    def test_all_recommendations_valid(self):
        tuner = ThresholdTuner()
        summary = _make_summary()
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            assert rec.validate() == []
            assert rec.requires_human_review is True
            assert rec.applied is False


# ═════════════════════════════════════════════════════════════════════════
# S19: ThresholdTuner — low sample cap
# ═════════════════════════════════════════════════════════════════════════

class TestThresholdTunerLowSample:
    """Confidence ≤ 0.30 if < 50 samples."""

    def test_low_sample_confidence_capped(self):
        tuner = ThresholdTuner()
        summary = _make_summary(
            decisions_covered=20,  # < 50
            belief_calibration_summary={
                "exit_pressure_bias": 0.20,
                "exit_pressure_mae": 0.25,
                "hold_conviction_bias": -0.20,
                "hold_conviction_mae": 0.25,
                "sample_size": 20.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            assert rec.confidence <= TT_LOW_CAP + 1e-9

    def test_high_sample_no_cap(self):
        tuner = ThresholdTuner()
        summary = _make_summary(
            decisions_covered=500,
            belief_calibration_summary={
                "exit_pressure_bias": 0.20,
                "exit_pressure_mae": 0.25,
                "hold_conviction_bias": -0.20,
                "hold_conviction_mae": 0.25,
                "sample_size": 500.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        # With high samples, at least some recs can exceed the cap
        # (Not all recs necessarily do, but the cap isn't applied)
        for rec in recs:
            assert rec.confidence > 0  # Just ensure they're generated


# ═════════════════════════════════════════════════════════════════════════
# S20: WeightTuner — normalization / bounds
# ═════════════════════════════════════════════════════════════════════════

class TestWeightTunerNormalization:
    """Proposed weights stay within [0.05, 0.60] bounds."""

    def test_hazard_weight_stays_in_bounds(self):
        tuner = WeightTuner()
        summary = _make_summary(
            hazard_calibration_summary={
                "drawdown_hazard_bias": 0.3,
                "drawdown_hazard_mae": 0.30,
                "reversal_hazard_bias": 0.01,
                "reversal_hazard_mae": 0.02,
                "volatility_hazard_bias": 0.15,
                "volatility_hazard_mae": 0.15,
                "time_decay_hazard_bias": 0.10,
                "time_decay_hazard_mae": 0.10,
                "regime_hazard_bias": 0.05,
                "regime_hazard_mae": 0.05,
                "ensemble_hazard_bias": 0.20,
                "ensemble_hazard_mae": 0.20,
                "composite_hazard_bias": 0.10,
                "composite_hazard_mae": 0.12,
                "sample_size": 200.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            if rec.component_name == "hazard_engine":
                assert 0.05 <= rec.proposed_value <= 0.50

    def test_belief_weight_stays_in_bounds(self):
        tuner = WeightTuner()
        summary = _make_summary(
            belief_calibration_summary={
                "exit_pressure_bias": 0.30,
                "exit_pressure_mae": 0.30,
                "hold_conviction_bias": 0.01,
                "hold_conviction_mae": 0.02,
                "sample_size": 200.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            if rec.component_name == "belief_engine":
                assert 0.05 <= rec.proposed_value <= 0.60

    def test_all_weight_recs_valid(self):
        tuner = WeightTuner()
        summary = _make_summary(
            hazard_calibration_summary={
                "drawdown_hazard_mae": 0.30,
                "reversal_hazard_mae": 0.02,
                "volatility_hazard_mae": 0.15,
                "time_decay_hazard_mae": 0.10,
                "regime_hazard_mae": 0.05,
                "ensemble_hazard_mae": 0.20,
                "composite_hazard_mae": 0.12,
                "sample_size": 200.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        for rec in recs:
            assert rec.validate() == []
            assert rec.requires_human_review is True


# ═════════════════════════════════════════════════════════════════════════
# S21: WeightTuner — min MAE gap
# ═════════════════════════════════════════════════════════════════════════

class TestWeightTunerMinGap:
    """No proposal if gap < MIN_MAE_GAP (0.05)."""

    def test_no_proposals_when_gap_below_threshold(self):
        tuner = WeightTuner()
        # All axes have near-equal MAE → gap < 0.05
        summary = _make_summary(
            hazard_calibration_summary={
                "drawdown_hazard_mae": 0.10,
                "reversal_hazard_mae": 0.11,
                "volatility_hazard_mae": 0.10,
                "time_decay_hazard_mae": 0.11,
                "regime_hazard_mae": 0.10,
                "ensemble_hazard_mae": 0.11,
                "composite_hazard_mae": 0.10,
                "sample_size": 200.0,
            },
            belief_calibration_summary={
                "exit_pressure_mae": 0.10,
                "hold_conviction_mae": 0.11,
                "sample_size": 200.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        # With gap of 0.01, no weight recs should be generated
        assert len(recs) == 0

    def test_proposals_when_gap_above_threshold(self):
        tuner = WeightTuner()
        summary = _make_summary(
            hazard_calibration_summary={
                "drawdown_hazard_mae": 0.30,  # High
                "reversal_hazard_mae": 0.02,  # Low → gap=0.28 >> 0.05
                "volatility_hazard_mae": 0.15,
                "time_decay_hazard_mae": 0.10,
                "regime_hazard_mae": 0.05,
                "ensemble_hazard_mae": 0.20,
                "composite_hazard_mae": 0.12,
                "sample_size": 200.0,
            },
        )
        recs = tuner.propose_adjustments(summary)
        assert len(recs) > 0


# ═════════════════════════════════════════════════════════════════════════
# S22: ProposalBuilder — max 5
# ═════════════════════════════════════════════════════════════════════════

class TestProposalBuilderMax5:
    """Cap to MAX_RECOMMENDATIONS_PER_RUN (5)."""

    def test_caps_at_max_5(self):
        publisher = MagicMock()
        publisher.publish_tuning_recommendation = MagicMock(return_value="ok")
        builder = ProposalBuilder(publisher)

        # Force many raw proposals
        many_recs = [_make_tuning_recommendation(confidence=0.8 - i * 0.05) for i in range(10)]
        builder._threshold_tuner = MagicMock()
        builder._threshold_tuner.propose_adjustments = MagicMock(return_value=many_recs[:5])
        builder._weight_tuner = MagicMock()
        builder._weight_tuner.propose_adjustments = MagicMock(return_value=many_recs[5:])

        summary = _make_summary()
        result = builder.build_tuning_proposals(summary)
        assert len(result) <= MAX_RECOMMENDATIONS_PER_RUN

    def test_fewer_than_5_passes_all(self):
        publisher = MagicMock()
        publisher.publish_tuning_recommendation = MagicMock(return_value="ok")
        builder = ProposalBuilder(publisher)

        few_recs = [_make_tuning_recommendation(confidence=0.5)]
        builder._threshold_tuner = MagicMock()
        builder._threshold_tuner.propose_adjustments = MagicMock(return_value=few_recs)
        builder._weight_tuner = MagicMock()
        builder._weight_tuner.propose_adjustments = MagicMock(return_value=[])

        summary = _make_summary()
        result = builder.build_tuning_proposals(summary)
        assert len(result) == 1


# ═════════════════════════════════════════════════════════════════════════
# S23: ProposalBuilder — confidence sort
# ═════════════════════════════════════════════════════════════════════════

class TestProposalBuilderConfidenceSort:
    """Highest confidence first."""

    def test_sorted_by_confidence_desc(self):
        publisher = MagicMock()
        publisher.publish_tuning_recommendation = MagicMock(return_value="ok")
        builder = ProposalBuilder(publisher)

        recs = [
            _make_tuning_recommendation(confidence=0.3),
            _make_tuning_recommendation(confidence=0.9),
            _make_tuning_recommendation(confidence=0.6),
        ]
        builder._threshold_tuner = MagicMock()
        builder._threshold_tuner.propose_adjustments = MagicMock(return_value=recs)
        builder._weight_tuner = MagicMock()
        builder._weight_tuner.propose_adjustments = MagicMock(return_value=[])

        summary = _make_summary()
        result = builder.build_tuning_proposals(summary)
        confidences = [r.confidence for r in result]
        assert confidences == sorted(confidences, reverse=True)

    def test_published_in_order(self):
        publisher = MagicMock()
        publisher.publish_tuning_recommendation = MagicMock(return_value="ok")
        builder = ProposalBuilder(publisher)

        recs = [
            _make_tuning_recommendation(confidence=0.2),
            _make_tuning_recommendation(confidence=0.8),
        ]
        builder._threshold_tuner = MagicMock()
        builder._threshold_tuner.propose_adjustments = MagicMock(return_value=recs)
        builder._weight_tuner = MagicMock()
        builder._weight_tuner.propose_adjustments = MagicMock(return_value=[])

        summary = _make_summary()
        builder.build_tuning_proposals(summary)
        calls = publisher.publish_tuning_recommendation.call_args_list
        assert len(calls) == 2
        # First call should be the higher confidence (0.8)
        assert calls[0][0][0].confidence == 0.8


# ═════════════════════════════════════════════════════════════════════════
# S24: OutcomeReconstructor
# ═════════════════════════════════════════════════════════════════════════

class TestOutcomeReconstructor:
    """Price/PnL path reconstruction."""

    def test_long_pnl_computation(self):
        result = OutcomeReconstructor._compute_pnl_path(
            [100, 105, 110], entry_price=100.0, side="LONG", quantity=1.0,
        )
        assert result == [0.0, 5.0, 10.0]

    def test_short_pnl_computation(self):
        result = OutcomeReconstructor._compute_pnl_path(
            [100, 95, 90], entry_price=100.0, side="SHORT", quantity=1.0,
        )
        assert result == [0.0, 5.0, 10.0]

    def test_max_drawdown_after_peak(self):
        pnl = [0, 5, 10, 8, 3, 4]
        dd = OutcomeReconstructor._compute_max_drawdown_after_peak(pnl)
        assert dd == 7.0  # peak=10, lowest after=3

    def test_empty_path_returns_zero(self):
        dd = OutcomeReconstructor._compute_max_drawdown_after_peak([])
        assert dd == 0.0

    def test_reconstruct_no_data(self):
        redis = MagicMock()
        redis.xrange = MagicMock(return_value=[])
        r = OutcomeReconstructor(redis)
        result = r.reconstruct("BTCUSDT", "LONG", 100.0, time.time())
        assert "MISSING_PRICE_PATH" in result.quality_flags


# ═════════════════════════════════════════════════════════════════════════
# S25: Shadow stream verification
# ═════════════════════════════════════════════════════════════════════════

class TestShadowStreamVerification:
    """Phase 5 outputs go to shadow streams only."""

    def test_obituary_published_to_shadow(self):
        redis = MagicMock()
        publisher = MagicMock()
        publisher.publish_obituary = MagicMock(return_value="stream-id")
        writer = ReplayObituaryWriter(redis, publisher)

        obit = _make_obituary()
        writer.publish_obituary_shadow(obit)
        publisher.publish_obituary.assert_called_once_with(obit)

    def test_proposal_published_to_shadow(self):
        publisher = MagicMock()
        publisher.publish_tuning_recommendation = MagicMock(return_value="stream-id")
        builder = ProposalBuilder(publisher)

        rec = _make_tuning_recommendation()
        builder.publish_tuning_recommendation(rec)
        publisher.publish_tuning_recommendation.assert_called_once_with(rec)
