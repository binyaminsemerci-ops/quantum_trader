"""
Phase 4 — Policy / Orchestration / Gateway: Comprehensive Test Suite.

Tests ALL Phase 4 modules:
  S1:  PolicyDecision contract
  S2:  ExitIntentCandidate contract
  S3:  ExitIntentValidationResult contract
  S4:  DecisionTrace contract
  S5:  PolicyConstraints constants
  S6:  ReasonCodes constants
  S7:  ExitPolicyEngine (7-step pipeline)
  S8:  ExitIntentGatewayValidator (schema + constraint)
  S9:  IdempotencyTracker
  S10: PayloadNormalizer
  S11: ExitAgentOrchestrator (integration, mock publisher)
  S12: Shadow stream verification
"""

from __future__ import annotations

import json
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ── Phase 4 contracts ────────────────────────────────────────────────────

from microservices.exit_brain_v1.models.policy_decision import PolicyDecision
from microservices.exit_brain_v1.models.exit_intent_candidate import ExitIntentCandidate
from microservices.exit_brain_v1.models.exit_intent_validation_result import (
    ExitIntentValidationResult,
)
from microservices.exit_brain_v1.models.decision_trace import DecisionTrace

# ── Phase 4 policy / engine / validators ─────────────────────────────────

from microservices.exit_brain_v1.policy.exit_policy_engine import ExitPolicyEngine
from microservices.exit_brain_v1.policy import policy_constraints as PC
from microservices.exit_brain_v1.policy import reason_codes as RC

from microservices.exit_brain_v1.validators.exit_intent_gateway_validator import (
    ExitIntentGatewayValidator,
    MAX_INTENT_AGE_SEC,
)
from microservices.exit_brain_v1.validators.idempotency import (
    IdempotencyTracker,
    DEFAULT_WINDOW_SEC,
    MAX_ENTRIES,
)
from microservices.exit_brain_v1.validators.payload_normalizer import (
    normalize_payload,
    check_required_fields,
)

# ── Phase 4 orchestrator ─────────────────────────────────────────────────

from microservices.exit_brain_v1.orchestrators.exit_agent_orchestrator import (
    ExitAgentOrchestrator,
)

# ── Upstream contracts (for factories) ───────────────────────────────────

from microservices.exit_brain_v1.models.position_exit_state import PositionExitState
from microservices.exit_brain_v1.models.belief_state import BeliefState
from microservices.exit_brain_v1.models.hazard_assessment import HazardAssessment
from microservices.exit_brain_v1.models.action_candidate import (
    ActionCandidate,
    VALID_ACTIONS,
    ACTION_EXIT_FRACTIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════

def _make_state(**overrides) -> PositionExitState:
    """Minimal valid PositionExitState with fresh timestamps."""
    now = time.time()
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        side="LONG",
        status="OPEN",
        entry_price=50000.0,
        current_price=51000.0,
        quantity=0.1,
        notional=5100.0,
        unrealized_pnl=100.0,
        unrealized_pnl_pct=0.02,
        open_timestamp=now - 3600,
        source_timestamps={"p33_snapshot": now - 5},
        data_quality_flags=[],
        shadow_only=True,
        peak_unrealized_pnl=120.0,
        max_adverse_excursion=30.0,
        max_favorable_excursion=150.0,
        atr=400.0,
        volatility_short=0.015,
    )
    defaults.update(overrides)
    return PositionExitState(**defaults)


def _make_belief(**overrides) -> BeliefState:
    """Minimal valid BeliefState."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        side="LONG",
        exit_pressure=0.4,
        hold_conviction=0.6,
        directional_edge=0.3,
        uncertainty_total=0.2,
        data_completeness=0.85,
        belief_timestamp=time.time(),
        belief_components={
            "ens_exit": 0.3, "rev_exit": 0.2, "geo_exit": 0.1,
            "ens_hold": 0.5, "trend_hold": 0.3, "geo_hold": 0.2,
            "ens_edge": 0.4, "regime_edge": 0.2,
            "ens_unc": 0.15, "disagreement_unc": 0.05,
            "data_unc": 0.05, "coverage": 0.9,
        },
        quality_flags=[],
        shadow_only=True,
    )
    defaults.update(overrides)
    return BeliefState(**defaults)


def _make_hazard(**overrides) -> HazardAssessment:
    """Minimal valid HazardAssessment."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        drawdown_hazard=0.3,
        reversal_hazard=0.2,
        volatility_hazard=0.25,
        time_decay_hazard=0.15,
        regime_hazard=0.2,
        ensemble_hazard=0.3,
        composite_hazard=0.233,
        dominant_hazard="drawdown",
        hazard_timestamp=time.time(),
        hazard_components={
            "drawdown": 0.3, "reversal": 0.2, "volatility": 0.25,
            "time_decay": 0.15, "regime": 0.2, "ensemble": 0.3,
        },
        quality_flags=[],
        shadow_only=True,
    )
    defaults.update(overrides)
    return HazardAssessment(**defaults)


def _make_candidates(**overrides) -> list[ActionCandidate]:
    """Build a full set of 7 scored candidates, sorted by net_utility desc."""
    now = time.time()
    base = [
        ("HOLD", 0.00, 0.50, 0.0),
        ("REDUCE_SMALL", 0.10, 0.45, 0.02),
        ("REDUCE_MEDIUM", 0.25, 0.40, 0.03),
        ("TAKE_PROFIT_PARTIAL", 0.50, 0.35, 0.0),
        ("TAKE_PROFIT_LARGE", 0.75, 0.30, 0.0),
        ("TIGHTEN_EXIT", 0.00, 0.25, 0.0),
        ("CLOSE_FULL", 1.00, 0.20, 0.05),
    ]
    candidates = []
    for action, frac, util, pen in base:
        candidates.append(ActionCandidate(
            position_id=overrides.get("position_id", "pos-001"),
            symbol=overrides.get("symbol", "BTCUSDT"),
            action=action,
            exit_fraction=frac,
            base_utility=util + pen,
            penalty_total=pen,
            net_utility=util,
            rank=0,  # assigned below
            utility_components={"base": util + pen},
            penalty_components={"total": pen},
            rationale=f"{action} utility={util:.2f}",
            scoring_timestamp=now,
            shadow_only=True,
        ))
    # Sort descending by net_utility, assign ranks
    candidates.sort(key=lambda c: c.net_utility, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1
    return candidates


def _make_policy_decision(**overrides) -> PolicyDecision:
    """Minimal valid PolicyDecision."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        decision_id=str(uuid.uuid4()),
        decision_timestamp=time.time(),
        chosen_action="HOLD",
        chosen_action_rank=1,
        chosen_action_utility=0.5,
        decision_confidence=0.7,
        decision_uncertainty=0.3,
        policy_passed=True,
        policy_blocks=[],
        policy_warnings=[],
        hazard_summary={"composite": 0.3},
        belief_summary={"exit_pressure": 0.4},
        utility_summary={"HOLD": 0.5},
        reason_codes=[],
        explanation_tags=[],
        shadow_only=True,
    )
    defaults.update(overrides)
    return PolicyDecision(**defaults)


def _make_intent(**overrides) -> ExitIntentCandidate:
    """Minimal valid ExitIntentCandidate."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        intent_id=str(uuid.uuid4()),
        intent_timestamp=time.time(),
        action_name="HOLD",
        intent_type="SHADOW_EXIT",
        target_reduction_pct=0.0,
        confidence=0.7,
        uncertainty=0.3,
        source_decision_id=str(uuid.uuid4()),
        idempotency_key="abc123deadbeef00",
        shadow_only=True,
    )
    defaults.update(overrides)
    return ExitIntentCandidate(**defaults)


def _make_validation_result(**overrides) -> ExitIntentValidationResult:
    """Minimal valid ExitIntentValidationResult."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        validation_timestamp=time.time(),
        candidate_action="HOLD",
        source_intent_id=str(uuid.uuid4()),
        is_valid=True,
        hard_blocks=[],
        soft_warnings=[],
        violated_constraints=[],
        normalized_candidate_payload={},
        validation_confidence=1.0,
        shadow_only=True,
    )
    defaults.update(overrides)
    return ExitIntentValidationResult(**defaults)


def _make_trace(**overrides) -> DecisionTrace:
    """Minimal valid DecisionTrace."""
    did = str(uuid.uuid4())
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        trace_id=str(uuid.uuid4()),
        trace_timestamp=time.time(),
        source_decision_id=did,
        upstream_versions={"state_ts": time.time()},
        all_candidates=[],
        chosen_action="HOLD",
        rejected_actions=[],
        decisive_factors=["chosen=HOLD"],
        uncertainty_penalties={},
        constraint_effects={},
        final_reasoning="HOLD (rank=1, utility=0.500, conf=0.700)",
        shadow_only=True,
    )
    defaults.update(overrides)
    return DecisionTrace(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# S1: PolicyDecision contract
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyDecision:

    def test_valid_policy_decision_has_no_errors(self):
        pd = _make_policy_decision()
        assert pd.validate() == []

    def test_empty_position_id_fails(self):
        pd = _make_policy_decision(position_id="")
        assert any("position_id" in e for e in pd.validate())

    def test_empty_symbol_fails(self):
        pd = _make_policy_decision(symbol="")
        assert any("symbol" in e for e in pd.validate())

    def test_empty_decision_id_fails(self):
        pd = _make_policy_decision(decision_id="")
        assert any("decision_id" in e for e in pd.validate())

    def test_bad_timestamp_fails(self):
        pd = _make_policy_decision(decision_timestamp=0)
        assert any("decision_timestamp" in e for e in pd.validate())

    def test_invalid_action_fails(self):
        pd = _make_policy_decision(chosen_action="NUKE")
        assert any("chosen_action" in e for e in pd.validate())

    def test_rank_below_one_fails(self):
        pd = _make_policy_decision(chosen_action_rank=0)
        assert any("chosen_action_rank" in e for e in pd.validate())

    def test_utility_out_of_range_fails(self):
        pd = _make_policy_decision(chosen_action_utility=1.5)
        assert any("chosen_action_utility" in e for e in pd.validate())

    def test_confidence_out_of_range_fails(self):
        pd = _make_policy_decision(decision_confidence=-0.1)
        assert any("decision_confidence" in e for e in pd.validate())

    def test_uncertainty_out_of_range_fails(self):
        pd = _make_policy_decision(decision_uncertainty=2.0)
        assert any("decision_uncertainty" in e for e in pd.validate())

    def test_policy_blocked_non_hold_fails(self):
        """If policy_passed=False, action MUST be HOLD."""
        pd = _make_policy_decision(policy_passed=False, chosen_action="CLOSE_FULL")
        errors = pd.validate()
        assert any("policy_passed=False" in e for e in errors)

    def test_policy_blocked_hold_passes(self):
        """policy_passed=False with HOLD is valid."""
        pd = _make_policy_decision(policy_passed=False, chosen_action="HOLD")
        assert pd.validate() == []

    def test_shadow_only_must_be_true(self):
        pd = _make_policy_decision(shadow_only=False)
        assert any("shadow_only" in e for e in pd.validate())

    def test_to_dict_has_all_keys(self):
        pd = _make_policy_decision(
            policy_blocks=["BLOCK_A"],
            policy_warnings=["WARN_B"],
            reason_codes=["CODE_C"],
        )
        d = pd.to_dict()
        assert d["position_id"] == "pos-001"
        assert d["shadow_only"] is True
        assert json.loads(d["policy_blocks"]) == ["BLOCK_A"]
        assert json.loads(d["policy_warnings"]) == ["WARN_B"]
        assert json.loads(d["reason_codes"]) == ["CODE_C"]

    def test_to_dict_json_fields_are_valid_json(self):
        pd = _make_policy_decision(
            hazard_summary={"c": 0.5},
            belief_summary={"ep": 0.3},
            utility_summary={"HOLD": 0.6},
        )
        d = pd.to_dict()
        for key in ["hazard_summary", "belief_summary", "utility_summary",
                     "policy_blocks", "policy_warnings", "reason_codes",
                     "explanation_tags"]:
            json.loads(d[key])  # must not raise

    def test_all_seven_actions_accepted(self):
        for action in VALID_ACTIONS:
            pd = _make_policy_decision(chosen_action=action, policy_passed=True)
            errors = pd.validate()
            action_errors = [e for e in errors if "chosen_action" in e]
            assert action_errors == [], f"Action {action} should be valid"


# ═══════════════════════════════════════════════════════════════════════════
# S2: ExitIntentCandidate contract
# ═══════════════════════════════════════════════════════════════════════════


class TestExitIntentCandidate:

    def test_valid_intent_has_no_errors(self):
        ei = _make_intent()
        assert ei.validate() == []

    def test_empty_position_id_fails(self):
        ei = _make_intent(position_id="")
        assert any("position_id" in e for e in ei.validate())

    def test_empty_symbol_fails(self):
        ei = _make_intent(symbol="")
        assert any("symbol" in e for e in ei.validate())

    def test_empty_intent_id_fails(self):
        ei = _make_intent(intent_id="")
        assert any("intent_id" in e for e in ei.validate())

    def test_bad_timestamp_fails(self):
        ei = _make_intent(intent_timestamp=-1)
        assert any("intent_timestamp" in e for e in ei.validate())

    def test_invalid_action_fails(self):
        ei = _make_intent(action_name="EXPLODE")
        assert any("action_name" in e for e in ei.validate())

    def test_wrong_intent_type_fails(self):
        ei = _make_intent(intent_type="MARKET_ORDER")
        assert any("intent_type" in e for e in ei.validate())

    def test_reduction_pct_out_of_range_fails(self):
        ei = _make_intent(target_reduction_pct=1.5)
        assert any("target_reduction_pct" in e for e in ei.validate())

    def test_confidence_out_of_range_fails(self):
        ei = _make_intent(confidence=-0.1)
        assert any("confidence" in e for e in ei.validate())

    def test_uncertainty_out_of_range_fails(self):
        ei = _make_intent(uncertainty=2.0)
        assert any("uncertainty" in e for e in ei.validate())

    def test_missing_source_decision_id_fails(self):
        ei = _make_intent(source_decision_id="")
        assert any("source_decision_id" in e for e in ei.validate())

    def test_missing_idempotency_key_fails(self):
        ei = _make_intent(idempotency_key="")
        assert any("idempotency_key" in e for e in ei.validate())

    def test_shadow_only_must_be_true(self):
        ei = _make_intent(shadow_only=False)
        assert any("shadow_only" in e for e in ei.validate())

    def test_to_dict_has_all_keys(self):
        ei = _make_intent(tighten_parameters={"mode": "shadow"})
        d = ei.to_dict()
        assert d["action_name"] == "HOLD"
        assert d["intent_type"] == "SHADOW_EXIT"
        assert d["shadow_only"] is True
        assert json.loads(d["tighten_parameters"]) == {"mode": "shadow"}

    def test_all_seven_actions_accepted(self):
        for action in VALID_ACTIONS:
            frac = ACTION_EXIT_FRACTIONS[action]
            ei = _make_intent(action_name=action, target_reduction_pct=frac)
            action_errors = [e for e in ei.validate() if "action_name" in e]
            assert action_errors == [], f"Action {action} should be valid"


# ═══════════════════════════════════════════════════════════════════════════
# S3: ExitIntentValidationResult contract
# ═══════════════════════════════════════════════════════════════════════════


class TestExitIntentValidationResult:

    def test_valid_result_has_no_errors(self):
        vr = _make_validation_result()
        assert vr.validate() == []

    def test_empty_position_id_fails(self):
        vr = _make_validation_result(position_id="")
        assert any("position_id" in e for e in vr.validate())

    def test_empty_symbol_fails(self):
        vr = _make_validation_result(symbol="")
        assert any("symbol" in e for e in vr.validate())

    def test_bad_timestamp_fails(self):
        vr = _make_validation_result(validation_timestamp=0)
        assert any("validation_timestamp" in e for e in vr.validate())

    def test_empty_candidate_action_fails(self):
        vr = _make_validation_result(candidate_action="")
        assert any("candidate_action" in e for e in vr.validate())

    def test_confidence_out_of_range_fails(self):
        vr = _make_validation_result(validation_confidence=1.5)
        assert any("validation_confidence" in e for e in vr.validate())

    def test_invalid_but_no_hard_blocks_fails(self):
        """is_valid=False requires at least one hard_block."""
        vr = _make_validation_result(is_valid=False, hard_blocks=[])
        assert any("hard_blocks" in e for e in vr.validate())

    def test_invalid_with_hard_blocks_passes(self):
        vr = _make_validation_result(
            is_valid=False,
            hard_blocks=["SCHEMA_VALIDATION_FAILED"],
        )
        assert vr.validate() == []

    def test_shadow_only_must_be_true(self):
        vr = _make_validation_result(shadow_only=False)
        assert any("shadow_only" in e for e in vr.validate())

    def test_to_dict_serializes_lists_as_json(self):
        vr = _make_validation_result(
            hard_blocks=["BLOCK_A"],
            soft_warnings=["WARN_B"],
            violated_constraints=["vc"],
        )
        d = vr.to_dict()
        assert json.loads(d["hard_blocks"]) == ["BLOCK_A"]
        assert json.loads(d["soft_warnings"]) == ["WARN_B"]
        assert json.loads(d["violated_constraints"]) == ["vc"]


# ═══════════════════════════════════════════════════════════════════════════
# S4: DecisionTrace contract
# ═══════════════════════════════════════════════════════════════════════════


class TestDecisionTrace:

    def test_valid_trace_has_no_errors(self):
        dt = _make_trace()
        assert dt.validate() == []

    def test_empty_position_id_fails(self):
        dt = _make_trace(position_id="")
        assert any("position_id" in e for e in dt.validate())

    def test_empty_symbol_fails(self):
        dt = _make_trace(symbol="")
        assert any("symbol" in e for e in dt.validate())

    def test_empty_trace_id_fails(self):
        dt = _make_trace(trace_id="")
        assert any("trace_id" in e for e in dt.validate())

    def test_bad_timestamp_fails(self):
        dt = _make_trace(trace_timestamp=0)
        assert any("trace_timestamp" in e for e in dt.validate())

    def test_empty_source_decision_id_fails(self):
        dt = _make_trace(source_decision_id="")
        assert any("source_decision_id" in e for e in dt.validate())

    def test_empty_chosen_action_fails(self):
        dt = _make_trace(chosen_action="")
        assert any("chosen_action" in e for e in dt.validate())

    def test_shadow_only_must_be_true(self):
        dt = _make_trace(shadow_only=False)
        assert any("shadow_only" in e for e in dt.validate())

    def test_to_dict_serializes_complex_fields_as_json(self):
        dt = _make_trace(
            upstream_versions={"state_ts": 1.0, "belief_ts": 2.0},
            all_candidates=[{"action": "HOLD"}],
            rejected_actions=[{"action": "CLOSE_FULL", "reason": "outranked"}],
            decisive_factors=["chosen=HOLD"],
            uncertainty_penalties={"uncertainty_total": 0.2},
            constraint_effects={"BLOCK": "forced HOLD"},
        )
        d = dt.to_dict()
        assert json.loads(d["upstream_versions"]) == {"state_ts": 1.0, "belief_ts": 2.0}
        assert json.loads(d["all_candidates"]) == [{"action": "HOLD"}]
        assert json.loads(d["rejected_actions"])[0]["action"] == "CLOSE_FULL"
        assert json.loads(d["decisive_factors"]) == ["chosen=HOLD"]
        assert json.loads(d["uncertainty_penalties"]) == {"uncertainty_total": 0.2}
        assert json.loads(d["constraint_effects"]) == {"BLOCK": "forced HOLD"}


# ═══════════════════════════════════════════════════════════════════════════
# S5: PolicyConstraints constants
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyConstraints:

    def test_uncertainty_hard_ceiling_value(self):
        assert PC.UNCERTAINTY_HARD_CEILING == 0.70

    def test_uncertainty_soft_ceiling_value(self):
        assert PC.UNCERTAINTY_SOFT_CEILING == 0.50

    def test_data_completeness_hard_floor_value(self):
        assert PC.DATA_COMPLETENESS_HARD_FLOOR == 0.40

    def test_data_completeness_soft_floor_value(self):
        assert PC.DATA_COMPLETENESS_SOFT_FLOOR == 0.60

    def test_min_action_conviction_value(self):
        assert PC.MIN_ACTION_CONVICTION == 0.15

    def test_prefer_hold_threshold_value(self):
        assert PC.PREFER_HOLD_THRESHOLD == 0.20

    def test_edge_neutral_band_value(self):
        assert PC.EDGE_NEUTRAL_BAND == 0.15

    def test_hazard_emergency_threshold_value(self):
        assert PC.HAZARD_EMERGENCY_THRESHOLD == 0.85

    def test_close_full_min_hazard_value(self):
        assert PC.CLOSE_FULL_MIN_HAZARD == 0.50

    def test_close_full_min_exit_pressure_value(self):
        assert PC.CLOSE_FULL_MIN_EXIT_PRESSURE == 0.70

    def test_max_upstream_age_value(self):
        assert PC.MAX_UPSTREAM_AGE_SEC == 120.0

    def test_safe_actions_are_frozen(self):
        assert isinstance(PC.SAFE_ACTIONS_HIGH_UNCERTAINTY, frozenset)
        assert PC.SAFE_ACTIONS_HIGH_UNCERTAINTY == {"HOLD", "TIGHTEN_EXIT"}

    def test_profit_required_actions_are_frozen(self):
        assert isinstance(PC.PROFIT_REQUIRED_ACTIONS, frozenset)
        assert PC.PROFIT_REQUIRED_ACTIONS == {"TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE"}

    def test_soft_ceiling_below_hard_ceiling(self):
        assert PC.UNCERTAINTY_SOFT_CEILING < PC.UNCERTAINTY_HARD_CEILING

    def test_soft_floor_above_hard_floor(self):
        assert PC.DATA_COMPLETENESS_SOFT_FLOOR > PC.DATA_COMPLETENESS_HARD_FLOOR


# ═══════════════════════════════════════════════════════════════════════════
# S6: ReasonCodes constants
# ═══════════════════════════════════════════════════════════════════════════


class TestReasonCodes:

    def test_seven_hard_block_codes(self):
        assert len(RC.ALL_HARD_BLOCKS) == 7

    def test_six_soft_warning_codes(self):
        assert len(RC.ALL_SOFT_WARNINGS) == 6

    def test_hard_blocks_are_frozenset(self):
        assert isinstance(RC.ALL_HARD_BLOCKS, frozenset)

    def test_soft_warnings_are_frozenset(self):
        assert isinstance(RC.ALL_SOFT_WARNINGS, frozenset)

    def test_no_overlap_hard_soft(self):
        """Hard blocks and soft warnings must not overlap."""
        assert RC.ALL_HARD_BLOCKS.isdisjoint(RC.ALL_SOFT_WARNINGS)

    def test_known_hard_block_codes_exist(self):
        expected = {
            RC.UNCERTAINTY_CEILING_BREACH,
            RC.DATA_COMPLETENESS_FLOOR,
            RC.PROFIT_TAKING_NO_PROFIT,
            RC.CLOSE_FULL_INSUFFICIENT_HAZARD,
            RC.STALE_UPSTREAM_DATA,
            RC.MISSING_UPSTREAM_DATA,
            RC.SHADOW_ONLY_VIOLATION,
        }
        assert expected == RC.ALL_HARD_BLOCKS

    def test_known_soft_warning_codes_exist(self):
        expected = {
            RC.INSUFFICIENT_CONVICTION,
            RC.EDGE_NEUTRAL_HOLD_PREFERRED,
            RC.HIGH_UNCERTAINTY_DAMPENING,
            RC.LOW_DATA_COMPLETENESS,
            RC.QUALITY_FLAGS_PRESENT,
            RC.PARTIAL_UPSTREAM,
        }
        assert expected == RC.ALL_SOFT_WARNINGS

    def test_emergency_override_code_exists(self):
        assert RC.HAZARD_EMERGENCY_OVERRIDE == "HAZARD_EMERGENCY_OVERRIDE"

    def test_holdback_codes_exist(self):
        assert RC.POLICY_FALLBACK_HOLD == "POLICY_FALLBACK_HOLD"
        assert RC.HOLD_PROMOTED_LOW_CONVICTION == "HOLD_PROMOTED_LOW_CONVICTION"
        assert RC.HOLD_PROMOTED_EDGE_NEUTRAL == "HOLD_PROMOTED_EDGE_NEUTRAL"

    def test_explanation_tags_are_human_readable(self):
        """All TAG_ constants should be multi-word strings."""
        for name in dir(RC):
            if name.startswith("TAG_"):
                val = getattr(RC, name)
                assert isinstance(val, str)
                assert len(val) > 5, f"{name} tag too short"


# ═══════════════════════════════════════════════════════════════════════════
# S7: ExitPolicyEngine (7-step pipeline)
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyEngineFailClosed:
    """Fail-closed behaviour: empty candidates, bad data → HOLD."""

    def test_empty_candidates_returns_hold(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate([], _make_belief(), _make_hazard(), _make_state())
        assert pd.chosen_action == "HOLD"
        assert pd.policy_passed is False
        assert RC.MISSING_UPSTREAM_DATA in pd.policy_blocks

    def test_result_always_validates_clean(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate([], _make_belief(), _make_hazard(), _make_state())
        assert pd.validate() == []


class TestPolicyEngineStaleness:
    """Step 1: Upstream freshness check."""

    def test_stale_data_forces_hold(self):
        engine = ExitPolicyEngine()
        state = _make_state(
            source_timestamps={"p33_snapshot": time.time() - 200},
        )
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), state,
        )
        assert pd.chosen_action == "HOLD"
        assert pd.policy_passed is False
        assert RC.STALE_UPSTREAM_DATA in pd.policy_blocks

    def test_fresh_data_does_not_block(self):
        engine = ExitPolicyEngine()
        state = _make_state(
            source_timestamps={"p33_snapshot": time.time() - 10},
        )
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), state,
        )
        assert RC.STALE_UPSTREAM_DATA not in pd.policy_blocks


class TestPolicyEngineDataCompleteness:
    """Step 2: Data completeness floor."""

    def test_low_completeness_forces_hold(self):
        engine = ExitPolicyEngine()
        belief = _make_belief(data_completeness=0.30)
        pd = engine.evaluate(
            _make_candidates(), belief, _make_hazard(), _make_state(),
        )
        assert pd.chosen_action == "HOLD"
        assert pd.policy_passed is False
        assert RC.DATA_COMPLETENESS_FLOOR in pd.policy_blocks

    def test_soft_completeness_warns(self):
        engine = ExitPolicyEngine()
        belief = _make_belief(data_completeness=0.55)
        pd = engine.evaluate(
            _make_candidates(), belief, _make_hazard(), _make_state(),
        )
        assert RC.LOW_DATA_COMPLETENESS in pd.policy_warnings

    def test_high_completeness_no_warnings(self):
        engine = ExitPolicyEngine()
        belief = _make_belief(data_completeness=0.90)
        pd = engine.evaluate(
            _make_candidates(), belief, _make_hazard(), _make_state(),
        )
        assert RC.DATA_COMPLETENESS_FLOOR not in pd.policy_blocks
        assert RC.LOW_DATA_COMPLETENESS not in pd.policy_warnings


class TestPolicyEngineUncertainty:
    """Step 3: Uncertainty ceiling."""

    def test_extreme_uncertainty_forces_hold(self):
        engine = ExitPolicyEngine()
        belief = _make_belief(uncertainty_total=0.80)
        pd = engine.evaluate(
            _make_candidates(), belief, _make_hazard(), _make_state(),
        )
        assert pd.chosen_action == "HOLD"
        assert pd.policy_passed is False
        assert RC.UNCERTAINTY_CEILING_BREACH in pd.policy_blocks

    def test_moderate_uncertainty_soft_warns(self):
        engine = ExitPolicyEngine()
        belief = _make_belief(uncertainty_total=0.55)
        pd = engine.evaluate(
            _make_candidates(), belief, _make_hazard(), _make_state(),
        )
        # Note: soft warn may appear if action is not in SAFE_ACTIONS
        assert RC.UNCERTAINTY_CEILING_BREACH not in pd.policy_blocks

    def test_low_uncertainty_no_blocks(self):
        engine = ExitPolicyEngine()
        belief = _make_belief(uncertainty_total=0.10)
        pd = engine.evaluate(
            _make_candidates(), belief, _make_hazard(), _make_state(),
        )
        assert RC.UNCERTAINTY_CEILING_BREACH not in pd.policy_blocks


class TestPolicyEngineEmergencyOverride:
    """Step 4a: Emergency hazard override → force CLOSE_FULL."""

    def test_extreme_composite_hazard_forces_close(self):
        engine = ExitPolicyEngine()
        hazard = _make_hazard(composite_hazard=0.90)
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert pd.chosen_action == "CLOSE_FULL"
        assert RC.HAZARD_EMERGENCY_OVERRIDE in pd.reason_codes

    def test_extreme_reversal_hazard_forces_close(self):
        engine = ExitPolicyEngine()
        hazard = _make_hazard(reversal_hazard=0.80)
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert pd.chosen_action == "CLOSE_FULL"
        assert RC.HAZARD_EMERGENCY_OVERRIDE in pd.reason_codes

    def test_extreme_drawdown_hazard_forces_close(self):
        engine = ExitPolicyEngine()
        hazard = _make_hazard(drawdown_hazard=0.60)
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert pd.chosen_action == "CLOSE_FULL"
        assert RC.HAZARD_EMERGENCY_OVERRIDE in pd.reason_codes

    def test_low_hazard_no_emergency(self):
        engine = ExitPolicyEngine()
        hazard = _make_hazard(
            composite_hazard=0.30, reversal_hazard=0.20, drawdown_hazard=0.25,
        )
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert RC.HAZARD_EMERGENCY_OVERRIDE not in pd.reason_codes


class TestPolicyEngineActionConstraints:
    """Step 4b: Action-specific constraints."""

    def test_take_profit_blocked_when_unprofitable(self):
        """TAKE_PROFIT requires positive PnL."""
        engine = ExitPolicyEngine()
        # Build candidates with TAKE_PROFIT_PARTIAL as rank 1
        cands = _make_candidates()
        for c in cands:
            if c.action == "TAKE_PROFIT_PARTIAL":
                c.net_utility = 0.99
                c.rank = 1
            else:
                c.net_utility = max(0.01, c.net_utility - 0.2)
        cands.sort(key=lambda c: c.net_utility, reverse=True)
        for i, c in enumerate(cands):
            c.rank = i + 1

        state = _make_state(unrealized_pnl=-50.0)
        pd = engine.evaluate(cands, _make_belief(), _make_hazard(), state)
        assert pd.chosen_action == "HOLD"
        assert RC.PROFIT_TAKING_NO_PROFIT in pd.policy_blocks

    def test_close_full_blocked_when_low_hazard_and_low_pressure(self):
        """CLOSE_FULL needs sufficient hazard or exit pressure."""
        engine = ExitPolicyEngine()
        cands = _make_candidates()
        for c in cands:
            if c.action == "CLOSE_FULL":
                c.net_utility = 0.99
                c.rank = 1
            else:
                c.net_utility = max(0.01, c.net_utility - 0.3)
        cands.sort(key=lambda c: c.net_utility, reverse=True)
        for i, c in enumerate(cands):
            c.rank = i + 1

        belief = _make_belief(exit_pressure=0.30)
        hazard = _make_hazard(
            composite_hazard=0.20,
            reversal_hazard=0.10,
            drawdown_hazard=0.10,
        )
        pd = engine.evaluate(cands, belief, hazard, _make_state())
        assert pd.chosen_action == "HOLD"
        assert RC.CLOSE_FULL_INSUFFICIENT_HAZARD in pd.policy_blocks


class TestPolicyEngineConviction:
    """Step 4c: Conviction check — low utility demotes to HOLD."""

    def test_low_conviction_demotes_to_hold(self):
        engine = ExitPolicyEngine()
        cands = _make_candidates()
        for c in cands:
            c.net_utility = 0.10  # Below MIN_ACTION_CONVICTION
        cands[0].action = "REDUCE_SMALL"
        cands[0].rank = 1

        belief = _make_belief(directional_edge=0.5)  # Not edge-neutral
        pd = engine.evaluate(cands, belief, _make_hazard(), _make_state())
        assert pd.chosen_action == "HOLD"
        assert RC.HOLD_PROMOTED_LOW_CONVICTION in pd.reason_codes


class TestPolicyEngineEdgeNeutral:
    """Step 4d: Edge-neutral HOLD bias."""

    def test_edge_neutral_prefers_hold(self):
        engine = ExitPolicyEngine()
        cands = _make_candidates()
        # Set up a non-HOLD action as rank 1 with utility below prefer_hold
        for c in cands:
            if c.action == "REDUCE_SMALL":
                c.net_utility = 0.18  # Above MIN_ACTION_CONVICTION but below PREFER_HOLD
                c.rank = 1
            elif c.action == "HOLD":
                c.net_utility = 0.16
                c.rank = 2
            else:
                c.net_utility = 0.05
        cands.sort(key=lambda c: c.net_utility, reverse=True)
        for i, c in enumerate(cands):
            c.rank = i + 1

        belief = _make_belief(
            directional_edge=0.05,  # Within EDGE_NEUTRAL_BAND (0.15)
        )
        pd = engine.evaluate(cands, belief, _make_hazard(), _make_state())
        assert pd.chosen_action == "HOLD"
        assert RC.HOLD_PROMOTED_EDGE_NEUTRAL in pd.reason_codes


class TestPolicyEngineHazardCloseBoost:
    """Step 4e: Hazard close boost (non-emergency)."""

    def test_hazard_boost_code_when_close_with_high_hazard(self):
        engine = ExitPolicyEngine()
        cands = _make_candidates()
        for c in cands:
            if c.action == "CLOSE_FULL":
                c.net_utility = 0.90
                c.rank = 1
            else:
                c.net_utility = max(0.01, c.net_utility - 0.5)
        cands.sort(key=lambda c: c.net_utility, reverse=True)
        for i, c in enumerate(cands):
            c.rank = i + 1

        belief = _make_belief(exit_pressure=0.80, directional_edge=-0.5)
        hazard = _make_hazard(
            composite_hazard=0.70,  # > HAZARD_CLOSE_BOOST_THRESHOLD (0.65)
            reversal_hazard=0.40,   # below emergency
            drawdown_hazard=0.40,   # below emergency
        )
        pd = engine.evaluate(cands, belief, hazard, _make_state())
        assert pd.chosen_action == "CLOSE_FULL"
        assert RC.HAZARD_CLOSE_BOOST in pd.reason_codes


class TestPolicyEngineDecisionOutput:
    """General output quality checks."""

    def test_decision_validates_clean(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert pd.validate() == []

    def test_decision_shadow_only_always_true(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert pd.shadow_only is True

    def test_decision_has_summaries(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert "composite" in pd.hazard_summary
        assert "exit_pressure" in pd.belief_summary
        assert len(pd.utility_summary) == 7  # All 7 actions

    def test_confidence_in_range(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert 0.0 <= pd.decision_confidence <= 1.0
        assert 0.0 <= pd.decision_uncertainty <= 1.0

    def test_decision_id_is_uuid(self):
        engine = ExitPolicyEngine()
        pd = engine.evaluate(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        uuid.UUID(pd.decision_id)  # Must not raise


# ═══════════════════════════════════════════════════════════════════════════
# S8: ExitIntentGatewayValidator
# ═══════════════════════════════════════════════════════════════════════════


class TestGatewayValidatorSchema:
    """Schema and field validation."""

    def test_valid_intent_passes(self):
        gw = ExitIntentGatewayValidator()
        result = gw.validate(_make_intent())
        assert result.is_valid is True
        assert result.hard_blocks == []

    def test_invalid_action_blocks(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(action_name="NUKE")
        result = gw.validate(intent)
        assert result.is_valid is False
        assert "INVALID_ACTION" in result.hard_blocks

    def test_schema_failure_blocks(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(position_id="", intent_id="")
        result = gw.validate(intent)
        assert result.is_valid is False
        assert "SCHEMA_VALIDATION_FAILED" in result.hard_blocks

    def test_shadow_only_violation_blocks(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(shadow_only=False)
        result = gw.validate(intent)
        assert result.is_valid is False
        assert RC.SHADOW_ONLY_VIOLATION in result.hard_blocks


class TestGatewayValidatorPayload:
    """Payload normalization and percentage sanity."""

    def test_reduction_pct_mismatch_warns(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(
            action_name="CLOSE_FULL",
            target_reduction_pct=0.50,  # should be 1.0
        )
        result = gw.validate(intent)
        assert any("reduction_pct_mismatch" in w for w in result.soft_warnings)

    def test_correct_reduction_pct_no_warning(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(
            action_name="CLOSE_FULL",
            target_reduction_pct=1.0,
        )
        result = gw.validate(intent)
        assert not any("reduction_pct_mismatch" in w for w in result.soft_warnings)

    def test_normalized_payload_in_result(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent()
        result = gw.validate(intent)
        assert isinstance(result.normalized_candidate_payload, dict)
        assert "action_name" in result.normalized_candidate_payload


class TestGatewayValidatorIdempotency:
    """Idempotency dedup."""

    def test_duplicate_intent_blocked(self):
        gw = ExitIntentGatewayValidator()
        intent1 = _make_intent(idempotency_key="same_key_123")
        intent2 = _make_intent(idempotency_key="same_key_123")
        r1 = gw.validate(intent1)
        r2 = gw.validate(intent2)
        assert r1.is_valid is True
        assert r2.is_valid is False
        assert "DUPLICATE_INTENT" in r2.hard_blocks

    def test_different_keys_both_pass(self):
        gw = ExitIntentGatewayValidator()
        r1 = gw.validate(_make_intent(idempotency_key="key_aaa"))
        r2 = gw.validate(_make_intent(idempotency_key="key_bbb"))
        assert r1.is_valid is True
        assert r2.is_valid is True


class TestGatewayValidatorFreshness:
    """Source freshness and evidence checks."""

    def test_stale_intent_warns(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(intent_timestamp=time.time() - 120)
        result = gw.validate(intent)
        assert any("intent_age" in w for w in result.soft_warnings)

    def test_zero_confidence_warns(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(confidence=0.0)
        result = gw.validate(intent)
        assert any("zero_confidence" in w for w in result.soft_warnings)

    def test_no_source_decision_warns(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(source_decision_id="")
        result = gw.validate(intent)
        # Empty source_decision_id triggers both schema error and warning
        assert result.is_valid is False


class TestGatewayValidatorOutput:
    """Output quality."""

    def test_result_validates_clean(self):
        gw = ExitIntentGatewayValidator()
        result = gw.validate(_make_intent())
        assert result.validate() == []

    def test_result_shadow_only_always_true(self):
        gw = ExitIntentGatewayValidator()
        result = gw.validate(_make_intent())
        assert result.shadow_only is True

    def test_valid_result_has_confidence_one(self):
        gw = ExitIntentGatewayValidator()
        result = gw.validate(_make_intent())
        assert result.validation_confidence == 1.0

    def test_invalid_result_has_confidence_zero(self):
        gw = ExitIntentGatewayValidator()
        intent = _make_intent(action_name="NUKE")
        result = gw.validate(intent)
        assert result.validation_confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# S9: IdempotencyTracker
# ═══════════════════════════════════════════════════════════════════════════


class TestIdempotencyTracker:

    def test_first_check_is_not_duplicate(self):
        tracker = IdempotencyTracker()
        assert tracker.is_duplicate("key_1") is False

    def test_second_check_is_duplicate(self):
        tracker = IdempotencyTracker()
        tracker.is_duplicate("key_1")
        assert tracker.is_duplicate("key_1") is True

    def test_different_keys_are_not_duplicates(self):
        tracker = IdempotencyTracker()
        tracker.is_duplicate("key_1")
        assert tracker.is_duplicate("key_2") is False

    def test_size_tracks_entries(self):
        tracker = IdempotencyTracker()
        assert tracker.size == 0
        tracker.is_duplicate("a")
        assert tracker.size == 1
        tracker.is_duplicate("b")
        assert tracker.size == 2

    def test_clear_removes_all(self):
        tracker = IdempotencyTracker()
        tracker.is_duplicate("a")
        tracker.is_duplicate("b")
        tracker.clear()
        assert tracker.size == 0
        assert tracker.is_duplicate("a") is False

    def test_expired_keys_evicted(self):
        tracker = IdempotencyTracker(window_sec=0.01)
        tracker.is_duplicate("old_key")
        import time as _t
        _t.sleep(0.05)
        assert tracker.is_duplicate("old_key") is False  # Evicted

    def test_max_entries_bounded(self):
        tracker = IdempotencyTracker()
        for i in range(MAX_ENTRIES + 100):
            tracker.is_duplicate(f"key_{i}")
        assert tracker.size <= MAX_ENTRIES

    def test_default_window_is_300(self):
        assert DEFAULT_WINDOW_SEC == 300.0


# ═══════════════════════════════════════════════════════════════════════════
# S10: PayloadNormalizer
# ═══════════════════════════════════════════════════════════════════════════


class TestPayloadNormalizer:

    def test_strips_unexpected_fields(self):
        raw = {"position_id": "p1", "rogue_field": "bad", "action_name": "HOLD"}
        normalized, warnings = normalize_payload(raw)
        assert "rogue_field" not in normalized
        assert any("stripped_fields" in w for w in warnings)

    def test_clamps_confidence_to_01(self):
        raw = {"confidence": 1.5, "uncertainty": -0.3, "target_reduction_pct": 2.0}
        normalized, _ = normalize_payload(raw)
        assert normalized["confidence"] == 1.0
        assert normalized["uncertainty"] == 0.0
        assert normalized["target_reduction_pct"] == 1.0

    def test_invalid_action_warns(self):
        raw = {"action_name": "NUKE"}
        _, warnings = normalize_payload(raw)
        assert any("invalid_action" in w for w in warnings)

    def test_wrong_intent_type_warns(self):
        raw = {"intent_type": "LIVE_EXIT"}
        _, warnings = normalize_payload(raw)
        assert any("unexpected_intent_type" in w for w in warnings)

    def test_reduction_pct_mismatch_warns(self):
        raw = {"action_name": "HOLD", "target_reduction_pct": 0.50}
        _, warnings = normalize_payload(raw)
        assert any("reduction_pct_mismatch" in w for w in warnings)

    def test_forces_shadow_only_true(self):
        raw = {"shadow_only": False}
        normalized, _ = normalize_payload(raw)
        assert normalized["shadow_only"] is True

    def test_check_required_fields_detects_missing(self):
        missing = check_required_fields({})
        assert "position_id" in missing
        assert "symbol" in missing
        assert "action_name" in missing

    def test_check_required_fields_passes_when_present(self):
        payload = {
            "position_id": "p1", "symbol": "BTCUSDT",
            "intent_id": "i1", "action_name": "HOLD",
            "intent_type": "SHADOW_EXIT", "source_decision_id": "d1",
            "idempotency_key": "k1",
        }
        assert check_required_fields(payload) == []


# ═══════════════════════════════════════════════════════════════════════════
# S11: ExitAgentOrchestrator (integration with mock publisher)
# ═══════════════════════════════════════════════════════════════════════════


class TestOrchestratorDecisionCycle:
    """Integration: full decision cycle with mock ShadowPublisher."""

    def _mock_publisher(self) -> MagicMock:
        pub = MagicMock()
        pub.publish_policy_decision = MagicMock()
        pub.publish_intent_candidate = MagicMock()
        pub.publish_intent_validation = MagicMock()
        pub.publish_decision_trace = MagicMock()
        return pub

    def test_returns_policy_decision(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert pd is not None
        assert isinstance(pd, PolicyDecision)

    def test_decision_validates_clean(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert pd.validate() == []

    def test_decision_shadow_only(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert pd.shadow_only is True

    def test_publishes_four_artifacts(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        pub.publish_policy_decision.assert_called_once()
        pub.publish_intent_candidate.assert_called_once()
        pub.publish_intent_validation.assert_called_once()
        pub.publish_decision_trace.assert_called_once()

    def test_published_decision_matches_return(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        published_pd = pub.publish_policy_decision.call_args[0][0]
        assert published_pd.decision_id == pd.decision_id

    def test_intent_has_correct_action(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        intent = pub.publish_intent_candidate.call_args[0][0]
        assert isinstance(intent, ExitIntentCandidate)
        assert intent.action_name in VALID_ACTIONS
        assert intent.intent_type == "SHADOW_EXIT"
        assert intent.shadow_only is True

    def test_intent_has_idempotency_key(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        intent = pub.publish_intent_candidate.call_args[0][0]
        assert len(intent.idempotency_key) == 16  # SHA256[:16]

    def test_intent_source_links_to_decision(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        intent = pub.publish_intent_candidate.call_args[0][0]
        assert intent.source_decision_id == pd.decision_id

    def test_validation_result_produced(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        vr = pub.publish_intent_validation.call_args[0][0]
        assert isinstance(vr, ExitIntentValidationResult)
        assert vr.shadow_only is True

    def test_trace_produced_with_all_candidates(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        trace = pub.publish_decision_trace.call_args[0][0]
        assert isinstance(trace, DecisionTrace)
        assert len(trace.all_candidates) == 7
        assert trace.shadow_only is True

    def test_trace_links_to_decision(self):
        pub = self._mock_publisher()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        trace = pub.publish_decision_trace.call_args[0][0]
        assert trace.source_decision_id == pd.decision_id


class TestOrchestratorFailClosed:
    """Orchestrator returns None on error, never raises."""

    def test_exception_returns_none(self):
        pub = MagicMock()
        pub.publish_policy_decision.side_effect = RuntimeError("boom")
        orch = ExitAgentOrchestrator(publisher=pub)
        result = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert result is None

    def test_emergency_via_orchestrator(self):
        """Emergency hazard triggers CLOSE_FULL through orchestrator."""
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        hazard = _make_hazard(composite_hazard=0.90)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert pd is not None
        assert pd.chosen_action == "CLOSE_FULL"


class TestOrchestratorIntentReductionPct:
    """Intent exit fraction matches chosen action."""

    def test_hold_intent_has_zero_reduction(self):
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        if pd and pd.chosen_action == "HOLD":
            intent = pub.publish_intent_candidate.call_args[0][0]
            assert intent.target_reduction_pct == 0.0

    def test_close_full_intent_has_full_reduction(self):
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        hazard = _make_hazard(composite_hazard=0.90)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert pd.chosen_action == "CLOSE_FULL"
        intent = pub.publish_intent_candidate.call_args[0][0]
        assert intent.target_reduction_pct == 1.0


class TestOrchestratorTightenParams:
    """TIGHTEN_EXIT includes tighten_parameters."""

    def test_tighten_exit_has_params(self):
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        # Build candidates with TIGHTEN_EXIT as top action
        cands = _make_candidates()
        for c in cands:
            if c.action == "TIGHTEN_EXIT":
                c.net_utility = 0.90
            else:
                c.net_utility = 0.05
        cands.sort(key=lambda c: c.net_utility, reverse=True)
        for i, c in enumerate(cands):
            c.rank = i + 1

        belief = _make_belief(directional_edge=-0.5)
        pd = orch.run_decision_cycle(cands, belief, _make_hazard(), _make_state())
        if pd and pd.chosen_action == "TIGHTEN_EXIT":
            intent = pub.publish_intent_candidate.call_args[0][0]
            assert "mode" in intent.tighten_parameters
            assert intent.tighten_parameters["mode"] == "shadow"


# ═══════════════════════════════════════════════════════════════════════════
# S12: Shadow stream verification
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4ShadowStreams:

    def test_shadow_stream_names(self):
        """Phase 4 shadow streams follow naming convention."""
        expected_streams = {
            "quantum:stream:exit.policy.shadow",
            "quantum:stream:exit.intent.candidate.shadow",
            "quantum:stream:exit.intent.validation.shadow",
            "quantum:stream:exit.decision.trace.shadow",
        }
        for stream in expected_streams:
            assert stream.startswith("quantum:stream:exit.")
            assert stream.endswith(".shadow")

    def test_policy_decision_serializable_for_stream(self):
        pd = _make_policy_decision()
        d = pd.to_dict()
        for k, v in d.items():
            assert isinstance(k, str), f"Key {k} must be str"
            assert isinstance(v, (str, int, float, bool)), (
                f"Value for {k} must be primitive, got {type(v)}"
            )

    def test_intent_candidate_serializable_for_stream(self):
        ei = _make_intent()
        d = ei.to_dict()
        for k, v in d.items():
            assert isinstance(k, str), f"Key {k} must be str"
            assert isinstance(v, (str, int, float, bool)), (
                f"Value for {k} must be primitive, got {type(v)}"
            )

    def test_validation_result_serializable_for_stream(self):
        vr = _make_validation_result()
        d = vr.to_dict()
        for k, v in d.items():
            assert isinstance(k, str), f"Key {k} must be str"
            assert isinstance(v, (str, int, float, bool)), (
                f"Value for {k} must be primitive, got {type(v)}"
            )

    def test_decision_trace_serializable_for_stream(self):
        dt = _make_trace()
        d = dt.to_dict()
        for k, v in d.items():
            assert isinstance(k, str), f"Key {k} must be str"
            assert isinstance(v, (str, int, float, bool)), (
                f"Value for {k} must be primitive, got {type(v)}"
            )

    def test_shadow_only_enforced_on_all_contracts(self):
        """Every Phase 4 contract has shadow_only=True."""
        pd = _make_policy_decision()
        ei = _make_intent()
        vr = _make_validation_result()
        dt = _make_trace()
        for obj in [pd, ei, vr, dt]:
            assert obj.shadow_only is True

    def test_forbidden_streams_not_written(self):
        """Phase 4 MUST NOT write to execution streams."""
        from microservices.exit_brain_v1.publishers.shadow_publisher import (
            _FORBIDDEN_STREAMS,
        )
        phase4_streams = {
            "quantum:stream:exit.policy.shadow",
            "quantum:stream:exit.intent.candidate.shadow",
            "quantum:stream:exit.intent.validation.shadow",
            "quantum:stream:exit.decision.trace.shadow",
        }
        for stream in phase4_streams:
            assert stream not in _FORBIDDEN_STREAMS

    def test_execution_streams_are_forbidden(self):
        """Verify the boundary streams are in forbidden list."""
        from microservices.exit_brain_v1.publishers.shadow_publisher import (
            _FORBIDDEN_STREAMS,
        )
        assert "quantum:stream:trade.intent" in _FORBIDDEN_STREAMS
        assert "quantum:stream:exit.intent" in _FORBIDDEN_STREAMS
        assert "quantum:stream:apply.plan" in _FORBIDDEN_STREAMS


# ═══════════════════════════════════════════════════════════════════════════
# S13: Full pipeline integration
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase4Integration:

    def test_full_pipeline_produces_all_artifacts(self):
        """Policy → Intent → Validation → Trace, all shadow-only."""
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )
        assert pd is not None

        intent = pub.publish_intent_candidate.call_args[0][0]
        vr = pub.publish_intent_validation.call_args[0][0]
        trace = pub.publish_decision_trace.call_args[0][0]

        # All four validate
        assert pd.validate() == []
        assert intent.validate() == []
        assert vr.validate() == []
        assert trace.validate() == []

        # All shadow only
        assert pd.shadow_only is True
        assert intent.shadow_only is True
        assert vr.shadow_only is True
        assert trace.shadow_only is True

    def test_short_position_pipeline(self):
        """Pipeline works with SHORT side."""
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        state = _make_state(side="SHORT")
        belief = _make_belief(side="SHORT")
        pd = orch.run_decision_cycle(
            _make_candidates(), belief, _make_hazard(), state,
        )
        assert pd is not None
        assert pd.validate() == []

    def test_all_to_dict_round_trips(self):
        """All Phase 4 contracts serialize without error."""
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), _make_state(),
        )

        pd = pub.publish_policy_decision.call_args[0][0]
        intent = pub.publish_intent_candidate.call_args[0][0]
        vr = pub.publish_intent_validation.call_args[0][0]
        trace = pub.publish_decision_trace.call_args[0][0]

        for obj in [pd, intent, vr, trace]:
            d = obj.to_dict()
            assert isinstance(d, dict)
            assert len(d) > 0

    def test_stale_data_blocks_full_pipeline(self):
        """Stale upstream → HOLD, but intent+trace still produced."""
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        state = _make_state(source_timestamps={"old": time.time() - 300})
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), _make_hazard(), state,
        )
        assert pd is not None
        assert pd.chosen_action == "HOLD"
        assert pd.policy_passed is False
        # Still publishes all four artifacts
        assert pub.publish_policy_decision.called
        assert pub.publish_intent_candidate.called
        assert pub.publish_intent_validation.called
        assert pub.publish_decision_trace.called

    def test_emergency_override_full_pipeline(self):
        """Emergency hazard → CLOSE_FULL through full pipeline."""
        pub = MagicMock()
        orch = ExitAgentOrchestrator(publisher=pub)
        hazard = _make_hazard(composite_hazard=0.90)
        pd = orch.run_decision_cycle(
            _make_candidates(), _make_belief(), hazard, _make_state(),
        )
        assert pd.chosen_action == "CLOSE_FULL"
        intent = pub.publish_intent_candidate.call_args[0][0]
        assert intent.action_name == "CLOSE_FULL"
        assert intent.target_reduction_pct == 1.0
