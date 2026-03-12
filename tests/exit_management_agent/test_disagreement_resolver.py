"""Tests for PATCH-11 disagreement resolver — conservative middle-action policy."""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.llm.disagreement_resolver import (
    ResolutionResult,
    resolve_disagreement,
)
from microservices.exit_management_agent.patch11_actions import (
    ACTION_TO_FAMILY,
    PATCH11_ACTIONS,
)


class TestAgreement:
    """When both judges pick the same action or same family."""

    def test_same_action_agrees(self):
        r = resolve_disagreement("HOLD", 0.8, "HOLD", 0.7, 0.2, 0.1)
        assert r.agreed is True
        assert r.resolved_action == "HOLD"

    def test_same_family_hold(self):
        r = resolve_disagreement("HOLD", 0.8, "DEFENSIVE_HOLD", 0.9, 0.2, 0.1)
        assert r.agreed is True
        # Higher confidence wins
        assert r.resolved_action == "DEFENSIVE_HOLD"

    def test_same_family_reduce(self):
        r = resolve_disagreement("REDUCE_25", 0.9, "REDUCE_50", 0.6, 0.2, 0.1)
        assert r.agreed is True
        assert r.resolved_action == "REDUCE_25"  # higher confidence

    def test_same_family_exit(self):
        r = resolve_disagreement("FULL_CLOSE", 0.7, "TOXICITY_UNWIND", 0.8, 0.2, 0.1)
        assert r.agreed is True
        assert r.resolved_action == "TOXICITY_UNWIND"


class TestQuarantineAlwaysWins:
    """QUARANTINE should always be the resolved action when either judge says it."""

    def test_quarantine_vs_hold(self):
        r = resolve_disagreement("QUARANTINE", 0.6, "HOLD", 0.9, 0.2, 0.1)
        assert r.resolved_action == "QUARANTINE"

    def test_hold_vs_quarantine(self):
        r = resolve_disagreement("HOLD", 0.9, "QUARANTINE", 0.6, 0.2, 0.1)
        assert r.resolved_action == "QUARANTINE"

    def test_quarantine_vs_full_close(self):
        r = resolve_disagreement("QUARANTINE", 0.5, "FULL_CLOSE", 0.9, 0.2, 0.1)
        assert r.resolved_action == "QUARANTINE"


class TestHoldVsExit:
    """HOLD family vs EXIT family → conservative middle action."""

    def test_hold_vs_full_close_low_hazard(self):
        r = resolve_disagreement("HOLD", 0.8, "FULL_CLOSE", 0.7, 0.3, 0.1)
        assert r.agreed is False
        # Low hazard: REDUCE_25
        assert r.resolved_action == "REDUCE_25"

    def test_hold_vs_full_close_high_hazard(self):
        r = resolve_disagreement("HOLD", 0.8, "FULL_CLOSE", 0.7, 0.8, 0.1)
        assert r.agreed is False
        # High hazard: HARVEST
        assert r.resolved_action == "HARVEST_70_KEEP_30"


class TestHoldVsReduce:
    """HOLD family vs REDUCE family → REDUCE_25."""

    def test_hold_vs_reduce_25(self):
        r = resolve_disagreement("HOLD", 0.8, "REDUCE_25", 0.5, 0.3, 0.1)
        assert r.resolved_action == "REDUCE_25"

    def test_hold_vs_reduce_50(self):
        r = resolve_disagreement("HOLD", 0.8, "REDUCE_50", 0.5, 0.3, 0.1)
        assert r.resolved_action == "REDUCE_25"


class TestExitVsHarvest:
    """EXIT → HARVEST unless extreme hazard."""

    def test_exit_vs_harvest_low_hazard(self):
        r = resolve_disagreement("FULL_CLOSE", 0.8, "HARVEST_70_KEEP_30", 0.7, 0.3, 0.1)
        assert r.resolved_action == "HARVEST_70_KEEP_30"

    def test_exit_vs_harvest_extreme_hazard(self):
        r = resolve_disagreement("FULL_CLOSE", 0.8, "HARVEST_70_KEEP_30", 0.7, 0.85, 0.1)
        # hazard >= 0.8: exit family wins → higher confidence
        assert r.resolved_action == "FULL_CLOSE"


class TestReduceVsExit:
    """REDUCE vs EXIT → REDUCE_50."""

    def test_reduce_25_vs_full_close(self):
        r = resolve_disagreement("REDUCE_25", 0.7, "FULL_CLOSE", 0.7, 0.4, 0.1)
        assert r.resolved_action == "REDUCE_50"


class TestToxicityDisagreement:
    """When TOXICITY_UNWIND is involved and toxicity is high, nudge toward REDUCE_50."""

    def test_high_toxicity_hold_vs_toxicity_unwind(self):
        r = resolve_disagreement("HOLD", 0.8, "TOXICITY_UNWIND", 0.5, 0.3, 0.75)
        # HOLD vs EXIT_FAMILY (TOXICITY_UNWIND) with low composite hazard → REDUCE_25
        assert r.resolved_action == "REDUCE_25"

    def test_high_toxicity_reduce_vs_toxicity_unwind(self):
        r = resolve_disagreement("REDUCE_25", 0.8, "TOXICITY_UNWIND", 0.5, 0.3, 0.75)
        # REDUCE vs EXIT → REDUCE_50
        assert r.resolved_action == "REDUCE_50"


class TestDefaultFallback:
    """Any unclassified pair → REDUCE_25."""

    def test_harvest_vs_reduce(self):
        r = resolve_disagreement("HARVEST_70_KEEP_30", 0.6, "REDUCE_25", 0.6, 0.2, 0.1)
        # This falls through to default: REDUCE_25
        assert r.resolved_action in PATCH11_ACTIONS


class TestResultValidation:
    """All resolved actions must be in PATCH11_ACTIONS."""

    @pytest.mark.parametrize("a", sorted(PATCH11_ACTIONS))
    @pytest.mark.parametrize("b", sorted(PATCH11_ACTIONS))
    def test_all_pairs_resolve_to_valid_action(self, a, b):
        r = resolve_disagreement(a, 0.7, b, 0.6, 0.4, 0.3)
        assert r.resolved_action in PATCH11_ACTIONS
        assert isinstance(r.resolution_method, str)
        assert isinstance(r.agreed, bool)

    def test_result_is_immutable(self):
        r = resolve_disagreement("HOLD", 0.8, "HOLD", 0.7, 0.2, 0.1)
        assert isinstance(r, ResolutionResult)
        with pytest.raises(AttributeError):
            r.resolved_action = "FLIP"
