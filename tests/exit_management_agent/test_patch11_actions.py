"""Tests for patch11_actions — action enum, families, reason codes, mappings."""
from __future__ import annotations

import pytest

from microservices.exit_management_agent.patch11_actions import (
    ACTION_FAMILIES,
    ACTION_TO_FAMILY,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFLICT_THRESHOLD,
    DEFAULT_HIGH_TOXICITY,
    DEFAULT_LARGE_POSITION_USDT,
    EBV1_TO_PATCH11,
    PATCH11_ACTIONS,
    PATCH11_QTY_MAP,
    VALID_REASON_CODES,
)


class TestPatch11Actions:
    """Constants completeness and sanity."""

    def test_exactly_8_actions(self):
        assert len(PATCH11_ACTIONS) == 8

    def test_no_flip(self):
        assert "FLIP" not in PATCH11_ACTIONS
        assert "FLIP_LONG" not in PATCH11_ACTIONS
        assert "FLIP_SHORT" not in PATCH11_ACTIONS

    def test_all_actions_have_qty(self):
        for action in PATCH11_ACTIONS:
            assert action in PATCH11_QTY_MAP, f"Missing qty for {action}"

    def test_hold_qty_is_none(self):
        assert PATCH11_QTY_MAP["HOLD"] is None
        assert PATCH11_QTY_MAP["DEFENSIVE_HOLD"] is None
        assert PATCH11_QTY_MAP["QUARANTINE"] is None

    def test_full_close_qty_is_one(self):
        assert PATCH11_QTY_MAP["FULL_CLOSE"] == 1.0
        assert PATCH11_QTY_MAP["TOXICITY_UNWIND"] == 1.0

    def test_partial_quantities(self):
        assert PATCH11_QTY_MAP["REDUCE_25"] == 0.25
        assert PATCH11_QTY_MAP["REDUCE_50"] == 0.50
        assert PATCH11_QTY_MAP["HARVEST_70_KEEP_30"] == 0.70


class TestActionFamilies:
    """Family grouping and reverse lookup."""

    def test_five_families(self):
        assert len(ACTION_FAMILIES) == 5

    def test_every_action_has_family(self):
        for action in PATCH11_ACTIONS:
            assert action in ACTION_TO_FAMILY, f"{action} has no family"

    def test_family_membership(self):
        assert ACTION_TO_FAMILY["HOLD"] == "HOLD_FAMILY"
        assert ACTION_TO_FAMILY["DEFENSIVE_HOLD"] == "HOLD_FAMILY"
        assert ACTION_TO_FAMILY["REDUCE_25"] == "REDUCE_FAMILY"
        assert ACTION_TO_FAMILY["REDUCE_50"] == "REDUCE_FAMILY"
        assert ACTION_TO_FAMILY["HARVEST_70_KEEP_30"] == "HARVEST_FAMILY"
        assert ACTION_TO_FAMILY["FULL_CLOSE"] == "EXIT_FAMILY"
        assert ACTION_TO_FAMILY["TOXICITY_UNWIND"] == "EXIT_FAMILY"
        assert ACTION_TO_FAMILY["QUARANTINE"] == "QUARANTINE_FAMILY"

    def test_families_are_disjoint(self):
        all_actions = set()
        for members in ACTION_FAMILIES.values():
            assert len(all_actions & members) == 0, "Families overlap"
            all_actions |= members
        assert all_actions == PATCH11_ACTIONS


class TestEBV1Mapping:
    """EB v1 → PATCH-11 action mapping."""

    def test_all_7_ebv1_actions_mapped(self):
        assert len(EBV1_TO_PATCH11) == 7

    def test_mapped_to_valid_patch11_actions(self):
        for ebv1, patch11 in EBV1_TO_PATCH11.items():
            assert patch11 in PATCH11_ACTIONS, f"{ebv1}→{patch11} not in PATCH11_ACTIONS"

    def test_specific_mappings(self):
        assert EBV1_TO_PATCH11["HOLD"] == "HOLD"
        assert EBV1_TO_PATCH11["REDUCE_SMALL"] == "REDUCE_25"
        assert EBV1_TO_PATCH11["CLOSE_FULL"] == "FULL_CLOSE"
        assert EBV1_TO_PATCH11["TIGHTEN_EXIT"] == "DEFENSIVE_HOLD"


class TestReasonCodes:
    """Reason code set sanity."""

    def test_has_reason_codes(self):
        assert len(VALID_REASON_CODES) >= 20

    def test_all_uppercase(self):
        for code in VALID_REASON_CODES:
            assert code == code.upper(), f"Reason code not uppercase: {code}"

    def test_specific_codes_present(self):
        assert "THESIS_DECAY" in VALID_REASON_CODES
        assert "TOXICITY_CRITICAL" in VALID_REASON_CODES
        assert "ENSEMBLE_CONSENSUS" in VALID_REASON_CODES
        assert "CROSS_CHECK_FAIL" in VALID_REASON_CODES


class TestThresholds:
    """Default thresholds are sensible."""

    def test_confidence_in_valid_range(self):
        assert 0.0 < DEFAULT_CONFIDENCE_THRESHOLD < 1.0

    def test_conflict_in_valid_range(self):
        assert 0.0 < DEFAULT_CONFLICT_THRESHOLD < 1.0

    def test_toxicity_in_valid_range(self):
        assert 0.0 < DEFAULT_HIGH_TOXICITY < 1.0

    def test_large_position_positive(self):
        assert DEFAULT_LARGE_POSITION_USDT > 0
