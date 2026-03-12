"""Tests for PATCH-11 LLM response validator.

Covers:
- Valid JSON acceptance
- Required field validation
- Action enum enforcement
- Confidence bounds
- Reason code validation
- Forbidden key rejection (no execution parameters)
- Unknown key rejection (strict mode)
- why_not key validation
- risk_note max length
"""
from __future__ import annotations

import json

import pytest

from microservices.exit_management_agent.llm.response_validator import (
    ValidatedResponse,
    validate_llm_response,
)
from microservices.exit_management_agent.patch11_actions import PATCH11_ACTIONS, VALID_REASON_CODES


def _valid_json(**overrides) -> str:
    """Build a valid LLM response JSON string with optional overrides."""
    base = {
        "action": "REDUCE_25",
        "confidence": 0.78,
        "reason_codes": ["THESIS_DECAY", "TOXICITY_RISING"],
        "why_not": {
            "HOLD": "thesis weakened",
            "FULL_CLOSE": "residual edge",
        },
        "risk_note": "Prefer partial de-risk.",
    }
    base.update(overrides)
    return json.dumps(base)


class TestValidResponse:
    """Happy path — valid responses should pass."""

    def test_minimal_valid(self):
        r = validate_llm_response(_valid_json())
        assert r.valid is True
        assert r.action == "REDUCE_25"
        assert r.confidence == 0.78
        assert "THESIS_DECAY" in r.reason_codes

    @pytest.mark.parametrize("action", sorted(PATCH11_ACTIONS))
    def test_all_valid_actions(self, action):
        r = validate_llm_response(_valid_json(action=action))
        assert r.valid is True
        assert r.action == action

    def test_confidence_zero(self):
        r = validate_llm_response(_valid_json(confidence=0.0))
        assert r.valid is True

    def test_confidence_one(self):
        r = validate_llm_response(_valid_json(confidence=1.0))
        assert r.valid is True

    def test_empty_why_not(self):
        r = validate_llm_response(_valid_json(why_not={}))
        assert r.valid is True

    def test_empty_risk_note(self):
        r = validate_llm_response(_valid_json(risk_note=""))
        assert r.valid is True


class TestInvalidJSON:
    """Non-JSON or malformed JSON should be rejected."""

    def test_not_json(self):
        r = validate_llm_response("this is not json")
        assert r.valid is False
        assert "json" in r.rejection_reason.lower()

    def test_empty_string(self):
        r = validate_llm_response("")
        assert r.valid is False

    def test_json_array(self):
        r = validate_llm_response("[1, 2, 3]")
        assert r.valid is False


class TestMissingFields:
    """Required fields must be present."""

    @pytest.mark.parametrize(
        "field", ["action", "confidence", "reason_codes", "why_not", "risk_note"]
    )
    def test_missing_required_field(self, field):
        data = json.loads(_valid_json())
        del data[field]
        r = validate_llm_response(json.dumps(data))
        assert r.valid is False
        assert field in r.rejection_reason.lower()


class TestActionValidation:
    """Action must be in PATCH11_ACTIONS enum."""

    def test_invalid_action(self):
        r = validate_llm_response(_valid_json(action="FLIP"))
        assert r.valid is False
        assert "action" in r.rejection_reason.lower()

    def test_invalid_action_lowercase(self):
        r = validate_llm_response(_valid_json(action="reduce_25"))
        assert r.valid is False

    def test_ebv1_action_rejected(self):
        r = validate_llm_response(_valid_json(action="CLOSE_FULL"))
        assert r.valid is False


class TestConfidenceValidation:
    """Confidence must be in [0, 1]."""

    def test_negative_confidence(self):
        r = validate_llm_response(_valid_json(confidence=-0.1))
        assert r.valid is False

    def test_above_one_confidence(self):
        r = validate_llm_response(_valid_json(confidence=1.01))
        assert r.valid is False

    def test_string_confidence(self):
        r = validate_llm_response(_valid_json(confidence="high"))
        assert r.valid is False


class TestReasonCodeValidation:
    """Reason codes must be in VALID_REASON_CODES."""

    def test_invalid_reason_code(self):
        r = validate_llm_response(_valid_json(reason_codes=["MADE_UP_CODE"]))
        assert r.valid is False
        assert "reason" in r.rejection_reason.lower()

    def test_empty_reason_codes_for_non_hold(self):
        r = validate_llm_response(_valid_json(action="REDUCE_25", reason_codes=[]))
        assert r.valid is False


class TestForbiddenKeys:
    """Execution parameters must never be present."""

    @pytest.mark.parametrize(
        "key", ["quantity", "size", "amount", "price", "order_type", "leverage", "stop_loss"]
    )
    def test_forbidden_key_rejected(self, key):
        data = json.loads(_valid_json())
        data[key] = "INJECTED"
        r = validate_llm_response(json.dumps(data))
        assert r.valid is False
        assert "forbidden" in r.rejection_reason.lower()


class TestUnknownKeys:
    """Strict mode rejects unexpected top-level keys."""

    def test_unknown_key_rejected(self):
        data = json.loads(_valid_json())
        data["extra_field"] = "unwanted"
        r = validate_llm_response(json.dumps(data))
        assert r.valid is False
        assert "unknown" in r.rejection_reason.lower()


class TestWhyNotValidation:
    """why_not keys must be valid PATCH-11 actions."""

    def test_invalid_why_not_key(self):
        r = validate_llm_response(
            _valid_json(why_not={"INVALID_ACTION": "reason"})
        )
        assert r.valid is False
        assert "why_not" in r.rejection_reason.lower()


class TestRiskNoteLength:
    """risk_note must not exceed max length."""

    def test_very_long_risk_note(self):
        r = validate_llm_response(_valid_json(risk_note="x" * 1001))
        assert r.valid is False
        assert "risk_note" in r.rejection_reason.lower()
