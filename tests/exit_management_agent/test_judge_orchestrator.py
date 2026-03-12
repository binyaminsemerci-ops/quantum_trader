"""Tests for PATCH-11 judge orchestrator — primary → fallback → disagreement → deterministic.

All tests use mocked Groq clients — no real API calls.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from microservices.exit_management_agent.llm.judge_orchestrator import (
    JudgeOrchestrator,
    JudgeResult,
    _deterministic_safe_policy,
)
from microservices.exit_management_agent.llm.groq_client import GroqModelClient, ThrottleSkipError
from microservices.exit_management_agent.patch11_actions import PATCH11_ACTIONS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_response(action="REDUCE_25", confidence=0.78, **overrides) -> str:
    """Build a valid LLM JSON response string."""
    data = {
        "action": action,
        "confidence": confidence,
        "reason_codes": ["THESIS_DECAY"],
        "why_not": {"HOLD": "thesis weak"},
        "risk_note": "partial de-risk",
    }
    data.update(overrides)
    return json.dumps(data)


def _make_ctx(**overrides):
    """Build a minimal EnsemblePipelineContext mock with JSON-serializable fields."""
    state = MagicMock()
    state.symbol = overrides.pop("symbol", "BTCUSDT")
    state.side = "LONG"
    state.entry_price = 30000.0
    state.current_price = 31500.0
    state.unrealized_pnl = 15.0
    state.unrealized_pnl_pct = 0.05
    state.hold_seconds = 3600.0
    state.notional = overrides.pop("notional", 500.0)
    state.leverage = 10

    geometry = MagicMock()
    geometry.mfe = 1700.0
    geometry.mae = 200.0
    geometry.drawdown_from_peak = 200.0
    geometry.profit_protection_ratio = 0.88
    geometry.momentum_decay = -0.02

    regime = MagicMock()
    regime.regime_label = "TREND"
    regime.regime_confidence = 0.75
    regime.trend_alignment = 0.8
    regime.reversal_risk = 0.15
    regime.chop_risk = 0.1

    belief = MagicMock()
    belief.exit_pressure = 0.3
    belief.hold_conviction = 0.7
    belief.directional_edge = 0.4
    belief.uncertainty_total = overrides.pop("uncertainty", 0.2)

    hazard = MagicMock()
    hazard.composite_hazard = overrides.pop("composite_hazard", 0.3)
    hazard.dominant_hazard = "time_decay"
    hazard.drawdown_hazard = 0.1
    hazard.reversal_hazard = 0.15
    hazard.volatility_hazard = 0.2
    hazard.time_decay_hazard = 0.3
    hazard.regime_hazard = 0.1
    hazard.ensemble_hazard = overrides.pop("ensemble_hazard", 0.2)

    decision = MagicMock()
    decision.chosen_action = overrides.pop("ensemble_action", "HOLD")
    decision.decision_confidence = 0.75
    decision.reason_codes = ["CONVICTION_HIGH"]

    ctx = MagicMock()
    ctx.state = state
    ctx.belief = belief
    ctx.hazard = hazard
    ctx.decision = decision
    ctx.geometry = geometry
    ctx.regime = regime
    ctx.candidates = []  # empty list, no candidates
    ctx.agg_signal = MagicMock()
    return ctx


def _mock_client(response: Optional[str] = None, side_effect=None) -> GroqModelClient:
    """Build a mock GroqModelClient."""
    client = MagicMock(spec=GroqModelClient)
    if side_effect:
        client.chat = AsyncMock(side_effect=side_effect)
    else:
        client.chat = AsyncMock(return_value=response or _valid_response())
    return client


# ── Deterministic safe policy ─────────────────────────────────────────────────

class TestDeterministicSafePolicy:
    """Emergency deterministic fallback when both LLMs fail."""

    def test_low_risk_holds(self):
        r = _deterministic_safe_policy(0.2, 0.1, "HOLD")
        assert r.action == "HOLD"
        assert r.source == "deterministic"

    def test_medium_risk_reduce_25(self):
        r = _deterministic_safe_policy(0.55, 0.2, "HOLD")
        assert r.action == "REDUCE_25"

    def test_high_risk_reduce_50(self):
        r = _deterministic_safe_policy(0.75, 0.3, "HOLD")
        assert r.action == "REDUCE_50"

    def test_high_toxicity_reduce_50(self):
        r = _deterministic_safe_policy(0.5, 0.75, "HOLD")
        assert r.action == "REDUCE_50"

    def test_emergency_full_close(self):
        r = _deterministic_safe_policy(0.95, 0.1, "HOLD")
        assert r.action == "FULL_CLOSE"

    def test_all_results_in_patch11(self):
        for ch in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for th in [0.0, 0.3, 0.7, 1.0]:
                r = _deterministic_safe_policy(ch, th, "HOLD")
                assert r.action in PATCH11_ACTIONS


# ── Normal path ───────────────────────────────────────────────────────────────

class TestNormalPath:
    """Primary succeeds, no soft failures → primary drives."""

    @pytest.mark.asyncio
    async def test_primary_success(self):
        primary = _mock_client(_valid_response("REDUCE_25", 0.85))
        fallback = _mock_client()
        judge = JudgeOrchestrator(primary, fallback)

        # Ensemble also recommends REDUCE_SMALL (maps to REDUCE_25) → no disagreement
        r = await judge.evaluate(ctx=_make_ctx(ensemble_action="REDUCE_SMALL"))
        assert r.action == "REDUCE_25"
        assert r.source == "primary"
        assert r.fallback_used is False
        assert r.confidence == 0.85
        fallback.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_primary_hold(self):
        primary = _mock_client(_valid_response("HOLD", 0.95, reason_codes=["EDGE_REMAINING"]))
        fallback = _mock_client()
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.action == "HOLD"
        assert r.source == "primary"


# ── Hard failure path ─────────────────────────────────────────────────────────

class TestHardFailure:
    """Primary fails (timeout, 429, invalid JSON) → fallback takes over."""

    @pytest.mark.asyncio
    async def test_timeout_falls_to_fallback(self):
        primary = _mock_client(side_effect=asyncio.TimeoutError())
        fallback = _mock_client(_valid_response("REDUCE_50", 0.7))
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.action == "REDUCE_50"
        assert r.source == "fallback"
        assert r.fallback_used is True

    @pytest.mark.asyncio
    async def test_throttle_falls_to_fallback(self):
        primary = _mock_client(side_effect=ThrottleSkipError("throttled"))
        fallback = _mock_client(_valid_response("HOLD", 0.8))
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.action == "HOLD"
        assert r.fallback_used is True

    @pytest.mark.asyncio
    async def test_invalid_json_falls_to_fallback(self):
        primary = _mock_client("this is not json")
        fallback = _mock_client(_valid_response("REDUCE_25", 0.75))
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.action == "REDUCE_25"
        assert r.fallback_used is True

    @pytest.mark.asyncio
    async def test_429_falls_to_fallback(self):
        exc = Exception("rate limited")
        exc.status = 429
        primary = _mock_client(side_effect=exc)
        fallback = _mock_client(_valid_response("HOLD", 0.9))
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.fallback_used is True

    @pytest.mark.asyncio
    async def test_5xx_falls_to_fallback(self):
        exc = Exception("server error")
        exc.status = 502
        primary = _mock_client(side_effect=exc)
        fallback = _mock_client(_valid_response("HOLD", 0.8))
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.fallback_used is True


# ── Double failure ────────────────────────────────────────────────────────────

class TestDoubleFailure:
    """Both LLMs fail → deterministic safe policy."""

    @pytest.mark.asyncio
    async def test_both_timeout(self):
        primary = _mock_client(side_effect=asyncio.TimeoutError())
        fallback = _mock_client(side_effect=asyncio.TimeoutError())
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx(composite_hazard=0.3))
        assert r.source == "deterministic"
        assert r.action in PATCH11_ACTIONS

    @pytest.mark.asyncio
    async def test_double_failure_emergency(self):
        primary = _mock_client(side_effect=asyncio.TimeoutError())
        fallback = _mock_client(side_effect=asyncio.TimeoutError())
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx(composite_hazard=0.95))
        assert r.action == "FULL_CLOSE"
        assert r.source == "deterministic"

    @pytest.mark.asyncio
    async def test_primary_invalid_fallback_timeout(self):
        primary = _mock_client("invalid json garbage")
        fallback = _mock_client(side_effect=asyncio.TimeoutError())
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.source == "deterministic"


# ── Soft failure → second opinion ─────────────────────────────────────────────

class TestSoftFailure:
    """Low confidence / contradictions → second opinion from fallback."""

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_second_opinion(self):
        primary = _mock_client(_valid_response("REDUCE_25", 0.40))
        fallback = _mock_client(_valid_response("REDUCE_25", 0.80))
        judge = JudgeOrchestrator(primary, fallback, confidence_threshold=0.60)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.second_opinion_used is True

    @pytest.mark.asyncio
    async def test_large_position_aggressive_action_triggers_second_opinion(self):
        # Large position + FULL_CLOSE → second opinion
        primary = _mock_client(_valid_response("FULL_CLOSE", 0.85))
        fallback = _mock_client(_valid_response("FULL_CLOSE", 0.80))
        judge = JudgeOrchestrator(primary, fallback, large_position_usdt=1000.0)

        r = await judge.evaluate(ctx=_make_ctx(notional=5000.0))
        assert r.second_opinion_used is True


# ── JudgeResult validation ────────────────────────────────────────────────────

class TestJudgeResult:
    """Result dataclass invariants."""

    @pytest.mark.asyncio
    async def test_result_has_latency(self):
        primary = _mock_client(_valid_response())
        fallback = _mock_client()
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_result_action_always_valid(self):
        primary = _mock_client(_valid_response("HARVEST_70_KEEP_30", 0.9))
        fallback = _mock_client()
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.action in PATCH11_ACTIONS

    @pytest.mark.asyncio
    async def test_qty_fraction_set(self):
        primary = _mock_client(_valid_response("REDUCE_50", 0.8))
        fallback = _mock_client()
        judge = JudgeOrchestrator(primary, fallback)

        r = await judge.evaluate(ctx=_make_ctx())
        assert r.qty_fraction == 0.50
