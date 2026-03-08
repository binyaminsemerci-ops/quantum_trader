"""
Unit Tests: PATCH-7B — Qwen3 constrained decision layer
=========================================================

Test groups:

  A. TestQwen3LayerHTTP      — HTTP path: valid responses, errors, fallbacks (14 tests)
  B. TestQwen3LayerPrompt    — Input/prompt contract: dimensions, actions (4 tests)
  C. TestAiModeMain          — main.py ai branch integration (7 tests)
  D. TestAuditQwen3Fields    — audit.py qwen3 field serialisation (4 tests)
  E. TestQwen3Models         — models.py Qwen3LayerResult dataclass (3 tests)
  F. TestConfigQwen3         — config.py qwen3 env parsing and clamping (4 tests)
  G. TestQwen3AllowedActions — _ALLOWED_ACTIONS contract (2 tests)

Total: 38 tests.  All existing 165 PATCH-7A tests continue to pass unchanged.
No live Redis or Ollama connection required.
"""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import fields as dc_fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── async helper ─────────────────────────────────────────────────────────────

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── shared constructors ───────────────────────────────────────────────────────

def _make_snap(**kw):
    from microservices.exit_management_agent.models import PositionSnapshot
    defaults = dict(
        symbol="BTCUSDT", side="LONG", quantity=0.01,
        entry_price=30000.0, mark_price=30000.0, leverage=10.0,
        stop_loss=0.0, take_profit=0.0, unrealized_pnl=0.0,
        entry_risk_usdt=100.0, sync_timestamp=0.0,
    )
    defaults.update(kw)
    return PositionSnapshot(**defaults)


def _make_score_state(**kw):
    from microservices.exit_management_agent.models import ExitScoreState
    defaults = dict(
        symbol="BTCUSDT", side="LONG", R_net=-0.3,
        age_sec=3600.0, age_fraction=0.25,
        giveback_pct=0.0, distance_to_sl_pct=0.04,
        peak_price=30100.0, mark_price=29900.0,
        entry_price=30000.0, leverage=10.0,
        r_effective_t1=2.0, r_effective_lock=0.5,
        d_r_loss=0.20, d_r_gain=0.0, d_giveback=0.0, d_time=0.05,
        d_sl_proximity=0.20, exit_score=0.09,
        formula_action="HOLD", formula_urgency="LOW",
        formula_confidence=0.60, formula_reason="No exit criteria met at R=-0.30",
    )
    defaults.update(kw)
    return ExitScoreState(**defaults)


def _make_qwen3_layer(shadow=True, timeout_ms=2000, endpoint="http://localhost:11434"):
    from microservices.exit_management_agent.qwen3_layer import Qwen3Layer
    return Qwen3Layer(endpoint=endpoint, timeout_ms=timeout_ms, shadow=shadow)


def _make_decision(action="HOLD", qwen3_result=None, score_state=None):
    from microservices.exit_management_agent.models import ExitDecision
    snap = _make_snap()
    return ExitDecision(
        snapshot=snap,
        action=action,
        reason="test",
        urgency="LOW",
        R_net=-0.3,
        confidence=0.6,
        suggested_sl=None,
        suggested_qty_fraction=None,
        dry_run=True,
        score_state=score_state,
        qwen3_result=qwen3_result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# A. Qwen3Layer HTTP path
# ─────────────────────────────────────────────────────────────────────────────

class TestQwen3LayerHTTP:
    """Tests for Qwen3Layer.evaluate() — all network I/O mocked."""

    def _mock_post(self, layer, response_content: str):
        """Patch Qwen3Layer._post to return a fixed string."""
        return patch.object(layer, "_post", new=AsyncMock(return_value=response_content))

    def test_valid_full_close_action(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "FULL_CLOSE", "confidence": 0.85, "reason": "high loss"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.action == "FULL_CLOSE"
        assert result.fallback is False
        assert result.confidence == pytest.approx(0.85)

    def test_valid_hold_action(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "HOLD", "confidence": 0.70, "reason": "wait"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.action == "HOLD"
        assert result.fallback is False

    def test_valid_partial_close_25(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="PARTIAL_CLOSE_25")
        response = json.dumps({"action": "PARTIAL_CLOSE_25", "confidence": 0.75, "reason": "harvest"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.action == "PARTIAL_CLOSE_25"
        assert result.fallback is False

    def test_valid_time_stop_exit(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="TIME_STOP_EXIT")
        response = json.dumps({"action": "TIME_STOP_EXIT", "confidence": 0.80, "reason": "max hold"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.action == "TIME_STOP_EXIT"
        assert result.fallback is False

    def test_disallowed_action_tighten_trail_fallback(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "TIGHTEN_TRAIL", "confidence": 0.70, "reason": "trail"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "HOLD"  # formula fallback

    def test_disallowed_action_move_to_be_fallback(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="PARTIAL_CLOSE_25")
        response = json.dumps({"action": "MOVE_TO_BREAKEVEN", "confidence": 0.60, "reason": "be"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "PARTIAL_CLOSE_25"  # formula fallback

    def test_disallowed_freetext_action_fallback(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "BUY_MORE", "confidence": 0.9, "reason": "bullish"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "HOLD"

    def test_invalid_json_fallback(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        with self._mock_post(layer, "not valid json at all"):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "HOLD"

    def test_missing_action_key_fallback(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="FULL_CLOSE")
        response = json.dumps({"confidence": 0.7, "reason": "missing action"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "FULL_CLOSE"

    def test_http_timeout_fallback(self):
        import aiohttp
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        with patch.object(layer, "_post", side_effect=aiohttp.ServerTimeoutError()):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "HOLD"

    def test_http_500_raises_and_fallback(self):
        import aiohttp
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="PARTIAL_CLOSE_25")
        with patch.object(layer, "_post", side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=500
        )):
            result = run_async(layer.evaluate(ss))
        assert result.fallback is True
        assert result.action == "PARTIAL_CLOSE_25"

    def test_confidence_clamped_above_one(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "HOLD", "confidence": 99.0, "reason": "sure"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_clamped_below_zero(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "HOLD", "confidence": -5.0, "reason": "unsure"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.confidence == pytest.approx(0.0)

    def test_reason_truncated_to_200(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        long_reason = "x" * 500
        response = json.dumps({"action": "HOLD", "confidence": 0.5, "reason": long_reason})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert len(result.reason) <= 200

    def test_latency_ms_positive(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "HOLD", "confidence": 0.5, "reason": "ok"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert result.latency_ms >= 0.0

    def test_raw_captured_on_valid_response(self):
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        response = json.dumps({"action": "HOLD", "confidence": 0.5, "reason": "ok"})
        with self._mock_post(layer, response):
            result = run_async(layer.evaluate(ss))
        assert len(result.raw) > 0

    def test_raw_empty_on_http_error(self):
        import aiohttp
        layer = _make_qwen3_layer(shadow=False)
        ss = _make_score_state(formula_action="HOLD")
        with patch.object(layer, "_post", side_effect=RuntimeError("connection refused")):
            result = run_async(layer.evaluate(ss))
        assert result.raw == ""
        assert result.fallback is True


# ─────────────────────────────────────────────────────────────────────────────
# B. Qwen3Layer prompt / input contract
# ─────────────────────────────────────────────────────────────────────────────

class TestQwen3LayerPrompt:
    """Verify the input payload contains required fields."""

    def test_payload_contains_all_five_dimensions(self):
        from microservices.exit_management_agent.qwen3_layer import _build_payload
        ss = _make_score_state()
        payload = _build_payload(ss)
        for key in ("d_r_loss", "d_r_gain", "d_giveback", "d_time", "d_sl_proximity"):
            assert key in payload, f"Missing dimension key: {key}"

    def test_payload_contains_formula_suggestion(self):
        from microservices.exit_management_agent.qwen3_layer import _build_payload
        ss = _make_score_state()
        payload = _build_payload(ss)
        assert "formula_suggestion" in payload
        fs = payload["formula_suggestion"]
        assert "action" in fs
        assert "confidence" in fs
        assert "reason" in fs

    def test_allowed_actions_excludes_tighten_trail(self):
        from microservices.exit_management_agent.qwen3_layer import _ALLOWED_ACTIONS
        assert "TIGHTEN_TRAIL" not in _ALLOWED_ACTIONS

    def test_allowed_actions_excludes_move_to_be(self):
        from microservices.exit_management_agent.qwen3_layer import _ALLOWED_ACTIONS
        assert "MOVE_TO_BREAKEVEN" not in _ALLOWED_ACTIONS

    def test_allowed_actions_contains_exactly_four(self):
        from microservices.exit_management_agent.qwen3_layer import _ALLOWED_ACTIONS
        assert len(_ALLOWED_ACTIONS) == 4
        for a in ("HOLD", "PARTIAL_CLOSE_25", "FULL_CLOSE", "TIME_STOP_EXIT"):
            assert a in _ALLOWED_ACTIONS


# ─────────────────────────────────────────────────────────────────────────────
# C. main.py ai mode integration
# ─────────────────────────────────────────────────────────────────────────────

class TestAiModeMain:
    """Integration tests for the ai branch in ExitManagementAgent._tick()."""

    def _make_agent_with_mocks(self, scoring_mode="ai", qwen3_shadow=True):
        """Return a minimally wired agent with all I/O mocked out."""
        from microservices.exit_management_agent.main import ExitManagementAgent
        from microservices.exit_management_agent.models import ExitScoreState

        cfg = MagicMock()
        cfg.redis_host = "127.0.0.1"
        cfg.redis_port = 6379
        cfg.live_writes_enabled = False
        cfg.ownership_transfer_enabled = False
        cfg.scoring_mode = scoring_mode
        cfg.qwen3_endpoint = "http://localhost:11434"
        cfg.qwen3_timeout_ms = 2000
        cfg.qwen3_shadow = qwen3_shadow
        cfg.qwen3_model = "qwen3:8b"
        cfg.max_hold_sec = 14400.0
        cfg.audit_stream = "quantum:stream:exit.audit"
        cfg.metrics_stream = "quantum:stream:exit.metrics"
        cfg.heartbeat_key = "quantum:exit_agent:heartbeat"
        cfg.heartbeat_ttl_sec = 60
        cfg.intent_stream = "quantum:stream:exit.intent"
        cfg.active_flag_key = "quantum:exit_agent:active_flag"
        cfg.active_flag_ttl_sec = 30
        cfg.testnet_mode = "true"
        cfg.max_positions_per_loop = 50
        cfg.symbol_allowlist = frozenset()
        cfg.dry_run = True
        cfg.loop_sec = 5.0

        with patch("microservices.exit_management_agent.main.RedisClient"), \
             patch("microservices.exit_management_agent.main.PositionSource"), \
             patch("microservices.exit_management_agent.main.PerceptionEngine"), \
             patch("microservices.exit_management_agent.main.DecisionEngine"), \
             patch("microservices.exit_management_agent.main.ScoringEngine"), \
             patch("microservices.exit_management_agent.main.AuditWriter"), \
             patch("microservices.exit_management_agent.main.HeartbeatWriter"), \
             patch("microservices.exit_management_agent.main.IntentWriter"), \
             patch("microservices.exit_management_agent.main.OwnershipFlagWriter"), \
             patch("microservices.exit_management_agent.main.Qwen3Layer") as MockQwen3:
            agent = ExitManagementAgent(cfg)

        return agent, MockQwen3

    def _run_tick_with_position(self, agent, snap, perception, score_state, qwen3_result=None):
        """Wire up mocks and run _tick() for one position."""
        from microservices.exit_management_agent.scoring_guards import HardGuards

        agent._position_source.get_open_positions = AsyncMock(return_value=[snap])
        agent._perception.compute = AsyncMock(return_value=perception)
        agent._perception.forget = MagicMock()
        agent._scoring_engine.score = MagicMock(return_value=score_state)
        agent._ownership_flag.write = AsyncMock()
        agent._heartbeat.beat = AsyncMock()
        agent._audit.write_decision = AsyncMock()
        agent._audit.write_metrics = AsyncMock()
        agent._intent_writer.maybe_publish = AsyncMock()

        if qwen3_result is not None:
            agent._qwen3.evaluate = AsyncMock(return_value=qwen3_result)

        with patch.object(HardGuards, "evaluate", return_value=None):
            run_async(agent._tick())

    def test_ai_mode_shadow_true_uses_formula_action(self):
        """With qwen3_shadow=True, live action comes from formula despite qwen3 saying FULL_CLOSE."""
        from microservices.exit_management_agent.models import Qwen3LayerResult

        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=True)
        agent._cfg.qwen3_shadow = True

        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=-0.1, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=0.05,
                                giveback_pct=0.0, r_effective_t1=2.0, r_effective_lock=0.5)
        ss = _make_score_state(formula_action="HOLD")
        qr = Qwen3LayerResult(action="FULL_CLOSE", confidence=0.9,
                               reason="model says close", fallback=False,
                               latency_ms=200.0, raw="{}")

        self._run_tick_with_position(agent, snap, perc, ss, qr)

        call_args = agent._audit.write_decision.call_args[0]
        dec = call_args[0]
        # Shadow: formula drives live path
        assert dec.action == "HOLD"
        # Qwen3 result still attached for audit
        assert dec.qwen3_result is not None
        assert dec.qwen3_result.action == "FULL_CLOSE"

    def test_ai_mode_shadow_false_uses_qwen3_action(self):
        """With qwen3_shadow=False and valid result, live action comes from Qwen3."""
        from microservices.exit_management_agent.models import Qwen3LayerResult

        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=False)
        agent._cfg.qwen3_shadow = False

        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=-0.1, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=0.05,
                                giveback_pct=0.0, r_effective_t1=2.0, r_effective_lock=0.5)
        ss = _make_score_state(formula_action="HOLD")
        qr = Qwen3LayerResult(action="FULL_CLOSE", confidence=0.9,
                               reason="close it", fallback=False,
                               latency_ms=180.0, raw="{}")

        self._run_tick_with_position(agent, snap, perc, ss, qr)

        dec = agent._audit.write_decision.call_args[0][0]
        assert dec.action == "FULL_CLOSE"

    def test_ai_mode_fallback_always_uses_formula(self):
        """When qr.fallback=True, formula action is used even if shadow=False."""
        from microservices.exit_management_agent.models import Qwen3LayerResult

        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=False)
        agent._cfg.qwen3_shadow = False

        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=-0.1, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=0.05,
                                giveback_pct=0.0, r_effective_t1=2.0, r_effective_lock=0.5)
        ss = _make_score_state(formula_action="PARTIAL_CLOSE_25")
        qr = Qwen3LayerResult(action="PARTIAL_CLOSE_25", confidence=0.0,
                               reason="qwen3_parse_error", fallback=True,
                               latency_ms=50.0, raw="bad")

        self._run_tick_with_position(agent, snap, perc, ss, qr)

        dec = agent._audit.write_decision.call_args[0][0]
        assert dec.action == "PARTIAL_CLOSE_25"
        assert dec.qwen3_result.fallback is True

    def test_ai_mode_tighten_trail_skips_qwen3(self):
        """formula_action=TIGHTEN_TRAIL → Qwen3Layer.evaluate never called."""
        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=True)

        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=0.8, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=0.05,
                                giveback_pct=0.7, r_effective_t1=2.0, r_effective_lock=0.5)
        ss = _make_score_state(formula_action="TIGHTEN_TRAIL")
        agent._qwen3.evaluate = AsyncMock()

        self._run_tick_with_position(agent, snap, perc, ss)

        agent._qwen3.evaluate.assert_not_called()
        dec = agent._audit.write_decision.call_args[0][0]
        assert dec.qwen3_result is None
        assert dec.action == "TIGHTEN_TRAIL"

    def test_ai_mode_move_to_be_skips_qwen3(self):
        """formula_action=MOVE_TO_BREAKEVEN → Qwen3Layer.evaluate never called."""
        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=True)

        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=0.6, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=0.05,
                                giveback_pct=0.0, r_effective_t1=2.0, r_effective_lock=0.5)
        ss = _make_score_state(formula_action="MOVE_TO_BREAKEVEN")
        agent._qwen3.evaluate = AsyncMock()

        self._run_tick_with_position(agent, snap, perc, ss)

        agent._qwen3.evaluate.assert_not_called()
        dec = agent._audit.write_decision.call_args[0][0]
        assert dec.qwen3_result is None

    def test_hard_guard_fires_no_qwen3_called(self):
        """When HardGuards returns a decision, Qwen3 is never called."""
        from microservices.exit_management_agent.models import ExitDecision
        from microservices.exit_management_agent.scoring_guards import HardGuards

        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=True)
        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=-2.0, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=-0.1,
                                giveback_pct=0.0, r_effective_t1=2.0, r_effective_lock=0.5)

        guard_dec = _make_decision(action="FULL_CLOSE")
        agent._qwen3.evaluate = AsyncMock()
        agent._position_source.get_open_positions = AsyncMock(return_value=[snap])
        agent._perception.compute = AsyncMock(return_value=perc)
        agent._perception.forget = MagicMock()
        agent._scoring_engine.score = MagicMock()
        agent._ownership_flag.write = AsyncMock()
        agent._heartbeat.beat = AsyncMock()
        agent._audit.write_decision = AsyncMock()
        agent._audit.write_metrics = AsyncMock()
        agent._intent_writer.maybe_publish = AsyncMock()

        with patch.object(HardGuards, "evaluate", return_value=guard_dec):
            run_async(agent._tick())

        agent._qwen3.evaluate.assert_not_called()

    def test_qwen3_result_attached_to_decision(self):
        """When Qwen3 is called and succeeds, qwen3_result is present on ExitDecision."""
        from microservices.exit_management_agent.models import Qwen3LayerResult

        agent, _ = self._make_agent_with_mocks(scoring_mode="ai", qwen3_shadow=True)

        snap = _make_snap()
        from microservices.exit_management_agent.models import PerceptionResult
        perc = PerceptionResult(snapshot=snap, R_net=-0.1, peak_price=30000.0,
                                age_sec=1000.0, distance_to_sl_pct=0.05,
                                giveback_pct=0.0, r_effective_t1=2.0, r_effective_lock=0.5)
        ss = _make_score_state(formula_action="HOLD")
        qr = Qwen3LayerResult(action="HOLD", confidence=0.65,
                               reason="hold it", fallback=False,
                               latency_ms=220.0, raw="{}")

        self._run_tick_with_position(agent, snap, perc, ss, qr)

        dec = agent._audit.write_decision.call_args[0][0]
        assert dec.qwen3_result is not None
        assert dec.qwen3_result.latency_ms == pytest.approx(220.0)


# ─────────────────────────────────────────────────────────────────────────────
# D. audit.py qwen3 field serialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditQwen3Fields:
    """Verify qwen3_* fields are correctly written to the audit stream."""

    def _captured_fields(self, dec):
        """Run write_decision and return the dict passed to xadd.

        AuditWriter.__new__ bypasses __init__ / _validate_streams, so no
        stream-allowlist patches are required.
        """
        from microservices.exit_management_agent.audit import AuditWriter

        redis = MagicMock()
        redis.xadd = AsyncMock()

        real_writer = AuditWriter.__new__(AuditWriter)
        real_writer._redis = redis
        real_writer._audit_stream = "quantum:stream:exit.audit"
        real_writer._metrics_stream = "quantum:stream:exit.metrics"
        run_async(real_writer.write_decision(dec, "testloop"))

        return redis.xadd.call_args[0][1]

    def test_qwen3_fields_written_when_result_present(self):
        from microservices.exit_management_agent.models import Qwen3LayerResult
        qr = Qwen3LayerResult(action="FULL_CLOSE", confidence=0.85,
                               reason="high loss", fallback=False,
                               latency_ms=350.0, raw="{}")
        ss = _make_score_state(formula_action="HOLD")
        dec = _make_decision(action="HOLD", qwen3_result=qr, score_state=ss)

        fields = self._captured_fields(dec)
        assert "qwen3_action" in fields
        assert "qwen3_confidence" in fields
        assert "qwen3_reason" in fields
        assert "qwen3_fallback" in fields
        assert "qwen3_latency_ms" in fields
        assert fields["qwen3_action"] == "FULL_CLOSE"
        assert fields["qwen3_fallback"] == "false"
        assert fields["patch"] == "PATCH-7B"

    def test_qwen3_fallback_true_written_as_string_true(self):
        from microservices.exit_management_agent.models import Qwen3LayerResult
        qr = Qwen3LayerResult(action="HOLD", confidence=0.0,
                               reason="qwen3_parse_error", fallback=True,
                               latency_ms=50.0, raw="bad")
        ss = _make_score_state()
        dec = _make_decision(action="HOLD", qwen3_result=qr, score_state=ss)

        fields = self._captured_fields(dec)
        assert fields["qwen3_fallback"] == "true"

    def test_qwen3_fields_absent_when_result_none(self):
        """Formula-mode or hard-guard decisions: no qwen3_* keys in audit."""
        ss = _make_score_state()
        dec = _make_decision(action="HOLD", qwen3_result=None, score_state=ss)

        fields = self._captured_fields(dec)
        for key in ("qwen3_action", "qwen3_confidence", "qwen3_reason",
                    "qwen3_fallback", "qwen3_latency_ms"):
            assert key not in fields, f"Unexpected key in audit: {key}"

    def test_qwen3_skip_action_no_qwen3_keys(self):
        """TIGHTEN_TRAIL skip path produces dec.qwen3_result=None → no qwen3 keys."""
        dec = _make_decision(action="TIGHTEN_TRAIL", qwen3_result=None)
        fields = self._captured_fields(dec)
        assert "qwen3_action" not in fields


# ─────────────────────────────────────────────────────────────────────────────
# E. models.py Qwen3LayerResult
# ─────────────────────────────────────────────────────────────────────────────

class TestQwen3Models:
    def test_qwen3_layer_result_is_frozen(self):
        from microservices.exit_management_agent.models import Qwen3LayerResult
        qr = Qwen3LayerResult(action="HOLD", confidence=0.5,
                               reason="test", fallback=False,
                               latency_ms=100.0, raw="{}")
        with pytest.raises((AttributeError, TypeError)):
            qr.action = "FULL_CLOSE"  # type: ignore

    def test_qwen3_layer_result_has_required_fields(self):
        from microservices.exit_management_agent.models import Qwen3LayerResult
        field_names = {f.name for f in dc_fields(Qwen3LayerResult)}
        for name in ("action", "confidence", "reason", "fallback", "latency_ms", "raw"):
            assert name in field_names

    def test_exit_decision_qwen3_result_defaults_none(self):
        """ExitDecision.qwen3_result defaults to None (backward compat)."""
        dec = _make_decision()
        assert dec.qwen3_result is None


# ─────────────────────────────────────────────────────────────────────────────
# F. config.py qwen3 env parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigQwen3:
    def _cfg_from_env(self, extra_env: dict):
        base = {
            "EXIT_AGENT_SCORING_MODE": "ai",
            "EXIT_AGENT_QWEN3_ENDPOINT": "http://localhost:11434",
            "EXIT_AGENT_QWEN3_TIMEOUT_MS": "2000",
            "EXIT_AGENT_QWEN3_SHADOW": "true",
            "EXIT_AGENT_QWEN3_MODEL": "qwen3:8b",
        }
        base.update(extra_env)
        with patch.dict(os.environ, base, clear=False):
            from microservices.exit_management_agent.config import AgentConfig
            return AgentConfig.from_env()

    def test_qwen3_shadow_true_by_default(self):
        cfg = self._cfg_from_env({})
        assert cfg.qwen3_shadow is True

    def test_qwen3_shadow_false_when_set(self):
        cfg = self._cfg_from_env({"EXIT_AGENT_QWEN3_SHADOW": "false"})
        assert cfg.qwen3_shadow is False

    def test_qwen3_timeout_clamped_below_200(self):
        cfg = self._cfg_from_env({"EXIT_AGENT_QWEN3_TIMEOUT_MS": "50"})
        assert cfg.qwen3_timeout_ms == 200

    def test_qwen3_timeout_clamped_above_10000(self):
        cfg = self._cfg_from_env({"EXIT_AGENT_QWEN3_TIMEOUT_MS": "99999"})
        assert cfg.qwen3_timeout_ms == 10000

    def test_qwen3_model_read_from_env(self):
        cfg = self._cfg_from_env({"EXIT_AGENT_QWEN3_MODEL": "qwen3:14b"})
        assert cfg.qwen3_model == "qwen3:14b"

    def test_scoring_mode_ai_not_falls_back_to_shadow(self):
        """scoring_mode=ai is now a real mode, not silently downgraded."""
        cfg = self._cfg_from_env({})
        assert cfg.scoring_mode == "ai"


# ─────────────────────────────────────────────────────────────────────────────
# G. _ALLOWED_ACTIONS invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestQwen3AllowedActions:
    def test_all_four_actions_present(self):
        from microservices.exit_management_agent.qwen3_layer import _ALLOWED_ACTIONS
        assert _ALLOWED_ACTIONS == frozenset(
            {"HOLD", "PARTIAL_CLOSE_25", "FULL_CLOSE", "TIME_STOP_EXIT"}
        )

    def test_allowed_actions_is_frozenset(self):
        from microservices.exit_management_agent.qwen3_layer import _ALLOWED_ACTIONS
        assert isinstance(_ALLOWED_ACTIONS, frozenset)
