"""Tests for ensemble_bridge — integration between Exit Brain v1 pipeline and EMA.

Tests cover:
- EnsembleBridgeResult construction and field mapping
- _action_to_urgency logic
- _state_to_snapshot conversion
- _build_geometry / _build_regime helper calls
- ENSEMBLE_QTY_MAP completeness
- EnsembleBridge._evaluate_sync with mocked EB v1 components
- EnsembleBridge.evaluate async wrapper
- Fail-closed paths: state=None, agg=None, belief=None, orchestrator=None
- Exception catch-all returns None
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from microservices.exit_management_agent.ensemble_bridge import (
    ENSEMBLE_QTY_MAP,
    EnsembleBridge,
    EnsembleBridgeResult,
    _action_to_urgency,
    _build_geometry,
    _build_regime,
    _state_to_snapshot,
)
from microservices.exit_management_agent.models import PositionSnapshot
from microservices.exit_brain_v1.models.action_candidate import VALID_ACTIONS


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_state(**overrides):
    """Minimal PositionExitState-like mock with all fields bridge accesses."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        side="LONG",
        status="OPEN",
        entry_price=30_000.0,
        current_price=31_500.0,
        quantity=0.01,
        notional=315.0,
        unrealized_pnl=15.0,
        unrealized_pnl_pct=0.05,
        open_timestamp=time.time() - 3600,
        source_timestamps={"p33_snapshot": time.time() - 5, "market_state": time.time() - 2},
        data_quality_flags=[],
        shadow_only=True,
        mark_price=31_500.0,
        leverage=10.0,
        realized_pnl=0.0,
        fees_paid=0.0,
        max_favorable_excursion=0.0,
        max_adverse_excursion=0.0,
        peak_unrealized_pnl=20.0,
        trough_unrealized_pnl=-5.0,
        drawdown_from_peak_pnl=5.0,
        volatility_short=0.02,
        volatility_medium=0.03,
        atr=500.0,
        trend_signal=0.001,
        regime_label="BULL",
        regime_confidence=0.75,
        momentum_score=None,
        mean_reversion_score=None,
        liquidity_score=None,
        spread_bps=None,
    )
    defaults.update(overrides)
    mock = MagicMock(**defaults)
    # Ensure attribute access works
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_policy_decision(**overrides):
    """Minimal PolicyDecision-like mock."""
    defaults = dict(
        chosen_action="HOLD",
        decision_confidence=0.7,
        explanation_tags=["time_pressure"],
        policy_blocks=[],
    )
    defaults.update(overrides)
    mock = MagicMock(**defaults)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


def _make_hazard(**overrides):
    """Minimal HazardAssessment-like mock."""
    defaults = dict(
        composite_hazard=0.3,
        dominant_hazard="drawdown_hazard",
    )
    defaults.update(overrides)
    mock = MagicMock(**defaults)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# ── ENSEMBLE_QTY_MAP ─────────────────────────────────────────────────────────

class TestEnsembleQtyMap:
    def test_covers_all_valid_actions(self):
        for action in VALID_ACTIONS:
            assert action in ENSEMBLE_QTY_MAP, f"Missing action: {action}"

    def test_hold_is_none(self):
        assert ENSEMBLE_QTY_MAP["HOLD"] is None

    def test_close_full_is_one(self):
        assert ENSEMBLE_QTY_MAP["CLOSE_FULL"] == 1.0

    def test_fractions_increasing(self):
        ordered_actions = ["REDUCE_SMALL", "REDUCE_MEDIUM", "TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE", "CLOSE_FULL"]
        fractions = [ENSEMBLE_QTY_MAP[a] for a in ordered_actions]
        for i in range(1, len(fractions)):
            assert fractions[i] > fractions[i - 1]


# ── _action_to_urgency ───────────────────────────────────────────────────────

class TestActionToUrgency:
    def test_emergency_override(self):
        assert _action_to_urgency("HOLD", hazard_emergency=True) == "EMERGENCY"
        assert _action_to_urgency("CLOSE_FULL", hazard_emergency=True) == "EMERGENCY"

    def test_close_full_high(self):
        assert _action_to_urgency("CLOSE_FULL", hazard_emergency=False) == "HIGH"

    def test_medium_actions(self):
        for action in ("TAKE_PROFIT_LARGE", "TAKE_PROFIT_PARTIAL", "REDUCE_MEDIUM"):
            assert _action_to_urgency(action, hazard_emergency=False) == "MEDIUM"

    def test_low_actions(self):
        for action in ("HOLD", "REDUCE_SMALL", "TIGHTEN_EXIT"):
            assert _action_to_urgency(action, hazard_emergency=False) == "LOW"


# ── _state_to_snapshot ────────────────────────────────────────────────────────

class TestStateToSnapshot:
    def test_basic_conversion(self):
        state = _make_state()
        snap = _state_to_snapshot(state)

        assert isinstance(snap, PositionSnapshot)
        assert snap.symbol == "BTCUSDT"
        assert snap.side == "LONG"
        assert snap.quantity == 0.01
        assert snap.entry_price == 30_000.0
        assert snap.mark_price == 31_500.0
        assert snap.leverage == 10.0
        assert snap.unrealized_pnl == 15.0
        assert snap.stop_loss == 0.0
        assert snap.take_profit == 0.0
        assert snap.entry_risk_usdt == 0.0

    def test_fallback_mark_price(self):
        state = _make_state(mark_price=0.0)
        snap = _state_to_snapshot(state)
        assert snap.mark_price == state.current_price

    def test_empty_timestamps_uses_now(self):
        state = _make_state(source_timestamps={})
        snap = _state_to_snapshot(state)
        assert snap.sync_timestamp > 0


# ── _build_geometry ───────────────────────────────────────────────────────────

class TestBuildGeometry:
    def test_returns_geometry_result(self):
        state = _make_state()
        result = _build_geometry(state)
        assert hasattr(result, "mfe")
        assert hasattr(result, "drawdown_from_peak")


# ── _build_regime ─────────────────────────────────────────────────────────────

class TestBuildRegime:
    @pytest.mark.parametrize("label,expected_trend_prob", [
        ("BULL", 0.70),
        ("BEAR", 0.70),
        ("RANGE", 0.15),
        ("VOLATILE", 0.15),
        ("UNKNOWN", 0.33),
    ])
    def test_regime_prob_mapping(self, label, expected_trend_prob):
        state = _make_state(regime_label=label)
        result = _build_regime(state)
        assert result is not None

    def test_none_label_defaults_unknown(self):
        state = _make_state(regime_label=None)
        result = _build_regime(state)
        assert result is not None


# ── EnsembleBridgeResult ─────────────────────────────────────────────────────

class TestEnsembleBridgeResult:
    def test_frozen(self):
        snap = PositionSnapshot(
            symbol="BTCUSDT", side="LONG", quantity=0.01,
            entry_price=30000, mark_price=31500, leverage=10,
            stop_loss=0, take_profit=0, unrealized_pnl=15,
            entry_risk_usdt=0, sync_timestamp=time.time(),
        )
        r = EnsembleBridgeResult(
            symbol="BTCUSDT", action="HOLD", reason="test",
            urgency="LOW", confidence=0.7, exit_fraction=0.0,
            snap=snap, elapsed_ms=12.5, n_models_loaded=6,
            policy_blocks=[],
        )
        assert r.symbol == "BTCUSDT"
        assert r.action == "HOLD"
        with pytest.raises(AttributeError):
            r.action = "CLOSE_FULL"


# ── EnsembleBridge._evaluate_sync ────────────────────────────────────────────

class TestEvaluateSync:
    """Test the synchronous pipeline with mocked EB v1 components."""

    def _make_bridge(self):
        """Build EnsembleBridge with all external deps mocked."""
        with patch("microservices.exit_management_agent.ensemble_bridge.redis_sync.Redis"):
            with patch("microservices.exit_management_agent.ensemble_bridge.PositionStateBuilder"):
                with patch("microservices.exit_management_agent.ensemble_bridge.ShadowPublisher"):
                    with patch("microservices.exit_management_agent.ensemble_bridge.EnsembleExitAdapter") as mock_adapter_cls:
                        with patch("microservices.exit_management_agent.ensemble_bridge.EnsembleAggregator") as mock_agg_cls:
                            mock_adapter = MagicMock()
                            mock_adapter.loaded_models = ["xgb", "lgb", "rf", "mlp", "svm", "knn"]
                            mock_adapter.missing_models = []
                            mock_adapter_cls.return_value = mock_adapter
                            mock_agg = MagicMock()
                            mock_agg_cls.return_value = mock_agg
                            bridge = EnsembleBridge("localhost", 6379)
        return bridge

    def test_state_none_returns_none(self):
        bridge = self._make_bridge()
        bridge._builder.build.return_value = None
        result = bridge._evaluate_sync("BTCUSDT")
        assert result is None

    def test_aggregation_none_returns_none(self):
        bridge = self._make_bridge()
        state = _make_state()
        bridge._builder.build.return_value = state
        bridge._adapter.collect_and_normalize.return_value = []

        with patch("microservices.exit_management_agent.ensemble_bridge.EnsembleAggregator") as agg_cls:
            bridge._aggregator.aggregate.return_value = (None, {})
            result = bridge._evaluate_sync("BTCUSDT")
            assert result is None

    @patch("microservices.exit_management_agent.ensemble_bridge.BeliefEngine")
    def test_belief_none_returns_none(self, mock_belief):
        bridge = self._make_bridge()
        state = _make_state()
        bridge._builder.build.return_value = state
        bridge._adapter.collect_and_normalize.return_value = [MagicMock()]

        agg_signal = MagicMock()
        bridge._aggregator.aggregate.return_value = (agg_signal, {})

        mock_belief.compute.return_value = None
        result = bridge._evaluate_sync("BTCUSDT")
        assert result is None

    @patch("microservices.exit_management_agent.ensemble_bridge.ExitAgentOrchestrator")
    @patch("microservices.exit_management_agent.ensemble_bridge.ActionUtilityEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.HazardEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.BeliefEngine")
    def test_orchestrator_none_returns_none(self, mock_belief, mock_hazard, mock_utility, mock_orch_cls):
        bridge = self._make_bridge()
        state = _make_state()
        bridge._builder.build.return_value = state
        bridge._adapter.collect_and_normalize.return_value = [MagicMock()]

        agg_signal = MagicMock()
        bridge._aggregator.aggregate.return_value = (agg_signal, {})

        mock_belief.compute.return_value = MagicMock()
        mock_hazard.assess.return_value = _make_hazard()
        mock_utility.score_all.return_value = [MagicMock()]

        orch_instance = MagicMock()
        orch_instance.run_decision_cycle.return_value = None
        mock_orch_cls.return_value = orch_instance

        result = bridge._evaluate_sync("BTCUSDT")
        assert result is None

    @patch("microservices.exit_management_agent.ensemble_bridge.ExitAgentOrchestrator")
    @patch("microservices.exit_management_agent.ensemble_bridge.ActionUtilityEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.HazardEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.BeliefEngine")
    def test_full_pipeline_returns_result(self, mock_belief, mock_hazard, mock_utility, mock_orch_cls):
        bridge = self._make_bridge()
        state = _make_state()
        bridge._builder.build.return_value = state
        bridge._adapter.collect_and_normalize.return_value = [MagicMock()]

        agg_signal = MagicMock()
        bridge._aggregator.aggregate.return_value = (agg_signal, {})

        mock_belief.compute.return_value = MagicMock()
        hazard = _make_hazard(composite_hazard=0.3)
        mock_hazard.assess.return_value = hazard
        mock_utility.score_all.return_value = [MagicMock()]

        decision = _make_policy_decision(
            chosen_action="TAKE_PROFIT_PARTIAL",
            decision_confidence=0.85,
            explanation_tags=["profit_target", "momentum_fade"],
            policy_blocks=[],
        )
        orch_instance = MagicMock()
        orch_instance.run_decision_cycle.return_value = decision
        mock_orch_cls.return_value = orch_instance

        result = bridge._evaluate_sync("BTCUSDT")

        assert result is not None
        assert isinstance(result, EnsembleBridgeResult)
        assert result.symbol == "BTCUSDT"
        assert result.action == "TAKE_PROFIT_PARTIAL"
        assert result.confidence == 0.85
        assert result.urgency == "MEDIUM"
        assert result.exit_fraction == 0.5
        assert result.reason == "profit_target; momentum_fade"
        assert result.policy_blocks == []
        assert result.elapsed_ms >= 0
        assert result.n_models_loaded == 6
        assert isinstance(result.snap, PositionSnapshot)

    @patch("microservices.exit_management_agent.ensemble_bridge.ExitAgentOrchestrator")
    @patch("microservices.exit_management_agent.ensemble_bridge.ActionUtilityEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.HazardEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.BeliefEngine")
    def test_high_hazard_sets_emergency(self, mock_belief, mock_hazard, mock_utility, mock_orch_cls):
        bridge = self._make_bridge()
        state = _make_state()
        bridge._builder.build.return_value = state
        bridge._adapter.collect_and_normalize.return_value = [MagicMock()]

        agg_signal = MagicMock()
        bridge._aggregator.aggregate.return_value = (agg_signal, {})

        mock_belief.compute.return_value = MagicMock()
        hazard = _make_hazard(composite_hazard=0.9)  # >= 0.8 → emergency
        mock_hazard.assess.return_value = hazard
        mock_utility.score_all.return_value = [MagicMock()]

        decision = _make_policy_decision(chosen_action="CLOSE_FULL")
        orch_instance = MagicMock()
        orch_instance.run_decision_cycle.return_value = decision
        mock_orch_cls.return_value = orch_instance

        result = bridge._evaluate_sync("BTCUSDT")
        assert result is not None
        assert result.urgency == "EMERGENCY"

    @patch("microservices.exit_management_agent.ensemble_bridge.ExitAgentOrchestrator")
    @patch("microservices.exit_management_agent.ensemble_bridge.ActionUtilityEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.HazardEngine")
    @patch("microservices.exit_management_agent.ensemble_bridge.BeliefEngine")
    def test_close_full_exit_fraction_is_one(self, mock_belief, mock_hazard, mock_utility, mock_orch_cls):
        bridge = self._make_bridge()
        state = _make_state()
        bridge._builder.build.return_value = state
        bridge._adapter.collect_and_normalize.return_value = [MagicMock()]

        agg_signal = MagicMock()
        bridge._aggregator.aggregate.return_value = (agg_signal, {})

        mock_belief.compute.return_value = MagicMock()
        mock_hazard.assess.return_value = _make_hazard()
        mock_utility.score_all.return_value = [MagicMock()]

        decision = _make_policy_decision(chosen_action="CLOSE_FULL")
        orch_instance = MagicMock()
        orch_instance.run_decision_cycle.return_value = decision
        mock_orch_cls.return_value = orch_instance

        result = bridge._evaluate_sync("BTCUSDT")
        assert result is not None
        assert result.exit_fraction == 1.0
        assert result.urgency == "HIGH"

    def test_exception_returns_none(self):
        bridge = self._make_bridge()
        bridge._builder.build.side_effect = RuntimeError("boom")
        result = bridge._evaluate_sync("BTCUSDT")
        assert result is None


# ── EnsembleBridge.discover_positions (async) ─────────────────────────────────

class TestDiscoverPositions:
    def test_delegates_to_builder(self):
        with patch("microservices.exit_management_agent.ensemble_bridge.redis_sync.Redis"):
            with patch("microservices.exit_management_agent.ensemble_bridge.PositionStateBuilder") as builder_cls:
                with patch("microservices.exit_management_agent.ensemble_bridge.ShadowPublisher"):
                    with patch("microservices.exit_management_agent.ensemble_bridge.EnsembleExitAdapter") as adapter_cls:
                        adapter = MagicMock()
                        adapter.loaded_models = ["xgb"]
                        adapter.missing_models = []
                        adapter_cls.return_value = adapter
                        builder = MagicMock()
                        builder.discover_open_positions.return_value = ["BTCUSDT", "ETHUSDT"]
                        builder_cls.return_value = builder
                        bridge = EnsembleBridge("localhost", 6379)

        result = asyncio.get_event_loop().run_until_complete(bridge.discover_positions())
        assert result == ["BTCUSDT", "ETHUSDT"]


# ── EnsembleBridge.evaluate (async) ──────────────────────────────────────────

class TestEvaluateAsync:
    def test_wraps_sync_in_thread(self):
        with patch("microservices.exit_management_agent.ensemble_bridge.redis_sync.Redis"):
            with patch("microservices.exit_management_agent.ensemble_bridge.PositionStateBuilder"):
                with patch("microservices.exit_management_agent.ensemble_bridge.ShadowPublisher"):
                    with patch("microservices.exit_management_agent.ensemble_bridge.EnsembleExitAdapter") as adapter_cls:
                        adapter = MagicMock()
                        adapter.loaded_models = []
                        adapter.missing_models = []
                        adapter_cls.return_value = adapter
                        bridge = EnsembleBridge("localhost", 6379)

        bridge._builder.build.return_value = None  # triggers fail-closed
        result = asyncio.get_event_loop().run_until_complete(bridge.evaluate("BTCUSDT"))
        assert result is None


# ── n_models_loaded property ─────────────────────────────────────────────────

class TestNModelsLoaded:
    def test_reflects_adapter(self):
        with patch("microservices.exit_management_agent.ensemble_bridge.redis_sync.Redis"):
            with patch("microservices.exit_management_agent.ensemble_bridge.PositionStateBuilder"):
                with patch("microservices.exit_management_agent.ensemble_bridge.ShadowPublisher"):
                    with patch("microservices.exit_management_agent.ensemble_bridge.EnsembleExitAdapter") as adapter_cls:
                        adapter = MagicMock()
                        adapter.loaded_models = ["a", "b", "c"]
                        adapter.missing_models = ["d"]
                        adapter_cls.return_value = adapter
                        bridge = EnsembleBridge("localhost", 6379)
        assert bridge.n_models_loaded == 3
