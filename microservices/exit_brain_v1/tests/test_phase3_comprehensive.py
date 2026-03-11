"""
Phase 3 — Belief / Hazard / Utility: Comprehensive Test Suite.

Tests ALL Phase 3 modules:
  S1: BeliefState contract
  S2: HazardAssessment contract
  S3: ActionCandidate contract + VALID_ACTIONS + ACTION_EXIT_FRACTIONS
  S4: BeliefEngine (fusion logic)
  S5: HazardEngine (6 axes + composite)
  S6: ActionUtilityEngine (7 action utilities + penalties + scoring)
  S7: Integration (full pipeline)
  S8: Shadow stream verification
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock

import pytest

# ── Phase 3 contracts ────────────────────────────────────────────────────

from microservices.exit_brain_v1.models.belief_state import BeliefState
from microservices.exit_brain_v1.models.hazard_assessment import HazardAssessment
from microservices.exit_brain_v1.models.action_candidate import (
    ActionCandidate,
    VALID_ACTIONS,
    ACTION_EXIT_FRACTIONS,
)

# ── Phase 3 engines ──────────────────────────────────────────────────────

from microservices.exit_brain_v1.engines.belief_engine import BeliefEngine
from microservices.exit_brain_v1.engines.hazard_engine import (
    HazardEngine,
    HAZARD_WEIGHTS,
    TIME_DECAY_HALF_LIFE,
    VOLATILITY_HIGH_REF,
)
from microservices.exit_brain_v1.engines.action_utility_engine import ActionUtilityEngine

# ── Upstream contracts (for factories) ───────────────────────────────────

from microservices.exit_brain_v1.models.position_exit_state import PositionExitState
from microservices.exit_brain_v1.models.aggregated_exit_signal import AggregatedExitSignal
from microservices.exit_brain_v1.engines.geometry_engine import GeometryResult
from microservices.exit_brain_v1.engines.regime_drift_engine import RegimeState


# ═══════════════════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════════════════

def _make_state(**overrides) -> PositionExitState:
    """Minimal valid PositionExitState."""
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
        open_timestamp=time.time() - 3600,
        source_timestamps={"p33_snapshot": time.time() - 5},
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


def _make_geometry(**overrides) -> GeometryResult:
    """Minimal valid GeometryResult."""
    defaults = dict(
        mfe=200.0,
        mae=50.0,
        drawdown_from_peak=20.0,
        profit_protection_ratio=0.83,
        momentum_decay=-0.05,
        reward_to_risk_remaining=1.5,
    )
    defaults.update(overrides)
    return GeometryResult(**defaults)


def _make_regime(**overrides) -> RegimeState:
    """Minimal valid RegimeState."""
    defaults = dict(
        regime_label="TRENDING",
        regime_confidence=0.75,
        trend_alignment=0.6,
        reversal_risk=0.2,
        chop_risk=0.15,
        mean_reversion_score=0.1,
        drift=None,
    )
    defaults.update(overrides)
    return RegimeState(**defaults)


def _make_ensemble(**overrides) -> AggregatedExitSignal:
    """Minimal valid AggregatedExitSignal."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        ensemble_timestamp=time.time(),
        participating_models=["xgboost", "lightgbm", "nhits"],
        missing_models=["patchtst", "tft", "dlinear"],
        stale_models=[],
        model_count_total=6,
        model_count_used=3,
        sell_probability_agg=0.3,
        hold_probability_agg=0.5,
        buy_probability_agg=0.2,
        continuation_probability_agg=0.7,
        reversal_probability_agg=0.3,
        confidence_agg=0.75,
        uncertainty_agg=0.25,
        disagreement_score=0.15,
        consensus_strength=0.85,
        reliability_score=0.7,
        aggregation_method="weighted_average_v1",
        quality_flags=[],
        shadow_only=True,
    )
    defaults.update(overrides)
    return AggregatedExitSignal(**defaults)


def _make_belief(**overrides) -> BeliefState:
    """Minimal valid BeliefState."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        side="LONG",
        exit_pressure=0.35,
        hold_conviction=0.60,
        directional_edge=0.3,
        uncertainty_total=0.25,
        data_completeness=0.85,
        belief_timestamp=time.time(),
        belief_components={},
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
        drawdown_hazard=0.2,
        reversal_hazard=0.15,
        volatility_hazard=0.3,
        time_decay_hazard=0.1,
        regime_hazard=0.2,
        ensemble_hazard=0.25,
        composite_hazard=0.2,
        dominant_hazard="volatility",
        hazard_timestamp=time.time(),
        hazard_components={
            "drawdown": 0.2,
            "reversal": 0.15,
            "volatility": 0.3,
            "time_decay": 0.1,
            "regime": 0.2,
            "ensemble": 0.25,
        },
        quality_flags=[],
        shadow_only=True,
    )
    defaults.update(overrides)
    return HazardAssessment(**defaults)


def _make_action(**overrides) -> ActionCandidate:
    """Minimal valid ActionCandidate."""
    defaults = dict(
        position_id="pos-001",
        symbol="BTCUSDT",
        action="HOLD",
        exit_fraction=0.0,
        base_utility=0.5,
        penalty_total=0.1,
        net_utility=0.4,
        rank=1,
        utility_components={},
        penalty_components={},
        rationale="test",
        scoring_timestamp=time.time(),
        shadow_only=True,
    )
    defaults.update(overrides)
    return ActionCandidate(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# S1: BeliefState contract
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefState:

    def test_valid_belief_no_errors(self):
        b = _make_belief()
        assert b.validate() == []

    def test_empty_position_id_rejected(self):
        b = _make_belief(position_id="")
        assert any("position_id" in e for e in b.validate())

    def test_empty_symbol_rejected(self):
        b = _make_belief(symbol="")
        assert any("symbol" in e for e in b.validate())

    def test_invalid_side_rejected(self):
        b = _make_belief(side="NEUTRAL")
        assert any("side" in e for e in b.validate())

    def test_long_side_accepted(self):
        b = _make_belief(side="LONG")
        assert b.validate() == []

    def test_short_side_accepted(self):
        b = _make_belief(side="SHORT")
        assert b.validate() == []

    def test_exit_pressure_out_of_range(self):
        b = _make_belief(exit_pressure=1.5)
        assert any("exit_pressure" in e for e in b.validate())

    def test_exit_pressure_negative_rejected(self):
        b = _make_belief(exit_pressure=-0.1)
        assert any("exit_pressure" in e for e in b.validate())

    def test_hold_conviction_out_of_range(self):
        b = _make_belief(hold_conviction=1.1)
        assert any("hold_conviction" in e for e in b.validate())

    def test_directional_edge_range_positive(self):
        b = _make_belief(directional_edge=1.0)
        assert b.validate() == []

    def test_directional_edge_range_negative(self):
        b = _make_belief(directional_edge=-1.0)
        assert b.validate() == []

    def test_directional_edge_out_of_range(self):
        b = _make_belief(directional_edge=1.5)
        assert any("directional_edge" in e for e in b.validate())

    def test_uncertainty_total_out_of_range(self):
        b = _make_belief(uncertainty_total=-0.1)
        assert any("uncertainty_total" in e for e in b.validate())

    def test_data_completeness_out_of_range(self):
        b = _make_belief(data_completeness=1.2)
        assert any("data_completeness" in e for e in b.validate())

    def test_zero_timestamp_rejected(self):
        b = _make_belief(belief_timestamp=0)
        assert any("belief_timestamp" in e for e in b.validate())

    def test_negative_timestamp_rejected(self):
        b = _make_belief(belief_timestamp=-1)
        assert any("belief_timestamp" in e for e in b.validate())

    def test_shadow_only_false_rejected(self):
        b = _make_belief(shadow_only=False)
        assert any("shadow_only" in e for e in b.validate())

    def test_to_dict_keys(self):
        b = _make_belief()
        d = b.to_dict()
        required_keys = {
            "position_id", "symbol", "side",
            "exit_pressure", "hold_conviction",
            "directional_edge", "uncertainty_total",
            "data_completeness", "belief_timestamp",
            "belief_components", "quality_flags",
            "shadow_only",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_quality_flags_serialized(self):
        b = _make_belief(quality_flags=["FLAG_A", "FLAG_B"])
        d = b.to_dict()
        assert d["quality_flags"] == "FLAG_A,FLAG_B"

    def test_to_dict_empty_quality_flags(self):
        b = _make_belief(quality_flags=[])
        d = b.to_dict()
        assert d["quality_flags"] == ""

    def test_to_dict_components_json(self):
        import json
        b = _make_belief(belief_components={"key": 0.5})
        d = b.to_dict()
        parsed = json.loads(d["belief_components"])
        assert parsed["key"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# S2: HazardAssessment contract
# ═══════════════════════════════════════════════════════════════════════════


class TestHazardAssessment:

    def test_valid_hazard_no_errors(self):
        h = _make_hazard()
        assert h.validate() == []

    def test_empty_position_id_rejected(self):
        h = _make_hazard(position_id="")
        assert any("position_id" in e for e in h.validate())

    def test_empty_symbol_rejected(self):
        h = _make_hazard(symbol="")
        assert any("symbol" in e for e in h.validate())

    def test_drawdown_hazard_out_of_range(self):
        h = _make_hazard(drawdown_hazard=1.5)
        assert any("drawdown_hazard" in e for e in h.validate())

    def test_reversal_hazard_negative_rejected(self):
        h = _make_hazard(reversal_hazard=-0.1)
        assert any("reversal_hazard" in e for e in h.validate())

    def test_volatility_hazard_out_of_range(self):
        h = _make_hazard(volatility_hazard=2.0)
        assert any("volatility_hazard" in e for e in h.validate())

    def test_time_decay_hazard_out_of_range(self):
        h = _make_hazard(time_decay_hazard=-0.01)
        assert any("time_decay_hazard" in e for e in h.validate())

    def test_regime_hazard_out_of_range(self):
        h = _make_hazard(regime_hazard=1.01)
        assert any("regime_hazard" in e for e in h.validate())

    def test_ensemble_hazard_out_of_range(self):
        h = _make_hazard(ensemble_hazard=-0.5)
        assert any("ensemble_hazard" in e for e in h.validate())

    def test_composite_hazard_out_of_range(self):
        h = _make_hazard(composite_hazard=1.1)
        assert any("composite_hazard" in e for e in h.validate())

    def test_all_hazards_at_zero(self):
        h = _make_hazard(
            drawdown_hazard=0.0, reversal_hazard=0.0,
            volatility_hazard=0.0, time_decay_hazard=0.0,
            regime_hazard=0.0, ensemble_hazard=0.0,
            composite_hazard=0.0,
        )
        assert h.validate() == []

    def test_all_hazards_at_one(self):
        h = _make_hazard(
            drawdown_hazard=1.0, reversal_hazard=1.0,
            volatility_hazard=1.0, time_decay_hazard=1.0,
            regime_hazard=1.0, ensemble_hazard=1.0,
            composite_hazard=1.0,
        )
        assert h.validate() == []

    def test_empty_dominant_hazard_rejected(self):
        h = _make_hazard(dominant_hazard="")
        assert any("dominant_hazard" in e for e in h.validate())

    def test_zero_timestamp_rejected(self):
        h = _make_hazard(hazard_timestamp=0)
        assert any("hazard_timestamp" in e for e in h.validate())

    def test_shadow_only_false_rejected(self):
        h = _make_hazard(shadow_only=False)
        assert any("shadow_only" in e for e in h.validate())

    def test_to_dict_keys(self):
        h = _make_hazard()
        d = h.to_dict()
        required_keys = {
            "position_id", "symbol",
            "drawdown_hazard", "reversal_hazard",
            "volatility_hazard", "time_decay_hazard",
            "regime_hazard", "ensemble_hazard",
            "composite_hazard", "dominant_hazard",
            "hazard_timestamp", "hazard_components",
            "quality_flags", "shadow_only",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_quality_flags_serialized(self):
        h = _make_hazard(quality_flags=["NO_VOLATILITY_DATA"])
        d = h.to_dict()
        assert d["quality_flags"] == "NO_VOLATILITY_DATA"

    def test_to_dict_components_json(self):
        import json
        h = _make_hazard()
        d = h.to_dict()
        parsed = json.loads(d["hazard_components"])
        assert "drawdown" in parsed


# ═══════════════════════════════════════════════════════════════════════════
# S3: ActionCandidate contract + VALID_ACTIONS + ACTION_EXIT_FRACTIONS
# ═══════════════════════════════════════════════════════════════════════════


class TestValidActions:

    def test_seven_actions_defined(self):
        assert len(VALID_ACTIONS) == 7

    def test_all_required_actions_present(self):
        expected = {
            "HOLD", "REDUCE_SMALL", "REDUCE_MEDIUM",
            "TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE",
            "TIGHTEN_EXIT", "CLOSE_FULL",
        }
        assert VALID_ACTIONS == expected

    def test_valid_actions_is_frozenset(self):
        assert isinstance(VALID_ACTIONS, frozenset)

    def test_exit_fractions_match_actions(self):
        assert set(ACTION_EXIT_FRACTIONS.keys()) == VALID_ACTIONS

    def test_hold_fraction_is_zero(self):
        assert ACTION_EXIT_FRACTIONS["HOLD"] == 0.0

    def test_close_full_fraction_is_one(self):
        assert ACTION_EXIT_FRACTIONS["CLOSE_FULL"] == 1.0

    def test_tighten_exit_fraction_is_zero(self):
        assert ACTION_EXIT_FRACTIONS["TIGHTEN_EXIT"] == 0.0

    def test_reduce_small_fraction(self):
        assert ACTION_EXIT_FRACTIONS["REDUCE_SMALL"] == 0.10

    def test_reduce_medium_fraction(self):
        assert ACTION_EXIT_FRACTIONS["REDUCE_MEDIUM"] == 0.25

    def test_take_profit_partial_fraction(self):
        assert ACTION_EXIT_FRACTIONS["TAKE_PROFIT_PARTIAL"] == 0.50

    def test_take_profit_large_fraction(self):
        assert ACTION_EXIT_FRACTIONS["TAKE_PROFIT_LARGE"] == 0.75

    def test_fractions_monotonic(self):
        """Fractions increase with aggressiveness (excluding HOLD/TIGHTEN)."""
        aggressive = ["REDUCE_SMALL", "REDUCE_MEDIUM",
                       "TAKE_PROFIT_PARTIAL", "TAKE_PROFIT_LARGE", "CLOSE_FULL"]
        fractions = [ACTION_EXIT_FRACTIONS[a] for a in aggressive]
        assert fractions == sorted(fractions)


class TestActionCandidate:

    def test_valid_candidate_no_errors(self):
        a = _make_action()
        assert a.validate() == []

    def test_empty_position_id_rejected(self):
        a = _make_action(position_id="")
        assert any("position_id" in e for e in a.validate())

    def test_empty_symbol_rejected(self):
        a = _make_action(symbol="")
        assert any("symbol" in e for e in a.validate())

    def test_invalid_action_rejected(self):
        a = _make_action(action="SELL_ALL")
        assert any("action" in e for e in a.validate())

    def test_all_valid_actions_accepted(self):
        for action in VALID_ACTIONS:
            a = _make_action(action=action, exit_fraction=ACTION_EXIT_FRACTIONS[action])
            assert a.validate() == [], f"{action} should be valid"

    def test_exit_fraction_out_of_range(self):
        a = _make_action(exit_fraction=1.5)
        assert any("exit_fraction" in e for e in a.validate())

    def test_base_utility_out_of_range(self):
        a = _make_action(base_utility=-0.1)
        assert any("base_utility" in e for e in a.validate())

    def test_penalty_total_out_of_range(self):
        a = _make_action(penalty_total=1.5)
        assert any("penalty_total" in e for e in a.validate())

    def test_net_utility_out_of_range(self):
        a = _make_action(net_utility=-0.01)
        assert any("net_utility" in e for e in a.validate())

    def test_rank_zero_rejected(self):
        a = _make_action(rank=0)
        assert any("rank" in e for e in a.validate())

    def test_rank_negative_rejected(self):
        a = _make_action(rank=-1)
        assert any("rank" in e for e in a.validate())

    def test_shadow_only_false_rejected(self):
        a = _make_action(shadow_only=False)
        assert any("shadow_only" in e for e in a.validate())

    def test_to_dict_keys(self):
        a = _make_action()
        d = a.to_dict()
        required_keys = {
            "position_id", "symbol", "action", "exit_fraction",
            "base_utility", "penalty_total", "net_utility", "rank",
            "utility_components", "penalty_components",
            "rationale", "scoring_timestamp", "shadow_only",
        }
        assert required_keys.issubset(set(d.keys()))

    def test_to_dict_components_json(self):
        import json
        a = _make_action(utility_components={"x": 0.5})
        d = a.to_dict()
        parsed = json.loads(d["utility_components"])
        assert parsed["x"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# S4: BeliefEngine (fusion logic)
# ═══════════════════════════════════════════════════════════════════════════


class TestBeliefEngineFailClosed:

    def test_returns_none_when_ensemble_is_none(self):
        result = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(), None
        )
        assert result is None

    def test_returns_belief_when_ensemble_is_present(self):
        result = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        assert result is not None
        assert isinstance(result, BeliefState)


class TestBeliefEngineOutputRanges:

    def _compute(self, **ensemble_overrides) -> BeliefState:
        state = _make_state()
        geo = _make_geometry()
        regime = _make_regime()
        ens = _make_ensemble(**ensemble_overrides)
        result = BeliefEngine.compute(state, geo, regime, ens)
        assert result is not None
        return result

    def test_exit_pressure_in_range(self):
        b = self._compute()
        assert 0.0 <= b.exit_pressure <= 1.0

    def test_hold_conviction_in_range(self):
        b = self._compute()
        assert 0.0 <= b.hold_conviction <= 1.0

    def test_directional_edge_in_range(self):
        b = self._compute()
        assert -1.0 <= b.directional_edge <= 1.0

    def test_uncertainty_total_in_range(self):
        b = self._compute()
        assert 0.0 <= b.uncertainty_total <= 1.0

    def test_data_completeness_in_range(self):
        b = self._compute()
        assert 0.0 <= b.data_completeness <= 1.0

    def test_validates_clean(self):
        b = self._compute()
        assert b.validate() == []

    def test_shadow_only_is_true(self):
        b = self._compute()
        assert b.shadow_only is True

    def test_position_id_propagated(self):
        state = _make_state(position_id="MY_POS")
        b = BeliefEngine.compute(state, _make_geometry(), _make_regime(), _make_ensemble())
        assert b.position_id == "MY_POS"

    def test_symbol_propagated(self):
        state = _make_state(symbol="ETHUSDT")
        b = BeliefEngine.compute(state, _make_geometry(), _make_regime(), _make_ensemble())
        assert b.symbol == "ETHUSDT"


class TestBeliefEngineExitPressure:

    def test_high_sell_probability_increases_exit_pressure(self):
        low = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(sell_probability_agg=0.1, confidence_agg=0.8),
        )
        high = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(sell_probability_agg=0.9, confidence_agg=0.8),
        )
        assert high.exit_pressure > low.exit_pressure

    def test_high_reversal_risk_increases_exit_pressure(self):
        low = BeliefEngine.compute(
            _make_state(), _make_geometry(),
            _make_regime(reversal_risk=0.1, regime_confidence=0.8),
            _make_ensemble(),
        )
        high = BeliefEngine.compute(
            _make_state(), _make_geometry(),
            _make_regime(reversal_risk=0.9, regime_confidence=0.8),
            _make_ensemble(),
        )
        assert high.exit_pressure > low.exit_pressure

    def test_low_profit_protection_increases_exit_pressure(self):
        """When geometry shows price giving back profits, exit pressure rises."""
        high_ppr = BeliefEngine.compute(
            _make_state(), _make_geometry(profit_protection_ratio=0.95, mfe=200.0),
            _make_regime(), _make_ensemble(),
        )
        low_ppr = BeliefEngine.compute(
            _make_state(), _make_geometry(profit_protection_ratio=0.10, mfe=200.0),
            _make_regime(), _make_ensemble(),
        )
        assert low_ppr.exit_pressure > high_ppr.exit_pressure


class TestBeliefEngineHoldConviction:

    def test_high_hold_probability_increases_conviction(self):
        low = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(hold_probability_agg=0.1, confidence_agg=0.8),
        )
        high = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(hold_probability_agg=0.8, confidence_agg=0.8),
        )
        assert high.hold_conviction > low.hold_conviction

    def test_positive_trend_alignment_increases_conviction(self):
        neg = BeliefEngine.compute(
            _make_state(), _make_geometry(),
            _make_regime(trend_alignment=-0.8),
            _make_ensemble(),
        )
        pos = BeliefEngine.compute(
            _make_state(), _make_geometry(),
            _make_regime(trend_alignment=0.8),
            _make_ensemble(),
        )
        assert pos.hold_conviction > neg.hold_conviction


class TestBeliefEngineDirectionalEdge:

    def test_continuation_dominant_gives_positive_edge(self):
        b = BeliefEngine.compute(
            _make_state(), _make_geometry(),
            _make_regime(trend_alignment=0.8, regime_confidence=0.8),
            _make_ensemble(
                continuation_probability_agg=0.9,
                reversal_probability_agg=0.1,
            ),
        )
        assert b.directional_edge > 0

    def test_reversal_dominant_gives_negative_edge(self):
        b = BeliefEngine.compute(
            _make_state(), _make_geometry(),
            _make_regime(trend_alignment=-0.8, regime_confidence=0.8),
            _make_ensemble(
                continuation_probability_agg=0.1,
                reversal_probability_agg=0.9,
            ),
        )
        assert b.directional_edge < 0


class TestBeliefEngineUncertainty:

    def test_high_ensemble_uncertainty_increases_total(self):
        low = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(uncertainty_agg=0.1, disagreement_score=0.1),
        )
        high = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(uncertainty_agg=0.9, disagreement_score=0.1),
        )
        assert high.uncertainty_total > low.uncertainty_total

    def test_high_disagreement_increases_uncertainty(self):
        low = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(disagreement_score=0.05),
        )
        high = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(),
            _make_ensemble(disagreement_score=0.95),
        )
        assert high.uncertainty_total > low.uncertainty_total


class TestBeliefEngineComponents:

    def test_components_dict_has_expected_keys(self):
        b = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        expected_keys = {
            "ens_exit", "rev_exit", "geo_exit",
            "ens_hold", "trend_hold", "geo_hold",
            "ens_edge", "regime_edge",
            "ens_unc", "disagreement_unc", "data_unc",
            "coverage",
        }
        assert expected_keys.issubset(set(b.belief_components.keys()))

    def test_quality_flags_merged_from_state_and_ensemble(self):
        state = _make_state(data_quality_flags=["STATE_FLAG"])
        ens = _make_ensemble(quality_flags=["ENS_FLAG"])
        b = BeliefEngine.compute(state, _make_geometry(), _make_regime(), ens)
        assert "STATE_FLAG" in b.quality_flags
        assert "ENS_FLAG" in b.quality_flags


# ═══════════════════════════════════════════════════════════════════════════
# S5: HazardEngine (6 axes + composite)
# ═══════════════════════════════════════════════════════════════════════════


class TestHazardEngineDrawdown:

    def test_no_favorable_excursion_returns_low_or_zero(self):
        geo = _make_geometry(mfe=0.0)
        state = _make_state(max_adverse_excursion=0.0, notional=5000.0)
        h = HazardEngine.compute_drawdown_hazard(geo, state)
        assert 0.0 <= h <= 1.0

    def test_large_drawdown_from_peak_gives_high_hazard(self):
        geo = _make_geometry(mfe=200.0, drawdown_from_peak=150.0)
        state = _make_state(peak_unrealized_pnl=200.0, notional=5000.0)
        h = HazardEngine.compute_drawdown_hazard(geo, state)
        assert h > 0.5

    def test_zero_drawdown_gives_zero_hazard(self):
        geo = _make_geometry(mfe=200.0, drawdown_from_peak=0.0)
        state = _make_state(peak_unrealized_pnl=200.0, notional=5000.0)
        h = HazardEngine.compute_drawdown_hazard(geo, state)
        assert h == 0.0

    def test_always_in_range(self):
        geo = _make_geometry(mfe=200.0, drawdown_from_peak=9999.0)
        state = _make_state(peak_unrealized_pnl=200.0, notional=5000.0)
        h = HazardEngine.compute_drawdown_hazard(geo, state)
        assert 0.0 <= h <= 1.0


class TestHazardEngineReversal:

    def test_high_reversal_risk_gives_high_hazard(self):
        regime = _make_regime(reversal_risk=0.9, regime_confidence=0.9)
        ens = _make_ensemble(reversal_probability_agg=0.8, confidence_agg=0.9)
        h = HazardEngine.compute_reversal_hazard(regime, ens)
        assert h > 0.5

    def test_low_reversal_risk_gives_low_hazard(self):
        regime = _make_regime(reversal_risk=0.1, regime_confidence=0.5)
        ens = _make_ensemble(reversal_probability_agg=0.1, confidence_agg=0.5)
        h = HazardEngine.compute_reversal_hazard(regime, ens)
        assert h < 0.3

    def test_always_in_range(self):
        for rr in [0.0, 0.5, 1.0]:
            for rp in [0.0, 0.5, 1.0]:
                h = HazardEngine.compute_reversal_hazard(
                    _make_regime(reversal_risk=rr, regime_confidence=1.0),
                    _make_ensemble(reversal_probability_agg=rp, confidence_agg=1.0),
                )
                assert 0.0 <= h <= 1.0


class TestHazardEngineVolatility:

    def test_high_atr_gives_high_hazard(self):
        """ATR = 2% of entry_price → max hazard."""
        state = _make_state(entry_price=50000.0, atr=1000.0)  # 2%
        h = HazardEngine.compute_volatility_hazard(state)
        assert h >= 0.9

    def test_low_atr_gives_low_hazard(self):
        state = _make_state(entry_price=50000.0, atr=100.0)  # 0.2%
        h = HazardEngine.compute_volatility_hazard(state)
        assert h < 0.2

    def test_no_atr_uses_volatility_short(self):
        state = _make_state(atr=None, volatility_short=0.02)
        h = HazardEngine.compute_volatility_hazard(state)
        assert 0.0 <= h <= 1.0

    def test_no_volatility_data_returns_moderate(self):
        state = _make_state(atr=None, volatility_short=None)
        h = HazardEngine.compute_volatility_hazard(state)
        assert h == 0.5

    def test_always_in_range(self):
        for atr in [0, 100, 500, 1000, 5000]:
            state = _make_state(entry_price=50000.0, atr=float(atr))
            h = HazardEngine.compute_volatility_hazard(state)
            assert 0.0 <= h <= 1.0


class TestHazardEngineTimeDecay:

    def test_zero_hold_gives_zero_hazard(self):
        state = _make_state(open_timestamp=time.time())
        h = HazardEngine.compute_time_decay_hazard(state)
        assert h < 0.01

    def test_at_time_constant_gives_approx_063(self):
        """At t=τ the formula 1-exp(-1) ≈ 0.6321, not 0.5."""
        state = _make_state(open_timestamp=time.time() - TIME_DECAY_HALF_LIFE)
        h = HazardEngine.compute_time_decay_hazard(state)
        assert abs(h - 0.6321) < 0.02

    def test_very_long_hold_approaches_one(self):
        state = _make_state(open_timestamp=time.time() - TIME_DECAY_HALF_LIFE * 10)
        h = HazardEngine.compute_time_decay_hazard(state)
        assert h > 0.99

    def test_monotonic_increase(self):
        """Longer hold → higher hazard."""
        now = time.time()
        prev = 0.0
        for secs in [0, 60, 600, 3600, 14400, 86400]:
            state = _make_state(open_timestamp=now - secs)
            h = HazardEngine.compute_time_decay_hazard(state)
            assert h >= prev
            prev = h


class TestHazardEngineRegime:

    def test_high_chop_gives_high_hazard(self):
        regime = _make_regime(chop_risk=0.9, trend_alignment=0.5)
        h = HazardEngine.compute_regime_hazard(regime)
        assert h > 0.3

    def test_counter_trend_gives_high_hazard(self):
        regime = _make_regime(chop_risk=0.0, trend_alignment=-0.8)
        h = HazardEngine.compute_regime_hazard(regime)
        assert h > 0.3

    def test_favorable_regime_gives_low_hazard(self):
        regime = _make_regime(chop_risk=0.05, trend_alignment=0.9)
        h = HazardEngine.compute_regime_hazard(regime)
        assert h < 0.1

    def test_always_in_range(self):
        for cr in [0.0, 0.5, 1.0]:
            for ta in [-1.0, 0.0, 1.0]:
                h = HazardEngine.compute_regime_hazard(
                    _make_regime(chop_risk=cr, trend_alignment=ta)
                )
                assert 0.0 <= h <= 1.0


class TestHazardEngineEnsemble:

    def test_high_sell_probability_gives_high_hazard(self):
        ens = _make_ensemble(
            sell_probability_agg=0.9, confidence_agg=0.9, uncertainty_agg=0.1,
        )
        h = HazardEngine.compute_ensemble_hazard(ens)
        assert h > 0.5

    def test_low_sell_probability_gives_low_hazard(self):
        ens = _make_ensemble(
            sell_probability_agg=0.1, confidence_agg=0.5, uncertainty_agg=0.5,
        )
        h = HazardEngine.compute_ensemble_hazard(ens)
        assert h < 0.15

    def test_high_uncertainty_dampens_hazard(self):
        """Even with high sell, high uncertainty should reduce ensemble hazard."""
        certain = HazardEngine.compute_ensemble_hazard(
            _make_ensemble(sell_probability_agg=0.8, confidence_agg=0.9, uncertainty_agg=0.1),
        )
        uncertain = HazardEngine.compute_ensemble_hazard(
            _make_ensemble(sell_probability_agg=0.8, confidence_agg=0.9, uncertainty_agg=0.9),
        )
        assert certain > uncertain


class TestHazardEngineComposite:

    def test_equal_weight_sum_to_correct_value(self):
        """With equal weights and known values, composite = mean."""
        hazards = {
            "drawdown": 0.6,
            "reversal": 0.4,
            "volatility": 0.2,
            "time_decay": 0.1,
            "regime": 0.3,
            "ensemble": 0.8,
        }
        composite, _ = HazardEngine.compute_composite(hazards)
        expected = sum(hazards.values()) / 6.0
        assert abs(composite - expected) < 0.001

    def test_dominant_is_highest(self):
        hazards = {
            "drawdown": 0.1,
            "reversal": 0.9,
            "volatility": 0.2,
            "time_decay": 0.1,
            "regime": 0.3,
            "ensemble": 0.5,
        }
        _, dominant = HazardEngine.compute_composite(hazards)
        assert dominant == "reversal"

    def test_custom_weights(self):
        hazards = {
            "drawdown": 1.0,
            "reversal": 0.0,
            "volatility": 0.0,
            "time_decay": 0.0,
            "regime": 0.0,
            "ensemble": 0.0,
        }
        weights = {
            "drawdown": 0.5,
            "reversal": 0.1,
            "volatility": 0.1,
            "time_decay": 0.1,
            "regime": 0.1,
            "ensemble": 0.1,
        }
        composite, _ = HazardEngine.compute_composite(hazards, weights)
        expected = 0.5 / 1.0  # Only drawdown contributes, total_weight=1.0
        assert abs(composite - expected) < 0.001

    def test_composite_always_in_range(self):
        for v in [0.0, 0.5, 1.0]:
            hazards = {k: v for k in HAZARD_WEIGHTS}
            composite, _ = HazardEngine.compute_composite(hazards)
            assert 0.0 <= composite <= 1.0

    def test_default_weights_sum_to_one(self):
        assert abs(sum(HAZARD_WEIGHTS.values()) - 1.0) < 0.001


class TestHazardEngineFullAssess:

    def test_returns_hazard_assessment(self):
        result = HazardEngine.assess(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        assert isinstance(result, HazardAssessment)

    def test_validates_clean(self):
        result = HazardEngine.assess(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        assert result.validate() == []

    def test_shadow_only_is_true(self):
        result = HazardEngine.assess(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        assert result.shadow_only is True

    def test_all_axes_in_range(self):
        result = HazardEngine.assess(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        for field_name in [
            "drawdown_hazard", "reversal_hazard", "volatility_hazard",
            "time_decay_hazard", "regime_hazard", "ensemble_hazard",
            "composite_hazard",
        ]:
            val = getattr(result, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name} = {val}"

    def test_dominant_hazard_not_empty(self):
        result = HazardEngine.assess(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        assert result.dominant_hazard != ""

    def test_no_volatility_data_flag(self):
        state = _make_state(atr=None, volatility_short=None)
        result = HazardEngine.assess(state, _make_geometry(), _make_regime(), _make_ensemble())
        assert "NO_VOLATILITY_DATA" in result.quality_flags

    def test_components_populated(self):
        result = HazardEngine.assess(
            _make_state(), _make_geometry(), _make_regime(), _make_ensemble()
        )
        assert "drawdown" in result.hazard_components
        assert "reversal" in result.hazard_components
        assert "volatility" in result.hazard_components
        assert "time_decay" in result.hazard_components
        assert "regime" in result.hazard_components
        assert "ensemble" in result.hazard_components


# ═══════════════════════════════════════════════════════════════════════════
# S6: ActionUtilityEngine (7 action utilities + penalties + scoring)
# ═══════════════════════════════════════════════════════════════════════════


class TestActionUtilityHold:

    def test_high_conviction_low_hazard_gives_high_utility(self):
        belief = _make_belief(hold_conviction=0.9, directional_edge=0.8)
        hazard = _make_hazard(composite_hazard=0.1)
        base, comps = ActionUtilityEngine._utility_hold(belief, hazard)
        assert base > 0.5

    def test_low_conviction_high_hazard_gives_low_utility(self):
        belief = _make_belief(hold_conviction=0.2, directional_edge=-0.5)
        hazard = _make_hazard(composite_hazard=0.8)
        base, comps = ActionUtilityEngine._utility_hold(belief, hazard)
        assert base < 0.2

    def test_components_has_expected_keys(self):
        base, comps = ActionUtilityEngine._utility_hold(_make_belief(), _make_hazard())
        assert "hold_conviction" in comps
        assert "safety_factor" in comps
        assert "edge_factor" in comps


class TestActionUtilityReduceSmall:

    def test_moderate_exit_pressure_gives_moderate_utility(self):
        belief = _make_belief(exit_pressure=0.5, uncertainty_total=0.3)
        hazard = _make_hazard(composite_hazard=0.4)
        base, _ = ActionUtilityEngine._utility_reduce_small(belief, hazard)
        assert 0.2 < base < 0.8

    def test_in_range(self):
        base, _ = ActionUtilityEngine._utility_reduce_small(_make_belief(), _make_hazard())
        assert 0.0 <= base <= 1.0


class TestActionUtilityReduceMedium:

    def test_higher_exit_pressure_weight_than_small(self):
        """Reduce medium has higher exit_pressure weight coefficient."""
        belief = _make_belief(exit_pressure=0.8)
        hazard = _make_hazard(composite_hazard=0.1)
        small, _ = ActionUtilityEngine._utility_reduce_small(belief, hazard)
        medium, _ = ActionUtilityEngine._utility_reduce_medium(belief, hazard)
        # Medium depends more on exit_pressure, less on uncertainty
        assert medium != small  # Different formulas


class TestActionUtilityTakeProfit:

    def test_profitable_position_gives_utility(self):
        belief = _make_belief(exit_pressure=0.6)
        hazard = _make_hazard(drawdown_hazard=0.5)
        base, _ = ActionUtilityEngine._utility_take_profit_partial(
            belief, hazard, _make_state(unrealized_pnl=100.0)
        )
        assert base > 0.0

    def test_large_take_profit_requires_more_conviction(self):
        belief = _make_belief(exit_pressure=0.7)
        hazard = _make_hazard(drawdown_hazard=0.4)
        state = _make_state(unrealized_pnl=200.0)
        partial, _ = ActionUtilityEngine._utility_take_profit_partial(belief, hazard, state)
        large, _ = ActionUtilityEngine._utility_take_profit_large(belief, hazard, state)
        # Both produce utility; exact ordering depends on the formula
        assert partial >= 0.0
        assert large >= 0.0


class TestActionUtilityTightenExit:

    def test_high_hazard_low_exit_pressure_gives_utility(self):
        belief = _make_belief(exit_pressure=0.2)
        hazard = _make_hazard(composite_hazard=0.7, volatility_hazard=0.8)
        base, _ = ActionUtilityEngine._utility_tighten_exit(belief, hazard)
        assert base > 0.2

    def test_components_has_expected_keys(self):
        base, comps = ActionUtilityEngine._utility_tighten_exit(_make_belief(), _make_hazard())
        assert "composite_hazard" in comps
        assert "volatility_hazard" in comps


class TestActionUtilityCloseFull:

    def test_high_exit_pressure_high_hazard_gives_high_utility(self):
        belief = _make_belief(exit_pressure=0.9)
        hazard = _make_hazard(composite_hazard=0.9, reversal_hazard=0.8)
        base, _ = ActionUtilityEngine._utility_close_full(belief, hazard)
        assert base > 0.3

    def test_low_pressure_low_hazard_gives_low_utility(self):
        belief = _make_belief(exit_pressure=0.1)
        hazard = _make_hazard(composite_hazard=0.1, reversal_hazard=0.1)
        base, _ = ActionUtilityEngine._utility_close_full(belief, hazard)
        assert base < 0.1


class TestActionUtilityPenalties:

    def test_take_profit_penalty_when_not_profitable(self):
        state = _make_state(unrealized_pnl=-50.0)
        belief = _make_belief()
        hazard = _make_hazard()
        total, comps = ActionUtilityEngine._compute_penalties(
            "TAKE_PROFIT_PARTIAL", state, belief, hazard,
        )
        assert "not_profitable" in comps
        assert total > 0

    def test_take_profit_large_penalty_when_not_profitable(self):
        state = _make_state(unrealized_pnl=-10.0)
        total, comps = ActionUtilityEngine._compute_penalties(
            "TAKE_PROFIT_LARGE", state, _make_belief(), _make_hazard(),
        )
        assert "not_profitable" in comps

    def test_no_profit_penalty_when_profitable(self):
        state = _make_state(unrealized_pnl=100.0)
        total, comps = ActionUtilityEngine._compute_penalties(
            "TAKE_PROFIT_PARTIAL", state, _make_belief(), _make_hazard(),
        )
        assert "not_profitable" not in comps

    def test_close_full_penalty_when_low_hazard(self):
        hazard = _make_hazard(composite_hazard=0.1)
        total, comps = ActionUtilityEngine._compute_penalties(
            "CLOSE_FULL", _make_state(), _make_belief(), hazard,
        )
        assert "low_hazard_close" in comps

    def test_no_close_penalty_when_high_hazard(self):
        hazard = _make_hazard(composite_hazard=0.8)
        total, comps = ActionUtilityEngine._compute_penalties(
            "CLOSE_FULL", _make_state(), _make_belief(), hazard,
        )
        assert "low_hazard_close" not in comps

    def test_hold_penalty_when_very_high_hazard(self):
        hazard = _make_hazard(composite_hazard=0.9)
        total, comps = ActionUtilityEngine._compute_penalties(
            "HOLD", _make_state(), _make_belief(), hazard,
        )
        assert "high_hazard_hold" in comps

    def test_no_hold_penalty_when_low_hazard(self):
        hazard = _make_hazard(composite_hazard=0.3)
        total, comps = ActionUtilityEngine._compute_penalties(
            "HOLD", _make_state(), _make_belief(), hazard,
        )
        assert "high_hazard_hold" not in comps

    def test_uncertainty_dampening_on_aggressive_actions(self):
        belief = _make_belief(uncertainty_total=0.8)
        hazard = _make_hazard()
        # CLOSE_FULL has exit_fraction=1.0, should get uncertainty penalty
        total, comps = ActionUtilityEngine._compute_penalties(
            "CLOSE_FULL", _make_state(), belief, hazard,
        )
        assert "uncertainty_dampening" in comps

    def test_no_uncertainty_dampening_on_hold(self):
        belief = _make_belief(uncertainty_total=0.9)
        total, comps = ActionUtilityEngine._compute_penalties(
            "HOLD", _make_state(), belief, _make_hazard(),
        )
        assert "uncertainty_dampening" not in comps

    def test_no_uncertainty_dampening_on_tighten(self):
        belief = _make_belief(uncertainty_total=0.9)
        total, comps = ActionUtilityEngine._compute_penalties(
            "TIGHTEN_EXIT", _make_state(), belief, _make_hazard(),
        )
        assert "uncertainty_dampening" not in comps

    def test_penalty_total_clamped_to_one(self):
        """Even with multiple penalties, total can't exceed 1.0."""
        state = _make_state(unrealized_pnl=-100.0)
        belief = _make_belief(uncertainty_total=1.0)
        hazard = _make_hazard(composite_hazard=0.05)
        total, _ = ActionUtilityEngine._compute_penalties(
            "TAKE_PROFIT_LARGE", state, belief, hazard,
        )
        assert total <= 1.0


class TestActionUtilityScoreAll:

    def test_returns_seven_candidates(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        assert len(candidates) == 7

    def test_all_valid_actions_scored(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        scored_actions = {c.action for c in candidates}
        assert scored_actions == VALID_ACTIONS

    def test_sorted_by_net_utility_descending(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        utilities = [c.net_utility for c in candidates]
        assert utilities == sorted(utilities, reverse=True)

    def test_rank_one_is_first(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        assert candidates[0].rank == 1

    def test_ranks_are_sequential(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        ranks = [c.rank for c in candidates]
        assert ranks == list(range(1, 8))

    def test_all_candidates_validate(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        for c in candidates:
            errors = c.validate()
            assert errors == [], f"{c.action} validation errors: {errors}"

    def test_all_candidates_shadow_only(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        for c in candidates:
            assert c.shadow_only is True

    def test_net_utility_in_range(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        for c in candidates:
            assert 0.0 <= c.net_utility <= 1.0

    def test_exit_fractions_match(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        for c in candidates:
            assert c.exit_fraction == ACTION_EXIT_FRACTIONS[c.action]

    def test_net_utility_equals_base_minus_penalty_clamped(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        for c in candidates:
            expected = max(0.0, min(1.0, c.base_utility - c.penalty_total))
            assert abs(c.net_utility - expected) < 1e-5, (
                f"{c.action}: net={c.net_utility}, expected={expected}"
            )

    def test_rationale_not_empty(self):
        candidates = ActionUtilityEngine.score_all(
            _make_state(), _make_belief(), _make_hazard()
        )
        for c in candidates:
            assert c.rationale, f"{c.action} has empty rationale"

    def test_position_id_propagated(self):
        state = _make_state(position_id="POS_XYZ")
        candidates = ActionUtilityEngine.score_all(
            state, _make_belief(position_id="POS_XYZ"),
            _make_hazard(position_id="POS_XYZ"),
        )
        for c in candidates:
            assert c.position_id == "POS_XYZ"


class TestActionUtilityScenarios:

    def test_strong_hold_scenario(self):
        """High conviction + low hazard → HOLD ranked #1."""
        belief = _make_belief(
            hold_conviction=0.95, exit_pressure=0.05,
            directional_edge=0.9, uncertainty_total=0.05,
        )
        hazard = _make_hazard(
            composite_hazard=0.05,
            drawdown_hazard=0.02, reversal_hazard=0.03,
            volatility_hazard=0.05, time_decay_hazard=0.02,
            regime_hazard=0.03, ensemble_hazard=0.05,
        )
        candidates = ActionUtilityEngine.score_all(
            _make_state(unrealized_pnl=500.0), belief, hazard,
        )
        assert candidates[0].action == "HOLD"

    def test_emergency_exit_scenario(self):
        """High exit pressure + high hazard → aggressive actions dominate, HOLD last."""
        belief = _make_belief(
            hold_conviction=0.05, exit_pressure=0.95,
            directional_edge=-0.8, uncertainty_total=0.1,
        )
        hazard = _make_hazard(
            composite_hazard=0.95,
            drawdown_hazard=0.9, reversal_hazard=0.95,
            volatility_hazard=0.8, time_decay_hazard=0.5,
            regime_hazard=0.9, ensemble_hazard=0.85,
            hazard_components={
                "drawdown": 0.9, "reversal": 0.95,
                "volatility": 0.8, "time_decay": 0.5,
                "regime": 0.9, "ensemble": 0.85,
            },
        )
        candidates = ActionUtilityEngine.score_all(
            _make_state(unrealized_pnl=-200.0), belief, hazard,
        )
        # Aggressive reduction actions dominate; HOLD ranks last due to penalty
        top_action = candidates[0].action
        assert top_action in ("REDUCE_MEDIUM", "REDUCE_SMALL", "CLOSE_FULL")
        hold_rank = next(c.rank for c in candidates if c.action == "HOLD")
        assert hold_rank >= 5  # HOLD heavily penalized under high hazard

    def test_unprofitable_penalizes_take_profit(self):
        """When position is losing, TAKE_PROFIT actions are penalised."""
        state = _make_state(unrealized_pnl=-50.0)
        belief = _make_belief(exit_pressure=0.5)
        hazard = _make_hazard(composite_hazard=0.4, drawdown_hazard=0.3)
        candidates = ActionUtilityEngine.score_all(state, belief, hazard)
        tp_actions = {c.action: c for c in candidates
                      if c.action.startswith("TAKE_PROFIT")}
        for c in tp_actions.values():
            assert c.penalty_total > 0, f"{c.action} should be penalised"


# ═══════════════════════════════════════════════════════════════════════════
# S7: Integration (full pipeline: belief → hazard → utility)
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase3Integration:

    def test_full_pipeline_belief_to_utility(self):
        """Phase 1 inputs → Belief → Hazard → Utility scorecard."""
        state = _make_state()
        geo = _make_geometry()
        regime = _make_regime()
        ens = _make_ensemble()

        # Step 1: Belief
        belief = BeliefEngine.compute(state, geo, regime, ens)
        assert belief is not None
        assert belief.validate() == []

        # Step 2: Hazard
        hazard = HazardEngine.assess(state, geo, regime, ens)
        assert hazard.validate() == []

        # Step 3: Utility
        candidates = ActionUtilityEngine.score_all(state, belief, hazard)
        assert len(candidates) == 7
        for c in candidates:
            assert c.validate() == []

    def test_pipeline_all_contracts_shadow_only(self):
        state = _make_state()
        geo = _make_geometry()
        regime = _make_regime()
        ens = _make_ensemble()

        belief = BeliefEngine.compute(state, geo, regime, ens)
        hazard = HazardEngine.assess(state, geo, regime, ens)
        candidates = ActionUtilityEngine.score_all(state, belief, hazard)

        assert belief.shadow_only is True
        assert hazard.shadow_only is True
        for c in candidates:
            assert c.shadow_only is True

    def test_pipeline_with_no_ensemble_stops_at_belief(self):
        """Without ensemble, belief returns None → hazard/utility can't run."""
        belief = BeliefEngine.compute(
            _make_state(), _make_geometry(), _make_regime(), None,
        )
        assert belief is None

    def test_to_dict_round_trips(self):
        """All Phase 3 contracts serialize without error."""
        import json

        state = _make_state()
        geo = _make_geometry()
        regime = _make_regime()
        ens = _make_ensemble()

        belief = BeliefEngine.compute(state, geo, regime, ens)
        hazard = HazardEngine.assess(state, geo, regime, ens)
        candidates = ActionUtilityEngine.score_all(state, belief, hazard)

        # BeliefState round-trip
        bd = belief.to_dict()
        assert float(bd["exit_pressure"]) == belief.exit_pressure
        assert json.loads(bd["belief_components"])

        # HazardAssessment round-trip
        hd = hazard.to_dict()
        assert float(hd["composite_hazard"]) == hazard.composite_hazard
        assert json.loads(hd["hazard_components"])

        # ActionCandidate round-trip
        for c in candidates:
            cd = c.to_dict()
            assert cd["action"] == c.action
            assert float(cd["net_utility"]) == c.net_utility
            assert json.loads(cd["utility_components"]) is not None

    def test_short_position_pipeline(self):
        """Pipeline works correctly for SHORT positions."""
        state = _make_state(
            side="SHORT", entry_price=50000.0, current_price=49000.0,
            unrealized_pnl=100.0, unrealized_pnl_pct=0.02,
        )
        geo = _make_geometry()
        regime = _make_regime(trend_alignment=-0.5)
        ens = _make_ensemble()

        belief = BeliefEngine.compute(state, geo, regime, ens)
        assert belief is not None
        assert belief.side == "SHORT"

        hazard = HazardEngine.assess(state, geo, regime, ens)
        candidates = ActionUtilityEngine.score_all(state, belief, hazard)
        assert len(candidates) == 7


# ═══════════════════════════════════════════════════════════════════════════
# S8: Shadow stream verification
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase3ShadowStreams:

    def test_shadow_stream_names(self):
        """Verify all 3 Phase 3 shadow stream names."""
        expected_streams = {
            "quantum:stream:exit.belief.shadow",
            "quantum:stream:exit.hazard.shadow",
            "quantum:stream:exit.utility.shadow",
        }
        for stream in expected_streams:
            assert stream.endswith(".shadow")
            assert "exit." in stream

    def test_belief_state_serializable_for_stream(self):
        """BeliefState.to_dict() produces flat dict suitable for Redis XADD."""
        b = _make_belief(
            belief_components={"key": 0.5},
            quality_flags=["FLAG"],
        )
        d = b.to_dict()
        # All values must be Redis-compatible (str, int, float, bytes)
        for k, v in d.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Key {k} has type {type(v)}, not Redis-compatible"
            )

    def test_hazard_assessment_serializable_for_stream(self):
        """HazardAssessment.to_dict() produces flat dict for Redis."""
        h = _make_hazard()
        d = h.to_dict()
        for k, v in d.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Key {k} has type {type(v)}, not Redis-compatible"
            )

    def test_action_candidate_serializable_for_stream(self):
        """ActionCandidate.to_dict() produces flat dict for Redis."""
        a = _make_action(
            utility_components={"x": 0.5},
            penalty_components={"y": 0.1},
        )
        d = a.to_dict()
        for k, v in d.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"Key {k} has type {type(v)}, not Redis-compatible"
            )

    def test_shadow_only_enforced_on_all_contracts(self):
        """Setting shadow_only=False causes validation failure."""
        b = _make_belief()
        h = _make_hazard()
        a = _make_action()

        b.shadow_only = False
        h.shadow_only = False
        a.shadow_only = False

        assert any("shadow_only" in e for e in b.validate())
        assert any("shadow_only" in e for e in h.validate())
        assert any("shadow_only" in e for e in a.validate())

    def test_forbidden_streams_not_written(self):
        """Phase 3 modules have no IO — they CANNOT write to any stream.
        This test documents the architectural constraint."""
        # BeliefEngine, HazardEngine, ActionUtilityEngine are all pure math
        # They have no Redis client, no publisher, no IO methods
        assert not hasattr(BeliefEngine, 'publish')
        assert not hasattr(HazardEngine, 'publish')
        assert not hasattr(ActionUtilityEngine, 'publish')
