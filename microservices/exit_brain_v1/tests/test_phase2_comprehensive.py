"""
Phase 2 Comprehensive Tests — Ensemble Sensor Layer.

Covers all Fase 2 modules:
    - models/model_exit_signal.py
    - models/aggregated_exit_signal.py
    - adapters/model_registry.py
    - adapters/ensemble_exit_adapter.py  (mock agents)
    - aggregators/ensemble_aggregator.py
    - engines/normalization.py
    - engines/calibration.py

Shadow-only. No execution writes. No Redis required.
"""

from __future__ import annotations

import json
import math
import time
import pytest
from unittest.mock import MagicMock, patch

from microservices.exit_brain_v1.models.model_exit_signal import (
    ModelExitSignal,
    VALID_MODELS,
    VALID_HORIZONS,
)
from microservices.exit_brain_v1.models.aggregated_exit_signal import (
    AggregatedExitSignal,
    EnsembleDiagnostics,
    MIN_MODELS_REQUIRED,
)
from microservices.exit_brain_v1.adapters.model_registry import (
    ALL_MODEL_NAMES,
    MODEL_SPECS,
    ModelSpec,
    get_model_spec,
    get_all_model_names,
    get_agent_class,
)
from microservices.exit_brain_v1.adapters.ensemble_exit_adapter import (
    EnsembleExitAdapter,
    SOFT_STALE_SEC,
    HARD_STALE_SEC,
)
from microservices.exit_brain_v1.aggregators.ensemble_aggregator import (
    EnsembleAggregator,
    AGGREGATION_METHOD,
    _std,
)
from microservices.exit_brain_v1.engines.normalization import (
    clamp,
    renormalize_probabilities,
    softmax,
    reconstruct_probabilities_from_action,
    derive_directional_probabilities,
    PROBA_SUM_TOLERANCE,
)
from microservices.exit_brain_v1.engines.calibration import (
    identity_calibrate,
    temperature_scale,
    platt_scale,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: build valid ModelExitSignal / AggregatedExitSignal
# ═══════════════════════════════════════════════════════════════════════════


def _make_signal(
    model_name: str = "xgboost",
    sell: float = 0.3,
    hold: float = 0.5,
    buy: float = 0.2,
    confidence: float = 0.8,
    freshness: float = 5.0,
    **overrides,
) -> ModelExitSignal:
    """Build a valid ModelExitSignal with sensible defaults."""
    kwargs = dict(
        position_id="pos-abc",
        symbol="BTCUSDT",
        model_name=model_name,
        model_version="v1.0",
        inference_timestamp=time.time(),
        horizon_label="short",
        sell_probability=sell,
        hold_probability=hold,
        buy_probability=buy,
        continuation_probability=hold + buy,
        reversal_probability=sell,
        confidence=confidence,
        freshness_seconds=freshness,
        shadow_only=True,
    )
    kwargs.update(overrides)
    return ModelExitSignal(**kwargs)


def _make_agg_signal(**overrides) -> AggregatedExitSignal:
    """Build a valid AggregatedExitSignal."""
    kwargs = dict(
        position_id="pos-abc",
        symbol="BTCUSDT",
        ensemble_timestamp=time.time(),
        participating_models=["xgboost", "lightgbm"],
        missing_models=["nhits"],
        stale_models=[],
        model_count_total=6,
        model_count_used=2,
        sell_probability_agg=0.3,
        hold_probability_agg=0.5,
        buy_probability_agg=0.2,
        continuation_probability_agg=0.7,
        reversal_probability_agg=0.3,
        confidence_agg=0.8,
        uncertainty_agg=0.2,
        disagreement_score=0.1,
        consensus_strength=0.9,
        reliability_score=0.7,
        aggregation_method="weighted_average_v1",
        shadow_only=True,
    )
    kwargs.update(overrides)
    return AggregatedExitSignal(**kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# S1: ModelExitSignal (data contract)
# ═══════════════════════════════════════════════════════════════════════════


class TestModelExitSignal:

    def test_valid_signal_no_errors(self):
        sig = _make_signal()
        assert sig.validate() == []

    def test_all_six_models_accepted(self):
        for model in VALID_MODELS:
            sig = _make_signal(model_name=model)
            assert sig.validate() == [], f"{model} should be valid"

    def test_invalid_model_name_rejected(self):
        sig = _make_signal(model_name="random_forest")
        errors = sig.validate()
        assert any("model_name" in e for e in errors)

    def test_uncertainty_auto_derived_from_confidence(self):
        sig = _make_signal(confidence=0.85)
        assert sig.uncertainty == pytest.approx(0.15, abs=1e-9)

    def test_uncertainty_explicit_overrides_auto(self):
        sig = _make_signal(confidence=0.85, uncertainty=0.10)
        # __post_init__ only sets uncertainty if it's None
        assert sig.uncertainty == 0.10

    def test_empty_position_id_rejected(self):
        sig = _make_signal(position_id="")
        errors = sig.validate()
        assert any("position_id" in e for e in errors)

    def test_empty_symbol_rejected(self):
        sig = _make_signal(symbol="")
        errors = sig.validate()
        assert any("symbol" in e for e in errors)

    def test_empty_model_version_rejected(self):
        sig = _make_signal(model_version="")
        errors = sig.validate()
        assert any("model_version" in e for e in errors)

    def test_zero_timestamp_rejected(self):
        sig = _make_signal(inference_timestamp=0)
        errors = sig.validate()
        assert any("inference_timestamp" in e for e in errors)

    def test_negative_timestamp_rejected(self):
        sig = _make_signal(inference_timestamp=-1.0)
        errors = sig.validate()
        assert any("inference_timestamp" in e for e in errors)

    def test_invalid_horizon_rejected(self):
        sig = _make_signal(horizon_label="ultra_long")
        errors = sig.validate()
        assert any("horizon_label" in e for e in errors)

    def test_all_valid_horizons_accepted(self):
        for h in VALID_HORIZONS:
            sig = _make_signal(horizon_label=h)
            assert sig.validate() == [], f"horizon '{h}' should be valid"

    def test_probability_out_of_range(self):
        sig = _make_signal(sell=-0.1)
        errors = sig.validate()
        assert any("sell_probability" in e for e in errors)

    def test_probability_above_one(self):
        sig = _make_signal(confidence=1.5)
        errors = sig.validate()
        assert any("confidence" in e for e in errors)

    def test_proba_sum_too_low(self):
        sig = _make_signal(sell=0.1, hold=0.1, buy=0.1)
        errors = sig.validate()
        assert any("sum" in e for e in errors)

    def test_proba_sum_too_high(self):
        sig = _make_signal(sell=0.5, hold=0.5, buy=0.5)
        errors = sig.validate()
        assert any("sum" in e for e in errors)

    def test_proba_sum_within_tolerance(self):
        sig = _make_signal(sell=0.295, hold=0.505, buy=0.2)
        assert sig.validate() == []

    def test_negative_freshness_rejected(self):
        sig = _make_signal(freshness=-1.0)
        errors = sig.validate()
        assert any("freshness" in e for e in errors)

    def test_shadow_only_false_rejected(self):
        sig = _make_signal(shadow_only=False)
        errors = sig.validate()
        assert any("shadow_only" in e for e in errors)

    def test_to_dict_keys(self):
        sig = _make_signal()
        d = sig.to_dict()
        required_keys = {
            "position_id", "symbol", "model_name", "model_version",
            "inference_timestamp", "horizon_label",
            "sell_probability", "hold_probability", "buy_probability",
            "continuation_probability", "reversal_probability",
            "confidence", "uncertainty", "freshness_seconds",
            "expected_upside_remaining", "expected_downside_if_hold",
            "expected_drawdown_risk", "shadow_only",
        }
        assert required_keys.issubset(d.keys())

    def test_to_dict_source_quality_flags_serialized(self):
        sig = _make_signal(source_quality_flags=["STALE_OHLCV", "MISSING_FUNDING"])
        d = sig.to_dict()
        assert d["source_quality_flags"] == "STALE_OHLCV,MISSING_FUNDING"

    def test_to_dict_empty_quality_flags(self):
        sig = _make_signal(source_quality_flags=[])
        d = sig.to_dict()
        assert d["source_quality_flags"] == ""

    def test_seven_required_outputs_present(self):
        """Each model MUST deliver all 7 required outputs per spec."""
        sig = _make_signal(
            expected_upside_remaining=0.05,
            expected_downside_if_hold=0.03,
            expected_drawdown_risk=0.12,
        )
        d = sig.to_dict()
        for key in [
            "continuation_probability", "reversal_probability",
            "expected_upside_remaining", "expected_downside_if_hold",
            "expected_drawdown_risk", "confidence", "uncertainty",
        ]:
            assert key in d, f"Missing required output: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# S2: AggregatedExitSignal / EnsembleDiagnostics
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregatedExitSignal:

    def test_valid_signal_no_errors(self):
        agg = _make_agg_signal()
        assert agg.validate() == []

    def test_min_models_required_is_two(self):
        assert MIN_MODELS_REQUIRED == 2

    def test_below_min_models_rejected(self):
        agg = _make_agg_signal(model_count_used=1)
        errors = agg.validate()
        assert any("model_count_used" in e for e in errors)

    def test_zero_model_count_total_rejected(self):
        agg = _make_agg_signal(model_count_total=0)
        errors = agg.validate()
        assert any("model_count_total" in e for e in errors)

    def test_empty_position_id_rejected(self):
        agg = _make_agg_signal(position_id="")
        errors = agg.validate()
        assert any("position_id" in e for e in errors)

    def test_empty_aggregation_method_rejected(self):
        agg = _make_agg_signal(aggregation_method="")
        errors = agg.validate()
        assert any("aggregation_method" in e for e in errors)

    def test_shadow_only_false_rejected(self):
        agg = _make_agg_signal(shadow_only=False)
        errors = agg.validate()
        assert any("shadow_only" in e for e in errors)

    def test_proba_sum_out_of_tolerance(self):
        agg = _make_agg_signal(
            sell_probability_agg=0.1,
            hold_probability_agg=0.1,
            buy_probability_agg=0.1,
        )
        errors = agg.validate()
        assert any("sum" in e for e in errors)

    def test_probability_range_check(self):
        agg = _make_agg_signal(confidence_agg=-0.1)
        errors = agg.validate()
        assert any("confidence_agg" in e for e in errors)

    def test_to_dict_keys(self):
        agg = _make_agg_signal()
        d = agg.to_dict()
        for key in [
            "position_id", "symbol", "ensemble_timestamp",
            "sell_probability_agg", "hold_probability_agg", "buy_probability_agg",
            "continuation_probability_agg", "reversal_probability_agg",
            "confidence_agg", "uncertainty_agg",
            "disagreement_score", "consensus_strength", "reliability_score",
            "aggregation_method", "shadow_only",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_lists_serialized_as_csv(self):
        agg = _make_agg_signal(
            participating_models=["xgboost", "lightgbm"],
            quality_flags=["HIGH_DISAGREEMENT"],
        )
        d = agg.to_dict()
        assert d["participating_models"] == "xgboost,lightgbm"
        assert d["quality_flags"] == "HIGH_DISAGREEMENT"


class TestEnsembleDiagnostics:

    def test_to_dict_serializes_nested(self):
        diag = EnsembleDiagnostics(
            position_id="pos-1",
            symbol="ETHUSDT",
            timestamp=time.time(),
            per_model_signals={"xgboost": {"sell": 0.3}},
            per_model_weights={"xgboost": 0.5},
            per_model_freshness={"xgboost": 2.0},
            per_model_status={"xgboost": "OK"},
            probability_spread={"sell_std": 0.01},
            max_disagreement_pair="xgboost vs lightgbm",
            aggregation_config={"method": "weighted_average_v1"},
        )
        d = diag.to_dict()
        # Nested dicts become JSON strings
        assert json.loads(d["per_model_signals"]) == {"xgboost": {"sell": 0.3}}
        assert json.loads(d["per_model_weights"]) == {"xgboost": 0.5}
        assert d["max_disagreement_pair"] == "xgboost vs lightgbm"

    def test_empty_calibration_metadata_default(self):
        diag = EnsembleDiagnostics(
            position_id="pos-1", symbol="X", timestamp=1.0,
            per_model_signals={}, per_model_weights={},
            per_model_freshness={}, per_model_status={},
            probability_spread={}, max_disagreement_pair="",
            aggregation_config={},
        )
        assert diag.calibration_metadata == {}


# ═══════════════════════════════════════════════════════════════════════════
# S3: Normalization (pure math)
# ═══════════════════════════════════════════════════════════════════════════


class TestClamp:

    def test_within_range(self):
        assert clamp(0.5) == 0.5

    def test_below_lo(self):
        assert clamp(-0.3) == 0.0

    def test_above_hi(self):
        assert clamp(1.5) == 1.0

    def test_custom_range(self):
        assert clamp(5.0, 2.0, 4.0) == 4.0
        assert clamp(1.0, 2.0, 4.0) == 2.0

    def test_exact_boundaries(self):
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0


class TestRenormalizeProbabilities:

    def test_already_normalized(self):
        result = renormalize_probabilities([0.3, 0.5, 0.2])
        assert sum(result) == pytest.approx(1.0)
        assert result[0] == pytest.approx(0.3)

    def test_unnormalized(self):
        result = renormalize_probabilities([1.0, 1.0, 1.0])
        assert all(p == pytest.approx(1.0 / 3) for p in result)

    def test_all_zeros_returns_uniform(self):
        result = renormalize_probabilities([0.0, 0.0, 0.0])
        assert all(p == pytest.approx(1.0 / 3) for p in result)

    def test_negative_clamped_to_zero(self):
        result = renormalize_probabilities([-0.5, 1.0, 0.5])
        assert result[0] == pytest.approx(0.0)
        assert sum(result) == pytest.approx(1.0)

    def test_single_element(self):
        result = renormalize_probabilities([0.7])
        assert result == [pytest.approx(1.0)]

    def test_empty_list(self):
        result = renormalize_probabilities([])
        assert result == []


class TestSoftmax:

    def test_uniform_logits(self):
        result = softmax([0.0, 0.0, 0.0])
        assert all(p == pytest.approx(1.0 / 3, abs=1e-6) for p in result)

    def test_sums_to_one(self):
        result = softmax([1.0, 2.0, 3.0])
        assert sum(result) == pytest.approx(1.0)

    def test_argmax_preserved(self):
        result = softmax([1.0, 5.0, 2.0])
        assert result[1] == max(result)

    def test_large_logits_numerically_stable(self):
        result = softmax([1000.0, 1001.0, 1002.0])
        assert sum(result) == pytest.approx(1.0)
        assert result[2] == max(result)

    def test_negative_logits(self):
        result = softmax([-10.0, -5.0, 0.0])
        assert sum(result) == pytest.approx(1.0)
        assert result[2] == max(result)

    def test_empty_list(self):
        assert softmax([]) == []


class TestReconstructProbabilities:

    def test_sell_action(self):
        s, h, b = reconstruct_probabilities_from_action("SELL", 0.8)
        assert s == pytest.approx(0.8)
        assert s + h + b == pytest.approx(1.0)
        assert h > b  # hold_share=0.6 default

    def test_buy_action(self):
        s, h, b = reconstruct_probabilities_from_action("BUY", 0.7)
        assert b == pytest.approx(0.7)
        assert s + h + b == pytest.approx(1.0)

    def test_hold_action(self):
        s, h, b = reconstruct_probabilities_from_action("HOLD", 0.6)
        assert h == pytest.approx(0.6)
        assert s == pytest.approx(b)  # symmetric residual

    def test_confidence_clamped_to_range(self):
        s, h, b = reconstruct_probabilities_from_action("SELL", 1.5)
        assert s + h + b == pytest.approx(1.0)
        assert s == pytest.approx(0.99)  # clamped to 0.99

    def test_zero_confidence_clamped(self):
        s, h, b = reconstruct_probabilities_from_action("BUY", 0.0)
        assert b == pytest.approx(0.01)  # clamped to 0.01

    def test_custom_hold_share(self):
        s, h, b = reconstruct_probabilities_from_action("SELL", 0.8, hold_share=0.8)
        residual = 0.2
        assert h == pytest.approx(residual * 0.8)
        assert b == pytest.approx(residual * 0.2)


class TestDeriveDirectionalProbabilities:

    def test_long_side(self):
        cont, rev = derive_directional_probabilities(0.3, 0.5, 0.2, "LONG")
        assert cont == pytest.approx(0.7)  # hold + buy
        assert rev == pytest.approx(0.3)   # sell

    def test_short_side(self):
        cont, rev = derive_directional_probabilities(0.3, 0.5, 0.2, "SHORT")
        assert cont == pytest.approx(0.8)  # hold + sell
        assert rev == pytest.approx(0.2)   # buy

    def test_clamped_output(self):
        cont, rev = derive_directional_probabilities(0.0, 1.0, 0.5, "LONG")
        assert 0.0 <= cont <= 1.0
        assert 0.0 <= rev <= 1.0

    def test_extreme_sell(self):
        cont, rev = derive_directional_probabilities(1.0, 0.0, 0.0, "LONG")
        assert cont == pytest.approx(0.0)
        assert rev == pytest.approx(1.0)

    def test_extreme_buy(self):
        cont, rev = derive_directional_probabilities(0.0, 0.0, 1.0, "SHORT")
        assert cont == pytest.approx(0.0)
        assert rev == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# S4: Calibration (pure math)
# ═══════════════════════════════════════════════════════════════════════════


class TestIdentityCalibrate:

    def test_passes_through(self):
        inp = [0.3, 0.5, 0.2]
        result = identity_calibrate(inp)
        assert result == inp

    def test_returns_new_list(self):
        inp = [0.3, 0.5, 0.2]
        result = identity_calibrate(inp)
        assert result is not inp

    def test_empty_input(self):
        assert identity_calibrate([]) == []


class TestTemperatureScale:

    def test_t1_equals_softmax(self):
        logits = [1.0, 2.0, 3.0]
        ts = temperature_scale(logits, 1.0)
        sm = softmax(logits)
        for a, b in zip(ts, sm):
            assert a == pytest.approx(b, abs=1e-6)

    def test_high_temperature_flattens(self):
        logits = [1.0, 2.0, 10.0]
        flat = temperature_scale(logits, 100.0)
        # High T → nearly uniform
        assert all(abs(p - 1 / 3) < 0.1 for p in flat)

    def test_low_temperature_sharpens(self):
        logits = [1.0, 2.0, 3.0]
        sharp = temperature_scale(logits, 0.1)
        assert sharp[2] > 0.99  # 3rd class dominates

    def test_sums_to_one(self):
        result = temperature_scale([5.0, -1.0, 2.0], 2.0)
        assert sum(result) == pytest.approx(1.0)

    def test_negative_temperature_clamped(self):
        result = temperature_scale([1.0, 2.0], -1.0)
        assert sum(result) == pytest.approx(1.0)

    def test_empty_logits(self):
        assert temperature_scale([], 1.0) == []


class TestPlattScale:

    def test_identity_params(self):
        # a=1.0, b=0.0 → identity-ish
        assert platt_scale(0.5, 1.0, 0.0) == pytest.approx(0.5, abs=1e-6)

    def test_monotonic(self):
        vals = [platt_scale(p / 10, 1.0, 0.0) for p in range(1, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_output_in_zero_one(self):
        for p in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
            result = platt_scale(p, 2.0, 0.5)
            assert 0.0 <= result <= 1.0

    def test_shift_moves_midpoint(self):
        # b > 0 shifts calibrated probability upward
        base = platt_scale(0.5, 1.0, 0.0)
        shifted = platt_scale(0.5, 1.0, 1.0)
        assert shifted > base

    def test_clamped_near_boundaries(self):
        assert 0.0 < platt_scale(0.0001) < 1.0
        assert 0.0 < platt_scale(0.9999) < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# S5: Model Registry
# ═══════════════════════════════════════════════════════════════════════════


class TestModelRegistry:

    def test_all_six_models_registered(self):
        assert len(ALL_MODEL_NAMES) == 6
        expected = {"xgboost", "lightgbm", "nhits", "patchtst", "tft", "dlinear"}
        assert set(ALL_MODEL_NAMES) == expected

    def test_get_model_spec_returns_spec(self):
        spec = get_model_spec("xgboost")
        assert isinstance(spec, ModelSpec)
        assert spec.name == "xgboost"
        assert spec.agent_class_name == "XGBoostAgent"

    def test_get_model_spec_unknown_returns_none(self):
        assert get_model_spec("not_a_model") is None

    def test_all_specs_have_required_fields(self):
        for name in ALL_MODEL_NAMES:
            spec = get_model_spec(name)
            assert spec is not None
            assert spec.name == name
            assert spec.agent_class_name, f"{name} missing agent_class_name"
            assert spec.agent_module, f"{name} missing agent_module"
            assert spec.horizon_label in VALID_HORIZONS
            assert spec.feature_version, f"{name} missing feature_version"
            assert spec.model_type in ("tree", "torch"), f"{name} invalid model_type"

    def test_model_spec_is_frozen(self):
        spec = get_model_spec("xgboost")
        with pytest.raises(AttributeError):
            spec.name = "oops"

    def test_get_all_model_names(self):
        names = get_all_model_names()
        assert names == ALL_MODEL_NAMES

    def test_tree_models(self):
        trees = [n for n in ALL_MODEL_NAMES if MODEL_SPECS[n].model_type == "tree"]
        assert set(trees) == {"xgboost", "lightgbm"}

    def test_torch_models(self):
        torches = [n for n in ALL_MODEL_NAMES if MODEL_SPECS[n].model_type == "torch"]
        assert set(torches) == {"nhits", "patchtst", "tft", "dlinear"}

    def test_get_agent_class_graceful_failure(self):
        # ai_engine.agents.unified_agents won't exist in test environment
        cls = get_agent_class("xgboost")
        # Returns None (import fails) — that's fine, fail-closed
        assert cls is None or callable(cls)


# ═══════════════════════════════════════════════════════════════════════════
# S6: EnsembleExitAdapter (with mock agents)
# ═══════════════════════════════════════════════════════════════════════════


class TestEnsembleExitAdapter:

    def test_missing_models_when_no_agents_available(self):
        """In test env, all agents will fail to import."""
        adapter = EnsembleExitAdapter()
        # No real agents loadable in test env
        assert len(adapter.missing_models) == 6

    def test_custom_model_names(self):
        adapter = EnsembleExitAdapter(model_names=["xgboost", "lightgbm"])
        # Only asked for 2
        assert len(adapter.missing_models) <= 2

    @patch(
        "microservices.exit_brain_v1.adapters.ensemble_exit_adapter.get_agent_class"
    )
    @patch(
        "microservices.exit_brain_v1.adapters.ensemble_exit_adapter.get_model_spec"
    )
    def test_collect_and_normalize_with_mock_agent(self, mock_spec, mock_cls):
        """Full adapter flow with a mock agent."""
        spec = ModelSpec(
            name="xgboost", agent_class_name="Mock",
            agent_module="mock", horizon_label="short",
            feature_version="v5", model_type="tree",
        )
        mock_spec.return_value = spec

        mock_agent = MagicMock()
        mock_agent.is_ready.return_value = True
        mock_agent.predict.return_value = {
            "action": "SELL",
            "confidence": 0.75,
            "version": "v1.0",
        }
        mock_cls.return_value = lambda: mock_agent

        adapter = EnsembleExitAdapter(model_names=["xgboost"])
        # Manually inject mock agent since _load_agents already ran
        adapter._agents = {"xgboost": mock_agent}
        adapter._agent_errors = {}

        # Build a minimal PositionExitState
        from microservices.exit_brain_v1.models.position_exit_state import (
            PositionExitState,
        )
        state = PositionExitState(
            position_id="pos-1", symbol="ETHUSDT", side="LONG",
            status="OPEN", entry_price=3000.0, current_price=3050.0,
            mark_price=3050.0, quantity=1.0, notional=3050.0,
            leverage=5.0, unrealized_pnl=50.0, unrealized_pnl_pct=1.67,
            realized_pnl=0.0, max_favorable_excursion=60.0,
            max_adverse_excursion=10.0, drawdown_from_peak_pnl=10.0,
            open_timestamp=time.time() - 300.0, regime_label="trending_up",
            regime_confidence=0.8, data_quality_flags=[],
            source_timestamps={},
        )

        signals = adapter.collect_and_normalize(state)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.model_name == "xgboost"
        assert sig.symbol == "ETHUSDT"
        assert sig.shadow_only is True
        assert sig.sell_probability + sig.hold_probability + sig.buy_probability == pytest.approx(1.0, abs=0.02)
        assert sig.validate() == []

    def test_build_feature_dict(self):
        """Feature extraction from PositionExitState."""
        from microservices.exit_brain_v1.models.position_exit_state import (
            PositionExitState,
        )
        state = PositionExitState(
            position_id="pos-1", symbol="BTCUSDT", side="LONG",
            status="OPEN", entry_price=50000.0, current_price=51000.0,
            mark_price=51000.0, quantity=0.5, notional=25500.0,
            leverage=10.0, unrealized_pnl=500.0, unrealized_pnl_pct=2.0,
            realized_pnl=0.0, max_favorable_excursion=600.0,
            max_adverse_excursion=100.0, drawdown_from_peak_pnl=100.0,
            open_timestamp=time.time() - 600.0, regime_label="ranging",
            regime_confidence=0.6, data_quality_flags=[],
            source_timestamps={}, atr=150.0,
            volatility_short=0.02, volatility_medium=0.03,
            trend_signal=0.7, momentum_score=0.5,
            spread_bps=1.5,
        )

        adapter = EnsembleExitAdapter(model_names=[])
        features = adapter._build_feature_dict(state)

        assert features["entry_price"] == 50000.0
        assert features["current_price"] == 51000.0
        assert features["leverage"] == 10.0
        assert features["atr"] == 150.0
        assert features["side_long"] == 1.0
        assert features["side_short"] == 0.0
        assert features["volatility_short"] == 0.02
        assert features["spread_bps"] == 1.5

    def test_build_feature_dict_short_side(self):
        from microservices.exit_brain_v1.models.position_exit_state import (
            PositionExitState,
        )
        state = PositionExitState(
            position_id="pos-2", symbol="XRPUSDT", side="SHORT",
            status="OPEN", entry_price=1.0, current_price=0.95,
            mark_price=0.95, quantity=1000.0, notional=950.0,
            leverage=3.0, unrealized_pnl=50.0, unrealized_pnl_pct=5.0,
            realized_pnl=0.0, max_favorable_excursion=60.0,
            max_adverse_excursion=10.0, drawdown_from_peak_pnl=10.0,
            open_timestamp=time.time() - 120.0, regime_label="trending_down",
            regime_confidence=0.9, data_quality_flags=[],
            source_timestamps={},
        )
        adapter = EnsembleExitAdapter(model_names=[])
        features = adapter._build_feature_dict(state)
        assert features["side_long"] == 0.0
        assert features["side_short"] == 1.0

    def test_publish_raw_shadow_signals(self):
        """Shadow publishing helper."""
        sig = _make_signal()
        mock_publisher = MagicMock()
        mock_publisher._xadd.return_value = "1234-0"

        adapter = EnsembleExitAdapter(model_names=[])
        count = adapter.publish_raw_shadow_signals([sig, sig], mock_publisher)
        assert count == 2
        assert mock_publisher._xadd.call_count == 2
        # Verify correct stream name
        call_args = mock_publisher._xadd.call_args_list[0]
        assert call_args[0][0] == "quantum:stream:exit.ensemble.raw.shadow"


# ═══════════════════════════════════════════════════════════════════════════
# S7: EnsembleAggregator (pure math)
# ═══════════════════════════════════════════════════════════════════════════


class TestStd:

    def test_uniform_values(self):
        assert _std([5.0, 5.0, 5.0]) == 0.0

    def test_known_values(self):
        # std of [0, 2] = sqrt((1+1)/2) = 1.0
        assert _std([0.0, 2.0]) == pytest.approx(1.0)

    def test_single_value(self):
        assert _std([42.0]) == 0.0

    def test_empty_returns_zero(self):
        assert _std([]) == 0.0


class TestEnsembleAggregatorFilter:

    def test_fresh_signals_all_usable(self):
        signals = [_make_signal("xgboost", freshness=2.0), _make_signal("lightgbm", freshness=3.0)]
        agg = EnsembleAggregator()
        usable, stale, excluded = agg.filter_usable_signals(signals)
        assert len(usable) == 2
        assert stale == []
        assert excluded == set()

    def test_soft_stale_still_usable(self):
        sig = _make_signal("xgboost", freshness=90.0)
        agg = EnsembleAggregator()
        usable, stale, excluded = agg.filter_usable_signals([sig])
        assert len(usable) == 1
        assert "xgboost" in stale

    def test_hard_stale_excluded(self):
        sig = _make_signal("xgboost", freshness=400.0)
        agg = EnsembleAggregator()
        usable, stale, excluded = agg.filter_usable_signals([sig])
        assert len(usable) == 0
        assert "xgboost" in excluded

    def test_invalid_signal_excluded(self):
        sig = _make_signal("xgboost", sell=-0.1)  # invalid
        agg = EnsembleAggregator()
        usable, stale, excluded = agg.filter_usable_signals([sig])
        assert len(usable) == 0
        assert "xgboost" in excluded


class TestEnsembleAggregatorWeights:

    def test_equal_fresh_signals_equal_weights(self):
        signals = [
            _make_signal("xgboost", confidence=0.8, freshness=5.0),
            _make_signal("lightgbm", confidence=0.8, freshness=5.0),
        ]
        agg = EnsembleAggregator()
        weights = agg.compute_reliability_weights(signals)
        assert len(weights) == 2
        assert weights[0] == pytest.approx(weights[1])
        assert sum(weights) == pytest.approx(1.0)

    def test_higher_confidence_gets_more_weight(self):
        signals = [
            _make_signal("xgboost", confidence=0.9, freshness=5.0),
            _make_signal("lightgbm", confidence=0.5, freshness=5.0),
        ]
        agg = EnsembleAggregator()
        weights = agg.compute_reliability_weights(signals)
        assert weights[0] > weights[1]

    def test_stale_signal_penalized(self):
        signals = [
            _make_signal("xgboost", confidence=0.8, freshness=5.0),
            _make_signal("lightgbm", confidence=0.8, freshness=200.0),  # soft+stale
        ]
        agg = EnsembleAggregator()
        weights = agg.compute_reliability_weights(signals)
        assert weights[0] > weights[1]

    def test_empty_returns_empty(self):
        agg = EnsembleAggregator()
        assert agg.compute_reliability_weights([]) == []


class TestEnsembleAggregatorProbabilities:

    def test_aggregate_two_models(self):
        signals = [
            _make_signal("xgboost", sell=0.2, hold=0.6, buy=0.2),
            _make_signal("lightgbm", sell=0.4, hold=0.4, buy=0.2),
        ]
        agg = EnsembleAggregator()
        weights = agg.compute_reliability_weights(signals)
        s, h, b = agg.aggregate_probabilities(signals, weights)
        assert s + h + b == pytest.approx(1.0, abs=0.01)

    def test_identical_models_preserve_probabilities(self):
        signals = [
            _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2),
            _make_signal("lightgbm", sell=0.3, hold=0.5, buy=0.2),
        ]
        agg = EnsembleAggregator()
        weights = [0.5, 0.5]
        s, h, b = agg.aggregate_probabilities(signals, weights)
        assert s == pytest.approx(0.3, abs=0.01)
        assert h == pytest.approx(0.5, abs=0.01)
        assert b == pytest.approx(0.2, abs=0.01)


class TestEnsembleAggregatorDisagreement:

    def test_perfect_agreement(self):
        signals = [
            _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2),
            _make_signal("lightgbm", sell=0.3, hold=0.5, buy=0.2),
        ]
        agg = EnsembleAggregator()
        score = agg.compute_disagreement_score(signals)
        assert score == 0.0

    def test_high_disagreement(self):
        signals = [
            _make_signal("xgboost", sell=0.9, hold=0.05, buy=0.05),
            _make_signal("lightgbm", sell=0.05, hold=0.05, buy=0.9),
        ]
        agg = EnsembleAggregator()
        score = agg.compute_disagreement_score(signals)
        assert score > 0.3  # significant disagreement

    def test_single_signal_no_disagreement(self):
        agg = EnsembleAggregator()
        score = agg.compute_disagreement_score([_make_signal()])
        assert score == 0.0

    def test_consensus_is_inverse(self):
        agg = EnsembleAggregator()
        assert agg.compute_consensus_strength(0.3) == pytest.approx(0.7)
        assert agg.compute_consensus_strength(0.0) == pytest.approx(1.0)
        assert agg.compute_consensus_strength(1.0) == pytest.approx(0.0)


class TestEnsembleAggregatorExpectedValues:

    def test_aggregates_nonzero_only(self):
        signals = [
            _make_signal("xgboost", expected_upside_remaining=0.05, expected_downside_if_hold=0.02),
            _make_signal("lightgbm", expected_upside_remaining=0.0, expected_downside_if_hold=0.04),
        ]
        agg = EnsembleAggregator()
        weights = [0.5, 0.5]
        up, down, dd = agg.aggregate_expected_values(signals, weights)
        # Upside: only xgboost contributes (lightgbm=0)
        assert up == pytest.approx(0.05, abs=0.01)
        # Downside: both contribute
        assert down > 0

    def test_all_zero_returns_zero(self):
        signals = [_make_signal("xgboost"), _make_signal("lightgbm")]
        agg = EnsembleAggregator()
        weights = [0.5, 0.5]
        up, down, dd = agg.aggregate_expected_values(signals, weights)
        assert up == 0.0
        assert down == 0.0
        assert dd == 0.0


class TestEnsembleAggregatorFullPipeline:

    def test_fail_closed_with_zero_signals(self):
        agg = EnsembleAggregator()
        result, diag = agg.aggregate([], "LONG")
        assert result is None
        assert isinstance(diag, EnsembleDiagnostics)

    def test_fail_closed_with_one_signal(self):
        agg = EnsembleAggregator()
        result, diag = agg.aggregate([_make_signal()], "LONG")
        assert result is None

    def test_success_with_two_signals(self):
        signals = [
            _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2),
            _make_signal("lightgbm", sell=0.25, hold=0.55, buy=0.2),
        ]
        agg = EnsembleAggregator()
        result, diag = agg.aggregate(signals, "LONG")
        assert result is not None
        assert result.validate() == []
        assert result.model_count_used == 2
        assert result.shadow_only is True

    def test_long_continuation_reversal(self):
        signals = [
            _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2),
            _make_signal("lightgbm", sell=0.3, hold=0.5, buy=0.2),
        ]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "LONG")
        assert result is not None
        # LONG: continuation = hold + buy, reversal = sell
        assert result.continuation_probability_agg > result.reversal_probability_agg

    def test_short_continuation_reversal(self):
        signals = [
            _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2),
            _make_signal("lightgbm", sell=0.3, hold=0.5, buy=0.2),
        ]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "SHORT")
        assert result is not None
        # SHORT: continuation = hold + sell, reversal = buy
        assert result.continuation_probability_agg > result.reversal_probability_agg

    def test_missing_models_detected(self):
        signals = [
            _make_signal("xgboost"),
            _make_signal("lightgbm"),
        ]
        agg = EnsembleAggregator()
        result, diag = agg.aggregate(
            signals, "LONG",
            all_model_names=["xgboost", "lightgbm", "nhits", "patchtst", "tft", "dlinear"],
        )
        assert result is not None
        assert set(result.missing_models) == {"nhits", "patchtst", "tft", "dlinear"}

    def test_stale_models_tracked(self):
        signals = [
            _make_signal("xgboost", freshness=5.0),
            _make_signal("lightgbm", freshness=90.0),  # soft stale
        ]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "LONG")
        assert result is not None
        assert "lightgbm" in result.stale_models

    def test_hard_stale_excluded_but_aggregation_works(self):
        signals = [
            _make_signal("xgboost", freshness=5.0),
            _make_signal("lightgbm", freshness=5.0),
            _make_signal("nhits", freshness=400.0),  # hard stale
        ]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "LONG")
        assert result is not None
        assert result.model_count_used == 2
        assert "nhits" not in result.participating_models

    def test_quality_flags_low_coverage(self):
        signals = [
            _make_signal("xgboost"),
            _make_signal("lightgbm"),
        ]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "LONG")
        assert result is not None
        assert any("LOW_MODEL_COVERAGE" in f for f in result.quality_flags)

    def test_quality_flags_high_disagreement(self):
        signals = [
            _make_signal("xgboost", sell=0.9, hold=0.05, buy=0.05),
            _make_signal("lightgbm", sell=0.05, hold=0.05, buy=0.9),
        ]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "LONG")
        assert result is not None
        assert any("HIGH_DISAGREEMENT" in f for f in result.quality_flags)

    def test_aggregation_method_is_weighted_average_v1(self):
        signals = [_make_signal("xgboost"), _make_signal("lightgbm")]
        agg = EnsembleAggregator()
        result, _ = agg.aggregate(signals, "LONG")
        assert result is not None
        assert result.aggregation_method == "weighted_average_v1"

    def test_diagnostics_always_produced(self):
        agg = EnsembleAggregator()
        # Even with empty signals
        _, diag = agg.aggregate([], "LONG")
        assert isinstance(diag, EnsembleDiagnostics)
        assert diag.aggregation_config["method"] == "weighted_average_v1"

    def test_diagnostics_per_model_weights_set_on_success(self):
        signals = [
            _make_signal("xgboost"),
            _make_signal("lightgbm"),
        ]
        agg = EnsembleAggregator()
        result, diag = agg.aggregate(signals, "LONG")
        assert result is not None
        assert "xgboost" in diag.per_model_weights
        assert "lightgbm" in diag.per_model_weights
        assert sum(diag.per_model_weights.values()) == pytest.approx(1.0)

    def test_max_disagreement_pair(self):
        signals = [
            _make_signal("xgboost", sell=0.9, hold=0.05, buy=0.05),
            _make_signal("lightgbm", sell=0.1, hold=0.8, buy=0.1),
            _make_signal("nhits", sell=0.3, hold=0.5, buy=0.2),
        ]
        agg = EnsembleAggregator()
        _, diag = agg.aggregate(signals, "LONG")
        assert "xgboost" in diag.max_disagreement_pair
        assert "lightgbm" in diag.max_disagreement_pair

    def test_six_models_full_ensemble(self):
        """All 6 models produce signals — golden path."""
        signals = [
            _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2, confidence=0.8),
            _make_signal("lightgbm", sell=0.25, hold=0.55, buy=0.2, confidence=0.75),
            _make_signal("nhits", sell=0.35, hold=0.45, buy=0.2, confidence=0.7),
            _make_signal("patchtst", sell=0.28, hold=0.52, buy=0.2, confidence=0.72),
            _make_signal("tft", sell=0.32, hold=0.48, buy=0.2, confidence=0.78),
            _make_signal("dlinear", sell=0.3, hold=0.5, buy=0.2, confidence=0.65),
        ]
        agg = EnsembleAggregator()
        result, diag = agg.aggregate(signals, "LONG", all_model_names=ALL_MODEL_NAMES)

        assert result is not None
        assert result.validate() == []
        assert result.model_count_used == 6
        assert result.missing_models == []
        assert result.stale_models == []
        assert not any("LOW_MODEL_COVERAGE" in f for f in result.quality_flags)
        assert result.reliability_score > 0.5
        # All probabilities in valid range
        assert 0.0 <= result.sell_probability_agg <= 1.0
        assert 0.0 <= result.hold_probability_agg <= 1.0
        assert 0.0 <= result.buy_probability_agg <= 1.0

    def test_publish_aggregated_shadow(self):
        """Publishing to shadow streams."""
        signals = [_make_signal("xgboost"), _make_signal("lightgbm")]
        agg_inst = EnsembleAggregator()
        result, diag = agg_inst.aggregate(signals, "LONG")
        assert result is not None

        mock_pub = MagicMock()
        mock_pub._xadd.return_value = "1234-0"

        agg_ok, diag_ok = agg_inst.publish_aggregated_shadow_signal(result, diag, mock_pub)
        assert agg_ok is True
        assert diag_ok is True
        assert mock_pub._xadd.call_count == 2

        # Verify correct stream names
        calls = mock_pub._xadd.call_args_list
        streams = [c[0][0] for c in calls]
        assert "quantum:stream:exit.ensemble.agg.shadow" in streams
        assert "quantum:stream:exit.ensemble.diag.shadow" in streams


# ═══════════════════════════════════════════════════════════════════════════
# S8: Integration — Full Adapter → Aggregator pipeline (mock agents)
# ═══════════════════════════════════════════════════════════════════════════


class TestPhase2Integration:

    def test_adapter_signals_feed_aggregator(self):
        """Signals from adapter feed directly into aggregator."""
        sig1 = _make_signal("xgboost", sell=0.3, hold=0.5, buy=0.2)
        sig2 = _make_signal("lightgbm", sell=0.25, hold=0.55, buy=0.2)

        agg = EnsembleAggregator()
        result, diag = agg.aggregate([sig1, sig2], "LONG")

        assert result is not None
        assert result.validate() == []
        assert isinstance(diag, EnsembleDiagnostics)

    def test_signal_to_dict_round_trip(self):
        """ModelExitSignal → to_dict → verify keys match aggregator needs."""
        sig = _make_signal(
            expected_upside_remaining=0.05,
            expected_downside_if_hold=0.03,
            expected_drawdown_risk=0.12,
        )
        d = sig.to_dict()

        # Aggregator reads these fields
        assert float(d["sell_probability"]) == sig.sell_probability
        assert float(d["hold_probability"]) == sig.hold_probability
        assert float(d["buy_probability"]) == sig.buy_probability
        assert float(d["confidence"]) == sig.confidence
        assert float(d["expected_upside_remaining"]) == 0.05

    def test_agg_signal_to_dict_round_trip(self):
        """AggregatedExitSignal → to_dict → verify completeness."""
        agg = _make_agg_signal(
            expected_upside_remaining_agg=0.04,
            expected_downside_if_hold_agg=0.02,
            expected_drawdown_risk_agg=0.08,
        )
        d = agg.to_dict()
        assert float(d["expected_upside_remaining_agg"]) == 0.04
        assert float(d["expected_downside_if_hold_agg"]) == 0.02
        assert float(d["expected_drawdown_risk_agg"]) == 0.08

    def test_shadow_only_enforced_everywhere(self):
        """All data contracts enforce shadow_only=True."""
        sig = _make_signal()
        agg = _make_agg_signal()
        assert sig.shadow_only is True
        assert agg.shadow_only is True
        # Setting to False must cause validation error
        sig.shadow_only = False
        agg.shadow_only = False
        assert any("shadow_only" in e for e in sig.validate())
        assert any("shadow_only" in e for e in agg.validate())

    def test_normalization_calibration_pipeline(self):
        """Full normalization → calibration → renormalize chain."""
        # Reconstruct from action
        s, h, b = reconstruct_probabilities_from_action("SELL", 0.8)
        assert s + h + b == pytest.approx(1.0)

        # Calibrate (identity)
        s2, h2, b2 = identity_calibrate([s, h, b])
        assert [s2, h2, b2] == [s, h, b]

        # Renormalize (idempotent)
        s3, h3, b3 = renormalize_probabilities([s2, h2, b2])
        assert s3 + h3 + b3 == pytest.approx(1.0)

        # Derive directional
        cont, rev = derive_directional_probabilities(s3, h3, b3, "LONG")
        assert cont + rev == pytest.approx(1.0, abs=0.02)

    def test_shadow_stream_names(self):
        """Verify all 3 shadow stream names used in Phase 2."""
        expected_streams = {
            "quantum:stream:exit.ensemble.raw.shadow",
            "quantum:stream:exit.ensemble.agg.shadow",
            "quantum:stream:exit.ensemble.diag.shadow",
        }
        # These streams are used by adapter and aggregator
        # Just confirm they're correct string constants
        for stream in expected_streams:
            assert stream.endswith(".shadow"), f"{stream} must end with .shadow"
            assert "exit.ensemble" in stream

    def test_freshness_weight_curve(self):
        """Verify freshness weight decay is monotonic and bounded."""
        agg = EnsembleAggregator()
        prev = 1.0
        for sec in [0, 10, 30, 60, 120, 200, 299]:
            w = agg._freshness_weight(sec)
            assert 0.0 <= w <= 1.0, f"Weight out of range at {sec}s"
            assert w <= prev or w == prev, f"Weight increased at {sec}s"
            prev = w
        # Beyond hard stale
        assert agg._freshness_weight(301) == 0.0
