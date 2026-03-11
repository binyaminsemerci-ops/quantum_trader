"""
EnsembleExitAdapter — IO boundary between ensemble models and exit brain.

This module:
1. Takes a PositionExitState
2. Calls each ensemble model's predict()
3. Normalizes raw outputs into ModelExitSignal
4. Publishes raw signals to shadow stream

It does NOT:
- Write orders
- Write to trade.intent, apply.plan, exit.intent
- Make exit decisions
- Aggregate signals (that's the aggregator's job)

This is the ONLY module in exit_brain_v1 that imports unified_agents.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..models.position_exit_state import PositionExitState
from ..models.model_exit_signal import ModelExitSignal
from ..engines.normalization import (
    clamp,
    reconstruct_probabilities_from_action,
    renormalize_probabilities,
    derive_directional_probabilities,
)
from ..engines.calibration import identity_calibrate
from .model_registry import (
    ALL_MODEL_NAMES,
    ModelSpec,
    get_agent_class,
    get_model_spec,
)

logger = logging.getLogger(__name__)

# Stale thresholds (seconds since inference)
SOFT_STALE_SEC = 60
HARD_STALE_SEC = 300


class EnsembleExitAdapter:
    """
    Adapter that collects predictions from all ensemble models
    and produces normalised ModelExitSignal objects.

    Lifecycle:
        adapter = EnsembleExitAdapter()
        signals = adapter.collect_and_normalize(position_exit_state)
        # signals is list[ModelExitSignal], possibly empty
    """

    def __init__(self, model_names: Optional[List[str]] = None) -> None:
        """
        Args:
            model_names: Which models to load. Defaults to all 6.
        """
        self._model_names = model_names or list(ALL_MODEL_NAMES)
        self._agents: Dict[str, Any] = {}
        self._agent_errors: Dict[str, str] = {}
        self._load_agents()

    # ── Agent lifecycle ──────────────────────────────────────────────────

    def _load_agents(self) -> None:
        """Instantiate all requested model agents. Fail-open per model."""
        for name in self._model_names:
            spec = get_model_spec(name)
            if spec is None:
                self._agent_errors[name] = f"No spec for model '{name}'"
                logger.warning("[ExitAdapter] No spec for model '%s', skipping", name)
                continue

            cls = get_agent_class(name)
            if cls is None:
                self._agent_errors[name] = f"Import failed for '{name}'"
                continue

            try:
                agent = cls()
                if hasattr(agent, "is_ready") and not agent.is_ready():
                    self._agent_errors[name] = f"Agent '{name}' loaded but not ready"
                    logger.warning("[ExitAdapter] Agent '%s' not ready", name)
                    continue
                self._agents[name] = agent
                logger.info("[ExitAdapter] Loaded agent '%s'", name)
            except Exception as e:
                self._agent_errors[name] = str(e)
                logger.error("[ExitAdapter] Failed to load '%s': %s", name, e)

    @property
    def loaded_models(self) -> List[str]:
        """List of successfully loaded model names."""
        return list(self._agents.keys())

    @property
    def missing_models(self) -> List[str]:
        """Model names that failed to load."""
        return [n for n in self._model_names if n not in self._agents]

    # ── Public API ───────────────────────────────────────────────────────

    def collect_and_normalize(
        self,
        state: PositionExitState,
    ) -> List[ModelExitSignal]:
        """
        Run all loaded models against the position state and return
        normalised ModelExitSignal per model.

        Fail-open per model: if one model errors, others still run.

        Args:
            state: Enriched position state from PositionStateBuilder.

        Returns:
            List of valid ModelExitSignal objects. May be empty.
        """
        features = self._build_feature_dict(state)
        signals: List[ModelExitSignal] = []

        for model_name, agent in self._agents.items():
            try:
                signal = self._predict_and_normalize(model_name, agent, state, features)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.error(
                    "[ExitAdapter] %s predict failed for %s: %s",
                    model_name, state.symbol, e,
                )

        logger.info(
            "[ExitAdapter] %s: %d/%d models produced signals",
            state.symbol, len(signals), len(self._agents),
        )
        return signals

    # ── Feature extraction ───────────────────────────────────────────────

    def _build_feature_dict(self, state: PositionExitState) -> Dict[str, float]:
        """
        Build a feature dictionary from PositionExitState for model input.

        ASSUMPTION: Models expect the same feature dict that ensemble_manager
        currently builds. This method converts PositionExitState fields to
        that expected format.
        """
        features: Dict[str, float] = {}

        # Core price features
        features["entry_price"] = state.entry_price
        features["current_price"] = state.current_price
        features["mark_price"] = state.mark_price
        features["quantity"] = state.quantity
        features["notional"] = state.notional
        features["leverage"] = state.leverage

        # PnL
        features["unrealized_pnl"] = state.unrealized_pnl
        features["unrealized_pnl_pct"] = state.unrealized_pnl_pct
        features["realized_pnl"] = state.realized_pnl

        # Excursions
        features["max_favorable_excursion"] = state.max_favorable_excursion
        features["max_adverse_excursion"] = state.max_adverse_excursion
        features["drawdown_from_peak_pnl"] = state.drawdown_from_peak_pnl

        # Time
        features["hold_seconds"] = state.hold_seconds

        # Volatility
        if state.volatility_short is not None:
            features["volatility_short"] = state.volatility_short
        if state.volatility_medium is not None:
            features["volatility_medium"] = state.volatility_medium
        if state.atr is not None:
            features["atr"] = state.atr

        # Regime
        if state.trend_signal is not None:
            features["trend_signal"] = state.trend_signal
        features["regime_confidence"] = state.regime_confidence

        # Scores
        if state.momentum_score is not None:
            features["momentum_score"] = state.momentum_score
        if state.mean_reversion_score is not None:
            features["mean_reversion_score"] = state.mean_reversion_score
        if state.liquidity_score is not None:
            features["liquidity_score"] = state.liquidity_score
        if state.spread_bps is not None:
            features["spread_bps"] = state.spread_bps

        # Side encoding
        features["side_long"] = 1.0 if state.side == "LONG" else 0.0
        features["side_short"] = 1.0 if state.side == "SHORT" else 0.0

        return features

    # ── Per-model prediction ─────────────────────────────────────────────

    def _predict_and_normalize(
        self,
        model_name: str,
        agent: Any,
        state: PositionExitState,
        features: Dict[str, float],
    ) -> Optional[ModelExitSignal]:
        """
        Call one model's predict() and normalize result to ModelExitSignal.

        Returns None on failure.
        """
        spec = get_model_spec(model_name)
        if spec is None:
            return None

        t0 = time.time()
        raw = agent.predict(state.symbol, features)
        inference_time = time.time()

        if raw is None:
            logger.warning("[ExitAdapter] %s returned None for %s", model_name, state.symbol)
            return None

        # Extract action + confidence from raw output
        action = raw.get("action", "HOLD")
        confidence = clamp(float(raw.get("confidence", 0.5)), 0.01, 0.99)
        version = raw.get("version", "unknown")

        # Reconstruct 3-class probabilities
        sell_p, hold_p, buy_p = reconstruct_probabilities_from_action(action, confidence)

        # Calibrate (v1: identity)
        sell_p, hold_p, buy_p = identity_calibrate([sell_p, hold_p, buy_p])

        # Renormalize (safety net)
        sell_p, hold_p, buy_p = renormalize_probabilities([sell_p, hold_p, buy_p])

        # Derive directional probabilities
        continuation_p, reversal_p = derive_directional_probabilities(
            sell_p, hold_p, buy_p, state.side,
        )

        freshness = time.time() - inference_time

        signal = ModelExitSignal(
            position_id=state.position_id,
            symbol=state.symbol,
            model_name=model_name,
            model_version=version,
            inference_timestamp=inference_time,
            horizon_label=spec.horizon_label,
            sell_probability=sell_p,
            hold_probability=hold_p,
            buy_probability=buy_p,
            continuation_probability=continuation_p,
            reversal_probability=reversal_p,
            confidence=confidence,
            freshness_seconds=freshness,
            feature_version=spec.feature_version,
            source_state_version=f"{state.position_id}@{state.source_timestamps}",
            source_quality_flags=list(state.data_quality_flags),
            shadow_only=True,
        )

        errors = signal.validate()
        if errors:
            logger.warning(
                "[ExitAdapter] %s signal for %s failed validation: %s",
                model_name, state.symbol, errors,
            )
            return None

        return signal

    # ── Shadow publishing helper ─────────────────────────────────────────

    def publish_raw_shadow_signals(
        self,
        signals: List[ModelExitSignal],
        shadow_publisher,
    ) -> int:
        """
        Publish each ModelExitSignal to the raw shadow stream.

        Args:
            signals: List of validated ModelExitSignal.
            shadow_publisher: ShadowPublisher instance with _xadd method.

        Returns:
            Count of successfully published signals.
        """
        stream = "quantum:stream:exit.ensemble.raw.shadow"
        published = 0
        for sig in signals:
            flat = {k: ("" if v is None else str(v)) for k, v in sig.to_dict().items()}
            entry_id = shadow_publisher._xadd(stream, flat)
            if entry_id is not None:
                published += 1
        return published
