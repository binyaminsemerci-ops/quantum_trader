"""ensemble_bridge: Bridge between Exit Brain v1 ensemble pipeline and Exit Management Agent.

Wraps the full Exit Brain v1 Phase 1→4 pipeline into a single async-callable
that returns an EnsembleBridgeResult compatible with ExitDecision construction.

This module is the ONLY integration point between the two systems.
Exit Brain v1 components are imported and called synchronously; we wrap
calls in asyncio.to_thread() because model inference is CPU-bound.

When scoring_mode="ensemble" in AgentConfig, this bridge replaces:
  - PerceptionEngine (state comes from PositionStateBuilder)
  - ScoringEngine / DecisionEngine (decisions come from ExitPolicyEngine)

Hard guards still fire BEFORE the ensemble pipeline.
Qwen3 can optionally refine the ensemble output (future extension).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import redis as redis_sync

from .models import PositionSnapshot

# Exit Brain v1 components (all synchronous)
from microservices.exit_brain_v1.services.position_state_builder import PositionStateBuilder
from microservices.exit_brain_v1.engines.geometry_engine import GeometryEngine, GeometryResult
from microservices.exit_brain_v1.engines.regime_drift_engine import RegimeDriftEngine, RegimeState
from microservices.exit_brain_v1.adapters.ensemble_exit_adapter import EnsembleExitAdapter
from microservices.exit_brain_v1.aggregators.ensemble_aggregator import EnsembleAggregator
from microservices.exit_brain_v1.engines.belief_engine import BeliefEngine
from microservices.exit_brain_v1.engines.hazard_engine import HazardEngine
from microservices.exit_brain_v1.engines.action_utility_engine import ActionUtilityEngine
from microservices.exit_brain_v1.orchestrators.exit_agent_orchestrator import ExitAgentOrchestrator
from microservices.exit_brain_v1.publishers.shadow_publisher import ShadowPublisher
from microservices.exit_brain_v1.models.action_candidate import ACTION_EXIT_FRACTIONS

_log = logging.getLogger("exit_management_agent.ensemble_bridge")

# Map from EB v1 action → qty fraction for ExitDecision
ENSEMBLE_QTY_MAP = {
    "HOLD": None,
    "REDUCE_SMALL": 0.10,
    "REDUCE_MEDIUM": 0.25,
    "TAKE_PROFIT_PARTIAL": 0.50,
    "TAKE_PROFIT_LARGE": 0.75,
    "TIGHTEN_EXIT": None,
    "CLOSE_FULL": 1.0,
}


@dataclass(frozen=True)
class EnsembleBridgeResult:
    """Result from the ensemble pipeline, ready for ExitDecision construction."""
    symbol: str
    action: str                 # e.g. HOLD, REDUCE_SMALL, CLOSE_FULL
    reason: str
    urgency: str                # LOW | MEDIUM | HIGH | EMERGENCY
    confidence: float
    exit_fraction: float
    snap: PositionSnapshot      # constructed from PositionExitState
    elapsed_ms: float
    n_models_loaded: int
    policy_blocks: list


@dataclass(frozen=True)
class EnsemblePipelineContext:
    """Rich intermediate state for PATCH-11 LLM consumption."""
    state: object          # PositionExitState
    geometry: object       # GeometryResult
    regime: object         # RegimeState
    belief: object         # BeliefState
    hazard: object         # HazardAssessment
    candidates: list       # List[ActionCandidate]
    decision: object       # PolicyDecision
    agg_signal: object     # AggregatedExitSignal


def _build_geometry(state) -> GeometryResult:
    """Compute Phase 1 geometry from position state."""
    peak_price = state.current_price
    trough_price = state.entry_price
    return GeometryEngine.compute_all(
        entry_price=state.entry_price,
        current_price=state.current_price,
        peak_price=peak_price,
        trough_price=trough_price,
        side=state.side,
        current_pnl=state.unrealized_pnl,
        peak_pnl=max(state.unrealized_pnl, 0),
        pnl_history=[state.unrealized_pnl],
    )


def _build_regime(state) -> RegimeState:
    """Compute Phase 1 regime analysis from position state."""
    label = state.regime_label or "UNKNOWN"
    default_probs = {"TREND": 0.33, "MR": 0.33, "CHOP": 0.34}
    if label in ("BULL", "BEAR"):
        default_probs = {"TREND": 0.70, "MR": 0.15, "CHOP": 0.15}
    elif label == "RANGE":
        default_probs = {"TREND": 0.15, "MR": 0.70, "CHOP": 0.15}
    elif label == "VOLATILE":
        default_probs = {"TREND": 0.15, "MR": 0.15, "CHOP": 0.70}

    return RegimeDriftEngine.summarize_regime_state(
        side=state.side,
        regime_probs=default_probs,
        mu=state.trend_signal or 0.0,
        sigma=state.volatility_short or 0.0,
        ts=state.regime_confidence,
    )


def _state_to_snapshot(state) -> PositionSnapshot:
    """Convert PositionExitState → PositionSnapshot for EMA compatibility."""
    return PositionSnapshot(
        symbol=state.symbol,
        side=state.side,
        quantity=state.quantity,
        entry_price=state.entry_price,
        mark_price=state.mark_price if state.mark_price else state.current_price,
        leverage=state.leverage,
        stop_loss=0.0,
        take_profit=0.0,
        unrealized_pnl=state.unrealized_pnl,
        entry_risk_usdt=0.0,
        sync_timestamp=max(state.source_timestamps.values(), default=time.time()),
    )


def _action_to_urgency(action: str, hazard_emergency: bool) -> str:
    """Map ensemble action + hazard to an urgency level."""
    if hazard_emergency:
        return "EMERGENCY"
    if action == "CLOSE_FULL":
        return "HIGH"
    if action in ("TAKE_PROFIT_LARGE", "TAKE_PROFIT_PARTIAL", "REDUCE_MEDIUM"):
        return "MEDIUM"
    return "LOW"


class EnsembleBridge:
    """
    Bridge that runs the full Exit Brain v1 pipeline for use inside
    Exit Management Agent's async main loop.

    Lifecycle:
        bridge = EnsembleBridge(redis_host, redis_port)
        symbols = await bridge.discover_positions()
        result = await bridge.evaluate(symbol)
        if result is not None:
            # build ExitDecision from result
    """

    def __init__(self, redis_host: str, redis_port: int) -> None:
        self._r = redis_sync.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=False,
        )
        self._builder = PositionStateBuilder(self._r)
        self._publisher = ShadowPublisher(self._r)
        self._adapter = EnsembleExitAdapter()
        self._aggregator = EnsembleAggregator()
        _log.info(
            "[EnsembleBridge] Initialised: %d/%d models loaded: %s",
            len(self._adapter.loaded_models),
            len(self._adapter.loaded_models) + len(self._adapter.missing_models),
            self._adapter.loaded_models,
        )
        if self._adapter.missing_models:
            _log.warning(
                "[EnsembleBridge] Missing models: %s", self._adapter.missing_models,
            )

    @property
    def n_models_loaded(self) -> int:
        return len(self._adapter.loaded_models)

    async def discover_positions(self) -> List[str]:
        """Discover open positions from EB v1's Redis sources (thread-safe)."""
        return await asyncio.to_thread(self._builder.discover_open_positions)

    async def evaluate(self, symbol: str) -> Optional[EnsembleBridgeResult]:
        """
        Run full Phase 1→4 pipeline for one symbol.

        Returns None on fail-closed (missing data, <2 models, etc.).
        Never raises — all exceptions caught and logged.
        """
        return await asyncio.to_thread(self._evaluate_sync, symbol)

    async def evaluate_for_judge(self, symbol: str) -> Optional[tuple]:
        """Run pipeline and return (EnsembleBridgeResult, EnsemblePipelineContext) for PATCH-11.

        Returns None on fail-closed.
        """
        return await asyncio.to_thread(self._run_pipeline, symbol)

    def _evaluate_sync(self, symbol: str) -> Optional[EnsembleBridgeResult]:
        """Synchronous pipeline — called from thread executor."""
        pair = self._run_pipeline(symbol)
        return pair[0] if pair is not None else None

    def _run_pipeline(self, symbol: str) -> Optional[tuple]:
        """Run full Phase 1→4 pipeline, return (EnsembleBridgeResult, EnsemblePipelineContext) or None."""
        t0 = time.monotonic()

        try:
            # ── Phase 1: State + Geometry + Regime ───────────────────
            state = self._builder.build(symbol)
            if state is None:
                _log.warning("[EnsembleBridge] %s: state build failed (fail-closed)", symbol)
                return None

            self._publisher.publish_state(state)

            geometry = _build_geometry(state)
            self._publisher.publish_geometry(symbol, state.side, geometry)

            regime = _build_regime(state)
            self._publisher.publish_regime(regime)

            # ── Phase 2: Ensemble signals + aggregation ──────────────
            signals = self._adapter.collect_and_normalize(state)

            agg_signal, diagnostics = self._aggregator.aggregate(
                signals=signals,
                side=state.side,
            )

            if agg_signal is None:
                _log.warning(
                    "[EnsembleBridge] %s: aggregation failed (<2 models), fail-closed", symbol,
                )
                return None

            # ── Phase 3: Belief + Hazard + Utility ───────────────────
            belief = BeliefEngine.compute(state, geometry, regime, agg_signal)
            if belief is None:
                _log.warning("[EnsembleBridge] %s: belief failed, fail-closed", symbol)
                return None
            self._publisher.publish_belief(belief)

            hazard = HazardEngine.assess(state, geometry, regime, agg_signal)
            self._publisher.publish_hazard(hazard)

            candidates = ActionUtilityEngine.score_all(
                state=state,
                belief=belief,
                hazard=hazard,
            )
            self._publisher.publish_utility(
                candidates=candidates,
                position_id=state.position_id,
                symbol=symbol,
            )

            # ── Phase 4: Policy + Orchestrator ───────────────────────
            orchestrator = ExitAgentOrchestrator(publisher=self._publisher)
            decision = orchestrator.run_decision_cycle(
                candidates=candidates,
                belief=belief,
                hazard=hazard,
                state=state,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            if decision is None:
                _log.warning("[EnsembleBridge] %s: orchestrator returned None", symbol)
                return None

            action = decision.chosen_action
            snap = _state_to_snapshot(state)
            hazard_emergency = hazard.composite_hazard >= 0.8
            urgency = _action_to_urgency(action, hazard_emergency)

            bridge_result = EnsembleBridgeResult(
                symbol=symbol,
                action=action,
                reason="; ".join(decision.explanation_tags) if decision.explanation_tags else action,
                urgency=urgency,
                confidence=decision.decision_confidence,
                exit_fraction=ENSEMBLE_QTY_MAP.get(action, 0.0) or 0.0,
                snap=snap,
                elapsed_ms=elapsed_ms,
                n_models_loaded=len(self._adapter.loaded_models),
                policy_blocks=list(decision.policy_blocks),
            )
            ctx = EnsemblePipelineContext(
                state=state,
                geometry=geometry,
                regime=regime,
                belief=belief,
                hazard=hazard,
                candidates=candidates,
                decision=decision,
                agg_signal=agg_signal,
            )
            return (bridge_result, ctx)

        except Exception:
            _log.exception("[EnsembleBridge] %s: unhandled error — fail-closed", symbol)
            return None
