#!/usr/bin/env python3
"""
run_exit_brain.py — Shadow-mode runner for Exit Brain v1.

Connects to Redis, discovers open positions, and runs the full
Phase 1→2→3→4 pipeline per position on a configurable loop interval.

All output goes to shadow streams only. No execution writes.
No apply.plan. No trade.intent. Fail-closed throughout.

Usage:
    python -m microservices.exit_brain_v1.run_exit_brain          # loop
    python -m microservices.exit_brain_v1.run_exit_brain --once   # single cycle
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional

import redis

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("exit_brain_v1.runner")

LOOP_INTERVAL_SEC = int(os.environ.get("EXIT_BRAIN_INTERVAL", "30"))
REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))


def build_geometry(state) -> GeometryResult:
    """Compute Phase 1 geometry from position state using compute_all."""
    peak_price = state.current_price  # best approximation without history
    trough_price = state.entry_price  # best approximation without history
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


def build_regime(state) -> RegimeState:
    """Compute Phase 1 regime analysis from position state."""
    # PositionExitState doesn't carry regime_probs; use defaults based on regime_label
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


def run_cycle(
    r: redis.Redis,
    builder: PositionStateBuilder,
    adapter: EnsembleExitAdapter,
    aggregator: EnsembleAggregator,
    publisher: ShadowPublisher,
) -> int:
    """
    Run one full decision cycle for all open positions.
    Returns count of positions processed.
    """
    symbols = builder.discover_open_positions()
    if not symbols:
        logger.info("[Cycle] No open positions found")
        return 0

    logger.info("[Cycle] Found %d open positions: %s", len(symbols), symbols)
    processed = 0

    for symbol in symbols:
        try:
            # ── Phase 1: Build state + geometry + regime ─────────────
            state = builder.build(symbol)
            if state is None:
                logger.warning("[Cycle] %s: state build failed (fail-closed skip)", symbol)
                continue

            publisher.publish_state(state)

            geometry = build_geometry(state)
            publisher.publish_geometry(symbol, state.side, geometry)

            regime = build_regime(state)
            publisher.publish_regime(regime)

            # ── Phase 2: Ensemble signals + aggregation ──────────────
            signals = adapter.collect_and_normalize(state)

            agg_signal, diagnostics = aggregator.aggregate(
                signals=signals,
                side=state.side,
            )

            if agg_signal is None:
                logger.warning("[Cycle] %s: ensemble aggregation failed (<2 models), skip", symbol)
                continue

            # ── Phase 3: Belief + Hazard + Utility ───────────────────
            belief = BeliefEngine.compute(state, geometry, regime, agg_signal)
            if belief is None:
                logger.warning("[Cycle] %s: belief computation failed, skip", symbol)
                continue
            publisher.publish_belief(belief)

            hazard = HazardEngine.assess(state, geometry, regime, agg_signal)
            publisher.publish_hazard(hazard)

            candidates = ActionUtilityEngine.score_all(
                state=state,
                belief=belief,
                hazard=hazard,
            )
            publisher.publish_utility(
                candidates=candidates,
                position_id=state.position_id,
                symbol=symbol,
            )

            # ── Phase 4: Policy + Orchestrator ───────────────────────
            orchestrator = ExitAgentOrchestrator(publisher=publisher)
            decision = orchestrator.run_decision_cycle(
                candidates=candidates,
                belief=belief,
                hazard=hazard,
                state=state,
            )

            action = decision.chosen_action if decision else "FAIL_CLOSED"
            logger.info(
                "[Cycle] %s: decision=%s (shadow-only)",
                symbol, action,
            )
            processed += 1

        except Exception:
            logger.exception("[Cycle] %s: unhandled error — fail-closed skip", symbol)
            continue

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Exit Brain v1 shadow runner")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    args = parser.parse_args()

    logger.info("═══ Exit Brain v1 Shadow Runner ═══")
    logger.info("Redis: %s:%d/%d", REDIS_HOST, REDIS_PORT, REDIS_DB)
    logger.info("Interval: %ds", LOOP_INTERVAL_SEC)
    logger.info("Mode: %s", "single-shot" if args.once else "loop")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    try:
        r.ping()
    except redis.ConnectionError:
        logger.error("Cannot connect to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        sys.exit(1)

    builder = PositionStateBuilder(r)
    publisher = ShadowPublisher(r)
    adapter = EnsembleExitAdapter()
    aggregator = EnsembleAggregator()

    cycle_count = 0
    while True:
        cycle_count += 1
        t0 = time.time()
        logger.info("── Cycle %d start ──", cycle_count)

        n = run_cycle(r, builder, adapter, aggregator, publisher)
        elapsed = time.time() - t0
        logger.info("── Cycle %d done: %d positions in %.2fs ──", cycle_count, n, elapsed)

        if args.once:
            logger.info("Single-shot mode — exiting")
            break

        time.sleep(LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    main()
