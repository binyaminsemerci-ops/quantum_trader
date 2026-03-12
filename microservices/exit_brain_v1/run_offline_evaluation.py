#!/usr/bin/env python3
"""
run_offline_evaluation.py — Periodic runner for Phase 5 Offline Evaluator.

Connects to Redis, runs the full evaluation pipeline over
a configurable time window (default: last 4 hours), and
publishes results to shadow streams.

Intended to be called periodically by a systemd timer.

Usage:
    python -m microservices.exit_brain_v1.run_offline_evaluation
    python -m microservices.exit_brain_v1.run_offline_evaluation --window-hours 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import redis

from microservices.exit_brain_v1.evaluators.offline_evaluator import OfflineEvaluator
from microservices.exit_brain_v1.publishers.shadow_publisher import ShadowPublisher
from microservices.exit_brain_v1.tuning.proposal_builder import ProposalBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("exit_brain_v1.offline_eval_runner")

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
DEFAULT_WINDOW_HOURS = float(os.environ.get("EVAL_WINDOW_HOURS", "4"))
HORIZON_SECONDS = float(os.environ.get("EVAL_HORIZON_SECONDS", "14400"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exit Brain v1 Offline Evaluator")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=DEFAULT_WINDOW_HOURS,
        help="Evaluation window in hours (default: 4)",
    )
    args = parser.parse_args()

    logger.info("═══ Exit Brain v1 Offline Evaluator ═══")
    logger.info("Redis: %s:%d/%d", REDIS_HOST, REDIS_PORT, REDIS_DB)
    logger.info("Window: %.1f hours, Horizon: %.0fs", args.window_hours, HORIZON_SECONDS)

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    try:
        r.ping()
    except redis.ConnectionError:
        logger.error("Cannot connect to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        sys.exit(1)

    publisher = ShadowPublisher(r)
    evaluator = OfflineEvaluator(
        redis_client=r,
        publisher=publisher,
        horizon_seconds=HORIZON_SECONDS,
    )

    end_ts = time.time()
    start_ts = end_ts - (args.window_hours * 3600)

    logger.info("Evaluation window: %.0f → %.0f", start_ts, end_ts)

    summary = evaluator.run_evaluation(start_ts, end_ts)

    if summary is None:
        logger.warning("Evaluation returned None (insufficient data)")
        sys.exit(0)

    logger.info(
        "Evaluation complete: %d obituaries, quality=%.3f, regret=%.3f",
        summary.obituaries_covered,
        summary.mean_decision_quality_score,
        summary.mean_regret_score,
    )

    # Run tuning proposals
    proposal_builder = ProposalBuilder(publisher)
    recommendations = proposal_builder.build_tuning_proposals(summary)
    if recommendations:
        logger.info("Generated %d tuning recommendations", len(recommendations))
        # Build calibration snapshot for reproducibility
        proposal_builder.build_calibration_snapshot(summary, recommendations)
    else:
        logger.info("No tuning recommendations generated")

    logger.info("═══ Offline Evaluator Done ═══")


if __name__ == "__main__":
    main()
