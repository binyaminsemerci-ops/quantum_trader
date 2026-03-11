"""
OfflineEvaluationSummary — Aggregated evaluation of Exit Brain decisions.

Phase 5 data contract. Shadow-only. Offline evaluation.

Produced by: offline_evaluator.py
Published to: quantum:stream:exit.eval.summary.shadow
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class OfflineEvaluationSummary:
    """
    Aggregated metrics from one offline evaluation run.

    Covers all evaluated decisions within a time window.
    Includes per-component calibration, baseline comparison,
    and optional per-symbol/regime breakdowns.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    evaluation_run_id: str                     # uuid4
    run_timestamp: float                       # Epoch when evaluation ran

    # ── Scope (REQUIRED) ─────────────────────────────────────────────────
    time_window_start: float                   # Epoch start of evaluated window
    time_window_end: float                     # Epoch end of evaluated window
    symbols_covered: List[str] = field(default_factory=list)
    positions_covered: int = 0
    decisions_covered: int = 0
    obituaries_covered: int = 0

    # ── Baselines (REQUIRED) ─────────────────────────────────────────────
    baseline_definitions: Dict[str, str] = field(default_factory=dict)
    # e.g. {"always_hold": "Never exit, measure PnL at horizon", ...}

    # ── Action distribution (REQUIRED) ───────────────────────────────────
    action_distribution: Dict[str, int] = field(default_factory=dict)
    # e.g. {"HOLD": 42, "CLOSE_FULL": 3, ...}

    # ── Aggregate quality scores (REQUIRED) ──────────────────────────────
    mean_decision_quality_score: float = 0.0   # [0,1]
    median_decision_quality_score: float = 0.0 # [0,1]
    mean_regret_score: float = 0.0             # [0,1] lower = better
    mean_preservation_score: float = 0.0       # [0,1] higher = better
    mean_opportunity_capture_score: float = 0.0  # [0,1] higher = better

    # ── Per-component calibration (REQUIRED) ─────────────────────────────
    belief_calibration_summary: Dict[str, float] = field(default_factory=dict)
    # e.g. {"exit_pressure_bias": 0.03, "exit_pressure_mae": 0.12, ...}

    hazard_calibration_summary: Dict[str, float] = field(default_factory=dict)
    # e.g. {"composite_hazard_bias": -0.05, "composite_hazard_mae": 0.15, ...}

    utility_ranking_summary: Dict[str, float] = field(default_factory=dict)
    # e.g. {"rank_accuracy": 0.65, "top3_accuracy": 0.85, "mean_rank_displacement": 0.8}

    policy_choice_summary: Dict[str, float] = field(default_factory=dict)
    # e.g. {"block_accuracy": 0.90, "pass_quality": 0.7, ...}

    # ── Baseline comparison (REQUIRED) ───────────────────────────────────
    baseline_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g. {"always_hold": {"mean_pnl": 1.2, "vs_exit_brain_pnl_delta": -0.5}, ...}

    # ── Segmentation (OPTIONAL) ──────────────────────────────────────────
    per_symbol_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_regime_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── Warnings (OPTIONAL) ──────────────────────────────────────────────
    sample_size_warnings: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.evaluation_run_id:
            errors.append("evaluation_run_id is empty")
        if self.run_timestamp <= 0:
            errors.append(f"run_timestamp must be > 0, got {self.run_timestamp}")
        if self.time_window_start <= 0:
            errors.append(f"time_window_start must be > 0, got {self.time_window_start}")
        if self.time_window_end <= self.time_window_start:
            errors.append("time_window_end must be > time_window_start")
        if self.positions_covered < 0:
            errors.append(f"positions_covered must be >= 0, got {self.positions_covered}")
        if self.decisions_covered < 0:
            errors.append(f"decisions_covered must be >= 0, got {self.decisions_covered}")

        for name, val in [
            ("mean_decision_quality_score", self.mean_decision_quality_score),
            ("median_decision_quality_score", self.median_decision_quality_score),
            ("mean_regret_score", self.mean_regret_score),
            ("mean_preservation_score", self.mean_preservation_score),
            ("mean_opportunity_capture_score", self.mean_opportunity_capture_score),
        ]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} must be in [0, 1], got {val}")

        if not self.shadow_only:
            errors.append("shadow_only MUST be True in Phase 5")

        return errors

    def to_dict(self) -> Dict:
        """Serialize to flat dict for Redis XADD."""
        import json
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "run_timestamp": self.run_timestamp,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "symbols_covered": json.dumps(self.symbols_covered),
            "positions_covered": self.positions_covered,
            "decisions_covered": self.decisions_covered,
            "obituaries_covered": self.obituaries_covered,
            "baseline_definitions": json.dumps(self.baseline_definitions),
            "action_distribution": json.dumps(self.action_distribution),
            "mean_decision_quality_score": self.mean_decision_quality_score,
            "median_decision_quality_score": self.median_decision_quality_score,
            "mean_regret_score": self.mean_regret_score,
            "mean_preservation_score": self.mean_preservation_score,
            "mean_opportunity_capture_score": self.mean_opportunity_capture_score,
            "belief_calibration_summary": json.dumps(self.belief_calibration_summary),
            "hazard_calibration_summary": json.dumps(self.hazard_calibration_summary),
            "utility_ranking_summary": json.dumps(self.utility_ranking_summary),
            "policy_choice_summary": json.dumps(self.policy_choice_summary),
            "baseline_comparison": json.dumps(self.baseline_comparison),
            "per_symbol_scores": json.dumps(self.per_symbol_scores),
            "per_regime_scores": json.dumps(self.per_regime_scores),
            "sample_size_warnings": json.dumps(self.sample_size_warnings),
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
