"""
ReplayEvaluationRecord — Per-decision replay evaluation.

Phase 5 data contract. Shadow-only. Offline evaluation.

Produced by: offline_evaluator (via counterfactual_evaluator)
Published to: quantum:stream:exit.replay.eval.shadow
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from .action_candidate import VALID_ACTIONS

logger = logging.getLogger(__name__)


@dataclass(frozen=False)
class ReplayEvaluationRecord:
    """
    Evaluation of one decision against realized outcomes.

    Compares predicted beliefs/hazards/utilities against realized proxies.
    Evaluates chosen action vs all counterfactual actions.

    shadow_only MUST be True.
    """

    # ── Identity (REQUIRED) ──────────────────────────────────────────────
    record_id: str                             # uuid4
    position_id: str
    symbol: str
    replay_timestamp: float                    # When this evaluation was run
    source_decision_timestamp: float           # When the original decision was made
    evaluated_horizon_seconds: float           # Lookahead window used (e.g. 14400)

    # ── Decision (REQUIRED) ──────────────────────────────────────────────
    chosen_action: str                         # What Exit Brain recommended
    actual_action: str = ""                    # What actually happened ∪{""}
    action_rank_at_decision: int = 1           # Rank of chosen action at decision time

    # ── Counterfactual (REQUIRED) ────────────────────────────────────────
    counterfactual_actions_evaluated: List[str] = field(default_factory=list)

    # ── Belief calibration (REQUIRED) ────────────────────────────────────
    predicted_exit_pressure: float = 0.0       # [0,1] from belief_snapshot
    realized_exit_pressure_proxy: float = 0.0  # [0,1] derived from price path
    predicted_hold_conviction: float = 0.0     # [0,1]
    realized_hold_conviction_proxy: float = 0.0  # [0,1]

    # ── Hazard calibration (REQUIRED) ────────────────────────────────────
    predicted_composite_hazard: float = 0.0    # [0,1]
    realized_hazard_proxy: float = 0.0         # [0,1]

    # ── Utility evaluation (REQUIRED) ────────────────────────────────────
    predicted_utility_by_action: Dict[str, float] = field(default_factory=dict)
    realized_utility_proxy_by_action: Dict[str, float] = field(default_factory=dict)

    # ── Ex-post analysis (REQUIRED) ──────────────────────────────────────
    ex_post_best_action: str = ""              # Best action in hindsight
    ex_post_best_utility_proxy: float = 0.0

    # ── Scores (REQUIRED) ────────────────────────────────────────────────
    decision_quality_score: float = 0.0        # [0,1] composite quality
    explanation_consistency_score: float = 0.0  # [0,1] did explanation match outcome?

    # ── Calibration errors (REQUIRED) ────────────────────────────────────
    calibration_errors: Dict[str, float] = field(default_factory=dict)
    # e.g. {"exit_pressure": pred - realized, "composite_hazard": pred - realized}

    # ── Quality (OPTIONAL) ───────────────────────────────────────────────
    quality_flags: List[str] = field(default_factory=list)

    # ── Safety (REQUIRED) ────────────────────────────────────────────────
    shadow_only: bool = True

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Return list of error strings. Empty = valid."""
        errors: List[str] = []

        if not self.record_id:
            errors.append("record_id is empty")
        if not self.position_id:
            errors.append("position_id is empty")
        if not self.symbol:
            errors.append("symbol is empty")
        if self.replay_timestamp <= 0:
            errors.append(f"replay_timestamp must be > 0, got {self.replay_timestamp}")
        if self.source_decision_timestamp <= 0:
            errors.append(f"source_decision_timestamp must be > 0, got {self.source_decision_timestamp}")
        if self.evaluated_horizon_seconds <= 0:
            errors.append(f"evaluated_horizon_seconds must be > 0, got {self.evaluated_horizon_seconds}")

        if self.chosen_action not in VALID_ACTIONS:
            errors.append(f"chosen_action '{self.chosen_action}' not in {VALID_ACTIONS}")
        if self.actual_action and self.actual_action not in VALID_ACTIONS:
            errors.append(f"actual_action '{self.actual_action}' not in {VALID_ACTIONS}")
        if self.ex_post_best_action and self.ex_post_best_action not in VALID_ACTIONS:
            errors.append(f"ex_post_best_action '{self.ex_post_best_action}' not in {VALID_ACTIONS}")

        if self.action_rank_at_decision < 1:
            errors.append(f"action_rank_at_decision must be >= 1, got {self.action_rank_at_decision}")

        for name, val in [
            ("predicted_exit_pressure", self.predicted_exit_pressure),
            ("realized_exit_pressure_proxy", self.realized_exit_pressure_proxy),
            ("predicted_hold_conviction", self.predicted_hold_conviction),
            ("realized_hold_conviction_proxy", self.realized_hold_conviction_proxy),
            ("predicted_composite_hazard", self.predicted_composite_hazard),
            ("realized_hazard_proxy", self.realized_hazard_proxy),
            ("decision_quality_score", self.decision_quality_score),
            ("explanation_consistency_score", self.explanation_consistency_score),
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
            "record_id": self.record_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "replay_timestamp": self.replay_timestamp,
            "source_decision_timestamp": self.source_decision_timestamp,
            "evaluated_horizon_seconds": self.evaluated_horizon_seconds,
            "chosen_action": self.chosen_action,
            "actual_action": self.actual_action,
            "action_rank_at_decision": self.action_rank_at_decision,
            "counterfactual_actions_evaluated": json.dumps(self.counterfactual_actions_evaluated),
            "predicted_exit_pressure": self.predicted_exit_pressure,
            "realized_exit_pressure_proxy": self.realized_exit_pressure_proxy,
            "predicted_hold_conviction": self.predicted_hold_conviction,
            "realized_hold_conviction_proxy": self.realized_hold_conviction_proxy,
            "predicted_composite_hazard": self.predicted_composite_hazard,
            "realized_hazard_proxy": self.realized_hazard_proxy,
            "predicted_utility_by_action": json.dumps(self.predicted_utility_by_action),
            "realized_utility_proxy_by_action": json.dumps(self.realized_utility_proxy_by_action),
            "ex_post_best_action": self.ex_post_best_action,
            "ex_post_best_utility_proxy": self.ex_post_best_utility_proxy,
            "decision_quality_score": self.decision_quality_score,
            "explanation_consistency_score": self.explanation_consistency_score,
            "calibration_errors": json.dumps(self.calibration_errors),
            "quality_flags": ",".join(self.quality_flags),
            "shadow_only": self.shadow_only,
        }
