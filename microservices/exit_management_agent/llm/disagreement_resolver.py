"""PATCH-11 — Disagreement resolver with conservative middle-action policy.

When primary judge and fallback disagree, this module applies a safe
middle-action policy to prevent catastrophic decisions.

No IO. No state. Pure logic.

Conservative middle-action rules:
  HOLD vs FULL_CLOSE        → REDUCE_25 or HARVEST_70_KEEP_30 (based on hazard)
  HOLD vs TOXICITY_UNWIND   → REDUCE_50 (if toxicity high)
  FULL_CLOSE vs HARVEST     → HARVEST_70_KEEP_30 (unless hard risk forces close)
  REDUCE_25 vs REDUCE_50    → REDUCE_25 (conservative)
  Any disagreement with QUARANTINE → QUARANTINE wins (safety)
  Any disagreement involving FLIP → QUARANTINE (not available in PATCH-11)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..patch11_actions import (
    ACTION_TO_FAMILY,
    PATCH11_ACTIONS,
)

_log = logging.getLogger("exit_management_agent.llm.disagreement_resolver")


@dataclass(frozen=True)
class ResolutionResult:
    """Result of disagreement resolution."""
    resolved_action: str           # Final action after resolution
    resolution_method: str         # How it was resolved
    primary_action: str
    fallback_action: str
    agreed: bool                   # True if same family


def resolve_disagreement(
    primary_action: str,
    primary_confidence: float,
    fallback_action: str,
    fallback_confidence: float,
    composite_hazard: float,
    toxicity_hazard: float = 0.0,
) -> ResolutionResult:
    """
    Resolve disagreement between primary and fallback judges.

    Args:
        primary_action: Action from Qwen3-32b.
        fallback_action: Action from GPT-OSS 20B.
        primary_confidence: Confidence from primary [0,1].
        fallback_confidence: Confidence from fallback [0,1].
        composite_hazard: Overall hazard level from hazard engine [0,1].
        toxicity_hazard: Toxicity component of hazard [0,1].

    Returns:
        ResolutionResult with the conservative middle action.
    """
    p_family = ACTION_TO_FAMILY.get(primary_action, "UNKNOWN")
    f_family = ACTION_TO_FAMILY.get(fallback_action, "UNKNOWN")

    # Same action family → proceed with primary
    if p_family == f_family:
        # Within same family, pick the one with higher confidence
        if primary_confidence >= fallback_confidence:
            return ResolutionResult(
                resolved_action=primary_action,
                resolution_method="SAME_FAMILY_PRIMARY",
                primary_action=primary_action,
                fallback_action=fallback_action,
                agreed=True,
            )
        else:
            return ResolutionResult(
                resolved_action=fallback_action,
                resolution_method="SAME_FAMILY_FALLBACK_HIGHER_CONF",
                primary_action=primary_action,
                fallback_action=fallback_action,
                agreed=True,
            )

    # Any QUARANTINE → QUARANTINE wins (safety-first)
    if primary_action == "QUARANTINE" or fallback_action == "QUARANTINE":
        return ResolutionResult(
            resolved_action="QUARANTINE",
            resolution_method="QUARANTINE_SAFETY",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    actions = frozenset({primary_action, fallback_action})
    families = frozenset({p_family, f_family})

    # HOLD_FAMILY vs EXIT_FAMILY → conservative partial exit
    if "HOLD_FAMILY" in families and "EXIT_FAMILY" in families:
        if composite_hazard >= 0.7:
            return ResolutionResult(
                resolved_action="HARVEST_70_KEEP_30",
                resolution_method="HOLD_vs_EXIT_HIGH_HAZARD",
                primary_action=primary_action,
                fallback_action=fallback_action,
                agreed=False,
            )
        return ResolutionResult(
            resolved_action="REDUCE_25",
            resolution_method="HOLD_vs_EXIT_LOW_HAZARD",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # HOLD_FAMILY vs REDUCE_FAMILY → conservative REDUCE_25
    if "HOLD_FAMILY" in families and "REDUCE_FAMILY" in families:
        return ResolutionResult(
            resolved_action="REDUCE_25",
            resolution_method="HOLD_vs_REDUCE",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # HOLD_FAMILY vs HARVEST_FAMILY → REDUCE_25 (conservative)
    if "HOLD_FAMILY" in families and "HARVEST_FAMILY" in families:
        return ResolutionResult(
            resolved_action="REDUCE_25",
            resolution_method="HOLD_vs_HARVEST",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # EXIT_FAMILY vs HARVEST_FAMILY → HARVEST unless hard risk
    if "EXIT_FAMILY" in families and "HARVEST_FAMILY" in families:
        if composite_hazard >= 0.8:
            return ResolutionResult(
                resolved_action="FULL_CLOSE",
                resolution_method="EXIT_vs_HARVEST_CRITICAL_HAZARD",
                primary_action=primary_action,
                fallback_action=fallback_action,
                agreed=False,
            )
        return ResolutionResult(
            resolved_action="HARVEST_70_KEEP_30",
            resolution_method="EXIT_vs_HARVEST_MODERATE",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # REDUCE_FAMILY vs EXIT_FAMILY → REDUCE_50
    if "REDUCE_FAMILY" in families and "EXIT_FAMILY" in families:
        return ResolutionResult(
            resolved_action="REDUCE_50",
            resolution_method="REDUCE_vs_EXIT",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # REDUCE_FAMILY vs HARVEST_FAMILY → REDUCE_50
    if "REDUCE_FAMILY" in families and "HARVEST_FAMILY" in families:
        return ResolutionResult(
            resolved_action="REDUCE_50",
            resolution_method="REDUCE_vs_HARVEST",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # Any toxicity-related disagreement with high toxicity → REDUCE_50
    if "TOXICITY_UNWIND" in actions and toxicity_hazard >= 0.7:
        return ResolutionResult(
            resolved_action="REDUCE_50",
            resolution_method="TOXICITY_COMPROMISE",
            primary_action=primary_action,
            fallback_action=fallback_action,
            agreed=False,
        )

    # Default fallback: most conservative between the two
    # Use REDUCE_25 as universal safe middle ground
    _log.warning(
        "[DisagreementResolver] Unhandled pair: %s (%s) vs %s (%s) → REDUCE_25",
        primary_action, p_family, fallback_action, f_family,
    )
    return ResolutionResult(
        resolved_action="REDUCE_25",
        resolution_method="DEFAULT_CONSERVATIVE",
        primary_action=primary_action,
        fallback_action=fallback_action,
        agreed=False,
    )
