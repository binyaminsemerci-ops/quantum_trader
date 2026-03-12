"""PATCH-11 — LLM judge orchestrator.

Main pipeline:
  1. Receive EnsemblePipelineContext from ensemble bridge
  2. Build structured prompt
  3. Call primary judge (Qwen3-32b)
  4. Validate response (strict JSON schema)
  5. Check for soft/hard failures
  6. If hard failure → call fallback (GPT-OSS 20B)
  7. If soft failure → call fallback for second opinion → resolve disagreement
  8. If double failure → deterministic safe policy
  9. Return JudgeResult

Never raises — all exceptions caught. Fail-closed → HOLD.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .groq_client import GroqModelClient, ThrottleSkipError
from .response_validator import ValidatedResponse, validate_llm_response
from .judge_prompt import build_user_prompt, get_system_prompt
from .disagreement_resolver import resolve_disagreement, ResolutionResult
from ..patch11_actions import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFLICT_THRESHOLD,
    DEFAULT_LARGE_POSITION_USDT,
    DEFAULT_HIGH_TOXICITY,
    PATCH11_ACTIONS,
    PATCH11_QTY_MAP,
    EBV1_TO_PATCH11,
)

_log = logging.getLogger("exit_management_agent.llm.judge_orchestrator")


@dataclass(frozen=True)
class JudgeResult:
    """Final output from the judge orchestrator."""
    action: str                    # PATCH-11 action
    confidence: float              # [0, 1]
    reason_codes: List[str]        # From LLM response
    why_not: dict                  # {action: reason}
    risk_note: str
    source: str                    # primary|fallback|disagreement|deterministic
    qty_fraction: Optional[float]  # From PATCH11_QTY_MAP
    latency_ms: float
    # Audit metadata
    primary_response: Optional[ValidatedResponse] = None
    fallback_response: Optional[ValidatedResponse] = None
    resolution: Optional[ResolutionResult] = None
    fallback_used: bool = False
    second_opinion_used: bool = False
    hard_failure_reason: Optional[str] = None
    soft_failure_reasons: List[str] = field(default_factory=list)
    shadow_metrics: dict = field(default_factory=dict)


# ── Deterministic safe policy for double failure ──────────────────────────

def _deterministic_safe_policy(
    composite_hazard: float,
    toxicity_hazard: float,
    ensemble_action: str,
) -> JudgeResult:
    """
    Emergency deterministic fallback when both LLM judges fail.

    Rules:
      - Low risk → HOLD (keep existing position)
      - High toxicity/risk → REDUCE_25 or REDUCE_50
      - Emergency → FULL_CLOSE only if composite_hazard >= 0.9
    """
    if composite_hazard >= 0.9:
        action = "FULL_CLOSE"
    elif composite_hazard >= 0.7 or toxicity_hazard >= 0.7:
        action = "REDUCE_50"
    elif composite_hazard >= 0.5:
        action = "REDUCE_25"
    else:
        action = "HOLD"

    return JudgeResult(
        action=action,
        confidence=0.3,
        reason_codes=["CONVICTION_LOW"],
        why_not={},
        risk_note="deterministic safe policy (double LLM failure)",
        source="deterministic",
        qty_fraction=PATCH11_QTY_MAP.get(action),
        latency_ms=0.0,
        hard_failure_reason="DOUBLE_LLM_FAILURE",
    )


# ── Soft failure detection ────────────────────────────────────────────────

def _detect_soft_failures(
    validated: ValidatedResponse,
    ctx: Any,
    confidence_threshold: float,
    conflict_threshold: float,
    large_position_usdt: float,
) -> List[str]:
    """
    Check for soft failure conditions that warrant a second opinion.

    Returns list of trigger reason strings. Empty = no soft failure.
    """
    reasons: List[str] = []

    # 1. Low confidence
    if validated.confidence < confidence_threshold:
        reasons.append(f"LOW_CONFIDENCE:{validated.confidence:.2f}")

    # 2. Contradictory reasoning: action is aggressive but reason_codes suggest hold
    hold_codes = {"EDGE_REMAINING", "CONVICTION_HIGH", "PROFIT_LOCKED"}
    exit_codes = {"THESIS_DECAY", "TOXICITY_CRITICAL", "REVERSAL_SIGNAL", "DRAWDOWN_RISK"}
    has_hold = bool(set(validated.reason_codes) & hold_codes)
    has_exit = bool(set(validated.reason_codes) & exit_codes)
    if has_hold and has_exit:
        reasons.append("CONTRADICTORY_REASONING")

    # 3. High-risk action (FULL_CLOSE, TOXICITY_UNWIND) on large position
    if ctx.state.notional >= large_position_usdt:
        if validated.action in ("FULL_CLOSE", "TOXICITY_UNWIND"):
            reasons.append(f"HIGH_RISK_LARGE_POSITION:${ctx.state.notional:.0f}")

    # 4. Action disagrees strongly with ensemble recommendation
    ensemble_action = EBV1_TO_PATCH11.get(ctx.decision.chosen_action, "HOLD")
    if validated.action != ensemble_action:
        from ..patch11_actions import ACTION_TO_FAMILY
        v_family = ACTION_TO_FAMILY.get(validated.action, "UNKNOWN")
        e_family = ACTION_TO_FAMILY.get(ensemble_action, "UNKNOWN")
        if v_family != e_family:
            reasons.append(f"ENSEMBLE_DISAGREEMENT:{ensemble_action}→{validated.action}")

    # 5. High uncertainty from belief engine
    if ctx.belief.uncertainty_total >= conflict_threshold:
        reasons.append(f"HIGH_UNCERTAINTY:{ctx.belief.uncertainty_total:.2f}")

    return reasons


class JudgeOrchestrator:
    """
    Orchestrates primary → fallback → disagreement resolution → gateway.

    Args:
        primary: GroqModelClient for Qwen3-32b
        fallback: GroqModelClient for GPT-OSS 20B
        confidence_threshold: Below this triggers soft failure
        conflict_threshold: Uncertainty above this triggers second opinion
        large_position_usdt: Positions above this get second opinion on aggressive actions
    """

    def __init__(
        self,
        primary: GroqModelClient,
        fallback: GroqModelClient,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        conflict_threshold: float = DEFAULT_CONFLICT_THRESHOLD,
        large_position_usdt: float = DEFAULT_LARGE_POSITION_USDT,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._confidence_threshold = confidence_threshold
        self._conflict_threshold = conflict_threshold
        self._large_position_usdt = large_position_usdt
        self._system_prompt = get_system_prompt()

    async def evaluate(self, ctx: Any) -> JudgeResult:
        """
        Run the full judge pipeline for one position.

        Args:
            ctx: EnsemblePipelineContext from ensemble_bridge.

        Returns:
            JudgeResult. Never raises. Fail-closed → HOLD or safe action.
        """
        t0 = time.monotonic()
        user_prompt = build_user_prompt(ctx)

        # ── Step 1: Call primary judge ────────────────────────────────
        primary_validated, primary_hard_failure = await self._call_judge(
            self._primary, user_prompt, "primary"
        )

        # ── Step 2: Handle hard failure ───────────────────────────────
        if primary_hard_failure is not None:
            _log.warning(
                "[JudgeOrchestrator] %s primary hard failure: %s → calling fallback",
                ctx.state.symbol, primary_hard_failure,
            )
            fallback_validated, fallback_hard_failure = await self._call_judge(
                self._fallback, user_prompt, "fallback"
            )

            if fallback_hard_failure is not None:
                # Double failure → deterministic safe policy
                _log.warning(
                    "[JudgeOrchestrator] %s DOUBLE FAILURE: primary=%s fallback=%s",
                    ctx.state.symbol, primary_hard_failure, fallback_hard_failure,
                )
                result = _deterministic_safe_policy(
                    composite_hazard=ctx.hazard.composite_hazard,
                    toxicity_hazard=getattr(ctx.hazard, "ensemble_hazard", 0.0),
                    ensemble_action=ctx.decision.chosen_action,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                return JudgeResult(
                    action=result.action,
                    confidence=result.confidence,
                    reason_codes=result.reason_codes,
                    why_not=result.why_not,
                    risk_note=result.risk_note,
                    source="deterministic",
                    qty_fraction=result.qty_fraction,
                    latency_ms=latency_ms,
                    primary_response=primary_validated,
                    fallback_response=fallback_validated,
                    fallback_used=True,
                    hard_failure_reason="DOUBLE_LLM_FAILURE",
                )

            # Fallback succeeded after primary hard failure
            latency_ms = (time.monotonic() - t0) * 1000.0
            return JudgeResult(
                action=fallback_validated.action,
                confidence=fallback_validated.confidence,
                reason_codes=fallback_validated.reason_codes,
                why_not=fallback_validated.why_not,
                risk_note=fallback_validated.risk_note,
                source="fallback",
                qty_fraction=PATCH11_QTY_MAP.get(fallback_validated.action),
                latency_ms=latency_ms,
                primary_response=primary_validated,
                fallback_response=fallback_validated,
                fallback_used=True,
                hard_failure_reason=primary_hard_failure,
            )

        # ── Step 3: Check for soft failures ───────────────────────────
        soft_failures = _detect_soft_failures(
            primary_validated,
            ctx,
            self._confidence_threshold,
            self._conflict_threshold,
            self._large_position_usdt,
        )

        if not soft_failures:
            # Normal path: primary succeeded, no issues
            latency_ms = (time.monotonic() - t0) * 1000.0
            return JudgeResult(
                action=primary_validated.action,
                confidence=primary_validated.confidence,
                reason_codes=primary_validated.reason_codes,
                why_not=primary_validated.why_not,
                risk_note=primary_validated.risk_note,
                source="primary",
                qty_fraction=PATCH11_QTY_MAP.get(primary_validated.action),
                latency_ms=latency_ms,
                primary_response=primary_validated,
            )

        # ── Step 4: Soft failure → call fallback for second opinion ───
        _log.info(
            "[JudgeOrchestrator] %s soft failure triggers: %s → second opinion",
            ctx.state.symbol, soft_failures,
        )
        fallback_validated, fallback_hard_failure = await self._call_judge(
            self._fallback, user_prompt, "fallback"
        )

        if fallback_hard_failure is not None:
            # Fallback failed on second opinion → use primary anyway
            _log.warning(
                "[JudgeOrchestrator] %s fallback failed for second opinion: %s "
                "→ using primary with soft warnings",
                ctx.state.symbol, fallback_hard_failure,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            return JudgeResult(
                action=primary_validated.action,
                confidence=primary_validated.confidence * 0.8,  # dampen
                reason_codes=primary_validated.reason_codes,
                why_not=primary_validated.why_not,
                risk_note=primary_validated.risk_note,
                source="primary",
                qty_fraction=PATCH11_QTY_MAP.get(primary_validated.action),
                latency_ms=latency_ms,
                primary_response=primary_validated,
                fallback_response=fallback_validated,
                second_opinion_used=True,
                soft_failure_reasons=soft_failures,
            )

        # ── Step 5: Resolve disagreement ──────────────────────────────
        resolution = resolve_disagreement(
            primary_action=primary_validated.action,
            primary_confidence=primary_validated.confidence,
            fallback_action=fallback_validated.action,
            fallback_confidence=fallback_validated.confidence,
            composite_hazard=ctx.hazard.composite_hazard,
            toxicity_hazard=getattr(ctx.hazard, "ensemble_hazard", 0.0),
        )

        latency_ms = (time.monotonic() - t0) * 1000.0

        if resolution.agreed:
            # Same family — use resolved action (higher confidence)
            winner = (
                primary_validated if resolution.resolved_action == primary_validated.action
                else fallback_validated
            )
            return JudgeResult(
                action=resolution.resolved_action,
                confidence=winner.confidence,
                reason_codes=winner.reason_codes,
                why_not=winner.why_not,
                risk_note=winner.risk_note,
                source="primary" if winner is primary_validated else "fallback",
                qty_fraction=PATCH11_QTY_MAP.get(resolution.resolved_action),
                latency_ms=latency_ms,
                primary_response=primary_validated,
                fallback_response=fallback_validated,
                resolution=resolution,
                second_opinion_used=True,
                soft_failure_reasons=soft_failures,
            )
        else:
            # Material disagreement — conservative middle action
            _log.info(
                "[JudgeOrchestrator] %s disagreement: %s vs %s → %s (%s)",
                ctx.state.symbol,
                primary_validated.action,
                fallback_validated.action,
                resolution.resolved_action,
                resolution.resolution_method,
            )
            return JudgeResult(
                action=resolution.resolved_action,
                confidence=min(primary_validated.confidence, fallback_validated.confidence) * 0.7,
                reason_codes=list(set(primary_validated.reason_codes + fallback_validated.reason_codes)),
                why_not={},
                risk_note=f"Disagreement resolved: {resolution.resolution_method}",
                source="disagreement",
                qty_fraction=PATCH11_QTY_MAP.get(resolution.resolved_action),
                latency_ms=latency_ms,
                primary_response=primary_validated,
                fallback_response=fallback_validated,
                resolution=resolution,
                second_opinion_used=True,
                soft_failure_reasons=soft_failures,
            )

    async def _call_judge(
        self,
        client: GroqModelClient,
        user_prompt: str,
        label: str,
    ) -> tuple[Optional[ValidatedResponse], Optional[str]]:
        """
        Call one LLM judge and validate the response.

        Returns:
            (validated_response, hard_failure_reason).
            If hard_failure_reason is not None, validated_response may be None.
        """
        try:
            raw = await client.chat(self._system_prompt, user_prompt)
        except ThrottleSkipError as exc:
            _log.warning("[JudgeOrchestrator] %s throttled: %s", label, exc)
            return (None, f"THROTTLE:{label}")
        except asyncio.TimeoutError:
            _log.warning("[JudgeOrchestrator] %s timeout", label)
            return (None, f"TIMEOUT:{label}")
        except Exception as exc:
            status = getattr(exc, "status", 0)
            if status == 429:
                _log.warning("[JudgeOrchestrator] %s 429 rate limited", label)
                return (None, f"RATE_LIMITED_429:{label}")
            if 500 <= status < 600:
                _log.warning("[JudgeOrchestrator] %s server error %d", label, status)
                return (None, f"SERVER_ERROR_{status}:{label}")
            _log.warning("[JudgeOrchestrator] %s error: %s", label, exc)
            return (None, f"HTTP_ERROR:{label}:{type(exc).__name__}")

        validated = validate_llm_response(raw)
        if not validated.valid:
            _log.warning(
                "[JudgeOrchestrator] %s invalid response: %s",
                label, validated.rejection_reason,
            )
            return (validated, f"VALIDATION_FAILED:{label}:{validated.rejection_reason}")

        return (validated, None)
