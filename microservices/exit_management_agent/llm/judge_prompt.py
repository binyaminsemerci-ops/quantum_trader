"""PATCH-11 — System and user prompt builders for LLM exit judges.

Builds structured prompts for primary (Qwen3-32b) and fallback (GPT-OSS 20B).
Both models use identical prompt contracts.

No IO. No state. Pure string formatting.
"""
from __future__ import annotations

import json
from typing import Any, List

from ..patch11_actions import (
    PATCH11_ACTIONS,
    VALID_REASON_CODES,
    EBV1_TO_PATCH11,
)

_SYSTEM_PROMPT = """\
You are a constrained exit policy judge for a quantitative cryptocurrency trading system.

You receive a JSON object containing:
1. Position state: symbol, side, entry price, current price, unrealized PnL, hold time
2. Math engine outputs: geometry (MFE, MAE, drawdown, momentum), regime (drift, trend alignment, reversal risk), belief (exit pressure, hold conviction, uncertainty), hazard (6-axis composite risk)
3. Ensemble model consensus: continuation probability, reversal probability, upside remaining
4. Utility-scored action candidates with rankings
5. Current ensemble recommendation

Your job: Pick ONE action from the allowed set, provide confidence, and explain why.

Allowed actions (exactly one):
  HOLD              — keep position as-is
  REDUCE_25         — exit 25%
  REDUCE_50         — exit 50%
  HARVEST_70_KEEP_30 — exit 70%, keep 30% runner
  FULL_CLOSE        — exit 100%
  DEFENSIVE_HOLD    — hold but tighten risk management
  TOXICITY_UNWIND   — emergency full exit due to toxicity
  QUARANTINE        — freeze position for manual review

Rules:
- Emergency stops are already handled upstream; you will never see them.
- FLIP is not available in PATCH-11.
- If uncertain, prefer conservative actions (HOLD, DEFENSIVE_HOLD, REDUCE_25).
- Do not invent execution parameters (quantity, price, order type, etc.).
- Do not suggest actions outside the allowed set.
- Provide your reasoning via reason_codes only from the known list.

You MUST respond with ONLY a JSON object — no markdown, no code fences, no explanation outside JSON:
{
  "action": "REDUCE_25",
  "confidence": 0.78,
  "reason_codes": ["THESIS_DECAY", "TOXICITY_RISING"],
  "why_not": {
    "HOLD": "thesis weakened and toxicity elevated",
    "FULL_CLOSE": "residual edge still positive"
  },
  "risk_note": "Prefer partial de-risk over full exit."
}

Valid reason codes: """ + ", ".join(sorted(VALID_REASON_CODES))


def get_system_prompt() -> str:
    """Return the system prompt (identical for primary and fallback)."""
    return _SYSTEM_PROMPT


def build_user_prompt(ctx: Any) -> str:
    """
    Build structured user prompt from EnsemblePipelineContext.

    Args:
        ctx: EnsemblePipelineContext from ensemble_bridge.

    Returns:
        JSON string with all decision-relevant state.
    """
    # Map EB v1 ranked candidates to PATCH-11 actions
    ranked_actions = []
    for c in ctx.candidates:
        mapped = EBV1_TO_PATCH11.get(c.action, c.action)
        ranked_actions.append({
            "action": mapped,
            "original_action": c.action,
            "net_utility": round(c.net_utility, 4),
            "rank": c.rank,
        })

    ensemble_action = EBV1_TO_PATCH11.get(
        ctx.decision.chosen_action, ctx.decision.chosen_action
    )

    payload = {
        "position": {
            "symbol": ctx.state.symbol,
            "side": ctx.state.side,
            "entry_price": round(ctx.state.entry_price, 6),
            "current_price": round(ctx.state.current_price, 6),
            "unrealized_pnl": round(ctx.state.unrealized_pnl, 4),
            "unrealized_pnl_pct": round(ctx.state.unrealized_pnl_pct, 4),
            "hold_seconds": round(ctx.state.hold_seconds, 1),
            "notional": round(ctx.state.notional, 2),
            "leverage": ctx.state.leverage,
        },
        "geometry": {
            "mfe": round(ctx.geometry.mfe, 6),
            "mae": round(ctx.geometry.mae, 6),
            "drawdown_from_peak": round(ctx.geometry.drawdown_from_peak, 4),
            "profit_protection_ratio": round(ctx.geometry.profit_protection_ratio, 4),
            "momentum_decay": round(ctx.geometry.momentum_decay, 6),
        },
        "regime": {
            "label": ctx.regime.regime_label,
            "confidence": round(ctx.regime.regime_confidence, 4),
            "trend_alignment": round(ctx.regime.trend_alignment, 4),
            "reversal_risk": round(ctx.regime.reversal_risk, 4),
            "chop_risk": round(ctx.regime.chop_risk, 4),
        },
        "belief": {
            "exit_pressure": round(ctx.belief.exit_pressure, 4),
            "hold_conviction": round(ctx.belief.hold_conviction, 4),
            "directional_edge": round(ctx.belief.directional_edge, 4),
            "uncertainty": round(ctx.belief.uncertainty_total, 4),
        },
        "hazard": {
            "composite": round(ctx.hazard.composite_hazard, 4),
            "dominant": ctx.hazard.dominant_hazard,
            "drawdown": round(ctx.hazard.drawdown_hazard, 4),
            "reversal": round(ctx.hazard.reversal_hazard, 4),
            "volatility": round(ctx.hazard.volatility_hazard, 4),
            "time_decay": round(ctx.hazard.time_decay_hazard, 4),
            "regime": round(ctx.hazard.regime_hazard, 4),
            "ensemble": round(ctx.hazard.ensemble_hazard, 4),
        },
        "utility_ranked_actions": ranked_actions,
        "ensemble_recommendation": {
            "action": ensemble_action,
            "confidence": round(ctx.decision.decision_confidence, 4),
            "reason_codes": list(ctx.decision.reason_codes),
        },
    }

    return json.dumps(payload, separators=(",", ":"))
