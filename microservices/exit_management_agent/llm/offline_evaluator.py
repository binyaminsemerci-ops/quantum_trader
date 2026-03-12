"""PATCH-11 — Offline evaluator for replay and post-close analysis ONLY.

STRUCTURAL ISOLATION: This module must NEVER be imported or called
from the live tick path (main.py). It is designed exclusively for:
  - Closed-trade obituary analysis (replay_obituary_writer)
  - Post-close "should we have exited earlier?" evaluation
  - Formula vs Qwen3 vs fallback comparison
  - Threshold tuning recommendations
  - Weekly deep audit reports

Models:
  - Primary evaluator: llama-3.3-70b-versatile (production, Groq)
  - Heavy evaluator:   openai/gpt-oss-120b (optional, rare deep audits)

This module does NOT have access to:
  - Live Redis connections
  - Position source
  - Ensemble bridge
  - Any live tick infrastructure
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .groq_client import GroqModelClient

_log = logging.getLogger("exit_management_agent.llm.offline_evaluator")

_EVALUATOR_SYSTEM_PROMPT = """\
You are an offline trade evaluation analyst for a quantitative cryptocurrency trading system.

You receive a CLOSED trade record including:
1. Trade details: symbol, side, entry/exit price, PnL, hold duration
2. Decision history: what actions were taken and when
3. Market regime at entry and exit
4. Ensemble model predictions at key points
5. LLM judge decisions (if available)

Your job: Evaluate the trade execution and provide actionable feedback.

You MUST respond with ONLY a JSON object:
{
  "overall_grade": "B+",
  "exit_timing": "LATE",
  "missed_opportunity_pct": 12.5,
  "should_have_action": "HARVEST_70_KEEP_30",
  "should_have_timestamp_offset_sec": -600,
  "key_learnings": ["Thesis decay signal was early enough", "Giveback risk was underweighted"],
  "threshold_suggestions": {
    "exit_pressure_threshold": 0.45,
    "giveback_hazard_weight": 0.25
  },
  "confidence": 0.72
}

Valid exit_timing values: EARLY, OPTIMAL, LATE, MISSED_EXIT, FORCED_EXIT
Valid overall_grade values: A+, A, A-, B+, B, B-, C+, C, C-, D, F
"""


@dataclass(frozen=True)
class OfflineEvaluation:
    """Result of offline trade evaluation."""
    symbol: str
    trade_id: str
    overall_grade: str
    exit_timing: str
    missed_opportunity_pct: float
    should_have_action: str
    key_learnings: List[str]
    threshold_suggestions: Dict[str, float]
    confidence: float
    raw: str
    evaluator_model: str
    success: bool
    error: Optional[str] = None


class OfflineEvaluator:
    """
    Offline-only trade evaluator. NOT for live tick path.

    Uses Llama 3.3 70B (primary) or GPT-OSS 120B (heavy) via Groq.

    IMPORTANT: This class must NEVER be instantiated in ExitManagementAgent.__init__
    or any code path reachable from _tick(). It is designed for batch/offline use only.
    """

    def __init__(
        self,
        evaluator_client: GroqModelClient,
        heavy_client: Optional[GroqModelClient] = None,
    ) -> None:
        self._evaluator = evaluator_client
        self._heavy = heavy_client

    async def evaluate_closed_trade(
        self,
        trade_record: Dict[str, Any],
        use_heavy: bool = False,
    ) -> OfflineEvaluation:
        """
        Evaluate a single closed trade.

        Args:
            trade_record: Dict with trade details, decision history, market state.
            use_heavy: If True and heavy client available, use GPT-OSS 120B.

        Returns:
            OfflineEvaluation. Never raises.
        """
        client = self._heavy if (use_heavy and self._heavy) else self._evaluator
        symbol = trade_record.get("symbol", "UNKNOWN")
        trade_id = trade_record.get("trade_id", "UNKNOWN")

        try:
            user_content = json.dumps(trade_record, separators=(",", ":"))
            raw = await client.chat(_EVALUATOR_SYSTEM_PROMPT, user_content)
            parsed = json.loads(raw.strip())

            return OfflineEvaluation(
                symbol=symbol,
                trade_id=trade_id,
                overall_grade=str(parsed.get("overall_grade", "?")),
                exit_timing=str(parsed.get("exit_timing", "UNKNOWN")),
                missed_opportunity_pct=float(parsed.get("missed_opportunity_pct", 0.0)),
                should_have_action=str(parsed.get("should_have_action", "")),
                key_learnings=list(parsed.get("key_learnings", [])),
                threshold_suggestions=dict(parsed.get("threshold_suggestions", {})),
                confidence=float(parsed.get("confidence", 0.0)),
                raw=raw[:2000],
                evaluator_model=client.model,
                success=True,
            )
        except Exception as exc:
            _log.warning(
                "[OfflineEvaluator] %s evaluation failed: %s", symbol, exc,
            )
            return OfflineEvaluation(
                symbol=symbol,
                trade_id=trade_id,
                overall_grade="?",
                exit_timing="UNKNOWN",
                missed_opportunity_pct=0.0,
                should_have_action="",
                key_learnings=[],
                threshold_suggestions={},
                confidence=0.0,
                raw="",
                evaluator_model=client.model,
                success=False,
                error=str(exc),
            )
