"""Portfolio selection integration leveraging the ML agent outputs."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - runtime import for agent package
    from ai_engine.agents.tft_agent import TFTAgent
    from ai_engine.agents.xgb_agent import XGBAgent, make_default_agent
    
    def make_tft_agent():
        """Create TFT agent as primary AI model."""
        agent = TFTAgent()
        if agent.load_model():
            return agent
        return None
        
except ImportError:  # pragma: no cover - allow running without ai_engine package
    XGBAgent = Any  # type: ignore
    TFTAgent = Any  # type: ignore

    def make_default_agent():  # type: ignore
        return None
    
    def make_tft_agent():  # type: ignore
        return None

from backend.config import LiquidityConfig

if TYPE_CHECKING:  # pragma: no cover - typing-only circular guard
    from backend.services.liquidity import LiquidityRecord
else:  # pragma: no cover - runtime fallback avoids circular import
    LiquidityRecord = Any  # type: ignore

logger = logging.getLogger(__name__)


class SelectionSignal(Dict[str, Any]):
    """Typed alias for downstream typing; entries mirror agent output."""


async def score_symbols_with_agent(
    symbols: Sequence[str],
    *,
    agent: Optional[Any] = None,
    limit: int = 240,
    top_n: Optional[int] = None,
) -> Dict[str, SelectionSignal]:
    """Fetch OHLCV via the agent helper and return per-symbol signals.

    The agent already bundles sentiment/news augmentation, so we rely on
    `scan_top_by_volume_from_api` when available. When the agent is missing,
    we fall back to neutral HOLD signals to keep the pipeline operational.
    """

    if not symbols:
        return {}

    # Try TFT agent first, fallback to XGBoost if needed
    if agent is None:
        agent = make_tft_agent()
        if agent:
            logger.info("Using TFT (Temporal Fusion Transformer) agent for predictions")
        else:
            agent = make_default_agent()
            if agent:
                logger.info("TFT not available, using XGBoost agent as primary")
    
    if agent is None:
        logger.info("Selection agent unavailable; returning neutral signals")
        return {symbol: SelectionSignal(action="HOLD", score=0.0) for symbol in symbols}

    unique_symbols = list(dict.fromkeys(symbol.upper() for symbol in symbols))
    target = top_n or len(unique_symbols)
    try:
        results = await agent.scan_top_by_volume_from_api(  # type: ignore[attr-defined]
            unique_symbols,
            top_n=target,
            limit=limit,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Agent scan failed, defaulting to HOLD signals: %s", exc)
        return {symbol: SelectionSignal(action="HOLD", score=0.0) for symbol in symbols}
    
    # Check TFT signal diversity - fallback to XGBoost if all HOLD
    if isinstance(agent, TFTAgent) and isinstance(results, Mapping):
        hold_count = sum(1 for r in results.values() if isinstance(r, Mapping) and r.get("action") == "HOLD")
        if hold_count > len(results) * 0.8:  # >80% HOLD signals
            logger.warning(
                "TFT producing %d/%d HOLD signals (%.1f%%) - switching to XGBoost fallback",
                hold_count, len(results), (hold_count/len(results)*100)
            )
            xgb_agent = make_default_agent()
            if xgb_agent:
                logger.info("Using XGBoost agent for more diverse signals")
                try:
                    results = await xgb_agent.scan_top_by_volume_from_api(
                        unique_symbols,
                        top_n=target,
                        limit=limit,
                    )
                except Exception as exc:
                    logger.warning("XGBoost fallback failed: %s", exc)

    signals: Dict[str, SelectionSignal] = {}
    for symbol in unique_symbols:
        payload = results.get(symbol) if isinstance(results, Mapping) else None
        if not isinstance(payload, Mapping):
            payload = {"action": "HOLD", "score": 0.0}
        signals[symbol] = SelectionSignal(payload)
    return signals


def _extract_base(symbol: str, quotes: Iterable[str]) -> str:
    symbol_upper = symbol.upper()
    for quote in quotes:
        if symbol_upper.endswith(quote.upper()):
            return symbol_upper[: -len(quote)]
    return symbol_upper


def _extract_quote(symbol: str, quotes: Iterable[str]) -> str:
    symbol_upper = symbol.upper()
    for quote in quotes:
        upper_quote = quote.upper()
        if symbol_upper.endswith(upper_quote):
            return upper_quote
    return "UNKNOWN"


def blend_liquidity_and_model(
    selection: Sequence[LiquidityRecord],
    signals: Mapping[str, Mapping[str, Any]],
    config: LiquidityConfig,
) -> List[LiquidityRecord]:
    """Return a new ordered selection with liquidity+model scores blended.

    Each incoming record receives `model_action`, `model_score`, and
    `allocation_score` fields populated. Diversification is applied by limiting
    repeats per base symbol according to configuration. We keep the target size
    equal to the incoming selection length, satisfying `selection_min`.
    """

    if not selection:
        return []

    liquidity_scores = [max(item.aggregate_score, 0.0) for item in selection]
    max_liquidity = max(liquidity_scores) if liquidity_scores else 1.0
    if max_liquidity <= 0:
        max_liquidity = 1.0

    provisional: List[LiquidityRecord] = []
    buy_signals = sell_signals = hold_signals = 0
    
    for record in selection:
        signal = signals.get(record.symbol.upper()) or signals.get(record.symbol)
        action = str(signal.get("action", "HOLD")).upper() if signal else "HOLD"
        score = float(signal.get("score", 0.0)) if signal else 0.0
        confidence = float(signal.get("confidence", 0.5)) if signal else 0.5

        # Track signal distribution
        if action == "BUY":
            buy_signals += 1
        elif action == "SELL":
            sell_signals += 1
        else:
            hold_signals += 1

        liquidity_component = max(record.aggregate_score, 0.0) / max_liquidity

        # Weight model component by confidence
        model_component = max(score, 0.0) * confidence
        if action == "HOLD":
            model_component *= 0.5
        elif action == "SELL":
            penalty = -model_component
            if score >= config.model_sell_threshold:
                model_component = penalty
            else:
                model_component = penalty * 0.5

        combined = (
            config.liquidity_weight * liquidity_component
            + config.model_weight * model_component
        )
        if combined < 0:
            combined = 0.0

        updated = replace(
            record,
            model_action=action,
            model_score=score,
            allocation_score=combined,
        )
        provisional.append(updated)
    
    # Log AI signal summary
    logger.info(
        "AI signals: BUY=%d (%.1f%%), SELL=%d (%.1f%%), HOLD=%d (%.1f%%)",
        buy_signals, 100*buy_signals/len(selection) if selection else 0,
        sell_signals, 100*sell_signals/len(selection) if selection else 0,
        hold_signals, 100*hold_signals/len(selection) if selection else 0
    )

    sorted_records = sorted(
        provisional,
        key=lambda item: (item.allocation_score, item.aggregate_score),
        reverse=True,
    )

    target_size = max(config.selection_min, min(len(selection), config.selection_max))

    base_counts: defaultdict[str, int] = defaultdict(int)
    kept: List[LiquidityRecord] = []
    leftovers: List[LiquidityRecord] = []
    for record in sorted_records:
        base = _extract_base(record.symbol, config.stable_quote_assets)
        if config.max_per_base > 0 and base_counts[base] >= config.max_per_base:
            leftovers.append(record)
            continue
        base_counts[base] += 1
        kept.append(record)
        if len(kept) >= target_size:
            break

    if len(kept) < target_size:
        needed = target_size - len(kept)
        kept.extend(leftovers[:needed])

    # If diversification trimmed below incoming length while still above target,
    # ensure deterministic size by cutting any extras beyond target.
    return kept[:target_size]


def summarize_selection(
    selection: Sequence[LiquidityRecord],
    *,
    config: LiquidityConfig,
) -> Dict[str, Any]:
    """Produce summary analytics for the blended selection."""

    if not selection:
        return {
            "selection_size": 0,
            "action_breakdown": {},
            "base_distribution": {},
            "quote_distribution": {},
            "average_liquidity_score": 0.0,
            "average_allocation_score": 0.0,
            "top_allocations": [],
        }

    selection_size = len(selection)
    action_breakdown: defaultdict[str, int] = defaultdict(int)
    base_distribution: defaultdict[str, int] = defaultdict(int)
    quote_distribution: defaultdict[str, int] = defaultdict(int)
    total_liquidity = 0.0
    total_allocation = 0.0

    for record in selection:
        base = _extract_base(record.symbol, config.stable_quote_assets)
        base_distribution[base] += 1
        quote = _extract_quote(record.symbol, config.stable_quote_assets)
        quote_distribution[quote] += 1
        action = getattr(record, "model_action", "HOLD") or "HOLD"
        action_breakdown[action.upper()] += 1
        total_liquidity += max(record.aggregate_score, 0.0)
        allocation_component = getattr(record, "allocation_score", record.aggregate_score)
        total_allocation += max(allocation_component, 0.0)

    average_liquidity = total_liquidity / selection_size if selection_size else 0.0
    average_allocation = total_allocation / selection_size if selection_size else 0.0

    top_sorted = sorted(
        selection,
        key=lambda item: getattr(item, "allocation_score", item.aggregate_score),
        reverse=True,
    )[:5]
    TOP_DECIMALS = 6
    top_allocations = [
        {
            "symbol": item.symbol,
            "allocation_score": round(
                float(getattr(item, "allocation_score", item.aggregate_score)),
                TOP_DECIMALS,
            ),
            "model_action": getattr(item, "model_action", "HOLD"),
            "model_score": round(float(getattr(item, "model_score", 0.0)), TOP_DECIMALS),
        }
        for item in top_sorted
    ]

    return {
        "selection_size": selection_size,
        "action_breakdown": dict(action_breakdown),
        "base_distribution": dict(base_distribution),
        "quote_distribution": dict(quote_distribution),
        "average_liquidity_score": round(average_liquidity, 6),
        "average_allocation_score": round(average_allocation, 6),
        "top_allocations": top_allocations,
    }


__all__ = [
    "score_symbols_with_agent",
    "blend_liquidity_and_model",
    "SelectionSignal",
    "summarize_selection",
]
