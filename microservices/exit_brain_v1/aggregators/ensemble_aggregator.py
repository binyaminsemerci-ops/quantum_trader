"""
EnsembleAggregator — Pure math aggregation of model exit signals.

No Redis. No IO. No side effects.

Takes list[ModelExitSignal] → AggregatedExitSignal.
Fail-closed: returns None if < 2 usable signals.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from ..models.model_exit_signal import ModelExitSignal, VALID_MODELS
from ..models.aggregated_exit_signal import (
    AggregatedExitSignal,
    EnsembleDiagnostics,
    MIN_MODELS_REQUIRED,
)
from ..engines.normalization import clamp, renormalize_probabilities

logger = logging.getLogger(__name__)

# Thresholds
SOFT_STALE_SEC = 60     # Stale penalty kicks in
HARD_STALE_SEC = 300    # Model excluded entirely
TOTAL_MODEL_COUNT = 6   # Expected total models in ensemble
LOW_COVERAGE_THRESHOLD = 4  # Below this count, flag LOW_MODEL_COVERAGE
AGGREGATION_METHOD = "weighted_average_v1"


class EnsembleAggregator:
    """
    Aggregates multiple ModelExitSignal into one AggregatedExitSignal.

    Pure computation — no IO, no Redis, no side effects.
    """

    def __init__(
        self,
        total_model_count: int = TOTAL_MODEL_COUNT,
        soft_stale_sec: float = SOFT_STALE_SEC,
        hard_stale_sec: float = HARD_STALE_SEC,
    ) -> None:
        self._total_count = total_model_count
        self._soft_stale = soft_stale_sec
        self._hard_stale = hard_stale_sec

    # ── Public API ───────────────────────────────────────────────────────

    def aggregate(
        self,
        signals: List[ModelExitSignal],
        side: str,
        all_model_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[AggregatedExitSignal], EnsembleDiagnostics]:
        """
        Aggregate model signals into a single exit signal.

        Args:
            signals: ModelExitSignal list from adapter.
            side: Position side ("LONG" or "SHORT").
            all_model_names: All expected model names (for missing detection).

        Returns:
            Tuple of (AggregatedExitSignal or None, EnsembleDiagnostics).
            Returns None for signal if < 2 usable models.
        """
        if all_model_names is None:
            all_model_names = list(VALID_MODELS)

        now = time.time()

        # Step 1: Filter usable signals
        usable, stale_names, excluded = self.filter_usable_signals(signals)

        # Determine missing
        present_names = {s.model_name for s in signals}
        missing_names = [n for n in all_model_names if n not in present_names]

        # Per-model status
        per_model_status: Dict[str, str] = {}
        for name in all_model_names:
            if name in missing_names:
                per_model_status[name] = "MISSING"
            elif name in excluded:
                per_model_status[name] = "HARD_STALE"
            elif name in stale_names:
                per_model_status[name] = "STALE"
            else:
                per_model_status[name] = "OK"

        # Build diagnostics (always produced)
        diag = self._build_diagnostics(
            signals, usable, stale_names, missing_names, per_model_status, now,
        )

        # Fail-closed: not enough models
        if len(usable) < MIN_MODELS_REQUIRED:
            logger.warning(
                "[Aggregator] Only %d usable signals (need %d), returning None",
                len(usable), MIN_MODELS_REQUIRED,
            )
            return None, diag

        # Step 2: Compute reliability weights
        weights = self.compute_reliability_weights(usable)

        # Step 3: Aggregate probabilities
        sell_agg, hold_agg, buy_agg = self.aggregate_probabilities(usable, weights)

        # Step 4: Derive continuation/reversal
        if side == "LONG":
            continuation_agg = hold_agg + buy_agg
            reversal_agg = sell_agg
        else:
            continuation_agg = hold_agg + sell_agg
            reversal_agg = buy_agg

        # Step 5: Aggregate expected values
        upside_agg, downside_agg, drawdown_agg = self.aggregate_expected_values(
            usable, weights,
        )

        # Step 6: Confidence
        confidence_agg = self._weighted_mean(
            [s.confidence for s in usable], weights,
        )
        uncertainty_agg = 1.0 - confidence_agg

        # Step 7: Disagreement
        disagreement = self.compute_disagreement_score(usable)

        # Step 8: Consensus
        consensus = self.compute_consensus_strength(disagreement)

        # Step 9: Reliability score
        mean_freshness_weight = (
            sum(self._freshness_weight(s.freshness_seconds) for s in usable) / len(usable)
        )
        reliability = (
            (len(usable) / self._total_count)
            * mean_freshness_weight
            * (1.0 - disagreement * 0.5)
        )

        # Quality flags
        quality_flags: List[str] = []
        if len(usable) < LOW_COVERAGE_THRESHOLD:
            quality_flags.append("LOW_MODEL_COVERAGE")
        if stale_names:
            quality_flags.append(f"STALE_MODELS:{','.join(stale_names)}")
        if missing_names:
            quality_flags.append(f"MISSING_MODELS:{','.join(missing_names)}")
        if disagreement > 0.3:
            quality_flags.append("HIGH_DISAGREEMENT")

        agg_signal = AggregatedExitSignal(
            position_id=usable[0].position_id,
            symbol=usable[0].symbol,
            ensemble_timestamp=now,
            participating_models=[s.model_name for s in usable],
            missing_models=missing_names,
            stale_models=stale_names,
            model_count_total=self._total_count,
            model_count_used=len(usable),
            sell_probability_agg=clamp(sell_agg),
            hold_probability_agg=clamp(hold_agg),
            buy_probability_agg=clamp(buy_agg),
            continuation_probability_agg=clamp(continuation_agg),
            reversal_probability_agg=clamp(reversal_agg),
            expected_upside_remaining_agg=upside_agg,
            expected_downside_if_hold_agg=downside_agg,
            expected_drawdown_risk_agg=clamp(drawdown_agg),
            confidence_agg=clamp(confidence_agg),
            uncertainty_agg=clamp(uncertainty_agg),
            disagreement_score=clamp(disagreement),
            consensus_strength=clamp(consensus),
            reliability_score=clamp(reliability),
            aggregation_method=AGGREGATION_METHOD,
            quality_flags=quality_flags,
            shadow_only=True,
        )

        errors = agg_signal.validate()
        if errors:
            logger.error("[Aggregator] Validation failed: %s", errors)
            return None, diag

        # Update diagnostics with final weights
        diag.per_model_weights = {
            usable[i].model_name: weights[i] for i in range(len(usable))
        }

        return agg_signal, diag

    # ── Filter ───────────────────────────────────────────────────────────

    def filter_usable_signals(
        self,
        signals: List[ModelExitSignal],
    ) -> Tuple[List[ModelExitSignal], List[str], set]:
        """
        Filter signals by staleness and validity.

        Returns:
            (usable_signals, stale_model_names, excluded_model_names)
        """
        usable: List[ModelExitSignal] = []
        stale_names: List[str] = []
        excluded: set = set()

        for s in signals:
            if s.freshness_seconds > self._hard_stale:
                excluded.add(s.model_name)
                logger.info("[Aggregator] Excluding %s: hard stale (%.0fs)", s.model_name, s.freshness_seconds)
                continue

            if s.freshness_seconds > self._soft_stale:
                stale_names.append(s.model_name)

            # Validate
            errors = s.validate()
            if errors:
                excluded.add(s.model_name)
                logger.warning("[Aggregator] Excluding %s: validation errors: %s", s.model_name, errors)
                continue

            usable.append(s)

        return usable, stale_names, excluded

    # ── Weights ──────────────────────────────────────────────────────────

    def compute_reliability_weights(
        self,
        signals: List[ModelExitSignal],
    ) -> List[float]:
        """
        Compute normalised reliability weights for usable signals.

        W_final(m) = W_base * W_freshness(m) * W_confidence(m)
        Then normalised to sum=1.

        Returns:
            List of weights, same order as input signals.
        """
        n = len(signals)
        if n == 0:
            return []

        base_weight = 1.0 / n

        raw_weights: List[float] = []
        for s in signals:
            w_fresh = self._freshness_weight(s.freshness_seconds)
            w_conf = clamp(s.confidence, 0.1, 1.0)
            raw_weights.append(base_weight * w_fresh * w_conf)

        total = sum(raw_weights)
        if total <= 0:
            return [1.0 / n] * n
        return [w / total for w in raw_weights]

    def _freshness_weight(self, freshness_seconds: float) -> float:
        """Compute freshness weight. 1.0 if fresh, decays linearly, min 0.1."""
        if freshness_seconds <= 30:
            return 1.0
        if freshness_seconds > self._hard_stale:
            return 0.0
        # Linear decay from 30s to hard_stale
        decay = (freshness_seconds - 30) / (self._hard_stale - 30)
        weight = max(0.1, 1.0 - decay)
        # Extra penalty for soft stale
        if freshness_seconds > self._soft_stale:
            weight *= 0.5
        return weight

    # ── Probability aggregation ──────────────────────────────────────────

    def aggregate_probabilities(
        self,
        signals: List[ModelExitSignal],
        weights: List[float],
    ) -> Tuple[float, float, float]:
        """
        Weighted average of [sell, hold, buy] probabilities.

        Returns renormalized (sell_agg, hold_agg, buy_agg).
        """
        sell_agg = sum(w * s.sell_probability for w, s in zip(weights, signals))
        hold_agg = sum(w * s.hold_probability for w, s in zip(weights, signals))
        buy_agg = sum(w * s.buy_probability for w, s in zip(weights, signals))

        # Renormalize
        sell_agg, hold_agg, buy_agg = renormalize_probabilities(
            [sell_agg, hold_agg, buy_agg]
        )
        return sell_agg, hold_agg, buy_agg

    # ── Expected value aggregation ───────────────────────────────────────

    def aggregate_expected_values(
        self,
        signals: List[ModelExitSignal],
        weights: List[float],
    ) -> Tuple[float, float, float]:
        """
        Weighted average of expected upside, downside, drawdown risk.

        Only considers signals where the value is non-zero (populated).
        """
        upside = self._weighted_mean_nonzero(
            [s.expected_upside_remaining for s in signals], weights,
        )
        downside = self._weighted_mean_nonzero(
            [s.expected_downside_if_hold for s in signals], weights,
        )
        drawdown = self._weighted_mean_nonzero(
            [s.expected_drawdown_risk for s in signals], weights,
        )
        return upside, downside, drawdown

    # ── Disagreement ─────────────────────────────────────────────────────

    def compute_disagreement_score(
        self,
        signals: List[ModelExitSignal],
    ) -> float:
        """
        Mean standard deviation of class probabilities across models.

        0.0 = perfect agreement, ~0.47 = maximum disagreement.
        Normalised to [0, 1] by dividing by 0.5.
        """
        if len(signals) < 2:
            return 0.0

        sells = [s.sell_probability for s in signals]
        holds = [s.hold_probability for s in signals]
        buys = [s.buy_probability for s in signals]

        std_sell = _std(sells)
        std_hold = _std(holds)
        std_buy = _std(buys)

        raw_disagreement = (std_sell + std_hold + std_buy) / 3.0
        return clamp(raw_disagreement / 0.5)

    # ── Consensus ────────────────────────────────────────────────────────

    def compute_consensus_strength(self, disagreement_score: float) -> float:
        """Consensus = 1 - disagreement."""
        return clamp(1.0 - disagreement_score)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_mean(values: List[float], weights: List[float]) -> float:
        if not values or not weights:
            return 0.0
        return sum(v * w for v, w in zip(values, weights))

    @staticmethod
    def _weighted_mean_nonzero(
        values: List[float], weights: List[float],
    ) -> float:
        """Weighted mean considering only non-zero values."""
        pairs = [(v, w) for v, w in zip(values, weights) if v != 0.0]
        if not pairs:
            return 0.0
        total_w = sum(w for _, w in pairs)
        if total_w <= 0:
            return 0.0
        return sum(v * w for v, w in pairs) / total_w

    # ── Diagnostics ──────────────────────────────────────────────────────

    def _build_diagnostics(
        self,
        all_signals: List[ModelExitSignal],
        usable: List[ModelExitSignal],
        stale_names: List[str],
        missing_names: List[str],
        per_model_status: Dict[str, str],
        timestamp: float,
    ) -> EnsembleDiagnostics:
        """Build diagnostics object for the aggregation cycle."""
        per_model_signals = {s.model_name: s.to_dict() for s in all_signals}
        per_model_freshness = {s.model_name: s.freshness_seconds for s in all_signals}

        # Probability spread (std per class)
        if len(usable) >= 2:
            sells = [s.sell_probability for s in usable]
            holds = [s.hold_probability for s in usable]
            buys = [s.buy_probability for s in usable]
            prob_spread = {
                "sell_std": _std(sells),
                "hold_std": _std(holds),
                "buy_std": _std(buys),
            }
        else:
            prob_spread = {"sell_std": 0.0, "hold_std": 0.0, "buy_std": 0.0}

        # Max disagreement pair
        max_pair = self._find_max_disagreement_pair(usable)

        position_id = all_signals[0].position_id if all_signals else ""
        symbol = all_signals[0].symbol if all_signals else ""

        return EnsembleDiagnostics(
            position_id=position_id,
            symbol=symbol,
            timestamp=timestamp,
            per_model_signals=per_model_signals,
            per_model_weights={},  # Updated after weight computation
            per_model_freshness=per_model_freshness,
            per_model_status=per_model_status,
            probability_spread=prob_spread,
            max_disagreement_pair=max_pair,
            aggregation_config={
                "method": AGGREGATION_METHOD,
                "soft_stale_sec": self._soft_stale,
                "hard_stale_sec": self._hard_stale,
                "min_models_required": MIN_MODELS_REQUIRED,
                "total_model_count": self._total_count,
            },
        )

    @staticmethod
    def _find_max_disagreement_pair(signals: List[ModelExitSignal]) -> str:
        """Find the pair of models with largest probability distance."""
        if len(signals) < 2:
            return ""

        max_dist = -1.0
        pair = ""
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                a, b = signals[i], signals[j]
                dist = (
                    abs(a.sell_probability - b.sell_probability)
                    + abs(a.hold_probability - b.hold_probability)
                    + abs(a.buy_probability - b.buy_probability)
                )
                if dist > max_dist:
                    max_dist = dist
                    pair = f"{a.model_name} vs {b.model_name}"
        return pair

    # ── Shadow publishing helper ─────────────────────────────────────────

    def publish_aggregated_shadow_signal(
        self,
        agg_signal: AggregatedExitSignal,
        diag: EnsembleDiagnostics,
        shadow_publisher,
    ) -> Tuple[bool, bool]:
        """
        Publish aggregated signal and diagnostics to shadow streams.

        Args:
            agg_signal: Aggregated exit signal.
            diag: Diagnostics for this cycle.
            shadow_publisher: ShadowPublisher instance.

        Returns:
            Tuple of (agg_published, diag_published).
        """
        agg_stream = "quantum:stream:exit.ensemble.agg.shadow"
        diag_stream = "quantum:stream:exit.ensemble.diag.shadow"

        agg_flat = {k: ("" if v is None else str(v)) for k, v in agg_signal.to_dict().items()}
        agg_ok = shadow_publisher._xadd(agg_stream, agg_flat) is not None

        diag_flat = {k: ("" if v is None else str(v)) for k, v in diag.to_dict().items()}
        diag_ok = shadow_publisher._xadd(diag_stream, diag_flat) is not None

        return agg_ok, diag_ok


# ── Module-level pure functions ──────────────────────────────────────────

def _std(values: List[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
