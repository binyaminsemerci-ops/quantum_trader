"""
Signal Quality Filter - P1-02
==============================

Filters noisy signals in high volatility conditions by requiring:
1. Minimum model agreement (≥3/4 models for same direction)
2. Minimum collective confidence in HIGH_VOL regime
3. Noise detection and tagging

Author: Quantum Trader AI Team
Date: December 3, 2025
Version: 1.0
"""

import logging
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Individual model prediction."""
    model_id: str
    direction: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    model_type: str  # "XGBoost", "LightGBM", "NHiTS", "Ensemble"


@dataclass
class FilterResult:
    """Signal quality filter result."""
    passed: bool
    reason: str
    is_noisy: bool
    model_agreement_pct: float
    collective_confidence: float
    volatility_regime: str
    recommended_action: Literal["ACCEPT", "REJECT", "REDUCE_SIZE"]


class SignalQualityFilter:
    """
    Signal quality filter for high volatility conditions (P1-02).
    
    Prevents trading on noisy/low-quality signals by requiring:
    - Model consensus in directional bias
    - Sufficient confidence levels
    - Volatility-adjusted thresholds
    """
    
    def __init__(
        self,
        min_model_agreement: float = 0.75,  # 75% = 3/4 models
        min_confidence_normal: float = 0.45,  # Normal volatility threshold
        min_confidence_high_vol: float = 0.65,  # High volatility threshold
        high_vol_threshold: float = 0.03,  # 3% ATR = high volatility
    ):
        """
        Initialize signal quality filter.
        
        Args:
            min_model_agreement: Minimum % of models agreeing (0.75 = 3/4 models)
            min_confidence_normal: Min confidence in normal volatility
            min_confidence_high_vol: Min confidence in high volatility
            high_vol_threshold: ATR % threshold for high volatility
        """
        self.min_model_agreement = min_model_agreement
        self.min_confidence_normal = min_confidence_normal
        self.min_confidence_high_vol = min_confidence_high_vol
        self.high_vol_threshold = high_vol_threshold
        
        logger.info(
            f"[P1-02] SignalQualityFilter initialized:\n"
            f"  Min model agreement: {min_model_agreement:.0%} (≥{int(min_model_agreement * 4)}/4 models)\n"
            f"  Min confidence (NORMAL): {min_confidence_normal:.0%}\n"
            f"  Min confidence (HIGH_VOL): {min_confidence_high_vol:.0%}\n"
            f"  High volatility threshold: {high_vol_threshold:.1%} ATR"
        )
    
    def filter_signal(
        self,
        symbol: str,
        model_predictions: List[ModelPrediction],
        atr_pct: float,
        metadata: Optional[Dict] = None
    ) -> FilterResult:
        """
        Filter trading signal based on model agreement and volatility.
        
        Args:
            symbol: Trading pair
            model_predictions: List of predictions from different models
            atr_pct: Current ATR as % (e.g., 0.03 = 3%)
            metadata: Additional context
            
        Returns:
            FilterResult with pass/fail decision and reasoning
        """
        # Determine volatility regime
        is_high_vol = atr_pct >= self.high_vol_threshold
        vol_regime = "HIGH_VOL" if is_high_vol else "NORMAL"
        
        # Get confidence threshold for current regime
        min_confidence = (
            self.min_confidence_high_vol if is_high_vol 
            else self.min_confidence_normal
        )
        
        # Count model directions
        buy_votes = sum(1 for p in model_predictions if p.direction == "BUY")
        sell_votes = sum(1 for p in model_predictions if p.direction == "SELL")
        hold_votes = sum(1 for p in model_predictions if p.direction == "HOLD")
        total_votes = len(model_predictions)
        
        # Determine consensus direction
        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus_direction = "BUY"
            consensus_votes = buy_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus_direction = "SELL"
            consensus_votes = sell_votes
        else:
            consensus_direction = "HOLD"
            consensus_votes = hold_votes
        
        # Calculate model agreement percentage
        model_agreement_pct = consensus_votes / total_votes if total_votes > 0 else 0.0
        
        # Calculate collective confidence (average of agreeing models)
        agreeing_models = [
            p for p in model_predictions 
            if p.direction == consensus_direction
        ]
        collective_confidence = (
            sum(p.confidence for p in agreeing_models) / len(agreeing_models)
            if agreeing_models else 0.0
        )
        
        # Check 1: Model agreement
        if model_agreement_pct < self.min_model_agreement:
            return FilterResult(
                passed=False,
                reason=f"Insufficient model agreement: {model_agreement_pct:.0%} < {self.min_model_agreement:.0%} "
                       f"({consensus_votes}/{total_votes} models agree on {consensus_direction})",
                is_noisy=True,
                model_agreement_pct=model_agreement_pct,
                collective_confidence=collective_confidence,
                volatility_regime=vol_regime,
                recommended_action="REJECT"
            )
        
        # Check 2: Collective confidence
        if collective_confidence < min_confidence:
            return FilterResult(
                passed=False,
                reason=f"Insufficient confidence in {vol_regime}: {collective_confidence:.1%} < {min_confidence:.1%}",
                is_noisy=True,
                model_agreement_pct=model_agreement_pct,
                collective_confidence=collective_confidence,
                volatility_regime=vol_regime,
                recommended_action="REJECT"
            )
        
        # Check 3: HIGH_VOL extra scrutiny
        if is_high_vol:
            # In high volatility, require even stricter criteria
            if consensus_direction == "HOLD":
                return FilterResult(
                    passed=False,
                    reason=f"HIGH_VOL consensus is HOLD - avoid trading in volatile conditions",
                    is_noisy=True,
                    model_agreement_pct=model_agreement_pct,
                    collective_confidence=collective_confidence,
                    volatility_regime=vol_regime,
                    recommended_action="REJECT"
                )
            
            # Check if confidence is borderline
            if collective_confidence < (min_confidence + 0.10):
                # Allow trade but suggest size reduction
                logger.warning(
                    f"[P1-02] {symbol} HIGH_VOL borderline confidence: {collective_confidence:.1%} "
                    f"(min: {min_confidence:.1%}) - suggesting REDUCE_SIZE"
                )
                return FilterResult(
                    passed=True,
                    reason=f"HIGH_VOL borderline - reduce position size",
                    is_noisy=False,
                    model_agreement_pct=model_agreement_pct,
                    collective_confidence=collective_confidence,
                    volatility_regime=vol_regime,
                    recommended_action="REDUCE_SIZE"
                )
        
        # All checks passed
        logger.info(
            f"[P1-02] ✅ {symbol} signal PASSED quality filter:\n"
            f"  Regime: {vol_regime} (ATR: {atr_pct:.2%})\n"
            f"  Agreement: {model_agreement_pct:.0%} ({consensus_votes}/{total_votes} → {consensus_direction})\n"
            f"  Confidence: {collective_confidence:.1%} (min: {min_confidence:.1%})"
        )
        
        return FilterResult(
            passed=True,
            reason=f"Signal meets quality criteria: {consensus_votes}/{total_votes} models agree with {collective_confidence:.1%} confidence",
            is_noisy=False,
            model_agreement_pct=model_agreement_pct,
            collective_confidence=collective_confidence,
            volatility_regime=vol_regime,
            recommended_action="ACCEPT"
        )
    
    def tag_noisy_signal(
        self,
        symbol: str,
        model_predictions: List[ModelPrediction],
        atr_pct: float
    ) -> Tuple[bool, str]:
        """
        Quick check if signal is noisy without full filtering.
        
        Args:
            symbol: Trading pair
            model_predictions: Model predictions
            atr_pct: Current ATR %
            
        Returns:
            (is_noisy, reason) tuple
        """
        # Calculate agreement
        directions = [p.direction for p in model_predictions]
        if not directions:
            return (True, "No model predictions")
        
        # Count most common direction
        from collections import Counter
        direction_counts = Counter(directions)
        most_common_count = direction_counts.most_common(1)[0][1]
        agreement_pct = most_common_count / len(directions)
        
        # Check if noisy
        if agreement_pct < self.min_model_agreement:
            return (
                True,
                f"Low agreement: {agreement_pct:.0%} < {self.min_model_agreement:.0%}"
            )
        
        # Check confidence
        avg_confidence = sum(p.confidence for p in model_predictions) / len(model_predictions)
        is_high_vol = atr_pct >= self.high_vol_threshold
        min_conf = self.min_confidence_high_vol if is_high_vol else self.min_confidence_normal
        
        if avg_confidence < min_conf:
            return (
                True,
                f"Low confidence: {avg_confidence:.1%} < {min_conf:.1%}"
            )
        
        return (False, "Signal not noisy")
    
    def adjust_position_size_for_quality(
        self,
        base_size_usd: float,
        filter_result: FilterResult,
    ) -> float:
        """
        Adjust position size based on signal quality.
        
        Args:
            base_size_usd: Original position size
            filter_result: Quality filter result
            
        Returns:
            Adjusted position size
        """
        if not filter_result.passed:
            return 0.0  # Rejected - no position
        
        if filter_result.recommended_action == "REDUCE_SIZE":
            # Reduce size by 50% for borderline signals
            adjusted_size = base_size_usd * 0.5
            logger.info(
                f"[P1-02] Position size reduced: ${base_size_usd:.2f} → ${adjusted_size:.2f} "
                f"(reason: {filter_result.reason})"
            )
            return adjusted_size
        
        # Accept full size
        return base_size_usd


# Singleton instance for easy access
_filter_instance: Optional[SignalQualityFilter] = None


def get_signal_quality_filter() -> SignalQualityFilter:
    """Get singleton signal quality filter instance."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = SignalQualityFilter()
    return _filter_instance


def reset_signal_quality_filter():
    """Reset singleton (for testing)."""
    global _filter_instance
    _filter_instance = None
