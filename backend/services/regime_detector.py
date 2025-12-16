"""Market Regime Detector - Deterministic regime classification."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def _parse_float(value: str | None, *, default: float) -> float:
    """Parse float from environment variable."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class RegimeType(str, Enum):
    """Market regime classifications."""
    TRENDING = "TRENDING"           # Strong directional movement
    RANGING = "RANGING"             # Choppy, sideways movement
    LOW_VOL = "LOW_VOL"            # Abnormally low volatility
    NORMAL_VOL = "NORMAL_VOL"      # Normal volatility
    HIGH_VOL = "HIGH_VOL"          # Elevated volatility
    EXTREME_VOL = "EXTREME_VOL"    # Extreme volatility / crisis mode


@dataclass
class RegimeConfig:
    """Configuration for regime detection thresholds."""
    
    # ATR/Price ratio thresholds (volatility classification)
    atr_ratio_low: float = 0.005        # < 0.5% = LOW_VOL
    atr_ratio_normal: float = 0.015     # 0.5-1.5% = NORMAL_VOL
    atr_ratio_high: float = 0.03        # 1.5-3% = HIGH_VOL
    # > 3% = EXTREME_VOL
    
    # ADX thresholds (trend strength)
    adx_trending: float = 25.0          # ADX > 25 = trending
    adx_strong_trend: float = 40.0      # ADX > 40 = strong trend
    
    # Range width threshold (% of price)
    range_width_threshold: float = 0.02  # 2% range = ranging market
    
    # EMA alignment threshold (trend confirmation)
    ema_alignment_pct: float = 0.02     # Price > 2% above EMA200 = uptrend
    
    @classmethod
    def from_env(cls) -> RegimeConfig:
        """Load configuration from environment variables."""
        return cls(
            atr_ratio_low=_parse_float(
                os.getenv("REGIME_ATR_RATIO_LOW"),
                default=0.005
            ),
            atr_ratio_normal=_parse_float(
                os.getenv("REGIME_ATR_RATIO_NORMAL"),
                default=0.015
            ),
            atr_ratio_high=_parse_float(
                os.getenv("REGIME_ATR_RATIO_HIGH"),
                default=0.03
            ),
            adx_trending=_parse_float(
                os.getenv("REGIME_ADX_TRENDING"),
                default=25.0
            ),
            adx_strong_trend=_parse_float(
                os.getenv("REGIME_ADX_STRONG_TREND"),
                default=40.0
            ),
            range_width_threshold=_parse_float(
                os.getenv("REGIME_RANGE_WIDTH_THRESHOLD"),
                default=0.02
            ),
            ema_alignment_pct=_parse_float(
                os.getenv("REGIME_EMA_ALIGNMENT_PCT"),
                default=0.02
            ),
        )


@dataclass
class RegimeIndicators:
    """Market indicators for regime detection."""
    price: float
    atr: float
    ema_200: Optional[float] = None
    adx: Optional[float] = None
    range_high: Optional[float] = None  # Recent high
    range_low: Optional[float] = None   # Recent low


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: RegimeType
    volatility_regime: RegimeType  # LOW_VOL, NORMAL_VOL, HIGH_VOL, EXTREME_VOL
    trend_regime: str              # TRENDING, RANGING, UNKNOWN
    atr_ratio: float               # ATR / Price
    details: Dict[str, any]        # Additional metrics for logging


class RegimeDetector:
    """
    Detect market regime based on price action, volatility, and trend indicators.
    
    Uses deterministic rules with configurable thresholds to classify markets into:
    - Volatility regimes: LOW_VOL, NORMAL_VOL, HIGH_VOL, EXTREME_VOL
    - Trend regimes: TRENDING, RANGING
    
    The primary regime returned is the volatility classification, with trend info
    available in the details for consumption by other components.
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None, event_bus = None):
        """
        Initialize regime detector.
        
        Args:
            config: Regime detection configuration. If None, loads from environment.
            event_bus: EventBus for publishing regime.changed events (optional)
        """
        self.config = config or RegimeConfig.from_env()
        self.event_bus = event_bus
        
        # Track previous regime to detect changes
        self.previous_regime: Optional[RegimeType] = None
        self.previous_trend_regime: Optional[str] = None
        
        logger.info(
            f"[OK] RegimeDetector initialized: "
            f"ATR ratios ({self.config.atr_ratio_low:.3f}, "
            f"{self.config.atr_ratio_normal:.3f}, "
            f"{self.config.atr_ratio_high:.3f}), "
            f"ADX trending={self.config.adx_trending}"
        )
        
        if self.event_bus:
            logger.info("[REGIME] EventBus integration enabled - regime.changed events will be published")
    
    def detect_regime(self, indicators: RegimeIndicators) -> RegimeResult:
        """
        Detect current market regime (synchronous version).
        
        Note: This version does NOT publish events. Use detect_regime_async() for event publication.
        
        Args:
            indicators: Market indicators (price, ATR, EMA, ADX, range)
        
        Returns:
            RegimeResult with regime classification and details
        """
        # Calculate ATR ratio (primary volatility measure)
        atr_ratio = indicators.atr / indicators.price if indicators.price > 0 else 0.0
        
        # Classify volatility regime
        volatility_regime = self._classify_volatility(atr_ratio)
        
        # Classify trend regime
        trend_regime = self._classify_trend(indicators)
        
        # Determine primary regime (use volatility as primary classifier)
        primary_regime = volatility_regime
        
        # Build details dictionary
        details = {
            "atr_ratio": atr_ratio,
            "atr_ratio_pct": atr_ratio * 100,
            "volatility_regime": volatility_regime.value,
            "trend_regime": trend_regime,
            "price": indicators.price,
            "atr": indicators.atr,
        }
        
        # Add optional indicators to details
        if indicators.ema_200 is not None:
            ema_distance_pct = (indicators.price - indicators.ema_200) / indicators.ema_200
            details["ema_200"] = indicators.ema_200
            details["ema_distance_pct"] = ema_distance_pct * 100
        
        if indicators.adx is not None:
            details["adx"] = indicators.adx
        
        if indicators.range_high is not None and indicators.range_low is not None:
            range_width = (indicators.range_high - indicators.range_low) / indicators.price
            details["range_width_pct"] = range_width * 100
        
        logger.debug(
            f"Regime detected: {primary_regime.value} "
            f"(ATR ratio: {atr_ratio:.4f}, Trend: {trend_regime})"
        )
        
        return RegimeResult(
            regime=primary_regime,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            atr_ratio=atr_ratio,
            details=details
        )
    
    async def detect_regime_async(self, indicators: RegimeIndicators, symbol: str = "GLOBAL") -> RegimeResult:
        """
        Detect current market regime and publish regime.changed event if changed.
        
        Args:
            indicators: Market indicators (price, ATR, EMA, ADX, range)
            symbol: Symbol being analyzed (for event context)
        
        Returns:
            RegimeResult with regime classification and details
        """
        from datetime import datetime, timezone
        
        # Perform detection
        result = self.detect_regime(indicators)
        
        # Check if regime changed
        regime_changed = False
        change_type = None
        
        if self.previous_regime is not None:
            # Check for volatility regime change
            if result.volatility_regime != self.previous_regime:
                regime_changed = True
                change_type = "VOLATILITY"
                logger.info(
                    f"[REGIME] 🔄 Volatility regime changed: "
                    f"{self.previous_regime.value} → {result.volatility_regime.value}"
                )
            
            # Check for trend regime change
            if result.trend_regime != self.previous_trend_regime:
                if not regime_changed:
                    regime_changed = True
                    change_type = "TREND"
                else:
                    change_type = "BOTH"  # Both volatility and trend changed
                
                logger.info(
                    f"[REGIME] 🔄 Trend regime changed: "
                    f"{self.previous_trend_regime} → {result.trend_regime}"
                )
        
        # Publish regime.changed event if changed and EventBus available
        if regime_changed and self.event_bus:
            try:
                await self.event_bus.publish("regime.changed", {
                    "symbol": symbol,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "change_type": change_type,
                    "old_volatility_regime": self.previous_regime.value if self.previous_regime else None,
                    "new_volatility_regime": result.volatility_regime.value,
                    "old_trend_regime": self.previous_trend_regime,
                    "new_trend_regime": result.trend_regime,
                    "atr_ratio": result.atr_ratio,
                    "details": result.details
                })
                logger.info(
                    f"[REGIME] ✅ Published regime.changed event: "
                    f"{change_type} change for {symbol}"
                )
            except Exception as e:
                logger.error(f"[REGIME] Failed to publish regime.changed event: {e}")
        
        # Update previous regime state
        self.previous_regime = result.volatility_regime
        self.previous_trend_regime = result.trend_regime
        
        return result
    
    def _classify_volatility(self, atr_ratio: float) -> RegimeType:
        """
        Classify volatility regime based on ATR/Price ratio.
        
        Args:
            atr_ratio: ATR divided by price
        
        Returns:
            Volatility regime classification
        """
        if atr_ratio < self.config.atr_ratio_low:
            return RegimeType.LOW_VOL
        elif atr_ratio < self.config.atr_ratio_normal:
            return RegimeType.NORMAL_VOL
        elif atr_ratio < self.config.atr_ratio_high:
            return RegimeType.HIGH_VOL
        else:
            return RegimeType.EXTREME_VOL
    
    def _classify_trend(self, indicators: RegimeIndicators) -> str:
        """
        Classify trend regime based on ADX and range width.
        
        Args:
            indicators: Market indicators
        
        Returns:
            Trend classification: "TRENDING", "RANGING", or "UNKNOWN"
        """
        # Use ADX if available (preferred method)
        if indicators.adx is not None:
            if indicators.adx >= self.config.adx_trending:
                return "TRENDING"
            else:
                return "RANGING"
        
        # Fall back to range width if ADX not available
        if indicators.range_high is not None and indicators.range_low is not None:
            range_width = (indicators.range_high - indicators.range_low) / indicators.price
            if range_width <= self.config.range_width_threshold:
                return "RANGING"
            else:
                return "TRENDING"
        
        # Cannot determine without indicators
        return "UNKNOWN"
    
    def get_regime_parameters(self, regime: RegimeType) -> Dict[str, float]:
        """
        Get suggested trading parameters for a given regime.
        
        This is a helper method that other components can use to adjust
        their behavior based on regime. Returns multipliers/adjustments.
        
        Args:
            regime: Market regime
        
        Returns:
            Dictionary of parameter adjustments
        """
        # Base parameters
        params = {
            "risk_multiplier": 1.0,      # Multiply base risk by this
            "sl_multiplier": 1.0,         # Multiply SL distance by this
            "tp_multiplier": 1.0,         # Multiply TP distance by this
            "min_confidence_adj": 0.0,    # Add to min confidence threshold
            "max_position_adj": 0,        # Adjust max concurrent positions
        }
        
        # Regime-specific adjustments
        if regime == RegimeType.LOW_VOL:
            # Low volatility: smaller stops, more aggressive
            params["risk_multiplier"] = 1.2
            params["sl_multiplier"] = 0.8
            params["tp_multiplier"] = 1.0
            params["min_confidence_adj"] = -0.05  # Allow slightly lower confidence
        
        elif regime == RegimeType.NORMAL_VOL:
            # Normal volatility: standard parameters
            params["risk_multiplier"] = 1.0
            params["sl_multiplier"] = 1.0
            params["tp_multiplier"] = 1.0
        
        elif regime == RegimeType.HIGH_VOL:
            # High volatility: wider stops, more conservative
            params["risk_multiplier"] = 0.8
            params["sl_multiplier"] = 1.2
            params["tp_multiplier"] = 1.3
            params["min_confidence_adj"] = 0.05  # Require higher confidence
        
        elif regime == RegimeType.EXTREME_VOL:
            # Extreme volatility: much wider stops, very conservative
            params["risk_multiplier"] = 0.5
            params["sl_multiplier"] = 1.5
            params["tp_multiplier"] = 1.5
            params["min_confidence_adj"] = 0.10  # Require much higher confidence
            params["max_position_adj"] = -1      # Reduce max positions by 1
        
        return params
    
    def get_global_regime(self) -> str:
        """
        Get a simplified global market regime string.
        
        This is a stub implementation that returns 'NORMAL' by default.
        In a full implementation, this would aggregate regime data across
        multiple instruments (e.g., BTC, ETH, SPY) to determine overall market state.
        
        Returns:
            Regime string: 'BULL', 'BEAR', 'CHOPPY', 'NORMAL', etc.
        """
        # TODO: Implement proper global regime detection
        # For now, return NORMAL as default
        return "NORMAL"


# Convenience function for quick regime detection
def detect_regime(
    price: float,
    atr: float,
    ema_200: Optional[float] = None,
    adx: Optional[float] = None,
    range_high: Optional[float] = None,
    range_low: Optional[float] = None,
    config: Optional[RegimeConfig] = None
) -> RegimeResult:
    """
    Convenience function to detect regime without creating detector instance.
    
    Args:
        price: Current price
        atr: Average True Range
        ema_200: 200-period EMA (optional)
        adx: Average Directional Index (optional)
        range_high: Recent range high (optional)
        range_low: Recent range low (optional)
        config: Configuration (optional, loads from env if None)
    
    Returns:
        RegimeResult
    """
    detector = RegimeDetector(config)
    indicators = RegimeIndicators(
        price=price,
        atr=atr,
        ema_200=ema_200,
        adx=adx,
        range_high=range_high,
        range_low=range_low
    )
    return detector.detect_regime(indicators)
