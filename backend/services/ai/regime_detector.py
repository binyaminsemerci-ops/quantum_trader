"""
Market Regime Detection Module

Detects market regimes for symbols based on technical indicators,
market structure, and liquidity conditions.

Used by Meta-Strategy Selector to choose appropriate trading strategies.

Author: Quantum Trader Team
Date: 2025-11-26
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    TREND_UP = "trend_up"                 # Strong uptrend
    TREND_DOWN = "trend_down"             # Strong downtrend
    RANGE_LOW_VOL = "range_low_vol"       # Sideways, low volatility
    RANGE_HIGH_VOL = "range_high_vol"     # Sideways, high volatility
    HIGH_VOLATILITY = "high_volatility"   # Extreme volatility/whipsaw
    ILLIQUID = "illiquid"                 # Low liquidity/dangerous
    UNKNOWN = "unknown"                   # Cannot determine


@dataclass
class MarketContext:
    """
    Market context data for regime detection.
    
    Attributes:
        symbol: Trading symbol
        timeframe: Timeframe (e.g., "15m", "1h")
        
        # Price & trend
        current_price: Current mark price
        sma_20: 20-period simple moving average
        sma_50: 50-period simple moving average
        ema_12: 12-period exponential moving average
        ema_26: 26-period exponential moving average
        
        # Volatility
        atr: Average True Range
        atr_pct: ATR as percentage of price
        bb_width: Bollinger Band width (as % of price)
        
        # Trend strength
        adx: Average Directional Index (0-100)
        trend_strength: Custom trend strength (-1 to +1)
        
        # Volume & liquidity
        volume_24h: 24-hour volume in USDT
        avg_volume_24h: Average 24h volume (last 7 days)
        depth_5bps: Orderbook depth within 5bps
        spread_bps: Bid-ask spread in basis points
        
        # Market structure
        funding_rate: Current funding rate
        open_interest: Open interest in USDT
        
        # Historical performance (for this symbol)
        recent_win_rate: Win rate for last N trades (0-1)
        recent_avg_r: Average R for last N trades
        trade_count: Number of historical trades
    """
    
    symbol: str
    timeframe: str = "15m"
    
    # Price & trend
    current_price: float = 0.0
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Volatility
    atr: float = 0.0
    atr_pct: float = 0.0
    bb_width: Optional[float] = None
    
    # Trend strength
    adx: Optional[float] = None
    trend_strength: float = 0.0  # -1 (down) to +1 (up)
    
    # Volume & liquidity
    volume_24h: float = 0.0
    avg_volume_24h: float = 0.0
    depth_5bps: float = 0.0
    spread_bps: float = 0.0
    
    # Market structure
    funding_rate: float = 0.0
    open_interest: float = 0.0
    
    # Historical performance
    recent_win_rate: Optional[float] = None
    recent_avg_r: Optional[float] = None
    trade_count: int = 0


@dataclass
class RegimeDetectionResult:
    """
    Result of regime detection.
    
    Attributes:
        regime: Detected market regime
        confidence: Detection confidence (0-1)
        reasoning: Human-readable explanation
        features: Feature values used for detection
    """
    
    regime: MarketRegime
    confidence: float
    reasoning: str
    features: Dict[str, float]


class RegimeDetector:
    """
    Market regime detector using rule-based analysis.
    
    Analyzes market context and classifies into one of several regimes.
    Future versions can use ML models, but we start with robust rules.
    """
    
    def __init__(
        self,
        # Trend thresholds
        trend_adx_threshold: float = 25.0,
        strong_trend_adx_threshold: float = 40.0,
        trend_strength_threshold: float = 0.3,
        
        # Volatility thresholds
        high_vol_atr_pct_threshold: float = 0.04,  # 4% ATR
        extreme_vol_atr_pct_threshold: float = 0.06,  # 6% ATR
        low_vol_atr_pct_threshold: float = 0.015,  # 1.5% ATR
        
        # Liquidity thresholds
        illiquid_volume_threshold: float = 2_000_000.0,  # $2M
        illiquid_depth_threshold: float = 50_000.0,  # $50k
        illiquid_spread_threshold: float = 10.0,  # 10 bps
    ):
        """
        Initialize regime detector.
        
        Args:
            trend_adx_threshold: ADX above this = trending
            strong_trend_adx_threshold: ADX above this = strong trend
            trend_strength_threshold: Trend strength above this = directional
            high_vol_atr_pct_threshold: ATR% above this = high vol
            extreme_vol_atr_pct_threshold: ATR% above this = extreme vol
            low_vol_atr_pct_threshold: ATR% below this = low vol
            illiquid_volume_threshold: Volume below this = illiquid
            illiquid_depth_threshold: Depth below this = illiquid
            illiquid_spread_threshold: Spread above this = illiquid
        """
        self.trend_adx_threshold = trend_adx_threshold
        self.strong_trend_adx_threshold = strong_trend_adx_threshold
        self.trend_strength_threshold = trend_strength_threshold
        
        self.high_vol_atr_pct_threshold = high_vol_atr_pct_threshold
        self.extreme_vol_atr_pct_threshold = extreme_vol_atr_pct_threshold
        self.low_vol_atr_pct_threshold = low_vol_atr_pct_threshold
        
        self.illiquid_volume_threshold = illiquid_volume_threshold
        self.illiquid_depth_threshold = illiquid_depth_threshold
        self.illiquid_spread_threshold = illiquid_spread_threshold
        
        logger.info(
            f"[OK] RegimeDetector initialized: "
            f"trend_adx={trend_adx_threshold}, high_vol={high_vol_atr_pct_threshold:.2%}"
        )
    
    def detect_regime(self, context: MarketContext) -> RegimeDetectionResult:
        """
        Detect market regime from context.
        
        Args:
            context: Market context data
            
        Returns:
            RegimeDetectionResult with regime classification
            
        Example:
            >>> detector = RegimeDetector()
            >>> context = MarketContext(
            ...     symbol="BTCUSDT",
            ...     atr_pct=0.025,
            ...     trend_strength=0.6,
            ...     adx=35.0,
            ...     volume_24h=50_000_000
            ... )
            >>> result = detector.detect_regime(context)
            >>> print(result.regime, result.confidence)
            MarketRegime.TREND_UP 0.85
        """
        
        # Calculate feature scores
        features = self._extract_features(context)
        
        # Check for ILLIQUID first (highest priority)
        if self._is_illiquid(context, features):
            return RegimeDetectionResult(
                regime=MarketRegime.ILLIQUID,
                confidence=0.95,
                reasoning=(
                    f"Illiquid market: volume=${context.volume_24h/1e6:.1f}M, "
                    f"depth=${context.depth_5bps/1e3:.0f}k, spread={context.spread_bps:.1f}bps"
                ),
                features=features
            )
        
        # Check for EXTREME VOLATILITY
        if self._is_extreme_volatility(context, features):
            return RegimeDetectionResult(
                regime=MarketRegime.HIGH_VOLATILITY,
                confidence=0.90,
                reasoning=(
                    f"Extreme volatility: ATR={context.atr_pct:.2%}, "
                    f"BB width={features.get('bb_width_pct', 0):.2%}"
                ),
                features=features
            )
        
        # Check for TRENDING markets
        trend_result = self._detect_trend(context, features)
        if trend_result is not None:
            return trend_result
        
        # Check for RANGING markets
        range_result = self._detect_range(context, features)
        if range_result is not None:
            return range_result
        
        # Default: UNKNOWN
        return RegimeDetectionResult(
            regime=MarketRegime.UNKNOWN,
            confidence=0.50,
            reasoning="Insufficient data or mixed signals",
            features=features
        )
    
    def _extract_features(self, context: MarketContext) -> Dict[str, float]:
        """Extract numerical features from context."""
        features = {
            "atr_pct": context.atr_pct,
            "trend_strength": context.trend_strength,
            "adx": context.adx or 0.0,
            "volume_24h": context.volume_24h,
            "depth_5bps": context.depth_5bps,
            "spread_bps": context.spread_bps,
        }
        
        # Add derived features
        if context.bb_width is not None:
            features["bb_width_pct"] = context.bb_width
        
        if context.sma_20 and context.current_price > 0:
            features["price_vs_sma20"] = (context.current_price - context.sma_20) / context.sma_20
        
        if context.sma_50 and context.current_price > 0:
            features["price_vs_sma50"] = (context.current_price - context.sma_50) / context.sma_50
        
        if context.ema_12 and context.ema_26:
            features["macd_signal"] = (context.ema_12 - context.ema_26) / context.current_price if context.current_price > 0 else 0.0
        
        if context.avg_volume_24h > 0:
            features["volume_ratio"] = context.volume_24h / context.avg_volume_24h
        
        return features
    
    def _is_illiquid(self, context: MarketContext, features: Dict) -> bool:
        """Check if market is illiquid/dangerous."""
        illiquid_count = 0
        
        if context.volume_24h < self.illiquid_volume_threshold:
            illiquid_count += 1
        
        if context.depth_5bps < self.illiquid_depth_threshold:
            illiquid_count += 1
        
        if context.spread_bps > self.illiquid_spread_threshold:
            illiquid_count += 1
        
        # Need at least 2 out of 3 illiquidity signals
        return illiquid_count >= 2
    
    def _is_extreme_volatility(self, context: MarketContext, features: Dict) -> bool:
        """Check for extreme volatility."""
        return context.atr_pct >= self.extreme_vol_atr_pct_threshold
    
    def _detect_trend(self, context: MarketContext, features: Dict) -> Optional[RegimeDetectionResult]:
        """Detect trending market."""
        adx = features.get("adx", 0)
        trend_str = features.get("trend_strength", 0)
        
        # Need both ADX and trend strength signals
        is_trending = (
            adx >= self.trend_adx_threshold and
            abs(trend_str) >= self.trend_strength_threshold
        )
        
        if not is_trending:
            return None
        
        # Determine direction
        if trend_str > 0:
            regime = MarketRegime.TREND_UP
            direction = "UP"
        else:
            regime = MarketRegime.TREND_DOWN
            direction = "DOWN"
        
        # Calculate confidence
        confidence = min(0.95, 0.6 + (adx / 100.0) * 0.2 + abs(trend_str) * 0.15)
        
        reasoning = (
            f"Trending {direction}: ADX={adx:.1f}, trend_strength={trend_str:.2f}, "
            f"ATR={context.atr_pct:.2%}"
        )
        
        return RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            reasoning=reasoning,
            features=features
        )
    
    def _detect_range(self, context: MarketContext, features: Dict) -> Optional[RegimeDetectionResult]:
        """Detect ranging market."""
        adx = features.get("adx", 0)
        atr_pct = features.get("atr_pct", 0)
        trend_str = abs(features.get("trend_strength", 0))
        
        # Range conditions: low ADX, low trend strength
        is_ranging = (
            adx < self.trend_adx_threshold and
            trend_str < self.trend_strength_threshold
        )
        
        if not is_ranging:
            return None
        
        # Classify by volatility
        if atr_pct >= self.high_vol_atr_pct_threshold:
            regime = MarketRegime.RANGE_HIGH_VOL
            vol_label = "HIGH"
        else:
            regime = MarketRegime.RANGE_LOW_VOL
            vol_label = "LOW"
        
        confidence = min(0.90, 0.65 + (1.0 - trend_str) * 0.25)
        
        reasoning = (
            f"Ranging {vol_label} VOL: ADX={adx:.1f}, trend_strength={trend_str:.2f}, "
            f"ATR={atr_pct:.2%}"
        )
        
        return RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            reasoning=reasoning,
            features=features
        )
    
    def get_regime_summary(self, context: MarketContext) -> str:
        """
        Get human-readable regime summary.
        
        Args:
            context: Market context
            
        Returns:
            String summary of detected regime
        """
        result = self.detect_regime(context)
        return f"{context.symbol} [{context.timeframe}]: {result.regime.value.upper()} (confidence={result.confidence:.2%}) - {result.reasoning}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_trend_strength(
    current_price: float,
    sma_20: Optional[float],
    sma_50: Optional[float],
    ema_12: Optional[float],
    ema_26: Optional[float],
) -> float:
    """
    Calculate trend strength score (-1 to +1).
    
    Uses multiple moving averages to determine trend direction and strength.
    
    Args:
        current_price: Current price
        sma_20: 20-period SMA
        sma_50: 50-period SMA
        ema_12: 12-period EMA
        ema_26: 26-period EMA
        
    Returns:
        Trend strength: +1 (strong up), 0 (no trend), -1 (strong down)
    """
    if current_price <= 0:
        return 0.0
    
    signals = []
    
    # Price vs SMAs
    if sma_20 is not None and sma_20 > 0:
        signals.append((current_price - sma_20) / sma_20)
    
    if sma_50 is not None and sma_50 > 0:
        signals.append((current_price - sma_50) / sma_50)
    
    # MACD-like signal
    if ema_12 is not None and ema_26 is not None:
        macd_signal = (ema_12 - ema_26) / current_price
        signals.append(macd_signal * 10)  # Scale up
    
    if not signals:
        return 0.0
    
    # Average and clip to [-1, 1]
    avg_signal = np.mean(signals)
    return float(np.clip(avg_signal * 5, -1.0, 1.0))  # Scale to reasonable range


def calculate_adx_from_highs_lows(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> float:
    """
    Calculate Average Directional Index (ADX).
    
    Simple ADX calculation for trend strength measurement.
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        period: ADX period (default 14)
        
    Returns:
        ADX value (0-100)
    """
    if len(highs) < period + 1:
        return 0.0
    
    # Calculate True Range
    tr1 = highs[1:] - lows[1:]
    tr2 = np.abs(highs[1:] - closes[:-1])
    tr3 = np.abs(lows[1:] - closes[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with EMA
    atr = np.mean(tr[-period:])
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
    
    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    
    # ADX is smoothed DX (simplified here as single value)
    adx = float(dx)
    
    return min(100.0, max(0.0, adx))


if __name__ == "__main__":
    # Demo
    print("🔍 REGIME DETECTOR DEMO\n")
    
    detector = RegimeDetector()
    
    # Test cases
    test_cases = [
        MarketContext(
            symbol="BTCUSDT",
            atr_pct=0.025,
            trend_strength=0.6,
            adx=35.0,
            volume_24h=50_000_000,
            depth_5bps=500_000,
            spread_bps=2.5,
        ),
        MarketContext(
            symbol="ETHUSDT",
            atr_pct=0.015,
            trend_strength=0.1,
            adx=18.0,
            volume_24h=30_000_000,
            depth_5bps=300_000,
            spread_bps=3.0,
        ),
        MarketContext(
            symbol="ALTUSDT",
            atr_pct=0.07,
            trend_strength=0.2,
            adx=22.0,
            volume_24h=1_500_000,
            depth_5bps=30_000,
            spread_bps=15.0,
        ),
    ]
    
    for ctx in test_cases:
        result = detector.detect_regime(ctx)
        print(f"Symbol: {ctx.symbol}")
        print(f"  Regime: {result.regime.value.upper()}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Reasoning: {result.reasoning}")
        print()
