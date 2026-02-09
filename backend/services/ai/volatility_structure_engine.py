"""
Volatility Structure Engine - Advanced Volatility Analysis
===========================================================

Provides multi-dimensional volatility insights:
- ATR trend detection (acceleration/deceleration)
- Cross-timeframe volatility comparison
- Volatility expansion/contraction signals
- Volatility regime classification

Phase 2D Enhancement for Quantum Trader AI OS

Author: Quantum Trader AI Team
Date: December 23, 2025
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict, deque
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class VolatilityStructureEngine:
    """
    Advanced volatility structure analysis engine.
    
    Provides:
    - ATR trend (acceleration/deceleration)
    - Cross-timeframe volatility comparison
    - Volatility expansion/contraction detection
    - Volatility regime classification
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        atr_trend_lookback: int = 5,
        volatility_expansion_threshold: float = 1.5,
        volatility_contraction_threshold: float = 0.5,
        history_size: int = 200
    ):
        """
        Initialize Volatility Structure Engine.
        
        Args:
            atr_period: ATR calculation period
            atr_trend_lookback: Periods to detect ATR trend
            volatility_expansion_threshold: Threshold for expansion detection (ratio)
            volatility_contraction_threshold: Threshold for contraction detection (ratio)
            history_size: Maximum history to keep per symbol
        """
        self.atr_period = atr_period
        self.atr_trend_lookback = atr_trend_lookback
        self.volatility_expansion_threshold = volatility_expansion_threshold
        self.volatility_contraction_threshold = volatility_contraction_threshold
        self.history_size = history_size
        
        # Per-symbol data storage
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._high_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._low_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._atr_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Timeframe windows for cross-TF analysis
        self.timeframe_windows = {
            "1m": 60,      # 1 hour of 1-minute bars
            "5m": 60,      # 5 hours of 5-minute bars
            "15m": 64,     # 16 hours of 15-minute bars
            "1h": 48       # 2 days of hourly bars
        }
        
        logger.info(
            f"[Volatility Structure Engine] Initialized: "
            f"ATR period={atr_period}, trend lookback={atr_trend_lookback}, "
            f"expansion threshold={volatility_expansion_threshold}x"
        )
    
    def update_price_data(
        self,
        symbol: str,
        close: float,
        high: Optional[float] = None,
        low: Optional[float] = None
    ):
        """
        Update price data for symbol.
        
        Args:
            symbol: Trading symbol
            close: Close price
            high: High price (optional, uses close if not provided)
            low: Low price (optional, uses close if not provided)
        """
        self._price_history[symbol].append(close)
        self._high_history[symbol].append(high if high is not None else close)
        self._low_history[symbol].append(low if low is not None else close)
    
    def calculate_atr(self, symbol: str) -> float:
        """
        Calculate Average True Range (ATR) for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            ATR value
        """
        if len(self._price_history[symbol]) < self.atr_period + 1:
            return 0.0
        
        prices = list(self._price_history[symbol])
        highs = list(self._high_history[symbol])
        lows = list(self._low_history[symbol])
        
        true_ranges = []
        for i in range(1, len(prices)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - prices[i-1])
            low_close = abs(lows[i] - prices[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < self.atr_period:
            return 0.0
        
        # Calculate ATR as SMA of true ranges
        atr = np.mean(true_ranges[-self.atr_period:])
        
        # Store ATR for trend analysis
        self._atr_history[symbol].append(atr)
        
        return float(atr)
    
    def calculate_atr_trend(self, symbol: str) -> Dict[str, float]:
        """
        Calculate ATR trend (acceleration/deceleration).
        
        Returns:
            Dict with:
            - atr_trend: Trend direction (-1 to 1)
            - atr_acceleration: Rate of change
            - atr_regime: "accelerating", "stable", "decelerating"
        """
        if len(self._atr_history[symbol]) < self.atr_trend_lookback + 1:
            return {
                "atr_trend": 0.0,
                "atr_acceleration": 0.0,
                "atr_regime": "unknown"
            }
        
        atr_values = list(self._atr_history[symbol])[-self.atr_trend_lookback-1:]
        
        # Calculate ATR changes
        atr_changes = np.diff(atr_values)
        
        # ATR trend: positive = accelerating volatility, negative = decelerating
        atr_trend = np.mean(atr_changes)
        
        # Normalize trend to [-1, 1]
        avg_atr = np.mean(atr_values)
        normalized_trend = np.tanh(atr_trend / avg_atr * 10) if avg_atr > 0 else 0.0
        
        # Calculate acceleration (2nd derivative)
        if len(atr_changes) >= 2:
            atr_acceleration = np.mean(np.diff(atr_changes))
        else:
            atr_acceleration = 0.0
        
        # Classify regime
        if normalized_trend > 0.3:
            atr_regime = "accelerating"
        elif normalized_trend < -0.3:
            atr_regime = "decelerating"
        else:
            atr_regime = "stable"
        
        return {
            "atr_trend": float(normalized_trend),
            "atr_acceleration": float(atr_acceleration),
            "atr_regime": atr_regime
        }
    
    def calculate_cross_timeframe_volatility(self, symbol: str) -> Dict[str, float]:
        """
        Calculate cross-timeframe volatility comparison.
        
        Compares short-term vs long-term volatility to detect:
        - Volatility expansion (short > long)
        - Volatility contraction (short < long)
        
        Returns:
            Dict with:
            - short_term_vol: Recent volatility (last 15 bars)
            - medium_term_vol: Medium volatility (last 50 bars)
            - long_term_vol: Long-term volatility (last 100 bars)
            - vol_ratio_short_medium: Short/Medium ratio
            - vol_ratio_short_long: Short/Long ratio
            - vol_regime: "expansion", "contraction", "normal"
        """
        prices = list(self._price_history[symbol])
        
        if len(prices) < 100:
            return {
                "short_term_vol": 0.0,
                "medium_term_vol": 0.0,
                "long_term_vol": 0.0,
                "vol_ratio_short_medium": 1.0,
                "vol_ratio_short_long": 1.0,
                "vol_regime": "unknown"
            }
        
        # Calculate volatility for different timeframes
        short_term_vol = self._calculate_volatility(prices[-15:])
        medium_term_vol = self._calculate_volatility(prices[-50:])
        long_term_vol = self._calculate_volatility(prices[-100:])
        
        # Calculate ratios
        vol_ratio_short_medium = short_term_vol / medium_term_vol if medium_term_vol > 0 else 1.0
        vol_ratio_short_long = short_term_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Classify volatility regime
        if vol_ratio_short_long > self.volatility_expansion_threshold:
            vol_regime = "expansion"
        elif vol_ratio_short_long < self.volatility_contraction_threshold:
            vol_regime = "contraction"
        else:
            vol_regime = "normal"
        
        return {
            "short_term_vol": float(short_term_vol),
            "medium_term_vol": float(medium_term_vol),
            "long_term_vol": float(long_term_vol),
            "vol_ratio_short_medium": float(vol_ratio_short_medium),
            "vol_ratio_short_long": float(vol_ratio_short_long),
            "vol_regime": vol_regime
        }
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate volatility (std dev of returns) for price series."""
        if len(prices) < 2:
            return 0.0
        
        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        
        return float(np.std(returns))
    
    async def get_structure(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get volatility structure for symbol (async wrapper for exit evaluator).
        
        Note: This is a fail-safe stub. For full volatility analysis, use 
        calculate_atr_trend() or calculate_cross_timeframe_volatility() after
        updating price data. Returns None to allow exit evaluator to skip
        volatility checks gracefully.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            None (fail-safe - volatility check will be skipped)
        """
        logger.debug(f"[VSE] get_structure called for {symbol} (stub - returning None)")
        return None
    
    def get_complete_volatility_analysis(self, symbol: str) -> Dict:
        """
        Get complete volatility structure analysis for symbol.
        
        Returns:
            Dict with all volatility metrics:
            - atr: Current ATR
            - atr_trend_data: ATR trend analysis
            - cross_tf_data: Cross-timeframe volatility
            - volatility_score: Combined volatility score (0-1)
            - volatility_regime: Overall regime classification
        """
        # Calculate ATR
        current_atr = self.calculate_atr(symbol)
        
        # Get ATR trend
        atr_trend_data = self.calculate_atr_trend(symbol)
        
        # Get cross-timeframe volatility
        cross_tf_data = self.calculate_cross_timeframe_volatility(symbol)
        
        # Calculate combined volatility score (0-1)
        # High score = high volatility expansion + accelerating ATR
        volatility_score = 0.0
        
        # ATR contribution (30%)
        if atr_trend_data["atr_trend"] > 0:
            volatility_score += 0.3 * abs(atr_trend_data["atr_trend"])
        
        # Cross-TF expansion contribution (40%)
        if cross_tf_data["vol_regime"] == "expansion":
            volatility_score += 0.4 * (cross_tf_data["vol_ratio_short_long"] - 1.0) / 2.0
        elif cross_tf_data["vol_regime"] == "contraction":
            volatility_score += 0.2  # Low volatility = some score
        
        # Current volatility level contribution (30%)
        if len(self._price_history[symbol]) >= 50:
            recent_prices = list(self._price_history[symbol])[-50:]
            current_vol = self._calculate_volatility(recent_prices)
            normalized_vol = min(current_vol * 100, 1.0)  # Normalize to [0, 1]
            volatility_score += 0.3 * normalized_vol
        
        # Clamp to [0, 1]
        volatility_score = max(0.0, min(1.0, volatility_score))
        
        # Determine overall volatility regime
        if atr_trend_data["atr_regime"] == "accelerating" and cross_tf_data["vol_regime"] == "expansion":
            overall_regime = "high_expansion"
        elif atr_trend_data["atr_regime"] == "decelerating" and cross_tf_data["vol_regime"] == "contraction":
            overall_regime = "low_contraction"
        elif cross_tf_data["vol_regime"] == "expansion":
            overall_regime = "expansion"
        elif cross_tf_data["vol_regime"] == "contraction":
            overall_regime = "contraction"
        else:
            overall_regime = "normal"
        
        return {
            "atr": current_atr,
            "atr_trend": atr_trend_data["atr_trend"],
            "atr_acceleration": atr_trend_data["atr_acceleration"],
            "atr_regime": atr_trend_data["atr_regime"],
            "short_term_vol": cross_tf_data["short_term_vol"],
            "medium_term_vol": cross_tf_data["medium_term_vol"],
            "long_term_vol": cross_tf_data["long_term_vol"],
            "vol_ratio_short_long": cross_tf_data["vol_ratio_short_long"],
            "vol_regime": cross_tf_data["vol_regime"],
            "volatility_score": volatility_score,
            "overall_regime": overall_regime
        }
