"""
Regime Detector - Analyzes market trends and defines regimes.
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Dict


class RegimeDetector:
    """Analyzes market trends and defines regimes"""
    
    def __init__(self):
        self.lookback_short = 50
        self.lookback_long = 200
        
        # Regime thresholds
        self.vol_low = 0.015
        self.vol_high = 0.03
        self.trend_threshold = 0.001
    
    def detect(self, prices: pd.Series) -> Dict:
        """
        Detect current market regime based on price action.
        
        Args:
            prices: Price series (at least 200 periods recommended)
            
        Returns:
            Dictionary with regime classification and metrics
        """
        if len(prices) < self.lookback_short:
            return {
                "regime": "UNKNOWN",
                "volatility": 0.0,
                "trend": 0.0,
                "confidence": 0.0,
                "reason": "insufficient_data"
            }
        
        # Calculate metrics
        returns = prices.pct_change().dropna()
        
        # Volatility (rolling std of returns)
        vol = returns.rolling(self.lookback_short).std().iloc[-1]
        
        # Trend (linear regression slope)
        recent_prices = prices[-self.lookback_short:]
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        # Normalized trend (percentage change per period)
        normalized_trend = trend / prices.iloc[-1]
        
        # Detect regime
        regime, confidence = self._classify_regime(vol, normalized_trend)
        
        return {
            "regime": regime,
            "volatility": float(vol),
            "trend": float(normalized_trend),
            "confidence": float(confidence),
            "price": float(prices.iloc[-1]),
            "samples": len(prices)
        }
    
    def _classify_regime(self, volatility: float, trend: float) -> tuple:
        """
        Classify market regime based on volatility and trend.
        
        Returns:
            Tuple of (regime_name, confidence)
        """
        # Range-bound: Low volatility, no clear trend
        if volatility < self.vol_low and abs(trend) < self.trend_threshold:
            return "RANGE", 0.9
        
        # Bull: Positive trend, moderate volatility
        elif trend > self.trend_threshold and volatility < self.vol_high:
            confidence = min(0.95, 0.7 + (trend / self.trend_threshold) * 0.1)
            return "BULL", confidence
        
        # Bear: Negative trend, moderate volatility
        elif trend < -self.trend_threshold and volatility < self.vol_high:
            confidence = min(0.95, 0.7 + abs(trend / self.trend_threshold) * 0.1)
            return "BEAR", confidence
        
        # Volatile: High volatility regardless of trend
        elif volatility >= self.vol_high:
            return "VOLATILE", 0.85
        
        # Uncertain: Doesn't fit clear patterns
        else:
            return "UNCERTAIN", 0.5
    
    def detect_advanced(self, prices: pd.Series, volume: pd.Series = None) -> Dict:
        """
        Advanced regime detection with volume confirmation.
        
        Args:
            prices: Price series
            volume: Optional volume series for confirmation
            
        Returns:
            Enhanced regime classification
        """
        basic_regime = self.detect(prices)
        
        if volume is not None and len(volume) == len(prices):
            # Volume confirmation
            vol_ma = volume.rolling(self.lookback_short).mean()
            recent_vol = volume[-10:].mean()
            
            if recent_vol > vol_ma.iloc[-1] * 1.2:
                basic_regime["volume_confirmation"] = "strong"
                basic_regime["confidence"] = min(0.95, basic_regime["confidence"] * 1.1)
            elif recent_vol < vol_ma.iloc[-1] * 0.8:
                basic_regime["volume_confirmation"] = "weak"
                basic_regime["confidence"] = max(0.3, basic_regime["confidence"] * 0.9)
            else:
                basic_regime["volume_confirmation"] = "neutral"
        
        return basic_regime
