"""Global Market Regime Detector - Determines overall market trend for safety rules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalRegime(Enum):
    """Global market regime classification."""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class GlobalRegimeDetectionResult:
    """Result of global regime detection."""
    regime: GlobalRegime
    btc_price: float
    btc_ema200: float
    price_vs_ema_pct: float  # Price as percentage of EMA200
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    notes: str = ""


class GlobalRegimeDetector:
    """
    Detects global market regime based on major market indicators (primarily BTCUSDT).
    
    This is used to enforce safety rules:
    - In UPTREND: Default to LONG-only, block most SHORTS
    - In DOWNTREND: Allow both directions
    - In SIDEWAYS: Allow both directions
    
    Detection is simple and reliable:
    - UPTREND: BTCUSDT price > EMA200 + significant buffer
    - DOWNTREND: BTCUSDT price < EMA200 - significant buffer
    - SIDEWAYS: Price within buffer of EMA200
    """
    
    def __init__(
        self,
        uptrend_threshold_pct: float = 2.0,  # Price must be 2% above EMA200 for UPTREND
        downtrend_threshold_pct: float = 2.0,  # Price must be 2% below EMA200 for DOWNTREND
    ):
        """
        Initialize global regime detector.
        
        Args:
            uptrend_threshold_pct: Percentage above EMA200 to classify as UPTREND
            downtrend_threshold_pct: Percentage below EMA200 to classify as DOWNTREND
        """
        self.uptrend_threshold = 1.0 + (uptrend_threshold_pct / 100.0)
        self.downtrend_threshold = 1.0 - (downtrend_threshold_pct / 100.0)
        
        logger.info(
            f"[OK] GlobalRegimeDetector initialized: "
            f"UPTREND if price > {self.uptrend_threshold:.3f}x EMA200, "
            f"DOWNTREND if price < {self.downtrend_threshold:.3f}x EMA200"
        )
    
    def detect_global_regime(
        self,
        btc_price: float,
        btc_ema200: float,
    ) -> GlobalRegimeDetectionResult:
        """
        Detect global market regime based on BTCUSDT.
        
        Args:
            btc_price: Current BTCUSDT price
            btc_ema200: BTCUSDT EMA200 value
            
        Returns:
            GlobalRegimeDetectionResult with regime classification
        """
        if btc_price <= 0 or btc_ema200 <= 0:
            logger.warning("Invalid BTC price or EMA200 data")
            return GlobalRegimeDetectionResult(
                regime=GlobalRegime.UNKNOWN,
                btc_price=btc_price,
                btc_ema200=btc_ema200,
                price_vs_ema_pct=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                notes="Invalid price data"
            )
        
        # Calculate price vs EMA ratio
        price_ratio = btc_price / btc_ema200
        price_vs_ema_pct = (price_ratio - 1.0) * 100.0
        
        # Classify regime
        if price_ratio >= self.uptrend_threshold:
            regime = GlobalRegime.UPTREND
            # Confidence increases with distance from threshold
            excess_pct = (price_ratio - self.uptrend_threshold) * 100.0
            confidence = min(1.0, 0.7 + (excess_pct / 10.0))  # 0.7 base + 0.1 per 1% excess
            notes = f"BTC {price_vs_ema_pct:+.1f}% above EMA200 (threshold: +{(self.uptrend_threshold-1.0)*100:.1f}%)"
            
        elif price_ratio <= self.downtrend_threshold:
            regime = GlobalRegime.DOWNTREND
            # Confidence increases with distance from threshold
            deficit_pct = (self.downtrend_threshold - price_ratio) * 100.0
            confidence = min(1.0, 0.7 + (deficit_pct / 10.0))
            notes = f"BTC {price_vs_ema_pct:.1f}% below EMA200 (threshold: -{(1.0-self.downtrend_threshold)*100:.1f}%)"
            
        else:
            regime = GlobalRegime.SIDEWAYS
            # Confidence is lower in sideways (less clear direction)
            confidence = 0.6
            notes = f"BTC {price_vs_ema_pct:+.1f}% vs EMA200 (within ±{(self.uptrend_threshold-1.0)*100:.1f}% range)"
        
        result = GlobalRegimeDetectionResult(
            regime=regime,
            btc_price=btc_price,
            btc_ema200=btc_ema200,
            price_vs_ema_pct=price_vs_ema_pct,
            confidence=confidence,
            timestamp=datetime.now(),
            notes=notes
        )
        
        logger.info(
            f"[GLOBE] Global Regime: {regime.value} "
            f"(BTC ${btc_price:.2f} vs EMA ${btc_ema200:.2f}, "
            f"{price_vs_ema_pct:+.2f}%, confidence {confidence:.0%})"
        )
        
        return result
    
    def should_block_shorts(
        self,
        global_regime_result: GlobalRegimeDetectionResult
    ) -> bool:
        """
        Determine if SHORT trades should be blocked based on global regime.
        
        Args:
            global_regime_result: Current global regime detection result
            
        Returns:
            True if shorts should be blocked, False otherwise
        """
        # Block shorts in clear UPTREND
        if global_regime_result.regime == GlobalRegime.UPTREND:
            if global_regime_result.confidence >= 0.7:
                logger.info(
                    f"[SAFETY] BLOCKING SHORTS: Strong global UPTREND detected "
                    f"(confidence {global_regime_result.confidence:.0%})"
                )
                return True
        
        return False
    
    def check_short_exception(
        self,
        global_regime_result: GlobalRegimeDetectionResult,
        symbol: str,
        symbol_price: float,
        symbol_ema200: float,
        short_confidence: float,
    ) -> tuple[bool, str]:
        """
        Check if a SHORT trade qualifies for exception in UPTREND.
        
        Exceptions are RARE and require ALL of:
        1. Global regime is UPTREND
        2. Local symbol is in DOWNTREND (price < EMA200)
        3. Short confidence >= exception threshold (from config)
        
        Args:
            global_regime_result: Current global regime
            symbol: Trading symbol
            symbol_price: Current price of symbol
            symbol_ema200: EMA200 of symbol
            short_confidence: AI confidence for SHORT signal
            
        Returns:
            Tuple of (allow_short, reason)
        """
        # Only check exceptions if global regime is UPTREND
        if global_regime_result.regime != GlobalRegime.UPTREND:
            return True, "Global regime not UPTREND - no exception needed"
        
        # Get exception threshold from config
        try:
            from config.config import get_uptrend_short_exception_threshold
            exception_threshold = get_uptrend_short_exception_threshold()
        except ImportError:
            exception_threshold = 0.65
        
        # Check if symbol is in local DOWNTREND
        if symbol_price >= symbol_ema200:
            return False, (
                f"Symbol {symbol} not in local DOWNTREND "
                f"(price ${symbol_price:.2f} >= EMA200 ${symbol_ema200:.2f})"
            )
        
        # Check if confidence meets exception threshold
        if short_confidence < exception_threshold:
            return False, (
                f"Confidence {short_confidence:.1%} < exception threshold {exception_threshold:.1%}"
            )
        
        # All exception criteria met
        reason = (
            f"[SAFETY] RARE SHORT ALLOWED in global UPTREND | "
            f"symbol {symbol} in local DOWNTREND "
            f"(${symbol_price:.2f} < ${symbol_ema200:.2f}), "
            f"confidence {short_confidence:.1%} >= {exception_threshold:.1%}"
        )
        logger.warning(reason)
        
        return True, reason


# Global instance (optional singleton pattern)
_global_detector: Optional[GlobalRegimeDetector] = None


def get_global_regime_detector() -> GlobalRegimeDetector:
    """Get or create global regime detector singleton."""
    global _global_detector
    if _global_detector is None:
        _global_detector = GlobalRegimeDetector()
    return _global_detector
