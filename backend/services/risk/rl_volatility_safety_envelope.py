"""
RL Volatility Safety Envelope - SPRINT 1 D4

Adds an extra safety layer on top of RL position sizing and leverage decisions
based on current market volatility. Prevents RL agent from taking excessive risk
during high/extreme volatility conditions.

Key Features:
- Volatility classification: LOW, NORMAL, HIGH, EXTREME (based on ATR/price)
- PolicyStore-driven limits per volatility bucket
- Caps RL-proposed leverage and risk% based on current market conditions
- Fail-safe: Allows RL decisions if envelope fails (fail-open)

Architecture:
- Reuses RegimeDetector for volatility calculation (ATR/price ratio)
- Reads limits from PolicyStore (volatility.{bucket}.max_leverage, etc.)
- Applied after RL decision but before Safety Governor
- Does not modify RL logic - pure safety wrapper

Integration:
- Called in EventDrivenExecutor after RL agent decides sizing
- Caps leverage and position size if volatility is HIGH/EXTREME
- Logs all adjustments for monitoring
"""

import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VolatilityBucket(str, Enum):
    """Volatility classification buckets."""
    LOW = "LOW"              # < 0.5% ATR/price
    NORMAL = "NORMAL"        # 0.5-1.5% ATR/price
    HIGH = "HIGH"            # 1.5-3.0% ATR/price
    EXTREME = "EXTREME"      # > 3.0% ATR/price


@dataclass
class VolatilityLimits:
    """Safety limits for a specific volatility bucket."""
    bucket: VolatilityBucket
    max_leverage: float
    max_risk_pct: float  # Max position size as % of equity
    
    def __str__(self):
        return f"{self.bucket.value}: max_lev={self.max_leverage:.1f}x, max_risk={self.max_risk_pct:.1%}"


@dataclass
class EnvelopeResult:
    """Result of applying volatility safety envelope."""
    original_leverage: float
    original_risk_pct: float
    capped_leverage: float
    capped_risk_pct: float
    volatility_bucket: VolatilityBucket
    limits: VolatilityLimits
    was_capped: bool
    
    def __str__(self):
        if self.was_capped:
            return (
                f"⚠️ CAPPED ({self.volatility_bucket.value}): "
                f"Leverage {self.original_leverage:.1f}x → {self.capped_leverage:.1f}x, "
                f"Risk {self.original_risk_pct:.1%} → {self.capped_risk_pct:.1%}"
            )
        else:
            return f"✅ PASS ({self.volatility_bucket.value}): Leverage {self.original_leverage:.1f}x, Risk {self.original_risk_pct:.1%}"


class RLVolatilitySafetyEnvelope:
    """
    Volatility-based safety envelope for RL position sizing.
    
    Caps RL-proposed leverage and position size based on current market volatility.
    Uses PolicyStore for dynamic configuration per volatility bucket.
    
    Example:
        envelope = RLVolatilitySafetyEnvelope(policy_store)
        
        # After RL decision
        result = envelope.apply_limits(
            symbol="BTCUSDT",
            atr_pct=0.025,  # 2.5% ATR
            proposed_leverage=20.0,
            proposed_risk_pct=0.08,  # 8% of equity
            equity_usd=10000.0
        )
        
        # Use capped values
        final_leverage = result.capped_leverage
        final_size = result.capped_risk_pct * equity_usd
    """
    
    def __init__(self, policy_store=None):
        """
        Initialize volatility safety envelope.
        
        Args:
            policy_store: PolicyStore instance for reading limits (optional)
        """
        self.policy_store = policy_store
        
        # Default ATR/price thresholds for volatility classification
        # These match RegimeDetector thresholds
        self.atr_threshold_low = 0.005      # 0.5%
        self.atr_threshold_normal = 0.015   # 1.5%
        self.atr_threshold_high = 0.03      # 3.0%
        
        # Cache for limits (to avoid repeated PolicyStore reads)
        self._limits_cache: Dict[VolatilityBucket, VolatilityLimits] = {}
        
        logger.info("[RL-ENVELOPE] Volatility Safety Envelope initialized")
    
    def get_volatility_bucket(self, atr_pct: float) -> VolatilityBucket:
        """
        Classify volatility based on ATR/price ratio.
        
        Args:
            atr_pct: ATR as percentage of price (e.g., 0.025 = 2.5%)
        
        Returns:
            VolatilityBucket (LOW, NORMAL, HIGH, EXTREME)
        """
        if atr_pct < self.atr_threshold_low:
            return VolatilityBucket.LOW
        elif atr_pct < self.atr_threshold_normal:
            return VolatilityBucket.NORMAL
        elif atr_pct < self.atr_threshold_high:
            return VolatilityBucket.HIGH
        else:
            return VolatilityBucket.EXTREME
    
    def get_limits_for_bucket(self, bucket: VolatilityBucket) -> VolatilityLimits:
        """
        Get safety limits for a volatility bucket from PolicyStore.
        
        Args:
            bucket: Volatility bucket
        
        Returns:
            VolatilityLimits with max_leverage and max_risk_pct
        """
        # Check cache first
        if bucket in self._limits_cache:
            return self._limits_cache[bucket]
        
        # Default limits (conservative fallback)
        default_limits = {
            VolatilityBucket.LOW: VolatilityLimits(
                bucket=VolatilityBucket.LOW,
                max_leverage=25.0,    # Full leverage allowed in low vol
                max_risk_pct=0.10     # 10% position size max
            ),
            VolatilityBucket.NORMAL: VolatilityLimits(
                bucket=VolatilityBucket.NORMAL,
                max_leverage=20.0,    # Slightly reduced
                max_risk_pct=0.08     # 8% position size max
            ),
            VolatilityBucket.HIGH: VolatilityLimits(
                bucket=VolatilityBucket.HIGH,
                max_leverage=15.0,    # Significantly reduced
                max_risk_pct=0.05     # 5% position size max
            ),
            VolatilityBucket.EXTREME: VolatilityLimits(
                bucket=VolatilityBucket.EXTREME,
                max_leverage=10.0,    # Very conservative
                max_risk_pct=0.03     # 3% position size max
            ),
        }
        
        # Get from PolicyStore if available
        if self.policy_store:
            try:
                bucket_key = bucket.value.lower()
                
                max_leverage = self.policy_store.get(
                    f"volatility.{bucket_key}.max_leverage",
                    default=default_limits[bucket].max_leverage
                )
                
                max_risk_pct = self.policy_store.get(
                    f"volatility.{bucket_key}.max_risk_pct",
                    default=default_limits[bucket].max_risk_pct
                )
                
                limits = VolatilityLimits(
                    bucket=bucket,
                    max_leverage=float(max_leverage),
                    max_risk_pct=float(max_risk_pct)
                )
                
                # Cache the result
                self._limits_cache[bucket] = limits
                return limits
                
            except Exception as e:
                logger.warning(f"[RL-ENVELOPE] Failed to read limits from PolicyStore: {e}")
                return default_limits[bucket]
        else:
            # No PolicyStore, use defaults
            return default_limits[bucket]
    
    def apply_limits(
        self,
        symbol: str,
        atr_pct: float,
        proposed_leverage: float,
        proposed_risk_pct: float,
        equity_usd: float
    ) -> EnvelopeResult:
        """
        Apply volatility-based safety limits to RL-proposed values.
        
        Args:
            symbol: Trading symbol (for logging)
            atr_pct: ATR as percentage of price (e.g., 0.025 = 2.5%)
            proposed_leverage: RL-proposed leverage
            proposed_risk_pct: RL-proposed position size as % of equity
            equity_usd: Current account equity in USD
        
        Returns:
            EnvelopeResult with original and capped values
        """
        # Classify volatility
        bucket = self.get_volatility_bucket(atr_pct)
        
        # Get limits for this bucket
        limits = self.get_limits_for_bucket(bucket)
        
        # Apply caps
        capped_leverage = min(proposed_leverage, limits.max_leverage)
        capped_risk_pct = min(proposed_risk_pct, limits.max_risk_pct)
        
        # Check if any capping occurred
        was_capped = (capped_leverage < proposed_leverage) or (capped_risk_pct < proposed_risk_pct)
        
        # Create result
        result = EnvelopeResult(
            original_leverage=proposed_leverage,
            original_risk_pct=proposed_risk_pct,
            capped_leverage=capped_leverage,
            capped_risk_pct=capped_risk_pct,
            volatility_bucket=bucket,
            limits=limits,
            was_capped=was_capped
        )
        
        # Log result
        if was_capped:
            logger.warning(
                f"🛡️ [RL-ENVELOPE] {symbol} | ATR={atr_pct:.2%} ({bucket.value}) | {result}"
            )
        else:
            logger.debug(
                f"✅ [RL-ENVELOPE] {symbol} | ATR={atr_pct:.2%} ({bucket.value}) | {result}"
            )
        
        return result
    
    def calculate_capped_position_size(
        self,
        equity_usd: float,
        capped_risk_pct: float,
        capped_leverage: float,
        price: float
    ) -> Tuple[float, float]:
        """
        Calculate final position size in USD and quantity based on capped values.
        
        Args:
            equity_usd: Account equity in USD
            capped_risk_pct: Capped position size as % of equity
            capped_leverage: Capped leverage
            price: Current asset price
        
        Returns:
            Tuple of (position_size_usd, quantity)
        """
        # Calculate position size in USD (margin)
        margin_usd = equity_usd * capped_risk_pct
        
        # Calculate notional position value (with leverage)
        notional_usd = margin_usd * capped_leverage
        
        # Calculate quantity
        quantity = notional_usd / price if price > 0 else 0.0
        
        return margin_usd, quantity
    
    def get_status(self) -> Dict:
        """
        Get current envelope status and cached limits.
        
        Returns:
            Dictionary with envelope configuration
        """
        return {
            "thresholds": {
                "low": f"{self.atr_threshold_low:.2%}",
                "normal": f"{self.atr_threshold_normal:.2%}",
                "high": f"{self.atr_threshold_high:.2%}"
            },
            "limits_cached": len(self._limits_cache),
            "policy_store_available": self.policy_store is not None,
            "cached_limits": {
                bucket.value: str(limits)
                for bucket, limits in self._limits_cache.items()
            }
        }


# Singleton instance getter
_envelope_instance: Optional[RLVolatilitySafetyEnvelope] = None


def get_rl_volatility_envelope(policy_store=None) -> RLVolatilitySafetyEnvelope:
    """
    Get singleton instance of RL Volatility Safety Envelope.
    
    Args:
        policy_store: PolicyStore instance (only used on first call)
    
    Returns:
        RLVolatilitySafetyEnvelope instance
    """
    global _envelope_instance
    
    if _envelope_instance is None:
        _envelope_instance = RLVolatilitySafetyEnvelope(policy_store)
        logger.info("[RL-ENVELOPE] Singleton instance created")
    
    return _envelope_instance
