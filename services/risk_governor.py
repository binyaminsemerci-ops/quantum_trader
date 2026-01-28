"""
Risk Governor - BRIDGE-PATCH v1.0

Enforces safety bounds on size/leverage before order execution.
All bounds are configurable via environment variables.

Design:
- Fail-closed: Conservative defaults, rejects on any policy violation
- Fail-open: Only rejects on safety critical violations (size/leverage out of range)
- Logged decision: Every ACCEPT/REJECT decision logged at INFO level
- Metadata: Returns detailed decision info for publishing to execution.result

Safety Policies:
1. Size bounds: [MIN_ORDER_USD .. MAX_POSITION_USD]
2. Leverage bounds: [5 .. 80]
3. Notional bounds: size * leverage <= MAX_NOTIONAL_USD
4. Confidence floor: if MIN_CONFIDENCE set, reject if signal confidence < threshold
5. Risk budget: if risk_budget_usd in intent, enforce it
"""

import os
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("risk-governor")


@dataclass
class GovernorConfig:
    """Risk Governor Configuration"""
    min_order_usd: float = 50.0
    max_position_usd: float = 10000.0
    max_notional_usd: float = 100000.0
    min_confidence: float = 0.0  # 0 = disabled
    max_leverage: int = 80
    min_leverage: int = 5
    fail_open: bool = False  # If True, only hard-fail on range violations
    
    @classmethod
    def from_env(cls) -> "GovernorConfig":
        """Load configuration from environment variables"""
        return cls(
            min_order_usd=float(os.getenv("MIN_ORDER_USD", "50")),
            max_position_usd=float(os.getenv("MAX_POSITION_USD", "10000")),
            max_notional_usd=float(os.getenv("MAX_NOTIONAL_USD", "100000")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.0")),
            max_leverage=int(os.getenv("AI_MAX_LEVERAGE", "80")),
            min_leverage=int(os.getenv("AI_MIN_LEVERAGE", "5")),
            fail_open=os.getenv("GOVERNOR_FAIL_OPEN", "false").lower() == "true",
        )


class RiskGovernor:
    """
    Risk Governor - Gate trades before execution
    
    Enforces safety bounds on position size, leverage, and notional exposure.
    Logs all decisions (ACCEPT/REJECT with reason).
    """
    
    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig.from_env()
        logger.info(f"✅ Risk Governor initialized: mode={'fail-open' if self.config.fail_open else 'fail-closed'}")
        logger.info(
            f"   Bounds: size=[${self.config.min_order_usd:.0f}..${self.config.max_position_usd:.0f}], "
            f"leverage=[{self.config.min_leverage}..{self.config.max_leverage}]x, "
            f"notional≤${self.config.max_notional_usd:.0f}"
        )
        if self.config.min_confidence > 0:
            logger.info(f"   Min confidence: {self.config.min_confidence:.2f}")
    
    def evaluate(
        self,
        symbol: str,
        action: str,
        confidence: float,
        position_size_usd: Optional[float] = None,
        leverage: Optional[float] = None,
        risk_budget_usd: Optional[float] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Evaluate if order meets safety policies.
        
        Args:
            symbol: Trading pair
            action: BUY/SELL/CLOSE
            confidence: Signal confidence [0..1]
            position_size_usd: Requested position size (optional, will use defaults)
            leverage: Requested leverage (optional, will use defaults)
            risk_budget_usd: Risk budget for position (optional)
        
        Returns:
            (approved: bool, reason: str, metadata: dict)
            
        Metadata includes:
            - clamped_size_usd: Final size after clamping
            - clamped_leverage: Final leverage after clamping
            - notional_usd: Final notional (size * leverage)
            - policies_checked: List of policies evaluated
        """
        policies_checked = []
        metadata = {}
        
        # P1: Use defaults if not provided
        size_usd = position_size_usd or self.config.max_position_usd * 0.1  # 10% by default
        leverage = leverage or 10.0
        
        # P2: Clamp size to bounds
        original_size = size_usd
        size_usd = max(self.config.min_order_usd, min(size_usd, self.config.max_position_usd))
        if size_usd != original_size:
            policies_checked.append(f"size clamped: ${original_size:.0f} → ${size_usd:.0f}")
        
        # P3: Clamp leverage to bounds
        original_lev = leverage
        leverage = max(self.config.min_leverage, min(leverage, self.config.max_leverage))
        if leverage != original_lev:
            policies_checked.append(f"leverage clamped: {original_lev:.1f}x → {leverage:.1f}x")
        
        # P4: Check notional exposure
        notional_usd = size_usd * leverage
        if notional_usd > self.config.max_notional_usd:
            # Hard fail: notional too large
            reason = (
                f"NOTIONAL_EXCEEDED: ${notional_usd:.0f} > ${self.config.max_notional_usd:.0f} "
                f"(${size_usd:.0f} * {leverage:.1f}x)"
            )
            logger.warning(f"[GOVERNOR] ❌ REJECT {symbol} {action}: {reason}")
            return False, reason, {'policies_checked': policies_checked, 'notional_usd': notional_usd}
        
        policies_checked.append(f"notional: ${notional_usd:.0f} ≤ ${self.config.max_notional_usd:.0f}")
        
        # P5: Confidence floor (if set)
        if self.config.min_confidence > 0 and confidence < self.config.min_confidence:
            reason = f"CONFIDENCE_BELOW_FLOOR: {confidence:.2f} < {self.config.min_confidence:.2f}"
            if self.config.fail_open:
                logger.info(f"[GOVERNOR] ⚠️  SOFT_WARN {symbol} {action}: {reason} (fail-open mode)")
            else:
                logger.warning(f"[GOVERNOR] ❌ REJECT {symbol} {action}: {reason}")
                return False, reason, {'policies_checked': policies_checked}
        
        policies_checked.append(f"confidence: {confidence:.2f} >= {self.config.min_confidence:.2f}")
        
        # P6: Risk budget (if provided)
        if risk_budget_usd and size_usd > risk_budget_usd:
            reason = f"EXCEEDS_RISK_BUDGET: ${size_usd:.0f} > ${risk_budget_usd:.0f}"
            if self.config.fail_open:
                logger.info(f"[GOVERNOR] ⚠️  SOFT_WARN {symbol} {action}: {reason} (fail-open mode)")
            else:
                logger.warning(f"[GOVERNOR] ❌ REJECT {symbol} {action}: {reason}")
                return False, reason, {'policies_checked': policies_checked}
        
        if risk_budget_usd:
            policies_checked.append(f"risk budget: ${size_usd:.0f} ≤ ${risk_budget_usd:.0f}")
        
        # All checks passed
        metadata = {
            'clamped_size_usd': size_usd,
            'clamped_leverage': leverage,
            'notional_usd': notional_usd,
            'policies_checked': policies_checked,
            'confidence': confidence
        }
        
        reason = f"PASS ({len(policies_checked)} policies ok)"
        logger.info(
            f"[GOVERNOR] ✅ ACCEPT {symbol} {action}: ${size_usd:.0f} @ {leverage:.1f}x "
            f"(notional=${notional_usd:.0f}, conf={confidence:.2f})"
        )
        
        return True, reason, metadata


# Global instance (lazy-loaded)
_governor_instance: Optional[RiskGovernor] = None


def get_risk_governor() -> RiskGovernor:
    """Get or create global Risk Governor instance"""
    global _governor_instance
    if _governor_instance is None:
        _governor_instance = RiskGovernor()
    return _governor_instance
