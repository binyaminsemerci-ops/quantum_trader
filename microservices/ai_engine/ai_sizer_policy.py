"""
AI Sizing Policy Bridge - BRIDGE-PATCH v1.0

Bridges AI-engine sizing decisions to execution service.
Supports shadow mode (no effect) and live mode (injected into trade.intent).

Design:
- Shadow mode: AI computes size/leverage but fields stored as ai_size_usd, ai_leverage, ai_harvest_policy
- Live mode: Fields copied to position_size_usd, leverage, harvest_policy before publishing
- All fields clamped by Risk Governor in execution service (fail-closed)
"""

import os
import logging
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("ai-sizer-policy")


class HarvestMode(Enum):
    """Exit strategy modes"""
    SCALPER = "scalper"  # Tight trailing, short max_time
    SWING = "swing"  # Wider trailing, medium max_time
    TREND_RUNNER = "trend_runner"  # Aggressive trailing, long max_time


@dataclass
class SizingConfig:
    """AI Sizing Configuration"""
    ai_sizing_mode: str = "shadow"  # shadow | live
    ai_max_leverage: int = 80
    ai_min_leverage: int = 5
    max_position_usd: float = 10000.0
    max_notional_usd: float = 100000.0
    min_order_usd: float = 50.0
    
    @classmethod
    def from_env(cls) -> "SizingConfig":
        """Load configuration from environment variables"""
        return cls(
            ai_sizing_mode=os.getenv("AI_SIZING_MODE", "shadow"),
            ai_max_leverage=int(os.getenv("AI_MAX_LEVERAGE", "80")),
            ai_min_leverage=int(os.getenv("AI_MIN_LEVERAGE", "5")),
            max_position_usd=float(os.getenv("MAX_POSITION_USD", "10000")),
            max_notional_usd=float(os.getenv("MAX_NOTIONAL_USD", "100000")),
            min_order_usd=float(os.getenv("MIN_ORDER_USD", "50")),
        )


class AISizerPolicy:
    """
    AI Sizing Policy - Coordinates size/leverage/harvest decisions
    
    Responsibilities:
    1. Compute desired position_size_usd, leverage, harvest_policy from AI models
    2. Format payload according to AI_SIZING_MODE (shadow vs live)
    3. Inject into trade.intent payload for execution service
    
    In shadow mode:
    - Fields stored as ai_size_usd, ai_leverage, ai_harvest_policy
    - Execution service logs what it WOULD do but doesn't execute
    
    In live mode:
    - Fields copied to position_size_usd, leverage, harvest_policy
    - Execution service executes with AI sizing
    
    All actual size/leverage bounds enforced by Risk Governor in execution service
    """
    
    def __init__(self, config: Optional[SizingConfig] = None):
        self.config = config or SizingConfig.from_env()
        logger.info(f"✅ AI Sizer Policy initialized: mode={self.config.ai_sizing_mode}")
        logger.info(
            f"   Bounds: leverage=[{self.config.ai_min_leverage}..{self.config.ai_max_leverage}]x, "
            f"position_usd=[{self.config.min_order_usd}..{self.config.max_position_usd}]"
        )
    
    def compute_size_and_leverage(
        self,
        signal_confidence: float,
        volatility_factor: float = 1.0,
        account_equity: float = 10000.0
    ) -> Tuple[float, float, Dict]:
        """
        Compute AI-recommended position size, leverage, and harvest policy.
        
        Args:
            signal_confidence: Signal confidence [0..1]
            volatility_factor: Market volatility adjustment [0.5..2.0]
            account_equity: Current account equity
        
        Returns:
            (size_usd, leverage, harvest_policy_dict)
        
        Formula:
        - Base allocation: 1% of account per signal confidence
        - Volatility adjustment: scale by volatility_factor
        - Leverage: confidence-based (higher confidence = higher leverage)
        - Policy: confidence-based exit mode selection
        """
        # Size: 0.5% to 2% of account based on confidence
        base_pct = 0.005 + (signal_confidence * 0.015)  # 0.5% + conf*1.5%
        size_usd = account_equity * base_pct * volatility_factor
        
        # Clamp to safety bounds
        size_usd = max(self.config.min_order_usd, 
                      min(size_usd, self.config.max_position_usd))
        
        # Leverage: 5x (low conf) to 80x (high conf)
        leverage_range = self.config.ai_max_leverage - self.config.ai_min_leverage
        leverage = self.config.ai_min_leverage + (signal_confidence * leverage_range)
        leverage = max(self.config.ai_min_leverage,
                      min(leverage, self.config.ai_max_leverage))
        
        # Harvest policy: based on confidence
        if signal_confidence < 0.6:
            policy_mode = HarvestMode.SCALPER  # Low conf = take profits fast
            trail_pct = 0.5
            max_time_sec = 1800  # 30 min max
        elif signal_confidence < 0.8:
            policy_mode = HarvestMode.SWING  # Medium conf = balanced
            trail_pct = 1.0
            max_time_sec = 3600  # 1 hour max
        else:
            policy_mode = HarvestMode.TREND_RUNNER  # High conf = let it run
            trail_pct = 2.0
            max_time_sec = 7200  # 2 hours max
        
        harvest_policy = {
            "mode": policy_mode.value,
            "trail_pct": trail_pct,
            "max_time_sec": max_time_sec,
            "partial_close_pct": 0.5 if signal_confidence > 0.75 else 0.0
        }
        
        return float(leverage), size_usd, harvest_policy
    
    def inject_into_payload(
        self,
        trade_intent_payload: Dict,
        signal_confidence: float,
        volatility_factor: float = 1.0,
        account_equity: float = 10000.0
    ) -> Dict:
        """
        Inject AI sizing fields into trade.intent payload.
        
        Args:
            trade_intent_payload: Existing trade.intent dict
            signal_confidence: Signal confidence for sizing
            volatility_factor: Market volatility adjustment
            account_equity: Current account equity
        
        Returns:
            Modified trade_intent_payload with AI fields injected
        
        Behavior depends on AI_SIZING_MODE:
        - shadow: Fields stored as ai_* (no execution effect)
        - live: Fields replace position_size_usd/leverage (execution applies)
        """
        # Compute AI sizing
        ai_leverage, ai_size_usd, ai_policy = self.compute_size_and_leverage(
            signal_confidence, volatility_factor, account_equity
        )
        
        # Always inject as ai_* fields (for audit/monitoring)
        trade_intent_payload["ai_leverage"] = ai_leverage
        trade_intent_payload["ai_size_usd"] = ai_size_usd
        trade_intent_payload["ai_harvest_policy"] = ai_policy
        
        # In live mode: copy to primary fields (execution uses these)
        if self.config.ai_sizing_mode == "live":
            trade_intent_payload["leverage"] = ai_leverage
            trade_intent_payload["position_size_usd"] = ai_size_usd
            trade_intent_payload["harvest_policy"] = ai_policy
            logger.info(
                f"[AI-SIZER] LIVE: {trade_intent_payload.get('symbol', 'N/A')} → "
                f"${ai_size_usd:.0f} @ {ai_leverage:.1f}x | policy={ai_policy['mode']}"
            )
        else:
            # Shadow mode: log what we WOULD do
            logger.debug(
                f"[AI-SIZER] SHADOW: {trade_intent_payload.get('symbol', 'N/A')} → "
                f"${ai_size_usd:.0f} @ {ai_leverage:.1f}x (NOT INJECTED)"
            )
        
        return trade_intent_payload


# Global instance (lazy-loaded)
_sizer_instance: Optional[AISizerPolicy] = None


def get_ai_sizer() -> AISizerPolicy:
    """Get or create global AI sizer instance"""
    global _sizer_instance
    if _sizer_instance is None:
        _sizer_instance = AISizerPolicy()
    return _sizer_instance
