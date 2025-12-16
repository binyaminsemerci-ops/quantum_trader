"""
Execution Service - Minimal Risk Stub

⚠️ TEMPORARY IMPLEMENTATION ⚠️

This is NOT production risk management.
This is a STUB to allow execution service to deploy independently.

TODO: Replace with proper Risk-Safety Service integration when available.

Design Decision:
- Allow most trades through with basic sanity checks
- Log all validations for audit
- Fail-safe: reject on doubt
"""
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskStub:
    """
    Minimal risk validation stub.
    
    This class provides BASIC sanity checks only:
    1. Symbol whitelist
    2. Max position size
    3. Leverage limits
    4. (Future: balance check)
    
    NOT A REPLACEMENT for proper risk management.
    """
    
    def __init__(
        self,
        allowed_symbols: Optional[list[str]] = None,
        max_position_usd: float = 1000.0,
        max_leverage: int = 10,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.allowed_symbols = set(allowed_symbols or [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT"
        ])
        self.max_position_usd = max_position_usd
        self.max_leverage = max_leverage
        
        logger.info(
            f"[RISK-STUB] Initialized: "
            f"symbols={len(self.allowed_symbols)}, "
            f"max_position=${max_position_usd}, "
            f"max_leverage={max_leverage}x, "
            f"enabled={enabled}"
        )
    
    async def validate_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        leverage: int,
        price_estimate: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Validate trade parameters.
        
        Returns:
            {
                "allowed": bool,
                "reason": Optional[str],
                "checks": {
                    "symbol_allowed": bool,
                    "size_ok": bool,
                    "leverage_ok": bool
                }
            }
        """
        if not self.enabled:
            logger.warning("[RISK-STUB] Risk validation DISABLED - allowing all trades")
            return {
                "allowed": True,
                "reason": "Risk stub disabled",
                "checks": {}
            }
        
        checks = {}
        rejection_reasons = []
        
        # 1. Symbol whitelist
        symbol_allowed = symbol in self.allowed_symbols
        checks["symbol_allowed"] = symbol_allowed
        if not symbol_allowed:
            rejection_reasons.append(f"Symbol {symbol} not in whitelist")
        
        # 2. Position size check
        # NOTE: `size` parameter is expected to be position value in USD
        position_value_usd = size
        size_ok = position_value_usd <= self.max_position_usd
        checks["size_ok"] = size_ok
        checks["position_value_usd"] = position_value_usd
        
        if not size_ok:
            rejection_reasons.append(
                f"Position size ${position_value_usd:.2f} exceeds max ${self.max_position_usd}"
            )
        
        # 3. Leverage check
        leverage_ok = leverage <= self.max_leverage
        checks["leverage_ok"] = leverage_ok
        if not leverage_ok:
            rejection_reasons.append(
                f"Leverage {leverage}x exceeds max {self.max_leverage}x"
            )
        
        # Decision
        allowed = all([symbol_allowed, size_ok, leverage_ok])
        reason = None if allowed else "; ".join(rejection_reasons)
        
        # Log validation
        log_level = logging.INFO if allowed else logging.WARNING
        logger.log(
            log_level,
            f"[RISK-STUB] Validation: symbol={symbol}, side={side}, "
            f"size={size}, leverage={leverage}x, "
            f"value_usd={position_value_usd}, "
            f"allowed={allowed}, reason={reason}"
        )
        
        return {
            "allowed": allowed,
            "reason": reason,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def add_symbol(self, symbol: str):
        """Add symbol to whitelist (runtime config)"""
        self.allowed_symbols.add(symbol)
        logger.info(f"[RISK-STUB] Added symbol: {symbol}")
    
    def remove_symbol(self, symbol: str):
        """Remove symbol from whitelist"""
        self.allowed_symbols.discard(symbol)
        logger.info(f"[RISK-STUB] Removed symbol: {symbol}")
    
    def update_limits(self, max_position_usd: Optional[float] = None, max_leverage: Optional[int] = None):
        """Update risk limits at runtime"""
        if max_position_usd:
            self.max_position_usd = max_position_usd
            logger.info(f"[RISK-STUB] Updated max_position: ${max_position_usd}")
        
        if max_leverage:
            self.max_leverage = max_leverage
            logger.info(f"[RISK-STUB] Updated max_leverage: {max_leverage}x")
