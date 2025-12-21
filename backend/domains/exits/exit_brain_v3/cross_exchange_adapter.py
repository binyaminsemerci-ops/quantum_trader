"""
Cross-Exchange Adapter for ExitBrain v3
Phase 4M+ Integration

Connects ExitBrain v3 to Cross-Exchange Intelligence Layer to:
- Read multi-exchange volatility, funding, and divergence data
- Calculate global volatility state
- Provide adaptive multipliers for ATR, TP, and SL
- Handle fail-safe fallback to local calculations
"""

import os
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CrossExchangeState:
    """Global market state from cross-exchange analysis"""
    timestamp: datetime
    exchange_divergence: float  # Price spread across exchanges (0-1)
    funding_delta: float  # Net funding rate change (-0.1 to +0.1)
    volatility_factor: float  # Combined volatility metric (0-5)
    num_exchanges: int  # Number of exchanges contributing data
    data_age_seconds: float  # How old is this data?
    is_stale: bool = False  # True if data > 60 seconds old


@dataclass
class VolatilityAdjustments:
    """Multipliers for ExitBrain calculations"""
    atr_multiplier: float = 1.0  # Multiply base ATR
    tp_multiplier: float = 1.0  # Multiply base TP
    sl_multiplier: float = 1.0  # Multiply base SL
    use_trailing: bool = True  # Enable/disable trailing
    reasoning: str = ""  # Why these adjustments?


class CrossExchangeAdapter:
    """
    Adapter between Cross-Exchange Intelligence Layer and ExitBrain v3
    
    Reads from: quantum:stream:exchange.normalized
    Publishes to: quantum:stream:exitbrain.status
    
    Fail-safe: If cross-exchange data stale > 60s, revert to local mode
    """
    
    def __init__(self, redis_client=None, enabled: bool = True):
        """
        Initialize adapter
        
        Args:
            redis_client: Redis connection (must support XREAD/XREVRANGE)
            enabled: If False, always use local mode
        """
        self.redis = redis_client
        self.enabled = enabled and (redis_client is not None)
        self.use_local_mode = not self.enabled
        
        # Stream names
        self.input_stream = "quantum:stream:exchange.normalized"
        self.status_stream = "quantum:stream:exitbrain.status"
        self.alert_stream = "quantum:stream:exitbrain.alerts"
        
        # State tracking
        self.last_state: Optional[CrossExchangeState] = None
        self.last_update: Optional[datetime] = None
        self.fail_count = 0
        self.total_reads = 0
        
        # Configuration
        self.stale_threshold_seconds = 60.0
        self.max_fail_count = 3  # After 3 fails, switch to local mode
        
        # Limits for safety
        self.min_tp_pct = 0.003  # 0.3% minimum TP
        self.min_sl_pct = 0.0015  # 0.15% minimum SL
        self.max_volatility_factor = 5.0  # Cap extreme volatility
        self.max_divergence = 1.0  # 100% max divergence
        
        logger.info(
            f"[CROSS-EXCHANGE-ADAPTER] Initialized | "
            f"Enabled: {self.enabled} | "
            f"Local Mode: {self.use_local_mode}"
        )
    
    async def get_global_volatility_state(self) -> CrossExchangeState:
        """
        Get current global market state from cross-exchange data
        
        Returns:
            CrossExchangeState with current metrics
        """
        if self.use_local_mode or not self.enabled:
            logger.debug("[CROSS-EXCHANGE-ADAPTER] Using local mode (cross-exchange disabled)")
            return self._get_local_fallback_state()
        
        try:
            # Read latest entry from normalized stream
            entries = await self.redis.xrevrange(
                self.input_stream,
                max='+',
                min='-',
                count=1
            )
            
            if not entries:
                logger.warning("[CROSS-EXCHANGE-ADAPTER] No data in normalized stream, using local mode")
                return await self._handle_fallback("No data in stream")
            
            # Parse entry
            entry_id, data = entries[0]
            timestamp = datetime.fromtimestamp(int(entry_id.split('-')[0]) / 1000, tz=timezone.utc)
            
            # Extract metrics
            state = CrossExchangeState(
                timestamp=timestamp,
                exchange_divergence=float(data.get(b'price_divergence', 0.0)),
                funding_delta=float(data.get(b'funding_delta', 0.0)),
                volatility_factor=float(data.get(b'volatility_factor', 1.0)),
                num_exchanges=int(data.get(b'num_exchanges', 0)),
                data_age_seconds=(datetime.now(timezone.utc) - timestamp).total_seconds()
            )
            
            # Check if stale
            if state.data_age_seconds > self.stale_threshold_seconds:
                state.is_stale = True
                logger.warning(
                    f"[CROSS-EXCHANGE-ADAPTER] Data stale: {state.data_age_seconds:.1f}s old, "
                    f"switching to local mode"
                )
                return await self._handle_fallback(f"Data stale ({state.data_age_seconds:.1f}s)")
            
            # Success - reset fail count
            self.fail_count = 0
            self.last_state = state
            self.last_update = datetime.now(timezone.utc)
            self.total_reads += 1
            
            logger.debug(
                f"[CROSS-EXCHANGE-ADAPTER] ✓ State updated | "
                f"Divergence: {state.exchange_divergence:.4f} | "
                f"Funding: {state.funding_delta:.4f} | "
                f"Volatility: {state.volatility_factor:.2f} | "
                f"Exchanges: {state.num_exchanges} | "
                f"Age: {state.data_age_seconds:.1f}s"
            )
            
            return state
        
        except Exception as e:
            logger.error(f"[CROSS-EXCHANGE-ADAPTER] Error reading stream: {e}")
            return await self._handle_fallback(f"Error: {str(e)}")
    
    def calculate_adjustments(
        self,
        state: CrossExchangeState,
        base_atr: float,
        base_tp: float,
        base_sl: float
    ) -> VolatilityAdjustments:
        """
        Calculate ATR/TP/SL adjustments based on cross-exchange state
        
        Logic:
        - Higher volatility → Wider SL, wider TP
        - Higher divergence → Wider SL (price uncertainty)
        - Positive funding delta → Wider TP (bullish momentum)
        - Negative funding delta → Tighter TP (bearish pressure)
        
        Args:
            state: Current cross-exchange state
            base_atr: Base ATR value
            base_tp: Base TP percentage
            base_sl: Base SL percentage
        
        Returns:
            VolatilityAdjustments with multipliers
        """
        if state.is_stale or self.use_local_mode:
            return VolatilityAdjustments(
                atr_multiplier=1.0,
                tp_multiplier=1.0,
                sl_multiplier=1.0,
                use_trailing=True,
                reasoning="Local mode (cross-exchange data unavailable)"
            )
        
        # Cap extreme values for safety
        vol_factor = min(state.volatility_factor, self.max_volatility_factor)
        divergence = min(state.exchange_divergence, self.max_divergence)
        funding = max(-0.1, min(0.1, state.funding_delta))  # Cap to ±10%
        
        # ATR adjustment: Increase with volatility
        # Formula: atr = base_atr * (1 + vol_factor * 0.6)
        atr_mult = 1.0 + (vol_factor * 0.6)
        
        # TP adjustment: Influenced by funding rate
        # Positive funding → wider TP (longs paying, but momentum is up)
        # Negative funding → tighter TP (shorts paying, momentum uncertain)
        # Formula: tp = base_tp * (1 + funding_delta * 0.8)
        tp_mult = 1.0 + (funding * 0.8)
        
        # SL adjustment: Widen with divergence
        # Higher divergence → more price uncertainty → wider SL
        # Formula: sl = base_sl * (1 + divergence * 0.4)
        sl_mult = 1.0 + (divergence * 0.4)
        
        # Trailing logic: Disable in extreme volatility
        use_trailing = vol_factor < 3.0  # Disable if volatility > 3x
        
        # Ensure minimums
        adjusted_tp = base_tp * tp_mult
        adjusted_sl = base_sl * sl_mult
        
        if adjusted_tp < self.min_tp_pct:
            tp_mult = self.min_tp_pct / base_tp
            logger.warning(f"[CROSS-EXCHANGE-ADAPTER] TP too tight, clamping to {self.min_tp_pct*100:.2f}%")
        
        if adjusted_sl < self.min_sl_pct:
            sl_mult = self.min_sl_pct / base_sl
            logger.warning(f"[CROSS-EXCHANGE-ADAPTER] SL too tight, clamping to {self.min_sl_pct*100:.2f}%")
        
        reasoning = (
            f"Vol:{vol_factor:.2f} Div:{divergence:.3f} Fund:{funding:.4f} | "
            f"ATR×{atr_mult:.2f} TP×{tp_mult:.2f} SL×{sl_mult:.2f}"
        )
        
        logger.info(f"[CROSS-EXCHANGE-ADAPTER] 🎯 Adjustments: {reasoning}")
        
        return VolatilityAdjustments(
            atr_multiplier=atr_mult,
            tp_multiplier=tp_mult,
            sl_multiplier=sl_mult,
            use_trailing=use_trailing,
            reasoning=reasoning
        )
    
    async def publish_status(self, adjustments: VolatilityAdjustments, state: CrossExchangeState):
        """
        Publish current ExitBrain status to Redis stream
        
        Stream: quantum:stream:exitbrain.status
        """
        if not self.redis or not self.enabled:
            return
        
        try:
            status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cross_exchange_active": not self.use_local_mode,
                "data_age_seconds": state.data_age_seconds if state else 0,
                "volatility_factor": state.volatility_factor if state else 0,
                "exchange_divergence": state.exchange_divergence if state else 0,
                "funding_delta": state.funding_delta if state else 0,
                "atr_multiplier": adjustments.atr_multiplier,
                "tp_multiplier": adjustments.tp_multiplier,
                "sl_multiplier": adjustments.sl_multiplier,
                "use_trailing": adjustments.use_trailing,
                "reasoning": adjustments.reasoning,
                "total_reads": self.total_reads,
                "fail_count": self.fail_count
            }
            
            await self.redis.xadd(
                self.status_stream,
                status,
                maxlen=1000  # Keep last 1000 status updates
            )
            
        except Exception as e:
            logger.error(f"[CROSS-EXCHANGE-ADAPTER] Failed to publish status: {e}")
    
    async def _handle_fallback(self, reason: str) -> CrossExchangeState:
        """
        Handle fallback to local mode
        
        Args:
            reason: Why we're falling back
        """
        self.fail_count += 1
        
        # Log alert to Redis
        if self.redis:
            try:
                await self.redis.xadd(
                    self.alert_stream,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "type": "FALLBACK_TO_LOCAL",
                        "reason": reason,
                        "fail_count": self.fail_count
                    },
                    maxlen=500
                )
            except:
                pass  # Best effort
        
        # Switch to local mode if too many failures
        if self.fail_count >= self.max_fail_count:
            self.use_local_mode = True
            logger.error(
                f"[CROSS-EXCHANGE-ADAPTER] ❌ Too many failures ({self.fail_count}), "
                f"permanently switching to local mode"
            )
        
        return self._get_local_fallback_state()
    
    def _get_local_fallback_state(self) -> CrossExchangeState:
        """Return neutral state for local calculations"""
        return CrossExchangeState(
            timestamp=datetime.now(timezone.utc),
            exchange_divergence=0.0,
            funding_delta=0.0,
            volatility_factor=1.0,  # Neutral
            num_exchanges=0,
            data_age_seconds=0.0,
            is_stale=True
        )
    
    def get_stats(self) -> Dict:
        """Get adapter statistics"""
        return {
            "enabled": self.enabled,
            "use_local_mode": self.use_local_mode,
            "total_reads": self.total_reads,
            "fail_count": self.fail_count,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_state": {
                "volatility_factor": self.last_state.volatility_factor if self.last_state else None,
                "exchange_divergence": self.last_state.exchange_divergence if self.last_state else None,
                "funding_delta": self.last_state.funding_delta if self.last_state else None,
                "data_age_seconds": self.last_state.data_age_seconds if self.last_state else None
            } if self.last_state else None
        }


# Global singleton instance
_cross_exchange_adapter: Optional[CrossExchangeAdapter] = None


def get_cross_exchange_adapter(redis_client=None, enabled: bool = None) -> CrossExchangeAdapter:
    """
    Get or create global CrossExchangeAdapter instance
    
    Args:
        redis_client: Redis client (required on first call)
        enabled: Override enable flag (defaults to env var CROSS_EXCHANGE_ENABLED)
    
    Returns:
        CrossExchangeAdapter singleton instance
    """
    global _cross_exchange_adapter
    
    if _cross_exchange_adapter is None:
        if enabled is None:
            enabled = os.getenv("CROSS_EXCHANGE_ENABLED", "true").lower() == "true"
        
        _cross_exchange_adapter = CrossExchangeAdapter(
            redis_client=redis_client,
            enabled=enabled
        )
        
        logger.info(
            f"[CROSS-EXCHANGE-ADAPTER] Global adapter initialized | "
            f"Enabled: {enabled}"
        )
    
    return _cross_exchange_adapter
