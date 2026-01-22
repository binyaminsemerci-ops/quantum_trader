#!/usr/bin/env python3
"""
Core Safety Kernel for Quantum Trader
======================================

Provides surgical safety layer at trade.intent publish boundary:
- Global and per-symbol rate limiting
- Circuit breaker SAFE MODE
- Fault event stream for observability

Design: Minimal code, fail-safe, does NOT touch strategy math.
"""

import os
import time
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SafetyKernel:
    """
    Core Safety Kernel - Last line of defense before trade.intent publish.
    
    Enforces:
    1. SAFE MODE flag check (manual override)
    2. Global rate limit (max intents per window)
    3. Per-symbol rate limit (max intents per symbol per window)
    4. Circuit breaker trips on violation → sets SAFE MODE + emits fault
    """
    
    def __init__(self, redis_client, config: Optional[Dict[str, Any]] = None):
        self.redis = redis_client
        
        # Load configuration from env with sane defaults
        self.window_sec = int(os.getenv("SAFETY_WINDOW_SEC", "10"))
        self.global_max = int(os.getenv("SAFETY_GLOBAL_MAX_INTENTS_PER_WINDOW", "20"))
        self.symbol_max = int(os.getenv("SAFETY_SYMBOL_MAX_INTENTS_PER_WINDOW", "5"))
        self.fault_cooldown_sec = int(os.getenv("SAFETY_FAULT_COOLDOWN_SEC", "300"))
        self.safe_mode_default = int(os.getenv("SAFETY_SAFE_MODE_DEFAULT", "0"))
        
        # Redis keys
        self.safe_mode_key = "quantum:safety:safe_mode"
        self.fault_stream = "quantum:stream:safety.fault"
        
        # Get router PID for fault events
        self.router_pid = os.getpid()
        
        logger.info(
            f"[SAFETY] Safety Kernel initialized | "
            f"window={self.window_sec}s global_max={self.global_max} "
            f"symbol_max={self.symbol_max} cooldown={self.fault_cooldown_sec}s"
        )
    
    def should_publish_intent(
        self, 
        symbol: str, 
        side: str,
        correlation_id: str = "",
        trace_id: str = ""
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Gate function: Check if trade.intent should be published.
        
        Returns:
            (allowed: bool, reason: str, meta: dict)
            
        If allowed=False, caller MUST NOT publish and should log reason.
        """
        try:
            # Check 1: SAFE MODE flag
            safe_mode = self.redis.get(self.safe_mode_key)
            if safe_mode and safe_mode.decode('utf-8') == "1":
                return False, "SAFE_MODE_ENABLED", {
                    "symbol": symbol,
                    "side": side,
                    "safe_mode": True
                }
            
            # Calculate current time bucket (10-second window)
            now = time.time()
            epoch_bucket = int(now // self.window_sec)
            
            # Check 2: Global rate limit
            global_key = f"quantum:safety:rate:global:{epoch_bucket}"
            global_count = self._safe_incr(global_key, self.window_sec + 2)
            
            if global_count > self.global_max:
                self._trip_circuit_breaker(
                    reason="GLOBAL_RATE_EXCEEDED",
                    symbol=symbol,
                    side=side,
                    global_count=global_count,
                    symbol_count=0,
                    correlation_id=correlation_id,
                    trace_id=trace_id
                )
                return False, "GLOBAL_RATE_EXCEEDED", {
                    "symbol": symbol,
                    "side": side,
                    "global_count": global_count,
                    "global_max": self.global_max,
                    "window_sec": self.window_sec,
                    "tripped": True
                }
            
            # Check 3: Per-symbol rate limit
            symbol_key = f"quantum:safety:rate:symbol:{symbol}:{epoch_bucket}"
            symbol_count = self._safe_incr(symbol_key, self.window_sec + 2)
            
            if symbol_count > self.symbol_max:
                self._trip_circuit_breaker(
                    reason="SYMBOL_RATE_EXCEEDED",
                    symbol=symbol,
                    side=side,
                    global_count=global_count,
                    symbol_count=symbol_count,
                    correlation_id=correlation_id,
                    trace_id=trace_id
                )
                return False, "SYMBOL_RATE_EXCEEDED", {
                    "symbol": symbol,
                    "side": side,
                    "symbol_count": symbol_count,
                    "symbol_max": self.symbol_max,
                    "window_sec": self.window_sec,
                    "tripped": True
                }
            
            # All checks passed
            return True, "OK", {
                "symbol": symbol,
                "side": side,
                "global_count": global_count,
                "symbol_count": symbol_count,
                "window_sec": self.window_sec
            }
            
        except Exception as e:
            # FAIL OPEN: On Redis errors, allow publish but log error
            logger.error(f"[SAFETY] Safety check failed (FAIL-OPEN): {e}")
            return True, "FAIL_OPEN_ERROR", {
                "symbol": symbol,
                "side": side,
                "error": str(e)
            }
    
    def _safe_incr(self, key: str, ttl_sec: int) -> int:
        """Atomically increment counter with TTL, return new value."""
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl_sec)
        result = pipe.execute()
        return int(result[0])
    
    def _trip_circuit_breaker(
        self,
        reason: str,
        symbol: str,
        side: str,
        global_count: int,
        symbol_count: int,
        correlation_id: str = "",
        trace_id: str = ""
    ):
        """
        Trip circuit breaker: Enable SAFE MODE and emit fault event.
        
        This is called when rate limits are exceeded.
        SAFE MODE will auto-expire after cooldown period.
        """
        try:
            # Set SAFE MODE flag with expiration
            self.redis.set(
                self.safe_mode_key,
                "1",
                ex=self.fault_cooldown_sec
            )
            
            # Emit fault event to stream
            fault_event = {
                "timestamp": str(int(time.time())),
                "reason": reason,
                "symbol": symbol,
                "side": side,
                "global_count": str(global_count),
                "symbol_count": str(symbol_count),
                "global_max": str(self.global_max),
                "symbol_max": str(self.symbol_max),
                "window_sec": str(self.window_sec),
                "cooldown_sec": str(self.fault_cooldown_sec),
                "router_pid": str(self.router_pid),
                "correlation_id": correlation_id,
                "trace_id": trace_id
            }
            
            self.redis.xadd(
                self.fault_stream,
                fault_event,
                maxlen=1000  # Keep last 1000 fault events
            )
            
            logger.error(
                f"[SAFETY] 🚨 CIRCUIT BREAKER TRIPPED | "
                f"reason={reason} symbol={symbol} side={side} | "
                f"global={global_count}/{self.global_max} "
                f"symbol={symbol_count}/{self.symbol_max} | "
                f"SAFE_MODE=ON for {self.fault_cooldown_sec}s | "
                f"corr={correlation_id}"
            )
            
        except Exception as e:
            logger.error(f"[SAFETY] Failed to trip circuit breaker: {e}")


def create_safety_kernel(redis_client, config: Optional[Dict[str, Any]] = None) -> SafetyKernel:
    """Factory function to create SafetyKernel instance."""
    return SafetyKernel(redis_client, config)
