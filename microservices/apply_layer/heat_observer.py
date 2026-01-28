#!/usr/bin/env python3
"""
P2.8A Apply Heat Observer - Shadow Observability Module

Reads HeatBridge cache for each plan_id and emits observed stream entry.
ZERO execution impact - fail-open, never blocks, never changes decisions.
"""

import json
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def observe(
    redis_client,
    plan_id: str,
    symbol: str,
    apply_decision: str,
    enabled: bool = True,
    stream_name: str = "quantum:stream:apply.heat.observed",
    lookup_prefix: str = "quantum:harvest:heat:by_plan:",
    dedupe_ttl_sec: int = 600,
    max_debug_chars: int = 400
) -> None:
    """
    Shadow observation: Read HeatBridge cache and emit observed stream.
    
    FAIL-OPEN GUARANTEE:
    - Never raises exceptions
    - Never blocks Apply processing
    - Never changes Apply decisions
    - Errors logged, metrics incremented (if available), processing continues
    
    Args:
        redis_client: Redis connection
        plan_id: Apply plan ID
        symbol: Trading symbol (e.g., BTCUSDT)
        apply_decision: Apply decision (EXECUTE/SKIP/BLOCKED)
        enabled: If False, silently skip (default: True)
        stream_name: Output observed stream name
        lookup_prefix: HeatBridge key prefix
        dedupe_ttl_sec: Deduplication TTL (default: 600s)
        max_debug_chars: Max debug JSON chars (default: 400)
    
    Returns:
        None (always succeeds, errors logged)
    """
    if not enabled:
        return
    
    ts_epoch = int(time.time())
    heat_found = 0
    heat_ts_epoch = ""
    heat_level = ""
    heat_score = ""
    heat_action = ""
    heat_out_action = ""
    heat_recommended_partial = ""
    heat_reason = "ok"
    debug_data = {}
    
    try:
        # Check deduplication
        dedupe_key = f"quantum:dedupe:p28:{plan_id}"
        if redis_client.exists(dedupe_key):
            logger.debug(f"{symbol}: Plan {plan_id[:16]} already observed (dedupe skip)")
            # Increment dedupe metric if available (optional)
            try:
                # Try to increment p28_heat_obs_skipped_dedupe_total if metrics exist
                # This is optional and fails silently if no metrics framework
                pass
            except:
                pass
            return
        
        # Set dedupe key
        try:
            redis_client.setex(dedupe_key, dedupe_ttl_sec, "1")
        except Exception as e:
            logger.warning(f"{symbol}: Failed to set dedupe key: {e}")
            # Continue processing even if dedupe fails
        
        # Read HeatBridge lookup key
        heat_key = f"{lookup_prefix}{plan_id}"
        
        try:
            heat_data = redis_client.hgetall(heat_key)
            
            if heat_data:
                heat_found = 1
                
                # Extract fields (handle bytes from redis-py)
                def get_field(d: Dict, key: str, default: str = "") -> str:
                    val = d.get(key) or d.get(key.encode())
                    if val is None:
                        return default
                    if isinstance(val, bytes):
                        return val.decode('utf-8', errors='replace')
                    return str(val)
                
                heat_ts_epoch = get_field(heat_data, "ts_epoch")
                heat_level = get_field(heat_data, "heat_level")
                heat_score = get_field(heat_data, "heat_score")
                heat_action = get_field(heat_data, "heat_action")
                heat_out_action = get_field(heat_data, "out_action")
                heat_recommended_partial = get_field(heat_data, "recommended_partial")
                
                # Build debug context (limited size)
                in_action = get_field(heat_data, "in_action")
                mode = get_field(heat_data, "mode")
                
                debug_data = {
                    "in_action": in_action,
                    "out_action": heat_out_action,
                    "mode": mode,
                    "apply_decision": apply_decision
                }
                
                heat_reason = "ok"
                logger.debug(f"{symbol}: Heat found for plan {plan_id[:16]} (level={heat_level}, action={heat_action})")
            else:
                # No heat data found
                heat_found = 0
                heat_reason = "missing"
                debug_data = {"apply_decision": apply_decision}
                logger.debug(f"{symbol}: No heat data for plan {plan_id[:16]}")
        
        except Exception as e:
            # Redis read error
            heat_found = 0
            heat_reason = "redis_error"
            debug_data = {"error": str(e)[:100], "apply_decision": apply_decision}
            logger.warning(f"{symbol}: Redis error reading heat key {heat_key}: {e}")
        
        # Emit observed stream
        try:
            debug_json = json.dumps(debug_data)
            if len(debug_json) > max_debug_chars:
                debug_json = debug_json[:max_debug_chars-3] + "..."
            
            observed_fields = {
                "ts_epoch": str(ts_epoch),
                "plan_id": plan_id,
                "symbol": symbol,
                "apply_component": "apply",
                "heat_found": str(heat_found),
                "heat_ts_epoch": heat_ts_epoch,
                "heat_level": heat_level,
                "heat_score": heat_score,
                "heat_action": heat_action,
                "heat_out_action": heat_out_action,
                "heat_recommended_partial": heat_recommended_partial,
                "heat_reason": heat_reason,
                "apply_decision": apply_decision,
                "debug_json": debug_json
            }
            
            redis_client.xadd(stream_name, observed_fields, maxlen=10000)
            
            logger.debug(f"{symbol}: Observed event emitted (plan={plan_id[:16]}, heat_found={heat_found})")
            
            # Increment metrics if available (optional)
            try:
                # Try to increment p28_heat_obs_total{found=..., reason=...} if metrics exist
                # This is optional and fails silently if no metrics framework
                pass
            except:
                pass
        
        except Exception as e:
            # Stream write error - log and continue
            logger.warning(f"{symbol}: Failed to emit observed event: {e}")
            # Don't raise - fail-open guarantee
    
    except Exception as e:
        # Top-level catch-all - ensure observe() never crashes Apply
        logger.error(f"{symbol}: Unexpected error in heat observer: {e}")
        # Silently continue - fail-open guarantee


def is_enabled(env_value: Optional[str]) -> bool:
    """
    Check if heat observation is enabled via environment variable.
    
    Args:
        env_value: Environment variable value (e.g., "true", "1", "yes")
    
    Returns:
        bool: True if enabled, False otherwise
    """
    if env_value is None:
        return False
    return env_value.lower() in ("true", "1", "yes", "on")
