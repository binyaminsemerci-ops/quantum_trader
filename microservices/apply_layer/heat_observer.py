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
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# P2.8A.3: Bounded thread pool for late observer (production-safe)
_late_observer_executor: Optional[ThreadPoolExecutor] = None
_late_observer_inflight: int = 0
_late_observer_inflight_lock = None  # Lazy init for thread safety

# P2.8A.2: Prometheus metrics (fail-open)
try:
    from prometheus_client import Counter
    p28_observed = Counter(
        'p28_observed_total',
        'Apply shadow observability events',
        ['obs_point', 'heat_found']
    )
    p28_heat_reason = Counter(
        'p28_heat_reason_total',
        'Apply heat lookup reasons',
        ['obs_point', 'reason']
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.debug("prometheus_client not available, P2.8A metrics disabled")


def observe(
    redis_client,
    plan_id: str,
    symbol: str,
    apply_decision: str = "",
    obs_point: str = "create_apply_plan",
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
        apply_decision: Apply decision (EXECUTE/SKIP/BLOCKED/UNKNOWN, default: "")
        obs_point: Observation point identifier (default: "create_apply_plan")
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
        # Check deduplication (per observation point)
        dedupe_key = f"quantum:dedupe:p28:{obs_point}:{plan_id}"
        if redis_client.exists(dedupe_key):
            logger.debug(f"{symbol}: Plan {plan_id[:16]} already observed at {obs_point} (dedupe skip)")
            # Increment dedupe metric if available (optional)
            try:
                # Try to increment p28_heat_obs_skipped_dedupe_total if metrics exist
                # This is optional and fails silently if no metrics framework
                pass
            except:
                pass
            return
        
        # Set dedupe key (per observation point)
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
                "obs_point": obs_point,
                "heat_found": str(heat_found),
                "heat_ts_epoch": heat_ts_epoch,
                "heat_level": heat_level,
                "heat_score": heat_score,
                "heat_action": heat_action,
                "heat_out_action": heat_out_action,
                "heat_recommended_partial": heat_recommended_partial,
                "heat_reason": heat_reason,
                "apply_decision": apply_decision or "",
                "debug_json": debug_json
            }
            
            redis_client.xadd(stream_name, observed_fields, maxlen=10000)
            
            logger.debug(f"{symbol}: Observed event emitted (plan={plan_id[:16]}, heat_found={heat_found})")
            
            # P2.8A.2: Increment Prometheus metrics (fail-open)
            if METRICS_AVAILABLE:
                try:
                    p28_observed.labels(
                        obs_point=obs_point,
                        heat_found=str(heat_found)
                    ).inc()
                    p28_heat_reason.labels(
                        obs_point=obs_point,
                        reason=heat_reason
                    ).inc()
                except Exception as e:
                    logger.debug(f"P2.8A.2: metrics increment failed: {e}")
        
        except Exception as e:
            # Stream write error - log and continue
            logger.warning(f"{symbol}: Failed to emit observed event: {e}")
            # Don't raise - fail-open guarantee
    
    except Exception as e:
        # Top-level catch-all - ensure observe() never crashes Apply
        logger.error(f"{symbol}: Unexpected error in heat observer: {e}")
        # Silently continue - fail-open guarantee


def observe_late_async(
    redis_client,
    plan_id: str,
    symbol: str,
    lookup_prefix: str,
    apply_decision: str = "",
    obs_stream: str = "quantum:stream:apply.heat.observed",
    max_wait_ms: int = 2000,
    poll_ms: int = 100,
    dedupe_ttl_sec: int = 600,
    max_debug_chars: int = 400,
    obs_point: str = "publish_plan_post",
    logger=None,
    max_workers: int = 4,
    max_inflight: int = 200
) -> None:
    """
    P2.8A.3: Delayed heat observation after publish_plan (non-blocking, bounded).
    
    Polls for HeatBridge by_plan key for up to max_wait_ms, then emits observed event.
    Uses bounded ThreadPoolExecutor to prevent resource leaks at high throughput.
    
    WHY THIS EXISTS:
    - Observer at create_apply_plan runs BEFORE publish → HeatBridge hasn't written by_plan yet
    - This late observer runs AFTER publish → HeatBridge has time to write by_plan key
    - Result: Better heat coverage without changing execution timing
    
    PRODUCTION SAFETY:
    - Bounded thread pool (max_workers) prevents unbounded thread spawn
    - Fail-open (swallows all exceptions)
    - Non-blocking (returns immediately)
    - Separate obs_point and dedupe key
    
    Args:
        redis_client: Redis connection
        plan_id: Apply plan ID
        symbol: Trading symbol
        lookup_prefix: HeatBridge key prefix (REQUIRED - must match P28_HEAT_LOOKUP_PREFIX)
        apply_decision: Apply decision
        obs_stream: Output observed stream name
        max_wait_ms: Max polling time in milliseconds (default: 2000)
        poll_ms: Polling interval in milliseconds (default: 100)
        dedupe_ttl_sec: Deduplication TTL
        max_debug_chars: Max debug JSON chars
        obs_point: Observation point identifier (default: "publish_plan_post")
        logger: Logger instance
        max_workers: Max thread pool size (default: 4)
        max_inflight: Max concurrent tasks (backpressure, default: 200)
    
    Returns:
        None (spawns background task, never blocks)
    """
    global _late_observer_executor, _late_observer_inflight, _late_observer_inflight_lock
    
    # Lazy init inflight lock (thread-safe)
    if _late_observer_inflight_lock is None:
        import threading
        _late_observer_inflight_lock = threading.Lock()
    
    # Check inflight limit (backpressure - fail-open on saturation)
    with _late_observer_inflight_lock:
        if _late_observer_inflight >= max_inflight:
            log = logger or logging.getLogger(__name__)
            log.debug(f"{symbol}: Late observer inflight limit reached ({_late_observer_inflight}/{max_inflight}), skipping (fail-open)")
            return  # Drop on saturation - better than OOM
        _late_observer_inflight += 1
    
    # Initialize bounded thread pool (singleton, production-safe)
    if _late_observer_executor is None:
        try:
            _late_observer_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="p28a3_late_obs"
            )
            log = logger or logging.getLogger(__name__)
            log.info(f"P2.8A.3: Late observer pool created (max_workers={max_workers})")
        except Exception as e:
            log = logger or logging.getLogger(__name__)
            log.warning(f"P2.8A.3: Failed to create thread pool: {e}")
            return  # Fail-open: don't crash Apply
    
    def _poll_and_observe():
        """Background polling logic."""
        global _late_observer_inflight, _late_observer_inflight_lock
        try:
            log = logger or logging.getLogger(__name__)
            heat_key = f"{lookup_prefix}{plan_id}"
            start_time = time.time()
            max_wait_sec = max_wait_ms / 1000.0
            poll_sec = poll_ms / 1000.0
            redis_errors = 0
            max_redis_errors = 2
            
            # Poll for by_plan key
            heat_found = False
            while (time.time() - start_time) < max_wait_sec:
                try:
                    if redis_client.exists(heat_key):
                        heat_found = True
                        break
                except Exception as e:
                    redis_errors += 1
                    log.warning(f"{symbol}: Redis error in late observer (attempt {redis_errors}/{max_redis_errors}): {e}")
                    if redis_errors >= max_redis_errors:
                        log.warning(f"{symbol}: Late observer aborting after {redis_errors} redis errors")
                        break
                time.sleep(poll_sec)
            
            # Emit observation (will read full heat data if found)
            observe(
                redis_client=redis_client,
                plan_id=plan_id,
                symbol=symbol,
                apply_decision=apply_decision,
                obs_point=obs_point,
                enabled=True,
                stream_name=obs_stream,
                lookup_prefix=lookup_prefix,
                dedupe_ttl_sec=dedupe_ttl_sec,
                max_debug_chars=max_debug_chars
            )
            
            wait_time_ms = int((time.time() - start_time) * 1000)
            if heat_found:
                log.debug(f"{symbol}: Late observer found heat after {wait_time_ms}ms (plan={plan_id[:16]})")
            else:
                reason = "timeout" if redis_errors < max_redis_errors else "redis_error"
                log.debug(f"{symbol}: Late observer no heat after {wait_time_ms}ms (reason={reason}, plan={plan_id[:16]})")
        
        except Exception as e:
            log = logger or logging.getLogger(__name__)
            log.warning(f"{symbol}: Late observer error: {e}")
            # Fail-open: don't crash, just log
        
        finally:
            # Always decrement inflight counter (backpressure cleanup)
            with _late_observer_inflight_lock:
                _late_observer_inflight -= 1
    
    # Submit to bounded thread pool (production-safe)
    try:
        _late_observer_executor.submit(_poll_and_observe)
    except Exception as e:
        log = logger or logging.getLogger(__name__)
        log.warning(f"{symbol}: Failed to submit late observer task: {e}")
        # Fail-open: don't crash Apply


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
