#!/usr/bin/env python3
"""
P3.5 Decision Intelligence Service

Consumes quantum:stream:apply.result and produces rolling-window aggregates
for live "why not trading" analytics.

Features:
- Per-minute bucket tracking (decision counts, reason counts, symbol breakdown)
- Rolling snapshot windows (1m, 5m, 15m, 1h)
- Reliable consumer group processing
- Low CPU consumption via tumbling windows
- Status tracking for monitoring
"""

import json
import logging
import os
import redis
import socket
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import signal
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_SYMBOL_BREAKDOWN = os.getenv("ENABLE_SYMBOL_BREAKDOWN", "true").lower() == "true"

# Stream configuration
STREAM_KEY = "quantum:stream:apply.result"
CONSUMER_GROUP = "p35_decision_intel"
CONSUMER_NAME = f"{socket.gethostname()}-{os.getpid()}"

# Redis key prefixes
BUCKET_PREFIX = "quantum:p35:bucket:"
DECISION_COUNTS_PREFIX = "quantum:p35:decision:counts:"
REASON_TOP_PREFIX = "quantum:p35:reason:top:"
STATUS_KEY = "quantum:p35:status"

# Bucket configuration
BUCKET_EXPIRY = 172800  # 48 hours
SNAPSHOT_WINDOWS = {
    "1m": (1, 60),      # 1 minute window, recompute every 60s
    "5m": (5, 300),     # 5 minute window, recompute every 300s
    "15m": (15, 900),   # 15 minute window, recompute every 900s
    "1h": (60, 3600),   # 1 hour window, recompute every 3600s
}
TOP_REASONS_LIMIT = 50

# Processing configuration
BATCH_SIZE = 100
ACK_INTERVAL = 10  # seconds
SNAPSHOT_COMPUTE_INTERVAL = 60  # seconds (primary snapshot compute cycle)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ============================================================================
# DECISION INTELLIGENCE ENGINE
# ============================================================================

class DecisionIntelligenceService:
    """Processes apply.result stream and generates decision analytics."""
    
    def __init__(self):
        """Initialize service with Redis connection."""
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        
        # Verify Redis connection
        try:
            self.redis.ping()
            logger.info(f"✅ Connected to Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
        
        # State tracking
        self.last_ack_time = time.time()
        self.last_snapshot_compute_time = time.time()
        self.processed_count = 0
        self.pending_ack_ids = []
        self.running = True
        
        # Current bucket (will update as time advances)
        self.current_bucket_key = None
        
        # Initialize consumer group
        self._ensure_consumer_group()
        
        # Initialize status
        self._init_status()
        
        logger.info(f"✅ Service initialized (consumer: {CONSUMER_NAME})")
    
    def _normalize_decision(self, data: dict) -> str:
        """
        Normalize decision field from apply.result event.
        
        Maps:
        - executed=true → EXECUTE
        - executed=false + error present → BLOCKED or SKIP
        - decision field (if present) → normalized value
        
        Returns normalized decision: EXECUTE, SKIP, BLOCKED, ERROR, or UNKNOWN:<raw>
        """
        # Try explicit decision field first
        if "decision" in data:
            raw = str(data["decision"]).strip().upper()
            # Normalize synonyms
            if raw in ["EXECUTE", "EXECUTED", "EXEC"]:
                return "EXECUTE"
            elif raw in ["SKIP", "SKIPPED"]:
                return "SKIP"
            elif raw in ["BLOCKED", "BLOCK"]:
                return "BLOCKED"
            elif raw in ["ERROR", "FAILED", "FAIL"]:
                return "ERROR"
            else:
                # Store unknown decision for debugging
                return f"UNKNOWN:{raw}"
        
        # Fallback: parse from executed boolean
        executed = data.get("executed", "")
        if executed == "true" or executed is True:
            return "EXECUTE"
        elif executed == "false" or executed is False:
            # Check error to determine SKIP vs BLOCKED
            error = self._normalize_reason(data)
            if error in ["none", ""]:
                return "SKIP"  # No error, just skipped
            else:
                return "BLOCKED"  # Has error, was blocked
        
        # Unknown state
        return "UNKNOWN"
    
    def _normalize_reason(self, data: dict) -> str:
        """
        Normalize reason/error field from apply.result event.
        
        Priority:
        1. Top-level 'error' field
        2. details JSON 'error' field
        3. Top-level 'reason' field
        4. Empty string → "none"
        
        Returns normalized reason or "none" if empty.
        """
        # Try top-level error
        if "error" in data and data["error"]:
            return str(data["error"]).strip()
        
        # Try details JSON
        if "details" in data:
            try:
                import json
                details = json.loads(data["details"])
                if "error" in details and details["error"]:
                    return str(details["error"]).strip()
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Try top-level reason
        if "reason" in data and data["reason"]:
            return str(data["reason"]).strip()
        
        # Default: no reason
        return "none"
    
    def _ensure_consumer_group(self):
        """Create consumer group if it doesn't exist."""
        try:
            self.redis.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
            logger.info(f"✅ Created consumer group: {CONSUMER_GROUP}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"✅ Consumer group already exists: {CONSUMER_GROUP}")
            else:
                raise
    
    def _init_status(self):
        """Initialize status tracking."""
        if not self.redis.exists(STATUS_KEY):
            self.redis.hset(STATUS_KEY, mapping={
                "last_id": "0",
                "last_ts": str(int(time.time())),
                "processed_total": "0",
                "pending_estimate": "0",
                "service_start_ts": str(int(time.time())),
                "consumer_name": CONSUMER_NAME,
            })
            logger.info(f"✅ Initialized status key: {STATUS_KEY}")
    
    def _get_current_bucket_key(self, ts: float = None) -> str:
        """Get bucket key for given timestamp (default: now)."""
        if ts is None:
            ts = time.time()
        dt = datetime.fromtimestamp(ts)
        bucket_time = dt.strftime("%Y%m%d%H%M")
        return f"{BUCKET_PREFIX}{bucket_time}"
    
    def _process_message(self, msg_id: str, data: Dict) -> bool:
        """
        Process single apply.result message.
        
        Args:
            msg_id: Redis stream message ID
            data: Message fields
            
        Returns:
            True if processed successfully, False otherwise
        """
        try:
            # Extract and normalize fields
            decision = self._normalize_decision(data)
            reason = self._normalize_reason(data)
            symbol = data.get("symbol", "")
            timestamp_str = data.get("timestamp", "")
            
            # Parse timestamp
            try:
                ts = float(timestamp_str) if timestamp_str else time.time()
            except ValueError:
                ts = time.time()
            
            # Get bucket key for this message's timestamp
            bucket_key = self._get_current_bucket_key(ts)
            
            # Track unknown decisions for debugging
            if decision.startswith("UNKNOWN"):
                # Store raw decision value in ZSET for debugging
                unknown_key = "quantum:p35:unknown_decision:top:5m"
                raw_value = data.get("decision", f"executed={data.get('executed')}")
                self.redis.zincrby(unknown_key, 1, raw_value)
                self.redis.expire(unknown_key, 300)  # 5 minutes TTL
                
                # Log sample (once per minute max)
                if self.processed_count % 100 == 0:
                    logger.warning(
                        f"⚠️  Unknown decision detected: {raw_value} "
                        f"(msg_id: {msg_id}, symbol: {symbol})"
                    )
            
            # Update per-minute bucket
            # - Decision count
            self.redis.hincrby(bucket_key, f"decision:{decision}", 1)
            
            # - Reason count
            self.redis.hincrby(bucket_key, f"reason:{reason}", 1)
            
            # - Symbol-reason breakdown (optional, if enabled)
            if ENABLE_SYMBOL_BREAKDOWN and symbol:
                self.redis.hincrby(
                    bucket_key,
                    f"symbol_reason:{symbol}:{reason}",
                    1
                )
            
            # Set expiry on bucket
            self.redis.expire(bucket_key, BUCKET_EXPIRY)
            
            # Track for ACK
            self.pending_ack_ids.append(msg_id)
            self.processed_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing message {msg_id}: {e}", exc_info=True)
            return False
    
    def _ack_messages(self):
        """ACK all pending messages."""
        if not self.pending_ack_ids:
            return
        
        try:
            for msg_id in self.pending_ack_ids:
                self.redis.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)
            
            count = len(self.pending_ack_ids)
            logger.debug(f"✅ ACKed {count} messages")
            self.pending_ack_ids = []
            
        except Exception as e:
            logger.error(f"❌ Error ACKing messages: {e}")
    
    def _compute_snapshots(self):
        """Recompute rolling window snapshots."""
        try:
            logger.debug("📊 Computing snapshots...")
            
            now = time.time()
            
            for window_name, (window_minutes, _) in SNAPSHOT_WINDOWS.items():
                # Collect all buckets within this window
                decision_counts = {}
                reason_counts = {}
                
                for i in range(window_minutes):
                    check_time = now - (i * 60)
                    bucket_key = self._get_current_bucket_key(check_time)
                    
                    # Get bucket data
                    try:
                        bucket_data = self.redis.hgetall(bucket_key)
                        
                        # Aggregate decisions and reasons
                        for field, count_str in bucket_data.items():
                            try:
                                count = int(count_str)
                            except ValueError:
                                continue
                            
                            if field.startswith("decision:"):
                                decision = field.split(":", 1)[1]
                                decision_counts[decision] = decision_counts.get(decision, 0) + count
                            
                            elif field.startswith("reason:") and not field.startswith("reason_reason:"):
                                reason = field.split(":", 1)[1]
                                reason_counts[reason] = reason_counts.get(reason, 0) + count
                    
                    except Exception as e:
                        logger.debug(f"Note: {bucket_key} not found or error: {e}")
                
                # Write decision counts hash
                decision_key = f"{DECISION_COUNTS_PREFIX}{window_name}"
                if decision_counts:
                    self.redis.delete(decision_key)
                    self.redis.hset(decision_key, mapping=decision_counts)
                    self.redis.expire(decision_key, 86400)  # 24h
                
                # Write top reasons zset
                reason_key = f"{REASON_TOP_PREFIX}{window_name}"
                if reason_counts:
                    self.redis.delete(reason_key)
                    for reason, count in reason_counts.items():
                        self.redis.zadd(reason_key, {reason: count})
                    self.redis.expire(reason_key, 86400)  # 24h
                    
                    # Trim to top N
                    rank_count = self.redis.zcard(reason_key)
                    if rank_count > TOP_REASONS_LIMIT:
                        self.redis.zremrangebyrank(reason_key, 0, rank_count - TOP_REASONS_LIMIT - 1)
            
            logger.debug("✅ Snapshots computed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error computing snapshots: {e}", exc_info=True)
    
    def _update_status(self):
        """Update status key with current metrics."""
        try:
            # Estimate pending messages
            try:
                pending = self.redis.xpending(STREAM_KEY, CONSUMER_GROUP)
                pending_count = pending.get("pending", 0) if isinstance(pending, dict) else 0
            except Exception:
                pending_count = 0
            
            self.redis.hset(STATUS_KEY, mapping={
                "last_id": "",  # Will be updated by consumer
                "last_ts": str(int(time.time())),
                "processed_total": str(self.processed_count),
                "pending_estimate": str(pending_count),
                "consumer_name": CONSUMER_NAME,
            })
        except Exception as e:
            logger.error(f"❌ Error updating status: {e}")
    
    def _check_snapshot_compute(self):
        """Check if it's time to recompute snapshots."""
        now = time.time()
        time_since_compute = now - self.last_snapshot_compute_time
        
        # Find the minimum recompute interval
        min_interval = min(interval for _, interval in SNAPSHOT_WINDOWS.values())
        
        if time_since_compute >= min_interval:
            self._compute_snapshots()
            self.last_snapshot_compute_time = now
    
    def run(self):
        """Main service loop."""
        logger.info("🚀 Starting consumer loop...")
        
        try:
            while self.running:
                try:
                    # Read messages from stream
                    messages = self.redis.xreadgroup(
                        groupname=CONSUMER_GROUP,
                        consumername=CONSUMER_NAME,
                        streams={STREAM_KEY: ">"},
                        count=BATCH_SIZE,
                        block=1000,  # 1 second timeout
                    )
                    
                    if messages:
                        # Process messages
                        for stream_key, msg_list in messages:
                            for msg_id, msg_data in msg_list:
                                self._process_message(msg_id, msg_data)
                    
                    # Check if it's time to ACK
                    now = time.time()
                    if now - self.last_ack_time >= ACK_INTERVAL:
                        self._ack_messages()
                        self.last_ack_time = now
                    
                    # Check if it's time to recompute snapshots
                    self._check_snapshot_compute()
                    
                    # Update status periodically
                    if self.processed_count % 100 == 0 and self.processed_count > 0:
                        self._update_status()
                        logger.info(f"📊 Processed {self.processed_count} messages")
                
                except redis.exceptions.ConnectionError:
                    logger.error("❌ Redis connection lost, retrying...")
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"❌ Error in consumer loop: {e}", exc_info=True)
                    time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("⏹️ Received interrupt signal")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("🛑 Shutting down...")
        
        # Final ACK
        try:
            self._ack_messages()
        except Exception as e:
            logger.error(f"Error during final ACK: {e}")
        
        # Final status update
        try:
            self._update_status()
        except Exception as e:
            logger.error(f"Error during final status update: {e}")
        
        logger.info(f"✅ Shutdown complete. Processed {self.processed_count} total messages.")
        self.running = False

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

def handle_signal(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"📬 Received signal {signum}")
    if service:
        service.shutdown()
    sys.exit(0)

# ============================================================================
# MAIN
# ============================================================================

service = None

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("P3.5 Decision Intelligence Service")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    logger.info(f"  Consumer Group: {CONSUMER_GROUP}")
    logger.info(f"  Consumer Name: {CONSUMER_NAME}")
    logger.info(f"  Symbol Breakdown: {ENABLE_SYMBOL_BREAKDOWN}")
    logger.info(f"  Batch Size: {BATCH_SIZE}")
    logger.info(f"  ACK Interval: {ACK_INTERVAL}s")
    logger.info("=" * 70)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    try:
        service = DecisionIntelligenceService()
        service.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
