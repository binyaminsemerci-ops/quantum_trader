#!/usr/bin/env python3
"""
P2.6 Portfolio Heat Gate - Hedge Fund OS Edition
=================================================

Portfolio-level safety gate between Harvest Kernel and Apply Layer.
Prevents premature FULL_CLOSE decisions based on portfolio-wide risk.

Core Function:
- Reads harvest.proposal stream
- Calculates Portfolio Heat = Σ(|notional| * sigma) / equity
- Downgrades FULL_CLOSE based on heat buckets:
  * COLD (< 0.25): FULL_CLOSE → PARTIAL_25
  * WARM (0.25-0.65): FULL_CLOSE → PARTIAL_75
  * HOT (≥ 0.65): FULL_CLOSE allowed
- Writes to harvest.calibrated stream

Invariants:
- FAIL-CLOSED: missing data → downgrade to PARTIAL_25
- MONOTONIC: only reduces aggressiveness, never increases
- AUDIT: all decisions traceable

Author: Quantum Trader Team - Hedge Fund OS
Date: 2026-01-26
"""
import os
import sys
import time
import json
import socket
import logging
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed")
    sys.exit(1)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("WARN: prometheus_client not available, metrics disabled")

# Logging
logging.basicConfig(
    level=os.getenv("P26_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] [P2.6-HEAT-GATE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """P2.6 Configuration"""
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Mode: shadow (log only) or enforce (modify proposals)
    MODE: str = os.getenv("P26_MODE", "shadow")
    
    # Heat thresholds
    HEAT_MIN: float = float(os.getenv("HEAT_MIN", "0.25"))
    HEAT_MAX: float = float(os.getenv("HEAT_MAX", "0.65"))
    
    # Polling
    POLL_SEC: int = int(os.getenv("P26_POLL_SEC", "2"))
    
    # Metrics
    METRICS_PORT: int = int(os.getenv("P26_METRICS_PORT", "8056"))
    
    # Streams
    INPUT_STREAM: str = "quantum:stream:harvest.proposal"
    OUTPUT_STREAM: str = "quantum:stream:harvest.calibrated"
    CONSUMER_GROUP: str = "p26_heat_gate"
    
    # Position data sources
    POSITION_SNAPSHOT_STREAM: str = "quantum:stream:position.snapshot"
    PORTFOLIO_STATE_KEY: str = "quantum:state:portfolio"


class PortfolioHeatGate:
    """Portfolio Heat Gate - P2.6"""
    
    def __init__(self, config: Config):
        self.config = config
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=False
        )
        self.consumer_name = f"{socket.gethostname()}_{os.getpid()}"
        
        # Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.metric_heat = Gauge("p26_heat_value", "Current portfolio heat value")
            self.metric_bucket = Gauge("p26_bucket", "Heat bucket state", ["state"])
            self.metric_downgrades = Counter("p26_actions_downgraded_total", "Total actions downgraded", ["from_action", "to_action", "reason"])
            self.metric_lag = Histogram("p26_stream_lag_ms", "Stream processing lag in ms")
            self.metric_processed = Counter("p26_proposals_processed_total", "Total proposals processed")
            self.metric_failures = Counter("p26_failures_total", "Total processing failures", ["reason"])
            self.metric_hash_writes = Counter("p26_hash_writes_total", "Total hash writes to proposal keys")
            self.metric_hash_failures = Counter("p26_hash_write_fail_total", "Total hash write failures")
            self.metric_enforce_mode = Gauge("p26_enforce_mode", "Enforce mode active (1=enforce, 0=shadow)")
        
        self._setup()
    
    def _setup(self):
        """Setup consumer group"""
        try:
            self.redis.xgroup_create(
                self.config.INPUT_STREAM,
                self.config.CONSUMER_GROUP,
                id="0",
                mkstream=True
            )
            logger.info(f"✅ Created consumer group: {self.config.CONSUMER_GROUP}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group: {e}")
            else:
                logger.info(f"✅ Consumer group exists: {self.config.CONSUMER_GROUP}")
    
    def _get_portfolio_heat(self) -> Tuple[float, str, Dict]:
        """
        Calculate portfolio heat from position snapshots and portfolio state.
        
        Returns:
            (heat_value, heat_bucket, context_dict)
        
        Formula:
            PortfolioHeat = Σ(|position_notional| * sigma) / equity_usd
        
        Buckets:
            COLD: heat < HEAT_MIN
            WARM: HEAT_MIN ≤ heat < HEAT_MAX
            HOT: heat ≥ HEAT_MAX
        """
        try:
            # Get equity from portfolio state
            portfolio_state = self.redis.hgetall(self.config.PORTFOLIO_STATE_KEY)
            
            if not portfolio_state:
                logger.warning("Portfolio state missing - FAIL-CLOSED")
                if PROMETHEUS_AVAILABLE:
                    self.metric_failures.labels(reason="portfolio_state_missing").inc()
                return 0.0, "COLD", {"error": "portfolio_state_missing", "fail_closed": True}
            
            equity_usd = float(portfolio_state.get(b"equity_usd", 0) or 0)
            if equity_usd <= 0:
                logger.warning(f"Invalid equity: {equity_usd} - FAIL-CLOSED")
                if PROMETHEUS_AVAILABLE:
                    self.metric_failures.labels(reason="invalid_equity").inc()
                return 0.0, "COLD", {"error": "invalid_equity", "equity": equity_usd, "fail_closed": True}
            
            # Get latest position snapshots
            # Read last 50 entries from position.snapshot stream
            snapshots = self.redis.xrevrange(
                self.config.POSITION_SNAPSHOT_STREAM,
                "+", "-",
                count=50
            )
            
            # Build position map: symbol → (notional, sigma)
            positions = {}
            for stream_id, data in snapshots:
                symbol = data.get(b"symbol", b"").decode()
                if not symbol or symbol in positions:
                    continue  # Take latest per symbol
                
                try:
                    notional = float(data.get(b"position_notional_usd", 0) or 0)
                    sigma = float(data.get(b"sigma", 0) or data.get(b"volatility", 0) or 0)
                    
                    if sigma > 0:  # Only include if we have volatility
                        positions[symbol] = {
                            "notional": abs(notional),
                            "sigma": sigma,
                            "risk_contribution": abs(notional) * sigma
                        }
                except (ValueError, TypeError) as e:
                    logger.debug(f"Skip {symbol}: invalid data - {e}")
                    continue
            
            # Calculate total risk
            total_risk = sum(p["risk_contribution"] for p in positions.values())
            heat = total_risk / max(equity_usd, 1.0)
            
            # Determine bucket
            if heat < self.config.HEAT_MIN:
                bucket = "COLD"
            elif heat < self.config.HEAT_MAX:
                bucket = "WARM"
            else:
                bucket = "HOT"
            
            context = {
                "equity_usd": equity_usd,
                "total_risk": total_risk,
                "positions_count": len(positions),
                "heat": heat,
                "bucket": bucket,
                "positions": positions
            }
            
            logger.debug(f"Portfolio Heat: {heat:.4f} ({bucket}) - {len(positions)} positions, equity=${equity_usd:.2f}")
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                self.metric_heat.set(heat)
                for b in ["COLD", "WARM", "HOT"]:
                    self.metric_bucket.labels(state=b).set(1 if b == bucket else 0)
            
            return heat, bucket, context
            
        except Exception as e:
            logger.error(f"Failed to calculate portfolio heat: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE:
                self.metric_failures.labels(reason="heat_calculation_error").inc()
            return 0.0, "COLD", {"error": str(e), "fail_closed": True}
    
    def _apply_gating_rule(self, original_action: str, heat_bucket: str, heat_value: float) -> Tuple[str, str]:
        """
        Apply portfolio heat gating rules.
        
        Rules:
            COLD: FULL_CLOSE → PARTIAL_25
            WARM: FULL_CLOSE → PARTIAL_75
            HOT: FULL_CLOSE allowed
            All PARTIAL actions pass unchanged
        
        Returns:
            (calibrated_action, reason)
        """
        # PARTIAL actions always pass
        if original_action.startswith("PARTIAL_"):
            return original_action, "partial_unchanged"
        
        # Non-FULL_CLOSE actions pass
        if original_action != "FULL_CLOSE":
            return original_action, "non_full_close_unchanged"
        
        # Apply heat-based downgrade for FULL_CLOSE
        if heat_bucket == "COLD":
            return "PARTIAL_25", "portfolio_heat_cold"
        elif heat_bucket == "WARM":
            return "PARTIAL_75", "portfolio_heat_warm"
        else:  # HOT
            return "FULL_CLOSE", "portfolio_heat_hot"
    
    def _process_proposal(self, stream_id: bytes, data: Dict) -> bool:
        """
        Process single harvest proposal.
        
        Returns:
            True if processed successfully (ACK), False otherwise
        """
        try:
            stream_id_str = stream_id.decode()
            
            # Extract proposal fields
            plan_id = data.get(b"plan_id", b"").decode()
            symbol = data.get(b"symbol", b"").decode()
            action = data.get(b"action", b"").decode()
            trace_id = data.get(b"trace_id", str(uuid.uuid4()).encode()).decode()
            
            if not plan_id or not symbol or not action:
                logger.warning(f"Missing required fields in proposal: {stream_id_str}")
                return True  # ACK but skip
            
            logger.info(f"📥 Proposal: {plan_id[:8]} | {symbol} {action}")
            
            # Get portfolio heat
            heat_value, heat_bucket, heat_context = self._get_portfolio_heat()
            
            # Apply gating rule
            calibrated_action, reason = self._apply_gating_rule(action, heat_bucket, heat_value)
            
            # Check if downgraded
            downgraded = calibrated_action != action
            
            if downgraded:
                logger.warning(
                    f"⚠️  DOWNGRADE: {symbol} {action} → {calibrated_action} "
                    f"(heat={heat_value:.4f} {heat_bucket}, reason={reason})"
                )
                if PROMETHEUS_AVAILABLE:
                    self.metric_downgrades.labels(
                        from_action=action,
                        to_action=calibrated_action,
                        reason=reason
                    ).inc()
            else:
                logger.info(f"✅ PASS: {symbol} {action} (heat={heat_value:.4f} {heat_bucket})")
            
            # Build calibrated proposal
            calibrated = {
                b"trace_id": trace_id.encode(),
                b"plan_id": plan_id.encode(),
                b"symbol": symbol.encode(),
                b"original_action": action.encode(),
                b"calibrated_action": calibrated_action.encode(),
                b"heat_value": str(heat_value).encode(),
                b"heat_bucket": heat_bucket.encode(),
                b"mode": self.config.MODE.encode(),
                b"calibrated": b"true",
                b"reason": reason.encode(),
                b"timestamp": str(int(time.time())).encode()
            }
            
            # Copy additional fields from original proposal
            for key in data:
                if key not in calibrated and key not in (b"action",):
                    calibrated[key] = data[key]
            
            # ALWAYS write to output stream (both shadow and enforce modes)
            msg_id = self.redis.xadd(self.config.OUTPUT_STREAM, calibrated)
            
            # In enforce mode, also write calibrated proposal to hash key that Apply Layer reads
            if self.config.MODE == "enforce":
                try:
                    # Write to quantum:harvest:proposal:{symbol} hash key
                    # This overwrites the original proposal with the calibrated one
                    hash_key = f"quantum:harvest:proposal:{symbol}"
                    
                    # Prepare hash data (string keys for Apply Layer compatibility)
                    hash_data = {
                        "trace_id": trace_id,
                        "plan_id": plan_id,
                        "symbol": symbol,
                        "action": calibrated_action,  # Use calibrated action
                        "original_action": action,
                        "heat_value": str(heat_value),
                        "heat_bucket": heat_bucket,
                        "calibrated": "1",
                        "calibrated_by": "p26_heat_gate",
                        "downgrade_reason": reason,
                        "ts": str(int(time.time()))
                    }
                    
                    # Copy additional fields from original proposal
                    for key in data:
                        key_str = key.decode() if isinstance(key, bytes) else key
                        if key_str not in hash_data and key_str not in ("action",):
                            val = data[key]
                            hash_data[key_str] = val.decode() if isinstance(val, bytes) else str(val)
                    
                    # Write to hash key
                    self.redis.hset(hash_key, mapping=hash_data)
                    # Set TTL to 5 minutes
                    self.redis.expire(hash_key, 300)
                    
                    logger.info(
                        f"📤 ENFORCE: {plan_id[:8]} | {symbol} {action}→{calibrated_action} | "
                        f"hash={hash_key} stream={msg_id.decode()}"
                    )
                    
                    if PROMETHEUS_AVAILABLE:
                        self.metric_hash_writes.inc()
                        
                except Exception as e:
                    logger.error(f"❌ HASH WRITE FAILED for {symbol}: {e} - FAIL-OPEN (keeping original proposal)")
                    if PROMETHEUS_AVAILABLE:
                        self.metric_hash_failures.inc()
                    # Fail-open: if hash write fails, don't crash - original proposal remains
            else:
                # Shadow mode: log comparison but don't affect apply layer
                logger.info(
                    f"🔍 SHADOW-COMPARE: {plan_id[:8]} | "
                    f"proposal={action} vs calibrated={calibrated_action} | "
                    f"heat={heat_value:.4f} {heat_bucket} | "
                    f"downgraded={downgraded} reason={reason}"
                )
            
            if PROMETHEUS_AVAILABLE:
                self.metric_processed.inc()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process proposal {stream_id}: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE:
                self.metric_failures.labels(reason="processing_error").inc()
            return False
    
    def run(self):
        """Main event loop"""
        logger.info("=" * 80)
        logger.info("P2.6 Portfolio Heat Gate - Hedge Fund OS")
        logger.info("=" * 80)
        logger.info(f"Redis: {self.config.REDIS_HOST}:{self.config.REDIS_PORT}/{self.config.REDIS_DB}")
        logger.info(f"Mode: {self.config.MODE.upper()}")
        logger.info(f"Heat Thresholds: COLD < {self.config.HEAT_MIN} < WARM < {self.config.HEAT_MAX} < HOT")
        logger.info(f"Consumer: {self.config.CONSUMER_GROUP} / {self.consumer_name}")
        logger.info(f"Input: {self.config.INPUT_STREAM}")
        logger.info(f"Output: {self.config.OUTPUT_STREAM}")
        logger.info(f"Metrics: :{self.config.METRICS_PORT}/metrics")
        logger.info("=" * 80)
        
        # Start metrics server
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(self.config.METRICS_PORT)
                logger.info(f"✅ Metrics server started on :{self.config.METRICS_PORT}")
                # Set enforce mode gauge
                self.metric_enforce_mode.set(1 if self.config.MODE == "enforce" else 0)
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")
        
        logger.info("🚀 Portfolio Heat Gate started")
        
        # Initial heat check
        heat, bucket, ctx = self._get_portfolio_heat()
        logger.info(f"📊 Initial Portfolio Heat: {heat:.4f} ({bucket})")
        
        while True:
            try:
                # Read from harvest.proposal stream
                messages = self.redis.xreadgroup(
                    self.config.CONSUMER_GROUP,
                    self.consumer_name,
                    {self.config.INPUT_STREAM: ">"},
                    count=10,
                    block=self.config.POLL_SEC * 1000
                )
                
                if not messages:
                    continue
                
                for stream_name, events in messages:
                    for stream_id, event_data in events:
                        # Calculate lag
                        try:
                            stream_ts = int(stream_id.decode().split("-")[0])
                            lag_ms = int(time.time() * 1000) - stream_ts
                            if PROMETHEUS_AVAILABLE:
                                self.metric_lag.observe(lag_ms)
                        except:
                            pass
                        
                        # Process proposal
                        success = self._process_proposal(stream_id, event_data)
                        
                        # ACK if processed
                        if success:
                            self.redis.xack(self.config.INPUT_STREAM, self.config.CONSUMER_GROUP, stream_id)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in event loop: {e}", exc_info=True)
                time.sleep(5)


def main():
    """Entry point"""
    config = Config()
    gate = PortfolioHeatGate(config)
    gate.run()


if __name__ == "__main__":
    main()
