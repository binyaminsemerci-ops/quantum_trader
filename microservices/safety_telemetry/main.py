#!/usr/bin/env python3
"""
P1 Safety Telemetry Exporter - Quantum Trader
Exposes Prometheus metrics for Safety Kernel + Router telemetry
Read-only, no impact on trading logic
"""

import os
import sys
import time
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import redis
from prometheus_client import Gauge, Info, Counter, start_http_server, REGISTRY

# Configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
PORT = int(os.getenv("PORT", "9105"))
SAFETY_WINDOW_SEC = int(os.getenv("SAFETY_WINDOW_SEC", "10"))
SAMPLE_INTERVAL_SEC = int(os.getenv("SAMPLE_INTERVAL_SEC", "15"))
FAULT_LOOKBACK_MAX = int(os.getenv("FAULT_LOOKBACK_MAX", "2000"))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("safety-telemetry")

# Prometheus Metrics

# A) Safe Mode
safe_mode_gauge = Gauge("quantum_safety_safe_mode", "Safe mode active (0=off, 1=on)")
safe_mode_ttl_gauge = Gauge("quantum_safety_safe_mode_ttl_seconds", "TTL of safe mode key (-1 if not set)")

# B) Faults
fault_stream_length_gauge = Gauge("quantum_safety_fault_stream_length", "Length of safety.fault stream")
faults_last_1h_gauge = Gauge("quantum_safety_faults_last_1h", "Count of faults in last 1 hour")
last_fault_timestamp_gauge = Gauge("quantum_safety_last_fault_timestamp", "Timestamp of last fault (unix seconds)")
last_fault_info = Info("quantum_safety_last_fault", "Last fault details")

# C) Trade Intent Stream
trade_intent_length_gauge = Gauge("quantum_trade_intent_stream_length", "Length of trade.intent stream")
trade_intent_rate_per_min_gauge = Gauge("quantum_trade_intent_rate_per_min", "Trade intents per minute (rolling)")

# D) Safety Rate Counters
safety_rate_global_gauge = Gauge("quantum_safety_rate_global_current_window", "Global rate counter for current window bucket")
safety_rate_symbol_info = Info("quantum_safety_rate_symbol_top5", "Top 5 symbols by current window count (symbol, count)")

# E) Exporter Health
redis_up_gauge = Gauge("quantum_safety_redis_up", "Redis connectivity (1=up, 0=down)")
redis_last_error_timestamp_gauge = Gauge("quantum_safety_redis_last_error_timestamp", "Last Redis error timestamp")
exporter_scrapes_total = Counter("quantum_safety_exporter_scrapes_total", "Total scrapes performed")
exporter_errors_total = Counter("quantum_safety_exporter_errors_total", "Total errors encountered")


class SafetyTelemetryExporter:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.trade_intent_samples = deque(maxlen=int(60 / SAMPLE_INTERVAL_SEC) + 1)  # Store ~60s of samples
        self.last_values = {}  # Cache last known values
        
    def connect_redis(self):
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            redis_up_gauge.set(1)
            return True
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            redis_up_gauge.set(0)
            redis_last_error_timestamp_gauge.set(time.time())
            return False
    
    def get_current_bucket(self) -> int:
        """Get current time bucket for safety rate counters"""
        return int(time.time() // SAFETY_WINDOW_SEC)
    
    def collect_safe_mode(self):
        """Collect safe mode metrics"""
        try:
            safe_mode_value = self.redis_client.get("quantum:safety:safe_mode")
            if safe_mode_value is not None:
                safe_mode_gauge.set(1 if int(safe_mode_value) else 0)
                ttl = self.redis_client.ttl("quantum:safety:safe_mode")
                safe_mode_ttl_gauge.set(ttl if ttl >= 0 else -1)
                self.last_values["safe_mode"] = (1 if int(safe_mode_value) else 0, ttl)
            else:
                safe_mode_gauge.set(0)
                safe_mode_ttl_gauge.set(-1)
                self.last_values["safe_mode"] = (0, -1)
        except Exception as e:
            logger.warning(f"Error collecting safe mode: {e}")
            # Use cached value if available
            if "safe_mode" in self.last_values:
                safe_mode_gauge.set(self.last_values["safe_mode"][0])
                safe_mode_ttl_gauge.set(self.last_values["safe_mode"][1])
    
    def collect_faults(self):
        """Collect fault stream metrics"""
        try:
            # Stream length
            fault_length = self.redis_client.xlen("quantum:stream:safety.fault")
            fault_stream_length_gauge.set(fault_length)
            
            if fault_length > 0:
                # Last fault
                last_fault = self.redis_client.xrevrange(
                    "quantum:stream:safety.fault",
                    "+", "-",
                    count=1
                )
                
                if last_fault:
                    fault_id, fault_data = last_fault[0]
                    timestamp = int(fault_data.get("timestamp", 0))
                    reason = fault_data.get("reason", "unknown")
                    symbol = fault_data.get("symbol", "")
                    side = fault_data.get("side", "")
                    
                    last_fault_timestamp_gauge.set(timestamp)
                    last_fault_info.info({
                        "reason": reason,
                        "symbol": symbol,
                        "side": side
                    })
                    
                    # Count faults in last hour
                    one_hour_ago = int(time.time()) - 3600
                    faults_1h = 0
                    
                    # Use XREVRANGE with max count to avoid scanning entire stream
                    recent_faults = self.redis_client.xrevrange(
                        "quantum:stream:safety.fault",
                        "+", "-",
                        count=min(FAULT_LOOKBACK_MAX, fault_length)
                    )
                    
                    for _, fault in recent_faults:
                        fault_ts = int(fault.get("timestamp", 0))
                        if fault_ts >= one_hour_ago:
                            faults_1h += 1
                        else:
                            break  # Stop when we hit older faults
                    
                    faults_last_1h_gauge.set(faults_1h)
                    self.last_values["faults"] = (fault_length, timestamp, faults_1h)
            else:
                last_fault_timestamp_gauge.set(0)
                faults_last_1h_gauge.set(0)
                self.last_values["faults"] = (0, 0, 0)
                
        except Exception as e:
            logger.warning(f"Error collecting faults: {e}")
            if "faults" in self.last_values:
                fault_stream_length_gauge.set(self.last_values["faults"][0])
                last_fault_timestamp_gauge.set(self.last_values["faults"][1])
                faults_last_1h_gauge.set(self.last_values["faults"][2])
    
    def collect_trade_intent(self):
        """Collect trade.intent stream metrics and compute rate"""
        try:
            stream_length = self.redis_client.xlen("quantum:stream:trade.intent")
            trade_intent_length_gauge.set(stream_length)
            
            # Store sample for rate calculation
            now = time.time()
            self.trade_intent_samples.append((now, stream_length))
            
            # Compute rate per minute from samples
            if len(self.trade_intent_samples) >= 2:
                oldest_sample = self.trade_intent_samples[0]
                newest_sample = self.trade_intent_samples[-1]
                
                time_delta = newest_sample[0] - oldest_sample[0]
                length_delta = newest_sample[1] - oldest_sample[1]
                
                if time_delta > 0:
                    rate_per_min = (length_delta / time_delta) * 60
                    trade_intent_rate_per_min_gauge.set(max(0, rate_per_min))  # Avoid negative due to MAXLEN trimming
                    self.last_values["trade_rate"] = rate_per_min
            
        except Exception as e:
            logger.warning(f"Error collecting trade.intent: {e}")
            if "trade_rate" in self.last_values:
                trade_intent_rate_per_min_gauge.set(self.last_values["trade_rate"])
    
    def collect_safety_rate_counters(self):
        """Collect safety rate counters (global + top5 symbols)"""
        try:
            bucket = self.get_current_bucket()
            
            # Global counter
            global_key = f"quantum:safety:rate:global:{bucket}"
            global_count = self.redis_client.get(global_key)
            if global_count:
                safety_rate_global_gauge.set(int(global_count))
                self.last_values["global_rate"] = int(global_count)
            else:
                safety_rate_global_gauge.set(0)
                self.last_values["global_rate"] = 0
            
            # Symbol counters - use SCAN with strict cap
            pattern = f"quantum:safety:rate:symbol:*:{bucket}"
            cursor = 0
            symbol_counts: List[Tuple[str, int]] = []
            scans_done = 0
            max_scans = 20  # Limit SCAN iterations
            
            while scans_done < max_scans:
                cursor, keys = self.redis_client.scan(
                    cursor,
                    match=pattern,
                    count=50
                )
                
                for key in keys:
                    try:
                        # Extract symbol from key: quantum:safety:rate:symbol:BTCUSDT:12345
                        parts = key.split(":")
                        if len(parts) >= 5:
                            symbol = parts[4]
                            count = int(self.redis_client.get(key) or 0)
                            symbol_counts.append((symbol, count))
                    except Exception as e:
                        logger.debug(f"Error parsing symbol key {key}: {e}")
                
                scans_done += 1
                if cursor == 0:
                    break
            
            # Top 5 symbols
            top5 = sorted(symbol_counts, key=lambda x: x[1], reverse=True)[:5]
            
            if top5:
                info_dict = {}
                for i, (symbol, count) in enumerate(top5):
                    info_dict[f"symbol_{i+1}"] = symbol
                    info_dict[f"count_{i+1}"] = str(count)
                safety_rate_symbol_info.info(info_dict)
                self.last_values["top5"] = top5
            else:
                safety_rate_symbol_info.info({"symbols": "none"})
                
        except Exception as e:
            logger.warning(f"Error collecting safety rate counters: {e}")
            if "global_rate" in self.last_values:
                safety_rate_global_gauge.set(self.last_values["global_rate"])
    
    def collect_metrics(self):
        """Main metrics collection cycle"""
        exporter_scrapes_total.inc()
        
        if not self.redis_client or not self.connect_redis():
            exporter_errors_total.inc()
            logger.warning("Redis unavailable, using cached values")
            return
        
        try:
            self.collect_safe_mode()
            self.collect_faults()
            self.collect_trade_intent()
            self.collect_safety_rate_counters()
            redis_up_gauge.set(1)
            
        except Exception as e:
            logger.error(f"Error during metrics collection: {e}")
            exporter_errors_total.inc()
            redis_up_gauge.set(0)
            redis_last_error_timestamp_gauge.set(time.time())
    
    def run(self):
        """Main run loop"""
        logger.info("="*60)
        logger.info("P1 Safety Telemetry Exporter")
        logger.info("="*60)
        logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
        logger.info(f"HTTP Port: {PORT}")
        logger.info(f"Sample Interval: {SAMPLE_INTERVAL_SEC}s")
        logger.info(f"Safety Window: {SAFETY_WINDOW_SEC}s")
        logger.info("="*60)
        
        # Start HTTP server
        start_http_server(PORT, addr="127.0.0.1")
        logger.info(f"✅ Metrics server listening on http://127.0.0.1:{PORT}/metrics")
        
        # Initial connection
        self.connect_redis()
        
        # Collection loop
        while True:
            try:
                self.collect_metrics()
                time.sleep(SAMPLE_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                exporter_errors_total.inc()
                time.sleep(SAMPLE_INTERVAL_SEC)


if __name__ == "__main__":
    exporter = SafetyTelemetryExporter()
    exporter.run()
