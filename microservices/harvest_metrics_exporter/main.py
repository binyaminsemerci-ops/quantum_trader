#!/usr/bin/env python3
"""
P2.7 Harvest Metrics Exporter
Reads harvest proposals from Redis and exposes Prometheus metrics on port 8042.
"""
import os
import sys
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Dict, Any, Optional
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Try importing redis, fallback to subprocess if not available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    import subprocess

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SEC", "5"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8042"))

# Known harvest actions for one-hot encoding
HARVEST_ACTIONS = ["NONE", "PARTIAL_25", "PARTIAL_50", "PARTIAL_75", "FULL_CLOSE_PROPOSED"]

# Global metrics cache
metrics_cache = {
    "last_update": 0,
    "data": {}
}
cache_lock = threading.Lock()


class RedisClient:
    """Redis client with fallback to redis-cli subprocess"""
    
    def __init__(self):
        self.use_lib = REDIS_AVAILABLE
        if self.use_lib:
            try:
                self.client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # Test connection
                self.client.ping()
                logger.info(f"Redis connected via redis-py: {REDIS_HOST}:{REDIS_PORT}")
            except Exception as e:
                logger.warning(f"redis-py failed: {e}, falling back to redis-cli")
                self.use_lib = False
    
    def hgetall(self, key: str) -> Dict[str, str]:
        """Get all fields from Redis hash"""
        if self.use_lib:
            try:
                return self.client.hgetall(key)
            except Exception as e:
                logger.error(f"hgetall failed for {key}: {e}")
                return {}
        else:
            # Fallback to redis-cli subprocess
            try:
                result = subprocess.check_output(
                    ["redis-cli", "-h", REDIS_HOST, "-p", str(REDIS_PORT), "HGETALL", key],
                    text=True,
                    timeout=5
                )
                lines = result.strip().split("\n")
                data = {}
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        data[lines[i]] = lines[i + 1]
                return data
            except Exception as e:
                logger.error(f"redis-cli hgetall failed for {key}: {e}")
                return {}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_timestamp(ts_str: Optional[str]) -> float:
    """Parse ISO timestamp to epoch seconds, return 0 if invalid"""
    if not ts_str:
        return 0.0
    try:
        # Try parsing ISO format
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


def fetch_harvest_data(redis_client: RedisClient) -> Dict[str, Dict[str, Any]]:
    """Fetch harvest proposal data from Redis for all symbols"""
    data = {}
    
    for symbol in SYMBOLS:
        key = f"quantum:harvest:proposal:{symbol}"
        try:
            raw = redis_client.hgetall(key)
            if not raw:
                logger.debug(f"No data for {symbol}")
                continue
            
            # Parse fields with safe defaults
            parsed = {
                "kill_score": safe_float(raw.get("kill_score")),
                "k_regime_flip": safe_float(raw.get("k_regime_flip")),
                "k_sigma_spike": safe_float(raw.get("k_sigma_spike")),
                "k_ts_drop": safe_float(raw.get("k_ts_drop")),
                "k_age_penalty": safe_float(raw.get("k_age_penalty")),
                "r_net": safe_float(raw.get("R_net") or raw.get("r_net")),
                "new_sl": safe_float(raw.get("new_sl_proposed")),
                "harvest_action": raw.get("harvest_action", "NONE"),
                "computed_at_utc": raw.get("computed_at_utc", ""),
            }
            
            # Parse timestamp
            parsed["last_update_epoch"] = parse_timestamp(parsed["computed_at_utc"])
            
            data[symbol] = parsed
            logger.debug(f"{symbol}: action={parsed['harvest_action']}, K={parsed['kill_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    
    return data


def format_prometheus_metrics(data: Dict[str, Dict[str, Any]]) -> str:
    """Format data as Prometheus text exposition format"""
    lines = []
    
    # Header
    lines.append("# HELP quantum_harvest_kill_score Kill score (edge collapse indicator)")
    lines.append("# TYPE quantum_harvest_kill_score gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_kill_score{{symbol="{symbol}"}} {values["kill_score"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_k_regime_flip Regime flip component (0 or 1)")
    lines.append("# TYPE quantum_harvest_k_regime_flip gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_k_regime_flip{{symbol="{symbol}"}} {values["k_regime_flip"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_k_sigma_spike Sigma spike component (0-2)")
    lines.append("# TYPE quantum_harvest_k_sigma_spike gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_k_sigma_spike{{symbol="{symbol}"}} {values["k_sigma_spike"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_k_ts_drop Trend strength drop component (0-0.5)")
    lines.append("# TYPE quantum_harvest_k_ts_drop gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_k_ts_drop{{symbol="{symbol}"}} {values["k_ts_drop"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_k_age_penalty Age penalty component (0-1)")
    lines.append("# TYPE quantum_harvest_k_age_penalty gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_k_age_penalty{{symbol="{symbol}"}} {values["k_age_penalty"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_r_net Net risk-reward ratio")
    lines.append("# TYPE quantum_harvest_r_net gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_r_net{{symbol="{symbol}"}} {values["r_net"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_new_sl Proposed new stop loss")
    lines.append("# TYPE quantum_harvest_new_sl gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_new_sl{{symbol="{symbol}"}} {values["new_sl"]}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_action Harvest action (one-hot encoded)")
    lines.append("# TYPE quantum_harvest_action gauge")
    for symbol, values in data.items():
        current_action = values["harvest_action"]
        for action in HARVEST_ACTIONS:
            value = 1.0 if current_action == action else 0.0
            lines.append(f'quantum_harvest_action{{symbol="{symbol}",action="{action}"}} {value}')
    
    lines.append("")
    lines.append("# HELP quantum_harvest_last_update_epoch Last update timestamp (seconds since epoch)")
    lines.append("# TYPE quantum_harvest_last_update_epoch gauge")
    for symbol, values in data.items():
        lines.append(f'quantum_harvest_last_update_epoch{{symbol="{symbol}"}} {values["last_update_epoch"]}')
    
    return "\n".join(lines) + "\n"


def poll_redis_worker(redis_client: RedisClient):
    """Background worker to poll Redis and update metrics cache"""
    logger.info(f"Starting Redis poll worker (interval={POLL_INTERVAL}s)")
    
    while True:
        try:
            data = fetch_harvest_data(redis_client)
            
            with cache_lock:
                metrics_cache["data"] = data
                metrics_cache["last_update"] = time.time()
            
            logger.info(f"Metrics updated: {len(data)} symbols")
            
        except Exception as e:
            logger.error(f"Poll worker error: {e}")
        
        time.sleep(POLL_INTERVAL)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics endpoint"""
    
    def log_message(self, format, *args):
        """Suppress default request logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/metrics":
            try:
                with cache_lock:
                    data = metrics_cache["data"]
                    cache_age = time.time() - metrics_cache["last_update"]
                
                # Format metrics
                metrics_text = format_prometheus_metrics(data)
                
                # Add meta metrics
                metrics_text += f"\n# HELP quantum_harvest_exporter_cache_age_seconds Cache age\n"
                metrics_text += f"# TYPE quantum_harvest_exporter_cache_age_seconds gauge\n"
                metrics_text += f"quantum_harvest_exporter_cache_age_seconds {cache_age:.1f}\n"
                
                # Send response
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(metrics_text.encode("utf-8"))
                
            except Exception as e:
                logger.error(f"Error serving metrics: {e}")
                self.send_error(500, f"Internal error: {e}")
        
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
        
        else:
            self.send_error(404, "Not found")


def main():
    logger.info("=== P2.7 Harvest Metrics Exporter ===")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT} (db={REDIS_DB})")
    logger.info(f"Symbols: {SYMBOLS}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Metrics port: {METRICS_PORT}")
    
    # Initialize Redis client
    redis_client = RedisClient()
    
    # Initial fetch
    logger.info("Performing initial Redis fetch...")
    data = fetch_harvest_data(redis_client)
    with cache_lock:
        metrics_cache["data"] = data
        metrics_cache["last_update"] = time.time()
    logger.info(f"Initial fetch complete: {len(data)} symbols")
    
    # Start background poll worker
    poll_thread = threading.Thread(target=poll_redis_worker, args=(redis_client,), daemon=True)
    poll_thread.start()
    
    # Start HTTP server
    server_address = ("", METRICS_PORT)
    httpd = HTTPServer(server_address, MetricsHandler)
    logger.info(f"HTTP server listening on port {METRICS_PORT}")
    logger.info("Endpoints: /metrics, /health")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
