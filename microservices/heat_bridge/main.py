"""
P2.7 HeatBridge - Shadow-Only Wiring Layer
===========================================
Consumes P2.6 heat decisions and produces fast lookup keys (plan_id, symbol).
Shadow-only, fail-open, no execution impact.
"""

import os
import sys
import time
import json
import logging
import redis
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Dict, Any, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics."""
    
    metrics_data = {}
    
    def do_GET(self):
        """Serve metrics in Prometheus format."""
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            
            lines = []
            for key, value in sorted(self.metrics_data.items()):
                lines.append(f"{key} {value}")
            
            self.wfile.write('\n'.join(lines).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health checks."""
    
    health_data = {"status": "ok", "last_ts_epoch": 0, "last_loop_ms": 0, "backlog": 0}
    
    def do_GET(self):
        """Serve health status."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.health_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


class HeatBridgeService:
    """P2.7 HeatBridge - shadow-only wiring layer for heat decisions."""
    
    def __init__(self):
        """Initialize service from environment."""
        self.stream_in = os.getenv("P27_STREAM_IN", "quantum:stream:harvest.heat.decision")
        self.group = os.getenv("P27_GROUP", "heat_bridge")
        self.consumer = os.getenv("P27_CONSUMER", "heat_bridge_1")
        self.poll_ms = int(os.getenv("P27_POLL_MS", "500"))
        self.batch = int(os.getenv("P27_BATCH", "10"))
        self.ttl_plan = int(os.getenv("P27_TTL_PLAN_SEC", "1800"))
        self.ttl_symbol = int(os.getenv("P27_TTL_SYMBOL_SEC", "1800"))
        self.dedupe_ttl = int(os.getenv("P27_DEDUPE_TTL_SEC", "120"))
        self.metrics_port = int(os.getenv("P27_METRICS_PORT", "8070"))
        self.health_port = int(os.getenv("P27_HEALTH_PORT", "8071"))
        
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=0,
            decode_responses=True
        )
        
        # Metrics
        self.metrics = {
            "p27_loops_total": 0,
            "p27_in_messages_total": 0,
            "p27_written_total_by_plan": 0,
            "p27_written_total_latest_symbol": 0,
            "p27_dedupe_skips_total": 0,
            "p27_errors_total_read": 0,
            "p27_errors_total_write": 0,
            "p27_errors_total_ack": 0,
            "p27_lag_ms": 0,
            "p27_last_ts_epoch": 0
        }
        
        # Health
        self.health = {
            "status": "ok",
            "last_ts_epoch": 0,
            "last_loop_ms": 0,
            "backlog": 0
        }
        
        logger.info(f"HeatBridge initialized")
        logger.info(f"Input stream: {self.stream_in}")
        logger.info(f"Consumer group: {self.group}/{self.consumer}")
        logger.info(f"TTLs: plan={self.ttl_plan}s, symbol={self.ttl_symbol}s, dedupe={self.dedupe_ttl}s")
        logger.info(f"Metrics port: {self.metrics_port}, Health port: {self.health_port}")
    
    def ensure_consumer_group(self):
        """Ensure Redis consumer group exists."""
        try:
            self.redis_client.xgroup_create(
                self.stream_in,
                self.group,
                id='0',
                mkstream=True
            )
            logger.info(f"Consumer group '{self.group}' created")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{self.group}' already exists")
            else:
                raise
    
    def is_deduped(self, plan_id: str) -> bool:
        """Check if plan_id was recently processed (deduplication)."""
        dedupe_key = f"quantum:dedupe:p27:{plan_id}"
        exists = self.redis_client.exists(dedupe_key)
        if exists:
            self.metrics["p27_dedupe_skips_total"] += 1
            return True
        
        # Set dedupe marker
        try:
            self.redis_client.setex(dedupe_key, self.dedupe_ttl, "1")
        except Exception as e:
            logger.warning(f"Failed to set dedupe marker: {e}")
        
        return False
    
    def write_lookup_keys(self, decision: Dict[str, Any]) -> bool:
        """
        Write heat decision to lookup keys.
        Returns True if successful, False otherwise.
        """
        plan_id = decision.get("plan_id", "unknown")
        symbol = decision.get("symbol", "UNKNOWN")
        
        # Prepare lookup data
        lookup_data = {
            "ts_epoch": decision.get("ts_epoch", ""),
            "symbol": symbol,
            "plan_id": plan_id,
            "in_action": decision.get("in_action", ""),
            "out_action": decision.get("out_action", ""),
            "heat_level": decision.get("heat_level", ""),
            "heat_score": decision.get("heat_score", ""),
            "heat_action": decision.get("heat_action", ""),
            "recommended_partial": decision.get("recommended_partial", ""),
            "reason": decision.get("reason", ""),
            "mode": decision.get("mode", ""),
            "debug_json": decision.get("debug_json", "{}")
        }
        
        try:
            # Write by_plan lookup key
            by_plan_key = f"quantum:harvest:heat:by_plan:{plan_id}"
            self.redis_client.hset(by_plan_key, mapping=lookup_data)
            self.redis_client.expire(by_plan_key, self.ttl_plan)
            self.metrics["p27_written_total_by_plan"] += 1
            
            # Write latest_symbol lookup key
            latest_key = f"quantum:harvest:heat:latest:{symbol}"
            latest_data = lookup_data.copy()
            latest_data["last_plan_id"] = plan_id
            self.redis_client.hset(latest_key, mapping=latest_data)
            self.redis_client.expire(latest_key, self.ttl_symbol)
            self.metrics["p27_written_total_latest_symbol"] += 1
            
            # Write latest_plan_id pointer
            pointer_key = f"quantum:harvest:heat:latest_plan_id:{symbol}"
            self.redis_client.setex(pointer_key, self.ttl_symbol, plan_id)
            
            logger.info(
                f"{symbol}/{plan_id}: written lookup keys (heat_action={decision.get('heat_action', 'NONE')})"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to write lookup keys: {e}")
            self.metrics["p27_errors_total_write"] += 1
            return False
    
    def process_message(self, msg_id: str, msg_data: Dict[str, Any]):
        """Process a single heat decision message."""
        try:
            plan_id = msg_data.get("plan_id", "unknown")
            symbol = msg_data.get("symbol", "UNKNOWN")
            
            # Check deduplication
            if self.is_deduped(plan_id):
                logger.debug(f"{symbol}/{plan_id}: deduplicated (skipped)")
                return True
            
            # Write lookup keys
            success = self.write_lookup_keys(msg_data)
            
            # Update metrics
            ts_epoch = msg_data.get("ts_epoch")
            if ts_epoch:
                try:
                    ts_epoch_int = int(ts_epoch)
                    self.metrics["p27_last_ts_epoch"] = ts_epoch_int
                    
                    # Calculate lag
                    lag_ms = int((time.time() - ts_epoch_int) * 1000)
                    self.metrics["p27_lag_ms"] = max(0, lag_ms)
                except (ValueError, TypeError):
                    pass
            
            self.metrics["p27_in_messages_total"] += 1
            
            return success
        
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            return False
    
    def get_backlog(self) -> int:
        """Estimate stream backlog."""
        try:
            pending = self.redis_client.xpending(self.stream_in, self.group)
            if pending and isinstance(pending, dict):
                return pending.get("num-pending-messages", 0)
        except Exception as e:
            logger.debug(f"Failed to get backlog: {e}")
        return 0
    
    def run_event_loop(self):
        """Main event loop: consume heat decisions and write lookup keys."""
        logger.info("Starting event loop...")
        
        while True:
            try:
                loop_start = time.time()
                
                # Read messages from stream
                messages = self.redis_client.xreadgroup(
                    self.group,
                    self.consumer,
                    {self.stream_in: '>'},
                    count=self.batch,
                    block=self.poll_ms
                )
                
                if messages:
                    for stream, msg_list in messages:
                        for msg_id, msg_data in msg_list:
                            # Process message
                            if self.process_message(msg_id, msg_data):
                                # Acknowledge on success
                                try:
                                    self.redis_client.xack(self.stream_in, self.group, msg_id)
                                except Exception as e:
                                    logger.warning(f"Failed to ack {msg_id}: {e}")
                                    self.metrics["p27_errors_total_ack"] += 1
                
                # Update health
                self.health["last_ts_epoch"] = self.metrics["p27_last_ts_epoch"]
                self.health["backlog"] = self.get_backlog()
                
                # Update loop metrics
                self.metrics["p27_loops_total"] += 1
                loop_ms = int((time.time() - loop_start) * 1000)
                self.health["last_loop_ms"] = loop_ms
            
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Event loop error: {e}")
                self.metrics["p27_errors_total_read"] += 1
                time.sleep(1)
    
    def export_metrics(self) -> Dict[str, float]:
        """Export metrics for HTTP handler."""
        exported = {
            "p27_loops_total": self.metrics["p27_loops_total"],
            "p27_in_messages_total": self.metrics["p27_in_messages_total"],
            f'p27_written_total{{kind="by_plan"}}': self.metrics["p27_written_total_by_plan"],
            f'p27_written_total{{kind="latest_symbol"}}': self.metrics["p27_written_total_latest_symbol"],
            "p27_dedupe_skips_total": self.metrics["p27_dedupe_skips_total"],
            f'p27_errors_total{{stage="read"}}': self.metrics["p27_errors_total_read"],
            f'p27_errors_total{{stage="write"}}': self.metrics["p27_errors_total_write"],
            f'p27_errors_total{{stage="ack"}}': self.metrics["p27_errors_total_ack"],
            "p27_lag_ms": self.metrics["p27_lag_ms"],
            "p27_last_ts_epoch": self.metrics["p27_last_ts_epoch"]
        }
        return exported
    
    def start_metrics_server(self):
        """Start metrics HTTP server in background thread."""
        def run_server():
            MetricsHandler.metrics_data = self.export_metrics()
            server = HTTPServer(('0.0.0.0', self.metrics_port), MetricsHandler)
            logger.info(f"Metrics server started on port {self.metrics_port}")
            
            while True:
                server.handle_request()
                MetricsHandler.metrics_data = self.export_metrics()
        
        thread = Thread(target=run_server, daemon=True)
        thread.start()
    
    def start_health_server(self):
        """Start health HTTP server in background thread."""
        def run_server():
            HealthHandler.health_data = self.health
            server = HTTPServer(('0.0.0.0', self.health_port), HealthHandler)
            logger.info(f"Health server started on port {self.health_port}")
            
            while True:
                server.handle_request()
                HealthHandler.health_data = self.health
        
        thread = Thread(target=run_server, daemon=True)
        thread.start()
    
    def run(self):
        """Start all services and run main loop."""
        try:
            # Test Redis connection
            self.redis_client.ping()
            logger.info("Redis connection OK")
            
            # Ensure consumer group
            self.ensure_consumer_group()
            
            # Start HTTP servers
            self.start_metrics_server()
            self.start_health_server()
            
            # Run main event loop
            self.run_event_loop()
        
        except Exception as e:
            logger.error(f"Service failed to start: {e}")
            sys.exit(1)


def main():
    """Entry point."""
    service = HeatBridgeService()
    service.run()


if __name__ == "__main__":
    main()
