"""
P2.6 Portfolio Heat Gate Microservice
======================================
Shadow-first, fail-open portfolio heat moderator.
Consumes harvest proposals, computes heat, emits moderation decisions.
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

from logic import HeatGateLogic


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
    
    health_data = {"status": "ok", "ts_epoch": int(time.time()), "last_loop_ms": 0}
    
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


class HeatGateService:
    """P2.6 Portfolio Heat Gate Service."""
    
    def __init__(self):
        """Initialize service from environment."""
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.metrics_port = int(os.getenv("HEAT_METRICS_PORT", "8068"))
        self.health_port = int(os.getenv("HEAT_HEALTH_PORT", "8069"))
        
        self.stream_in = os.getenv("HEAT_STREAM_IN", "quantum:stream:harvest.proposal")
        self.stream_out = os.getenv("HEAT_STREAM_OUT", "quantum:stream:harvest.heat.decision")
        self.group = os.getenv("HEAT_GROUP", "heat_gate")
        self.consumer = os.getenv("HEAT_CONSUMER", "heat_gate_1")
        self.poll_ms = int(os.getenv("HEAT_POLL_MS", "500"))
        self.state_key = os.getenv("HEAT_STATE_KEY", "quantum:portfolio:state")
        
        # Initialize Redis clients
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=0,
            decode_responses=True
        )
        
        # Initialize logic engine with full config
        config = {k: v for k, v in os.environ.items()}
        self.logic = HeatGateLogic(config)
        
        # Metrics
        self.metrics = {
            "p26_loops_total": 0,
            "p26_in_messages_total": {},
            "p26_out_messages_total": {},
            "p26_heat_score": {},
            "p26_failopen_total": {}
        }
        
        # Health
        self.health = {"status": "ok", "ts_epoch": int(time.time()), "last_loop_ms": 0}
        
        logger.info(f"Heat Gate initialized: {self.stream_in} → {self.stream_out}")
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
    
    def read_portfolio_state(self) -> Optional[Dict[str, Any]]:
        """Read portfolio state from Redis."""
        try:
            state = self.redis_client.hgetall(self.state_key)
            if not state:
                return None
            return state
        except Exception as e:
            logger.warning(f"Failed to read portfolio state: {e}")
            return None
    
    def write_heat_decision(self, decision: Dict[str, Any]):
        """Write heat decision to output stream and shadow key."""
        symbol = decision["symbol"]
        
        # Write to output stream
        try:
            self.redis_client.xadd(self.stream_out, decision)
        except Exception as e:
            logger.error(f"Failed to write heat decision stream: {e}")
        
        # Write shadow key
        shadow_key = f"quantum:harvest:heat:{symbol}"
        try:
            shadow_data = {
                "ts_epoch": decision["ts_epoch"],
                "last_plan_id": decision["plan_id"],
                "heat_level": decision["heat_level"],
                "heat_score": decision["heat_score"],
                "heat_action": decision["heat_action"],
                "out_action": decision["out_action"],
                "recommended_partial": decision.get("recommended_partial", ""),
                "reason": decision["reason"],
                "debug_json": decision.get("debug_json", "{}")
            }
            self.redis_client.hset(shadow_key, mapping=shadow_data)
            self.redis_client.expire(shadow_key, 600)  # 10 min TTL
        except Exception as e:
            logger.error(f"Failed to write shadow key: {e}")
    
    def update_metrics(self, decision: Dict[str, Any]):
        """Update Prometheus metrics."""
        # Update heat score gauge
        symbol = decision["symbol"]
        heat_score = decision["heat_score"]
        metric_key = f'p26_heat_score{{symbol="{symbol}"}}'
        self.metrics["p26_heat_score"][metric_key] = heat_score
        
        # Update message counters
        in_action = decision["in_action"]
        in_key = f'p26_in_messages_total{{action="{in_action}"}}'
        self.metrics["p26_in_messages_total"][in_key] = \
            self.metrics["p26_in_messages_total"].get(in_key, 0) + 1
        
        heat_action = decision["heat_action"]
        heat_level = decision["heat_level"]
        out_key = f'p26_out_messages_total{{heat_action="{heat_action}",heat_level="{heat_level}"}}'
        self.metrics["p26_out_messages_total"][out_key] = \
            self.metrics["p26_out_messages_total"].get(out_key, 0) + 1
        
        # Update fail-open counter
        if decision["reason"] != "ok":
            reason = decision["reason"]
            fail_key = f'p26_failopen_total{{reason="{reason}"}}'
            self.metrics["p26_failopen_total"][fail_key] = \
                self.metrics["p26_failopen_total"].get(fail_key, 0) + 1
    
    def export_metrics(self) -> Dict[str, float]:
        """Export metrics for HTTP handler."""
        exported = {}
        exported["p26_loops_total"] = self.metrics["p26_loops_total"]
        
        for key, val in self.metrics["p26_in_messages_total"].items():
            exported[key] = val
        for key, val in self.metrics["p26_out_messages_total"].items():
            exported[key] = val
        for key, val in self.metrics["p26_heat_score"].items():
            exported[key] = val
        for key, val in self.metrics["p26_failopen_total"].items():
            exported[key] = val
        
        return exported
    
    def process_message(self, msg_id: str, msg_data: Dict[str, Any]):
        """Process a single harvest proposal message."""
        try:
            # Read portfolio state
            portfolio_state = self.read_portfolio_state()
            
            # Process with logic engine
            decision = self.logic.process_harvest_proposal(msg_data, portfolio_state)
            
            # Write decision
            self.write_heat_decision(decision)
            
            # Update metrics
            self.update_metrics(decision)
            
            # Log
            symbol = decision["symbol"]
            heat_level = decision["heat_level"]
            heat_action = decision["heat_action"]
            out_action = decision["out_action"]
            reason = decision["reason"]
            
            if reason == "ok":
                logger.info(
                    f"{symbol}: {heat_level.upper()} (score={decision['heat_score']:.3f}) "
                    f"→ {heat_action} (out={out_action})"
                )
            else:
                logger.warning(
                    f"{symbol}: FAIL-OPEN ({reason}) → out={out_action}"
                )
        
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
    
    def run_event_loop(self):
        """Main event loop: consume harvest proposals and emit decisions."""
        logger.info("Starting event loop...")
        
        while True:
            try:
                loop_start = time.time()
                
                # Read messages from stream
                messages = self.redis_client.xreadgroup(
                    self.group,
                    self.consumer,
                    {self.stream_in: '>'},
                    count=10,
                    block=self.poll_ms
                )
                
                if messages:
                    for stream, msg_list in messages:
                        for msg_id, msg_data in msg_list:
                            self.process_message(msg_id, msg_data)
                            
                            # Acknowledge message
                            try:
                                self.redis_client.xack(self.stream_in, self.group, msg_id)
                            except Exception as e:
                                logger.warning(f"Failed to ack {msg_id}: {e}")
                
                # Update loop metrics
                self.metrics["p26_loops_total"] += 1
                loop_ms = int((time.time() - loop_start) * 1000)
                self.health["last_loop_ms"] = loop_ms
                self.health["ts_epoch"] = int(time.time())
            
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Event loop error: {e}")
                time.sleep(1)
    
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
    service = HeatGateService()
    service.run()


if __name__ == "__main__":
    main()
