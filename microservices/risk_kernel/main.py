#!/usr/bin/env python3
"""
Risk Kernel — Unified Process for All Risk Services (OP 7E)
============================================================

Consolidates 7 separate risk services into one process:
  1. Governor (stream: apply.plan) — entry gate
  2. Heat Gate (stream: harvest.proposal) — exit heat moderator
  3. Portfolio Gate (stream: harvest.proposal) — exit portfolio gate
  4. Portfolio Heat Gate (stream: harvest.proposal) — exit heat calibrator
  5. Risk Proposal Publisher (poll: 10s) — SL/TP calculator
  6. Capital Allocation (poll: 5s) — per-symbol budget allocator
  7. Portfolio Governance (poll: 30s) — macro policy controller

Architecture:
  - Each component runs in its own daemon thread
  - Single Redis connection pool shared where possible
  - Unified Prometheus metrics on one port
  - Unified health endpoint
  - Original modules imported wholesale (no rewrite)
"""

import os
import sys
import time
import json
import signal
import logging
import threading
import importlib
import redis

from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
HEALTH_PORT = int(os.getenv('RISK_KERNEL_HEALTH_PORT', '8070'))
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))

# Component toggles — disable individual components if needed
ENABLE_GOVERNOR = os.getenv('RK_ENABLE_GOVERNOR', 'true').lower() == 'true'
ENABLE_HEAT_GATE = os.getenv('RK_ENABLE_HEAT_GATE', 'true').lower() == 'true'
ENABLE_PORTFOLIO_GATE = os.getenv('RK_ENABLE_PORTFOLIO_GATE', 'true').lower() == 'true'
# Portfolio Heat Gate disabled by default — overlaps 90% with Heat Gate + Portfolio Gate
# and conflicts on prometheus metric names (p26_*). Enable only if the other two are off.
ENABLE_PORTFOLIO_HEAT_GATE = os.getenv('RK_ENABLE_PORTFOLIO_HEAT_GATE', 'false').lower() == 'true'
ENABLE_RISK_PROPOSAL = os.getenv('RK_ENABLE_RISK_PROPOSAL', 'true').lower() == 'true'
ENABLE_CAPITAL_ALLOCATION = os.getenv('RK_ENABLE_CAPITAL_ALLOCATION', 'true').lower() == 'true'
ENABLE_PORTFOLIO_GOVERNANCE = os.getenv('RK_ENABLE_PORTFOLIO_GOVERNANCE', 'true').lower() == 'true'

# ============================================================================
# PATH SETUP — import each service module from its directory
# ============================================================================
BASE = os.path.dirname(os.path.abspath(__file__))
MICROSERVICES = os.path.dirname(BASE)

# ============================================================================
# COMPONENT HEALTH TRACKING
# ============================================================================
component_health: Dict[str, Dict[str, Any]] = {}
health_lock = threading.Lock()
shutdown_event = threading.Event()


def update_health(name: str, status: str = "ok", error: str = ""):
    """Thread-safe health update for a component."""
    with health_lock:
        component_health[name] = {
            "status": status,
            "ts_epoch": int(time.time()),
            "error": error,
        }


# ============================================================================
# HEALTH HTTP SERVER
# ============================================================================
class HealthHandler(BaseHTTPRequestHandler):
    """Unified health endpoint for all risk kernel components."""

    def do_GET(self):
        if self.path == '/health':
            with health_lock:
                data = {
                    "service": "risk-kernel",
                    "ts_epoch": int(time.time()),
                    "components": dict(component_health),
                }
            all_ok = all(c["status"] == "ok" for c in data["components"].values())
            data["status"] = "ok" if all_ok else "degraded"

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress noisy HTTP logs


def start_health_server():
    """Start unified health HTTP server in a daemon thread."""
    def run():
        server = HTTPServer(('0.0.0.0', HEALTH_PORT), HealthHandler)
        logger.info(f"Health server on port {HEALTH_PORT}")
        server.serve_forever()

    t = threading.Thread(target=run, name="health-server", daemon=True)
    t.start()


# ============================================================================
# COMPONENT WRAPPERS — each runs in its own thread
# ============================================================================

def run_governor():
    """Thread target: import and run Governor."""
    name = "governor"
    update_health(name, "starting")
    try:
        gov_dir = os.path.join(MICROSERVICES, 'governor')
        sys.path.insert(0, gov_dir)
        # Also need parent for risk_guard import
        sys.path.insert(0, MICROSERVICES)

        from governor.main import Governor, Config

        # Disable Governor's own prometheus (we use unified health)
        # Override metrics port to avoid conflict
        os.environ['GOV_METRICS_PORT'] = os.getenv('GOV_METRICS_PORT', '8044')

        from prometheus_client import start_http_server as prom_start
        try:
            prom_start(int(os.getenv('GOV_METRICS_PORT', '8044')))
            logger.info(f"Governor prometheus on port {os.getenv('GOV_METRICS_PORT', '8044')}")
        except OSError:
            logger.warning("Governor prometheus port already in use, skipping")

        governor = Governor(Config)
        update_health(name, "ok")
        logger.info("Governor started")
        governor.run()
    except Exception as e:
        logger.error(f"Governor crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


def run_heat_gate():
    """Thread target: import and run Heat Gate."""
    name = "heat_gate"
    update_health(name, "starting")
    try:
        hg_dir = os.path.join(MICROSERVICES, 'heat_gate')
        sys.path.insert(0, hg_dir)

        from heat_gate.main import HeatGateService

        svc = HeatGateService()
        update_health(name, "ok")
        logger.info("Heat Gate started")
        svc.run()
    except Exception as e:
        logger.error(f"Heat Gate crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


def run_portfolio_gate():
    """Thread target: import and run Portfolio Gate."""
    name = "portfolio_gate"
    update_health(name, "starting")
    try:
        pg_dir = os.path.join(MICROSERVICES, 'portfolio_gate')
        sys.path.insert(0, pg_dir)

        from portfolio_gate.main import PortfolioGate

        # Override prometheus to avoid port conflict
        os.environ.setdefault('PG_METRICS_PORT', '8047')

        gate = PortfolioGate()
        update_health(name, "ok")
        logger.info("Portfolio Gate started")
        gate.run()
    except Exception as e:
        logger.error(f"Portfolio Gate crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


def run_portfolio_heat_gate():
    """Thread target: import and run Portfolio Heat Gate."""
    name = "portfolio_heat_gate"
    update_health(name, "starting")
    try:
        phg_dir = os.path.join(MICROSERVICES, 'portfolio_heat_gate')
        sys.path.insert(0, phg_dir)

        from portfolio_heat_gate.main import PortfolioHeatGate, Config as PHGConfig

        gate = PortfolioHeatGate(PHGConfig())
        update_health(name, "ok")
        logger.info("Portfolio Heat Gate started")
        gate.run()
    except Exception as e:
        logger.error(f"Portfolio Heat Gate crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


def run_risk_proposal():
    """Thread target: import and run Risk Proposal Publisher."""
    name = "risk_proposal"
    update_health(name, "starting")
    try:
        rp_dir = os.path.join(MICROSERVICES, 'risk_proposal_publisher')
        sys.path.insert(0, rp_dir)
        # risk_kernel_stops is in ai_engine/
        ai_dir = os.path.join(MICROSERVICES, 'ai_engine')
        sys.path.insert(0, ai_dir)

        from risk_proposal_publisher.main import RiskProposalPublisher

        redis_url = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}")
        symbols = [s.strip() for s in os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")]
        interval = int(os.getenv("PUBLISH_INTERVAL_SEC", "10"))

        publisher = RiskProposalPublisher(
            redis_url=redis_url,
            symbols=symbols,
            publish_interval=interval,
        )
        update_health(name, "ok")
        logger.info("Risk Proposal Publisher started")
        publisher.run_loop()
    except Exception as e:
        logger.error(f"Risk Proposal Publisher crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


def run_capital_allocation():
    """Thread target: import and run Capital Allocation."""
    name = "capital_allocation"
    update_health(name, "starting")
    try:
        ca_dir = os.path.join(MICROSERVICES, 'capital_allocation')
        sys.path.insert(0, ca_dir)

        from capital_allocation.main import AllocationEngine, RedisClient

        # Start prometheus for allocation
        from prometheus_client import start_http_server as prom_start
        try:
            port = int(os.getenv('P29_METRICS_PORT', '8059'))
            prom_start(port)
            logger.info(f"Capital Allocation prometheus on port {port}")
        except OSError:
            logger.warning("Capital Allocation prometheus port already in use, skipping")

        redis_client = RedisClient()
        engine = AllocationEngine(redis_client)
        update_health(name, "ok")
        logger.info("Capital Allocation started")
        engine.run_loop()
    except Exception as e:
        logger.error(f"Capital Allocation crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


def run_portfolio_governance():
    """Thread target: import and run Portfolio Governance."""
    name = "portfolio_governance"
    update_health(name, "starting")
    try:
        pg_dir = os.path.join(MICROSERVICES, 'portfolio_governance')
        sys.path.insert(0, pg_dir)

        from portfolio_governance.governance_controller import PortfolioGovernanceAgent

        redis_url = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")
        agent = PortfolioGovernanceAgent(redis_url=redis_url)
        update_health(name, "ok")
        logger.info("Portfolio Governance started")
        agent.run(interval=30)
    except Exception as e:
        logger.error(f"Portfolio Governance crashed: {e}", exc_info=True)
        update_health(name, "error", str(e))


# ============================================================================
# MAIN
# ============================================================================
COMPONENTS = [
    ("Governor",              ENABLE_GOVERNOR,              run_governor),
    ("Heat Gate",             ENABLE_HEAT_GATE,             run_heat_gate),
    ("Portfolio Gate",        ENABLE_PORTFOLIO_GATE,        run_portfolio_gate),
    ("Portfolio Heat Gate",   ENABLE_PORTFOLIO_HEAT_GATE,   run_portfolio_heat_gate),
    ("Risk Proposal",        ENABLE_RISK_PROPOSAL,         run_risk_proposal),
    ("Capital Allocation",   ENABLE_CAPITAL_ALLOCATION,    run_capital_allocation),
    ("Portfolio Governance",  ENABLE_PORTFOLIO_GOVERNANCE,  run_portfolio_governance),
]


def main():
    logger.info("=" * 60)
    logger.info("=== RISK KERNEL STARTING (OP 7E) ===")
    logger.info("=" * 60)

    # Test Redis connectivity
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        r.close()
        logger.info("Redis connection OK")
    except Exception as e:
        logger.critical(f"Redis connection failed: {e}")
        sys.exit(1)

    # Start unified health server
    start_health_server()

    # Launch component threads
    threads = []
    for label, enabled, target in COMPONENTS:
        if enabled:
            t = threading.Thread(target=target, name=label, daemon=True)
            t.start()
            threads.append((label, t))
            logger.info(f"  [+] {label} → thread started")
        else:
            logger.info(f"  [-] {label} → DISABLED")
            update_health(label.lower().replace(" ", "_"), "disabled")

    logger.info(f"Risk Kernel running with {len(threads)} components")

    # Keep main thread alive — wait for shutdown signal
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Watchdog: check thread health every 30s
    while not shutdown_event.is_set():
        shutdown_event.wait(30)
        if shutdown_event.is_set():
            break

        alive_count = sum(1 for _, t in threads if t.is_alive())
        dead = [(label, t) for label, t in threads if not t.is_alive()]

        if dead:
            for label, t in dead:
                logger.error(f"DEAD THREAD: {label}")
                update_health(label.lower().replace(" ", "_"), "dead")

        logger.info(f"Watchdog: {alive_count}/{len(threads)} threads alive")

    logger.info("Risk Kernel stopped")


if __name__ == '__main__':
    main()
