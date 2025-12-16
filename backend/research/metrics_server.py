"""
Prometheus metrics HTTP server for Strategy Generator AI.

Exposes metrics on :9090/metrics for Prometheus scraping.
"""

import logging
import time
from prometheus_client import start_http_server, REGISTRY
from backend.database import SessionLocal
from backend.research.postgres_repository import PostgresStrategyRepository
from backend.research.metrics import update_status_counts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_metrics_server(port: int = 9090, update_interval: int = 60):
    """
    Start Prometheus metrics server.
    
    Args:
        port: HTTP port for metrics endpoint
        update_interval: Seconds between metric updates
    """
    logger.info(f"Starting Prometheus metrics server on port {port}")
    
    # Start HTTP server
    start_http_server(port)
    logger.info(f"✅ Metrics available at http://localhost:{port}/metrics")
    
    # Update metrics periodically
    repo = PostgresStrategyRepository(SessionLocal)
    
    while True:
        try:
            # Update strategy status counts
            update_status_counts(repo)
            logger.debug("Updated strategy status metrics")
            
            # Sleep until next update
            time.sleep(update_interval)
            
        except KeyboardInterrupt:
            logger.info("Shutting down metrics server...")
            break
        except Exception as e:
            logger.error(f"Error updating metrics: {e}", exc_info=True)
            time.sleep(update_interval)


if __name__ == "__main__":
    import os
    port = int(os.getenv("METRICS_PORT", "9090"))
    update_interval = int(os.getenv("METRICS_UPDATE_INTERVAL", "60"))
    run_metrics_server(port, update_interval)
