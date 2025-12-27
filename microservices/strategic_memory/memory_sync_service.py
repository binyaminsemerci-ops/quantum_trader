"""
Strategic Memory Sync Service - Phase 4S
Main orchestration service for strategic memory synchronization
"""
import os
import time
import signal
import sys
from datetime import datetime
import structlog

from memory_loader import MemoryLoader
from pattern_analyzer import PatternAnalyzer
from reinforcement_feedback import ReinforcementFeedback

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


class StrategicMemorySync:
    """
    Main service for strategic memory synchronization
    
    Runs continuous loop to:
    1. Load memory from all strategic components
    2. Analyze patterns between strategies and results
    3. Generate reinforcement feedback for AI Engine
    """
    
    def __init__(self, redis_url: str = None, interval: int = 60):
        """
        Initialize Strategic Memory Sync service
        
        Args:
            redis_url: Redis connection URL (defaults to env var REDIS_URL)
            interval: Analysis interval in seconds (default: 60)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.interval = interval
        self.running = True
        self.iteration = 0
        
        # Initialize components
        self.loader = MemoryLoader(redis_url=self.redis_url)
        self.analyzer = PatternAnalyzer()
        self.feedback = ReinforcementFeedback(redis_url=self.redis_url)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(
            "StrategicMemorySync initialized",
            redis_url=self.redis_url,
            interval=interval
        )
    
    def run(self) -> None:
        """Main service loop"""
        logger.info("=" * 50)
        logger.info("Strategic Memory Sync - Starting")
        logger.info("=" * 50)
        logger.info(f"Start time: {datetime.utcnow().isoformat()}Z")
        logger.info(f"Redis: {self.redis_url}")
        logger.info(f"Analysis interval: {self.interval}s")
        
        while self.running:
            try:
                self.iteration += 1
                start_time = time.time()
                
                logger.info("Starting memory sync iteration", iteration=self.iteration)
                
                # Step 1: Load strategic memory
                memory = self.loader.load()
                
                total_samples = sum([
                    len(memory.get("meta_stream", [])),
                    len(memory.get("pnl_stream", [])),
                    len(memory.get("recent_trades", []))
                ])
                
                if total_samples < 3:
                    logger.warning(
                        "Insufficient memory samples for analysis",
                        iteration=self.iteration,
                        samples=total_samples
                    )
                else:
                    # Step 2: Analyze patterns
                    analysis = self.analyzer.analyze(memory)
                    
                    # Step 3: Generate and push feedback
                    feedback = self.feedback.push_feedback(analysis)
                    
                    elapsed = time.time() - start_time
                    logger.info(
                        "Memory sync iteration complete",
                        iteration=self.iteration,
                        samples=total_samples,
                        best_regime=analysis.get("best_regime", ["UNKNOWN"])[0],
                        policy=feedback.get("updated_policy"),
                        confidence=feedback.get("confidence_boost"),
                        elapsed_ms=round(elapsed * 1000, 2)
                    )
                
                # Sleep until next iteration
                time.sleep(self.interval)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(
                    "Error in memory sync iteration",
                    iteration=self.iteration,
                    error=str(e),
                    error_type=type(e).__name__
                )
                # Continue after error with longer sleep
                time.sleep(self.interval)
        
        logger.info("Strategic Memory Sync service stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal", signal=signum)
        self.running = False


def main():
    """Main entry point"""
    # Get configuration from environment
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    interval = int(os.getenv("MEMORY_SYNC_INTERVAL", "60"))
    
    # Create and run service
    service = StrategicMemorySync(redis_url=redis_url, interval=interval)
    
    try:
        service.run()
    except Exception as e:
        logger.error("Fatal error in main", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
