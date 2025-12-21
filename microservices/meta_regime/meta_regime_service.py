"""
Meta-Regime Service - Main service for regime detection and correlation.
"""
import os
import time
import signal
import sys
from datetime import datetime
import pandas as pd
import structlog
import redis
import json
from regime_detector import RegimeDetector
from regime_memory import RegimeMemory
from correlator import MetaRegimeCorrelator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


class MetaRegimeService:
    """Main service for meta-regime analysis"""
    
    def __init__(self):
        """Initialize the meta-regime service"""
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = redis.from_url(self.redis_url)
        
        self.detector = RegimeDetector()
        self.memory = RegimeMemory(redis_url=self.redis_url)
        self.correlator = MetaRegimeCorrelator(redis_url=self.redis_url)
        
        self.interval = int(os.getenv("REGIME_INTERVAL", "30"))
        self.running = True
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
        
        logger.info(
            "MetaRegimeService initialized",
            redis_url=self.redis_url,
            interval=self.interval
        )
    
    def _shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        logger.info("Shutdown signal received", signal=signum)
        self.running = False
    
    def get_market_data(self, symbol: str = "BTCUSDT") -> pd.Series:
        """
        Get market data from Redis stream (populated by cross-exchange).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Price series
        """
        try:
            # Read from cross-exchange stream: quantum:stream:exchange.raw
            # Format: {exchange, symbol, timestamp, open, high, low, close, volume}
            stream_data = self.redis_client.xrevrange(
                "quantum:stream:exchange.raw",
                count=500  # Get more entries to filter by symbol
            )
            
            if not stream_data:
                logger.warning("No market data available", symbol=symbol)
                return None
            
            # Filter and extract close prices for the specified symbol
            prices = []
            for entry_id, data in stream_data:
                if data.get(b'symbol', b'').decode() == symbol:
                    close_price = data.get(b'close', b'0').decode()
                    if close_price and close_price != '0':
                        prices.append(float(close_price))
            
            if len(prices) >= 50:
                prices.reverse()  # Oldest to newest
                return pd.Series(prices)
            
            # If insufficient data for requested symbol
            logger.warning(
                "Insufficient market data",
                symbol=symbol,
                found=len(prices),
                required=50
            )
            return None
            
        except Exception as e:
            logger.error("Failed to get market data", error=str(e), symbol=symbol)
            return None
    
    def get_portfolio_pnl(self) -> float:
        """
        Get current portfolio PnL from governance or portfolio intelligence.
        
        Returns:
            Current PnL value
        """
        try:
            # Try to get from portfolio governance memory
            score_key = "quantum:governance:portfolio_score"
            score = self.redis_client.get(score_key)
            
            if score:
                # Convert score (0-1) to approximate PnL (-0.5 to 0.5)
                score_val = float(score)
                return (score_val - 0.5)
            
            # Try to get from portfolio intelligence
            pnl_key = "quantum:portfolio:total_pnl"
            pnl = self.redis_client.get(pnl_key)
            
            if pnl:
                return float(pnl)
            
            # Default to 0 if no data
            return 0.0
            
        except Exception as e:
            logger.error("Failed to get portfolio PnL", error=str(e))
            return 0.0
    
    def run(self):
        """Main service loop"""
        logger.info("==============================================")
        logger.info("Meta-Regime Service - Starting")
        logger.info("==============================================")
        logger.info(f"Start time: {datetime.utcnow().isoformat()}")
        logger.info(f"Redis: {self.redis_url}")
        logger.info(f"Analysis interval: {self.interval}s")
        
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                start_time = time.time()
                
                # Get market data
                prices = self.get_market_data()
                
                if prices is not None and len(prices) >= 50:
                    # Detect regime
                    regime_info = self.detector.detect(prices)
                    
                    # Get portfolio PnL
                    pnl = self.get_portfolio_pnl()
                    regime_info["pnl"] = pnl
                    regime_info["iteration"] = iteration
                    
                    # Record in memory
                    self.memory.record(regime_info)
                    
                    # Correlate and update governance
                    correlation = self.correlator.correlate()
                    
                    # Update governance policy based on regime
                    policy_updated = self.correlator.update_governance_from_regime(regime_info)
                    
                    # Log summary
                    logger.info(
                        "Regime analysis complete",
                        iteration=iteration,
                        regime=regime_info["regime"],
                        volatility=regime_info["volatility"],
                        trend=regime_info["trend"],
                        confidence=regime_info["confidence"],
                        pnl=pnl,
                        policy_updated=policy_updated,
                        duration_ms=(time.time() - start_time) * 1000
                    )
                    
                    if correlation:
                        best_regime = correlation.get("best_regime")
                        best_stats = correlation.get("statistics", {})
                        logger.info(
                            "Best performing regime",
                            regime=best_regime,
                            avg_pnl=best_stats.get("avg_pnl", 0),
                            win_rate=best_stats.get("win_rate", 0),
                            samples=best_stats.get("count", 0)
                        )
                
                else:
                    logger.warning(
                        "Insufficient market data for regime detection",
                        iteration=iteration,
                        samples=len(prices) if prices is not None else 0
                    )
                
                # Sleep until next iteration
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(
                    "Error in regime analysis loop",
                    error=str(e),
                    iteration=iteration,
                    exc_info=True
                )
                time.sleep(5)  # Short sleep on error
        
        logger.info("Meta-Regime Service shutting down")


def main():
    """Entry point"""
    try:
        service = MetaRegimeService()
        service.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
