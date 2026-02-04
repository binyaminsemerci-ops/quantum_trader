#!/usr/bin/env python3
"""
Performance Tracker - Symbol Profitability Analytics
====================================================

Tracks historical trading performance per symbol:
- Win rate (% profitable trades)
- Avg P&L% per trade
- Sharpe ratio (risk-adjusted returns)
- Total trades count

Writes to: quantum:symbol:performance:{symbol}

Used by: AI Universe Generator (30% weight in hybrid scoring)

Author: Quantum Trader Team
Date: 2026-02-04
"""
import os
import sys
import time
import json
import math
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed")
    sys.exit(1)

import numpy as np

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

SCAN_INTERVAL = int(os.getenv("PERF_TRACKER_INTERVAL", "300"))  # 5 minutes
MIN_TRADES_FOR_STATS = int(os.getenv("MIN_TRADES_FOR_STATS", "5"))
LOOKBACK_DAYS = int(os.getenv("PERF_LOOKBACK_DAYS", "30"))  # Last 30 days

LOG_LEVEL = os.getenv("PERF_TRACKER_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [PERF-TRACKER] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        
    def get_all_traded_symbols(self) -> List[str]:
        """Get list of all symbols with ledger entries"""
        try:
            pattern = "quantum:ledger:*"
            keys = self.redis.keys(pattern)
            symbols = [k.decode().replace("quantum:ledger:", "") for k in keys]
            return symbols
        except Exception as e:
            logger.error(f"Failed to get traded symbols: {e}")
            return []
    
    def get_ledger_data(self, symbol: str) -> Optional[Dict]:
        """Get current ledger data for symbol"""
        try:
            key = f"quantum:ledger:{symbol}"
            data = self.redis.hgetall(key)
            
            if not data:
                return None
            
            # Parse relevant fields
            ledger = {
                "symbol": symbol,
                "total_trades": int(data.get(b"total_trades", b"0").decode()),
                "winning_trades": int(data.get(b"winning_trades", b"0").decode()),
                "losing_trades": int(data.get(b"losing_trades", b"0").decode()),
                "total_pnl_usdt": float(data.get(b"total_pnl_usdt", b"0.0").decode()),
                "total_fees_usdt": float(data.get(b"total_fees_usdt", b"0.0").decode()),
                "total_volume_usdt": float(data.get(b"total_volume_usdt", b"0.0").decode()),
                "realized_pnl": float(data.get(b"realized_pnl", b"0.0").decode()),
            }
            
            # Parse trade history if available
            trade_history_str = data.get(b"trade_history", b"[]").decode()
            try:
                ledger["trade_history"] = json.loads(trade_history_str)
            except:
                ledger["trade_history"] = []
            
            return ledger
            
        except Exception as e:
            logger.warning(f"Failed to get ledger for {symbol}: {e}")
            return None
    
    def calculate_performance_metrics(self, symbol: str, ledger: Dict) -> Optional[Dict]:
        """
        Calculate performance metrics from ledger data
        
        Returns:
            win_rate: 0.0-1.0 (fraction of profitable trades)
            avg_pnl_pct: Average P&L% per trade
            sharpe_ratio: Risk-adjusted return (annualized)
            total_trades: Number of completed trades
        """
        try:
            total_trades = ledger["total_trades"]
            winning_trades = ledger["winning_trades"]
            losing_trades = ledger["losing_trades"]
            
            if total_trades < MIN_TRADES_FOR_STATS:
                logger.debug(f"{symbol}: Only {total_trades} trades, need {MIN_TRADES_FOR_STATS}")
                return None
            
            # Win rate
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
            
            # Average P&L%
            total_pnl = ledger["total_pnl_usdt"]
            total_volume = ledger["total_volume_usdt"]
            avg_pnl_pct = (total_pnl / total_volume * 100) if total_volume > 0 else 0.0
            
            # Sharpe ratio (need trade-by-trade P&L for accurate calculation)
            trade_history = ledger.get("trade_history", [])
            
            if len(trade_history) >= MIN_TRADES_FOR_STATS:
                # Filter recent trades (last 30 days)
                cutoff_ts = time.time() - (LOOKBACK_DAYS * 86400)
                recent_trades = [
                    t for t in trade_history 
                    if t.get("timestamp", 0) > cutoff_ts
                ]
                
                if len(recent_trades) >= MIN_TRADES_FOR_STATS:
                    returns = [t.get("pnl_pct", 0.0) for t in recent_trades]
                    mean_return = np.mean(returns)
                    std_return = np.std(returns) if len(returns) > 1 else 0.0
                    
                    # Annualized Sharpe (assuming ~100 trades/year for futures)
                    sharpe_ratio = (mean_return / std_return * math.sqrt(100)) if std_return > 0 else 0.0
                else:
                    # Fallback: simple estimate
                    sharpe_ratio = (avg_pnl_pct / 2.0) if avg_pnl_pct > 0 else 0.0
            else:
                # Fallback: simple estimate based on win rate
                sharpe_ratio = (win_rate - 0.5) * 2.0  # Range: -1.0 to 1.0
            
            metrics = {
                "win_rate": round(win_rate, 4),
                "avg_pnl_pct": round(avg_pnl_pct, 4),
                "sharpe_ratio": round(sharpe_ratio, 4),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl_usdt": round(total_pnl, 2),
                "last_updated": int(time.time())
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {symbol}: {e}")
            return None
    
    def save_performance(self, symbol: str, metrics: Dict) -> bool:
        """Save performance metrics to Redis"""
        try:
            key = f"quantum:symbol:performance:{symbol}"
            
            # Convert to bytes
            data = {
                b"win_rate": str(metrics["win_rate"]).encode(),
                b"avg_pnl_pct": str(metrics["avg_pnl_pct"]).encode(),
                b"sharpe_ratio": str(metrics["sharpe_ratio"]).encode(),
                b"total_trades": str(metrics["total_trades"]).encode(),
                b"winning_trades": str(metrics["winning_trades"]).encode(),
                b"losing_trades": str(metrics["losing_trades"]).encode(),
                b"total_pnl_usdt": str(metrics["total_pnl_usdt"]).encode(),
                b"last_updated": str(metrics["last_updated"]).encode()
            }
            
            self.redis.hset(key, mapping=data)
            
            logger.info(
                f"✅ {symbol}: win_rate={metrics['win_rate']:.2%} "
                f"avg_pnl={metrics['avg_pnl_pct']:.2f}% "
                f"sharpe={metrics['sharpe_ratio']:.2f} "
                f"trades={metrics['total_trades']}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save performance for {symbol}: {e}")
            return False
    
    def process_symbol(self, symbol: str) -> bool:
        """Process one symbol: read ledger, calculate metrics, save performance"""
        try:
            # Get ledger data
            ledger = self.get_ledger_data(symbol)
            if not ledger:
                logger.debug(f"{symbol}: No ledger data")
                return False
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics(symbol, ledger)
            if not metrics:
                logger.debug(f"{symbol}: Insufficient data for metrics")
                return False
            
            # Save to Redis
            return self.save_performance(symbol, metrics)
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            return False
    
    def run_scan_cycle(self):
        """Run one complete scan of all traded symbols"""
        try:
            logger.info("Starting performance scan...")
            start_time = time.time()
            
            # Get all traded symbols
            symbols = self.get_all_traded_symbols()
            logger.info(f"Found {len(symbols)} symbols with ledger data")
            
            if not symbols:
                logger.warning("No symbols found with ledger data")
                return
            
            # Process each symbol
            processed = 0
            updated = 0
            
            for symbol in symbols:
                processed += 1
                if self.process_symbol(symbol):
                    updated += 1
                
                # Progress logging every 10 symbols
                if processed % 10 == 0:
                    logger.info(f"Progress: {processed}/{len(symbols)} symbols processed")
            
            elapsed = time.time() - start_time
            logger.info(
                f"✅ Scan complete: {processed} processed, {updated} updated "
                f"in {elapsed:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"Scan cycle failed: {e}")
    
    def run_forever(self):
        """Main loop: scan periodically"""
        logger.info(f"Performance Tracker started (scan interval: {SCAN_INTERVAL}s)")
        logger.info(f"Min trades for stats: {MIN_TRADES_FOR_STATS}")
        logger.info(f"Lookback period: {LOOKBACK_DAYS} days")
        
        while True:
            try:
                self.run_scan_cycle()
                
                logger.info(f"Next scan in {SCAN_INTERVAL}s...")
                time.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(60)  # Wait 1 min on error


def main():
    logger.info("=" * 70)
    logger.info("PERFORMANCE TRACKER - Symbol Profitability Analytics")
    logger.info("=" * 70)
    
    tracker = PerformanceTracker()
    tracker.run_forever()


if __name__ == "__main__":
    main()
