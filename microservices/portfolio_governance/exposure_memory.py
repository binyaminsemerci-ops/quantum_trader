"""
Exposure Memory System
======================
Core memory system for tracking portfolio exposure, PnL, and confidence metrics.

This module provides:
- Rolling window memory of trade events
- Statistical summarization of portfolio performance
- Portfolio Score calculation based on PnL, confidence, and volatility
- Redis Streams integration for distributed event sourcing
"""

import datetime
import redis
import json
from collections import deque
from typing import Dict, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExposureMemory:
    """
    Manages a rolling window of portfolio exposure events.
    
    Tracks:
    - Trade PnL history
    - AI confidence scores
    - Volatility regimes
    - Leverage utilization
    
    Provides portfolio score calculation for governance decisions.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", window: int = 500):
        """
        Initialize Exposure Memory.
        
        Args:
            redis_url: Redis connection URL
            window: Size of rolling window for memory (default 500 events)
        """
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.memory = deque(maxlen=window)
        self.window = window
        logger.info(f"ExposureMemory initialized with window={window}")
    
    def record(self, data: Dict) -> bool:
        """
        Record a new exposure event.
        
        Data format:
        {
            "timestamp": "2025-12-21T12:00:00",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "leverage": 20,
            "pnl": 0.32,
            "confidence": 0.72,
            "volatility": 0.14,
            "position_size": 1000.0,
            "exit_reason": "dynamic_tp"
        }
        
        Args:
            data: Event data dictionary
            
        Returns:
            True if recorded successfully
        """
        try:
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.datetime.utcnow().isoformat()
            
            # Store in local memory
            self.memory.append(data)
            
            # Store in Redis Stream for distributed access
            stream_data = {k: str(v) for k, v in data.items()}
            self.client.xadd("quantum:stream:portfolio.memory", stream_data)
            
            # Update latest portfolio score in Redis
            score = self.get_portfolio_score()
            self.client.set("quantum:governance:score", str(score))
            
            logger.debug(f"Recorded event for {data.get('symbol')}, PnL={data.get('pnl')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record exposure event: {e}")
            return False
    
    def summarize(self) -> Dict:
        """
        Generate statistical summary of exposure memory.
        
        Returns:
            Dictionary with aggregated metrics:
            - samples: Number of events in memory
            - avg_pnl: Average PnL per trade
            - avg_confidence: Average AI confidence
            - avg_volatility: Average market volatility
            - avg_leverage: Average leverage used
            - win_rate: Percentage of profitable trades
        """
        if not self.memory:
            return {}
        
        total_samples = len(self.memory)
        
        # Calculate averages
        avg_pnl = sum(d.get("pnl", 0) for d in self.memory) / total_samples
        avg_conf = sum(d.get("confidence", 0) for d in self.memory) / total_samples
        vol_mean = sum(d.get("volatility", 0) for d in self.memory) / total_samples
        leverage_mean = sum(d.get("leverage", 1) for d in self.memory) / total_samples
        
        # Calculate win rate
        winning_trades = sum(1 for d in self.memory if d.get("pnl", 0) > 0)
        win_rate = winning_trades / total_samples if total_samples > 0 else 0
        
        # Calculate volatility of returns
        pnls = [d.get("pnl", 0) for d in self.memory]
        pnl_variance = sum((p - avg_pnl) ** 2 for p in pnls) / total_samples
        pnl_volatility = pnl_variance ** 0.5
        
        return {
            "samples": total_samples,
            "avg_pnl": round(avg_pnl, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_volatility": round(vol_mean, 4),
            "avg_leverage": round(leverage_mean, 2),
            "win_rate": round(win_rate, 4),
            "pnl_volatility": round(pnl_volatility, 4)
        }
    
    def get_portfolio_score(self) -> float:
        """
        Calculate Portfolio Score based on performance metrics.
        
        Formula:
            score = (avg_pnl * avg_confidence * win_rate) / max(avg_volatility, 0.01)
        
        Higher score indicates:
        - Better PnL performance
        - Higher AI confidence
        - Better win rate
        - Lower market volatility (more stable conditions)
        
        Returns:
            Portfolio score (typically -1.0 to 5.0 range)
        """
        s = self.summarize()
        if not s or s["samples"] < 10:
            return 0.0
        
        # Core logic: performance * confidence * win_rate / volatility
        # Adjusted to penalize high volatility and reward consistent wins
        numerator = s["avg_pnl"] * s["avg_confidence"] * s["win_rate"]
        denominator = max(s["avg_volatility"], 0.01)
        
        score = numerator / denominator
        
        # Normalize to reasonable range
        return round(score, 3)
    
    def get_symbol_stats(self, symbol: str) -> Dict:
        """
        Get statistics for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            
        Returns:
            Dictionary with symbol-specific metrics
        """
        symbol_events = [e for e in self.memory if e.get("symbol") == symbol]
        
        if not symbol_events:
            return {"symbol": symbol, "samples": 0}
        
        total = len(symbol_events)
        avg_pnl = sum(e.get("pnl", 0) for e in symbol_events) / total
        wins = sum(1 for e in symbol_events if e.get("pnl", 0) > 0)
        
        return {
            "symbol": symbol,
            "samples": total,
            "avg_pnl": round(avg_pnl, 4),
            "win_rate": round(wins / total, 4) if total > 0 else 0,
            "avg_leverage": round(sum(e.get("leverage", 1) for e in symbol_events) / total, 2)
        }
    
    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """
        Get most recent exposure events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        return list(self.memory)[-limit:]
    
    def clear_memory(self) -> None:
        """Clear all exposure memory. Use with caution!"""
        self.memory.clear()
        logger.warning("Exposure memory cleared!")
    
    def get_memory_health(self) -> Dict:
        """
        Get health status of exposure memory.
        
        Returns:
            Dictionary with health metrics
        """
        summary = self.summarize()
        score = self.get_portfolio_score()
        
        # Determine health status
        if not summary:
            status = "initializing"
        elif summary["samples"] < 50:
            status = "warming_up"
        elif score < 0.2:
            status = "poor"
        elif score < 0.5:
            status = "moderate"
        elif score < 0.8:
            status = "good"
        else:
            status = "excellent"
        
        return {
            "status": status,
            "score": score,
            "samples": summary.get("samples", 0),
            "window_utilization": round(summary.get("samples", 0) / self.window, 2) if summary else 0
        }
