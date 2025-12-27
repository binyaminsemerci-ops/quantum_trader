"""
Regime Memory - Long-term memory of market regimes and their outcomes.
"""
import redis
import json
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional
import structlog

logger = structlog.get_logger()


class RegimeMemory:
    """Long-term memory for regime analysis and correlation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", maxlen: int = 1000):
        """
        Initialize regime memory.
        
        Args:
            redis_url: Redis connection URL
            maxlen: Maximum buffer size
        """
        self.client = redis.from_url(redis_url)
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
        
        logger.info("RegimeMemory initialized", maxlen=maxlen)
    
    def record(self, regime_data: Dict) -> None:
        """
        Record a regime observation.
        
        Args:
            regime_data: Dictionary with regime, volatility, trend, pnl, etc.
        """
        # Add timestamp
        if "timestamp" not in regime_data:
            regime_data["timestamp"] = datetime.utcnow().isoformat()
        
        # Add to local buffer
        self.buffer.append(regime_data)
        
        # Store in Redis stream
        try:
            self.client.xadd(
                "quantum:stream:meta.regime",
                regime_data,
                maxlen=self.maxlen
            )
        except Exception as e:
            logger.error("Failed to record regime", error=str(e))
    
    def summarize(self) -> Dict:
        """
        Summarize regime performance.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.buffer:
            return {}
        
        by_regime = defaultdict(lambda: {
            "count": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "volatility": [],
            "confidence": []
        })
        
        for data in self.buffer:
            regime = data.get("regime", "UNKNOWN")
            pnl = data.get("pnl", 0.0)
            vol = data.get("volatility", 0.0)
            conf = data.get("confidence", 0.0)
            
            stats = by_regime[regime]
            stats["count"] += 1
            stats["total_pnl"] += pnl
            
            if pnl > 0:
                stats["wins"] += 1
            elif pnl < 0:
                stats["losses"] += 1
            
            stats["volatility"].append(vol)
            stats["confidence"].append(conf)
        
        # Calculate averages
        for regime, stats in by_regime.items():
            if stats["count"] > 0:
                stats["avg_pnl"] = stats["total_pnl"] / stats["count"]
                stats["win_rate"] = stats["wins"] / stats["count"]
                stats["avg_volatility"] = sum(stats["volatility"]) / len(stats["volatility"])
                stats["avg_confidence"] = sum(stats["confidence"]) / len(stats["confidence"])
                
                # Remove raw lists to reduce size
                del stats["volatility"]
                del stats["confidence"]
        
        return dict(by_regime)
    
    def get_best_regime(self) -> Optional[tuple]:
        """
        Get the regime with best average PnL.
        
        Returns:
            Tuple of (regime_name, statistics) or None
        """
        summary = self.summarize()
        if not summary:
            return None
        
        # Filter regimes with at least 5 samples
        valid_regimes = {
            regime: stats 
            for regime, stats in summary.items() 
            if stats["count"] >= 5
        }
        
        if not valid_regimes:
            return None
        
        best = max(valid_regimes.items(), key=lambda kv: kv[1]["avg_pnl"])
        return best
    
    def get_worst_regime(self) -> Optional[tuple]:
        """
        Get the regime with worst average PnL.
        
        Returns:
            Tuple of (regime_name, statistics) or None
        """
        summary = self.summarize()
        if not summary:
            return None
        
        # Filter regimes with at least 5 samples
        valid_regimes = {
            regime: stats 
            for regime, stats in summary.items() 
            if stats["count"] >= 5
        }
        
        if not valid_regimes:
            return None
        
        worst = min(valid_regimes.items(), key=lambda kv: kv[1]["avg_pnl"])
        return worst
    
    def get_regime_stats(self, regime: str) -> Optional[Dict]:
        """
        Get statistics for a specific regime.
        
        Args:
            regime: Regime name
            
        Returns:
            Statistics dictionary or None
        """
        summary = self.summarize()
        return summary.get(regime)
    
    def clear(self) -> None:
        """Clear all regime memory"""
        self.buffer.clear()
        try:
            self.client.delete("quantum:stream:meta.regime")
            logger.info("RegimeMemory cleared")
        except Exception as e:
            logger.error("Failed to clear regime memory", error=str(e))
