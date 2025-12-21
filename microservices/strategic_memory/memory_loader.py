"""
Memory Loader - Fetches data from Redis streams and keys
"""
import redis
import json
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger()


class MemoryLoader:
    """Retrieves strategic memory data from Redis streams and keys"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis = redis.from_url(redis_url, decode_responses=False)
        logger.info("MemoryLoader initialized", redis_url=redis_url)
    
    def load(self) -> Dict[str, Any]:
        """
        Load all strategic memory data from Redis
        
        Returns:
            Dictionary containing:
            - portfolio_policy: Current governance policy
            - preferred_regime: Preferred market regime
            - meta_stream: Recent meta-regime observations
            - pnl_stream: Recent portfolio PnL data
            - exposure: Current exposure summary
            - leverage: Current leverage settings
            - exit_stats: Exit brain statistics
        """
        try:
            memory = {
                "portfolio_policy": self._get_key("quantum:governance:policy"),
                "preferred_regime": self._get_key("quantum:governance:preferred_regime"),
                "regime_stats": self._get_json("quantum:governance:regime_stats"),
                "meta_stream": self._get_stream("quantum:stream:meta.regime", count=50),
                "pnl_stream": self._get_stream("quantum:stream:portfolio.memory", count=50),
                "exposure": self._get_json("quantum:exposure:summary"),
                "leverage": self._get_json("quantum:risk:leverage_limits"),
                "exit_stats": self._get_json("quantum:exit:statistics"),
                "recent_trades": self._get_stream("quantum:stream:trade.results", count=30)
            }
            
            samples = sum([
                len(memory.get("meta_stream", [])),
                len(memory.get("pnl_stream", [])),
                len(memory.get("recent_trades", []))
            ])
            
            logger.info(
                "Memory loaded successfully",
                samples=samples,
                has_policy=memory["portfolio_policy"] is not None,
                has_regime=memory["preferred_regime"] is not None
            )
            
            return memory
            
        except Exception as e:
            logger.error("Failed to load memory", error=str(e))
            return self._empty_memory()
    
    def _get_key(self, key: str) -> Optional[str]:
        """Get string value from Redis key"""
        try:
            value = self.redis.get(key)
            if value:
                return value.decode('utf-8') if isinstance(value, bytes) else value
            return None
        except Exception as e:
            logger.warning("Failed to get key", key=key, error=str(e))
            return None
    
    def _get_json(self, key: str) -> Optional[Dict]:
        """Get JSON value from Redis key"""
        try:
            value = self.redis.get(key)
            if value:
                decoded = value.decode('utf-8') if isinstance(value, bytes) else value
                return json.loads(decoded)
            return None
        except Exception as e:
            logger.warning("Failed to get JSON key", key=key, error=str(e))
            return None
    
    def _get_stream(self, stream: str, count: int = 50) -> List[tuple]:
        """Get recent entries from Redis stream"""
        try:
            entries = self.redis.xrevrange(stream, count=count)
            return entries if entries else []
        except Exception as e:
            logger.warning("Failed to get stream", stream=stream, error=str(e))
            return []
    
    def _empty_memory(self) -> Dict[str, Any]:
        """Return empty memory structure"""
        return {
            "portfolio_policy": None,
            "preferred_regime": None,
            "regime_stats": None,
            "meta_stream": [],
            "pnl_stream": [],
            "exposure": None,
            "leverage": None,
            "exit_stats": None,
            "recent_trades": []
        }
