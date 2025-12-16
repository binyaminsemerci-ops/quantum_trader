"""
Redis Opportunity Store for OpportunityRanker
Implements OpportunityStore protocol
"""

import logging
import json
from typing import Optional
from datetime import datetime
import redis
from backend.services.opportunity_ranker import OpportunityRanking

logger = logging.getLogger(__name__)


class RedisOpportunityStore:
    """Real implementation of OpportunityStore using Redis."""
    
    # TTL for rankings: 1 hour
    DEFAULT_TTL = 3600
    
    # Key prefix for Redis
    KEY_PREFIX = "opportunity_rank:"
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize with Redis client.
        
        Args:
            redis_client: Configured Redis client instance
        """
        self.redis = redis_client
        logger.info("RedisOpportunityStore initialized")
    
    def update(self, ranking: OpportunityRanking) -> None:
        """
        Store opportunity ranking in Redis.
        
        Args:
            ranking: OpportunityRanking to store
        """
        try:
            # Create key
            key = f"{self.KEY_PREFIX}{ranking.symbol}"
            
            # Serialize to JSON
            data = {
                "symbol": ranking.symbol,
                "overall_score": ranking.overall_score,
                "rank": ranking.rank,
                "metric_scores": ranking.metric_scores,
                "metadata": ranking.metadata,
                "timestamp": ranking.timestamp.isoformat()
            }
            
            # Store in Redis with TTL
            self.redis.setex(
                key,
                self.DEFAULT_TTL,
                json.dumps(data)
            )
            
            logger.debug(f"Stored ranking for {ranking.symbol}: {ranking.overall_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to store ranking for {ranking.symbol}: {e}")
            raise
    
    def get(self, symbol: str) -> Optional[OpportunityRanking]:
        """
        Retrieve opportunity ranking from Redis.
        
        Args:
            symbol: Trading pair
            
        Returns:
            OpportunityRanking if found, else None
        """
        try:
            # Get from Redis
            key = f"{self.KEY_PREFIX}{symbol}"
            data_str = self.redis.get(key)
            
            if not data_str:
                logger.debug(f"No ranking found for {symbol}")
                return None
            
            # Deserialize from JSON
            data = json.loads(data_str)
            
            # Reconstruct OpportunityRanking
            ranking = OpportunityRanking(
                symbol=data["symbol"],
                overall_score=data["overall_score"],
                rank=data["rank"],
                metric_scores=data["metric_scores"],
                metadata=data.get("metadata", {}),
                timestamp=datetime.fromisoformat(data["timestamp"])
            )
            
            return ranking
            
        except Exception as e:
            logger.error(f"Failed to retrieve ranking for {symbol}: {e}")
            return None
    
    def get_all(self) -> list[OpportunityRanking]:
        """
        Get all stored rankings.
        
        Returns:
            List of OpportunityRanking objects
        """
        try:
            # Find all ranking keys
            pattern = f"{self.KEY_PREFIX}*"
            keys = self.redis.keys(pattern)
            
            rankings = []
            for key in keys:
                try:
                    data_str = self.redis.get(key)
                    if data_str:
                        data = json.loads(data_str)
                        ranking = OpportunityRanking(
                            symbol=data["symbol"],
                            overall_score=data["overall_score"],
                            rank=data["rank"],
                            metric_scores=data["metric_scores"],
                            metadata=data.get("metadata", {}),
                            timestamp=datetime.fromisoformat(data["timestamp"])
                        )
                        rankings.append(ranking)
                except Exception as e:
                    logger.warning(f"Failed to parse ranking for key {key}: {e}")
                    continue
            
            # Sort by rank
            rankings.sort(key=lambda r: r.rank)
            
            logger.debug(f"Retrieved {len(rankings)} rankings from Redis")
            return rankings
            
        except Exception as e:
            logger.error(f"Failed to retrieve all rankings: {e}")
            return []
    
    def clear_all(self) -> None:
        """Delete all stored rankings."""
        try:
            pattern = f"{self.KEY_PREFIX}*"
            keys = self.redis.keys(pattern)
            
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} rankings from Redis")
            else:
                logger.debug("No rankings to clear")
                
        except Exception as e:
            logger.error(f"Failed to clear rankings: {e}")
            raise
