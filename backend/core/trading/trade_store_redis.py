"""
SPRINT 1 - D5: Redis TradeStore Backend
=======================================

High-performance trade persistence using Redis.

Features:
- Redis hash storage (trade:{id})
- Atomic operations
- TTL support for closed trades
- Concurrent-safe
- Fast queries
- Survives restarts (if Redis has persistence enabled)

Adapted from existing TradeStateStore with enhanced Trade model support.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.core.trading.trade_store_base import (
    Trade,
    TradeStore,
    TradeStoreBase,
    TradeStatus,
    TradeSide,
)

logger = logging.getLogger(__name__)


class TradeStoreRedis(TradeStoreBase):
    """
    Redis-backed trade storage.
    
    High-performance primary storage when Redis is available.
    Uses Redis hashes for structured storage with atomic operations.
    """
    
    KEY_PREFIX = "trade:"
    CLOSED_TTL_DAYS = 30  # Keep closed trades for 30 days
    
    def __init__(self, redis_client: Redis):
        """
        Initialize Redis TradeStore.
        
        Args:
            redis_client: Async Redis client
        """
        super().__init__()
        self.backend_name = "Redis"
        self.redis = redis_client
        
        logger.info("[TradeStore] Redis backend configured")
    
    async def initialize(self) -> None:
        """Initialize and verify Redis connection."""
        try:
            await self.redis.ping()
            self._initialized = True
            logger.info("[TradeStore] Redis backend initialized and connected")
        except Exception as e:
            logger.error(f"[TradeStore] Redis initialization failed: {e}")
            raise
    
    def _get_key(self, trade_id: str) -> str:
        """Get Redis key for trade."""
        return f"{self.KEY_PREFIX}{trade_id}"
    
    async def save_new_trade(self, trade: Trade) -> None:
        """
        Save a new trade to Redis (atomic operation).
        
        Uses Redis hash for structured storage with optional TTL.
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            key = self._get_key(trade.trade_id)
            data = trade.to_dict()
            
            # Convert all values to strings for Redis hash
            redis_data = {
                k: str(v) if v is not None else ""
                for k, v in data.items()
            }
            
            # Atomic operation: delete old + set new + set TTL
            async with self.redis.pipeline(transaction=True) as pipe:
                # Delete existing hash (ensures clean state)
                pipe.delete(key)
                
                # Set all fields atomically
                pipe.hset(key, mapping=redis_data)
                
                # Set TTL for closed trades
                if trade.status == TradeStatus.CLOSED:
                    pipe.expire(key, self.CLOSED_TTL_DAYS * 86400)
                
                await pipe.execute()
            
            logger.debug(
                f"[TradeStore] Saved trade to Redis: {trade.trade_id} | "
                f"{trade.symbol} {trade.side.value} | ${trade.margin_usd:.2f}"
            )
        
        except redis.RedisError as e:
            logger.error(f"[TradeStore] Failed to save trade {trade.trade_id} to Redis: {e}")
            raise
    
    async def update_trade(self, trade_id: str, **fields) -> bool:
        """
        Update specific fields of an existing trade.
        
        Uses Redis HSET for atomic field updates.
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        if not fields:
            return False
        
        try:
            key = self._get_key(trade_id)
            
            # Check if trade exists
            exists = await self.redis.exists(key)
            if not exists:
                logger.warning(f"[TradeStore] Trade not found for update: {trade_id}")
                return False
            
            # Add updated_at timestamp
            fields['updated_at'] = datetime.utcnow().isoformat()
            
            # Convert values to strings
            redis_fields = {
                k: str(v) if v is not None else ""
                for k, v in fields.items()
            }
            
            # Update fields atomically
            await self.redis.hset(key, mapping=redis_fields)
            
            logger.debug(f"[TradeStore] Updated trade {trade_id}: {list(fields.keys())}")
            return True
        
        except redis.RedisError as e:
            logger.error(f"[TradeStore] Failed to update trade {trade_id}: {e}")
            return False
    
    async def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Retrieve trade by ID from Redis.
        
        Returns:
            Trade object or None if not found
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            key = self._get_key(trade_id)
            data = await self.redis.hgetall(key)
            
            if not data:
                return None
            
            # Decode bytes to strings and convert to proper types
            decoded_data = {}
            for k, v in data.items():
                key_str = k.decode() if isinstance(k, bytes) else k
                val_str = v.decode() if isinstance(v, bytes) else v
                
                # Convert empty strings back to None
                decoded_data[key_str] = val_str if val_str != "" else None
            
            # Convert string numbers back to floats/ints
            for key in ['quantity', 'leverage', 'margin_usd', 'entry_price', 'sl_price', 
                       'tp_price', 'trail_percent', 'exit_price', 'pnl_usd', 'pnl_pct', 
                       'r_multiple', 'entry_fee_usd', 'exit_fee_usd', 'funding_fees_usd',
                       'confidence', 'rl_leverage_original']:
                if decoded_data.get(key):
                    try:
                        decoded_data[key] = float(decoded_data[key])
                    except (ValueError, TypeError):
                        pass
            
            return Trade.from_dict(decoded_data)
        
        except redis.RedisError as e:
            logger.error(f"[TradeStore] Failed to get trade {trade_id} from Redis: {e}")
            return None
        except Exception as e:
            logger.error(f"[TradeStore] Failed to parse trade {trade_id}: {e}")
            return None
    
    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Trade]:
        """
        Get all open trades, optionally filtered by symbol.
        
        Uses Redis SCAN to iterate through all trade keys.
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            pattern = f"{self.KEY_PREFIX}*"
            open_trades = []
            
            # Scan for all trade keys
            async for key in self.redis.scan_iter(match=pattern, count=100):
                trade_id = key.decode().replace(self.KEY_PREFIX, "")
                trade = await self.get_trade_by_id(trade_id)
                
                if trade and trade.status == TradeStatus.OPEN:
                    if symbol is None or trade.symbol == symbol:
                        open_trades.append(trade)
            
            logger.debug(
                f"[TradeStore] Found {len(open_trades)} open trades in Redis" +
                (f" for {symbol}" if symbol else "")
            )
            return open_trades
        
        except redis.RedisError as e:
            logger.error(f"[TradeStore] Failed to get open trades from Redis: {e}")
            return []
    
    async def mark_trade_closed(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        close_reason: str,
        exit_fee_usd: float = 0.0
    ) -> bool:
        """
        Mark trade as closed and calculate final PnL.
        
        Atomically updates trade and sets TTL for auto-cleanup.
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            # Get existing trade
            trade = await self.get_trade_by_id(trade_id)
            
            if not trade:
                logger.warning(f"[TradeStore] Trade not found for close: {trade_id}")
                return False
            
            # Update trade with exit info and calculate PnL
            trade.update_exit(exit_price, exit_time, close_reason, exit_fee_usd)
            
            # Save updated trade (with TTL for closed status)
            await self.save_new_trade(trade)
            
            logger.info(
                f"[TradeStore] Closed trade {trade_id}: "
                f"PnL=${trade.pnl_usd:.2f} ({trade.pnl_pct:.2f}%), R={trade.r_multiple:.2f}"
            )
            return True
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to close trade {trade_id}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self._initialized:
            return {
                "error": "Not initialized",
                "backend": "Redis",
                "initialized": False,
            }
        
        try:
            pattern = f"{self.KEY_PREFIX}*"
            total_count = 0
            open_count = 0
            closed_count = 0
            total_pnl = 0.0
            
            # Scan all trade keys
            async for key in self.redis.scan_iter(match=pattern, count=100):
                total_count += 1
                trade_id = key.decode().replace(self.KEY_PREFIX, "")
                trade = await self.get_trade_by_id(trade_id)
                
                if trade:
                    if trade.status == TradeStatus.OPEN:
                        open_count += 1
                    elif trade.status == TradeStatus.CLOSED:
                        closed_count += 1
                        total_pnl += trade.pnl_usd
            
            return {
                "total_trades": total_count,
                "open_trades": open_count,
                "closed_trades": closed_count,
                "total_pnl_usd": total_pnl,
                "backend": "Redis",
                "initialized": self._initialized,
            }
        
        except Exception as e:
            logger.error(f"[TradeStore] Failed to get stats from Redis: {e}")
            return {
                "error": str(e),
                "backend": "Redis",
                "initialized": self._initialized,
            }
    
    async def list_all_trades(
        self,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """
        List all trades (open and closed).
        
        Args:
            limit: Optional max number of trades to return
        
        Returns:
            List of Trade objects
        """
        if not self._initialized:
            raise RuntimeError("TradeStore not initialized")
        
        try:
            pattern = f"{self.KEY_PREFIX}*"
            all_trades = []
            count = 0
            
            async for key in self.redis.scan_iter(match=pattern, count=100):
                if limit and count >= limit:
                    break
                
                trade_id = key.decode().replace(self.KEY_PREFIX, "")
                trade = await self.get_trade_by_id(trade_id)
                
                if trade:
                    all_trades.append(trade)
                    count += 1
            
            logger.debug(f"[TradeStore] Found {len(all_trades)} total trades in Redis")
            return all_trades
        
        except redis.RedisError as e:
            logger.error(f"[TradeStore] Failed to list trades from Redis: {e}")
            return []
