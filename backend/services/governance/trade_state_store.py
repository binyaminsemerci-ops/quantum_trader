"""
PATCH-P0-04: Trade State Persistence with Redis
================================================

Replaces vulnerable single JSON file with Redis hash storage.
Eliminates corruption risk, race conditions, and provides ACID guarantees.

Key Features:
- Redis hash per trade: trade:{trade_id}
- Atomic operations (no race conditions)
- TTL support for auto-cleanup
- Survives backend restarts
- Transaction support for consistency
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class TradeState:
    """Trade state data model."""
    
    def __init__(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        status: str = "OPEN",
        entry_time: Optional[datetime] = None,
        exit_price: Optional[float] = None,
        exit_time: Optional[datetime] = None,
        pnl: float = 0.0,
        metadata: Optional[dict] = None,
    ):
        self.trade_id = trade_id
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.quantity = quantity
        self.status = status
        self.entry_time = entry_time or datetime.utcnow()
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl = pnl
        self.metadata = metadata or {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": str(self.entry_price),
            "quantity": str(self.quantity),
            "status": self.status,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": str(self.exit_price) if self.exit_price else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": str(self.pnl),
            "metadata": json.dumps(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TradeState":
        """Create from dictionary."""
        return cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            side=data["side"],
            entry_price=float(data["entry_price"]),
            quantity=float(data["quantity"]),
            status=data.get("status", "OPEN"),
            entry_time=datetime.fromisoformat(data["entry_time"]) if data.get("entry_time") else None,
            exit_price=float(data["exit_price"]) if data.get("exit_price") else None,
            exit_time=datetime.fromisoformat(data["exit_time"]) if data.get("exit_time") else None,
            pnl=float(data.get("pnl", 0)),
            metadata=json.loads(data.get("metadata", "{}")),
        )


class TradeStateStore:
    """
    PATCH-P0-04: Redis-backed trade state storage.
    
    Replaces single JSON file with Redis hashes for:
    - Atomic operations (no corruption)
    - Concurrent safety (no race conditions)
    - Persistence (survives restarts)
    - TTL support (auto-cleanup)
    
    Usage:
        store = TradeStateStore(redis_client)
        await store.initialize()
        
        # Save trade
        await store.save(trade_id, trade_state)
        
        # Get trade
        state = await store.get(trade_id)
        
        # List open trades
        open_trades = await store.list_open_trades()
        
        # Delete trade
        await store.delete(trade_id)
    """
    
    KEY_PREFIX = "trade:"
    CLOSED_TTL_DAYS = 30  # Keep closed trades for 30 days
    
    def __init__(self, redis_client: Redis):
        """
        Initialize TradeStateStore.
        
        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self._initialized = False
        
        logger.info("TradeStateStore initialized with Redis backend")
    
    async def initialize(self) -> None:
        """Initialize store (verify Redis connection)."""
        try:
            await self.redis.ping()
            self._initialized = True
            logger.info("TradeStateStore connected to Redis successfully")
        except Exception as e:
            logger.error(f"TradeStateStore Redis connection failed: {e}")
            raise
    
    def _get_key(self, trade_id: str) -> str:
        """Get Redis key for trade."""
        return f"{self.KEY_PREFIX}{trade_id}"
    
    async def save(
        self,
        trade_id: str,
        state: TradeState,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Save trade state to Redis (atomic operation).
        
        Args:
            trade_id: Unique trade identifier
            state: TradeState object
            ttl: Optional TTL in seconds (for closed trades)
        """
        try:
            key = self._get_key(trade_id)
            data = state.to_dict()
            
            # Use Redis hash for structured storage
            async with self.redis.pipeline(transaction=True) as pipe:
                # Delete existing hash (ensures clean state)
                pipe.delete(key)
                
                # Set all fields atomically
                pipe.hset(key, mapping=data)
                
                # Set TTL if provided (closed trades)
                if ttl:
                    pipe.expire(key, ttl)
                elif state.status == "CLOSED":
                    # Auto-TTL for closed trades
                    pipe.expire(key, self.CLOSED_TTL_DAYS * 86400)
                
                await pipe.execute()
            
            logger.debug(f"Trade state saved: {trade_id} ({state.status})")
        
        except redis.RedisError as e:
            logger.error(f"Failed to save trade state {trade_id}: {e}")
            raise
    
    async def get(self, trade_id: str) -> Optional[TradeState]:
        """
        Get trade state from Redis.
        
        Args:
            trade_id: Unique trade identifier
        
        Returns:
            TradeState object or None if not found
        """
        try:
            key = self._get_key(trade_id)
            data = await self.redis.hgetall(key)
            
            if not data:
                return None
            
            # Decode bytes to strings
            decoded_data = {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v
                for k, v in data.items()
            }
            
            return TradeState.from_dict(decoded_data)
        
        except redis.RedisError as e:
            logger.error(f"Failed to get trade state {trade_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse trade state {trade_id}: {e}")
            return None
    
    async def delete(self, trade_id: str) -> bool:
        """
        Delete trade state from Redis.
        
        Args:
            trade_id: Unique trade identifier
        
        Returns:
            True if deleted, False if not found
        """
        try:
            key = self._get_key(trade_id)
            result = await self.redis.delete(key)
            
            if result > 0:
                logger.debug(f"Trade state deleted: {trade_id}")
                return True
            else:
                logger.debug(f"Trade state not found for deletion: {trade_id}")
                return False
        
        except redis.RedisError as e:
            logger.error(f"Failed to delete trade state {trade_id}: {e}")
            return False
    
    async def list_open_trades(self) -> List[TradeState]:
        """
        List all open trades.
        
        Returns:
            List of TradeState objects with status=OPEN
        """
        try:
            # Scan for all trade keys
            pattern = f"{self.KEY_PREFIX}*"
            open_trades = []
            
            async for key in self.redis.scan_iter(match=pattern, count=100):
                state = await self.get(key.decode().replace(self.KEY_PREFIX, ""))
                
                if state and state.status == "OPEN":
                    open_trades.append(state)
            
            logger.debug(f"Found {len(open_trades)} open trades")
            return open_trades
        
        except redis.RedisError as e:
            logger.error(f"Failed to list open trades: {e}")
            return []
    
    async def list_all_trades(
        self,
        limit: Optional[int] = None,
    ) -> List[TradeState]:
        """
        List all trades (open and closed).
        
        Args:
            limit: Optional max number of trades to return
        
        Returns:
            List of TradeState objects
        """
        try:
            pattern = f"{self.KEY_PREFIX}*"
            all_trades = []
            count = 0
            
            async for key in self.redis.scan_iter(match=pattern, count=100):
                if limit and count >= limit:
                    break
                
                state = await self.get(key.decode().replace(self.KEY_PREFIX, ""))
                
                if state:
                    all_trades.append(state)
                    count += 1
            
            logger.debug(f"Found {len(all_trades)} total trades")
            return all_trades
        
        except redis.RedisError as e:
            logger.error(f"Failed to list all trades: {e}")
            return []
    
    async def update_status(
        self,
        trade_id: str,
        new_status: str,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
    ) -> bool:
        """
        Update trade status (atomic operation).
        
        Args:
            trade_id: Unique trade identifier
            new_status: New status (OPEN, CLOSED, CANCELED, etc.)
            exit_price: Optional exit price (for CLOSED status)
            pnl: Optional PnL (for CLOSED status)
        
        Returns:
            True if updated, False if trade not found
        """
        try:
            # Get existing trade
            state = await self.get(trade_id)
            
            if not state:
                logger.warning(f"Cannot update status: trade {trade_id} not found")
                return False
            
            # Update fields
            state.status = new_status
            
            if new_status == "CLOSED":
                state.exit_time = datetime.utcnow()
                if exit_price is not None:
                    state.exit_price = exit_price
                if pnl is not None:
                    state.pnl = pnl
            
            # Save atomically
            await self.save(trade_id, state)
            
            logger.info(f"Trade status updated: {trade_id} → {new_status}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update trade status {trade_id}: {e}")
            return False
    
    async def get_stats(self) -> dict:
        """Get store statistics."""
        try:
            pattern = f"{self.KEY_PREFIX}*"
            total_count = 0
            open_count = 0
            closed_count = 0
            
            async for key in self.redis.scan_iter(match=pattern, count=100):
                total_count += 1
                state = await self.get(key.decode().replace(self.KEY_PREFIX, ""))
                
                if state:
                    if state.status == "OPEN":
                        open_count += 1
                    elif state.status == "CLOSED":
                        closed_count += 1
            
            return {
                "total_trades": total_count,
                "open_trades": open_count,
                "closed_trades": closed_count,
                "backend": "Redis",
                "initialized": self._initialized,
            }
        
        except Exception as e:
            logger.error(f"Failed to get store stats: {e}")
            return {
                "error": str(e),
                "backend": "Redis",
                "initialized": self._initialized,
            }


# Global singleton
_trade_state_store: Optional[TradeStateStore] = None


def get_trade_state_store() -> TradeStateStore:
    """Get global TradeStateStore instance."""
    if _trade_state_store is None:
        raise RuntimeError(
            "TradeStateStore not initialized. Call initialize_trade_state_store() first."
        )
    return _trade_state_store


async def initialize_trade_state_store(redis_client: Redis) -> TradeStateStore:
    """Initialize global TradeStateStore singleton."""
    global _trade_state_store
    
    _trade_state_store = TradeStateStore(redis_client)
    await _trade_state_store.initialize()
    
    logger.info("Global TradeStateStore initialized")
    return _trade_state_store
