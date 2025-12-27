"""Episodic Memory - Event and Episode Storage.

Stores individual events and episodes (trades, system events, risk events,
CEO decisions, strategy shifts) with full context and metadata.

Architecture:
- Redis for short-term buffer (recent events, fast queries)
- Periodic batch to Postgres for long-term storage
- Rich metadata: timestamp, trace_id, regime, risk_mode, global_mode
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class EpisodeType(str, Enum):
    """Types of episodes stored in episodic memory."""
    
    TRADE = "trade"
    SYSTEM_EVENT = "system_event"
    RISK_EVENT = "risk_event"
    CEO_DECISION = "ceo_decision"
    STRATEGY_SHIFT = "strategy_shift"
    REGIME_CHANGE = "regime_change"
    BLACK_SWAN = "black_swan"
    OUTAGE = "outage"
    FAILOVER = "failover"


@dataclass
class Episode:
    """
    Single episode/event with full context.
    
    An episode represents a significant event in the trading system's
    lifecycle, with complete metadata for later analysis.
    """
    
    episode_id: str
    episode_type: EpisodeType
    timestamp: datetime
    
    # Event-specific data
    data: dict[str, Any]
    
    # Context at time of episode
    trace_id: Optional[str] = None
    regime: Optional[str] = None
    risk_mode: Optional[str] = None
    global_mode: Optional[str] = None
    
    # Performance metrics (if applicable)
    pnl: Optional[float] = None
    drawdown: Optional[float] = None
    risk_score: Optional[float] = None
    
    # Tags for querying
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["episode_type"] = self.episode_type.value
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Episode:
        """Create Episode from dictionary."""
        data = data.copy()
        data["episode_type"] = EpisodeType(data["episode_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class EpisodicMemory:
    """
    Episodic memory storage and retrieval.
    
    Features:
    - Store episodes in Redis with TTL (short-term)
    - Periodic batch write to Postgres (long-term)
    - Rich query API:
      - By episode type
      - By time range
      - By context (regime, mode, etc.)
      - By performance (high loss, high profit)
      - By tags
    - Memory consolidation (Redis -> Postgres)
    
    Usage:
        memory = EpisodicMemory(redis_client)
        await memory.initialize()
        
        # Store a trade episode
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            episode_type=EpisodeType.TRADE,
            timestamp=datetime.now(),
            data={
                "symbol": "BTCUSDT",
                "side": "LONG",
                "entry": 43000,
                "exit": 43500,
                "size": 100,
            },
            regime="TRENDING_UP",
            risk_mode="AGGRESSIVE_SMALL_ACCOUNT",
            global_mode="GROWTH",
            pnl=50.0,
            tags=["profitable", "btc"],
        )
        await memory.store_episode(episode)
        
        # Query
        black_swans = await memory.query_by_type(EpisodeType.BLACK_SWAN, days=30)
        high_loss_trades = await memory.query_by_loss_threshold(loss_threshold=100)
    """
    
    REDIS_PREFIX = "quantum:memory:episode:"
    REDIS_INDEX = "quantum:memory:episode_index"
    DEFAULT_TTL = 86400 * 7  # 7 days in Redis
    BATCH_SIZE = 100
    BATCH_INTERVAL = 300  # 5 minutes
    
    def __init__(
        self,
        redis_client: Redis,
        postgres_connection: Optional[Any] = None,
        ttl: int = DEFAULT_TTL,
    ):
        """
        Initialize EpisodicMemory.
        
        Args:
            redis_client: Async Redis client
            postgres_connection: Optional Postgres connection for long-term storage
            ttl: TTL for Redis entries (seconds)
        """
        self.redis = redis_client
        self.postgres = postgres_connection
        self.ttl = ttl
        
        # Background tasks
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        
        # In-memory buffer for batching
        self._buffer: list[Episode] = []
        self._buffer_lock = asyncio.Lock()
        
        logger.info(f"EpisodicMemory initialized: ttl={ttl}s")
    
    async def initialize(self) -> None:
        """Initialize episodic memory and start background tasks."""
        # Verify Redis connection
        await self.redis.ping()
        
        # Create Postgres table if needed
        if self.postgres:
            await self._create_postgres_table()
        
        logger.info("EpisodicMemory initialized successfully")
    
    async def start(self) -> None:
        """Start background batch processing."""
        if self._running:
            logger.warning("EpisodicMemory already running")
            return
        
        self._running = True
        
        # Start batch task
        self._batch_task = asyncio.create_task(self._batch_loop())
        
        logger.info("EpisodicMemory started")
    
    async def stop(self) -> None:
        """Stop background tasks and flush buffer."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel batch task
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining buffer
        await self._flush_buffer()
        
        logger.info("EpisodicMemory stopped")
    
    async def store_episode(self, episode: Episode) -> None:
        """
        Store an episode in Redis and buffer for Postgres.
        
        Args:
            episode: Episode to store
        """
        # Store in Redis with TTL
        key = f"{self.REDIS_PREFIX}{episode.episode_id}"
        data = json.dumps(episode.to_dict())
        
        await self.redis.setex(key, self.ttl, data)
        
        # Add to index for querying
        await self._add_to_index(episode)
        
        # Add to buffer for Postgres batch
        if self.postgres:
            async with self._buffer_lock:
                self._buffer.append(episode)
        
        logger.debug(
            f"Stored episode: type={episode.episode_type.value}, "
            f"id={episode.episode_id}"
        )
    
    async def store_event(
        self,
        event_type: EpisodeType,
        data: dict[str, Any],
        trace_id: Optional[str] = None,
        regime: Optional[str] = None,
        risk_mode: Optional[str] = None,
        global_mode: Optional[str] = None,
        pnl: Optional[float] = None,
        drawdown: Optional[float] = None,
        risk_score: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> Episode:
        """
        Store an event (convenience method).
        
        Args:
            event_type: Type of event
            data: Event-specific data
            trace_id: Optional trace ID
            regime: Optional regime at time of event
            risk_mode: Optional risk mode
            global_mode: Optional global mode
            pnl: Optional PnL
            drawdown: Optional drawdown
            risk_score: Optional risk score
            tags: Optional tags
        
        Returns:
            Created Episode
        """
        episode = Episode(
            episode_id=str(uuid.uuid4()),
            episode_type=event_type,
            timestamp=datetime.now(),
            data=data,
            trace_id=trace_id,
            regime=regime,
            risk_mode=risk_mode,
            global_mode=global_mode,
            pnl=pnl,
            drawdown=drawdown,
            risk_score=risk_score,
            tags=tags or [],
        )
        
        await self.store_episode(episode)
        return episode
    
    async def query_by_type(
        self,
        episode_type: EpisodeType,
        days: int = 7,
        limit: int = 100,
    ) -> list[Episode]:
        """
        Query episodes by type.
        
        Args:
            episode_type: Type of episode
            days: Look back N days
            limit: Maximum number of results
        
        Returns:
            List of episodes
        """
        since = datetime.now() - timedelta(days=days)
        
        # Query from index
        episodes = await self._query_index(
            episode_type=episode_type,
            since=since,
            limit=limit,
        )
        
        return episodes
    
    async def query_by_context(
        self,
        regime: Optional[str] = None,
        risk_mode: Optional[str] = None,
        global_mode: Optional[str] = None,
        days: int = 7,
        limit: int = 100,
    ) -> list[Episode]:
        """
        Query episodes by context.
        
        Args:
            regime: Optional regime filter
            risk_mode: Optional risk mode filter
            global_mode: Optional global mode filter
            days: Look back N days
            limit: Maximum number of results
        
        Returns:
            List of episodes matching context
        """
        since = datetime.now() - timedelta(days=days)
        
        episodes = await self._query_index(
            regime=regime,
            risk_mode=risk_mode,
            global_mode=global_mode,
            since=since,
            limit=limit,
        )
        
        return episodes
    
    async def query_by_loss_threshold(
        self,
        loss_threshold: float,
        days: int = 7,
        limit: int = 100,
    ) -> list[Episode]:
        """
        Query episodes with loss exceeding threshold.
        
        Args:
            loss_threshold: Minimum loss (positive number)
            days: Look back N days
            limit: Maximum number of results
        
        Returns:
            List of episodes with high loss
        """
        since = datetime.now() - timedelta(days=days)
        
        episodes = await self._query_index(
            since=since,
            limit=limit * 2,  # Over-fetch for filtering
        )
        
        # Filter by loss
        high_loss = [
            ep for ep in episodes
            if ep.pnl is not None and ep.pnl < -loss_threshold
        ]
        
        return high_loss[:limit]
    
    async def query_by_profit_threshold(
        self,
        profit_threshold: float,
        days: int = 7,
        limit: int = 100,
    ) -> list[Episode]:
        """
        Query episodes with profit exceeding threshold.
        
        Args:
            profit_threshold: Minimum profit
            days: Look back N days
            limit: Maximum number of results
        
        Returns:
            List of episodes with high profit
        """
        since = datetime.now() - timedelta(days=days)
        
        episodes = await self._query_index(
            since=since,
            limit=limit * 2,
        )
        
        # Filter by profit
        high_profit = [
            ep for ep in episodes
            if ep.pnl is not None and ep.pnl > profit_threshold
        ]
        
        return high_profit[:limit]
    
    async def query_by_tags(
        self,
        tags: list[str],
        days: int = 7,
        limit: int = 100,
        match_all: bool = False,
    ) -> list[Episode]:
        """
        Query episodes by tags.
        
        Args:
            tags: Tags to match
            days: Look back N days
            limit: Maximum number of results
            match_all: If True, require all tags; if False, require any tag
        
        Returns:
            List of episodes matching tags
        """
        since = datetime.now() - timedelta(days=days)
        
        episodes = await self._query_index(
            since=since,
            limit=limit * 2,
        )
        
        # Filter by tags
        if match_all:
            matching = [
                ep for ep in episodes
                if all(tag in ep.tags for tag in tags)
            ]
        else:
            matching = [
                ep for ep in episodes
                if any(tag in ep.tags for tag in tags)
            ]
        
        return matching[:limit]
    
    async def _add_to_index(self, episode: Episode) -> None:
        """Add episode to Redis sorted set index."""
        # Use timestamp as score for chronological sorting
        score = episode.timestamp.timestamp()
        
        # Store episode ID with metadata
        member = json.dumps({
            "episode_id": episode.episode_id,
            "type": episode.episode_type.value,
            "regime": episode.regime,
            "risk_mode": episode.risk_mode,
            "global_mode": episode.global_mode,
            "tags": episode.tags,
        })
        
        await self.redis.zadd(self.REDIS_INDEX, {member: score})
        
        # Trim index to last 10,000 entries
        await self.redis.zremrangebyrank(self.REDIS_INDEX, 0, -10001)
    
    async def _query_index(
        self,
        episode_type: Optional[EpisodeType] = None,
        regime: Optional[str] = None,
        risk_mode: Optional[str] = None,
        global_mode: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Episode]:
        """Query episodes from index with filters."""
        # Get recent entries from sorted set
        min_score = since.timestamp() if since else "-inf"
        max_score = "+inf"
        
        # Fetch from index
        results = await self.redis.zrangebyscore(
            self.REDIS_INDEX,
            min_score,
            max_score,
            start=0,
            num=limit * 2,  # Over-fetch for filtering
        )
        
        episodes: list[Episode] = []
        
        for result in results:
            # Parse index entry
            entry = json.loads(result)
            
            # Apply filters
            if episode_type and entry["type"] != episode_type.value:
                continue
            if regime and entry["regime"] != regime:
                continue
            if risk_mode and entry["risk_mode"] != risk_mode:
                continue
            if global_mode and entry["global_mode"] != global_mode:
                continue
            
            # Fetch full episode
            episode = await self._get_episode(entry["episode_id"])
            if episode:
                episodes.append(episode)
            
            if len(episodes) >= limit:
                break
        
        return episodes
    
    async def _get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve episode from Redis."""
        key = f"{self.REDIS_PREFIX}{episode_id}"
        data = await self.redis.get(key)
        
        if not data:
            return None
        
        return Episode.from_dict(json.loads(data))
    
    async def _batch_loop(self) -> None:
        """Background task to batch episodes to Postgres."""
        while self._running:
            try:
                await asyncio.sleep(self.BATCH_INTERVAL)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")
    
    async def _flush_buffer(self) -> None:
        """Flush buffered episodes to Postgres."""
        if not self.postgres:
            return
        
        async with self._buffer_lock:
            if not self._buffer:
                return
            
            episodes = self._buffer[: self.BATCH_SIZE]
            self._buffer = self._buffer[self.BATCH_SIZE :]
        
        try:
            await self._write_to_postgres(episodes)
            logger.info(f"Flushed {len(episodes)} episodes to Postgres")
        except Exception as e:
            logger.error(f"Failed to write episodes to Postgres: {e}")
            # Re-add to buffer
            async with self._buffer_lock:
                self._buffer.extend(episodes)
    
    async def _write_to_postgres(self, episodes: list[Episode]) -> None:
        """Write episodes to Postgres."""
        # TODO: Implement Postgres batch insert
        # This would use asyncpg or similar:
        #
        # INSERT INTO episodes (
        #     episode_id, episode_type, timestamp, data,
        #     trace_id, regime, risk_mode, global_mode,
        #     pnl, drawdown, risk_score, tags
        # ) VALUES ($1, $2, ...) ON CONFLICT DO NOTHING
        
        logger.debug(f"Would write {len(episodes)} episodes to Postgres")
    
    async def _create_postgres_table(self) -> None:
        """Create Postgres table for episodes."""
        # TODO: Implement table creation
        # CREATE TABLE IF NOT EXISTS episodes (
        #     episode_id TEXT PRIMARY KEY,
        #     episode_type TEXT NOT NULL,
        #     timestamp TIMESTAMP NOT NULL,
        #     data JSONB NOT NULL,
        #     trace_id TEXT,
        #     regime TEXT,
        #     risk_mode TEXT,
        #     global_mode TEXT,
        #     pnl FLOAT,
        #     drawdown FLOAT,
        #     risk_score FLOAT,
        #     tags TEXT[]
        # );
        # CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
        # CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(episode_type);
        pass
    
    async def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        # Count episodes in index
        total_episodes = await self.redis.zcard(self.REDIS_INDEX)
        
        # Buffer size
        async with self._buffer_lock:
            buffer_size = len(self._buffer)
        
        return {
            "total_episodes_redis": total_episodes,
            "buffer_size": buffer_size,
            "ttl": self.ttl,
            "batch_interval": self.BATCH_INTERVAL,
        }
