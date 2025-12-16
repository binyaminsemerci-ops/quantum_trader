"""Semantic Memory - Pattern and Knowledge Storage.

Stores learned patterns, correlations, and lessons learned from
episodic memory. Semantic memory represents generalized knowledge
extracted from specific experiences.

Architecture:
- Patterns stored in Redis for fast retrieval
- Periodic jobs extract patterns from episodic memory
- Support for pattern types: regime_shift, correlation, performance, lesson
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


class PatternType(str, Enum):
    """Types of patterns stored in semantic memory."""
    
    REGIME_SHIFT = "regime_shift"
    CORRELATION = "correlation"
    PERFORMANCE = "performance"
    LESSON_LEARNED = "lesson_learned"
    RISK_PATTERN = "risk_pattern"
    STRATEGY_PATTERN = "strategy_pattern"
    MARKET_BEHAVIOR = "market_behavior"


@dataclass
class Pattern:
    """
    Learned pattern or knowledge.
    
    A pattern represents generalized knowledge extracted from
    episodic memory, such as:
    - "After X days of high volatility, market often enters range"
    - "RL performance correlates negatively with volatility"
    - "CEO mode switch to DEFENSIVE precedes drawdown reduction"
    """
    
    pattern_id: str
    pattern_type: PatternType
    description: str
    
    # Evidence supporting this pattern
    evidence: dict[str, Any]
    
    # Statistical measures
    confidence: float  # 0.0 to 1.0
    sample_size: int
    
    # Metadata
    discovered_at: datetime
    last_updated: datetime
    
    # Tags for querying
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["pattern_type"] = self.pattern_type.value
        d["discovered_at"] = self.discovered_at.isoformat()
        d["last_updated"] = self.last_updated.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pattern:
        """Create Pattern from dictionary."""
        data = data.copy()
        data["pattern_type"] = PatternType(data["pattern_type"])
        data["discovered_at"] = datetime.fromisoformat(data["discovered_at"])
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class SemanticMemory:
    """
    Semantic memory storage and pattern learning.
    
    Features:
    - Store learned patterns in Redis
    - Extract patterns from episodic memory
    - Query patterns by type, tags, confidence
    - Update patterns as new evidence emerges
    - Pattern consolidation (merge similar patterns)
    
    Usage:
        memory = SemanticMemory(redis_client)
        await memory.initialize()
        
        # Store a pattern
        pattern = Pattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type=PatternType.REGIME_SHIFT,
            description="After 3+ days of high volatility, market enters range 70% of time",
            evidence={
                "occurrences": 14,
                "total_observations": 20,
                "average_days_before_shift": 3.2,
            },
            confidence=0.70,
            sample_size=20,
            discovered_at=datetime.now(),
            last_updated=datetime.now(),
            tags=["volatility", "regime_shift", "range"],
        )
        await memory.store_pattern(pattern)
        
        # Query patterns
        regime_patterns = await memory.get_patterns(PatternType.REGIME_SHIFT)
        high_confidence = await memory.query_by_confidence(min_confidence=0.75)
    """
    
    REDIS_PREFIX = "quantum:memory:pattern:"
    REDIS_INDEX = "quantum:memory:pattern_index"
    PATTERN_EXTRACTION_INTERVAL = 3600  # 1 hour
    MIN_SAMPLE_SIZE = 5
    MIN_CONFIDENCE = 0.60
    
    def __init__(
        self,
        redis_client: Redis,
        postgres_connection: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
    ):
        """
        Initialize SemanticMemory.
        
        Args:
            redis_client: Async Redis client
            postgres_connection: Optional Postgres connection
            episodic_memory: Optional EpisodicMemory instance for pattern extraction
        """
        self.redis = redis_client
        self.postgres = postgres_connection
        self.episodic = episodic_memory
        
        # Background tasks
        self._running = False
        self._extraction_task: Optional[asyncio.Task] = None
        
        logger.info("SemanticMemory initialized")
    
    async def initialize(self) -> None:
        """Initialize semantic memory."""
        await self.redis.ping()
        
        if self.postgres:
            await self._create_postgres_table()
        
        logger.info("SemanticMemory initialized successfully")
    
    async def start(self) -> None:
        """Start background pattern extraction."""
        if self._running:
            logger.warning("SemanticMemory already running")
            return
        
        self._running = True
        
        # Start pattern extraction task
        if self.episodic:
            self._extraction_task = asyncio.create_task(self._extraction_loop())
        
        logger.info("SemanticMemory started")
    
    async def stop(self) -> None:
        """Stop background tasks."""
        if not self._running:
            return
        
        self._running = False
        
        if self._extraction_task:
            self._extraction_task.cancel()
            try:
                await self._extraction_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SemanticMemory stopped")
    
    async def store_pattern(self, pattern: Pattern) -> None:
        """
        Store a pattern.
        
        Args:
            pattern: Pattern to store
        """
        # Store in Redis
        key = f"{self.REDIS_PREFIX}{pattern.pattern_id}"
        data = json.dumps(pattern.to_dict())
        
        await self.redis.set(key, data)
        
        # Add to index
        await self._add_to_index(pattern)
        
        logger.debug(
            f"Stored pattern: type={pattern.pattern_type.value}, "
            f"confidence={pattern.confidence:.2f}"
        )
    
    async def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Retrieve a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
        
        Returns:
            Pattern or None
        """
        key = f"{self.REDIS_PREFIX}{pattern_id}"
        data = await self.redis.get(key)
        
        if not data:
            return None
        
        return Pattern.from_dict(json.loads(data))
    
    async def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        limit: int = 50,
    ) -> list[Pattern]:
        """
        Get patterns by type.
        
        Args:
            pattern_type: Optional pattern type filter
            limit: Maximum results
        
        Returns:
            List of patterns
        """
        return await self._query_index(
            pattern_type=pattern_type,
            limit=limit,
        )
    
    async def query_by_confidence(
        self,
        min_confidence: float,
        pattern_type: Optional[PatternType] = None,
        limit: int = 50,
    ) -> list[Pattern]:
        """
        Query patterns by minimum confidence.
        
        Args:
            min_confidence: Minimum confidence threshold
            pattern_type: Optional pattern type filter
            limit: Maximum results
        
        Returns:
            List of high-confidence patterns
        """
        patterns = await self._query_index(
            pattern_type=pattern_type,
            limit=limit * 2,
        )
        
        # Filter by confidence
        high_confidence = [
            p for p in patterns
            if p.confidence >= min_confidence
        ]
        
        return high_confidence[:limit]
    
    async def query_by_tags(
        self,
        tags: list[str],
        match_all: bool = False,
        limit: int = 50,
    ) -> list[Pattern]:
        """
        Query patterns by tags.
        
        Args:
            tags: Tags to match
            match_all: If True, require all tags; if False, require any tag
            limit: Maximum results
        
        Returns:
            List of patterns matching tags
        """
        patterns = await self._query_index(limit=limit * 2)
        
        # Filter by tags
        if match_all:
            matching = [
                p for p in patterns
                if all(tag in p.tags for tag in tags)
            ]
        else:
            matching = [
                p for p in patterns
                if any(tag in p.tags for tag in tags)
            ]
        
        return matching[:limit]
    
    async def query_pattern(
        self,
        topic: str,
        tags: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[Pattern]:
        """
        Query patterns by topic (description search).
        
        Args:
            topic: Topic to search for in pattern descriptions
            tags: Optional tags filter
            limit: Maximum results
        
        Returns:
            List of matching patterns
        """
        patterns = await self._query_index(limit=limit * 3)
        
        # Filter by topic (simple substring match)
        topic_lower = topic.lower()
        matching = [
            p for p in patterns
            if topic_lower in p.description.lower()
        ]
        
        # Apply tags filter if provided
        if tags:
            matching = [
                p for p in matching
                if any(tag in p.tags for tag in tags)
            ]
        
        return matching[:limit]
    
    async def get_all_patterns(self, limit: int = 100) -> list[Pattern]:
        """
        Get all patterns.
        
        Args:
            limit: Maximum results
        
        Returns:
            List of all patterns
        """
        return await self._query_index(limit=limit)
    
    async def update_pattern(
        self,
        pattern_id: str,
        new_evidence: dict[str, Any],
        new_confidence: Optional[float] = None,
        new_sample_size: Optional[int] = None,
    ) -> Optional[Pattern]:
        """
        Update a pattern with new evidence.
        
        Args:
            pattern_id: Pattern ID
            new_evidence: New evidence to merge
            new_confidence: Optional updated confidence
            new_sample_size: Optional updated sample size
        
        Returns:
            Updated pattern or None
        """
        pattern = await self.get_pattern(pattern_id)
        
        if not pattern:
            return None
        
        # Merge evidence
        pattern.evidence.update(new_evidence)
        
        # Update confidence and sample size
        if new_confidence is not None:
            pattern.confidence = new_confidence
        if new_sample_size is not None:
            pattern.sample_size = new_sample_size
        
        pattern.last_updated = datetime.now()
        
        await self.store_pattern(pattern)
        
        return pattern
    
    async def _add_to_index(self, pattern: Pattern) -> None:
        """Add pattern to Redis sorted set index."""
        # Use confidence as score for ranking
        score = pattern.confidence
        
        # Store pattern ID with metadata
        member = json.dumps({
            "pattern_id": pattern.pattern_id,
            "type": pattern.pattern_type.value,
            "tags": pattern.tags,
            "sample_size": pattern.sample_size,
        })
        
        await self.redis.zadd(self.REDIS_INDEX, {member: score})
        
        # Trim to top 1000 patterns
        await self.redis.zremrangebyrank(self.REDIS_INDEX, 0, -1001)
    
    async def _query_index(
        self,
        pattern_type: Optional[PatternType] = None,
        limit: int = 50,
    ) -> list[Pattern]:
        """Query patterns from index."""
        # Get patterns sorted by confidence (descending)
        results = await self.redis.zrevrange(
            self.REDIS_INDEX,
            0,
            limit * 2,
        )
        
        patterns: list[Pattern] = []
        
        for result in results:
            entry = json.loads(result)
            
            # Apply type filter
            if pattern_type and entry["type"] != pattern_type.value:
                continue
            
            # Fetch full pattern
            pattern = await self.get_pattern(entry["pattern_id"])
            if pattern:
                patterns.append(pattern)
            
            if len(patterns) >= limit:
                break
        
        return patterns
    
    async def _extraction_loop(self) -> None:
        """Background task to extract patterns from episodic memory."""
        while self._running:
            try:
                await asyncio.sleep(self.PATTERN_EXTRACTION_INTERVAL)
                await self._extract_patterns()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pattern extraction loop: {e}")
    
    async def _extract_patterns(self) -> None:
        """Extract patterns from episodic memory."""
        if not self.episodic:
            return
        
        logger.info("Starting pattern extraction")
        
        # Extract different types of patterns
        await self._extract_regime_shift_patterns()
        await self._extract_performance_patterns()
        await self._extract_risk_patterns()
        
        logger.info("Pattern extraction complete")
    
    async def _extract_regime_shift_patterns(self) -> None:
        """Extract regime shift patterns."""
        # TODO: Implement pattern extraction logic
        # Example: "After 3+ days of high volatility, market enters range"
        #
        # Steps:
        # 1. Query regime_change episodes from episodic memory
        # 2. Analyze preceding volatility patterns
        # 3. Calculate correlation
        # 4. Create pattern if confidence > threshold
        pass
    
    async def _extract_performance_patterns(self) -> None:
        """Extract performance patterns."""
        # TODO: Implement performance pattern extraction
        # Example: "RL performs poorly when volatility > X"
        pass
    
    async def _extract_risk_patterns(self) -> None:
        """Extract risk patterns."""
        # TODO: Implement risk pattern extraction
        # Example: "Risk alerts cluster around mode switches"
        pass
    
    async def _create_postgres_table(self) -> None:
        """Create Postgres table for patterns."""
        # TODO: Implement table creation
        # CREATE TABLE IF NOT EXISTS patterns (
        #     pattern_id TEXT PRIMARY KEY,
        #     pattern_type TEXT NOT NULL,
        #     description TEXT NOT NULL,
        #     evidence JSONB NOT NULL,
        #     confidence FLOAT NOT NULL,
        #     sample_size INT NOT NULL,
        #     discovered_at TIMESTAMP NOT NULL,
        #     last_updated TIMESTAMP NOT NULL,
        #     tags TEXT[]
        # );
        pass
    
    async def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        total_patterns = await self.redis.zcard(self.REDIS_INDEX)
        
        return {
            "total_patterns": total_patterns,
            "extraction_interval": self.PATTERN_EXTRACTION_INTERVAL,
            "min_confidence": self.MIN_CONFIDENCE,
            "min_sample_size": self.MIN_SAMPLE_SIZE,
        }
