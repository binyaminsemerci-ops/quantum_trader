"""Policy Memory - Historical Policy States and Outcomes.

Stores historical policy configurations with performance context,
allowing agents to learn from past policy decisions.

Architecture:
- Policy snapshots stored with full context
- Link to performance outcomes (PnL, drawdown, risk events)
- Similarity search for "what happened last time we used this policy?"
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class PolicySnapshot:
    """
    Historical policy state with context and outcomes.
    
    A policy snapshot captures:
    - Complete policy configuration at a point in time
    - Context (regime, market conditions, system state)
    - Performance outcomes (PnL, drawdown, risk events)
    """
    
    snapshot_id: str
    timestamp: datetime
    
    # Policy configuration
    global_mode: str
    risk_mode: str
    leverage: float
    max_positions: int
    position_size_pct: float
    risk_per_trade_pct: float
    
    # Full policy config (for detailed comparison)
    policy_config: dict[str, Any]
    
    # Context at snapshot time
    context: dict[str, Any] = field(default_factory=dict)
    
    # Performance outcomes (filled in after observation period)
    outcomes: Optional[dict[str, Any]] = None
    
    # Tags for querying
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySnapshot:
        """Create PolicySnapshot from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class PolicyMemory:
    """
    Policy memory storage and retrieval.
    
    Features:
    - Store policy snapshots with context
    - Track performance outcomes over time
    - Similarity search for similar past policies
    - Suggest policy adjustments based on historical data
    
    Usage:
        memory = PolicyMemory(redis_client)
        await memory.initialize()
        
        # Log current policy state
        snapshot = await memory.log_policy_state(
            policy_config={...},
            context={"regime": "TRENDING_UP", "volatility": 0.45},
        )
        
        # Later, update with outcomes
        await memory.update_outcomes(
            snapshot.snapshot_id,
            outcomes={
                "total_pnl": 150.0,
                "win_rate": 0.65,
                "max_drawdown": 0.05,
                "risk_events": 2,
            },
        )
        
        # Query similar policies
        similar = await memory.lookup_similar_policy_states(
            context={"regime": "TRENDING_UP", "volatility": 0.45},
        )
    """
    
    REDIS_PREFIX = "quantum:memory:policy:"
    REDIS_INDEX = "quantum:memory:policy_index"
    OUTCOME_UPDATE_INTERVAL = 3600  # 1 hour
    OBSERVATION_PERIOD_HOURS = 24  # Observe outcomes for 24 hours
    
    def __init__(
        self,
        redis_client: Redis,
        postgres_connection: Optional[Any] = None,
    ):
        """
        Initialize PolicyMemory.
        
        Args:
            redis_client: Async Redis client
            postgres_connection: Optional Postgres connection
        """
        self.redis = redis_client
        self.postgres = postgres_connection
        
        # Background tasks
        self._running = False
        self._outcome_task: Optional[asyncio.Task] = None
        
        logger.info("PolicyMemory initialized")
    
    async def initialize(self) -> None:
        """Initialize policy memory."""
        await self.redis.ping()
        
        if self.postgres:
            await self._create_postgres_table()
        
        logger.info("PolicyMemory initialized successfully")
    
    async def start(self) -> None:
        """Start background outcome tracking."""
        if self._running:
            logger.warning("PolicyMemory already running")
            return
        
        self._running = True
        
        # Start outcome update task
        self._outcome_task = asyncio.create_task(self._outcome_update_loop())
        
        logger.info("PolicyMemory started")
    
    async def stop(self) -> None:
        """Stop background tasks."""
        if not self._running:
            return
        
        self._running = False
        
        if self._outcome_task:
            self._outcome_task.cancel()
            try:
                await self._outcome_task
            except asyncio.CancelledError:
                pass
        
        logger.info("PolicyMemory stopped")
    
    async def log_policy_state(
        self,
        policy_config: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> PolicySnapshot:
        """
        Log current policy state.
        
        Args:
            policy_config: Full policy configuration
            context: Optional context (regime, volatility, etc.)
            tags: Optional tags
        
        Returns:
            Created PolicySnapshot
        """
        snapshot = PolicySnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            global_mode=policy_config.get("global_mode", "UNKNOWN"),
            risk_mode=policy_config.get("active_mode", "UNKNOWN"),
            leverage=policy_config.get("leverage", 1.0),
            max_positions=policy_config.get("max_positions", 0),
            position_size_pct=policy_config.get("position_size_pct", 0.0),
            risk_per_trade_pct=policy_config.get("risk_per_trade_pct", 0.0),
            policy_config=policy_config,
            context=context or {},
            tags=tags or [],
        )
        
        await self.store_snapshot(snapshot)
        
        return snapshot
    
    async def store_snapshot(self, snapshot: PolicySnapshot) -> None:
        """
        Store a policy snapshot.
        
        Args:
            snapshot: PolicySnapshot to store
        """
        # Store in Redis
        key = f"{self.REDIS_PREFIX}{snapshot.snapshot_id}"
        data = json.dumps(snapshot.to_dict())
        
        await self.redis.set(key, data)
        
        # Add to index
        await self._add_to_index(snapshot)
        
        logger.debug(
            f"Stored policy snapshot: mode={snapshot.global_mode}, "
            f"leverage={snapshot.leverage}"
        )
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[PolicySnapshot]:
        """
        Retrieve a policy snapshot by ID.
        
        Args:
            snapshot_id: Snapshot ID
        
        Returns:
            PolicySnapshot or None
        """
        key = f"{self.REDIS_PREFIX}{snapshot_id}"
        data = await self.redis.get(key)
        
        if not data:
            return None
        
        return PolicySnapshot.from_dict(json.loads(data))
    
    async def update_outcomes(
        self,
        snapshot_id: str,
        outcomes: dict[str, Any],
    ) -> Optional[PolicySnapshot]:
        """
        Update a snapshot with observed outcomes.
        
        Args:
            snapshot_id: Snapshot ID
            outcomes: Outcomes dict (pnl, win_rate, drawdown, etc.)
        
        Returns:
            Updated PolicySnapshot or None
        """
        snapshot = await self.get_snapshot(snapshot_id)
        
        if not snapshot:
            return None
        
        snapshot.outcomes = outcomes
        
        await self.store_snapshot(snapshot)
        
        logger.debug(f"Updated policy snapshot outcomes: id={snapshot_id}")
        
        return snapshot
    
    async def get_recent_snapshots(
        self,
        days: int = 7,
        limit: int = 50,
    ) -> list[PolicySnapshot]:
        """
        Get recent policy snapshots.
        
        Args:
            days: Look back N days
            limit: Maximum results
        
        Returns:
            List of recent snapshots
        """
        since = datetime.now() - timedelta(days=days)
        
        return await self._query_index(
            since=since,
            limit=limit,
        )
    
    async def lookup_similar_policy_states(
        self,
        context: dict[str, Any],
        days: int = 30,
        limit: int = 10,
    ) -> list[PolicySnapshot]:
        """
        Find similar policy states from history.
        
        Args:
            context: Context to match (regime, volatility, etc.)
            days: Look back N days
            limit: Maximum results
        
        Returns:
            List of similar snapshots
        """
        since = datetime.now() - timedelta(days=days)
        
        snapshots = await self._query_index(
            since=since,
            limit=limit * 3,  # Over-fetch for similarity filtering
        )
        
        # Calculate similarity scores
        scored = []
        for snapshot in snapshots:
            score = self._calculate_similarity(context, snapshot.context)
            scored.append((score, snapshot))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [s for _, s in scored[:limit]]
    
    async def suggest_policy_adjustments(
        self,
        current_context: dict[str, Any],
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """
        Suggest policy adjustments based on historical data.
        
        Args:
            current_context: Current context
            lookback_days: Days to look back
        
        Returns:
            Suggestions dict
        """
        # Find similar historical states
        similar = await self.lookup_similar_policy_states(
            context=current_context,
            days=lookback_days,
            limit=20,
        )
        
        # Filter for snapshots with outcomes
        with_outcomes = [s for s in similar if s.outcomes]
        
        if not with_outcomes:
            return {
                "suggestions": [],
                "confidence": 0.0,
                "reason": "Insufficient historical data",
            }
        
        # Analyze outcomes
        best_performers = sorted(
            with_outcomes,
            key=lambda s: s.outcomes.get("total_pnl", 0),
            reverse=True,
        )[:5]
        
        # Extract common characteristics of best performers
        suggestions = []
        
        # Average leverage from top performers
        avg_leverage = sum(s.leverage for s in best_performers) / len(best_performers)
        suggestions.append({
            "parameter": "leverage",
            "suggested_value": round(avg_leverage, 2),
            "reason": f"Top performers used avg leverage of {avg_leverage:.2f}",
        })
        
        # Most common mode
        modes = [s.global_mode for s in best_performers]
        most_common_mode = max(set(modes), key=modes.count)
        suggestions.append({
            "parameter": "global_mode",
            "suggested_value": most_common_mode,
            "reason": f"Mode {most_common_mode} performed best in similar context",
        })
        
        # Calculate confidence based on sample size and consistency
        confidence = min(len(with_outcomes) / 20.0, 1.0)
        
        return {
            "suggestions": suggestions,
            "confidence": confidence,
            "sample_size": len(with_outcomes),
            "reason": f"Based on {len(with_outcomes)} similar historical states",
        }
    
    def _calculate_similarity(
        self,
        context1: dict[str, Any],
        context2: dict[str, Any],
    ) -> float:
        """
        Calculate similarity score between two contexts.
        
        Args:
            context1: First context
            context2: Second context
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not context1 or not context2:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        # Regime match (weight: 0.4)
        if "regime" in context1 and "regime" in context2:
            if context1["regime"] == context2["regime"]:
                score += 0.4
            total_weight += 0.4
        
        # Volatility proximity (weight: 0.3)
        if "volatility" in context1 and "volatility" in context2:
            vol1 = float(context1["volatility"])
            vol2 = float(context2["volatility"])
            vol_similarity = 1.0 - min(abs(vol1 - vol2) / max(vol1, vol2, 1e-6), 1.0)
            score += 0.3 * vol_similarity
            total_weight += 0.3
        
        # Trend strength proximity (weight: 0.2)
        if "trend_strength" in context1 and "trend_strength" in context2:
            trend1 = float(context1["trend_strength"])
            trend2 = float(context2["trend_strength"])
            trend_similarity = 1.0 - min(abs(trend1 - trend2), 1.0)
            score += 0.2 * trend_similarity
            total_weight += 0.2
        
        # Market condition match (weight: 0.1)
        if "market_condition" in context1 and "market_condition" in context2:
            if context1["market_condition"] == context2["market_condition"]:
                score += 0.1
            total_weight += 0.1
        
        # Normalize by total weight
        if total_weight > 0:
            return score / total_weight
        
        return 0.0
    
    async def _add_to_index(self, snapshot: PolicySnapshot) -> None:
        """Add snapshot to Redis sorted set index."""
        # Use timestamp as score for chronological sorting
        score = snapshot.timestamp.timestamp()
        
        # Store snapshot ID with metadata
        member = json.dumps({
            "snapshot_id": snapshot.snapshot_id,
            "global_mode": snapshot.global_mode,
            "risk_mode": snapshot.risk_mode,
            "leverage": snapshot.leverage,
        })
        
        await self.redis.zadd(self.REDIS_INDEX, {member: score})
        
        # Trim to last 1000 snapshots
        await self.redis.zremrangebyrank(self.REDIS_INDEX, 0, -1001)
    
    async def _query_index(
        self,
        since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[PolicySnapshot]:
        """Query policy snapshots from index."""
        # Get entries sorted by timestamp
        min_score = since.timestamp() if since else "-inf"
        max_score = "+inf"
        
        results = await self.redis.zrangebyscore(
            self.REDIS_INDEX,
            min_score,
            max_score,
            start=0,
            num=limit,
        )
        
        snapshots: list[PolicySnapshot] = []
        
        for result in results:
            entry = json.loads(result)
            
            # Fetch full snapshot
            snapshot = await self.get_snapshot(entry["snapshot_id"])
            if snapshot:
                snapshots.append(snapshot)
        
        return snapshots
    
    async def _outcome_update_loop(self) -> None:
        """Background task to update policy outcomes."""
        while self._running:
            try:
                await asyncio.sleep(self.OUTCOME_UPDATE_INTERVAL)
                await self._update_pending_outcomes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in outcome update loop: {e}")
    
    async def _update_pending_outcomes(self) -> None:
        """Update outcomes for snapshots in observation period."""
        # Get snapshots from observation period
        lookback_hours = self.OBSERVATION_PERIOD_HOURS
        since = datetime.now() - timedelta(hours=lookback_hours * 2)
        
        snapshots = await self._query_index(since=since, limit=100)
        
        for snapshot in snapshots:
            # Skip if already has outcomes
            if snapshot.outcomes:
                continue
            
            # Check if observation period has passed
            age = datetime.now() - snapshot.timestamp
            if age.total_seconds() < self.OBSERVATION_PERIOD_HOURS * 3600:
                continue
            
            # Calculate outcomes
            # TODO: Query episodic memory for trades/events during observation period
            # For now, just mark as observed
            await self.update_outcomes(
                snapshot.snapshot_id,
                outcomes={
                    "observed": True,
                    "observation_period_hours": lookback_hours,
                },
            )
    
    async def _create_postgres_table(self) -> None:
        """Create Postgres table for policy snapshots."""
        # TODO: Implement table creation
        # CREATE TABLE IF NOT EXISTS policy_snapshots (
        #     snapshot_id TEXT PRIMARY KEY,
        #     timestamp TIMESTAMP NOT NULL,
        #     global_mode TEXT NOT NULL,
        #     risk_mode TEXT NOT NULL,
        #     leverage FLOAT NOT NULL,
        #     max_positions INT NOT NULL,
        #     position_size_pct FLOAT NOT NULL,
        #     risk_per_trade_pct FLOAT NOT NULL,
        #     policy_config JSONB NOT NULL,
        #     context JSONB,
        #     outcomes JSONB,
        #     tags TEXT[]
        # );
        pass
    
    async def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        total_snapshots = await self.redis.zcard(self.REDIS_INDEX)
        
        # Count snapshots with outcomes
        recent_snapshots = await self.get_recent_snapshots(days=30, limit=100)
        with_outcomes = sum(1 for s in recent_snapshots if s.outcomes)
        
        return {
            "total_snapshots": total_snapshots,
            "recent_with_outcomes": with_outcomes,
            "observation_period_hours": self.OBSERVATION_PERIOD_HOURS,
            "outcome_update_interval": self.OUTCOME_UPDATE_INTERVAL,
        }
