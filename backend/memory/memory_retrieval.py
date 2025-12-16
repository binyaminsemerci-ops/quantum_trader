"""Memory Retrieval and Summarization.

Provides advanced query capabilities and periodic memory summarization
for creating daily/weekly/monthly memory reports.

Features:
- Multi-dimensional memory queries
- Relevance scoring and ranking
- Periodic memory summarization
- Memory snapshot generation
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.core.event_bus import EventBus
from backend.memory.episodic_memory import Episode, EpisodeType, EpisodicMemory
from backend.memory.policy_memory import PolicyMemory, PolicySnapshot
from backend.memory.semantic_memory import Pattern, PatternType, SemanticMemory

logger = logging.getLogger(__name__)


@dataclass
class MemorySummary:
    """
    Periodic memory summary.
    
    Aggregates key insights from episodic, semantic, and policy memory
    for a specific time period.
    """
    
    summary_id: str
    period_start: datetime
    period_end: datetime
    
    # Episodic summary
    total_trades: int
    profitable_trades: int
    loss_trades: int
    total_pnl: float
    avg_pnl: float
    max_profit: float
    max_loss: float
    
    # Risk events
    total_risk_events: int
    critical_risk_events: int
    
    # CEO decisions
    total_ceo_decisions: int
    mode_switches: int
    
    # Regime changes
    regime_distribution: dict[str, int] = field(default_factory=dict)
    
    # Key patterns
    key_patterns: list[dict[str, Any]] = field(default_factory=list)
    
    # Policy insights
    policy_changes: int = 0
    best_performing_policy: Optional[dict[str, Any]] = None
    
    # Tags
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary_id": self.summary_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_trades": self.total_trades,
            "profitable_trades": self.profitable_trades,
            "loss_trades": self.loss_trades,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "total_risk_events": self.total_risk_events,
            "critical_risk_events": self.critical_risk_events,
            "total_ceo_decisions": self.total_ceo_decisions,
            "mode_switches": self.mode_switches,
            "regime_distribution": self.regime_distribution,
            "key_patterns": self.key_patterns,
            "policy_changes": self.policy_changes,
            "best_performing_policy": self.best_performing_policy,
            "tags": self.tags,
        }


class MemoryRetrieval:
    """
    Advanced memory retrieval and summarization.
    
    Features:
    - Complex queries across memory types
    - Relevance scoring and ranking
    - Periodic summary generation
    - Memory snapshot creation
    
    Usage:
        retrieval = MemoryRetrieval(
            redis_client,
            event_bus,
            episodic_memory,
            semantic_memory,
            policy_memory,
        )
        await retrieval.initialize()
        await retrieval.start()
        
        # Generate summary
        summary = await retrieval.generate_summary(days=7)
        
        # Query relevant memories
        memories = await retrieval.query_relevant_memories(
            query="high profit trades in TRENDING_UP regime",
            limit=10,
        )
    """
    
    SUMMARY_INTERVAL = 86400  # Daily summaries
    REDIS_SUMMARY_PREFIX = "quantum:memory:summary:"
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: EventBus,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        policy_memory: PolicyMemory,
    ):
        """
        Initialize MemoryRetrieval.
        
        Args:
            redis_client: Async Redis client
            event_bus: EventBus for publishing summaries
            episodic_memory: EpisodicMemory instance
            semantic_memory: SemanticMemory instance
            policy_memory: PolicyMemory instance
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.policy = policy_memory
        
        # Background tasks
        self._running = False
        self._summary_task: Optional[asyncio.Task] = None
        
        logger.info("MemoryRetrieval initialized")
    
    async def initialize(self) -> None:
        """Initialize memory retrieval."""
        await self.redis.ping()
        logger.info("MemoryRetrieval initialized successfully")
    
    async def start(self) -> None:
        """Start periodic summarization."""
        if self._running:
            logger.warning("MemoryRetrieval already running")
            return
        
        self._running = True
        
        # Start summary task
        self._summary_task = asyncio.create_task(self._summary_loop())
        
        logger.info("MemoryRetrieval started")
    
    async def stop(self) -> None:
        """Stop background tasks."""
        if not self._running:
            return
        
        self._running = False
        
        if self._summary_task:
            self._summary_task.cancel()
            try:
                await self._summary_task
            except asyncio.CancelledError:
                pass
        
        logger.info("MemoryRetrieval stopped")
    
    async def generate_summary(
        self,
        days: int = 1,
        include_patterns: bool = True,
        include_policy_insights: bool = True,
    ) -> MemorySummary:
        """
        Generate memory summary for period.
        
        Args:
            days: Number of days to summarize
            include_patterns: Include semantic patterns
            include_policy_insights: Include policy analysis
        
        Returns:
            MemorySummary
        """
        period_end = datetime.now()
        period_start = period_end - timedelta(days=days)
        
        # Query trades
        trades = await self.episodic.query_by_type(
            EpisodeType.TRADE,
            days=days,
            limit=1000,
        )
        
        # Calculate trade metrics
        profitable = [t for t in trades if t.pnl and t.pnl > 0]
        losses = [t for t in trades if t.pnl and t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in trades if t.pnl)
        avg_pnl = total_pnl / len(trades) if trades else 0
        max_profit = max((t.pnl for t in trades if t.pnl), default=0)
        max_loss = min((t.pnl for t in trades if t.pnl), default=0)
        
        # Query risk events
        risk_events = await self.episodic.query_by_type(
            EpisodeType.RISK_EVENT,
            days=days,
            limit=500,
        )
        critical_risk = [
            e for e in risk_events
            if e.data.get("severity") == "CRITICAL"
        ]
        
        # Query CEO decisions
        ceo_decisions = await self.episodic.query_by_type(
            EpisodeType.CEO_DECISION,
            days=days,
            limit=500,
        )
        mode_switches = sum(
            1 for d in ceo_decisions
            if d.data.get("action") == "mode_switch"
        )
        
        # Regime distribution
        regime_dist: dict[str, int] = {}
        for trade in trades:
            regime = trade.regime or "UNKNOWN"
            regime_dist[regime] = regime_dist.get(regime, 0) + 1
        
        # Create summary
        summary = MemorySummary(
            summary_id=f"summary_{period_start.strftime('%Y%m%d')}_{days}d",
            period_start=period_start,
            period_end=period_end,
            total_trades=len(trades),
            profitable_trades=len(profitable),
            loss_trades=len(losses),
            total_pnl=total_pnl,
            avg_pnl=avg_pnl,
            max_profit=max_profit,
            max_loss=max_loss,
            total_risk_events=len(risk_events),
            critical_risk_events=len(critical_risk),
            total_ceo_decisions=len(ceo_decisions),
            mode_switches=mode_switches,
            regime_distribution=regime_dist,
        )
        
        # Add key patterns
        if include_patterns:
            patterns = await self.semantic.query_by_confidence(
                min_confidence=0.70,
                limit=10,
            )
            summary.key_patterns = [
                {
                    "type": p.pattern_type.value,
                    "description": p.description,
                    "confidence": p.confidence,
                }
                for p in patterns
            ]
        
        # Add policy insights
        if include_policy_insights:
            policy_snapshots = await self.policy.get_recent_snapshots(days=days)
            summary.policy_changes = len(policy_snapshots)
            
            # Find best performing policy
            snapshots_with_outcomes = [
                s for s in policy_snapshots
                if s.outcomes and "total_pnl" in s.outcomes
            ]
            if snapshots_with_outcomes:
                best = max(
                    snapshots_with_outcomes,
                    key=lambda s: s.outcomes["total_pnl"],
                )
                summary.best_performing_policy = {
                    "global_mode": best.global_mode,
                    "risk_mode": best.risk_mode,
                    "leverage": best.leverage,
                    "pnl": best.outcomes["total_pnl"],
                }
        
        # Store summary
        await self._store_summary(summary)
        
        # Publish event
        await self.event_bus.publish("memory_summary_created", summary.to_dict())
        
        logger.info(f"Generated memory summary: {summary.summary_id}")
        
        return summary
    
    async def query_relevant_memories(
        self,
        query: str,
        days: int = 30,
        limit: int = 10,
    ) -> dict[str, list[Any]]:
        """
        Query relevant memories based on natural language query.
        
        Args:
            query: Natural language query
            days: Look back N days
            limit: Maximum results per type
        
        Returns:
            Dictionary with episodes, patterns, and policy snapshots
        """
        query_lower = query.lower()
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Query episodes by tags
        episodes: list[Episode] = []
        if keywords:
            episodes = await self.episodic.query_by_tags(
                tags=keywords,
                days=days,
                limit=limit,
                match_all=False,
            )
        
        # Query patterns by topic
        patterns: list[Pattern] = []
        if keywords:
            patterns = await self.semantic.query_pattern(
                topic=query,
                tags=keywords,
                limit=limit,
            )
        
        # Query policy snapshots
        policy_snapshots: list[PolicySnapshot] = []
        if "policy" in query_lower or "mode" in query_lower:
            policy_snapshots = await self.policy.get_recent_snapshots(
                days=days,
                limit=limit,
            )
        
        return {
            "episodes": [e.to_dict() for e in episodes],
            "patterns": [p.to_dict() for p in patterns],
            "policy_snapshots": [p.to_dict() for p in policy_snapshots],
        }
    
    async def get_daily_memory_report(
        self,
        date: Optional[datetime] = None,
    ) -> Optional[MemorySummary]:
        """
        Get daily memory report for a specific date.
        
        Args:
            date: Date to get report for (default: today)
        
        Returns:
            MemorySummary or None
        """
        if date is None:
            date = datetime.now()
        
        summary_id = f"summary_{date.strftime('%Y%m%d')}_1d"
        key = f"{self.REDIS_SUMMARY_PREFIX}{summary_id}"
        
        data = await self.redis.get(key)
        if not data:
            return None
        
        summary_dict = json.loads(data)
        
        # Convert back to MemorySummary
        summary_dict["period_start"] = datetime.fromisoformat(summary_dict["period_start"])
        summary_dict["period_end"] = datetime.fromisoformat(summary_dict["period_end"])
        
        return MemorySummary(**summary_dict)
    
    async def get_risk_memory_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Generate risk-focused memory summary.
        
        Args:
            days: Look back N days
        
        Returns:
            Risk summary dictionary
        """
        # Query risk events
        risk_events = await self.episodic.query_by_type(
            EpisodeType.RISK_EVENT,
            days=days,
            limit=500,
        )
        
        # Query high-loss trades
        high_loss_trades = await self.episodic.query_by_loss_threshold(
            loss_threshold=50,
            days=days,
            limit=100,
        )
        
        # Group risk events by severity
        by_severity: dict[str, int] = {}
        for event in risk_events:
            severity = event.data.get("severity", "MEDIUM")
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Query risk patterns
        risk_patterns = await self.semantic.get_patterns(
            pattern_type=PatternType.RISK_PATTERN,
            limit=10,
        )
        
        return {
            "period_days": days,
            "total_risk_events": len(risk_events),
            "risk_events_by_severity": by_severity,
            "high_loss_trades": len(high_loss_trades),
            "total_loss_from_high_loss": sum(
                t.pnl for t in high_loss_trades if t.pnl
            ),
            "risk_patterns": [
                {
                    "description": p.description,
                    "confidence": p.confidence,
                }
                for p in risk_patterns
            ],
        }
    
    async def get_strategy_memory_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Generate strategy-focused memory summary.
        
        Args:
            days: Look back N days
        
        Returns:
            Strategy summary dictionary
        """
        # Query strategy shifts
        strategy_shifts = await self.episodic.query_by_type(
            EpisodeType.STRATEGY_SHIFT,
            days=days,
            limit=200,
        )
        
        # Query trades by regime
        regimes = ["TRENDING_UP", "TRENDING_DOWN", "RANGING", "CHOPPY"]
        regime_performance: dict[str, dict[str, float]] = {}
        
        for regime in regimes:
            trades = await self.episodic.query_by_context(
                regime=regime,
                days=days,
                limit=500,
            )
            
            if trades:
                total_pnl = sum(t.pnl for t in trades if t.pnl)
                win_rate = sum(1 for t in trades if t.pnl and t.pnl > 0) / len(trades)
                
                regime_performance[regime] = {
                    "total_trades": len(trades),
                    "total_pnl": total_pnl,
                    "win_rate": win_rate,
                }
        
        # Query strategy patterns
        strategy_patterns = await self.semantic.get_patterns(
            pattern_type=PatternType.STRATEGY_PATTERN,
            limit=10,
        )
        
        return {
            "period_days": days,
            "total_strategy_shifts": len(strategy_shifts),
            "regime_performance": regime_performance,
            "strategy_patterns": [
                {
                    "description": p.description,
                    "confidence": p.confidence,
                }
                for p in strategy_patterns
            ],
        }
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from query."""
        # Simple keyword extraction
        important_words = [
            "profit", "loss", "risk", "regime", "trending", "ranging",
            "volatile", "btc", "eth", "mode", "switch", "alert",
            "ceo", "strategy", "policy", "drawdown",
        ]
        
        keywords = []
        for word in important_words:
            if word in query:
                keywords.append(word)
        
        return keywords
    
    async def _store_summary(self, summary: MemorySummary) -> None:
        """Store summary in Redis."""
        key = f"{self.REDIS_SUMMARY_PREFIX}{summary.summary_id}"
        data = json.dumps(summary.to_dict())
        
        # Store with 30-day TTL
        await self.redis.setex(key, 86400 * 30, data)
    
    async def _summary_loop(self) -> None:
        """Background task to generate periodic summaries."""
        while self._running:
            try:
                await asyncio.sleep(self.SUMMARY_INTERVAL)
                
                # Generate daily summary
                await self.generate_summary(days=1)
                
                # Generate weekly summary (on Sundays)
                if datetime.now().weekday() == 6:
                    await self.generate_summary(days=7)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in summary loop: {e}")
