"""Memory Engine - Unified entry point for all memory operations.

The Memory Engine provides a high-level API for AI agents to interact
with the memory system without knowing the underlying storage details.

Architecture:
- Routes storage to episodic, semantic, and policy memory
- Provides unified query interface
- Coordinates memory consolidation
- Integrates with EventBus for memory updates
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.memory.episodic_memory import Episode, EpisodeType, EpisodicMemory
from backend.memory.policy_memory import PolicyMemory, PolicySnapshot
from backend.memory.semantic_memory import Pattern, PatternType, SemanticMemory

logger = logging.getLogger(__name__)


class MemoryEngine:
    """
    Unified memory engine for Quantum Trader v5.
    
    The Memory Engine is the entry point for all AI agents to:
    - Store events and episodes
    - Store learned patterns
    - Track policy history
    - Query memories
    - Generate summaries
    
    Integration:
    - AI CEO: Store decisions, query similar past states
    - AI Risk Officer: Store risk events, query historical risk patterns
    - AI Strategy Officer: Store strategy shifts, query performance patterns
    
    Usage:
        engine = MemoryEngine(redis_client, event_bus, policy_store)
        await engine.initialize()
        await engine.start()
        
        # Store a trade episode
        await engine.store_event(
            event_type=EpisodeType.TRADE,
            data={"symbol": "BTCUSDT", "pnl": 50},
            regime="TRENDING_UP",
            pnl=50.0,
        )
        
        # Query similar episodes
        similar = await engine.query_memory(
            episode_type=EpisodeType.TRADE,
            context={"regime": "TRENDING_UP"},
            days=30,
        )
        
        # Get summary
        summary = await engine.summarize(days=7)
    """
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: EventBus,
        policy_store: PolicyStore,
        postgres_connection: Optional[Any] = None,
    ):
        """
        Initialize Memory Engine.
        
        Args:
            redis_client: Async Redis client
            event_bus: EventBus instance for publishing memory updates
            policy_store: PolicyStore for policy context
            postgres_connection: Optional Postgres connection
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.policy_store = policy_store
        
        # Memory components
        self.episodic = EpisodicMemory(redis_client, postgres_connection)
        self.semantic = SemanticMemory(redis_client, postgres_connection)
        self.policy = PolicyMemory(redis_client, postgres_connection)
        
        # Running state
        self._running = False
        self._instance_id = f"memory_engine_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"MemoryEngine initialized: instance_id={self._instance_id}")
    
    async def initialize(self) -> None:
        """Initialize all memory components."""
        await self.episodic.initialize()
        await self.semantic.initialize()
        await self.policy.initialize()
        
        # Subscribe to events for automatic memory storage
        self._subscribe_to_events()
        
        logger.info("MemoryEngine initialized successfully")
    
    async def start(self) -> None:
        """Start memory engine and background tasks."""
        if self._running:
            logger.warning("MemoryEngine already running")
            return
        
        self._running = True
        
        # Start memory components
        await self.episodic.start()
        await self.semantic.start()
        await self.policy.start()
        
        logger.info("MemoryEngine started")
    
    async def stop(self) -> None:
        """Stop memory engine and flush buffers."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop memory components
        await self.episodic.stop()
        await self.semantic.stop()
        await self.policy.stop()
        
        logger.info("MemoryEngine stopped")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to events for automatic memory storage."""
        # Trade events
        self.event_bus.subscribe("position_opened", self._handle_position_opened)
        self.event_bus.subscribe("position_closed", self._handle_position_closed)
        
        # CEO decisions
        self.event_bus.subscribe("ceo_decision", self._handle_ceo_decision)
        self.event_bus.subscribe("ceo_mode_switch", self._handle_ceo_mode_switch)
        
        # Risk events
        self.event_bus.subscribe("risk_alert", self._handle_risk_alert)
        
        # Strategy events
        self.event_bus.subscribe("strategy_recommendation", self._handle_strategy_recommendation)
        
        # System events
        self.event_bus.subscribe("system_degraded", self._handle_system_degraded)
        
        # Regime changes
        self.event_bus.subscribe("regime_detected", self._handle_regime_detected)
        
        # Policy updates
        self.event_bus.subscribe("policy_updated", self._handle_policy_updated)
        
        logger.info("MemoryEngine subscribed to events")
    
    # -------------------------------------------------------------------------
    # High-level API
    # -------------------------------------------------------------------------
    
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
        Store an event in episodic memory.
        
        Args:
            event_type: Type of event
            data: Event-specific data
            trace_id: Optional trace ID
            regime: Optional regime
            risk_mode: Optional risk mode
            global_mode: Optional global mode
            pnl: Optional PnL
            drawdown: Optional drawdown
            risk_score: Optional risk score
            tags: Optional tags
        
        Returns:
            Created Episode
        """
        # Get current context if not provided
        if not regime or not risk_mode or not global_mode:
            policy = await self.policy_store.get_policy()
            regime = regime or "UNKNOWN"
            risk_mode = risk_mode or policy.active_mode.value
            global_mode = global_mode or "UNKNOWN"
        
        episode = await self.episodic.store_event(
            event_type=event_type,
            data=data,
            trace_id=trace_id,
            regime=regime,
            risk_mode=risk_mode,
            global_mode=global_mode,
            pnl=pnl,
            drawdown=drawdown,
            risk_score=risk_score,
            tags=tags,
        )
        
        # Publish memory event
        await self.event_bus.publish(
            "memory_stored",
            {
                "episode_id": episode.episode_id,
                "episode_type": episode.episode_type.value,
                "timestamp": episode.timestamp.isoformat(),
            },
        )
        
        return episode
    
    async def store_episode(self, episode: Episode) -> None:
        """
        Store a complete episode.
        
        Args:
            episode: Episode to store
        """
        await self.episodic.store_episode(episode)
        
        await self.event_bus.publish(
            "memory_stored",
            {
                "episode_id": episode.episode_id,
                "episode_type": episode.episode_type.value,
                "timestamp": episode.timestamp.isoformat(),
            },
        )
    
    async def query_memory(
        self,
        episode_type: Optional[EpisodeType] = None,
        context: Optional[dict[str, str]] = None,
        tags: Optional[list[str]] = None,
        days: int = 7,
        limit: int = 100,
    ) -> list[Episode]:
        """
        Query episodic memory.
        
        Args:
            episode_type: Optional episode type filter
            context: Optional context filter (regime, risk_mode, global_mode)
            tags: Optional tags filter
            days: Look back N days
            limit: Maximum results
        
        Returns:
            List of matching episodes
        """
        if tags:
            return await self.episodic.query_by_tags(tags, days, limit)
        
        if context:
            return await self.episodic.query_by_context(
                regime=context.get("regime"),
                risk_mode=context.get("risk_mode"),
                global_mode=context.get("global_mode"),
                days=days,
                limit=limit,
            )
        
        if episode_type:
            return await self.episodic.query_by_type(episode_type, days, limit)
        
        # Default: query all recent episodes
        return await self.episodic.query_by_type(
            EpisodeType.TRADE,  # Default to trades
            days=days,
            limit=limit,
        )
    
    async def summarize(
        self,
        days: int = 7,
        include_patterns: bool = True,
        include_policy_history: bool = True,
    ) -> dict[str, Any]:
        """
        Generate memory summary.
        
        Args:
            days: Look back N days
            include_patterns: Include semantic patterns
            include_policy_history: Include policy snapshots
        
        Returns:
            Memory summary dictionary
        """
        summary = {
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
        }
        
        # Episodic summary
        trades = await self.episodic.query_by_type(EpisodeType.TRADE, days)
        risk_events = await self.episodic.query_by_type(EpisodeType.RISK_EVENT, days)
        ceo_decisions = await self.episodic.query_by_type(EpisodeType.CEO_DECISION, days)
        
        summary["episodic"] = {
            "total_trades": len(trades),
            "total_risk_events": len(risk_events),
            "total_ceo_decisions": len(ceo_decisions),
            "total_pnl": sum(ep.pnl for ep in trades if ep.pnl),
            "avg_pnl": (
                sum(ep.pnl for ep in trades if ep.pnl) / len(trades)
                if trades else 0
            ),
        }
        
        # Semantic patterns
        if include_patterns:
            patterns = await self.semantic.get_all_patterns(limit=20)
            summary["semantic"] = {
                "total_patterns": len(patterns),
                "patterns": [
                    {
                        "type": p.pattern_type.value,
                        "description": p.description,
                        "confidence": p.confidence,
                    }
                    for p in patterns
                ],
            }
        
        # Policy history
        if include_policy_history:
            policy_snapshots = await self.policy.get_recent_snapshots(days)
            summary["policy"] = {
                "total_snapshots": len(policy_snapshots),
                "mode_changes": len(set(s.global_mode for s in policy_snapshots)),
            }
        
        # Publish summary event
        await self.event_bus.publish("memory_summary_created", summary)
        
        return summary
    
    # -------------------------------------------------------------------------
    # Event handlers (automatic memory storage)
    # -------------------------------------------------------------------------
    
    async def _handle_position_opened(self, event: dict[str, Any]) -> None:
        """Handle position_opened event."""
        try:
            await self.store_event(
                event_type=EpisodeType.TRADE,
                data={
                    "action": "opened",
                    "symbol": event.get("symbol"),
                    "side": event.get("side"),
                    "size": event.get("size"),
                    "entry_price": event.get("price"),
                },
                trace_id=event.get("trace_id"),
                tags=["position_opened", event.get("symbol", "")],
            )
        except Exception as e:
            logger.error(f"Error storing position_opened: {e}")
    
    async def _handle_position_closed(self, event: dict[str, Any]) -> None:
        """Handle position_closed event."""
        try:
            pnl = event.get("pnl", 0)
            tags = ["position_closed", event.get("symbol", "")]
            
            if pnl > 0:
                tags.append("profitable")
            elif pnl < 0:
                tags.append("loss")
            
            await self.store_event(
                event_type=EpisodeType.TRADE,
                data={
                    "action": "closed",
                    "symbol": event.get("symbol"),
                    "side": event.get("side"),
                    "size": event.get("size"),
                    "exit_price": event.get("price"),
                    "pnl": pnl,
                    "duration": event.get("duration"),
                },
                trace_id=event.get("trace_id"),
                pnl=pnl,
                tags=tags,
            )
        except Exception as e:
            logger.error(f"Error storing position_closed: {e}")
    
    async def _handle_ceo_decision(self, event: dict[str, Any]) -> None:
        """Handle ceo_decision event."""
        try:
            await self.store_event(
                event_type=EpisodeType.CEO_DECISION,
                data={
                    "decision": event.get("decision"),
                    "reasoning": event.get("reasoning"),
                    "mode": event.get("mode"),
                    "confidence": event.get("confidence"),
                },
                trace_id=event.get("trace_id"),
                global_mode=event.get("mode"),
                tags=["ceo_decision", event.get("mode", "")],
            )
        except Exception as e:
            logger.error(f"Error storing ceo_decision: {e}")
    
    async def _handle_ceo_mode_switch(self, event: dict[str, Any]) -> None:
        """Handle ceo_mode_switch event."""
        try:
            await self.store_event(
                event_type=EpisodeType.CEO_DECISION,
                data={
                    "action": "mode_switch",
                    "old_mode": event.get("old_mode"),
                    "new_mode": event.get("new_mode"),
                    "reason": event.get("reason"),
                },
                trace_id=event.get("trace_id"),
                global_mode=event.get("new_mode"),
                tags=["mode_switch", event.get("new_mode", "")],
            )
        except Exception as e:
            logger.error(f"Error storing ceo_mode_switch: {e}")
    
    async def _handle_risk_alert(self, event: dict[str, Any]) -> None:
        """Handle risk_alert event."""
        try:
            severity = event.get("severity", "MEDIUM")
            
            await self.store_event(
                event_type=EpisodeType.RISK_EVENT,
                data={
                    "alert_type": event.get("alert_type"),
                    "severity": severity,
                    "message": event.get("message"),
                    "risk_score": event.get("risk_score"),
                },
                trace_id=event.get("trace_id"),
                risk_score=event.get("risk_score"),
                tags=["risk_alert", severity.lower()],
            )
        except Exception as e:
            logger.error(f"Error storing risk_alert: {e}")
    
    async def _handle_strategy_recommendation(self, event: dict[str, Any]) -> None:
        """Handle strategy_recommendation event."""
        try:
            await self.store_event(
                event_type=EpisodeType.STRATEGY_SHIFT,
                data={
                    "primary_strategy": event.get("primary_strategy"),
                    "disabled_strategies": event.get("disabled_strategies", []),
                    "reason": event.get("reason"),
                },
                trace_id=event.get("trace_id"),
                tags=["strategy_recommendation"],
            )
        except Exception as e:
            logger.error(f"Error storing strategy_recommendation: {e}")
    
    async def _handle_system_degraded(self, event: dict[str, Any]) -> None:
        """Handle system_degraded event."""
        try:
            await self.store_event(
                event_type=EpisodeType.SYSTEM_EVENT,
                data={
                    "event": "degraded",
                    "reason": event.get("reason"),
                    "affected_components": event.get("affected_components", []),
                },
                trace_id=event.get("trace_id"),
                tags=["system_degraded", "outage"],
            )
        except Exception as e:
            logger.error(f"Error storing system_degraded: {e}")
    
    async def _handle_regime_detected(self, event: dict[str, Any]) -> None:
        """Handle regime_detected event."""
        try:
            await self.store_event(
                event_type=EpisodeType.REGIME_CHANGE,
                data={
                    "old_regime": event.get("old_regime"),
                    "new_regime": event.get("new_regime"),
                    "confidence": event.get("confidence"),
                },
                trace_id=event.get("trace_id"),
                regime=event.get("new_regime"),
                tags=["regime_change", event.get("new_regime", "")],
            )
        except Exception as e:
            logger.error(f"Error storing regime_detected: {e}")
    
    async def _handle_policy_updated(self, event: dict[str, Any]) -> None:
        """Handle policy_updated event."""
        try:
            # Store in policy memory
            policy = await self.policy_store.get_policy()
            
            await self.policy.log_policy_state(
                policy_config=policy.dict(),
                context={
                    "updated_by": event.get("updated_by"),
                    "reason": event.get("reason"),
                },
            )
        except Exception as e:
            logger.error(f"Error storing policy_updated: {e}")
    
    async def get_status(self) -> dict[str, Any]:
        """Get memory engine status."""
        episodic_stats = await self.episodic.get_statistics()
        semantic_stats = await self.semantic.get_statistics()
        policy_stats = await self.policy.get_statistics()
        
        return {
            "instance_id": self._instance_id,
            "running": self._running,
            "episodic": episodic_stats,
            "semantic": semantic_stats,
            "policy": policy_stats,
        }
