"""AI Strategy Officer - Intelligent strategy monitoring and recommendation agent.

The AI Strategy Officer continuously monitors strategy and model performance,
providing recommendations for optimal strategy selection and configuration.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from backend.ai_strategy.strategy_brain import (
    ModelPerformance,
    StrategyBrain,
    StrategyPerformance,
    StrategyRecommendation,
)
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore

logger = logging.getLogger(__name__)


class AI_StrategyOfficer:
    """
    AI Strategy Officer - Intelligent Strategy Monitoring Agent.
    
    Responsibilities:
    - Monitor strategy performance (trend, mean reversion, breakout, range)
    - Monitor model performance (XGBoost, LightGBM, N-HiTS, PatchTST)
    - Analyze win rates, Sharpe ratios, profit factors
    - Recommend primary and fallback strategies
    - Identify underperforming strategies
    - Suggest meta-strategy adjustments
    - Provide strategy state to AI CEO
    
    Integration:
    - Runs in analytics-os-service
    - Uses EventBus v2 for communication
    - Publishes strategy_state_update, strategy_recommendation, strategy_alert
    - Integrates with Analytics OS (PBA, PIL, PAL)
    """
    
    DEFAULT_ANALYSIS_INTERVAL = 60.0  # seconds
    
    def __init__(
        self,
        redis_client: Redis,
        event_bus: EventBus,
        policy_store: PolicyStore,
        brain: Optional[StrategyBrain] = None,
        analysis_interval: float = DEFAULT_ANALYSIS_INTERVAL,
    ):
        """
        Initialize AI Strategy Officer.
        
        Args:
            redis_client: Async Redis client
            event_bus: EventBus instance
            policy_store: PolicyStore instance
            brain: Optional StrategyBrain instance (for testing)
            analysis_interval: Seconds between analyses
        """
        self.redis = redis_client
        self.event_bus = event_bus
        self.policy_store = policy_store
        self.brain = brain or StrategyBrain()
        self.analysis_interval = analysis_interval
        
        # State tracking
        self._running = False
        self._analysis_task: Optional[asyncio.Task] = None
        
        # Cached data
        self._strategy_performances: dict[str, StrategyPerformance] = {}
        self._model_performances: dict[str, ModelPerformance] = {}
        self._current_regime = "UNKNOWN"
        
        # Instance ID
        self._instance_id = f"ai_so_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"AI_StrategyOfficer initialized: instance_id={self._instance_id}, "
            f"analysis_interval={analysis_interval}s"
        )
    
    async def initialize(self) -> None:
        """Initialize AI Strategy Officer."""
        # Verify EventBus
        await self.event_bus.initialize()
        
        # Verify PolicyStore
        policy = await self.policy_store.get_policy()
        logger.info(f"AI_SO verified PolicyStore: mode={policy.active_mode.value}")
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Load initial performance data
        await self._load_performance_data()
        
        logger.info("AI_StrategyOfficer initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        # Position events
        self.event_bus.subscribe("position_opened", self._handle_position_event)
        self.event_bus.subscribe("position_closed", self._handle_position_event)
        
        # Strategy events
        self.event_bus.subscribe("strategy_executed", self._handle_strategy_executed)
        
        # Model events
        self.event_bus.subscribe("model_updated", self._handle_model_updated)
        self.event_bus.subscribe("model_prediction", self._handle_model_prediction)
        
        # Regime updates
        self.event_bus.subscribe("regime_detected", self._handle_regime_detected)
        
        logger.info("AI_SO subscribed to events")
    
    async def _load_performance_data(self) -> None:
        """Load strategy and model performance data from Redis/database."""
        try:
            # Try to load cached performance data
            # In production, this would query from database/analytics
            logger.info("AI_SO loading performance data...")
            
            # Initialize with sample data for now
            # In production, query actual performance from PAL/PIL
            self._strategy_performances = {}
            self._model_performances = {}
            
            logger.info("AI_SO performance data loaded")
        except Exception as e:
            logger.error(f"Failed to load performance data: {e}")
    
    async def start(self) -> None:
        """Start AI Strategy Officer analysis loop."""
        if self._running:
            logger.warning("AI_SO already running")
            return
        
        self._running = True
        
        # Start EventBus processing
        await self.event_bus.start()
        
        # Start analysis loop
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("AI_StrategyOfficer started")
    
    async def stop(self) -> None:
        """Stop AI Strategy Officer gracefully."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel analysis task
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AI_StrategyOfficer stopped")
    
    async def _analysis_loop(self) -> None:
        """Main strategy analysis loop."""
        logger.info(f"AI_SO analysis loop started (interval={self.analysis_interval}s)")
        
        while self._running:
            try:
                # Check if Strategy Officer is enabled
                policy = await self.policy_store.get_policy()
                active_config = policy.get_active_config()
                
                # Check for enable_ai_so flag (if exists)
                if hasattr(active_config, 'enable_ai_so') and not active_config.enable_ai_so:
                    logger.debug("AI_SO disabled in PolicyStore")
                    await asyncio.sleep(self.analysis_interval)
                    continue
                
                # Gather performance data
                await self._refresh_performance_data()
                
                # Generate trace ID
                trace_id = f"so_analysis_{uuid.uuid4().hex[:12]}"
                
                # Perform analysis
                strategies = list(self._strategy_performances.values())
                models = list(self._model_performances.values())
                
                recommendation = self.brain.analyze_and_recommend(
                    strategies=strategies,
                    models=models,
                    current_regime=self._current_regime,
                    current_primary_strategy=active_config.primary_strategy if hasattr(active_config, 'primary_strategy') else None,
                )
                
                # Log recommendation
                logger.info(
                    f"[{trace_id}] Strategy Recommendation: "
                    f"primary={recommendation.primary_strategy}, "
                    f"meta={recommendation.meta_strategy_mode}, "
                    f"win_rate={recommendation.overall_win_rate:.2%}, "
                    f"confidence={recommendation.confidence:.2f}"
                )
                
                # Publish recommendation
                await self._publish_recommendation(recommendation, trace_id)
                
            except Exception as e:
                logger.error(f"Error in strategy analysis loop: {e}", exc_info=True)
            
            # Wait for next cycle
            await asyncio.sleep(self.analysis_interval)
    
    async def _refresh_performance_data(self) -> None:
        """
        Refresh strategy and model performance data.
        
        In production, this would:
        - Query PAL (Portfolio Analysis Logger) for strategy performance
        - Query PIL (Position Info Logger) for trade outcomes
        - Query Model Supervisor for model performance
        - Calculate performance metrics
        """
        # For now, maintain cached data from events
        # In production, implement actual data refresh from analytics DB
        pass
    
    async def _publish_recommendation(
        self,
        recommendation: StrategyRecommendation,
        trace_id: str,
    ) -> None:
        """Publish strategy recommendation to EventBus."""
        # Publish strategy state update (for AI CEO)
        await self.event_bus.publish(
            "strategy_state_update",
            {
                "win_rate": recommendation.overall_win_rate,
                "sharpe_ratio": recommendation.overall_sharpe,
                "primary_strategy": recommendation.primary_strategy,
                "meta_strategy_mode": recommendation.meta_strategy_mode,
                "best_performing": recommendation.best_performing_strategy,
                "worst_performing": recommendation.worst_performing_strategy,
                "confidence": recommendation.confidence,
            },
            trace_id=trace_id,
        )
        
        # Publish full recommendation
        await self.event_bus.publish(
            "strategy_recommendation",
            recommendation.to_dict(),
            trace_id=trace_id,
        )
        
        # Publish alerts if needed
        if recommendation.alerts:
            await self.event_bus.publish(
                "strategy_alert",
                {
                    "alerts": recommendation.alerts,
                    "primary_strategy": recommendation.primary_strategy,
                    "disabled_strategies": recommendation.disabled_strategies,
                    "timestamp": recommendation.recommendation_timestamp.isoformat(),
                },
                trace_id=trace_id,
            )
    
    # Event handlers
    
    async def _handle_position_event(self, event_data: dict) -> None:
        """Handle position opened/closed events."""
        logger.debug(f"AI_SO received position event: {event_data.get('event_type')}")
        # Update strategy performance based on position outcome
        
        payload = event_data.get("payload", {})
        if "strategy_name" in payload and "pnl" in payload:
            strategy_name = payload["strategy_name"]
            pnl = payload["pnl"]
            
            # Update strategy performance
            if strategy_name in self._strategy_performances:
                perf = self._strategy_performances[strategy_name]
                perf.total_trades += 1
                
                if pnl > 0:
                    perf.winning_trades += 1
                else:
                    perf.losing_trades += 1
                
                perf.total_pnl += pnl
                perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0.0
                perf.last_update = datetime.utcnow()
    
    async def _handle_strategy_executed(self, event_data: dict) -> None:
        """Handle strategy executed event."""
        logger.debug(f"AI_SO received strategy executed: {event_data}")
    
    async def _handle_model_updated(self, event_data: dict) -> None:
        """Handle model updated/retrained event."""
        payload = event_data.get("payload", {})
        model_name = payload.get("model_name")
        
        if model_name and model_name in self._model_performances:
            self._model_performances[model_name].last_retrain = datetime.utcnow()
            logger.info(f"AI_SO updated model retrain time: {model_name}")
    
    async def _handle_model_prediction(self, event_data: dict) -> None:
        """Handle model prediction event."""
        payload = event_data.get("payload", {})
        model_name = payload.get("model_name")
        confidence = payload.get("confidence", 0.0)
        
        if model_name:
            # Track prediction
            if model_name not in self._model_performances:
                # Initialize model performance tracking
                self._model_performances[model_name] = ModelPerformance(
                    model_name=model_name,
                    model_type="unknown",
                    total_predictions=0,
                    correct_predictions=0,
                    accuracy=0.5,
                    avg_confidence=0.0,
                    high_confidence_accuracy=0.5,
                    predictions_traded=0,
                    trades_profitable=0,
                    economic_win_rate=0.5,
                    last_prediction=datetime.utcnow(),
                    last_retrain=datetime.utcnow() - timedelta(days=1),
                    is_enabled=True,
                )
            
            perf = self._model_performances[model_name]
            perf.total_predictions += 1
            perf.last_prediction = datetime.utcnow()
    
    async def _handle_regime_detected(self, event_data: dict) -> None:
        """Handle regime detection event."""
        payload = event_data.get("payload", {})
        regime = payload.get("regime", "UNKNOWN")
        
        if regime != self._current_regime:
            logger.info(f"AI_SO detected regime change: {self._current_regime} → {regime}")
            self._current_regime = regime
    
    # Public API
    
    async def get_status(self) -> dict:
        """Get AI Strategy Officer status."""
        recent_recommendations = self.brain.get_recommendation_history(limit=10)
        
        return {
            "instance_id": self._instance_id,
            "running": self._running,
            "analysis_interval": self.analysis_interval,
            "tracked_strategies": len(self._strategy_performances),
            "tracked_models": len(self._model_performances),
            "current_regime": self._current_regime,
            "recent_recommendations_count": len(recent_recommendations),
            "last_recommendation": recent_recommendations[-1].to_dict() if recent_recommendations else None,
        }
    
    async def force_analysis(self) -> StrategyRecommendation:
        """Force immediate strategy analysis (for testing/debugging)."""
        logger.info("AI_SO forcing immediate analysis")
        
        await self._refresh_performance_data()
        
        strategies = list(self._strategy_performances.values())
        models = list(self._model_performances.values())
        
        recommendation = self.brain.analyze_and_recommend(
            strategies=strategies,
            models=models,
            current_regime=self._current_regime,
        )
        
        trace_id = f"so_forced_{uuid.uuid4().hex[:8]}"
        await self._publish_recommendation(recommendation, trace_id)
        
        return recommendation
    
    async def register_strategy_performance(
        self,
        strategy_name: str,
        performance: StrategyPerformance,
    ) -> None:
        """Register or update strategy performance data (for manual updates)."""
        self._strategy_performances[strategy_name] = performance
        logger.info(f"AI_SO registered strategy performance: {strategy_name}")
    
    async def register_model_performance(
        self,
        model_name: str,
        performance: ModelPerformance,
    ) -> None:
        """Register or update model performance data (for manual updates)."""
        self._model_performances[model_name] = performance
        logger.info(f"AI_SO registered model performance: {model_name}")
