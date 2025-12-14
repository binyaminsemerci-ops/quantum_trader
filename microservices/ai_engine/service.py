"""
AI Engine Service - Core Service Logic

Orchestrates AI inference pipeline:
1. Market data → Ensemble models
2. Ensemble → Signal + Confidence
3. Signal → Meta-Strategy Selector → Strategy
4. Signal + Strategy → RL Position Sizing → Size + Leverage
5. Publish ai.decision.made event → execution-service
"""
import asyncio
import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Core dependencies
from backend.core.event_bus import EventBus
from backend.core.event_buffer import EventBuffer
from backend.core.health_contract import (
    ServiceHealth, DependencyHealth, DependencyStatus,
    check_redis_health, check_http_endpoint_health
)

# Local imports
from .models import (
    MarketTickEvent, MarketKlineEvent, TradeClosedEvent, PolicyUpdatedEvent,
    AISignalGeneratedEvent, StrategySelectedEvent, SizingDecidedEvent, AIDecisionMadeEvent,
    SignalAction, MarketRegime, StrategyID,
    ComponentHealth
    # NOTE: ServiceHealth removed from here - using health_contract version instead
)
from .config import settings

logger = logging.getLogger(__name__)


class AIEngineService:
    """
    AI Engine Service
    
    Responsibilities:
    - AI model inference (ensemble voting)
    - Meta-strategy selection (RL-based)
    - RL position sizing
    - Market regime detection
    - Trade intent generation
    """
    
    def __init__(self):
        # Core components (initialized in start())
        self.event_bus: Optional[EventBus] = None
        self.event_buffer: Optional[EventBuffer] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # AI modules (lazy-loaded)
        self.ensemble_manager = None
        self.meta_strategy_selector = None
        self.rl_sizing_agent = None
        self.regime_detector = None
        self.memory_manager = None
        self.model_supervisor = None
        
        # State tracking
        self._running = False
        self._event_loop_task: Optional[asyncio.Task] = None
        self._regime_update_task: Optional[asyncio.Task] = None
        self._signals_generated = 0
        self._models_loaded = 0
        self._start_time = datetime.now(timezone.utc)
        
        logger.info("[AI-ENGINE] Service initialized")
    
    async def start(self):
        """Start the service."""
        if self._running:
            logger.warning("[AI-ENGINE] Service already running")
            return
        
        try:
            # Initialize Redis client
            import redis.asyncio as redis
            logger.info(f"[AI-ENGINE] Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}...")
            redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False
            )
            await redis_client.ping()
            logger.info("[AI-ENGINE] ✅ Redis connected")
            
            # Initialize EventBus
            logger.info("[AI-ENGINE] Initializing EventBus...")
            self.event_bus = EventBus(redis_client=redis_client, service_name="ai-engine")
            
            # Subscribe to events
            self.event_bus.subscribe("market.tick", self._handle_market_tick)
            self.event_bus.subscribe("market.klines", self._handle_market_klines)
            self.event_bus.subscribe("trade.closed", self._handle_trade_closed)
            self.event_bus.subscribe("policy.updated", self._handle_policy_updated)
            logger.info("[AI-ENGINE] ✅ EventBus subscriptions active")
            
            # Initialize EventBuffer
            from pathlib import Path
            self.event_buffer = EventBuffer(buffer_dir=Path("data/event_buffers/ai_engine"))
            logger.info("[AI-ENGINE] ✅ EventBuffer ready")
            
            # Initialize HTTP client for risk-safety-service
            self.http_client = httpx.AsyncClient(
                base_url=settings.RISK_SAFETY_SERVICE_URL,
                timeout=5.0
            )
            logger.info(f"[AI-ENGINE] ✅ HTTP client ready (risk-safety: {settings.RISK_SAFETY_SERVICE_URL})")
            
            # Load AI modules
            await self._load_ai_modules()
            
            # Start background tasks
            self._running = True
            self._event_loop_task = asyncio.create_task(self._event_processing_loop())
            if settings.REGIME_DETECTION_ENABLED:
                self._regime_update_task = asyncio.create_task(self._regime_update_loop())
            
            logger.info("[AI-ENGINE] ✅ Service started successfully")
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] ❌ Failed to start service: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the service."""
        if not self._running:
            return
        
        logger.info("[AI-ENGINE] Stopping service...")
        self._running = False
        
        # Cancel background tasks
        for task in [self._event_loop_task, self._regime_update_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush event buffer
        # NOTE: EventBuffer.flush() not implemented yet
        # if self.event_buffer:
        #     self.event_buffer.flush()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("[AI-ENGINE] ✅ Service stopped")
    
    # ========================================================================
    # AI MODULE LOADING
    # ========================================================================
    
    async def _load_ai_modules(self):
        """Load all AI modules."""
        logger.info("[AI-ENGINE] Loading AI modules...")
        
        try:
            # 1. Ensemble Manager
            if settings.ENSEMBLE_MODELS:
                logger.info(f"[AI-ENGINE] Loading ensemble: {settings.ENSEMBLE_MODELS}")
                # DISABLED TEMPORARILY - ensemble_manager.py missing
                logger.warning("[AI-ENGINE] ⚠️ Ensemble loading disabled (module not found)")
                # from ai_engine.ensemble_manager import EnsembleManager
                # 
                # self.ensemble_manager = EnsembleManager(
                #     weights=settings.ENSEMBLE_WEIGHTS,
                #     min_consensus=settings.MIN_CONSENSUS,
                #     xgb_model_path=settings.XGB_MODEL_PATH,
                #     xgb_scaler_path=settings.XGB_SCALER_PATH
                # )
                # self._models_loaded += len(settings.ENSEMBLE_MODELS)
                # logger.info(f"[AI-ENGINE] ✅ Ensemble loaded ({self._models_loaded} models)")
            
            # 2. Meta-Strategy Selector
            # DISABLED TEMPORARILY - constructor signature mismatch
            logger.warning("[AI-ENGINE] ⚠️ Meta-Strategy loading disabled (parameter mismatch)")
            # if settings.META_STRATEGY_ENABLED:
            #     logger.info("[AI-ENGINE] Loading Meta-Strategy Selector...")
            #     from backend.services.ai.meta_strategy_selector import MetaStrategySelector
            #     
            #     self.meta_strategy_selector = MetaStrategySelector(
            #         epsilon=settings.META_STRATEGY_EPSILON,
            #         alpha=settings.META_STRATEGY_ALPHA,
            #         discount=settings.META_STRATEGY_DISCOUNT,
            #         state_path=settings.META_STRATEGY_STATE_PATH
            #     )
            #     logger.info("[AI-ENGINE] ✅ Meta-Strategy Selector loaded")
            
            # 3. RL Position Sizing Agent
            # DISABLED TEMPORARILY - module loading issues
            logger.warning("[AI-ENGINE] ⚠️ RL Position Sizing loading disabled")
            # if settings.RL_SIZING_ENABLED:
            #     logger.info("[AI-ENGINE] Loading RL Position Sizing Agent...")
            #     from backend.services.ai.rl_position_sizing_agent import RLPositionSizingAgent
            #     
            #     self.rl_sizing_agent = RLPositionSizingAgent(
            #         policy_store=None,  # Will fetch from risk-safety-service
            #         epsilon=settings.RL_SIZING_EPSILON,
            #         alpha=settings.RL_SIZING_ALPHA,
            #         discount=settings.RL_SIZING_DISCOUNT,
            #         state_path=settings.RL_SIZING_STATE_PATH
            #     )
            #     logger.info("[AI-ENGINE] ✅ RL Position Sizing loaded")
            
            # 4. Regime Detector
            # DISABLED TEMPORARILY - module loading issues
            logger.warning("[AI-ENGINE] ⚠️ Regime Detector loading disabled")
            # if settings.REGIME_DETECTION_ENABLED:
            #     logger.info("[AI-ENGINE] Loading Regime Detector...")
            #     from backend.services.ai.regime_detector import RegimeDetector
            #     
            #     self.regime_detector = RegimeDetector()
            #     logger.info("[AI-ENGINE] ✅ Regime Detector loaded")
            
            # 5. Memory State Manager
            # DISABLED TEMPORARILY - parameter mismatch
            logger.warning("[AI-ENGINE] ⚠️ Memory State Manager loading disabled")
            # if settings.MEMORY_STATE_ENABLED:
            #     logger.info("[AI-ENGINE] Loading Memory State Manager...")
            #     from backend.services.ai.memory_state_manager import MemoryStateManager
            #     
            #     self.memory_manager = MemoryStateManager(
            #         lookback_hours=settings.MEMORY_LOOKBACK_HOURS
            #     )
            #     logger.info("[AI-ENGINE] ✅ Memory State Manager loaded")
            
            # 6. Model Supervisor
            # DISABLED TEMPORARILY - module loading issues
            logger.warning("[AI-ENGINE] ⚠️ Model Supervisor loading disabled")
            # if settings.MODEL_SUPERVISOR_ENABLED:
            #     logger.info("[AI-ENGINE] Loading Model Supervisor...")
            #     from backend.services.ai.model_supervisor import ModelSupervisor
            #     
            #     self.model_supervisor = ModelSupervisor(
            #         bias_threshold=settings.MODEL_SUPERVISOR_BIAS_THRESHOLD,
            #         min_samples=settings.MODEL_SUPERVISOR_MIN_SAMPLES
            #     )
            #     logger.info("[AI-ENGINE] ✅ Model Supervisor loaded")
            
            logger.info(f"[AI-ENGINE] ✅ All AI modules loaded (0 models - all disabled for now)")
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] ❌ Failed to load AI modules: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================
    
    async def _handle_market_tick(self, event_data: Dict[str, Any]):
        """
        Handle market.tick event.
        
        This is the main trigger for signal generation.
        
        Flow:
        1. Extract symbol + price from event
        2. Run ensemble inference
        3. Run meta-strategy selection
        4. Run RL position sizing
        5. Publish ai.decision.made event
        """
        try:
            symbol = event_data.get("symbol")
            price = event_data.get("price", 0.0)
            
            if not symbol or price <= 0:
                return
            
            logger.debug(f"[AI-ENGINE] Market tick: {symbol} @ {price}")
            
            # Generate full signal
            decision = await self.generate_signal(symbol, current_price=price)
            
            if decision:
                self._signals_generated += 1
                logger.info(
                    f"[AI-ENGINE] 🎯 Decision: {symbol} {decision.side.upper()} "
                    f"(confidence={decision.confidence:.2f}, size=${decision.position_size_usd:.0f})"
                )
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error handling market.tick: {e}", exc_info=True)
    
    async def _handle_market_klines(self, event_data: Dict[str, Any]):
        """Handle market.klines event (candle data)."""
        try:
            symbol = event_data.get("symbol")
            logger.debug(f"[AI-ENGINE] Market klines update: {symbol}")
            
            # Update regime detector if enabled
            if self.regime_detector and settings.REGIME_DETECTION_ENABLED:
                # TODO: Update regime with new candle data
                pass
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error handling market.klines: {e}", exc_info=True)
    
    async def _handle_trade_closed(self, event_data: Dict[str, Any]):
        """
        Handle trade.closed event from execution-service.
        
        This is used for continuous learning:
        - Update Meta-Strategy Q-values
        - Update RL Sizing Q-values
        - Feed to Continuous Learning Manager
        """
        try:
            trade_id = event_data.get("trade_id")
            pnl_percent = event_data.get("pnl_percent", 0.0)
            model = event_data.get("model", "unknown")
            strategy = event_data.get("strategy")
            
            logger.info(f"[AI-ENGINE] Trade closed: {trade_id} (PnL={pnl_percent:.2f}%, model={model})")
            
            # TODO: Update Meta-Strategy Q-table
            if self.meta_strategy_selector and strategy:
                # reward = pnl_percent / 100.0  # Convert to R units
                pass
            
            # TODO: Update RL Sizing Q-table
            if self.rl_sizing_agent:
                pass
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error handling trade.closed: {e}", exc_info=True)
    
    async def _handle_policy_updated(self, event_data: Dict[str, Any]):
        """Handle policy.updated event from risk-safety-service."""
        key = event_data.get("key")
        new_value = event_data.get("new_value")
        logger.info(f"[AI-ENGINE] Policy updated: {key} = {new_value}")
        
        # TODO: Refresh policy snapshot if needed
    
    # ========================================================================
    # SIGNAL GENERATION (MAIN PIPELINE)
    # ========================================================================
    
    async def generate_signal(
        self,
        symbol: str,
        current_price: Optional[float] = None
    ) -> Optional[AIDecisionMadeEvent]:
        """
        Generate complete AI trading signal.
        
        Pipeline:
        1. Ensemble inference → action + confidence
        2. Meta-Strategy selection → strategy
        3. RL Position Sizing → size + leverage + TP/SL
        4. Publish ai.decision.made event
        
        Returns:
            AIDecisionMadeEvent if signal generated, None otherwise
        """
        try:
            # Step 1: Ensemble inference
            if not self.ensemble_manager:
                logger.warning(f"[AI-ENGINE] Ensemble not loaded, skipping {symbol}")
                return None
            
            logger.debug(f"[AI-ENGINE] Running ensemble for {symbol}...")
            
            # TODO: Fetch market data (OHLCV) for symbol
            # For now, placeholder
            market_data = {}
            
            ensemble_result = await asyncio.to_thread(
                self.ensemble_manager.predict,
                symbol=symbol,
                market_data=market_data
            )
            
            if not ensemble_result or ensemble_result.get("action") == "HOLD":
                logger.debug(f"[AI-ENGINE] No actionable signal for {symbol}")
                return None
            
            action = ensemble_result["action"]
            ensemble_confidence = ensemble_result.get("confidence", 0.0)
            model_votes = ensemble_result.get("votes", {})
            consensus = ensemble_result.get("consensus", 0)
            
            # Check minimum confidence
            if ensemble_confidence < settings.MIN_SIGNAL_CONFIDENCE:
                logger.debug(
                    f"[AI-ENGINE] Signal rejected: {symbol} confidence={ensemble_confidence:.2f} "
                    f"< {settings.MIN_SIGNAL_CONFIDENCE:.2f}"
                )
                return None
            
            logger.info(
                f"[AI-ENGINE] ✅ Ensemble: {symbol} {action} "
                f"(confidence={ensemble_confidence:.2f}, consensus={consensus}/4)"
            )
            
            # Publish intermediate event
            await self.event_bus.publish("ai.signal_generated", AISignalGeneratedEvent(
                symbol=symbol,
                action=SignalAction(action.lower()),
                confidence=ensemble_confidence,
                ensemble_confidence=ensemble_confidence,
                model_votes=model_votes,
                consensus=consensus,
                timestamp=datetime.now(timezone.utc).isoformat()
            ).dict())
            
            # Step 2: Meta-Strategy selection
            strategy_id = StrategyID.DEFAULT
            strategy_name = "default"
            regime = MarketRegime.UNKNOWN
            
            if self.meta_strategy_selector:
                logger.debug(f"[AI-ENGINE] Selecting strategy for {symbol}...")
                
                # Detect regime
                if self.regime_detector:
                    regime = await asyncio.to_thread(
                        self.regime_detector.detect_regime,
                        symbol=symbol
                    )
                
                # Select strategy
                strategy_decision = await asyncio.to_thread(
                    self.meta_strategy_selector.select_strategy,
                    symbol=symbol,
                    regime=regime,
                    confidence=ensemble_confidence
                )
                
                strategy_id = strategy_decision.strategy_id
                strategy_name = strategy_decision.strategy_profile.name
                
                logger.info(
                    f"[AI-ENGINE] ✅ Strategy: {symbol} → {strategy_name} "
                    f"(regime={regime.value}, q={strategy_decision.q_values.get(strategy_id.value, 0.0):.3f})"
                )
                
                # Publish intermediate event
                await self.event_bus.publish("strategy.selected", StrategySelectedEvent(
                    symbol=symbol,
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    regime=regime,
                    confidence=strategy_decision.confidence,
                    reasoning=strategy_decision.reasoning,
                    is_exploration=strategy_decision.is_exploration,
                    q_values=strategy_decision.q_values,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ).dict())
            
            # Step 3: RL Position Sizing
            position_size_usd = 100.0  # Default
            leverage = 1
            tp_percent = 0.06  # 6%
            sl_percent = 0.025  # 2.5%
            
            if self.rl_sizing_agent:
                logger.debug(f"[AI-ENGINE] Calculating position size for {symbol}...")
                
                # TODO: Get portfolio state from execution-service
                portfolio_exposure = 0.0
                
                sizing_decision = await asyncio.to_thread(
                    self.rl_sizing_agent.decide_size,
                    symbol=symbol,
                    confidence=ensemble_confidence,
                    regime=regime,
                    portfolio_exposure=portfolio_exposure
                )
                
                position_size_usd = sizing_decision.position_size_usd
                leverage = int(sizing_decision.leverage)
                tp_percent = sizing_decision.tp_percent
                sl_percent = sizing_decision.sl_percent
                
                logger.info(
                    f"[AI-ENGINE] ✅ Sizing: {symbol} ${position_size_usd:.0f} @ {leverage}x "
                    f"(risk={sizing_decision.risk_pct:.2f}%, TP={tp_percent*100:.1f}%, SL={sl_percent*100:.1f}%)"
                )
                
                # Publish intermediate event
                await self.event_bus.publish("sizing.decided", SizingDecidedEvent(
                    symbol=symbol,
                    position_size_usd=position_size_usd,
                    leverage=leverage,
                    risk_pct=sizing_decision.risk_pct,
                    confidence=sizing_decision.confidence,
                    reasoning=sizing_decision.reasoning,
                    tp_percent=tp_percent,
                    sl_percent=sl_percent,
                    partial_tp_enabled=sizing_decision.partial_tp_enabled,
                    timestamp=datetime.now(timezone.utc).isoformat()
                ).dict())
            
            # Step 4: Build final decision
            decision = AIDecisionMadeEvent(
                symbol=symbol,
                side=SignalAction(action.lower()),
                confidence=ensemble_confidence,
                entry_price=current_price,
                quantity=position_size_usd,
                leverage=leverage,
                stop_loss=current_price * (1 - sl_percent) if action.upper() == "BUY" else current_price * (1 + sl_percent),
                take_profit=current_price * (1 + tp_percent) if action.upper() == "BUY" else current_price * (1 - tp_percent),
                trail_percent=None,  # No trailing for now
                model="ensemble",
                ensemble_confidence=ensemble_confidence,
                strategy=strategy_name,
                meta_strategy=strategy_id.value,
                regime=regime,
                position_size_usd=position_size_usd,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Step 5: Publish final decision
            await self.event_bus.publish("ai.decision.made", decision.dict())
            
            logger.info(
                f"[AI-ENGINE] 🚀 AI DECISION PUBLISHED: {symbol} {action} "
                f"(${position_size_usd:.0f} @ {leverage}x, confidence={ensemble_confidence:.2f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error generating signal for {symbol}: {e}", exc_info=True)
            return None
    
    # ========================================================================
    # BACKGROUND TASKS
    # ========================================================================
    
    async def _event_processing_loop(self):
        """Background loop for processing buffered events."""
        logger.info("[AI-ENGINE] Event processing loop started")
        
        while self._running:
            try:
                # Process buffered events
                # NOTE: EventBuffer.pop() not implemented yet - using EventBus directly
                # if self.event_buffer:
                #     while True:
                #         event = self.event_buffer.pop()
                #         if event is None:
                #             break
                #         
                #         event_type = event.get("type")
                #         event_data = event.get("data")
                #         if event_type and event_data:
                #             await self.event_bus.publish(event_type, event_data)
                #             logger.debug(f"[AI-ENGINE] Replayed buffered event: {event_type}")
                
                await asyncio.sleep(5.0)  # Reduced frequency since not processing
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AI-ENGINE] Error in event processing loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
        
        logger.info("[AI-ENGINE] Event processing loop stopped")
    
    async def _regime_update_loop(self):
        """Background loop for updating market regime."""
        logger.info("[AI-ENGINE] Regime update loop started")
        
        while self._running:
            try:
                if self.regime_detector:
                    # TODO: Update regime for all active symbols
                    pass
                
                await asyncio.sleep(settings.REGIME_UPDATE_INTERVAL_SEC)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AI-ENGINE] Error in regime update loop: {e}", exc_info=True)
                await asyncio.sleep(60.0)
        
        logger.info("[AI-ENGINE] Regime update loop stopped")
    
    # ========================================================================
    # QUERY METHODS (for REST API)
    # ========================================================================
    
    async def get_health(self) -> Dict[str, Any]:
        """Get service health status (standardized format)."""
        dependencies = {}
        
        # Check EventBus (Redis)
        try:
            if hasattr(self.event_bus, 'redis_client') and self.event_bus.redis_client:
                dependencies["redis"] = await check_redis_health(self.event_bus.redis_client)
            else:
                dependencies["redis"] = DependencyHealth(
                    status=DependencyStatus.DOWN,
                    error="Redis client not initialized"
                )
        except Exception as e:
            dependencies["redis"] = DependencyHealth(
                status=DependencyStatus.DOWN,
                error=str(e)
            )
        
        # Check EventBus overall
        dependencies["eventbus"] = DependencyHealth(
            status=DependencyStatus.OK if self.event_bus and self._running else DependencyStatus.DOWN
        )
        
        # Check Risk-Safety Service
        try:
            if self.http_client:
                dependencies["risk_safety_service"] = await check_http_endpoint_health(
                    self.http_client,
                    f"{settings.RISK_SAFETY_SERVICE_URL}/health"
                )
            else:
                dependencies["risk_safety_service"] = DependencyHealth(
                    status=DependencyStatus.NOT_APPLICABLE
                )
        except Exception as e:
            dependencies["risk_safety_service"] = DependencyHealth(
                status=DependencyStatus.DOWN,
                error=str(e)
            )
        
        # Service-specific metrics
        metrics = {
            "models_loaded": self._models_loaded,
            "signals_generated_total": self._signals_generated,
            "ensemble_enabled": self.ensemble_manager is not None,
            "meta_strategy_enabled": settings.META_STRATEGY_ENABLED,
            "rl_sizing_enabled": settings.RL_SIZING_ENABLED,
            "running": self._running
        }
        
        # Build standardized health response
        try:
            health = ServiceHealth.create(
                service_name="ai-engine-service",
                version=settings.VERSION,
                start_time=self._start_time,
                dependencies=dependencies,
                metrics=metrics
            )
            return health.to_dict()
        except Exception as e:
            # Fallback to simple response
            logger.error(f"[AI-ENGINE] Health check failed: {e}", exc_info=True)
            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            return {
                "service": "ai-engine-service",
                "status": "DEGRADED",
                "version": settings.VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": round(uptime, 2),
                "error": str(e),
                "running": self._running
            }
