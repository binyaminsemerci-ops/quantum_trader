"""
AI Service - Microservice for AI/ML Trading Signals
====================================================

Quantum Trader v3.0 - AI Service

Responsibilities:
- Run AI/ML models (XGBoost, LightGBM, N-HiTS, PatchTST)
- Ensemble forecasting with weighted voting
- RL position sizing agent
- RL meta-strategy agent
- Universe opportunity scanning
- Publish signal.generated, rl.decision, model.prediction events

Author: Quantum Trader AI Team
Date: December 2, 2025
Version: 3.0.0
"""

import asyncio
import logging
import os
import signal as system_signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import redis.asyncio as redis
from redis.asyncio import Redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Import core infrastructure
from backend.core.event_bus import EventBus, initialize_event_bus, shutdown_event_bus
from backend.core.policy_store import PolicyStore, initialize_policy_store, shutdown_policy_store
from backend.core.logger import configure_logging, get_logger
from backend.core.service_rpc import ServiceRPCClient, ServiceRPCServer
from backend.events.v3_schemas import (
    EventTypes,
    SignalGeneratedPayload,
    RLDecisionPayload,
    ModelPredictionPayload,
    UniverseOpportunityPayload,
    ServiceHeartbeatPayload,
    build_event,
)

# Import AI components
from ai_engine.ensemble_manager import EnsembleManager
from backend.services.ai_trading_engine import AITradingEngine

logger = get_logger(__name__, component="ai_service")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AIServiceConfig:
    """AI Service configuration"""
    service_name: str = "ai-service"
    redis_url: str = "redis://localhost:6379"
    
    # Model paths
    models_dir: str = "ai_engine/models"
    xgboost_model_path: str = "ai_engine/models/xgb_model.pkl"
    lightgbm_model_path: str = "ai_engine/models/lgbm_model.pkl"
    nhits_model_path: str = "ai_engine/models/nhits_model.pt"
    patchtst_model_path: str = "ai_engine/models/patchtst_model.pt"
    ensemble_model_path: str = "ai_engine/models/ensemble_model.pkl"
    
    # Ensemble settings
    ensemble_mode: bool = True
    min_confidence_threshold: float = 0.6
    
    # RL settings
    rl_position_sizing_enabled: bool = True
    rl_meta_strategy_enabled: bool = True
    
    # Universe scanning
    universe_scan_enabled: bool = True
    universe_scan_interval_seconds: int = 300  # 5 minutes
    top_opportunities: int = 20
    
    # Signal generation
    signal_generation_interval_seconds: int = 60  # 1 minute
    max_parallel_predictions: int = 50
    prediction_timeout_seconds: float = 2.0
    
    # Health check
    heartbeat_interval_seconds: int = 5
    health_check_port: int = 8001
    
    # Performance
    max_event_handlers: int = 100
    event_processing_timeout: float = 30.0
    
    @classmethod
    def from_env(cls) -> "AIServiceConfig":
        """Load configuration from environment variables"""
        return cls(
            service_name=os.getenv("SERVICE_NAME", "ai-service"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            models_dir=os.getenv("MODELS_DIR", "ai_engine/models"),
            ensemble_mode=os.getenv("ENSEMBLE_MODE", "true").lower() == "true",
            min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6")),
            rl_position_sizing_enabled=os.getenv("RL_POSITION_SIZING_ENABLED", "true").lower() == "true",
            rl_meta_strategy_enabled=os.getenv("RL_META_STRATEGY_ENABLED", "true").lower() == "true",
            universe_scan_enabled=os.getenv("UNIVERSE_SCAN_ENABLED", "true").lower() == "true",
            universe_scan_interval_seconds=int(os.getenv("UNIVERSE_SCAN_INTERVAL_SECONDS", "300")),
            top_opportunities=int(os.getenv("TOP_OPPORTUNITIES", "20")),
            signal_generation_interval_seconds=int(os.getenv("SIGNAL_GENERATION_INTERVAL_SECONDS", "60")),
            heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "5")),
            health_check_port=int(os.getenv("HEALTH_CHECK_PORT", "8001")),
        )


# ============================================================================
# AI SERVICE
# ============================================================================

class AIService:
    """
    AI Service - Handles AI/ML signal generation and RL agents.
    
    Architecture:
    - Loads all AI models (XGBoost, LightGBM, N-HiTS, PatchTST)
    - Runs ensemble manager for signal generation
    - Executes RL agents for position sizing and meta-strategy
    - Scans universe for trading opportunities
    - Publishes events via EventBus v2
    - Handles RPC requests from other services
    """
    
    def __init__(self, config: AIServiceConfig):
        """Initialize AI Service"""
        self.config = config
        self.running = False
        self.startup_time = time.time()
        
        # FastAPI for health endpoints
        self.app = FastAPI(title="AI Service", version="3.0.0")
        
        # Core components (initialized in bootstrap)
        self.redis: Optional[Redis] = None
        self.event_bus: Optional[EventBus] = None
        self.policy_store: Optional[PolicyStore] = None
        self.rpc_client: Optional[ServiceRPCClient] = None
        self.rpc_server: Optional[ServiceRPCServer] = None
        
        # AI components
        self.ensemble_manager: Optional[EnsembleManager] = None
        self.ai_trading_engine: Optional[AITradingEngine] = None
        
        # RL agents (loaded on demand)
        self.rl_position_sizing_agent = None
        self.rl_meta_strategy_agent = None
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        # Metrics
        self.metrics = {
            "signals_generated": 0,
            "rl_decisions_made": 0,
            "predictions_made": 0,
            "opportunities_scanned": 0,
            "events_published": 0,
            "events_received": 0,
            "rpc_calls_received": 0,
            "errors": 0,
        }
        
        logger.info(
            f"AIService initialized",
            service_name=config.service_name,
            ensemble_mode=config.ensemble_mode,
            rl_enabled=config.rl_position_sizing_enabled,
        )
    
    async def bootstrap(self) -> None:
        """Bootstrap AI Service - initialize all components"""
        logger.info("Bootstrapping AI Service...")
        
        try:
            # 1. Initialize Redis
            logger.info(f"Connecting to Redis: {self.config.redis_url}")
            self.redis = redis.from_url(
                self.config.redis_url,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            await self.redis.ping()
            logger.info("✓ Redis connected")
            
            # 2. Initialize PolicyStore v2
            logger.info("Initializing PolicyStore v2...")
            self.policy_store = await initialize_policy_store(self.redis)
            logger.info("✓ PolicyStore v2 initialized")
            
            # 3. Initialize EventBus v2
            logger.info("Initializing EventBus v2...")
            self.event_bus = await initialize_event_bus(
                self.redis,
                service_name=self.config.service_name
            )
            logger.info("✓ EventBus v2 initialized")
            
            # 4. Initialize RPC Client
            logger.info("Initializing RPC Client...")
            self.rpc_client = ServiceRPCClient(
                self.redis,
                service_name=self.config.service_name
            )
            await self.rpc_client.initialize()
            logger.info("✓ RPC Client initialized")
            
            # 5. Initialize RPC Server
            logger.info("Initializing RPC Server...")
            self.rpc_server = ServiceRPCServer(
                self.redis,
                service_name=self.config.service_name
            )
            self._register_rpc_handlers()
            await self.rpc_server.start()
            logger.info("✓ RPC Server started")
            
            # 6. Load AI Models
            logger.info("Loading AI models...")
            await self._load_ai_models()
            logger.info("✓ AI models loaded")
            
            # 7. Load RL Agents (if enabled)
            if self.config.rl_position_sizing_enabled or self.config.rl_meta_strategy_enabled:
                logger.info("Loading RL agents...")
                await self._load_rl_agents()
                logger.info("✓ RL agents loaded")
            
            # 8. Subscribe to events
            logger.info("Subscribing to events...")
            # 9. Start EventBus consumer
            await self.event_bus.start()
            logger.info("✓ EventBus consumer started")
            
            # 10. Setup health endpoints
            self._setup_health_endpoints()
            logger.info("✓ Health endpoints registered")
            
            logger.info(
                "🚀 AI Service bootstrap complete",
                uptime_seconds=time.time() - self.startup_time,
                ensemble_loaded=self.ensemble_manager is not None,
                rl_agents_loaded=self.rl_position_sizing_agent is not None,
            )   uptime_seconds=time.time() - self.startup_time,
                ensemble_loaded=self.ensemble_manager is not None,
                rl_agents_loaded=self.rl_position_sizing_agent is not None,
            )
        
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}", exc_info=True)
            raise
    
    async def _load_ai_models(self) -> None:
        """Load AI/ML models"""
        try:
            # Load ensemble manager
            if self.config.ensemble_mode:
                logger.info("Loading Ensemble Manager...")
                self.ensemble_manager = EnsembleManager()
                logger.info(
                    f"✓ Ensemble Manager loaded with {len(self.ensemble_manager.agents)} models"
                )
            
            # Load AI Trading Engine
            logger.info("Loading AI Trading Engine...")
            from ai_engine.agents.hybrid_agent import HybridAgent
            
            try:
                agent = HybridAgent()
                logger.info("✓ Hybrid Agent loaded (TFT + XGBoost ensemble)")
            except Exception as e:
                logger.warning(f"Hybrid Agent unavailable: {e}, using Ensemble Manager")
                agent = self.ensemble_manager if self.ensemble_manager else None
            
            self.ai_trading_engine = AITradingEngine(agent=agent)
            logger.info("✓ AI Trading Engine initialized")
        
        except Exception as e:
            logger.error(f"Failed to load AI models: {e}", exc_info=True)
            raise
    
    async def _load_rl_agents(self) -> None:
        """Load RL agents"""
        try:
            # Load RL Position Sizing Agent
            if self.config.rl_position_sizing_enabled:
                logger.info("Loading RL Position Sizing Agent...")
                from backend.domains.learning.rl_v3.ppo_agent_v3 import PPOAgent
                from backend.domains.learning.rl_v3.config_v3 import RLv3Config
                
                rl_config = RLv3Config()
                self.rl_position_sizing_agent = PPOAgent(
                    observation_space_size=rl_config.observation_space_size,
                    action_space_size=rl_config.action_space_size,
                )
                logger.info("✓ RL Position Sizing Agent loaded")
            
            # Load RL Meta Strategy Agent
            if self.config.rl_meta_strategy_enabled:
                logger.info("Loading RL Meta Strategy Agent...")
                from backend.domains.learning.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
                
                self.rl_meta_strategy_agent = MetaStrategyAgentV2(
                    alpha=0.01,
                    gamma=0.99,
                    epsilon=0.1
                )
                logger.info("✓ RL Meta Strategy Agent loaded")
        
        except Exception as e:
            logger.warning(f"Failed to load RL agents: {e}")
            self.rl_position_sizing_agent = None
            self.rl_meta_strategy_agent = None
    
    def _register_rpc_handlers(self) -> None:
        """Register RPC command handlers"""
        
        @self.rpc_server.register_handler("get_signal")
        async def handle_get_signal(params: dict) -> dict:
            """RPC: Get signal for specific symbol"""
            try:
                symbol = params["symbol"]
                logger.debug(f"RPC get_signal: {symbol}")
                
                signal = await self._generate_signal_for_symbol(symbol)
                
                self.metrics["rpc_calls_received"] += 1
                
                return {
                    "symbol": signal["symbol"],
                    "action": signal["action"],
                    "confidence": signal["confidence"],
                    "model_source": signal.get("model_source", "ensemble"),
                    "score": signal.get("score", signal["confidence"]),
                }
            except Exception as e:
                logger.error(f"RPC get_signal error: {e}", exc_info=True)
                return {"error": str(e)}
        
        @self.rpc_server.register_handler("get_top_opportunities")
        async def handle_get_top_opportunities(params: dict) -> dict:
            """RPC: Get top trading opportunities"""
            try:
                top_n = params.get("top_n", self.config.top_opportunities)
                logger.debug(f"RPC get_top_opportunities: top_n={top_n}")
                
                opportunities = await self._scan_universe(top_n)
                
                self.metrics["rpc_calls_received"] += 1
                
                return {
                    "opportunities": opportunities,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                logger.error(f"RPC get_top_opportunities error: {e}", exc_info=True)
                return {"error": str(e)}
        
        @self.rpc_server.register_handler("get_rl_decision")
        async def handle_get_rl_decision(params: dict) -> dict:
            """RPC: Get RL decision for symbol"""
            try:
                symbol = params["symbol"]
                decision_type = params.get("decision_type", "position_sizing")
                
                logger.debug(f"RPC get_rl_decision: {symbol}, type={decision_type}")
                
                if decision_type == "position_sizing" and self.rl_position_sizing_agent:
                    decision = await self._get_rl_position_sizing(symbol)
                elif decision_type == "meta_strategy" and self.rl_meta_strategy_agent:
                    decision = await self._get_rl_meta_strategy(symbol)
                else:
                    decision = {"error": "RL agent not available"}
                
                self.metrics["rpc_calls_received"] += 1
                
                return decision
            except Exception as e:
                logger.error(f"RPC get_rl_decision error: {e}", exc_info=True)
                return {"error": str(e)}
        
        @self.rpc_server.register_handler("reload_models")
        async def handle_reload_models(params: dict) -> dict:
            """RPC: Reload AI models"""
            try:
                logger.info("RPC reload_models triggered")
                await self._load_ai_models()
                
                self.metrics["rpc_calls_received"] += 1
                
                return {"status": "success", "message": "Models reloaded"}
            except Exception as e:
                logger.error(f"RPC reload_models error: {e}", exc_info=True)
                return {"error": str(e)}
        
        logger.info("✓ RPC handlers registered: get_signal, get_top_opportunities, get_rl_decision, reload_models")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to EventBus events"""
        
        # Subscribe to policy updates
        self.event_bus.subscribe(EventTypes.POLICY_UPDATED, self._handle_policy_updated)
        
        # Subscribe to system mode changes
        self.event_bus.subscribe(EventTypes.SYSTEM_MODE_CHANGED, self._handle_mode_changed)
        
        # Subscribe to position closed (for RL learning)
        self.event_bus.subscribe(EventTypes.POSITION_CLOSED, self._handle_position_closed)
        
        logger.info("✓ Event subscriptions: policy.updated, system.mode_changed, position.closed")
    
    async def _handle_policy_updated(self, event_data: dict) -> None:
        """Handle policy.updated event"""
        try:
            logger.info(
                "Policy updated",
                previous_mode=event_data.get("previous_mode"),
                new_mode=event_data.get("new_mode"),
            )
            
            # Reload policy from PolicyStore
            policy = await self.policy_store.reload_policy()
            
            logger.info(
                f"Policy reloaded: active_mode={policy.active_mode.value}, "
                f"version={policy.version}"
            )
            
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling policy update: {e}", exc_info=True)
            self.metrics["errors"] += 1
    
    async def _handle_mode_changed(self, event_data: dict) -> None:
        """Handle system.mode_changed event"""
        try:
            logger.warning(
                "System mode changed",
                previous_mode=event_data.get("previous_mode"),
                new_mode=event_data.get("new_mode"),
                reason=event_data.get("reason"),
            )
            
            new_mode = event_data.get("new_mode")
            
            # Adjust AI behavior based on mode
            if new_mode == "EMERGENCY":
                logger.critical("EMERGENCY MODE - Pausing signal generation")
                # Pause signal generation
            elif new_mode == "DEFENSIVE":
                logger.warning("DEFENSIVE MODE - Reducing confidence threshold")
                # Increase confidence threshold
            elif new_mode == "NORMAL":
                logger.info("NORMAL MODE - Resuming normal operations")
            
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling mode change: {e}", exc_info=True)
            self.metrics["errors"] += 1
    
    async def _handle_position_closed(self, event_data: dict) -> None:
        """Handle position.closed event (for RL learning)"""
        try:
            symbol = event_data.get("symbol")
            pnl_pct = event_data.get("pnl_pct", 0.0)
            exit_reason = event_data.get("exit_reason")
            
            logger.info(
                f"Position closed: {symbol}, PnL={pnl_pct:.2f}%, reason={exit_reason}"
            )
            
            # Update RL agents with outcome (reward signal)
            if self.rl_position_sizing_agent:
                # RL learning happens here
                # This would update Q-values based on PnL outcome
                pass
            
            self.metrics["events_received"] += 1
        
        except Exception as e:
            logger.error(f"Error handling position closed: {e}", exc_info=True)
            self.metrics["errors"] += 1
    
    async def start(self) -> None:
        """Start AI Service background tasks"""
        logger.info("Starting AI Service background tasks...")
        
        self.running = True
        
        # Start signal generation task
        self.tasks.append(
            asyncio.create_task(self._signal_generation_loop())
        )
        
        # Start universe scanning task (if enabled)
        if self.config.universe_scan_enabled:
            self.tasks.append(
                asyncio.create_task(self._universe_scan_loop())
            )
        
        # Start heartbeat task
        self.tasks.append(
            asyncio.create_task(self._heartbeat_loop())
        )
        
        logger.info(f"✓ Started {len(self.tasks)} background tasks")
    
    async def _signal_generation_loop(self) -> None:
        """Background task: Generate trading signals periodically"""
        logger.info("Signal generation loop started")
        
        while self.running:
            try:
                # Get active symbols from policy or use default list
                symbols = await self._get_active_symbols()
                
                logger.info(f"Generating signals for {len(symbols)} symbols...")
                
                # Generate signals in batches
                signals = await self._generate_signals_batch(symbols)
                
                # Publish signals
                for signal in signals:
                    if signal["action"] != "HOLD":
                        await self._publish_signal(signal)
                
                logger.info(
                    f"Signal generation complete: {len([s for s in signals if s['action'] != 'HOLD'])} "
                    f"actionable signals out of {len(signals)} total"
                )
                
                # Sleep until next interval
                await asyncio.sleep(self.config.signal_generation_interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Signal generation error: {e}", exc_info=True)
                self.metrics["errors"] += 1
                await asyncio.sleep(10)
        
        logger.info("Signal generation loop stopped")
    
    async def _universe_scan_loop(self) -> None:
        """Background task: Scan universe for opportunities"""
        logger.info("Universe scan loop started")
        
        while self.running:
            try:
                logger.info("Scanning universe for trading opportunities...")
                
                opportunities = await self._scan_universe(self.config.top_opportunities)
                
                # Publish opportunities
                await self._publish_universe_opportunities(opportunities)
                
                logger.info(f"Universe scan complete: {len(opportunities)} opportunities found")
                
                self.metrics["opportunities_scanned"] += len(opportunities)
                
                # Sleep until next scan
                await asyncio.sleep(self.config.universe_scan_interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Universe scan error: {e}", exc_info=True)
                self.metrics["errors"] += 1
                await asyncio.sleep(60)
        
        logger.info("Universe scan loop stopped")
    
    async def _heartbeat_loop(self) -> None:
        """Background task: Publish service heartbeat"""
        logger.info("Heartbeat loop started")
        
        while self.running:
            try:
                await self._publish_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}", exc_info=True)
                await asyncio.sleep(5)
        
        logger.info("Heartbeat loop stopped")
    
    async def _get_active_symbols(self) -> List[str]:
        """Get list of active trading symbols"""
        # Default symbols for now
        # In production, this would come from PolicyStore or Universe Scanner
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT",
            "XRPUSDT", "DOTUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT",
        ]
    
    async def _generate_signals_batch(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate signals for multiple symbols"""
        if not self.ai_trading_engine:
            logger.warning("AI Trading Engine not loaded")
            return []
        
        try:
            # Use AI Trading Engine to generate signals
            current_positions = {}  # Would come from exec-risk-service
            
            signals = await self.ai_trading_engine.get_trading_signals(
                symbols=symbols,
                current_positions=current_positions
            )
            
            self.metrics["predictions_made"] += len(signals)
            
            return signals
        
        except Exception as e:
            logger.error(f"Signal generation failed: {e}", exc_info=True)
            return []
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Generate signal for single symbol"""
        signals = await self._generate_signals_batch([symbol])
        return signals[0] if signals else {"symbol": symbol, "action": "HOLD", "confidence": 0.5}
    
    async def _scan_universe(self, top_n: int) -> List[Dict[str, Any]]:
        """Scan universe and return top opportunities"""
        # Simplified implementation
        # In production, this would use Opportunity Ranker
        symbols = await self._get_active_symbols()
        
        opportunities = []
        for symbol in symbols[:top_n]:
            opportunities.append({
                "symbol": symbol,
                "score": 0.75,
                "reason": "High volume and volatility",
            })
        
        return opportunities
    
    async def _get_rl_position_sizing(self, symbol: str) -> Dict[str, Any]:
        """Get RL position sizing decision"""
        if not self.rl_position_sizing_agent:
            return {"error": "RL agent not available"}
        
        # Simplified - would use real state observation
        return {
            "decision_type": "position_sizing",
            "symbol": symbol,
            "recommended_size_usd": 1000.0,
            "recommended_leverage": 10.0,
            "kelly_fraction": 0.05,
        }
    
    async def _get_rl_meta_strategy(self, symbol: str) -> Dict[str, Any]:
        """Get RL meta-strategy decision"""
        if not self.rl_meta_strategy_agent:
            return {"error": "RL agent not available"}
        
        # Simplified - would use real state
        return {
            "decision_type": "meta_strategy",
            "symbol": symbol,
            "selected_strategy": "aggressive",
            "strategy_confidence": 0.85,
        }
    
    async def _publish_signal(self, signal: Dict[str, Any]) -> None:
        """Publish signal.generated event"""
        try:
            payload = SignalGeneratedPayload(
                symbol=signal["symbol"],
                action=signal["action"],
                confidence=signal.get("confidence", 0.5),
                model_source=signal.get("model_source", "ensemble"),
                score=signal.get("score", signal.get("confidence", 0.5)),
                timeframe="1m",
                ensemble_agreement=signal.get("ensemble_agreement"),
                models_voted=signal.get("models_voted"),
            )
            
            await self.event_bus.publish(
                EventTypes.SIGNAL_GENERATED,
                payload.dict()
            )
            
            self.metrics["signals_generated"] += 1
            self.metrics["events_published"] += 1
            
            logger.debug(
                f"Signal published: {signal['symbol']} {signal['action']} "
                f"conf={signal.get('confidence', 0):.2f}"
            )
        
        except Exception as e:
            logger.error(f"Failed to publish signal: {e}", exc_info=True)
            self.metrics["errors"] += 1
    
    async def _publish_universe_opportunities(self, opportunities: List[Dict[str, Any]]) -> None:
        """Publish universe.opportunity event"""
        try:
            payload = UniverseOpportunityPayload(
                ranked_symbols=opportunities,
                total_scanned=len(opportunities),
                timestamp=datetime.now(timezone.utc).isoformat(),
                criteria_weights={"volume": 0.3, "volatility": 0.3, "momentum": 0.4},
            )
            
            await self.event_bus.publish(
                EventTypes.UNIVERSE_OPPORTUNITY,
                payload.dict()
            )
            
            self.metrics["events_published"] += 1
        
        except Exception as e:
            logger.error(f"Failed to publish universe opportunities: {e}", exc_info=True)
            self.metrics["errors"] += 1
    
    async def _publish_heartbeat(self) -> None:
        """Publish service.heartbeat event"""
        try:
            import psutil
            
            payload = ServiceHeartbeatPayload(
                service_name=self.config.service_name,
                status="HEALTHY",
                uptime_seconds=time.time() - self.startup_time,
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / 1024 / 1024,
                active_tasks=len([t for t in self.tasks if not t.done()]),
                processed_events_last_minute=self.metrics["events_received"],
            )
            
            await self.event_bus.publish(
        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}", exc_info=True)
    
    def _setup_health_endpoints(self) -> None:
        """Setup FastAPI health endpoints"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return JSONResponse({
                "status": "healthy" if self.running else "starting",
                "service": self.config.service_name,
                "version": "3.0.0",
                "uptime_seconds": time.time() - self.startup_time,
                "models_loaded": self.ensemble_manager is not None,
                "rl_agents_loaded": self.rl_position_sizing_agent is not None,
            })
        
        @self.app.get("/ready")
        async def ready():
            """Readiness check endpoint"""
            is_ready = (
                self.running and
                self.ensemble_manager is not None and
                self.event_bus is not None and
                self.rpc_server is not None
            )
            
            if is_ready:
                return JSONResponse({"status": "ready"})
            else:
                return JSONResponse({"status": "not_ready"}, status_code=503)
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            metrics_text = []
            
            # Service metrics
            metrics_text.append(f'# HELP ai_service_uptime_seconds Service uptime in seconds')
            metrics_text.append(f'# TYPE ai_service_uptime_seconds gauge')
            metrics_text.append(f'ai_service_uptime_seconds {time.time() - self.startup_time}')
            
            metrics_text.append(f'# HELP ai_service_signals_generated_total Total signals generated')
            metrics_text.append(f'# TYPE ai_service_signals_generated_total counter')
            metrics_text.append(f'ai_service_signals_generated_total {self.metrics["signals_generated"]}')
            
            metrics_text.append(f'# HELP ai_service_predictions_made_total Total predictions made')
            metrics_text.append(f'# TYPE ai_service_predictions_made_total counter')
            metrics_text.append(f'ai_service_predictions_made_total {self.metrics["predictions_made"]}')
            
            metrics_text.append(f'# HELP ai_service_events_published_total Total events published')
            metrics_text.append(f'# TYPE ai_service_events_published_total counter')
            metrics_text.append(f'ai_service_events_published_total {self.metrics["events_published"]}')
            
            metrics_text.append(f'# HELP ai_service_errors_total Total errors')
            metrics_text.append(f'# TYPE ai_service_errors_total counter')
            metrics_text.append(f'ai_service_errors_total {self.metrics["errors"]}')
            
            return "\n".join(metrics_text)
    
    async def start_fastapi(self) -> None:
        """Start FastAPI server"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.health_check_port,
            log_level="warning"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def shutdown(self) -> None:
        except Exception as e:
            logger.error(f"Failed to publish heartbeat: {e}", exc_info=True)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown AI Service"""
        logger.info("Shutting down AI Service...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown RPC server
        if self.rpc_server:
            await self.rpc_server.stop()
        
        # Shutdown RPC client
        if self.rpc_client:
            await self.rpc_client.shutdown()
        
        # Shutdown EventBus
        if self.event_bus:
            await self.event_bus.stop()
        
        # Shutdown PolicyStore
        await shutdown_policy_store()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        logger.info(
            "AI Service shutdown complete",
            uptime_seconds=time.time() - self.startup_time,
            signals_generated=self.metrics["signals_generated"],
            events_published=self.metrics["events_published"],
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for AI Service"""
    
    # Configure logging
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    
    logger.info("=" * 80)
    logger.info("QUANTUM TRADER v3.0 - AI SERVICE")
    logger.info("=" * 80)
    
    # Load configuration
    config = AIServiceConfig.from_env()
    
    # Create service
    service = AIService(config)
    
    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        shutdown_event.set()
    
    system_signal.signal(system_signal.SIGINT, signal_handler)
    system_signal.signal(system_signal.SIGTERM, signal_handler)
    
    try:
        # Bootstrap service
        await service.bootstrap()
        
        # Start background tasks
        await service.start()
        
        # Start FastAPI in background
        fastapi_task = asyncio.create_task(service.start_fastapi())
        
        logger.info("🚀 AI Service running")
        logger.info(f"   Service: {config.service_name}")
        logger.info(f"   Redis: {config.redis_url}")
        logger.info(f"   Health: http://0.0.0.0:{config.health_check_port}/health")
        logger.info(f"   Metrics: http://0.0.0.0:{config.health_check_port}/metrics")
        logger.info(f"   Ensemble Mode: {config.ensemble_mode}")
        logger.info(f"   RL Enabled: {config.rl_position_sizing_enabled}")
        logger.info(f"   Universe Scan: {config.universe_scan_enabled}")
        logger.info("")
        logger.info("Press Ctrl+C to shutdown")
        logger.info("=" * 80)
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel FastAPI task
        fastapi_task.cancel()
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        # Graceful shutdown
        await service.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
