"""
AI Engine Service - Core Service Logic

Orchestrates AI inference pipeline:
1. Market data → Ensemble models
2. Ensemble → Signal + Confidence
3. Signal → Meta-Strategy Selector → Strategy
4. Signal + Strategy → RL Position Sizing → Size + Leverage
5. Publish ai.decision.made event → execution-service
"""
import time
import asyncio
import logging
import httpx
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from collections import deque, defaultdict
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# P1-B: JSON logging with correlation_id
from shared.logging_config import setup_json_logging, set_correlation_id, get_correlation_id

# Standard library imports
import os  # Added for environment variables

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
from backend.microservices.ai_engine.services.model_supervisor_governance import ModelSupervisorGovernance
from backend.microservices.ai_engine.services.adaptive_retrainer import AdaptiveRetrainer
from backend.microservices.ai_engine.services.model_validation_layer import ModelValidationLayer

# 🔥 PHASE 1 FUTURES INTELLIGENCE MODULES 🔥
from backend.services.ai.drift_detection_manager import DriftDetectionManager
from backend.services.ai.reinforcement_signal_manager import ReinforcementSignalManager
from backend.services.risk.funding_rate_filter import FundingRateFilter
from backend.services.continuous_learning import ContinuousLearningManager, ShadowEvaluator, RetrainingConfig, ModelVersion, ModelStage
from backend.services.ai.volatility_structure_engine import VolatilityStructureEngine
from backend.services.ai.orderbook_imbalance_module import OrderbookImbalanceModule
from backend.services.ai.risk_mode_predictor import RiskModePredictor
from backend.services.ai.strategy_selector import StrategySelector, TradingStrategy
from backend.services.ai.system_health_monitor import SystemHealthMonitor
from backend.services.ai.performance_benchmarker import PerformanceBenchmarker
from backend.services.ai.adaptive_threshold_manager import AdaptiveThresholdManager
from backend.services.binance_market_data import BinanceMarketDataFetcher

# 🔥 PHASE 2.2: CEO Brain Orchestration
from backend.services.orchestration.orchestrator import get_orchestrator

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
        
        # 🔥 RATE LIMITING STATE
        self.signal_times = deque(maxlen=20)  # Last 20 signal timestamps
        self.last_signal_by_symbol = defaultdict(float)  # symbol -> timestamp
        self.max_signals_per_min = int(os.getenv("MAX_SIGNALS_PER_MINUTE", "6"))
        self.symbol_cooldown_sec = int(os.getenv("SYMBOL_COOLDOWN_SECONDS", "120"))
        self.meta_strategy_selector = None
        self.rl_sizing_agent = None
        self.regime_detector = None
        self.memory_manager = None
        self.model_supervisor = None
        self.supervisor_governance = None  # Phase 4D+4E: Model Supervisor & Predictive Governance
        self.adaptive_retrainer = None  # Phase 4F: Adaptive Retraining Pipeline
        self.model_validator = None  # Phase 4G: Model Validation Layer
        
        # 🔥 PHASE 1: Futures Intelligence Modules
        self.cross_exchange_aggregator = None  # Cross-exchange volatility, divergence, lead/lag
        self.funding_rate_filter = None        # Funding pressure, crowd bias, squeeze detection
        self.drift_detector = None             # Model drift detection & retrain triggers
        self.rl_signal_manager = None          # PnL feedback loop & confidence calibration
        
        # 🔥 PHASE 2C: Continuous Learning Manager
        self.continuous_learning_manager = None  # Auto-retrain models based on real trade data
        self._clm_trade_buffer: List[Dict] = []   # Buffer for CLM trade outcomes
        
        # 🔥 PHASE 2D: Volatility Structure Engine
        self.volatility_structure_engine = None  # ATR-trend, cross-TF volatility analysis
        
        # 🔥 PHASE 2B: Orderbook Imbalance Module
        self.orderbook_imbalance = None  # Real-time orderbook depth analysis
        self._binance_fetcher = None  # Binance market data fetcher
        self._active_symbols: List[str] = []  # Symbols to track orderbook for
        
        # 🔥 PHASE 3A: Risk Mode Predictor
        self.risk_mode_predictor = None  # ML-based dynamic risk management
        
        # 🔥 PHASE 3B: Strategy Selector
        self.strategy_selector = None  # Intelligent strategy selection
        
        # 🔥 PHASE 3C: System Health Monitor
        self.health_monitor = None  # Real-time system health monitoring
        self.performance_benchmarker = None  # Performance tracking and benchmarking
        self.adaptive_threshold_manager = None  # Adaptive threshold learning
        
        # 🔥 PHASE 3C ADAPTERS: Exit Brain Integration & Confidence Calibration
        self.exit_brain_performance_adapter = None  # Performance-adaptive TP/SL
        self.confidence_calibrator = None  # Confidence score calibration
        
        # 🔥 PHASE 2.2: CEO Brain Orchestrator
        self.orchestrator = None  # CEO Brain + Strategy Brain + Risk Brain coordination
        
        # State tracking
        self._running = False
        self._active_symbols: List[str] = []  # Symbols to track for orderbook
        self._orderbook_client = None  # Binance client for orderbook fetching
        self._event_loop_task: Optional[asyncio.Task] = None
        self._regime_update_task: Optional[asyncio.Task] = None
        self._signals_generated = 0
        self._models_loaded = 0
        self._start_time = datetime.now(timezone.utc)
        
        # Governance tracking (Phase 4D+4E)
        self._governance_predictions: Dict[str, Dict[str, np.ndarray]] = {}  # {symbol: {model: predictions}}
        self._governance_actuals: Dict[str, List[float]] = {}  # {symbol: [actual_prices]}
        self._governance_pnl: Dict[str, Dict[str, float]] = {}  # {symbol: {model: pnl}}
        
        # Price history for indicator calculations (symbol -> list of prices)
        self._price_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[float]] = {}
        self._history_max_len = 120  # Keep 2 minutes at 1 tick/sec
        
        logger.info("[AI-ENGINE] Service initialized")
    
    async def start(self):
        """Start the service."""
        if self._running:
            logger.warning("[AI-ENGINE] Service already running")
            return
        
        # P1-B: Setup JSON logging with correlation_id
        setup_json_logging('ai_engine')
        logger.info("[AI-ENGINE] JSON logging initialized")
        
        try:
            # Initialize Redis client
            import redis.asyncio as redis
            logger.info(f"[AI-ENGINE] Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}...")
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=False
            )
            await self.redis_client.ping()
            logger.info("[AI-ENGINE] ✅ Redis connected")
            
            # Initialize EventBus
            logger.info("[AI-ENGINE] Initializing EventBus...")
            self.event_bus = EventBus(redis_client=self.redis_client, service_name="ai-engine")
            
            # Subscribe to events
            self.event_bus.subscribe("market.tick", self._handle_market_tick)
            self.event_bus.subscribe("exchange.raw", self._handle_exchange_raw)  # 🔥 FIX: Use live exchange data
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
            
            # Start EventBus consumer (CRITICAL - starts reading from Redis Streams)
            await self.event_bus.start()
            logger.info("[AI-ENGINE] ✅ EventBus consumer started")
            
            # 🔥 PHASE 2B: Start orderbook data feed
            if self.orderbook_imbalance:
                asyncio.create_task(self._fetch_orderbook_loop())
                logger.info("[PHASE 2B] 📖 Orderbook data feed started")
            
            # 🔥 PHASE 3C: Start health monitoring loop
            if self.health_monitor:
                asyncio.create_task(self.health_monitor.start_monitoring())
                logger.info("[PHASE 3C] 🏥 Health monitoring loop started")
            
            # 🔥 PHASE 3C-2: Start performance benchmarking loop
            if self.performance_benchmarker:
                asyncio.create_task(self.performance_benchmarker.start_benchmarking())
                logger.info("[PHASE 3C-2] 📊 Performance benchmarking loop started")
            
            # 🔥 PHASE 3C-3: Start adaptive learning loop
            if self.adaptive_threshold_manager:
                asyncio.create_task(self.adaptive_threshold_manager.start_learning())
                logger.info("[PHASE 3C-3] 🧠 Adaptive learning loop started")
            
            if settings.REGIME_DETECTION_ENABLED:
                self._regime_update_task = asyncio.create_task(self._regime_update_loop())
            
            # 🔥 FIX: Set running flag to True
            self._running = True
            
            logger.info("[AI-ENGINE] ✅ Service started successfully (running=True)")
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] ❌ Failed to start service: {e}", exc_info=True)
            raise
    
    async def stop(self):
        """Stop the service."""
        if not self._running:
            return
        
        logger.info("[AI-ENGINE] Stopping service...")
        self._running = False
        
        # Stop EventBus consumer
        if self.event_bus:
            await self.event_bus.stop()
            logger.info("[AI-ENGINE] EventBus consumer stopped")
        
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
                try:
                    from ai_engine.ensemble_manager import EnsembleManager
                    
                    self.ensemble_manager = EnsembleManager(
                        weights=settings.ENSEMBLE_WEIGHTS,
                        min_consensus=settings.MIN_CONSENSUS,
                        enabled_models=settings.ENSEMBLE_MODELS,
                        xgb_model_path=settings.XGB_MODEL_PATH,
                        xgb_scaler_path=settings.XGB_SCALER_PATH
                    )
                    self._models_loaded += len(settings.ENSEMBLE_MODELS)
                    logger.info(f"[AI-ENGINE] ✅ Ensemble loaded ({self._models_loaded} models)")
                except Exception as e:
                    logger.warning(f"[AI-ENGINE] ⚠️ Ensemble loading failed: {e}")
                    self.ensemble_manager = None
            
            # 2. Meta-Strategy Selector
            if settings.META_STRATEGY_ENABLED:
                logger.info("[AI-ENGINE] Loading Meta-Strategy Selector...")
                from backend.services.ai.meta_strategy_selector import MetaStrategySelector
                from pathlib import Path
                
                self.meta_strategy_selector = MetaStrategySelector(
                    epsilon=settings.META_STRATEGY_EPSILON,
                    alpha=settings.META_STRATEGY_ALPHA,
                    state_file=Path(settings.META_STRATEGY_STATE_PATH),  # FIXED: state_path → state_file, added Path()
                )
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Meta-Strategy Selector loaded")
            
            # 3. RL Position Sizing Agent
            if settings.RL_SIZING_ENABLED:
                logger.info("[AI-ENGINE] Loading RL Position Sizing Agent...")
                from backend.services.ai.rl_position_sizing_agent import RLPositionSizingAgent
                
                self.rl_sizing_agent = RLPositionSizingAgent(
                    policy_store=None,  # Will fetch from risk-safety-service
                    state_file=settings.RL_SIZING_STATE_PATH,  # FIXED: state_path → state_file
                    learning_rate=settings.RL_SIZING_ALPHA,  # FIXED: alpha → learning_rate
                    discount_factor=settings.RL_SIZING_DISCOUNT,  # FIXED: discount → discount_factor
                    exploration_rate=settings.RL_SIZING_EPSILON,  # FIXED: epsilon → exploration_rate
                )
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ RL Position Sizing loaded")
            
            # 4. Regime Detector
            if settings.REGIME_DETECTION_ENABLED:
                logger.info("[AI-ENGINE] Loading Regime Detector...")
                from backend.services.ai.regime_detector import RegimeDetector
                
                self.regime_detector = RegimeDetector()
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Regime Detector loaded")
            
            # 5. Memory State Manager
            if settings.MEMORY_STATE_ENABLED:
                logger.info("[AI-ENGINE] Loading Memory State Manager...")
                from backend.services.ai.memory_state_manager import MemoryStateManager
                
                self.memory_manager = MemoryStateManager(
                    checkpoint_path="/app/data/memory_state.json",  # FIXED: removed lookback_hours, added checkpoint_path
                    ewma_alpha=0.3
                )
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Memory State Manager loaded")
            
            # 6. Model Supervisor
            if settings.MODEL_SUPERVISOR_ENABLED:
                logger.info("[AI-ENGINE] Loading Model Supervisor...")
                from backend.services.ai.model_supervisor import ModelSupervisor
                
                self.model_supervisor = ModelSupervisor(
                    data_dir="/app/data",  # FIXED: correct parameters (no bias_threshold/min_samples)
                    analysis_window_days=30,
                    recent_window_days=7
                )
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Model Supervisor loaded")
            
            # 7. Model Supervisor & Predictive Governance (Phase 4D + 4E)
            if self.ensemble_manager and settings.ENSEMBLE_MODELS:
                logger.info("[AI-ENGINE] 🧠 Initializing Model Supervisor & Governance (Phase 4D+4E)...")
                
                # Initialize governance system
                self.supervisor_governance = ModelSupervisorGovernance(
                    drift_threshold=0.05,  # 5% MAPE threshold
                    retrain_interval=3600,  # 1 hour
                    smooth=0.3  # 30% smoothing for weight adjustment
                )
                
                # Register ensemble models for supervision
                # Map friendly names to settings names
                model_mapping = {
                    "PatchTST": "patchtst",
                    "NHiTS": "nhits",
                    "XGBoost": "xgb",
                    "LightGBM": "lgbm"
                }
                
                for display_name, config_name in model_mapping.items():
                    if config_name in settings.ENSEMBLE_MODELS:
                        self.supervisor_governance.register(display_name, None)
                
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Model Supervisor & Governance active")
                logger.info(f"[PHASE 4D+4E] Supervisor + Predictive Governance active")
            
            # 8. Adaptive Retraining Pipeline (Phase 4F)
            if self.ensemble_manager and settings.ENSEMBLE_MODELS:
                logger.info("[AI-ENGINE] 🔄 Initializing Adaptive Retraining Pipeline (Phase 4F)...")
                
                self.adaptive_retrainer = AdaptiveRetrainer(
                    data_api=None,  # Will be replaced with actual data API
                    model_paths={
                        "patchtst": "/app/models/patchtst_adaptive.pth",
                        "nhits": "/app/models/nhits_adaptive.pth"
                    },
                    retrain_interval=14400,  # 4 hours
                    min_data_points=5000,
                    max_epochs=2
                )
                
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Adaptive Retraining Pipeline active")
                logger.info(f"[PHASE 4F] Adaptive Retrainer initialized - Interval: 4h")
            
            # 9. Model Validation Layer (Phase 4G)
            if self.adaptive_retrainer:
                logger.info("[AI-ENGINE] 🔍 Initializing Model Validation Layer (Phase 4G)...")
                
                self.model_validator = ModelValidationLayer(
                    model_paths={
                        "patchtst": "/app/models/patchtst.pth",
                        "nhits": "/app/models/nhits.pth"
                    },
                    val_data_api=None  # Will use same data API as retrainer
                )
                
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Model Validation Layer active")
                logger.info(f"[PHASE 4G] Validator initialized - Criteria: 3% MAPE improvement + better Sharpe")
            
            # 🔥 PHASE 1 MODULES - FUTURES INTELLIGENCE 🔥
            
            # 10. Cross-Exchange Normalizer
            if settings.CROSS_EXCHANGE_ENABLED:
                logger.info("[AI-ENGINE] 🌐 Initializing Cross-Exchange Normalizer (Phase 1)...")
                try:
                    from microservices.ai_engine.cross_exchange_aggregator import CrossExchangeAggregator
                    
                    self.cross_exchange_aggregator = CrossExchangeAggregator(
                        symbols=settings.CROSS_EXCHANGE_SYMBOLS,
                        redis_host=settings.REDIS_HOST,
                        redis_port=settings.REDIS_PORT
                    )
                    # Start aggregation in background
                    asyncio.create_task(self.cross_exchange_aggregator.start())
                    
                    self._models_loaded += 1
                    logger.info("[AI-ENGINE] ✅ Cross-Exchange Normalizer active")
                    logger.info(f"[PHASE 1] Cross-Exchange: {len(settings.CROSS_EXCHANGE_SYMBOLS)} symbols, {len(settings.CROSS_EXCHANGE_EXCHANGES)} exchanges")
                except Exception as e:
                    logger.warning(f"[AI-ENGINE] ⚠️ Cross-Exchange failed to load: {e}")
                    self.cross_exchange_aggregator = None
            
            # 11. Funding Rate Filter
            if settings.FUNDING_RATE_ENABLED:
                logger.info("[AI-ENGINE] 💰 Initializing Funding Rate Filter (Phase 1)...")
                try:
                    self.funding_rate_filter = FundingRateFilter(
                        max_funding_rate=settings.MAX_FUNDING_RATE,
                        warn_funding_rate=settings.WARN_FUNDING_RATE,
                        use_testnet=os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
                    )
                    
                    self._models_loaded += 1
                    logger.info("[AI-ENGINE] ✅ Funding Rate Filter active")
                    logger.info(f"[PHASE 1] Funding: Max={settings.MAX_FUNDING_RATE*100:.2f}%, Warn={settings.WARN_FUNDING_RATE*100:.2f}%")
                except Exception as e:
                    logger.warning(f"[AI-ENGINE] ⚠️ Funding Rate Filter failed: {e}")
                    self.funding_rate_filter = None
            
            # 12. Drift Detection Manager
            if settings.DRIFT_DETECTION_ENABLED:
                logger.info("[AI-ENGINE] 📊 Initializing Drift Detection Manager (Phase 1)...")
                try:
                    self.drift_detector = DriftDetectionManager(
                        psi_moderate_threshold=settings.DRIFT_PSI_THRESHOLD_MODERATE,
                        psi_severe_threshold=settings.DRIFT_PSI_THRESHOLD_SEVERE
                    )
                    
                    self._models_loaded += 1
                    logger.info("[AI-ENGINE] ✅ Drift Detection Manager active")
                    logger.info(f"[PHASE 1] Drift: PSI moderate={settings.DRIFT_PSI_THRESHOLD_MODERATE}, severe={settings.DRIFT_PSI_THRESHOLD_SEVERE}")
                except Exception as e:
                    logger.warning(f"[AI-ENGINE] ⚠️ Drift Detection failed: {e}")
                    self.drift_detector = None
            
            # 13. Reinforcement Signal Manager (PnL Feedback Loop)
            if settings.REINFORCEMENT_SIGNAL_ENABLED:
                logger.info("[AI-ENGINE] 🎯 Initializing Reinforcement Signal Manager (Phase 1)...")
                try:
                    self.rl_signal_manager = ReinforcementSignalManager(
                        learning_rate=settings.RL_SIGNAL_LEARNING_RATE,
                        discount_factor=settings.RL_SIGNAL_DISCOUNT_FACTOR,
                        checkpoint_path=settings.RL_SIGNAL_STATE_PATH
                    )
                    
                    self._models_loaded += 1
                    logger.info("[AI-ENGINE] ✅ Reinforcement Signal Manager active")
                    logger.info(f"[PHASE 1] RL Signal: lr={settings.RL_SIGNAL_LEARNING_RATE}, discount={settings.RL_SIGNAL_DISCOUNT_FACTOR}")
                except Exception as e:
                    logger.warning(f"[AI-ENGINE] ⚠️ RL Signal Manager failed: {e}")
                    self.rl_signal_manager = None
            
            # 14. Continuous Learning Manager (Phase 2C) - Auto-retraining with real trade data
            logger.info("[AI-ENGINE] 📚 Initializing Continuous Learning Manager (Phase 2C)...")
            try:
                # Use InMemoryEventBus for CLM (separate from Redis EventBus)
                from backend.services.eventbus import InMemoryEventBus
                clm_eventbus = InMemoryEventBus()
                
                self.continuous_learning_manager = ContinuousLearningManager(
                    eventbus=clm_eventbus,
                    evaluator=ShadowEvaluator()
                )
                
                # Register models for continuous learning
                for model_name in ["ensemble", "lstm", "xgboost", "lightgbm"]:
                    initial_model = ModelVersion(
                        model_name=model_name,
                        version="1.0",
                        stage=ModelStage.LIVE,
                        model_type=model_name
                    )
                    
                    retrain_config = RetrainingConfig(
                        model_name=model_name,
                        retrain_interval_hours=24.0,
                        min_new_samples=5000,
                        shadow_evaluation_hours=2.0
                    )
                    
                    self.continuous_learning_manager.register_model(model_name, initial_model, retrain_config)
                
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Continuous Learning Manager active")
                logger.info("[PHASE 2C] CLM: Min samples=5000, shadow eval=2h, registered 4 models")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ CLM failed to initialize: {e}")
                self.continuous_learning_manager = None
            
            # 15. Volatility Structure Engine (Phase 2D) - ATR-trend & cross-TF volatility
            logger.info("[AI-ENGINE] 📊 Initializing Volatility Structure Engine (Phase 2D)...")
            try:
                self.volatility_structure_engine = VolatilityStructureEngine(
                    atr_period=14,
                    atr_trend_lookback=5,
                    volatility_expansion_threshold=1.5,
                    volatility_contraction_threshold=0.5,
                    history_size=200
                )
                
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Volatility Structure Engine active")
                logger.info("[PHASE 2D] VSE: ATR trend detection, cross-TF volatility, regime classification")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Volatility Structure Engine failed: {e}")
                self.volatility_structure_engine = None
            
            # 16. Orderbook Imbalance Module (Phase 2B) - Real-time orderflow analysis
            logger.info("[AI-ENGINE] 📖 Initializing Orderbook Imbalance Module (Phase 2B)...")
            try:
                self.orderbook_imbalance = OrderbookImbalanceModule(
                    depth_levels=20,
                    delta_volume_window=100,
                    large_order_threshold_pct=0.01,
                    history_size=50
                )
                
                # Initialize Binance fetcher for orderbook data
                self._binance_fetcher = BinanceMarketDataFetcher()
                
                # 🔥 PHASE 1.1 FIX: Use UniverseManager (top coins by 24h volume from main base + L1 + L2)
                try:
                    from backend.services.universe_manager import get_universe_manager
                    max_symbols = int(os.getenv("QT_MAX_SYMBOLS", "20"))
                    
                    logger.info("[PHASE 1.1] 🌐 Loading universe from UniverseManager (24h volume: main base + L1 + L2)...")
                    universe_mgr = get_universe_manager()
                    await universe_mgr.initialize()
                    
                    self._active_symbols = universe_mgr.get_symbols()[:max_symbols]
                    logger.info(f"[PHASE 1.1] ✅ Loaded {len(self._active_symbols)} symbols by 24h volume")
                    logger.info(f"[PHASE 1.1] Symbols: {', '.join(self._active_symbols[:10])}...")
                except Exception as e:
                    logger.error(f"[PHASE 1.1] ❌ Failed to load UniverseManager: {e}")
                    logger.warning("[PHASE 1.1] Falling back to top 3 by volume...")
                    # Fallback to safe defaults (top 3 by volume)
                    self._active_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
                
                # Start orderbook data feed loop
                asyncio.create_task(self._fetch_orderbook_loop())
                
                self._models_loaded += 1
                logger.info("[AI-ENGINE] ✅ Orderbook Imbalance Module active")
                logger.info(f"[PHASE 2B] OBI: Tracking orderbook for {len(self._active_symbols)} symbols")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Orderbook Imbalance Module failed: {e}")
                self.orderbook_imbalance = None
            
            logger.info(f"[AI-ENGINE] ✅ All AI modules loaded ({self._models_loaded} models active)")
            logger.info("[PHASE 1] 🚀 Futures Intelligence Stack: ACTIVATED")
            if self.continuous_learning_manager:
                logger.info("[PHASE 2C] 🎓 Continuous Learning: ONLINE")
            if self.volatility_structure_engine:
                logger.info("[PHASE 2D] 📈 Volatility Structure Engine: ONLINE")
            if self.orderbook_imbalance:
                logger.info("[PHASE 2B] 📖 Orderbook Imbalance: ONLINE")
            
            # 🔥 PHASE 3A: Risk Mode Predictor - Dynamic risk management
            logger.info("[AI-ENGINE] 📊 Initializing Risk Mode Predictor (Phase 3A)...")
            try:
                self.risk_mode_predictor = RiskModePredictor(
                    volatility_engine=self.volatility_structure_engine,
                    orderbook_module=self.orderbook_imbalance,
                    vol_threshold_high=0.7,
                    vol_threshold_low=0.3,
                    imbalance_threshold=0.3
                )
                logger.info("[PHASE 3A] RMP: Volatility + orderbook + market conditions")
                logger.info("[PHASE 3A] 📊 Risk Mode Predictor: ONLINE")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Risk Mode Predictor failed: {e}")
                self.risk_mode_predictor = None
            
            # 🔥 PHASE 3B: Strategy Selector - Intelligent strategy selection
            logger.info("[AI-ENGINE] 🎯 Initializing Strategy Selector (Phase 3B)...")
            try:
                self.strategy_selector = StrategySelector(
                    volatility_engine=self.volatility_structure_engine,
                    orderbook_module=self.orderbook_imbalance,
                    risk_mode_predictor=self.risk_mode_predictor,
                    confidence_threshold=0.60
                )
                logger.info("[PHASE 3B] SS: Phase 2D + 2B + 3A integration")
                logger.info("[PHASE 3B] 🎯 Strategy Selector: ONLINE")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Strategy Selector failed: {e}")
                self.strategy_selector = None
            
            # 🔥 PHASE 3C: System Health Monitor - Real-time monitoring
            logger.info("[AI-ENGINE] 🏥 Initializing System Health Monitor (Phase 3C)...")
            try:
                self.health_monitor = SystemHealthMonitor(
                    check_interval_sec=60,        # Check every 60 seconds
                    alert_retention_hours=24,     # Keep alerts for 24 hours
                    metrics_history_size=1000     # Keep 1000 health checks
                )
                
                # Link all modules to health monitor
                self.health_monitor.set_modules(
                    orderbook_module=self.orderbook_imbalance,
                    volatility_engine=self.volatility_structure_engine,
                    risk_mode_predictor=self.risk_mode_predictor,
                    strategy_selector=self.strategy_selector,
                    ensemble_manager=self.ensemble_manager
                )
                
                logger.info("[PHASE 3C] SHM: All modules linked (2B, 2D, 3A, 3B, ensemble)")
                logger.info("[PHASE 3C] 🏥 System Health Monitor: ONLINE")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ System Health Monitor failed: {e}")
                self.health_monitor = None
            
            # 🔥 PHASE 3C-2: Performance Benchmarker
            logger.info("[AI-ENGINE] 📊 Initializing Performance Benchmarker (Phase 3C-2)...")
            try:
                self.performance_benchmarker = PerformanceBenchmarker(
                    benchmark_interval_sec=300,      # Benchmark every 5 minutes
                    history_retention_hours=168,     # Keep 7 days of data
                    latency_sample_size=1000,        # 1000 latency samples
                    regression_threshold_pct=20.0    # 20% performance drop = regression
                )
                
                # Link all modules
                self.performance_benchmarker.set_modules(
                    orderbook_module=self.orderbook_imbalance,
                    volatility_engine=self.volatility_structure_engine,
                    risk_mode_predictor=self.risk_mode_predictor,
                    strategy_selector=self.strategy_selector,
                    ensemble_manager=self.ensemble_manager
                )
                
                logger.info("[PHASE 3C-2] PB: Benchmarking all modules (5min interval)")
                logger.info("[PHASE 3C-2] 📊 Performance Benchmarker: ONLINE")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Performance Benchmarker failed: {e}")
                self.performance_benchmarker = None
            
            # 🔥 PHASE 3C-3: Adaptive Threshold Manager
            logger.info("[AI-ENGINE] 🧠 Initializing Adaptive Threshold Manager (Phase 3C-3)...")
            try:
                self.adaptive_threshold_manager = AdaptiveThresholdManager(
                    learning_rate=0.1,                  # 10% learning rate
                    min_samples_for_learning=100,       # Need 100 samples to learn
                    false_positive_target=0.05,         # Target 5% false positive rate
                    adjustment_interval_hours=24,       # Review thresholds daily
                    confidence_threshold=0.7            # 70% confidence to apply
                )
                
                logger.info("[PHASE 3C-3] ATM: Learning optimal thresholds (24h review cycle)")
                logger.info("[PHASE 3C-3] 🧠 Adaptive Threshold Manager: ONLINE")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Adaptive Threshold Manager failed: {e}")
                self.adaptive_threshold_manager = None
            
            logger.info("[AI-ENGINE] 🔍 DEBUG: Reached adapter initialization section")
            
            # 🔥 PHASE 3C ADAPTERS: Exit Brain Performance Adapter
            logger.info("[AI-ENGINE] 🎯 Initializing Exit Brain Performance Adapter (Phase 3C)...")
            try:
                from backend.services.ai.exit_brain_performance_adapter import ExitBrainPerformanceAdapter
                
                self.exit_brain_performance_adapter = ExitBrainPerformanceAdapter(
                    performance_benchmarker=self.performance_benchmarker,
                    adaptive_threshold_manager=self.adaptive_threshold_manager,
                    system_health_monitor=self.health_monitor,
                    default_tp_multipliers=(1.0, 2.5, 4.0),
                    default_sl_multiplier=1.5,
                    min_sample_size=20,
                    health_threshold=70.0
                )
                
                logger.info("[PHASE 3C] ✅ Exit Brain Performance Adapter initialized")
                logger.info("[PHASE 3C] 🎯 Features: Adaptive TP/SL, Health gating, Predictive tightening")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Exit Brain Performance Adapter failed: {e}")
                self.exit_brain_performance_adapter = None
            
            # 🔥 PHASE 3C ADAPTERS: Confidence Calibrator
            print("🔥 DEBUG [LINE 722]: Starting Confidence Calibrator init...", flush=True)
            logger.info("[AI-ENGINE] 🎯 Initializing Confidence Calibrator (Phase 3C)...")
            try:
                print("🔥 DEBUG [LINE 725]: Importing ConfidenceCalibrator...", flush=True)
                from backend.services.ai.confidence_calibrator import ConfidenceCalibrator
                
                print("🔥 DEBUG [LINE 728]: Creating ConfidenceCalibrator instance...", flush=True)
                self.confidence_calibrator = ConfidenceCalibrator(
                    performance_benchmarker=self.performance_benchmarker,
                    smoothing_factor=0.7,
                    min_confidence=0.1,
                    max_confidence=0.95,
                    min_sample_size=20
                )
                
                print("🔥 DEBUG [LINE 736]: ConfidenceCalibrator created successfully!", flush=True)
                logger.info("[PHASE 3C] ✅ Confidence Calibrator initialized")
                logger.info("[PHASE 3C] 🎯 Features: Historical accuracy calibration, Module-specific tracking")
                print("🔥 DEBUG [LINE 739]: Logged Confidence Calibrator success", flush=True)
            except Exception as e:
                print(f"🔥 DEBUG [LINE 741]: Confidence Calibrator exception: {e}", flush=True)
                logger.warning(f"[AI-ENGINE] ⚠️ Confidence Calibrator failed: {e}")
                self.confidence_calibrator = None
            
            print("🔥 DEBUG [LINE 745]: BEFORE ORCHESTRATOR INIT - THIS IS THE CRITICAL LINE!", flush=True)
            
            # 🔥 PHASE 2.2: CEO Brain Orchestrator
            print("🔥 DEBUG [LINE 748]: About to initialize orchestrator...", flush=True)
            logger.info("[AI-ENGINE] 🧠 Initializing CEO Brain Orchestrator (Phase 2.2)...")
            try:
                print("🔥 DEBUG [LINE 751]: Calling get_orchestrator()...", flush=True)
                self.orchestrator = get_orchestrator()
                print("🔥 DEBUG [LINE 753]: get_orchestrator() returned successfully!", flush=True)
                logger.info("[PHASE 2.2] ✅ CEO Brain Orchestrator initialized")
                logger.info("[PHASE 2.2] 🧠 Features: CEO Brain (mode), Strategy Brain (eval), Risk Brain (sizing)")
            except Exception as e:
                print(f"🔥 DEBUG [LINE 757]: Exception in orchestrator init: {e}", flush=True)
                logger.warning(f"[AI-ENGINE] ⚠️ CEO Brain Orchestrator failed: {e}")
                self.orchestrator = None
                
            print("🔥 DEBUG [LINE 761]: AFTER ORCHESTRATOR INIT BLOCK", flush=True)
                
        except Exception as e:
            print(f"🔥 DEBUG: CRITICAL EXCEPTION in _load_ai_modules: {e}", flush=True)
            logger.error(f"[AI-ENGINE] ❌ Critical error loading AI modules: {e}", exc_info=True)
            raise
            
        print("🔥 DEBUG: _load_ai_modules() COMPLETED SUCCESSFULLY!", flush=True)
        
    # ========================================================================
    
    async def update_price_history(self, symbol: str, price: float, volume: float = 0.0):
        """Update price history from API requests or market.tick events."""
        if not symbol or price <= 0:
            logger.warning(f"[AI-ENGINE] ⚠️ Invalid price history update: symbol={symbol}, price={price}")
            return
        
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            self._volume_history[symbol] = []
            logger.info(f"[AI-ENGINE] 🆕 Creating new price history for {symbol}")
        
        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)
        
        # Trim to max length
        if len(self._price_history[symbol]) > self._history_max_len:
            self._price_history[symbol] = self._price_history[symbol][-self._history_max_len:]
            self._volume_history[symbol] = self._volume_history[symbol][-self._history_max_len:]
        
        # 🔥 PHASE 2D: Feed price data to Volatility Structure Engine
        if self.volatility_structure_engine:
            try:
                self.volatility_structure_engine.update_price_data(symbol, price)
            except Exception as vol_error:
                logger.error(f"[AI-ENGINE] Volatility engine update failed: {vol_error}")
        
        logger.info(f"[AI-ENGINE] ✅ Price history updated: {symbol} @ ${price:.2f} (len={len(self._price_history[symbol])})")
    
    async def update_orderbook(self, symbol: str, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """
        Update orderbook data for orderbook imbalance analysis.
        
        Args:
            symbol: Trading pair
            bids: List of (price, quantity) tuples for bids
            asks: List of (price, quantity) tuples for asks
        """
        # 🔥 PHASE 2B: Feed orderbook data to Orderbook Imbalance Module
        if self.orderbook_imbalance:
            try:
                self.orderbook_imbalance.update_orderbook(symbol, bids, asks)
                logger.debug(f"[PHASE 2B] Orderbook updated for {symbol}: {len(bids)} bids, {len(asks)} asks")
            except Exception as ob_error:
                logger.error(f"[AI-ENGINE] Orderbook update failed: {ob_error}")
    
    async def _fetch_orderbook_loop(self):
        """
        Periodically fetch orderbook data for active symbols.
        
        Phase 2B: REST API polling approach (1-2 updates/sec per symbol).
        Alternative: WebSocket for real-time updates (10-100 updates/sec).
        """
        from backend.services.binance_market_data import BinanceMarketDataFetcher
        
        try:
            self._orderbook_client = BinanceMarketDataFetcher()
            logger.info("[PHASE 2B] 📖 Orderbook client initialized")
        except Exception as e:
            logger.error(f"[PHASE 2B] Failed to initialize orderbook client: {e}")
            return
        
        while self._running:
            try:
                # Track symbols from price history
                if not self._active_symbols and self._price_history:
                    self._active_symbols = list(self._price_history.keys())[:10]  # Limit to 10 symbols
                    logger.info(f"[PHASE 2B] Tracking {len(self._active_symbols)} symbols for orderbook")
                
                for symbol in self._active_symbols:
                    try:
                        # Fetch orderbook (limit=20 levels)
                        book = self._orderbook_client.client.futures_order_book(symbol=symbol, limit=20)
                        
                        # Convert to expected format
                        bids = [(float(p), float(q)) for p, q in book['bids']]
                        asks = [(float(p), float(q)) for p, q in book['asks']]
                        
                        # Update orderbook module
                        await self.update_orderbook(symbol, bids, asks)
                        
                    except Exception as symbol_error:
                        logger.debug(f"[PHASE 2B] Orderbook fetch failed for {symbol}: {symbol_error}")
                        continue
                
                # Fetch every 1 second (can be adjusted)
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"[PHASE 2B] Orderbook fetch loop error: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    
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
            logger.info(f"[AI-ENGINE] 🎯 Received market.tick event: {event_data}")
            symbol = event_data.get("symbol")
            price = event_data.get("price", 0.0)
            volume = event_data.get("volume", 0.0)
            
            if not symbol or price <= 0:
                logger.warning(f"[AI-ENGINE] Invalid market tick: symbol={symbol}, price={price}")
                return
            
            # Update price history
            await self.update_price_history(symbol, price, volume)
            
            logger.info(f"[AI-ENGINE] Processing tick: {symbol} @ ${price:.2f} "
                       f"(history: {len(self._price_history[symbol])} ticks)")
            
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
    
    async def _handle_exchange_raw(self, event_data: Dict[str, Any]):
        """
        Handle exchange.raw event - LIVE market data from cross-exchange aggregator.
        
        🔥 FIX: Since market.tick stream is stale (frozen Dec 16), we now read directly
        from exchange.raw which has 1.96M live events with current BTC/ETH/SOL prices.
        
        Flow:
        1. Extract symbol + close price from exchange.raw event
        2. Convert to market.tick format
        3. Run ensemble inference
        4. Run meta-strategy selection
        5. Run RL position sizing
        6. Publish ai.decision.made event
        
        NOTE: exchange.raw events come directly as fields (not wrapped in "payload"),
        so event_data will be empty {} from EventBus. We need to access raw stream.
        """
        try:
            # EventBus gives us empty payload for exchange.raw (fields not in JSON)
            # So we skip this handler - aggregator publishes normalized prices instead
            logger.debug(f"[AI-ENGINE] Skipping exchange.raw handler (using normalized prices)")
            return
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error handling exchange.raw: {e}", exc_info=True)
    
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
        - Track PnL for Model Supervisor & Governance (Phase 4D+4E)
        """
        try:
            trade_id = event_data.get("trade_id")
            symbol = event_data.get("symbol", "unknown")
            pnl_percent = event_data.get("pnl_percent", 0.0)
            model = event_data.get("model", "unknown")
            strategy = event_data.get("strategy")
            
            logger.info(f"[AI-ENGINE] Trade closed: {trade_id} (PnL={pnl_percent:.2f}%, model={model})")
            
            # Phase 4D+4E: Track PnL for governance
            if self.supervisor_governance and symbol:
                if symbol not in self._governance_pnl:
                    self._governance_pnl[symbol] = {}
                
                # Distribute PnL to all models (simplified - in reality you'd track per model)
                for model_name in ["PatchTST", "NHiTS", "XGBoost", "LightGBM"]:
                    self._governance_pnl[symbol][model_name] = pnl_percent
                
                logger.info(f"[Governance] PnL tracked for {symbol}: {pnl_percent:.2f}%")
            
            # TODO: Update Meta-Strategy Q-table
            if self.meta_strategy_selector and strategy:
                # reward = pnl_percent / 100.0  # Convert to R units
                pass
            
            # TODO: Update RL Sizing Q-table
            if self.rl_sizing_agent:
                pass
            
            # 🔥 PHASE 1: Feed trade outcome to Reinforcement Signal Manager
            if self.rl_signal_manager:
                try:
                    from backend.services.ai.reinforcement_signal_manager import TradeOutcome
                    
                    outcome = TradeOutcome(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        action=event_data.get("action", "unknown"),
                        confidence=event_data.get("confidence", 0.5),
                        pnl=pnl_percent / 100.0,  # Convert to decimal
                        position_size=event_data.get("position_size", 0.0),
                        entry_price=event_data.get("entry_price", 0.0)
                    )
                    
                    await asyncio.to_thread(self.rl_signal_manager.learn_from_outcome, outcome)
                    logger.info(f"[PHASE 1] RL Signal learned from {symbol} trade: PnL={pnl_percent:.2f}%")
                except Exception as rl_error:
                    logger.error(f"[AI-ENGINE] RL Signal Manager learning failed: {rl_error}")
            
            # 🔥 PHASE 2C: Feed trade outcome to Continuous Learning Manager
            if self.continuous_learning_manager:
                try:
                    # Build trade outcome for CLM
                    trade_outcome = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "action": event_data.get("action", "unknown"),
                        "confidence": event_data.get("confidence", 0.5),
                        "pnl_percent": pnl_percent,
                        "model": model,
                        "strategy": strategy,
                        "entry_price": event_data.get("entry_price", 0.0),
                        "exit_price": event_data.get("exit_price", 0.0),
                        "position_size": event_data.get("position_size", 0.0)
                    }
                    
                    # Add to buffer
                    self._clm_trade_buffer.append(trade_outcome)
                    
                    # Check if we should trigger retraining
                    buffer_size = len(self._clm_trade_buffer)
                    
                    if buffer_size >= 5000:
                        logger.info(f"[PHASE 2C] CLM buffer reached {buffer_size} trades - triggering retrain check")
                        
                        # Trigger retraining for all registered models
                        for model_name in ["ensemble", "lstm", "xgboost", "lightgbm"]:
                            try:
                                new_model = await self.continuous_learning_manager.retrain_model(
                                    model_name=model_name,
                                    training_data=self._clm_trade_buffer
                                )
                                logger.info(f"[PHASE 2C] ✅ Retrained {model_name} to v{new_model.version} with {buffer_size} samples")
                                
                                # Check if shadow should be promoted
                                should_promote = await self.continuous_learning_manager.check_shadow_promotion(model_name)
                                if should_promote:
                                    logger.info(f"[PHASE 2C] 🎉 Promoted {model_name} v{new_model.version} to LIVE")
                            except Exception as retrain_error:
                                logger.error(f"[PHASE 2C] Retraining {model_name} failed: {retrain_error}")
                        
                        # Clear buffer after retraining
                        self._clm_trade_buffer.clear()
                        logger.info("[PHASE 2C] CLM buffer cleared, waiting for next 5000 trades")
                    else:
                        # Log progress every 100 trades
                        if buffer_size % 100 == 0:
                            logger.info(f"[PHASE 2C] CLM progress: {buffer_size}/5000 trades collected ({buffer_size/50:.1f}%)")
                
                except Exception as clm_error:
                    logger.error(f"[AI-ENGINE] CLM processing failed: {clm_error}", exc_info=True)
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error handling trade.closed: {e}", exc_info=True)
    
    async def _handle_policy_updated(self, event_data: Dict[str, Any]):
        """Handle policy.updated event from risk-safety-service."""
        key = event_data.get("key")
        new_value = event_data.get("new_value")
        logger.info(f"[AI-ENGINE] Policy updated: {key} = {new_value}")
        
        # TODO: Refresh policy snapshot if needed
    
    # ========================================================================
    # INDICATOR CALCULATIONS
    # ========================================================================
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI from price history."""
        if len(prices) < period + 1:
            return 50.0  # Neutral if insufficient data
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26) -> float:
        """Calculate MACD line (fast EMA - slow EMA)."""
        if len(prices) < slow:
            return 0.0  # Neutral if insufficient data
        
        def ema(data: List[float], period: int) -> float:
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        fast_ema = ema(prices[-fast:], fast)
        slow_ema = ema(prices[-slow:], slow)
        return fast_ema - slow_ema
    
    def _calculate_volume_ratio(self, volumes: List[float], window: int = 20) -> float:
        """Calculate current volume relative to average."""
        if len(volumes) < 2:
            return 1.0
        
        current_vol = volumes[-1]
        if len(volumes) < window:
            avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else current_vol
        else:
            avg_vol = sum(volumes[-window:-1]) / (window - 1)
        
        if avg_vol == 0:
            return 1.0
        
        return current_vol / avg_vol
    
    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate momentum as percentage change over period."""
        if len(prices) < period + 1:
            return 0.0
        
        old_price = prices[-period-1]
        current_price = prices[-1]
        
        if old_price == 0:
            return 0.0
        
        return ((current_price - old_price) / old_price) * 100
    
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
        0. ESS Kill Switch check → block if emergency stop active
        1. Ensemble inference → action + confidence
        2. Meta-Strategy selection → strategy
        3. RL Position Sizing → size + leverage + TP/SL
        4. Publish ai.decision.made event
        
        Returns:
            AIDecisionMadeEvent if signal generated, None otherwise
        """
        # 🔥 PHASE 3C: Track signal generation for health monitoring and benchmarking
        start_time = datetime.utcnow()
        success = False
        
        try:
            logger.info(f"[AI-ENGINE] 🔍 generate_signal START: {symbol}, price={current_price}")
            
            # Get current price if not provided
            if current_price is None:
                prices = self._price_history.get(symbol, [])
                if prices:
                    current_price = prices[-1]
                    logger.info(f"[AI-ENGINE] ✅ Got price from history: ${current_price:.2f}")
                else:
                    logger.warning(f"[AI-ENGINE] ❌ No price data for {symbol}, cannot generate signal")
                    return None
            
            logger.info(f"[AI-ENGINE] ✅ Price confirmed: {symbol} @ ${current_price:.2f}")
            
            # Step 0: ESS Kill Switch - Check emergency stop
            logger.info(f"[AI-ENGINE] 🔍 Checking emergency stop...")
            emergency_stop = await self.redis_client.get("trading:emergency_stop")
            logger.info(f"[AI-ENGINE] ✅ Emergency stop check: {emergency_stop}")
            if emergency_stop == b"1":
                logger.critical(
                    f"[AI-ENGINE] 🚨 EMERGENCY STOP ACTIVE - Signal generation blocked for {symbol}"
                )
                return None
            
            # Step 1: Ensemble inference
            if not self.ensemble_manager:
                logger.warning(f"[AI-ENGINE] Ensemble not loaded, using fallback for {symbol}")
                # Set to default values that will trigger fallback logic
                action = "HOLD"
                ensemble_confidence = 0.60
                votes_info = {"fallback": True}
                # Still calculate features for fallback logic
                prices = self._price_history.get(symbol, [])
                volumes = self._volume_history.get(symbol, [])
                features = {
                    "price": current_price,
                    "price_change": self._calculate_momentum(prices, period=1),  # 1-period price change for LGBM
                    "rsi_14": self._calculate_rsi(prices, period=14),
                    "macd": self._calculate_macd(prices, fast=12, slow=26),
                    "volume_ratio": self._calculate_volume_ratio(volumes, window=20),
                    "momentum_10": self._calculate_momentum(prices, period=10),
                }
            else:
                logger.debug(f"[AI-ENGINE] Running ensemble for {symbol}...")
                
                # Calculate real technical indicators from price history
                prices = self._price_history.get(symbol, [])
                volumes = self._volume_history.get(symbol, [])
                
                logger.info(f"[AI-ENGINE] 📊 Price history: {symbol} has {len(prices)} data points")
                
                # Base features
                features = {
                    "price": current_price,
                    "price_change": self._calculate_momentum(prices, period=1),
                    "rsi_14": self._calculate_rsi(prices, period=14),
                    "macd": self._calculate_macd(prices, fast=12, slow=26),
                    "volume_ratio": self._calculate_volume_ratio(volumes, window=20),
                    "momentum_10": self._calculate_momentum(prices, period=10),
                }
                
                # 🔥 PHASE 1: Add Cross-Exchange Features
                if self.cross_exchange_aggregator:
                    try:
                        cross_exchange_data = await asyncio.to_thread(
                            self.cross_exchange_aggregator.get_latest_features,
                            symbol
                        )
                        if cross_exchange_data:
                            features["volatility_factor"] = cross_exchange_data.get("volatility_factor", 1.0)
                            features["exchange_divergence"] = cross_exchange_data.get("divergence", 0.0)
                            features["lead_lag_score"] = cross_exchange_data.get("lead_lag", 0.0)
                            logger.info(f"[PHASE 1] Cross-Exchange: volatility={features['volatility_factor']:.3f}, "
                                      f"divergence={features['exchange_divergence']:.4f}")
                    except Exception as e:
                        logger.warning(f"[PHASE 1] Cross-Exchange feature extraction failed: {e}")
                
                # 🔥 PHASE 1: Add Funding Rate Features
                if self.funding_rate_filter:
                    try:
                        funding_data = await asyncio.to_thread(
                            self.funding_rate_filter.get_funding_features,
                            symbol
                        )
                        if funding_data:
                            features["funding_rate"] = funding_data.get("current_funding", 0.0)
                            features["funding_delta"] = funding_data.get("funding_delta", 0.0)
                            features["crowded_side_score"] = funding_data.get("crowd_bias", 0.0)
                            logger.info(f"[PHASE 1] Funding: rate={features['funding_rate']:.5f}, "
                                      f"crowd_bias={features['crowded_side_score']:.2f}")
                    except Exception as e:
                        logger.warning(f"[PHASE 1] Funding rate feature extraction failed: {e}")
                
                # 🔥 PHASE 2D: Add Volatility Structure Features (ATR-trend, cross-TF)
                if self.volatility_structure_engine:
                    try:
                        vol_analysis = await asyncio.to_thread(
                            self.volatility_structure_engine.get_complete_volatility_analysis,
                            symbol
                        )
                        
                        # Add all 11 volatility metrics to features
                        features["atr"] = vol_analysis["atr"]
                        features["atr_trend"] = vol_analysis["atr_trend"]
                        features["atr_acceleration"] = vol_analysis["atr_acceleration"]
                        features["short_term_vol"] = vol_analysis["short_term_vol"]
                        features["medium_term_vol"] = vol_analysis["medium_term_vol"]
                        features["long_term_vol"] = vol_analysis["long_term_vol"]
                        features["vol_ratio_short_long"] = vol_analysis["vol_ratio_short_long"]
                        features["volatility_score"] = vol_analysis["volatility_score"]
                        
                        logger.info(f"[PHASE 2D] Volatility: ATR={vol_analysis['atr']:.4f}, "
                                  f"trend={vol_analysis['atr_trend']:.2f} ({vol_analysis['atr_regime']}), "
                                  f"score={vol_analysis['volatility_score']:.3f}, regime={vol_analysis['overall_regime']}")
                    except Exception as vol_error:
                        logger.warning(f"[PHASE 2D] Volatility feature extraction failed: {vol_error}")
                
                # 🔥 PHASE 2B: Add Orderbook Imbalance Features
                if self.orderbook_imbalance:
                    try:
                        orderbook_metrics = await asyncio.to_thread(
                            self.orderbook_imbalance.get_metrics,
                            symbol
                        )
                        
                        if orderbook_metrics:
                            # Add 5 key orderbook metrics to features
                            features["orderflow_imbalance"] = orderbook_metrics.orderflow_imbalance
                            features["delta_volume"] = orderbook_metrics.delta_volume
                            features["bid_ask_spread_pct"] = orderbook_metrics.bid_ask_spread_pct
                            features["order_book_depth_ratio"] = orderbook_metrics.order_book_depth_ratio
                            features["large_order_presence"] = orderbook_metrics.large_order_presence
                            
                            logger.info(f"[PHASE 2B] Orderbook: imbalance={orderbook_metrics.orderflow_imbalance:.3f}, "
                                      f"delta={orderbook_metrics.delta_volume:.2f}, "
                                      f"depth_ratio={orderbook_metrics.order_book_depth_ratio:.3f}, "
                                      f"large_orders={orderbook_metrics.large_order_presence:.2f}")
                    except Exception as ob_error:
                        logger.warning(f"[PHASE 2B] Orderbook feature extraction failed: {ob_error}")
                
                logger.info(f"[AI-ENGINE] Features for {symbol}: RSI={features['rsi_14']:.1f}, "
                            f"MACD={features['macd']:.4f}, VolumeRatio={features['volume_ratio']:.2f}, "
                            f"Momentum={features['momentum_10']:.2f}%")
                
                # 🔥 PHASE 3A: Predict risk mode
                risk_signal = None
                risk_multiplier = 1.0
                if self.risk_mode_predictor:
                    try:
                        risk_signal = await asyncio.to_thread(
                            self.risk_mode_predictor.predict_risk_mode,
                            symbol=symbol,
                            current_price=current_price,
                            market_conditions={}  # TODO: Add BTC dominance, funding rates, fear/greed
                        )
                        risk_multiplier = self.risk_mode_predictor.get_risk_multiplier(risk_signal.mode)
                        
                        logger.info(f"[PHASE 3A] {symbol} Risk: {risk_signal.mode.value} "
                                  f"(confidence={risk_signal.confidence:.1%}, "
                                  f"regime={risk_signal.regime.value}, "
                                  f"multiplier={risk_multiplier:.2f}x)")
                        logger.info(f"[PHASE 3A] {symbol} Scores: vol={risk_signal.volatility_score:.2f}, "
                                  f"flow={risk_signal.orderflow_score:.2f}, "
                                  f"market={risk_signal.market_condition_score:.2f}")
                        logger.info(f"[PHASE 3A] {symbol} Reason: {risk_signal.reason}")
                    except Exception as e:
                        logger.warning(f"[PHASE 3A] Risk prediction failed for {symbol}: {e}")
                        risk_multiplier = 1.0
                
                # Call ensemble predict - returns (action, confidence, info_dict)
                action, ensemble_confidence, votes_info = await asyncio.to_thread(
                    self.ensemble_manager.predict,
                    symbol=symbol,
                    features=features
                )
                logger.info(f"[AI-ENGINE] 🎯 Ensemble returned: action='{action}' (type={type(action)}), confidence={ensemble_confidence}")
                
                # 🔥 PHASE 3B: Select optimal trading strategy
                strategy_selection = None
                selected_strategy = "momentum_conservative"  # Default
                if self.strategy_selector:
                    try:
                        strategy_selection = await asyncio.to_thread(
                            self.strategy_selector.select_strategy,
                            symbol=symbol,
                            current_price=current_price,
                            ensemble_confidence=ensemble_confidence,
                            market_conditions={}  # TODO: Add external market data
                        )
                        selected_strategy = strategy_selection.primary_strategy.value
                        
                        logger.info(f"[PHASE 3B] {symbol} Strategy: {selected_strategy} "
                                  f"(conf={strategy_selection.confidence:.1%}, "
                                  f"align={strategy_selection.market_alignment_score:.2f})")
                        logger.info(f"[PHASE 3B] {symbol} Reasoning: {strategy_selection.reasoning}")
                        
                        if strategy_selection.secondary_strategy:
                            logger.info(f"[PHASE 3B] {symbol} Secondary: {strategy_selection.secondary_strategy.value}")
                    except Exception as e:
                        logger.warning(f"[PHASE 3B] Strategy selection failed for {symbol}: {e}")
            
            # FALLBACK: If ML models return HOLD with low confidence, use rule-based signals for testing
            fallback_triggered = False
            if action == "HOLD" and 0.50 <= ensemble_confidence <= 0.65:
                rsi = features.get('rsi_14', 50)
                macd = features.get('macd', 0)
                
                # Log feature values for debugging
                logger.info(f"[AI-ENGINE] 🔍 FALLBACK CHECK {symbol}: RSI={rsi:.1f}, MACD={macd:.4f}, price_history_len={len(self._price_history.get(symbol, []))}")
                
                # TEMPORARY: Very relaxed thresholds for immediate testing
                # When history < 15 points, use extreme relaxed logic
                history_len = len(self._price_history.get(symbol, []))
                if history_len < 15:
                    # Use simple alternating pattern for testing when no real RSI available
                    import hashlib
                    symbol_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
                    if symbol_hash % 3 == 0:  # ~33% BUY
                        action = "BUY"
                        ensemble_confidence = 0.68
                        fallback_triggered = True
                        logger.info(f"[AI-ENGINE] 🔥 FALLBACK BUY signal (testing mode): {symbol}")
                    elif symbol_hash % 3 == 1:  # ~33% SELL
                        action = "SELL"
                        ensemble_confidence = 0.68
                        fallback_triggered = True
                        logger.info(f"[AI-ENGINE] 🔥 FALLBACK SELL signal (testing mode): {symbol}")
                    # else: HOLD (~33%)
                else:
                    # Normal RSI-based fallback when enough history
                    if rsi < 35 and macd > -0.001:  # Slightly oversold + neutral/bullish momentum
                        action = "BUY"
                        ensemble_confidence = 0.72
                        fallback_triggered = True
                        logger.info(f"[AI-ENGINE] 🔥 FALLBACK BUY signal: {symbol} RSI={rsi:.1f}, MACD={macd:.4f}")
                    elif rsi > 65 and macd < 0.001:  # Slightly overbought + neutral/bearish momentum
                        action = "SELL"
                        ensemble_confidence = 0.72
                        fallback_triggered = True
                        logger.info(f"[AI-ENGINE] 🔥 FALLBACK SELL signal: {symbol} RSI={rsi:.1f}, MACD={macd:.4f}")
            
            logger.info(f"[AI-ENGINE] 🔍 Action check: repr={repr(action)}, equals_HOLD={action == 'HOLD'}, fallback={fallback_triggered}")
            
            if not action or action == "HOLD":
                logger.info(f"[AI-ENGINE] ⚠️ No actionable signal for {symbol}")
                return None
            
            logger.info(f"[AI-ENGINE] ✅ Action confirmed: {action} (confidence={ensemble_confidence:.2f})")
            
            # Build model_votes and consensus - if fallback triggered, use synthetic values
            if fallback_triggered:
                model_votes = {action: "fallback"}
                consensus = 1
            else:
                model_votes = votes_info.get("votes", {})
                consensus = votes_info.get("consensus_count", 0)  # FIX: int not str
            
            # 🔥 PHASE 1: Apply RL Signal Manager Confidence Calibration
            if self.rl_signal_manager and not fallback_triggered:
                try:
                    calibrated_confidence = await asyncio.to_thread(
                        self.rl_signal_manager.calibrate_confidence,
                        symbol=symbol,
                        raw_confidence=ensemble_confidence,
                        action=action
                    )
                    logger.info(f"[PHASE 1] RL Calibration: {ensemble_confidence:.3f} → {calibrated_confidence:.3f}")
                    ensemble_confidence = calibrated_confidence
                except Exception as e:
                    logger.warning(f"[PHASE 1] Confidence calibration failed: {e}")
            
            # 🔥 PHASE 1: Check Funding Rate Filter (block high-cost trades)
            if self.funding_rate_filter:
                try:
                    funding_check = await asyncio.to_thread(
                        self.funding_rate_filter.should_block_trade_simple,
                        symbol
                    )
                    if funding_check.get("blocked", False):
                        logger.warning(
                            f"[PHASE 1] 🚫 FUNDING FILTER BLOCKED: {symbol} "
                            f"(rate={funding_check.get('funding_rate', 0):.5f}, "
                            f"reason={funding_check.get('reason', 'unknown')})"
                        )
                        return None
                except Exception as e:
                    logger.warning(f"[PHASE 1] Funding filter check failed: {e}")
            
            # 🔥 PHASE 1: Check Drift Detection (block if model drifted)
            if self.drift_detector:
                try:
                    drift_status = await asyncio.to_thread(
                        self.drift_detector.check_drift,
                        symbol=symbol,
                        features=features,
                        prediction=ensemble_confidence
                    )
                    if drift_status.get("severity") in ["SEVERE", "CRITICAL"]:
                        # 🔥 PHASE 1 FIX: Allow signals on testnet for data collection
                        is_testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
                        if is_testnet:
                            logger.warning(
                                f"[PHASE 1] ⚠️  DRIFT DETECTED: {symbol} "
                                f"(severity={drift_status.get('severity')}, "
                                f"psi={drift_status.get('psi', 0):.3f}) - ALLOWING on testnet for data collection"
                            )
                        else:
                            logger.warning(
                                f"[PHASE 1] 🚫 DRIFT DETECTED: {symbol} "
                                f"(severity={drift_status.get('severity')}, "
                                f"psi={drift_status.get('psi', 0):.3f}) - Signal blocked"
                            )
                            # Trigger retraining if CLM available
                            if self.adaptive_retrainer:
                                logger.info(f"[PHASE 1] Triggering retrain due to drift...")
                            return None
                except Exception as e:
                    logger.warning(f"[PHASE 1] Drift detection check failed: {e}")
            
            # Check minimum confidence
            logger.info(f"[AI-ENGINE] 🔍 Confidence check: {ensemble_confidence:.2f} vs min {settings.MIN_SIGNAL_CONFIDENCE:.2f}")
            if ensemble_confidence < settings.MIN_SIGNAL_CONFIDENCE:
                logger.info(
                    f"[AI-ENGINE] ❌ Signal rejected: {symbol} confidence={ensemble_confidence:.2f} "
                    f"< {settings.MIN_SIGNAL_CONFIDENCE:.2f}"
                )
                return None
            
            logger.info(f"[AI-ENGINE] ✅ Confidence check passed!")
            
            logger.info(
                f"[AI-ENGINE] ✅ Ensemble: {symbol} {action} "
                f"(confidence={ensemble_confidence:.2f}, consensus={consensus}/4)"
            )
            
            # Phase 4D+4E: Run Governance Cycle
            if self.supervisor_governance and not fallback_triggered:
                try:
                    # Store predictions for governance tracking
                    if symbol not in self._governance_predictions:
                        self._governance_predictions[symbol] = {}
                    
                    # Create synthetic predictions dict (in real scenario, ensemble_manager would return per-model predictions)
                    model_predictions = {
                        "PatchTST": np.array([current_price * (1.01 if action == "BUY" else 0.99)]),
                        "NHiTS": np.array([current_price * (1.01 if action == "BUY" else 0.99)]),
                        "XGBoost": np.array([current_price * (1.01 if action == "BUY" else 0.99)]),
                        "LightGBM": np.array([current_price * (1.01 if action == "BUY" else 0.99)])
                    }
                    self._governance_predictions[symbol] = model_predictions
                    
                    # Track actual prices for comparison
                    if symbol not in self._governance_actuals:
                        self._governance_actuals[symbol] = []
                    self._governance_actuals[symbol].append(current_price)
                    if len(self._governance_actuals[symbol]) > 100:
                        self._governance_actuals[symbol].pop(0)
                    
                    # Get PnL data (if available from closed trades)
                    pnl_data = self._governance_pnl.get(symbol, {
                        "PatchTST": 0.0, "NHiTS": 0.0, "XGBoost": 0.0, "LightGBM": 0.0
                    })
                    
                    # Run governance cycle
                    if len(self._governance_actuals[symbol]) >= 2:
                        actuals_array = np.array(self._governance_actuals[symbol][-1:])  # Use latest actual
                        weights = self.supervisor_governance.run_cycle(
                            predictions=model_predictions,
                            actuals=actuals_array,
                            pnl=pnl_data
                        )
                        
                        logger.info(f"[Governance] Cycle complete for {symbol} - Weights: {weights}")
                    
                except Exception as e:
                    logger.error(f"[Governance] Error in cycle: {e}", exc_info=True)
            
            # 🔥 PHASE 3C: Calibrate confidence before publishing signal
            calibrated_confidence = ensemble_confidence
            if self.confidence_calibrator:
                try:
                    calibration_result = await self.confidence_calibrator.calibrate_confidence(
                        signal_source="ensemble_manager",
                        raw_confidence=ensemble_confidence,
                        symbol=symbol,
                        metadata={"action": action, "consensus": consensus}
                    )
                    calibrated_confidence = calibration_result.calibrated_confidence
                    
                    logger.info(
                        f"[PHASE 3C] 📊 Confidence calibrated for {symbol}: "
                        f"{ensemble_confidence:.2%} → {calibrated_confidence:.2%} "
                        f"(factor: {calibration_result.calibration_factor:.3f})"
                    )
                except Exception as e:
                    logger.warning(f"[PHASE 3C] ⚠️ Confidence calibration failed for {symbol}: {e}")
            
            # Publish intermediate event
            await self.event_bus.publish("ai.signal_generated", AISignalGeneratedEvent(
                symbol=symbol,
                action=SignalAction(action.lower()),
                confidence=calibrated_confidence,  # Use calibrated confidence
                ensemble_confidence=ensemble_confidence,  # Keep original for comparison
                model_votes=model_votes,
                consensus=consensus,
                timestamp=datetime.now(timezone.utc).isoformat()
            ).dict())
            
            # Step 2: Meta-Strategy selection
            strategy_id = StrategyID.DEFAULT
            strategy_name = "default"
            regime = MarketRegime.UNKNOWN
            
            # TEMPORARY: Bypass Meta-Strategy and RegimeDetector to test basic signal flow
            # TODO: Fix RegimeDetector.detect_regime() API signature mismatch
            if False and self.meta_strategy_selector:  # Disabled temporarily
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
            
            # Step 3: RL Position Sizing (position_size_usd ONLY)
            # NOTE: Leverage skal beregnes av ExitBrain v3.5 (ILF), ikke her!
            position_size_usd = 200.0  # Default fallback
            leverage = 1  # Placeholder (ExitBrain overstyrer)
            tp_percent = 0.06  # 6% (ExitBrain overstyrer)
            sl_percent = 0.025  # 2.5% (ExitBrain overstyrer)
            
            # 🔥 AI-DETERMINED POSITION SIZE (RL Agent)
            if self.rl_sizing_agent:
                logger.debug(f"[AI-ENGINE] Calculating position size for {symbol}...")
                
                # Get ATR from features (for volatility-based sizing)
                atr_value = features.get("atr", 0.02)  # Default 2% if not available
                atr_pct = atr_value  # Already in percentage format
                
                # TODO: Get real portfolio state from execution-service
                portfolio_exposure = 0.0
                
                # TODO: Get real account equity from execution-service
                equity_usd = 10000.0  # Default $10K account
                
                sizing_decision = await asyncio.to_thread(
                    self.rl_sizing_agent.decide_sizing,  # ✅ FIXED: decide_sizing (not decide_size)
                    symbol=symbol,
                    confidence=ensemble_confidence,
                    atr_pct=atr_pct,  # ✅ ADDED: ATR percentage
                    current_exposure_pct=portfolio_exposure,  # ✅ FIXED: parameter name
                    equity_usd=equity_usd,  # ✅ ADDED: account equity
                    adx=None,  # Optional: ADX indicator (TODO: add if available)
                    trend_strength=None  # Optional: trend strength (TODO: calculate)
                )
                
                position_size_usd = sizing_decision.position_size_usd
                # NOTE: Leverage fra RL Agent ignoreres - ExitBrain v3.5 (ILF) bestemmer leverage!
                leverage = 1  # Placeholder for trade.intent (eksekutor overstyrer med ExitBrain)
                tp_percent = sizing_decision.tp_percent
                sl_percent = sizing_decision.sl_percent
                
                # 🔥 PHASE 3A: Apply risk multiplier
                original_size = position_size_usd
                position_size_usd = position_size_usd * risk_multiplier
            # 🔥 TESTNET SIZING: Cap position size
            if os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true":
                original_size = position_size_usd
                position_size_usd = min(position_size_usd, 10.0)  # Max 0 on testnet
                if position_size_usd != original_size:
                    logger.info(f"[TESTNET] Capped position:  → ")
            
                if risk_multiplier != 1.0:
                    logger.info(f"[PHASE 3A] Risk-adjusted position: ${original_size:.0f} → ${position_size_usd:.0f} "
                              f"(multiplier={risk_multiplier:.2f}x, mode={risk_signal.mode.value if risk_signal else 'N/A'})")
                
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
            
            # 🔥 PHASE 2.2: CEO Brain Orchestration Check
            # Evaluate signal through CEO Brain + Strategy Brain + Risk Brain
            if self.orchestrator:
                logger.info(f"[PHASE 2.2] 🧠 Orchestrating signal through CEO Brain pipeline...")
                try:
                    orchestration_signal = {
                        "symbol": symbol,
                        "direction": action.upper(),
                        "confidence": ensemble_confidence,
                        "entry_price": current_price,
                        "position_size_usd": position_size_usd,
                        "leverage": leverage,
                        "regime": regime.value if regime != MarketRegime.UNKNOWN else "unknown",
                        "strategy": strategy_id.value,
                    }
                    
                    orchestration_result = await self.orchestrator.evaluate_signal(orchestration_signal)
                    
                    logger.info(
                        f"[PHASE 2.2] Orchestration result: {orchestration_result.final_decision} "
                        f"(CEO mode={orchestration_result.operating_mode}, "
                        f"strategy_approved={orchestration_result.strategy_approved})"
                    )
                    
                    # If orchestrator rejects, skip publishing trade.intent
                    if orchestration_result.final_decision == "SKIP":
                        logger.warning(
                            f"[PHASE 2.2] 🚫 Signal BLOCKED by orchestrator: {symbol} "
                            f"Reason: {orchestration_result.decision_reason}"
                        )
                        return None
                    
                    # If orchestrator delays, log but continue (can be used for future queueing)
                    if orchestration_result.final_decision == "DELAY":
                        logger.info(
                            f"[PHASE 2.2] ⏸️  Signal DELAYED by orchestrator: {symbol} "
                            f"Reason: {orchestration_result.decision_reason}"
                        )
                        # For now, continue with execution (future: implement queue)
                    
                    # Update position size and leverage from Risk Brain if provided
                    if orchestration_result.position_size > 0:
                        position_size_usd = orchestration_result.position_size
                        logger.info(f"[PHASE 2.2] Risk Brain adjusted size: ${position_size_usd:.2f}")
                    
                    if orchestration_result.leverage > 0:
                        leverage = orchestration_result.leverage
                        logger.info(f"[PHASE 2.2] Risk Brain adjusted leverage: {leverage:.1f}x")
                    
                except Exception as e:
                    logger.error(f"[PHASE 2.2] Orchestration failed: {e}", exc_info=True)
                    # Continue with original signal if orchestration fails
                    logger.warning("[PHASE 2.2] Falling back to original signal (orchestration failed)")
            
            # Step 6: Publish trade.intent for Execution Service
            # NOTE: ExitBrain v3.5 vil beregne leverage via ILF basert på metadata
            
            # Extract consensus details from votes_info
            consensus_count = votes_info.get("consensus_count", 0) if not fallback_triggered else 1
            
            # ALWAYS preserve per-model telemetry - augment with fallback if needed
            model_breakdown = votes_info.get("models", {})
            if fallback_triggered:
                model_breakdown["fallback"] = {
                    "action": action,
                    "confidence": ensemble_confidence,
                    "reason": "consensus_not_met_or_hold_signal",
                    "triggered_by": "rsi_macd_rules" if not self.testnet_mode else "testnet_hash_pattern"
                }
            
                        # TESTNET SIZING: Cap position size
            if os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true":
                position_size_usd = min(position_size_usd, 10.0)
            
            trade_intent_payload = {
                "symbol": symbol,
                "side": action.upper(),
                "position_size_usd": position_size_usd,
                "leverage": leverage,  # Placeholder - ExitBrain overstyrer
                "entry_price": current_price,
                "stop_loss": decision.stop_loss,  # Preliminary - ExitBrain overstyrer
                "take_profit": decision.take_profit,  # Preliminary - ExitBrain overstyrer
                "confidence": ensemble_confidence,
                "timestamp": decision.timestamp,
                "model": "ensemble",
                "meta_strategy": strategy_id.value,
                # 🔥 CONSENSUS DETAILS for Dashboard filtering
                "consensus_count": consensus_count,
                "total_models": 4,  # XGB, LGBM, N-HiTS, PatchTST
                "model_breakdown": model_breakdown,
                # 🔥 METADATA FOR EXITBRAIN v3.5 (ILF + AdaptiveLeverageEngine)
                "atr_value": features.get("atr", 0.02),
                "volatility_factor": features.get("volatility_factor", 1.0),
                "exchange_divergence": features.get("exchange_divergence", 0.0),
                "funding_rate": features.get("funding_rate", 0.0),
                "regime": regime.value if regime != MarketRegime.UNKNOWN else "unknown"
            }
            # 🔥 RATE LIMITING: Global + Per-Symbol
            now = time.time()
            symbol = trade_intent_payload.get("symbol", "UNKNOWN")
            confidence = trade_intent_payload.get("confidence", 0)
            min_confidence = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.75"))
            
            # Filter 1: Confidence threshold
            if confidence < min_confidence:
                logger.debug(f"[RATE-LIMIT] ⛔ {symbol} skipped: confidence {confidence:.2f} < {min_confidence}")
                return None
            
            # Filter 2: Per-symbol cooldown
            last_time = self.last_signal_by_symbol.get(symbol, 0)
            if now - last_time < self.symbol_cooldown_sec:
                wait_sec = int(self.symbol_cooldown_sec - (now - last_time))
                logger.debug(f"[RATE-LIMIT] ⛔ {symbol} skipped: cooldown ({wait_sec}s remaining)")
                return None
            
            # Filter 3: Global rate limit (sliding window - last 60 seconds)
            self.signal_times = deque([t for t in self.signal_times if now - t < 60], maxlen=20)
            if len(self.signal_times) >= self.max_signals_per_min:
                logger.debug(f"[RATE-LIMIT] ⛔ {symbol} skipped: global limit ({len(self.signal_times)}/{self.max_signals_per_min})")
                return None
            
            # ALLOWED - record signal
            self.signal_times.append(now)
            self.last_signal_by_symbol[symbol] = now
            logger.info(f"[RATE-LIMIT] ✅ {symbol} allowed (confidence={confidence:.2f}, rate={len(self.signal_times)}/min)")
            
            # Log telemetry visibility before publishing
            model_keys = list(model_breakdown.keys())
            logger.info(
                f"[TELEMETRY] Publishing trade.intent: {symbol} | "
                f"breakdown_keys={model_keys} | fallback_used={fallback_triggered} | "
                f"consensus={consensus_count}/{trade_intent_payload.get('total_models', 4)} | action={action}"
            )
            
            print(f"[DEBUG] About to publish trade.intent: {trade_intent_payload}")
            await self.event_bus.publish("trade.intent", trade_intent_payload)
            print(f"[DEBUG] trade.intent published to Redis!")
            
            logger.info(
                f"[AI-ENGINE] 🚀 AI DECISION PUBLISHED: {symbol} {action} "
                f"(${position_size_usd:.0f} @ {leverage}x, confidence={ensemble_confidence:.2f})"
            )
            logger.debug(f"[AI-ENGINE] trade.intent event sent to Execution Service")
            
            # 🔥 PHASE 3C: Record successful signal generation
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if self.health_monitor:
                self.health_monitor.record_signal_attempt(success=True, latency_ms=latency_ms)
            
            # 🔥 PHASE 3C-2: Record performance metrics
            if self.performance_benchmarker:
                self.performance_benchmarker.record_latency('ensemble', latency_ms)
            
            # 🔥 PHASE 3C-3: Record metric for learning
            if self.adaptive_threshold_manager:
                self.adaptive_threshold_manager.record_metric('ensemble', 'latency_ms', latency_ms)
            
            return decision
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error generating signal for {symbol}: {e}", exc_info=True)
            
            # 🔥 PHASE 3C: Record failed signal generation
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if self.health_monitor:
                self.health_monitor.record_signal_attempt(success=False, latency_ms=latency_ms)
                self.health_monitor.record_error()
            
            # 🔥 PHASE 3C-2: Record error
            if self.performance_benchmarker:
                self.performance_benchmarker.record_error('ensemble')
            
            # 🔥 PHASE 3C-3: Record metric for learning
            if self.adaptive_threshold_manager:
                self.adaptive_threshold_manager.record_metric('ensemble', 'error_rate', 1.0)
            
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
                
                # Phase 4F: Run adaptive retraining cycle
                if self.adaptive_retrainer:
                    try:
                        result = self.adaptive_retrainer.run_cycle()
                        if result.get("status") == "success":
                            logger.info(f"[Retrainer] Cycle completed - Models: {result.get('models_retrained', [])}")
                            
                            # Phase 4G: Validate retrained models
                            if self.model_validator:
                                try:
                                    validation_result = self.model_validator.run_validation_cycle()
                                    logger.info(f"[Validator] Validation complete: {validation_result}")
                                except Exception as val_e:
                                    logger.error(f"[Validator] Validation error: {val_e}")
                    except Exception as e:
                        logger.error(f"[Retrainer] Cycle error: {e}")
                
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
            if not self.event_bus:
                # Service not started yet
                dependencies["redis"] = DependencyHealth(
                    status=DependencyStatus.DOWN,
                    error="Service not started - event_bus not initialized"
                )
            elif hasattr(self.event_bus, 'redis') and self.event_bus.redis:
                # EventBus stores Redis client as 'self.redis' (see EventBus.__init__)
                # Use the actual Redis client for health check
                dependencies["redis"] = await check_redis_health(self.event_bus.redis)
            else:
                # EventBus initialized but Redis client missing (should never happen)
                dependencies["redis"] = DependencyHealth(
                    status=DependencyStatus.DOWN,
                    error="Redis client not found in EventBus"
                )
        except Exception as e:
            # Unexpected error during Redis health check
            logger.error(f"[AI-ENGINE] Redis health check failed: {e}", exc_info=True)
            dependencies["redis"] = DependencyHealth(
                status=DependencyStatus.DOWN,
                error=str(e)
            )
        
        # Check EventBus overall
        dependencies["eventbus"] = DependencyHealth(
            status=DependencyStatus.OK if self.event_bus and self._running else DependencyStatus.DOWN
        )
        
        # Check Risk-Safety Service
        # TEMPORARY HOTFIX: Disabled while fixing Exit Brain v3 integration
        # TODO: Re-enable after event_driven_executor Exit Brain integration complete
        # See: AI_RISK_SAFETY_FIX_ACTION_PLAN.md - Phase 2
        dependencies["risk_safety_service"] = DependencyHealth(
            status=DependencyStatus.NOT_APPLICABLE,
            details={"note": "Risk-Safety Service integration pending Exit Brain v3 fix"}
        )
        # Original code (to restore after fix):
        # try:
        #     if self.http_client:
        #         dependencies["risk_safety_service"] = await check_http_endpoint_health(
        #             self.http_client,
        #             f"{settings.RISK_SAFETY_SERVICE_URL}/health"
        #         )
        #     else:
        #         dependencies["risk_safety_service"] = DependencyHealth(
        #             status=DependencyStatus.NOT_APPLICABLE
        #         )
        # except Exception as e:
        #     dependencies["risk_safety_service"] = DependencyHealth(
        #         status=DependencyStatus.DOWN,
        #         error=str(e)
        #     )
        
        # Service-specific metrics
        metrics = {
            "models_loaded": self._models_loaded,
            "signals_generated_total": self._signals_generated,
            "ensemble_enabled": self.ensemble_manager is not None,
            "meta_strategy_enabled": settings.META_STRATEGY_ENABLED,
            "rl_sizing_enabled": settings.RL_SIZING_ENABLED,
            "running": self._running,
            "governance_active": self.supervisor_governance is not None,  # Phase 4D+4E
            "cross_exchange_intelligence": os.getenv("CROSS_EXCHANGE_ENABLED", "false").lower() == "true",  # Phase 4M+
            "intelligent_leverage_v2": os.getenv("INTELLIGENT_LEVERAGE_ENABLED", "true").lower() == "true",  # Phase 4O+
            "rl_position_sizing": os.getenv("RL_POSITION_SIZING_ENABLED", "true").lower() == "true",  # Phase 4O+
            "adaptive_leverage_enabled": os.getenv("ADAPTIVE_LEVERAGE_ENABLED", "true").lower() == "true",  # Phase 4N
        }
        
        # Add cross-exchange stream status if enabled (Phase 4M+)
        if metrics["cross_exchange_intelligence"]:
            try:
                if self.event_bus and hasattr(self.event_bus, 'redis'):
                    # Check normalized stream length
                    stream_len = await self.event_bus.redis.xlen("quantum:stream:exchange.normalized")
                    metrics["cross_exchange_stream"] = {
                        "active": stream_len > 0,
                        "entries": stream_len,
                        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                        "status": "OK" if stream_len > 0 else "NO_DATA"
                    }
                else:
                    metrics["cross_exchange_stream"] = {"status": "REDIS_NOT_AVAILABLE"}
            except Exception as e:
                logger.error(f"[Cross-Exchange] Error checking stream: {e}")
                metrics["cross_exchange_stream"] = {"status": "ERROR", "error": str(e)}
        
        # Add Intelligent Leverage v2 status if enabled (Phase 4O+)
        if metrics["intelligent_leverage_v2"]:
            try:
                if self.event_bus and hasattr(self.event_bus, 'redis'):
                    # Check PnL stream length (where ILFv2 calculations published)
                    pnl_stream_len = await self.event_bus.redis.xlen("quantum:stream:exitbrain.pnl")
                    
                    # Get recent leverages for averaging
                    recent_leverages = []
                    recent_confidences = []
                    recent_divergences = []
                    recent_vols = []
                    
                    if pnl_stream_len > 0:
                        # Read last 100 entries
                        messages = await self.event_bus.redis.xrevrange(
                            "quantum:stream:exitbrain.pnl",
                            count=100
                        )
                        for msg_id, data in messages:
                            if 'dynamic_leverage' in data:
                                recent_leverages.append(float(data['dynamic_leverage']))
                            if 'confidence' in data:
                                recent_confidences.append(float(data['confidence']))
                            if 'exch_divergence' in data:
                                recent_divergences.append(float(data['exch_divergence']))
                            if 'volatility' in data:
                                recent_vols.append(float(data['volatility']))
                    
                    metrics["intelligent_leverage"] = {
                        "enabled": True,
                        "version": "ILFv2",
                        "range": "5-80x",
                        "avg_leverage": round(np.mean(recent_leverages), 2) if recent_leverages else 0.0,
                        "avg_confidence": round(np.mean(recent_confidences), 2) if recent_confidences else 0.0,
                        "avg_divergence": round(np.mean(recent_divergences), 4) if recent_divergences else 0.0,
                        "avg_volatility": round(np.mean(recent_vols), 2) if recent_vols else 0.0,
                        "calculations_total": pnl_stream_len,
                        "cross_exchange_integrated": True,
                        "status": "OK"
                    }
                else:
                    metrics["intelligent_leverage"] = {"status": "REDIS_NOT_AVAILABLE"}
            except Exception as e:
                logger.error(f"[ILF-v2] Error getting status: {e}")
                metrics["intelligent_leverage"] = {"status": "ERROR", "error": str(e)}
        
        # Add RL Position Sizing status if enabled (Phase 4O+)
        if metrics["rl_position_sizing"]:
            try:
                from microservices.rl_sizing_agent.rl_agent import get_rl_agent
                rl_agent = get_rl_agent()
                rl_stats = rl_agent.get_statistics()
                
                metrics["rl_agent"] = {
                    "enabled": True,
                    "policy_version": f"v3.{rl_stats.get('policy_updates', 0)}",
                    "trades_processed": rl_stats.get('trades_processed', 0),
                    "policy_updates": rl_stats.get('policy_updates', 0),
                    "reward_mean": round(rl_stats.get('avg_reward', 0.0), 4),
                    "pytorch_available": rl_stats.get('pytorch_available', False),
                    "experiences_buffered": rl_stats.get('experiences_buffered', 0),
                    "status": "OK" if rl_stats.get('policy_loaded', False) else "NO_POLICY"
                }
            except Exception as e:
                logger.error(f"[RL-Agent] Error getting status: {e}")
                metrics["rl_agent"] = {"status": "ERROR", "error": str(e)}
        
        # Add Exposure Balancer status if enabled (Phase 4P)
        exposure_balancer_enabled = os.getenv("EXPOSURE_BALANCER_ENABLED", "true").lower() == "true"
        metrics["exposure_balancer_enabled"] = exposure_balancer_enabled
        
        if exposure_balancer_enabled:
            try:
                from microservices.exposure_balancer.exposure_balancer import get_exposure_balancer
                balancer = get_exposure_balancer(redis_client=None, config=None)  # Gets existing instance
                balancer_stats = balancer.get_statistics()
                
                metrics["exposure_balancer"] = {
                    "enabled": True,
                    "version": "v1.0",
                    "actions_taken": balancer_stats.get('actions_taken', 0),
                    "actions_by_type": balancer_stats.get('actions_by_type', {}),
                    "last_metrics": {
                        "margin_utilization": round(balancer_stats['last_metrics'].get('margin_utilization', 0.0), 4),
                        "symbol_count": balancer_stats['last_metrics'].get('symbol_count', 0),
                        "avg_confidence": round(balancer_stats['last_metrics'].get('avg_confidence', 0.0), 4),
                        "cross_divergence": round(balancer_stats['last_metrics'].get('cross_divergence', 0.0), 4)
                    },
                    "limits": balancer_stats.get('limits', {}),
                    "status": "OK"
                }
            except Exception as e:
                logger.error(f"[Exposure-Balancer] Error getting status: {e}")
                metrics["exposure_balancer"] = {"status": "ERROR", "error": str(e)}
        
        # Add adaptive leverage status if enabled (Phase 4N)
        if metrics["adaptive_leverage_enabled"]:
            try:
                from backend.domains.exits.exit_brain_v3.v35_integration import get_v35_integration
                v35 = get_v35_integration()
                
                if v35.enabled:
                    pnl_stats = v35.get_pnl_stats()
                    
                    # Check PnL stream length
                    pnl_stream_len = 0
                    if self.event_bus and hasattr(self.event_bus, 'redis'):
                        pnl_stream_len = await self.event_bus.redis.xlen("quantum:stream:exitbrain.pnl")
                    
                    metrics["adaptive_leverage_status"] = {
                        "enabled": True,
                        "models": 1,
                        "volatility_source": "cross_exchange",
                        "avg_pnl_last_20": round(pnl_stats['avg_pnl'], 4),
                        "win_rate": round(pnl_stats['win_rate'], 4),
                        "total_trades": pnl_stats['total_trades'],
                        "pnl_stream_entries": pnl_stream_len,
                        "status": "OK"
                    }
                else:
                    metrics["adaptive_leverage_status"] = {
                        "enabled": False,
                        "status": "DISABLED"
                    }
            except Exception as e:
                logger.error(f"[Adaptive-Leverage] Error getting status: {e}")
                metrics["adaptive_leverage_status"] = {"status": "ERROR", "error": str(e)}
        
        # Add Portfolio Governance status (Phase 4Q)
        portfolio_governance_enabled = os.getenv("PORTFOLIO_GOVERNANCE_ENABLED", "true").lower() == "true"
        metrics["portfolio_governance_enabled"] = portfolio_governance_enabled
        
        if portfolio_governance_enabled:
            try:
                if self.event_bus and hasattr(self.event_bus, 'redis'):
                    # Get current policy and score from Redis
                    policy = await self.event_bus.redis.get("quantum:governance:policy")
                    score = await self.event_bus.redis.get("quantum:governance:score")
                    params_json = await self.event_bus.redis.get("quantum:governance:params")
                    
                    # Get memory stream length
                    memory_samples = await self.event_bus.redis.xlen("quantum:stream:portfolio.memory")
                    
                    # Parse policy parameters
                    params = {}
                    if params_json:
                        import json
                        params = json.loads(params_json.decode('utf-8') if isinstance(params_json, bytes) else params_json)
                    
                    metrics["portfolio_governance"] = {
                        "enabled": True,
                        "policy": policy.decode('utf-8') if isinstance(policy, bytes) else (policy or "BALANCED"),
                        "score": float(score.decode('utf-8') if isinstance(score, bytes) else (score or "0.0")),
                        "memory_samples": memory_samples,
                        "current_parameters": {
                            "max_leverage": params.get("max_leverage", "N/A"),
                            "min_confidence": params.get("min_confidence", "N/A"),
                            "max_concurrent_positions": params.get("max_concurrent_positions", "N/A")
                        },
                        "status": "OK" if memory_samples > 0 else "WARMING_UP"
                    }
                else:
                    metrics["portfolio_governance"] = {"status": "REDIS_NOT_AVAILABLE"}
            except Exception as e:
                logger.error(f"[Portfolio-Governance] Error getting status: {e}")
                metrics["portfolio_governance"] = {"status": "ERROR", "error": str(e)}
        
        # Add Meta-Regime Correlator status (Phase 4R)
        meta_regime_enabled = os.getenv("META_REGIME_ENABLED", "true").lower() == "true"
        metrics["meta_regime_enabled"] = meta_regime_enabled
        
        if meta_regime_enabled:
            try:
                if self.event_bus and hasattr(self.event_bus, 'redis'):
                    # Get preferred regime from Redis
                    preferred_regime = await self.event_bus.redis.get("quantum:governance:preferred_regime")
                    
                    # Get regime statistics
                    regime_stats_json = await self.event_bus.redis.get("quantum:governance:regime_stats")
                    
                    # Get regime stream length
                    regime_samples = await self.event_bus.redis.xlen("quantum:stream:meta.regime")
                    
                    # Parse regime stats
                    regime_stats = {}
                    if regime_stats_json:
                        import json
                        regime_stats = json.loads(regime_stats_json.decode('utf-8') if isinstance(regime_stats_json, bytes) else regime_stats_json)
                    
                    # Get best regime info
                    best_regime = None
                    best_pnl = -float('inf')
                    for regime_name, stats in regime_stats.items():
                        if stats.get('count', 0) >= 5 and stats.get('avg_pnl', -999) > best_pnl:
                            best_regime = regime_name
                            best_pnl = stats['avg_pnl']
                    
                    metrics["meta_regime"] = {
                        "enabled": True,
                        "preferred": preferred_regime.decode('utf-8') if isinstance(preferred_regime, bytes) else (preferred_regime or "UNKNOWN"),
                        "samples": regime_samples,
                        "best_regime": best_regime,
                        "best_pnl": round(best_pnl, 4) if best_regime else None,
                        "regimes_detected": len(regime_stats),
                        "status": "active" if regime_samples > 0 else "warming_up"
                    }
                else:
                    metrics["meta_regime"] = {"status": "redis_not_available"}
            except Exception as e:
                logger.error(f"[Meta-Regime] Error getting status: {e}")
                metrics["meta_regime"] = {"status": "error", "error": str(e)}
        
        # Add governance status if active
        if self.supervisor_governance:
            try:
                governance_status = self.supervisor_governance.get_status()
                metrics["governance"] = governance_status
            except Exception as e:
                logger.error(f"[Governance] Error getting status: {e}")
                metrics["governance"] = {"error": str(e)}
        
        # Add strategic memory feedback (Phase 4S)
        try:
            import json
            if self.event_bus and self.event_bus.redis:
                feedback_data = await self.event_bus.redis.get("quantum:feedback:strategic_memory")
                if feedback_data:
                    feedback_str = feedback_data.decode('utf-8') if isinstance(feedback_data, bytes) else feedback_data
                    feedback = json.loads(feedback_str)
                    metrics["strategic_memory"] = {
                        "status": "active",
                        "preferred_regime": feedback.get("preferred_regime", "UNKNOWN"),
                        "recommended_policy": feedback.get("updated_policy", "CONSERVATIVE"),
                        "confidence_boost": feedback.get("confidence_boost", 0.0),
                        "leverage_hint": feedback.get("leverage_hint", 1.0),
                        "performance": feedback.get("regime_performance", {}),
                        "last_update": feedback.get("timestamp")
                    }
                else:
                    metrics["strategic_memory"] = {
                        "status": "warming_up",
                        "message": "Waiting for first analysis cycle"
                    }
            else:
                metrics["strategic_memory"] = {"status": "redis_not_available"}
        except Exception as e:
            logger.error(f"[StrategicMemory] Error getting feedback: {e}")
            metrics["strategic_memory"] = {"status": "error", "error": str(e)}
        
        # Add strategic evolution status (Phase 4T)
        try:
            import json
            if self.event_bus and self.event_bus.redis:
                # Get selected models
                selected_data = await self.event_bus.redis.get("quantum:evolution:selected")
                rankings_data = await self.event_bus.redis.get("quantum:evolution:rankings")
                mutated_data = await self.event_bus.redis.get("quantum:evolution:mutated")
                retrain_count = await self.event_bus.redis.get("quantum:evolution:retrain_count")
                
                evolution_metrics = {"status": "active"}
                
                if selected_data:
                    selected_str = selected_data.decode('utf-8') if isinstance(selected_data, bytes) else selected_data
                    selected = json.loads(selected_str)
                    evolution_metrics["selected_models"] = selected.get("models", [])
                    evolution_metrics["top_scores"] = selected.get("scores", [])
                
                if rankings_data:
                    rankings_str = rankings_data.decode('utf-8') if isinstance(rankings_data, bytes) else rankings_data
                    rankings = json.loads(rankings_str)
                    evolution_metrics["strategies_evaluated"] = len(rankings)
                
                if mutated_data:
                    mutated_str = mutated_data.decode('utf-8') if isinstance(mutated_data, bytes) else mutated_data
                    mutated = json.loads(mutated_str)
                    evolution_metrics["mutation_count"] = len(mutated)
                
                if retrain_count:
                    count_str = retrain_count.decode('utf-8') if isinstance(retrain_count, bytes) else retrain_count
                    evolution_metrics["total_retrains"] = int(count_str)
                
                metrics["strategic_evolution"] = evolution_metrics
            else:
                metrics["strategic_evolution"] = {"status": "redis_not_available"}
        except Exception as e:
            logger.error(f"[StrategicEvolution] Error getting status: {e}")
            metrics["strategic_evolution"] = {"status": "error", "error": str(e)}
        
        # Add Phase 4U: Model Federation metrics
        try:
            if self.event_bus and self.event_bus.redis:
                consensus_data = await self.event_bus.redis.get("quantum:consensus:signal")
                trust_weights = await self.event_bus.redis.hgetall("quantum:trust:history")
                
                federation_metrics = {"status": "active" if consensus_data else "inactive"}
                
                if consensus_data:
                    consensus_str = consensus_data.decode('utf-8') if isinstance(consensus_data, bytes) else consensus_data
                    consensus = json.loads(consensus_str)
                    federation_metrics["consensus_signal"] = consensus
                    federation_metrics["active_models"] = consensus.get("models_used", 0)
                
                if trust_weights:
                    decoded_weights = {}
                    for k, v in trust_weights.items():
                        key = k.decode('utf-8') if isinstance(k, bytes) else k
                        val = v.decode('utf-8') if isinstance(v, bytes) else v
                        decoded_weights[key] = float(val)
                    federation_metrics["trusted_weights"] = decoded_weights
                
                metrics["model_federation"] = federation_metrics
            else:
                metrics["model_federation"] = {"status": "redis_not_available"}
        except Exception as e:
            logger.error(f"[ModelFederation] Error getting status: {e}")
            metrics["model_federation"] = {"status": "error", "error": str(e)}
        
        # Add adaptive retrainer status if active (Phase 4F)
        if self.adaptive_retrainer:
            try:
                retrainer_status = self.adaptive_retrainer.get_status()
                metrics["adaptive_retrainer"] = retrainer_status
            except Exception as e:
                logger.error(f"[Retrainer] Error getting status: {e}")
        
        # Add model validator status if active (Phase 4G)
        if self.model_validator:
            try:
                validator_status = self.model_validator.get_status()
                metrics["model_validator"] = validator_status
            except Exception as e:
                logger.error(f"[Validator] Error getting status: {e}")
                metrics["model_validator"] = {"error": str(e)}
                metrics["adaptive_retrainer"] = {"error": str(e)}
        
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
