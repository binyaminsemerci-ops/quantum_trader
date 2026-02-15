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
from microservices.ai_engine.rl_influence import RLInfluenceV2
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

# [SimpleCLM] Trade outcome recorder - optional dependency
try:
    from .simple_clm import SimpleCLM
    SIMPLE_CLM_AVAILABLE = True
except ImportError as e:
    SIMPLE_CLM_AVAILABLE = False
    SimpleCLM = None
    print(f"[WARN] SimpleCLM import failed: {e}", flush=True)

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
        
        # [SimpleCLM] Trade outcome recorder (passive collection)
        self.simple_clm = None
        
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
        
        # 🔥 PHASE 3D: AI-Driven Exit Evaluator
        self.exit_evaluator = None  # Intelligent profit-taking decisions
        
        # 🔥 PHASE 2.2: CEO Brain Orchestrator
        self.orchestrator = None  # CEO Brain + Strategy Brain + Risk Brain coordination
        
        # State tracking
        self._running = False
        self._active_symbols: List[str] = []  # Symbols to track for orderbook
        self._orderbook_client = None  # Binance client for orderbook fetching
        self._event_loop_task: Optional[asyncio.Task] = None
        self._regime_update_task: Optional[asyncio.Task] = None
        self._normalized_stream_task: Optional[asyncio.Task] = None
        self._signals_generated = 0
        self._models_loaded = 0
        self._start_time = datetime.now(timezone.utc)
        self._normalized_stream_last_id = "$"
        self._cross_exchange_features: Dict[str, Dict[str, Any]] = {}
        self._xchg_stream = os.getenv("CROSS_EXCHANGE_STREAM", "quantum:stream:exchange.normalized")
        self._xchg_group = os.getenv("CROSS_EXCHANGE_GROUP", "quantum:group:ai-engine:exchange.normalized")
        self._xchg_consumer = os.getenv("CROSS_EXCHANGE_CONSUMER", f"ai-engine-{os.getpid()}")
        self._xchg_stale_sec = int(os.getenv("CROSS_EXCHANGE_STALE_SEC", "120"))
        self._xchg_log_last = 0.0
        self._xchg_stale_log = 0.0
        
        # Testnet mode flag
        self.testnet_mode = os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true"
        
        # Governance tracking (Phase 4D+4E)
        self._governance_predictions: Dict[str, Dict[str, np.ndarray]] = {}  # {symbol: {model: predictions}}
        self._governance_actuals: Dict[str, List[float]] = {}  # {symbol: [actual_prices]}
        self._governance_pnl: Dict[str, Dict[str, float]] = {}  # {symbol: {model: pnl}}
        
        # OHLCV history for indicator calculations (symbol -> list of OHLCV dicts)
        self._price_history: Dict[str, List[float]] = {}  # Kept for backward compatibility
        self._volume_history: Dict[str, List[float]] = {}
        self._ohlcv_history: Dict[str, List[Dict[str, float]]] = {}  # For v5 features
        self._history_max_len = 120  # Keep 2 minutes at 1 tick/sec
        
        # 🔥 RL Confidence Calibration
        self._rl_cal_enabled = os.getenv("RL_CALIBRATION_ENABLED", "true").lower() == "true"
        self._rl_cal_window = int(os.getenv("RL_CALIBRATION_WINDOW_TRADES", "200"))
        self._rl_cal_min_trades = int(os.getenv("RL_CALIBRATION_MIN_TRADES", "40"))
        self._rl_cal_decay = float(os.getenv("RL_CALIBRATION_DECAY", "0.985"))
        self._rl_cal_alpha = float(os.getenv("RL_CALIBRATION_SMOOTH_ALPHA", "0.25"))
        self._rl_cal_floor = float(os.getenv("RL_CALIBRATION_CONF_FLOOR", "0.35"))
        self._rl_cal_ceil = float(os.getenv("RL_CALIBRATION_CONF_CEIL", "0.90"))
        self._rl_cal_key = os.getenv("RL_CALIBRATION_KEY", "quantum:rl:calibration:v1")
        self._rl_cal_stream = os.getenv("RL_CALIBRATION_SOURCE_STREAM", "quantum:stream:trade.closed")
        self._rl_cal_group = os.getenv("RL_CALIBRATION_CONSUMER_GROUP", "quantum:group:ai-engine:trade.closed")
        self._rl_cal_consumer = os.getenv("RL_CALIBRATION_CONSUMER", f"ai-engine-rlcal-{os.getpid()}")
        self._rl_cal_task: Optional[asyncio.Task] = None
        
        # 🔥 RL_PROOF logging throttle (observability only)
        self._rl_proof_last_log: Dict[str, float] = {}  # {symbol: timestamp}
        self._rl_proof_throttle_sec = 30
        
        logger.info("[AI-ENGINE] Service initialized")
    
    async def start(self):
        """Start the service."""
        if self._running:
            logger.warning("[AI-ENGINE] Service already running")
            return
        
        # 🔥 FIX: Set running flag FIRST (before creating background tasks)
        self._running = True
        
        # P1-B: Setup JSON logging with correlation_id
        setup_json_logging('ai_engine')
        logger.info("[AI-ENGINE] JSON logging initialized")
        
        try:
            # Initialize Redis client with connection pool
            import redis.asyncio as redis
            from redis.asyncio.connection import ConnectionPool
            
            logger.info(f"[AI-ENGINE] Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}...")
            
            # Create connection pool with proper limits
            pool = ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                max_connections=50,  # Limit concurrent connections
                socket_timeout=30.0,  # Generous timeout for slow operations
                socket_connect_timeout=15.0,  # Connection timeout
                health_check_interval=30,  # Check connection health every 30s
                decode_responses=False,
            )
            
            self.redis_client = redis.Redis(
                connection_pool=pool,
                retry_on_timeout=True,  # Auto-retry on timeout
            )
            
            # Retry Redis ping on startup
            for attempt in range(3):
                try:
                    await self.redis_client.ping()
                    logger.info("[AI-ENGINE] ✅ Redis connected (pool: max=50)")
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"[AI-ENGINE] Redis ping failed (attempt {attempt+1}/3): {e}, retrying...")
                    await asyncio.sleep(2)
            
            self.rl_influence = RLInfluenceV2(self.redis_client, logger)
            
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

            if settings.CROSS_EXCHANGE_ENABLED:
                await self._ensure_cross_exchange_group()
                if self._normalized_stream_task and not self._normalized_stream_task.done():
                    self._normalized_stream_task.cancel()
                    try:
                        await self._normalized_stream_task
                    except asyncio.CancelledError:
                        pass
                self._normalized_stream_task = asyncio.create_task(self._consume_cross_exchange_stream())
                logger.info("[AI-ENGINE] ✅ Cross-Exchange normalized stream consumer started")
            
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
            
            # 🔥 RL Calibration Consumer
            if self._rl_cal_enabled:
                await self._ensure_rl_calibration_group()
                self._rl_cal_task = asyncio.create_task(self._consume_rl_calibration_stream())
                logger.info(f"[RL-CAL] ✅ Calibration consumer started (stream={self._rl_cal_stream}, group={self._rl_cal_group})")
            
            if settings.REGIME_DETECTION_ENABLED:
                self._regime_update_task = asyncio.create_task(self._regime_update_loop())
            
            # [SimpleCLM] Start trade outcome recorder background tasks
            if self.simple_clm:
                await self.simple_clm.start()
                logger.info("[sCLM] ✅ Background monitoring started")
            
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
        
        # Stop RL calibration consumer
        if self._rl_cal_task and not self._rl_cal_task.done():
            self._rl_cal_task.cancel()
            try:
                await self._rl_cal_task
            except asyncio.CancelledError:
                pass
            logger.info("[RL-CAL] Consumer stopped")
        
        # Stop EventBus consumer
        if self.event_bus:
            await self.event_bus.stop()
            logger.info("[AI-ENGINE] EventBus consumer stopped")
        
        # [SimpleCLM] Stop background monitoring
        if self.simple_clm:
            await self.simple_clm.stop()
            logger.info("[sCLM] Background monitoring stopped")
        
        # Cancel background tasks
        for task in [self._event_loop_task, self._regime_update_task, self._normalized_stream_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._normalized_stream_task = None
        
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
            logger.info(f"[DEPLOY-DEBUG] settings.ENSEMBLE_MODELS = {getattr(settings, 'ENSEMBLE_MODELS', 'ATTR_MISSING')}")
            logger.info(f"[DEPLOY-DEBUG] bool(settings.ENSEMBLE_MODELS) = {bool(getattr(settings, 'ENSEMBLE_MODELS', False))}")
            
            if settings.ENSEMBLE_MODELS:
                logger.info(f"[AI-ENGINE] Loading ensemble: {settings.ENSEMBLE_MODELS}")
                try:
                    from ai_engine.ensemble_manager import EnsembleManager
                    logger.info("[DEPLOY-DEBUG] EnsembleManager imported successfully")
                    
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
                logger.info("[AI-ENGINE] 🌐 Preparing Cross-Exchange normalized stream consumer (Phase 1)...")
                try:
                    self.cross_exchange_aggregator = None
                    self._cross_exchange_features.clear()
                    self._normalized_stream_last_id = "$"
                    self._models_loaded += 1
                    # Use _active_symbols from Universe Service (loaded later in PHASE 2B)
                    logger.info(
                        f"[PHASE 1] Cross-Exchange stream ready (symbols will load from Universe Service), "
                        f"{len(settings.CROSS_EXCHANGE_EXCHANGES)} exchanges"
                    )
                except Exception as e:
                    logger.warning(f"[AI-ENGINE] ⚠️ Cross-Exchange stream setup failed: {e}")
            
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
            
            # [SimpleCLM] Initialize trade outcome recorder (passive collection)
            if SIMPLE_CLM_AVAILABLE and os.getenv("SIMPLE_CLM_ENABLED", "true").lower() == "true":
                logger.info("[AI-ENGINE] 📝 Initializing SimpleCLM (trade outcome recorder)...")
                try:
                    storage_path = os.getenv("SIMPLE_CLM_STORAGE", "/home/qt/quantum_trader/data/clm_trades.jsonl")
                    self.simple_clm = SimpleCLM(
                        storage_path=storage_path,
                        win_threshold=float(os.getenv("SIMPLE_CLM_WIN_THRESHOLD", "0.5")),
                        loss_threshold=float(os.getenv("SIMPLE_CLM_LOSS_THRESHOLD", "-0.5")),
                        starvation_hours=float(os.getenv("SIMPLE_CLM_STARVATION_HOURS", "1.0")),
                        stats_log_interval_seconds=int(os.getenv("SIMPLE_CLM_STATS_INTERVAL", "300"))
                    )
                    self._models_loaded += 1
                    logger.info("[sCLM] ✅ Initialized: storage={storage_path}")
                except Exception as e:
                    logger.error(f"[sCLM] ❌ Initialization failed: {e}", exc_info=True)
                    self.simple_clm = None
            else:
                logger.info("[sCLM] Disabled or not available")
            
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
            
            # 🔥 PHASE 3D: AI-Driven Exit Evaluator
            logger.info("[AI-ENGINE] 🎯 Initializing Exit Evaluator (Phase 3D)...")
            try:
                from microservices.ai_engine.exit_evaluator import ExitEvaluator
                
                self.exit_evaluator = ExitEvaluator(
                    regime_detector=self.regime_detector,
                    vse=self.volatility_structure_engine,
                    ensemble=self.ensemble_manager
                )
                
                logger.info("[PHASE 3D] ✅ Exit Evaluator initialized")
                logger.info("[PHASE 3D] 🧠 Features: Intelligent profit-taking, Multi-factor scoring, Dynamic percentages")
            except Exception as e:
                logger.warning(f"[AI-ENGINE] ⚠️ Exit Evaluator failed: {e}")
                self.exit_evaluator = None
            
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
            self._ohlcv_history[symbol] = []
            logger.info(f"[AI-ENGINE] 🆕 Creating new price history for {symbol}")
        
        self._price_history[symbol].append(price)
        self._volume_history[symbol].append(volume)
        
        # Update OHLCV data for v5 features
        if self._ohlcv_history[symbol]:
            last_candle = self._ohlcv_history[symbol][-1]
            last_candle["high"] = max(last_candle["high"], price)
            last_candle["low"] = min(last_candle["low"], price)
            last_candle["close"] = price
            last_candle["volume"] += volume
        else:
            self._ohlcv_history[symbol].append({
                "open": price, "high": price, "low": price, "close": price, "volume": volume
            })
        
        # Trim to max length
        if len(self._price_history[symbol]) > self._history_max_len:
            self._price_history[symbol] = self._price_history[symbol][-self._history_max_len:]
            self._volume_history[symbol] = self._volume_history[symbol][-self._history_max_len:]
            self._ohlcv_history[symbol] = self._ohlcv_history[symbol][-self._history_max_len:]
        
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

    async def _ensure_cross_exchange_group(self):
        """Create consumer group for normalized stream if missing."""
        try:
            await self.redis_client.xgroup_create(
                name=self._xchg_stream,
                groupname=self._xchg_group,
                id="$",
                mkstream=True,
            )
            logger.info(
                f"[AI-ENGINE] ✅ Cross-exchange group created: stream={self._xchg_stream}, group={self._xchg_group}"
            )
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.info(
                    f"[AI-ENGINE] Cross-exchange group already exists: stream={self._xchg_stream}, group={self._xchg_group}"
                )
            else:
                logger.warning(
                    f"[AI-ENGINE] Cross-exchange group create error: {e} (stream={self._xchg_stream}, group={self._xchg_group})"
                )

    async def _consume_cross_exchange_stream(self):
        """Continuously consume normalized cross-exchange data from Redis."""
        stream_key = self._xchg_stream
        group = self._xchg_group
        consumer = self._xchg_consumer
        last_seen_id = self._normalized_stream_last_id or "$"
        logger.warning(
            f"[AI-ENGINE] 📡 Starting cross-exchange consumer at {stream_key} (group={group}, consumer={consumer}, last_seen={last_seen_id})"
        )
        
        # Create a dedicated Redis connection for this consumer
        # This avoids connection pool contention with other consumers
        import redis.asyncio as redis
        from .config import settings
        dedicated_redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=False,
            socket_timeout=30.0,
            socket_connect_timeout=15.0,
            single_connection_client=True,  # Use single dedicated connection
        )
        
        try:
            await dedicated_redis.ping()
            logger.warning("[AI-ENGINE] ✅ Dedicated Redis connection established for cross-exchange consumer")
        except Exception as e:
            logger.error(f"[AI-ENGINE] Failed to establish dedicated Redis connection: {e}")
            return
        
        logger.warning(f"[AI-ENGINE] TRACE-1: _running={self._running}, entering loop...")
        
        loop_count = 0
        while self._running:
            loop_count += 1
            logger.warning(f"[AI-ENGINE] TRACE-LOOP: Loop #{loop_count}")
            try:
                # Use dedicated Redis connection with blocking read
                messages = await dedicated_redis.xreadgroup(
                    groupname=group,
                    consumername=consumer,
                    streams={stream_key: ">"},
                    count=50,
                    block=2000,  # 2 second block - dedicated connection won't starve
                )
                
                if not messages:
                    continue
                    
                logger.warning(f"[AI-ENGINE] 📨 XREADGROUP returned: {len(messages)} streams")

                ack_ids: List[Any] = []
                consumed = 0
                last_symbol: Optional[str] = None

                for _, entries in messages:
                    for message_id, raw_data in entries:
                        last_seen_id = message_id.decode("utf-8") if isinstance(message_id, bytes) else message_id

                        parsed = self._parse_cross_exchange_entry(raw_data)
                        if not parsed:
                            continue

                        parsed["last_id"] = last_seen_id
                        parsed["received_at"] = datetime.utcnow()
                        symbol = parsed["symbol"]
                        last_symbol = symbol
                        self._cross_exchange_features[symbol] = parsed
                        ack_ids.append(message_id)
                        consumed += 1

                        tick_event = {
                            "symbol": symbol,
                            "price": parsed["avg_price"],
                            "volume": 0.0,
                            "source": "exchange.normalized",
                            "num_exchanges": parsed["num_exchanges"],
                            "cross_exchange_timestamp": parsed["source_timestamp"],
                        }

                        try:
                            await self._handle_market_tick(tick_event)
                        except Exception as tick_error:
                            logger.error(f"[AI-ENGINE] Error handling normalized tick: {tick_error}", exc_info=True)

                if ack_ids:
                    try:
                        await dedicated_redis.xack(stream_key, group, *ack_ids)
                    except Exception as ack_error:
                        logger.warning(f"[AI-ENGINE] XACK failed for {len(ack_ids)} ids: {ack_error}")
                if last_seen_id:
                    self._normalized_stream_last_id = last_seen_id

                now_ts = time.time()
                latest = self._cross_exchange_features.get(last_symbol) if (consumed and last_symbol) else None
                if consumed and latest and (now_ts - self._xchg_log_last) >= 5:
                    self._xchg_log_last = now_ts
                    cache_size = len(self._cross_exchange_features)
                    logger.info(
                        f"[AI-ENGINE] xchg-consumed {consumed} msgs, last_id={last_seen_id}, "
                        f"cache_size={cache_size}, divergence={latest.get('price_divergence', 0.0):.5f}, "
                        f"spread_bps={latest.get('xchg_spread_bps', 0.0):.2f}"
                    )

            except asyncio.CancelledError:
                logger.info("[AI-ENGINE] Cross-exchange consumer cancelled")
                break
            except asyncio.TimeoutError:
                logger.warning("[AI-ENGINE] XREADGROUP timed out - retrying...")
                continue
            except Exception as e:
                logger.error(f"[AI-ENGINE] Cross-exchange consumer failure: {e}", exc_info=True)
                await asyncio.sleep(2)

        # Cleanup dedicated connection
        try:
            await dedicated_redis.close()
            logger.info("[AI-ENGINE] Dedicated Redis connection closed")
        except Exception:
            pass
        logger.info("[AI-ENGINE] Cross-exchange consumer stopped")

    def _parse_cross_exchange_entry(self, raw_data: Dict[Any, Any]) -> Optional[Dict[str, Any]]:
        """Decode and enrich a Redis stream entry from exchange.normalized."""
        decoded = {}
        for key, value in raw_data.items():
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            value_str = value.decode("utf-8") if isinstance(value, bytes) else value
            decoded[key_str] = value_str

        if "init" in decoded:
            return None

        symbol = decoded.get("symbol")
        avg_price = self._safe_float(decoded.get("avg_price"))
        if not symbol or avg_price is None or avg_price <= 0:
            return None

        price_divergence = self._safe_float(decoded.get("price_divergence")) or 0.0
        num_exchanges = self._safe_int(decoded.get("num_exchanges")) or 0
        binance_price = self._safe_float(decoded.get("binance_price"))
        bybit_price = self._safe_float(decoded.get("bybit_price"))
        coinbase_price = self._safe_float(decoded.get("coinbase_price"))
        funding_delta = self._safe_float(decoded.get("funding_delta")) or 0.0
        source_timestamp = self._safe_int(decoded.get("timestamp")) or int(time.time())

        volatility_factor = (price_divergence / avg_price) if avg_price else 0.0
        if not np.isfinite(volatility_factor):
            volatility_factor = 0.0

        lead_lag = 0.0
        if binance_price is not None and bybit_price is not None and avg_price:
            lead_lag = (binance_price - bybit_price) / avg_price if avg_price else 0.0
            if not np.isfinite(lead_lag):
                lead_lag = 0.0

        spread_abs = None
        spread_bps = None
        if binance_price is not None and bybit_price is not None and avg_price:
            spread_abs = binance_price - bybit_price
            spread_bps = (spread_abs / avg_price) * 10000 if avg_price else 0.0
            if not np.isfinite(spread_bps):
                spread_bps = 0.0
            if not np.isfinite(spread_abs):
                spread_abs = 0.0

        return {
            "symbol": symbol,
            "avg_price": avg_price,
            "price_divergence": price_divergence,
            "num_exchanges": num_exchanges,
            "binance_price": binance_price,
            "bybit_price": bybit_price,
            "coinbase_price": coinbase_price,
            "funding_delta": funding_delta,
            "volatility_factor": volatility_factor,
            "exchange_divergence": price_divergence,
            "lead_lag_score": lead_lag,
            "source_timestamp": source_timestamp,
            "xchg_avg_price": avg_price,
            "xchg_price_divergence": price_divergence,
            "xchg_num_exchanges": num_exchanges,
            "xchg_binance_price": binance_price if binance_price is not None else 0.0,
            "xchg_bybit_price": bybit_price if bybit_price is not None else 0.0,
            "xchg_spread_abs": spread_abs if spread_abs is not None else 0.0,
            "xchg_spread_bps": spread_bps if spread_bps is not None else 0.0,
        }

    def _safe_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "null", "none"}:
                return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() in {"null", "none"}:
                return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _is_cross_exchange_data_fresh(self, data: Dict[str, Any], max_age_seconds: int = 120) -> bool:
        received_at = data.get("received_at")
        if received_at is None:
            return False
        age = (datetime.utcnow() - received_at).total_seconds()
        return age <= max_age_seconds
    
    def _apply_cross_exchange_features(self, symbol: str, features: dict) -> dict:
        """
        Merge cross-exchange normalized features into the feature map, regardless of ensemble/fallback path.
        """
        try:
            cross = self._cross_exchange_features.get(symbol)
            if not cross:
                return features

            received_at = cross.get("received_at")
            if not received_at or not self._is_cross_exchange_data_fresh(cross, self._xchg_stale_sec):
                return features

            # Canonical key used by trade.intent: exchange_divergence must be non-zero when xchg is present
            features["exchange_divergence"] = float(cross.get("price_divergence") or 0.0)

            # Optional richer signals
            features["xchg_avg_price"] = float(cross.get("avg_price") or 0.0)
            features["xchg_price_divergence"] = float(cross.get("price_divergence") or 0.0)
            features["xchg_num_exchanges"] = int(cross.get("num_exchanges") or 0)

            # Spread metrics if available
            if "spread_abs" in cross:
                features["xchg_spread_abs"] = float(cross.get("spread_abs") or 0.0)
            if "spread_bps" in cross:
                features["xchg_spread_bps"] = float(cross.get("spread_bps") or 0.0)

            # Optional prices (string/None safe)
            features["xchg_binance_price"] = cross.get("binance_price")
            features["xchg_bybit_price"] = cross.get("bybit_price")
            features["xchg_coinbase_price"] = cross.get("coinbase_price")

            logger.debug(f"[AI-ENGINE] Cross-Exchange merged for {symbol}: div={features['exchange_divergence']:.4f}")
            return features

        except Exception as e:
            logger.warning(f"[AI-ENGINE] xchg merge failed for {symbol}: {e}")
            return features

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
    
    def _decode_redis_payload(self, raw: dict) -> dict:
        """
        Robust decoder for Redis Stream payloads (bytes keys/values → str).
        
        Handles trade.closed events that arrive with bytes keys/values from Redis.
        Must be called BEFORE any field access to ensure data integrity for CLM.
        
        Args:
            raw: Redis message_data with bytes keys/values or already-decoded dict
        
        Returns:
            Decoded dict with string keys and string values
        """
        decoded = {}
        for k, v in raw.items():
            # Decode key if bytes
            key = k.decode('utf-8') if isinstance(k, bytes) else k
            # Decode value if bytes
            val = v.decode('utf-8') if isinstance(v, bytes) else v
            decoded[key] = val
        return decoded
    
    async def _handle_trade_closed(self, event_data: Dict[str, Any]):
        """
        Handle trade.closed event from execution-service.
        
        This is used for continuous learning:
        - Update Meta-Strategy Q-values
        - Update RL Sizing Q-values
        - Feed to Continuous Learning Manager
        - Track PnL for Model Supervisor & Governance (Phase 4D+4E)
        """
        import json  # For payload parsing
        
        try:
            # 🔥 CRITICAL FIX: Robust bytes→str decoding for trade.closed events
            # Problem: Redis XADD stores all keys/values as bytes. When EventBus reads
            # with XREADGROUP, message_data contains {b'entry_price': b'0.06226', ...}.
            # Previous logic failed to decode properly, resulting in empty dict {},
            # which caused CLM to reject all trades (entry_price=0.0).
            
            # Step 1: Immediate bytes decoding if raw Redis format detected
            if event_data and any(isinstance(k, bytes) for k in event_data.keys()):
                # Raw bytes keys found - decode everything immediately
                event_data = self._decode_redis_payload(event_data)
                logger.debug(f"[REDIS-DECODE] ✅ Decoded {len(event_data)} fields from bytes")
            
            # Step 2: Early exit if empty (failed upstream)
            if not event_data or len(event_data) == 0:
                logger.warning("[AI-ENGINE] ⚠️ Empty trade.closed event - skipping")
                return
            
            # Step 3: Handle EventBus wrapper format (payload field with JSON string)
            if "payload" in event_data:
                payload_json = event_data.get("payload", "{}")
                if isinstance(payload_json, bytes):
                    payload_json = payload_json.decode('utf-8')
                event_data = json.loads(payload_json) if payload_json and payload_json != "{}" else {}
                if event_data:
                    logger.debug("[REDIS-DECODE] ✅ Extracted from EventBus payload wrapper")
            
            # Step 4: Final validation - must have symbol field for valid trade
            if not event_data or "symbol" not in event_data:
                logger.warning("[AI-ENGINE] ⚠️ trade.closed missing 'symbol' - skipping")
                return
            
            trade_id = event_data.get("trade_id")
            symbol = event_data.get("symbol", "unknown")
            pnl_percent = float(event_data.get("pnl_percent", 0.0))
            model = event_data.get("model_id", event_data.get("model", "unknown"))  # Try model_id first, fallback to model
            strategy = event_data.get("strategy")
            entry_price = float(event_data.get("entry_price", 0.0))
            exit_price = float(event_data.get("exit_price", 0.0))
            
            logger.info(f"[AI-ENGINE] Trade closed: {trade_id} | {symbol} | Entry={entry_price} Exit={exit_price} | PnL={pnl_percent:.2f}% | model={model}")
            
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
            
            # [SimpleCLM] Record trade outcome (passive collection, validation, labeling, persistence)
            if self.simple_clm:
                try:
                    # Build complete trade record with all required fields
                    trade_record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "side": event_data.get("action", "BUY").upper(),  # BUY/SELL
                        "entry_price": float(event_data.get("entry_price", 0.0)),
                        "exit_price": float(event_data.get("exit_price", 0.0)),
                        "pnl_percent": float(pnl_percent),
                        "confidence": float(event_data.get("confidence", 0.5)),
                        "model_id": f"{model}_{strategy}",
                        "strategy_id": strategy,
                        "position_size": float(event_data.get("position_size", 0.0)),
                        "exit_reason": event_data.get("exit_reason", "unknown")
                    }
                    
                    # Debug: Verify decoded data integrity before CLM validation
                    logger.debug(f"[sCLM] Submitting trade: entry_price={trade_record['entry_price']}, exit_price={trade_record['exit_price']}, pnl={trade_record['pnl_percent']}")
                    
                    # Record trade (validates, labels, persists)
                    success, error = self.simple_clm.record_trade(trade_record)
                    
                    if success:
                        logger.debug(f"[sCLM] ✅ Recorded: {symbol} PnL={pnl_percent:+.2f}%")
                    else:
                        logger.warning(f"[sCLM] ❌ Rejected trade: {error}")
                
                except Exception as sclm_error:
                    logger.error(f"[sCLM] ❌ Recording failed: {sclm_error}", exc_info=True)
            
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
    
    def _calculate_v5_features(self, symbol: str) -> Dict[str, float]:
        """
        Calculate all 18 features required by XGBoost v5.
        
        Returns dict with keys:
        price_change, high_low_range, volume_change, volume_ma_ratio,
        ema_10, ema_20, ema_50, rsi_14, macd, macd_signal, macd_hist,
        bb_position, volatility_20, momentum_10, momentum_20,
        ema_10_20_cross, ema_10_50_cross, volume_ratio
        """
        ohlcv = self._ohlcv_history.get(symbol, [])
        prices = self._price_history.get(symbol, [])
        volumes = self._volume_history.get(symbol, [])
        
        if len(prices) < 2:
            # Return neutral values if insufficient data
            return {
                "price_change": 0.0, "high_low_range": 0.0, "volume_change": 0.0,
                "volume_ma_ratio": 1.0, "ema_10": prices[-1] if prices else 0.0,
                "ema_20": prices[-1] if prices else 0.0, "ema_50": prices[-1] if prices else 0.0,
                "rsi_14": 50.0, "macd": 0.0, "macd_signal": 0.0, "macd_hist": 0.0,
                "bb_position": 0.5, "volatility_20": 0.0, "momentum_10": 0.0,
                "momentum_20": 0.0, "ema_10_20_cross": 0.0, "ema_10_50_cross": 0.0,
                "volume_ratio": 1.0
            }
        
        current_price = prices[-1]
        
        # Helper: EMA calculation
        def ema(data: List[float], period: int) -> float:
            if len(data) < period:
                return data[-1] if data else 0.0
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        # 1. price_change (1-period)
        price_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) >= 2 else 0.0
        
        # 2. high_low_range
        if ohlcv and len(ohlcv) > 0:
            last_candle = ohlcv[-1]
            high_low_range = (last_candle["high"] - last_candle["low"]) / current_price if current_price > 0 else 0.0
        else:
            high_low_range = 0.0
        
        # 3. volume_change
        volume_change = ((volumes[-1] - volumes[-2]) / volumes[-2]) * 100 if len(volumes) >= 2 and volumes[-2] > 0 else 0.0
        
        # 4. volume_ma_ratio (current volume / MA(20))
        if len(volumes) >= 20:
            volume_ma = sum(volumes[-20:]) / 20
            volume_ma_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1.0
        else:
            volume_ma_ratio = 1.0
        
        # 5-7. EMAs (10, 20, 50)
        ema_10 = ema(prices, 10)
        ema_20 = ema(prices, 20)
        ema_50 = ema(prices, 50)
        
        # 8. RSI (14)
        rsi_14 = self._calculate_rsi(prices, 14)
        
        # 9-11. MACD (12, 26, 9)
        if len(prices) >= 26:
            fast_ema = ema(prices, 12)
            slow_ema = ema(prices, 26)
            macd = fast_ema - slow_ema
            
            # MACD signal line (9-period EMA of MACD)
            # For simplicity, approximate with single value (would need history of MACD values)
            macd_signal = macd * 0.9  # Simplified approximation
            macd_hist = macd - macd_signal
        else:
            macd = macd_signal = macd_hist = 0.0
        
        # 12. bb_position (Bollinger Band position)
        if len(prices) >= 20:
            bb_ma = sum(prices[-20:]) / 20
            bb_std = (sum([(p - bb_ma) ** 2 for p in prices[-20:]]) / 20) ** 0.5
            bb_upper = bb_ma + (2 * bb_std)
            bb_lower = bb_ma - (2 * bb_std)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        else:
            bb_position = 0.5
        
        # 13. volatility_20 (20-period rolling std of returns)
        if len(prices) >= 21:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-20, 0)]
            volatility_20 = (sum([r ** 2 for r in returns]) / len(returns)) ** 0.5
        else:
            volatility_20 = 0.0
        
        # 14-15. momentum (10 and 20 period)
        momentum_10 = self._calculate_momentum(prices, 10)
        momentum_20 = self._calculate_momentum(prices, 20)
        
        # 16-17. EMA crosses
        ema_10_20_cross = (ema_10 - ema_20) / ema_20 if ema_20 > 0 else 0.0
        ema_10_50_cross = (ema_10 - ema_50) / ema_50 if ema_50 > 0 else 0.0
        
        # 18. volume_ratio (same as volume_ma_ratio, kept for compatibility)
        volume_ratio = volume_ma_ratio
        
        return {
            "price_change": price_change,
            "high_low_range": high_low_range,
            "volume_change": volume_change,
            "volume_ma_ratio": volume_ma_ratio,
            "ema_10": ema_10,
            "ema_20": ema_20,
            "ema_50": ema_50,
            "rsi_14": rsi_14,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "bb_position": bb_position,
            "volatility_20": volatility_20,
            "momentum_10": momentum_10,
            "momentum_20": momentum_20,
            "ema_10_20_cross": ema_10_20_cross,
            "ema_10_50_cross": ema_10_50_cross,
            "volume_ratio": volume_ratio
        }
    
    async def _refresh_emergency_stop_cache(self):
        """Background task to refresh emergency stop cache without blocking signal generation."""
        try:
            logger.info("[AI-ENGINE] 🔄 Background: Refreshing emergency stop cache...")
            # This may block, but it's in a separate task so it won't block signal generation
            emergency_stop = await self.redis_client.get("trading:emergency_stop")
            self._emergency_stop_cache = {
                'value': emergency_stop,
                'last_check': time.time(),
                'updating': False
            }
            logger.info(f"[AI-ENGINE] ✅ Emergency stop cache refreshed: {emergency_stop}")
        except Exception as e:
            logger.warning(f"[AI-ENGINE] ⚠️ Emergency stop cache refresh failed: {e}")
            self._emergency_stop_cache['updating'] = False
    
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
        # Default risk context to avoid unbound errors before predictors run
        risk_signal = None
        risk_multiplier = 1.0
        
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
            
            # Step 0: ESS Kill Switch - Check emergency stop (cached, non-blocking)
            # NOTE: Direct Redis get() blocks indefinitely on aioredis even with socket_timeout
            # So we cache the value and refresh asynchronously - NEVER block signal generation
            now = time.time()
            if not hasattr(self, '_emergency_stop_cache'):
                self._emergency_stop_cache = {'value': None, 'last_check': 0, 'updating': False}
            
            # Start background refresh if needed (non-blocking)
            if now - self._emergency_stop_cache['last_check'] > 30 and not self._emergency_stop_cache['updating']:
                self._emergency_stop_cache['updating'] = True
                asyncio.create_task(self._refresh_emergency_stop_cache())
            
            # Always use cached value (never block)
            emergency_stop = self._emergency_stop_cache['value']
            
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
                # Calculate v5 features for fallback logic
                features = self._calculate_v5_features(symbol)
                features["price"] = current_price
                # Apply cross-exchange features (critical fix)
                features = self._apply_cross_exchange_features(symbol, features)
            else:
                logger.debug(f"[AI-ENGINE] Running ensemble for {symbol}...")
                
                # Calculate real technical indicators from price history
                prices = self._price_history.get(symbol, [])
                volumes = self._volume_history.get(symbol, [])
                
                logger.info(f"[AI-ENGINE] 📊 Price history: {symbol} has {len(prices)} data points")
                
                # Calculate all 18 v5 features
                features = self._calculate_v5_features(symbol)
                
                # Add current price for backward compatibility
                features["price"] = current_price
                
                # Apply cross-exchange features (unified with fallback path)
                features = self._apply_cross_exchange_features(symbol, features)
                
                # Set defaults for any missing xchg features
                features.setdefault("exchange_divergence", 0.0)
                features.setdefault("xchg_avg_price", 0.0)
                features.setdefault("xchg_price_divergence", 0.0)
                features.setdefault("xchg_num_exchanges", 0)
                features.setdefault("xchg_binance_price", None)
                features.setdefault("xchg_bybit_price", None)
                features.setdefault("xchg_coinbase_price", None)
                features.setdefault("xchg_spread_abs", 0.0)
                features.setdefault("xchg_spread_bps", 0.0)
                
                # 🔥 PHASE 1: Add Funding Rate Features (5s timeout)
                if self.funding_rate_filter:
                    try:
                        funding_data = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.funding_rate_filter.get_funding_features,
                                symbol
                            ),
                            timeout=5.0
                        )
                        if funding_data:
                            features["funding_rate"] = funding_data.get("current_funding", 0.0)
                            features["funding_delta"] = funding_data.get("funding_delta", 0.0)
                            features["crowded_side_score"] = funding_data.get("crowd_bias", 0.0)
                            logger.info(f"[PHASE 1] Funding: rate={features['funding_rate']:.5f}, "
                                      f"crowd_bias={features['crowded_side_score']:.2f}")
                    except asyncio.TimeoutError:
                        logger.warning(f"[PHASE 1] Funding rate timeout (5s) for {symbol}")
                    except Exception as e:
                        logger.warning(f"[PHASE 1] Funding rate feature extraction failed: {e}")
                
                # 🔥 PHASE 2D: Add Volatility Structure Features (5s timeout)
                if self.volatility_structure_engine:
                    try:
                        vol_analysis = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.volatility_structure_engine.get_complete_volatility_analysis,
                                symbol
                            ),
                            timeout=5.0
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
                    except asyncio.TimeoutError:
                        logger.warning(f"[PHASE 2D] Volatility timeout (5s) for {symbol}")
                    except Exception as vol_error:
                        logger.warning(f"[PHASE 2D] Volatility feature extraction failed: {vol_error}")
                
                # 🔥 PHASE 2B: Add Orderbook Imbalance Features (5s timeout)
                if self.orderbook_imbalance:
                    try:
                        orderbook_metrics = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.orderbook_imbalance.get_metrics,
                                symbol
                            ),
                            timeout=5.0
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
                    except asyncio.TimeoutError:
                        logger.warning(f"[PHASE 2B] Orderbook timeout (5s) for {symbol}")
                    except Exception as ob_error:
                        logger.warning(f"[PHASE 2B] Orderbook feature extraction failed: {ob_error}")
                
                logger.info(f"[AI-ENGINE] Features for {symbol}: RSI={features['rsi_14']:.1f}, "
                            f"MACD={features['macd']:.4f}, VolumeRatio={features['volume_ratio']:.2f}, "
                            f"Momentum={features['momentum_10']:.2f}%")
                
                # 🔥 PHASE 3A: Predict risk mode (5s timeout)
                risk_signal = None
                risk_multiplier = 1.0
                if self.risk_mode_predictor:
                    try:
                        risk_signal = await asyncio.wait_for(
                            asyncio.to_thread(
                                self.risk_mode_predictor.predict_risk_mode,
                                symbol=symbol,
                                current_price=current_price,
                                market_conditions={}  # TODO: Add BTC dominance, funding rates, fear/greed
                            ),
                            timeout=5.0
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
                    except asyncio.TimeoutError:
                        logger.warning(f"[PHASE 3A] Risk prediction timeout (5s) for {symbol}")
                        risk_multiplier = 1.0
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
            
            # FALLBACK: If ML models return HOLD (regardless of confidence), use rule-based signals for exploration
            # EXPANDED THRESHOLD: Changed from 0.65 to 0.98 to enable trades during high-confidence HOLD periods
            # This allows CLM data collection even when ensemble is conservative
            fallback_triggered = False
            if action == "HOLD" and 0.50 <= ensemble_confidence <= 0.98:
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
                    # EXPLORATION MODE: Relaxed RSI thresholds to enable data collection
                    # RSI < 45 (was 35) + MACD > -0.002 (was -0.001) = BUY
                    # RSI > 55 (was 65) + MACD < 0.002 (was 0.001) = SELL
                    if rsi < 45 and macd > -0.002:  # Moderately oversold + neutral/bullish momentum
                        action = "BUY"
                        ensemble_confidence = 0.72
                        fallback_triggered = True
                        logger.info(f"[AI-ENGINE] 🔥 FALLBACK BUY signal: {symbol} RSI={rsi:.1f}, MACD={macd:.4f}")
                    elif rsi > 55 and macd < 0.002:  # Moderately overbought + neutral/bearish momentum
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
            
            # 🔥 PHASE 1: Apply RL Signal Manager Confidence Calibration (5s timeout)
            if self.rl_signal_manager and not fallback_triggered:
                try:
                    calibrated_confidence = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.rl_signal_manager.calibrate_confidence,
                            symbol=symbol,
                            raw_confidence=ensemble_confidence,
                            action=action
                        ),
                        timeout=5.0
                    )
                    logger.info(f"[PHASE 1] RL Calibration: {ensemble_confidence:.3f} → {calibrated_confidence:.3f}")
                    ensemble_confidence = calibrated_confidence
                except asyncio.TimeoutError:
                    logger.warning(f"[PHASE 1] RL Calibration timeout (5s) for {symbol}")
                except Exception as e:
                    logger.warning(f"[PHASE 1] Confidence calibration failed: {e}")
            
            # 🔥 PHASE 1: Check Funding Rate Filter (5s timeout)
            if self.funding_rate_filter:
                try:
                    funding_check = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.funding_rate_filter.should_block_trade_simple,
                            symbol
                        ),
                        timeout=5.0
                    )
                    if funding_check.get("blocked", False):
                        logger.warning(
                            f"[PHASE 1] 🚫 FUNDING FILTER BLOCKED: {symbol} "
                            f"(rate={funding_check.get('funding_rate', 0):.5f}, "
                            f"reason={funding_check.get('reason', 'unknown')})"
                        )
                        return None
                except asyncio.TimeoutError:
                    logger.warning(f"[PHASE 1] Funding check timeout (5s) for {symbol} - continuing anyway")
                except Exception as e:
                    logger.warning(f"[PHASE 1] Funding filter check failed: {e}")
            
            # 🔥 PHASE 1: Check Drift Detection (block if model drifted) - WITH TIMEOUT
            if self.drift_detector:
                try:
                    # 🔥 TIMEOUT: Max 5 seconds for drift check to prevent hangs
                    drift_status = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.drift_detector.check_drift,
                            symbol=symbol,
                            features=features,
                            prediction=ensemble_confidence
                        ),
                        timeout=5.0
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
                            # Continue processing signal on testnet
                        else:
                            logger.warning(
                                f"[PHASE 1] 🚫 DRIFT DETECTED: {symbol} "
                                f"(severity={drift_status.get('severity')}, "
                                f"psi={drift_status.get('psi', 0):.3f}) - Signal blocked"
                            )
                            # Trigger retraining if CLM available
                            if self.adaptive_retrainer:
                                logger.info(f"[PHASE 1] Triggering retrain due to drift...")
                            return None  # Only block on mainnet
                except asyncio.TimeoutError:
                    logger.error(f"[PHASE 1] ⏰ Drift detection TIMEOUT (5s) for {symbol} - allowing signal")
                    # Continue processing signal if drift check times out (fail-open for liveness)
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
            
            # Step 3: RL Position Sizing + TradingMathematician
            # 🔥 AKTIVERT: Math AI beregner optimal leverage og sizing
            
            # 🔥 POLICY-DRIVEN (fail-closed): NO FALLBACK VALUES!
            # If RL agent fails to provide leverage/sizing → SKIP trade
            position_size_usd = None  # MUST come from RL agent
            leverage = None  # MUST come from RL agent or policy
            tp_percent = 0.06  # 6% default (TODO: also from policy/RL)
            sl_percent = 0.025  # 2.5% default (TODO: also from policy/RL)
            
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
                # 🔥 BRUKER LEVERAGE FRA RL AGENT (Math AI beregning)
                leverage = int(round(sizing_decision.leverage))  # Convert float to int for Pydantic
                tp_percent = sizing_decision.tp_percent
                sl_percent = sizing_decision.sl_percent
                
                # 🔥 PHASE 3A: Apply risk multiplier
                original_size = position_size_usd
                position_size_usd = position_size_usd * risk_multiplier
            # 🔥 TESTNET SIZING: Reasonable cap for testing ($500 for diversification)
            if os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true":
                original_size = position_size_usd
                position_size_usd = min(position_size_usd, 500.0)  # Max $500 on testnet (better diversification)
                if position_size_usd != original_size:
                    logger.info(f"[TESTNET] Capped position: ${original_size:.0f} → ${position_size_usd:.0f}")
            
                if risk_multiplier != 1.0:
                    logger.info(f"[PHASE 3A] Risk-adjusted position: ${original_size:.0f} → ${position_size_usd:.0f} "
                              f"(multiplier={risk_multiplier:.2f}x, mode={risk_signal.mode.value if risk_signal else 'N/A'})")
                
                logger.info(
                    f"[AI-ENGINE] 🔥 DYNAMIC SIZING: {symbol} ${position_size_usd:.0f} @ {leverage:.1f}x "
                    f"(risk={sizing_decision.risk_pct:.2f}%, TP={tp_percent*100:.1f}%, SL={sl_percent*100:.1f}%) "
                    f"[{sizing_decision.reasoning[:80]}...]"
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
            else:
                # 🔥 FAIL-CLOSED: RL agent not available or failed
                if fallback_triggered:
                    # EXPLORATION FALLBACK: Use minimal sizing for CLM data collection
                    logger.warning(f"[AI-ENGINE] RL_AGENT_NOT_AVAILABLE for {symbol} - Using exploration fallback sizing")
                    position_size_usd = 50.0  # Minimal size for data collection ($50 = 0.5% of $10K account)
                    leverage = 5  # Conservative leverage
                    tp_percent = 0.02  # 2% take profit
                    sl_percent = 0.01  # 1% stop loss
                    logger.info(f"[AI-ENGINE] 🧪 EXPLORATION SIZING: {symbol} ${position_size_usd:.0f} @ {leverage}x (TP={tp_percent*100:.1f}%, SL={sl_percent*100:.1f}%)")
                else:
                    logger.error(f"[AI-ENGINE] RL_AGENT_NOT_AVAILABLE for {symbol} - SKIPPING trade")
                    return  # SKIP trade - no fallback for production signals!
            
            # 🔥 POLICY VALIDATION: Ensure leverage was set
            if leverage is None or leverage <= 0:
                logger.error(
                    f"[AI-ENGINE] POLICY_MISSING_LEVERAGE for {symbol} "
                    f"(RL agent failed to provide leverage) - SKIPPING trade"
                )
                return  # SKIP trade - no fallback to 10x!
            
            if position_size_usd is None or position_size_usd <= 0:
                logger.error(
                    f"[AI-ENGINE] POLICY_MISSING_SIZE for {symbol} "
                    f"(RL agent failed to provide position size) - SKIPPING trade"
                )
                return  # SKIP trade - no fallback to $200!
            
            # Step 4: Build final decision
            
            # 🐛 DEBUG: Log SL/TP calculation
            calculated_sl = current_price * (1 - sl_percent) if action.upper() == "BUY" else current_price * (1 + sl_percent)
            calculated_tp = current_price * (1 + tp_percent) if action.upper() == "BUY" else current_price * (1 - tp_percent)
            sl_formula = f"(1-{sl_percent:.4f})" if action.upper() == "BUY" else f"(1+{sl_percent:.4f})"
            tp_formula = f"(1+{tp_percent:.4f})" if action.upper() == "BUY" else f"(1-{tp_percent:.4f})"
            logger.warning(
                f"[SL_DEBUG] {symbol} {action.upper()}: "
                f"price={current_price:.6f}, sl_pct={sl_percent:.4f}, tp_pct={tp_percent:.4f} | "
                f"SL_CALC={sl_formula}={calculated_sl:.6f}, "
                f"TP_CALC={tp_formula}={calculated_tp:.6f}"
            )
            
            decision = AIDecisionMadeEvent(
                symbol=symbol,
                side=SignalAction(action.lower()),
                confidence=ensemble_confidence,
                entry_price=current_price,
                quantity=position_size_usd,
                leverage=leverage,
                stop_loss=calculated_sl,
                take_profit=calculated_tp,
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
                    
                    # Update position size from Risk Brain if provided (but keep Math AI leverage!)
                    if orchestration_result.position_size > 0:
                        # Only adjust size down if Risk Brain is more conservative
                        if orchestration_result.position_size < position_size_usd:
                            position_size_usd = orchestration_result.position_size
                            logger.info(f"[PHASE 2.2] Risk Brain reduced size: ${position_size_usd:.2f}")
                        else:
                            logger.info(f"[PHASE 2.2] Risk Brain size ${orchestration_result.position_size:.2f} >= Math AI ${position_size_usd:.2f}, keeping Math AI")
                    
                    # 🔥 DISABLED: Do NOT override Math AI leverage with Risk Brain (Math AI is more sophisticated)
                    if orchestration_result.leverage > 0:
                        logger.info(f"[PHASE 2.2] Risk Brain suggested leverage: {orchestration_result.leverage:.1f}x (IGNORED - using Math AI {leverage}x)")
                        # leverage = orchestration_result.leverage  # DISABLED: Keep Math AI leverage!
                    
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
            
                        # TESTNET SIZING: Cap position size (second cap after orchestration)
            if os.getenv("BINANCE_USE_TESTNET", "false").lower() == "true":
                position_size_usd = min(position_size_usd, 500.0)  # Max $500 on testnet
            
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
            
            # RL Bootstrap v2 (shadow_gated)
            rl_meta = {}
            rl_data = None
            try:
                rl_data = await self.rl_influence.fetch(symbol) if getattr(self, 'rl_influence', None) else None
                
                # 🔍 RL_PROOF: Log RL block execution (observability only)
                now_proof = time.time()
                last_proof = self._rl_proof_last_log.get(symbol, 0)
                if now_proof - last_proof > self._rl_proof_throttle_sec:
                    self._rl_proof_last_log[symbol] = now_proof
                    logger.info(
                        f"[AI-ENGINE] RL_PROOF symbol={symbol}, ens_action={action}, "
                        f"ens_conf={ensemble_confidence:.2f}, rl_data={'FOUND' if rl_data else 'NONE'}"
                    )
                
                action, rl_meta = self.rl_influence.apply_shadow(symbol, action, float(ensemble_confidence), rl_data) if getattr(self, 'rl_influence', None) else (action, {})
                
                # 🔍 RL_PROOF: Log shadow result (observability only)
                if now_proof - last_proof > self._rl_proof_throttle_sec:
                    gate_reason = rl_meta.get('rl_gate_reason', 'unknown')
                    rl_effect = rl_meta.get('rl_effect', 'none')
                    rl_policy_age = rl_meta.get('rl_policy_age_sec', -1)
                    rl_conf = rl_meta.get('rl_confidence', 0.0)
                    logger.info(
                        f"[AI-ENGINE] RL_PROOF_RESULT symbol={symbol}, gate_reason={gate_reason}, "
                        f"rl_effect={rl_effect}, policy_age={rl_policy_age}s, rl_conf={rl_conf:.2f}"
                    )
            except Exception:
                rl_meta = {}
            
            # 🔥 RATE LIMITING: Global + Per-Symbol
            now = time.time()
            symbol = trade_intent_payload.get("symbol", "UNKNOWN")
            confidence = trade_intent_payload.get("confidence", 0)
            min_confidence = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.75"))
            
            # Filter 1: Confidence threshold
            if confidence < min_confidence:
                logger.debug(f"[RATE-LIMIT] ⛔ {symbol} skipped: confidence {confidence:.2f} < {min_confidence}")
                return None
            
            # Merge RL metadata
            trade_intent_payload = {**trade_intent_payload, **rl_meta}
            
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
            
            # 🔥 BRIDGE-PATCH: AI Sizing Injection (shadow or live mode)
            logger.info(f"[BRIDGE-PATCH] Entering AI sizer block for {symbol}...")
            try:
                from microservices.ai_engine.ai_sizer_policy import get_ai_sizer
                sizer = get_ai_sizer()
                volatility_factor = features.get("volatility_factor", 1.0)
                # TODO: Get real account equity from Binance account info
                account_equity = 10000.0  # Placeholder
                trade_intent_payload = sizer.inject_into_payload(
                    trade_intent_payload,
                    signal_confidence=ensemble_confidence,
                    volatility_factor=volatility_factor,
                    account_equity=account_equity
                )
            except Exception as e:
                logger.error(f"[BRIDGE-PATCH] AI Sizer failed: {e}", exc_info=True)
                # Continue with original payload if sizer fails (fail-closed)
            
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
    # AI-DRIVEN EXIT EVALUATION (PHASE 3D)
    # ========================================================================
    
    async def evaluate_exit(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        🧠 AI-driven exit evaluation for dynamic profit-taking.
        
        Replaces hardcoded R-level thresholds with intelligent decisions based on:
        - Regime detection (exit if regime flipped)
        - Volatility dynamics (hold if expanding, exit if contracting)
        - Ensemble confidence (exit if degraded significantly)
        - R-momentum (hold if accelerating, exit if stalling)
        - Peak distance (hold if near peak, exit if far from peak)
        - Position age (exit old positions to rotate capital)
        
        Args:
            position_data: Dict with keys:
                - symbol: str
                - side: "LONG"/"SHORT"
                - entry_price: float
                - current_price: float
                - position_qty: float
                - entry_timestamp: int
                - age_sec: int
                - R_net: float
                - R_history: List[float] (optional)
                - entry_regime: str (optional)
                - entry_confidence: float (optional)
                - peak_price: float (optional)
        
        Returns:
            Dict with keys:
                - action: "HOLD", "PARTIAL_CLOSE", or "CLOSE"
                - percentage: float (0.0-1.0)
                - reason: str (human-readable decision reason)
                - factors: Dict (all evaluated factors)
                - current_regime: str
                - hold_score: int
                - exit_score: int
                - timestamp: int
        """
        start_time = datetime.utcnow()
        
        try:
            # Fallback if exit_evaluator not initialized
            if not self.exit_evaluator:
                logger.warning("[AI-ENGINE] ⚠️ Exit evaluator not available, using fallback")
                return {
                    "action": "HOLD",
                    "percentage": 0.0,
                    "reason": "evaluator_unavailable",
                    "factors": {},
                    "current_regime": "UNKNOWN",
                    "hold_score": 0,
                    "exit_score": 0,
                    "timestamp": int(time.time())
                }
            
            # Call exit evaluator
            evaluation = await self.exit_evaluator.evaluate_exit(position_data)
            
            # Convert ExitEvaluation dataclass to dict
            result = {
                "action": evaluation.action,
                "percentage": evaluation.percentage,
                "reason": evaluation.reason,
                "factors": evaluation.factors,
                "current_regime": evaluation.current_regime,
                "hold_score": evaluation.hold_score,
                "exit_score": evaluation.exit_score,
                "timestamp": evaluation.timestamp
            }
            
            # Log decision
            symbol = position_data.get("symbol", "UNKNOWN")
            R_net = position_data.get("R_net", 0)
            logger.info(
                f"[AI-EXIT] {symbol} R={R_net:.2f}: {evaluation.action} {int(evaluation.percentage*100)}% "
                f"(hold={evaluation.hold_score} vs exit={evaluation.exit_score}) - {evaluation.reason}"
            )
            
            # Publish to Redis event stream for monitoring
            try:
                await self.redis_client.xadd(
                    "quantum:stream:ai.exit.decision",
                    {
                        "symbol": symbol,
                        "action": evaluation.action,
                        "percentage": str(evaluation.percentage),
                        "reason": evaluation.reason,
                        "hold_score": str(evaluation.hold_score),
                        "exit_score": str(evaluation.exit_score),
                        "regime": evaluation.current_regime,
                        "R_net": str(R_net),
                        "timestamp": str(evaluation.timestamp)
                    }
                )
            except Exception as e:
                logger.warning(f"[AI-EXIT] Failed to publish to Redis stream: {e}")
            
            # Record performance metrics
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if self.performance_benchmarker:
                self.performance_benchmarker.record_latency('exit_evaluator', latency_ms)
            
            if self.health_monitor:
                self.health_monitor.record_signal_attempt(success=True, latency_ms=latency_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"[AI-ENGINE] Error evaluating exit for {position_data.get('symbol', 'UNKNOWN')}: {e}", exc_info=True)
            
            # Record error
            if self.performance_benchmarker:
                self.performance_benchmarker.record_error('exit_evaluator')
            
            if self.health_monitor:
                self.health_monitor.record_error()
            
            # Return safe fallback
            return {
                "action": "HOLD",
                "percentage": 0.0,
                "reason": f"error:{str(e)[:50]}",
                "factors": {},
                "current_regime": "UNKNOWN",
                "hold_score": 0,
                "exit_score": 0,
                "timestamp": int(time.time())
            }
    
    # ========================================================================
    # RL CONFIDENCE CALIBRATION
    # ========================================================================
    
    async def _ensure_rl_calibration_group(self):
        """Ensure Redis consumer group exists for RL calibration stream."""
        try:
            await self.redis_client.xgroup_create(
                name=self._rl_cal_stream,
                groupname=self._rl_cal_group,
                id='0',
                mkstream=True
            )
            logger.info(f"[RL-CAL] Created consumer group: {self._rl_cal_group}")
        except Exception as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"[RL-CAL] Consumer group already exists: {self._rl_cal_group}")
            else:
                logger.error(f"[RL-CAL] Failed to create consumer group: {e}")
    
    async def _consume_rl_calibration_stream(self):
        """Consume trade.closed events for RL confidence calibration."""
        logger.info(f"[RL-CAL] Consumer started: {self._rl_cal_consumer}")
        last_id = ">"
        
        while self._running:
            try:
                # Read from stream
                result = await self.redis_client.xreadgroup(
                    groupname=self._rl_cal_group,
                    consumername=self._rl_cal_consumer,
                    streams={self._rl_cal_stream: last_id},
                    count=10,
                    block=5000
                )
                
                if not result:
                    await asyncio.sleep(1)
                    continue
                
                for stream_name, messages in result:
                    for msg_id, data in messages:
                        try:
                            await self._process_calibration_event(msg_id, data)
                            # ACK after successful processing
                            await self.redis_client.xack(self._rl_cal_stream, self._rl_cal_group, msg_id)
                        except Exception as e:
                            logger.error(f"[RL-CAL] Error processing event {msg_id}: {e}")
                            # Still ACK to avoid blocking stream
                            await self.redis_client.xack(self._rl_cal_stream, self._rl_cal_group, msg_id)
            
            except asyncio.CancelledError:
                logger.info("[RL-CAL] Consumer task cancelled")
                break
            except Exception as e:
                logger.error(f"[RL-CAL] Consumer error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _process_calibration_event(self, msg_id: str, data: Dict):
        """Process single trade.closed event for calibration stats update."""
        import json
        
        try:
            # Parse event data
            symbol = data.get(b'symbol', b'').decode('utf-8') if isinstance(data.get(b'symbol'), bytes) else data.get('symbol', '')
            pnl = float(data.get(b'pnl', 0) if isinstance(data.get(b'pnl'), bytes) else data.get('pnl', 0))
            
            if not symbol:
                return
            
            # Determine win/loss
            is_win = pnl > 0
            new_win = 1.0 if is_win else 0.0
            
            # Load existing stats
            cal_key = f"{self._rl_cal_key}:{symbol}"
            stats = await self.redis_client.hgetall(cal_key)
            
            if stats:
                # Decode existing stats
                trades = int(stats.get(b'trades', 0) if isinstance(stats.get(b'trades'), bytes) else stats.get('trades', 0))
                wins = int(stats.get(b'wins', 0) if isinstance(stats.get(b'wins'), bytes) else stats.get('wins', 0))
                losses = int(stats.get(b'losses', 0) if isinstance(stats.get(b'losses'), bytes) else stats.get('losses', 0))
                ema_winrate = float(stats.get(b'ema_winrate', 0.5) if isinstance(stats.get(b'ema_winrate'), bytes) else stats.get('ema_winrate', 0.5))
            else:
                trades = 0
                wins = 0
                losses = 0
                ema_winrate = 0.5  # Start neutral
            
            # Update stats
            trades += 1
            if is_win:
                wins += 1
            else:
                losses += 1
            
            # Update EMA winrate
            ema_winrate = ema_winrate * self._rl_cal_decay + new_win * (1 - self._rl_cal_decay)
            
            # Save updated stats
            await self.redis_client.hset(cal_key, mapping={
                'trades': trades,
                'wins': wins,
                'losses': losses,
                'ema_winrate': ema_winrate,
                'updated_ts': int(time.time())
            })
            
            logger.debug(f"[RL-CAL] Updated {symbol}: trades={trades}, ema_winrate={ema_winrate:.3f}")
        
        except Exception as e:
            logger.error(f"[RL-CAL] Error processing calibration event: {e}", exc_info=True)
    
    def _calibrate_rl_confidence(self, symbol: str, raw_conf: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calibrate RL confidence based on historical win/loss EMA.
        
        Args:
            symbol: Trading symbol
            raw_conf: Raw RL confidence from policy
            
        Returns:
            (calibrated_confidence, metadata)
        """
        try:
            # Check if calibration enabled
            if not self._rl_cal_enabled:
                return raw_conf, {"cal": "off"}
            
            # Load calibration stats (sync Redis call - use blocking)
            import redis
            sync_client = redis.Redis(
                host=self.redis_client.connection_pool.connection_kwargs['host'],
                port=self.redis_client.connection_pool.connection_kwargs['port'],
                db=self.redis_client.connection_pool.connection_kwargs['db'],
                decode_responses=False
            )
            
            cal_key = f"{self._rl_cal_key}:{symbol}"
            stats = sync_client.hgetall(cal_key)
            
            if not stats:
                return raw_conf, {"cal": "insufficient", "trades": 0}
            
            # Parse stats
            trades = int(stats.get(b'trades', 0))
            ema_winrate = float(stats.get(b'ema_winrate', 0.5))
            
            # Check minimum trades threshold
            if trades < self._rl_cal_min_trades:
                return raw_conf, {"cal": "insufficient", "trades": trades}
            
            # Compute calibrated confidence
            # Map winrate (0..1) to multiplier (0.5..1.5)
            multiplier = 0.5 + ema_winrate
            
            # Apply smoothing
            cal_conf = raw_conf * (1 - self._rl_cal_alpha) + (raw_conf * multiplier) * self._rl_cal_alpha
            
            # Clamp to floor/ceil
            cal_conf = max(self._rl_cal_floor, min(self._rl_cal_ceil, cal_conf))
            
            meta = {
                "cal": "on",
                "trades": trades,
                "ema_winrate": round(ema_winrate, 3),
                "mult": round(multiplier, 3),
                "raw": round(raw_conf, 3),
                "calibrated": round(cal_conf, 3)
            }
            
            return cal_conf, meta
        
        except Exception as e:
            logger.error(f"[RL-CAL] Calibration error for {symbol}: {e}")
            return raw_conf, {"cal": "error", "error": str(e)}
    
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
                
                # Phase 4F: Run adaptive retraining cycle (with timeout to prevent hangs)
                if self.adaptive_retrainer:
                    try:
                        # 🔥 TIMEOUT: Max 5 minutes for retrain cycle to prevent hangs
                        result = await asyncio.wait_for(
                            asyncio.to_thread(self.adaptive_retrainer.run_cycle),
                            timeout=300.0
                        )
                        if result.get("status") == "success":
                            logger.info(f"[Retrainer] Cycle completed - Models: {result.get('models_retrained', [])}")
                            
                            # Phase 4G: Validate retrained models (with timeout)
                            if self.model_validator:
                                try:
                                    validation_result = await asyncio.wait_for(
                                        asyncio.to_thread(self.model_validator.run_validation_cycle),
                                        timeout=120.0
                                    )
                                    logger.info(f"[Validator] Validation complete: {validation_result}")
                                except asyncio.TimeoutError:
                                    logger.error("[Validator] Validation timeout (120s) - skipping")
                                except Exception as val_e:
                                    logger.error(f"[Validator] Validation error: {val_e}")
                    except asyncio.TimeoutError:
                        logger.error("[Retrainer] Cycle timeout (300s) - skipping this cycle")
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
                        "symbols": getattr(self, '_active_symbols', []),
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
        
        # Add RL calibration stats
        if self._rl_cal_enabled:
            try:
                cal_keys = await self.redis_client.keys(f"{self._rl_cal_key}:*")
                metrics["rl_calibration"] = {
                    "enabled": True,
                    "symbols_tracked": len(cal_keys),
                    "stream": self._rl_cal_stream,
                    "min_trades": self._rl_cal_min_trades
                }
            except Exception as e:
                logger.error(f"[RL-CAL] Error getting stats: {e}")
        
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
