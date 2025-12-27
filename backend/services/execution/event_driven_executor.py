"""
Event-driven execution engine: AI continuously monitors market and trades
when it detects strong signals, without fixed time intervals.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.ai_trading_engine import AITradingEngine
from backend.services.execution.execution import run_portfolio_rebalance, build_execution_adapter
from backend.config.execution import load_execution_config
from backend.config.risk import load_risk_config
from backend.database import SessionLocal

# [TARGET] RISK MANAGEMENT INTEGRATION
from backend.config.risk_management import load_risk_management_config, ConsensusType
from backend.services.risk_management import (
    TradeLifecycleManager,
    SignalQuality,
    MarketConditions,
)
from backend.services.market_data_helpers import (
    fetch_market_conditions,
    get_consensus_type,
    normalize_action,
)

# [TARGET] QUANT MODULES INTEGRATION
from backend.services.regime_detector import RegimeDetector, RegimeConfig
from backend.services.cost_model import CostModel, CostConfig, estimate_trade_cost
from backend.services.symbol_performance import (
    SymbolPerformanceManager,
    SymbolPerformanceConfig,
    TradeResult,
)
from backend.services.execution.exit_policy_regime_config import get_exit_params
from backend.services.monitoring.logging_extensions import (
    enrich_trade_entry,
    enrich_trade_exit,
    format_trade_log_message,
)
from backend.services.execution.hybrid_tpsl import place_hybrid_orders
from backend.services.execution.position_invariant import (
    get_position_invariant_enforcer,
    PositionInvariantViolation,
)

# [PHASE 1] Exit Order Gateway for observability
try:
    from backend.services.execution.exit_order_gateway import submit_exit_order
    EXIT_GATEWAY_AVAILABLE = True
except ImportError:
    EXIT_GATEWAY_AVAILABLE = False
    logger_gateway = logging.getLogger(__name__ + ".exit_gateway")
    logger_gateway.warning("[EXIT_GATEWAY] Not available - will place orders directly")

# [EXIT BRAIN V3] Advanced exit strategy orchestration
try:
    from backend.domains.exits.exit_brain_v3.router import ExitRouter
    from backend.domains.exits.exit_brain_v3.integration import build_context_from_position
    EXIT_BRAIN_V3_AVAILABLE = True
    logger_exit_brain = logging.getLogger(__name__ + ".exit_brain_v3")
    logger_exit_brain.info("[OK] Exit Brain v3 available")
except ImportError as e:
    EXIT_BRAIN_V3_AVAILABLE = False
    logger_exit_brain = logging.getLogger(__name__ + ".exit_brain_v3")
    logger_exit_brain.warning(f"[WARNING] Exit Brain v3 not available: {e}")
from backend.services.governance.orchestrator_policy import (
    OrchestratorPolicy,
    OrchestratorConfig,
    create_risk_state,
    create_symbol_performance,
    create_cost_metrics,
)
from backend.services.risk.funding_rate_filter import FundingRateFilter
from backend.services.governance.policy_observer import PolicyObserver
from backend.services.governance.orchestrator_config import (
    OrchestratorIntegrationConfig,
    OrchestratorMode
)
# 🛑 MODEL SUPERVISOR: Bias detection and model performance tracking
from backend.services.ai.model_supervisor import ModelSupervisor
from backend.utils.universe import load_universe

# [NEW] META-STRATEGY SELECTOR INTEGRATION
try:
    from backend.services.meta_strategy_integration import get_meta_strategy_integration
    META_STRATEGY_AVAILABLE = True
    logger_meta = logging.getLogger(__name__ + ".meta_strategy")
    logger_meta.info("[OK] Meta-Strategy Selector available")
except ImportError as e:
    META_STRATEGY_AVAILABLE = False
    logger_meta = logging.getLogger(__name__ + ".meta_strategy")
    logger_meta.warning(f"[WARNING] Meta-Strategy Selector not available: {e}")

# [NEW] AI SYSTEM INTEGRATION
try:
    from backend.services.system_services import get_ai_services, AISystemServices, SubsystemMode
    from backend.services.integration_hooks import (
        pre_trade_universe_filter,
        pre_trade_risk_check,
        pre_trade_portfolio_check,
        pre_trade_confidence_adjustment,
        pre_trade_position_sizing,
        execution_order_type_selection,
        execution_slippage_check,
        post_trade_position_classification,
        post_trade_amplification_check,
        periodic_self_healing_check,
        periodic_ai_hfos_coordination
    )
    AI_INTEGRATION_AVAILABLE = True
    logger_integration = logging.getLogger(__name__ + ".ai_integration")
    logger_integration.info("[OK] AI System Integration available")
except ImportError as e:
    AI_INTEGRATION_AVAILABLE = False
    logger_integration = logging.getLogger(__name__ + ".ai_integration")
    logger_integration.warning(f"[WARNING] AI System Integration not available: {e}")

# [NEW] MSC AI INTEGRATION
try:
    from backend.services.msc_ai_integration import QuantumPolicyStoreMSC
    MSC_AI_AVAILABLE = True
    logger_msc = logging.getLogger(__name__ + ".msc_ai")
    logger_msc.info("[OK] MSC AI Policy Reader available")
except ImportError as e:
    MSC_AI_AVAILABLE = False
    logger_msc = logging.getLogger(__name__ + ".msc_ai")
    logger_msc.warning(f"[WARNING] MSC AI not available: {e}")

# [NEW] EMERGENCY STOP SYSTEM (ESS) - SPRINT 1 D3
try:
    from backend.core.safety.ess import EmergencyStopSystem
    from backend.events.listeners.ess_listener import ESSEventListener
    ESS_AVAILABLE = True
    logger_ess = logging.getLogger(__name__ + ".ess")
    logger_ess.info("[OK] Emergency Stop System available")
except ImportError as e:
    ESS_AVAILABLE = False
    logger_ess = logging.getLogger(__name__ + ".ess")
    logger_ess.warning(f"[WARNING] Emergency Stop System not available: {e}")

# [NEW] TRADESTORE - SPRINT 1 D5
try:
    from backend.core.trading import get_trade_store, Trade, TradeSide, TradeStatus
    TRADESTORE_AVAILABLE = True
    logger_tradestore = logging.getLogger(__name__ + ".tradestore")
    logger_tradestore.info("[OK] TradeStore available")
except ImportError as e:
    TRADESTORE_AVAILABLE = False
    logger_tradestore = logging.getLogger(__name__ + ".tradestore")
    logger_tradestore.warning(f"[WARNING] TradeStore not available: {e}")

# [NEW] RL VOLATILITY SAFETY ENVELOPE - SPRINT 1 D4
try:
    from backend.services.risk.rl_volatility_safety_envelope import get_rl_volatility_envelope
    RL_ENVELOPE_AVAILABLE = True
    logger_envelope = logging.getLogger(__name__ + ".rl_envelope")
    logger_envelope.info("[OK] RL Volatility Safety Envelope available")
except ImportError as e:
    RL_ENVELOPE_AVAILABLE = False
    logger_envelope = logging.getLogger(__name__ + ".rl_envelope")
    logger_envelope.warning(f"[WARNING] RL Volatility Safety Envelope not available: {e}")

logger = logging.getLogger(__name__)

# [TARGET] CRITICAL: Shared TP/SL storage path for position_monitor
TPSL_STORAGE_PATH = Path("/app/tmp/quantum_tpsl.json")

# Create directory if not exists
TPSL_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)


class EventDrivenExecutor:
    """
    Continuously monitors market signals and executes trades when AI
    detects high-confidence opportunities, regardless of time.
    """

    def __init__(
        self,
        ai_engine: AITradingEngine,
        symbols: List[str],
        confidence_threshold: float = 0.60,  # BALANCED: Trade good consensus signals (was 0.70)
        check_interval_seconds: int = 30,
        cooldown_seconds: int = 300,
        ai_services: Optional['AISystemServices'] = None,  # [NEW] AI System Services
        app_state: Optional[any] = None,  # [NEW] App state for Safety Governor access
        event_bus: Optional[any] = None,  # [CRITICAL FIX #6] EventBus for strategy.switched events
    ):
        self.ai_engine = ai_engine
        self.symbols = symbols
        self.confidence_threshold = confidence_threshold
        self.check_interval = check_interval_seconds
        self.cooldown = cooldown_seconds
        
        # [SPRINT 5 - PATCH #3] Signal Flood Throttling
        from collections import deque
        self._signal_queue_max_size = int(os.getenv("QT_SIGNAL_QUEUE_MAX", "20"))  # REDUCED: 100 → 20 for quality
        self._signal_queue = deque(maxlen=self._signal_queue_max_size)
        self._dropped_signals_count = 0
        logger.info(f"[PATCH #3] Signal queue throttling enabled: max_size={self._signal_queue_max_size}")
        
        # [TP v3] Mutex locks for thread-safe TP/SL updates
        self._tp_update_locks: Dict[str, asyncio.Lock] = {}
        logger.info("[TP v3] TP/SL update mutex initialized for race condition prevention")
        
        # [NEW] Store app state for Safety Governor access
        self._app_state = app_state
        
        # [CRITICAL FIX #6] Store EventBus and subscribe to strategy.switched
        self.event_bus = event_bus
        self.current_strategy = "moderate"  # Default strategy
        if self.event_bus:
            self.event_bus.subscribe("strategy.switched", self._handle_strategy_switch)
            logger.info("[OK] EventDrivenExecutor subscribed to strategy.switched events")
        
        # [NEW] AI SYSTEM INTEGRATION
        if AI_INTEGRATION_AVAILABLE:
            self.ai_services = ai_services or get_ai_services()
            logger.info("[OK] AI System Services integrated into EventDrivenExecutor")
            
            # [AI-OS] Load PAL and PBA references for trade decision enhancement
            if self.ai_services and hasattr(self.ai_services, 'pal') and self.ai_services.pal:
                self.pal = self.ai_services.pal
                logger.info("[AI-OS] PAL (Profit Amplification Layer) available")
            else:
                self.pal = None
                logger.warning("[AI-OS] PAL not available")
            
            if self.ai_services and hasattr(self.ai_services, 'pba'):
                self.pba = self.ai_services.pba
                logger.info("[AI-OS] PBA (Portfolio Balancer AI) available")
            else:
                self.pba = None
                logger.warning("[AI-OS] PBA not available")
        else:
            self.ai_services = None
            self.pal = None
            self.pba = None
            logger.info("[INFO] AI System Services not available - using default behavior")
        
        # [NEW] META-STRATEGY SELECTOR: AI-powered strategy selection
        if META_STRATEGY_AVAILABLE:
            try:
                meta_enabled = os.getenv("META_STRATEGY_ENABLED", "true").lower() == "true"
                meta_epsilon = float(os.getenv("META_STRATEGY_EPSILON", "0.10"))
                meta_alpha = float(os.getenv("META_STRATEGY_ALPHA", "0.20"))
                
                self.meta_strategy = get_meta_strategy_integration(
                    enabled=meta_enabled,
                    epsilon=meta_epsilon,
                    alpha=meta_alpha
                )
                
                metrics = self.meta_strategy.get_metrics()
                logger.info(
                    f"[OK] Meta-Strategy Selector initialized: "
                    f"enabled={metrics['enabled']}, epsilon={metrics['epsilon']:.0%}, alpha={metrics['alpha']:.0%}"
                )
            except Exception as e:
                logger.error(f"[ERROR] Meta-Strategy initialization failed: {e}", exc_info=True)
                self.meta_strategy = None
        else:
            self.meta_strategy = None
            logger.info("[INFO] Meta-Strategy Selector not available")
        
        # OPTIMIZATION: Enable direct execution to bypass slow rebalancing
        self._direct_execute = os.getenv("QT_EVENT_DIRECT_EXECUTE", "1") == "1"
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_rebalance_time: Optional[datetime] = None
        
        # Load risk config for position sizing
        self._risk_config = load_risk_config()
        self._execution_config = load_execution_config()
        self._adapter = None
        
        # [TARGET] RISK MANAGEMENT: Initialize TradeLifecycleManager with AI engine
        self._rm_config = load_risk_management_config()
        self._trade_manager = TradeLifecycleManager(self._rm_config, ai_engine=ai_engine)
        logger.info("[OK] Risk Management layer initialized with AI feedback loop")
        
        # [TARGET] QUANT MODULES: Initialize advanced quantitative trading modules
        # [REGIME FIX] Pass EventBus for regime.changed event publication
        self.regime_detector = RegimeDetector(
            config=RegimeConfig(
                atr_ratio_low=0.005,
                atr_ratio_normal=0.015,
                atr_ratio_high=0.03,
                adx_trending=25.0,
                adx_strong_trend=40.0,
                range_width_threshold=0.02,
                ema_alignment_pct=0.02
            ),
            event_bus=event_bus  # Enable regime.changed event publishing
        )
        
        self.cost_model = CostModel(
            config=CostConfig(
                maker_fee_rate=0.0002,  # 0.02%
                taker_fee_rate=0.0004,  # 0.04%
                base_slippage_bps=2.0,
                volatility_slippage_factor=50.0,
                funding_rate_per_8h=0.0001
            )
        )
        
        self.symbol_perf = SymbolPerformanceManager(
            config=SymbolPerformanceConfig(
                min_trades_for_adjustment=5,
                poor_winrate_threshold=0.30,
                good_winrate_threshold=0.55,
                poor_avg_R_threshold=0.0,
                good_avg_R_threshold=1.5,
                poor_risk_multiplier=0.5,
                good_risk_multiplier=1.0,
                disable_after_losses=10,
                reenable_after_wins=3,
                persistence_file="data/symbol_performance.json"
            )
        )
        
        # [NEW] FUNDING RATE FILTER: Prevent high funding cost positions
        self.funding_filter = FundingRateFilter(
            max_funding_rate=0.001,  # 0.1% per 8h maximum
            warn_funding_rate=0.0005,  # 0.05% per 8h warning
        )
        logger.info("[MONEY_WITH_WINGS] Funding Rate Filter initialized")
        
        # [TARGET] ORCHESTRATOR: Top-level policy engine that unifies all subsystems
        orchestrator_policy_store = getattr(self, 'policy_store', None)  # Get from executor if attached
        self.orchestrator = OrchestratorPolicy(
            config=OrchestratorConfig(
                base_confidence=0.60,  # BALANCED: Good signals (was 0.70)
                base_risk_pct=1.0,
                daily_dd_limit=5.0,  # 5% daily DD limit for testnet
                losing_streak_limit=5,
                max_open_positions=10,  # Synced with QT_MAX_POSITIONS
                total_exposure_limit=20.0,  # 20% max exposure
                policy_update_interval_sec=60
            ),
            policy_store=orchestrator_policy_store  # Pass PolicyStore for dynamic filtering
        )
        
        if orchestrator_policy_store:
            logger.info("[OK] Orchestrator initialized with PolicyStore integration")
        
        # [TARGET] ORCHESTRATOR INTEGRATION CONFIG: LIVE MODE - Step 1 (Signal Filtering)
        self.orch_config = OrchestratorIntegrationConfig.create_live_mode_gradual()
        
        # [TARGET] POLICY OBSERVER: Logs what policy would do without enforcement
        self.policy_observer = PolicyObserver(
            log_dir=self.orch_config.observation_log_dir
        )
        
        # 🛑 MODEL SUPERVISOR: Bias detection and model performance tracking
        self.model_supervisor = ModelSupervisor()  # Mode read from QT_MODEL_SUPERVISOR_MODE env var
        logger.info("[OK] ModelSupervisor initialized")
        
        # [NEW] MSC AI POLICY READER: Read supreme AI decisions
        if MSC_AI_AVAILABLE:
            try:
                self.msc_policy_store = QuantumPolicyStoreMSC()
                logger.info("[OK] MSC AI Policy Reader initialized")
            except Exception as e:
                logger.error(f"[ERROR] MSC AI Policy Reader initialization failed: {e}", exc_info=True)
                self.msc_policy_store = None
        else:
            self.msc_policy_store = None
        
        # [NEW] STRATEGY RUNTIME ENGINE: AI-driven signal generation
        self._strategy_runtime_available = False
        try:
            from backend.services.strategy_runtime_integration import get_strategy_runtime_engine, check_strategy_runtime_health
            
            # Initialize engine (now uses AI models)
            self.strategy_runtime_engine = get_strategy_runtime_engine()
            health = check_strategy_runtime_health()
            
            if health['status'] == 'healthy':
                self._strategy_runtime_available = True
                logger.info(
                    f"[OK] Strategy Runtime Engine initialized (AI-driven): "
                    f"{health['active_strategies']} active strategies"
                )
            else:
                logger.warning(f"[WARNING] Strategy Runtime Engine unhealthy: {health.get('error')}")
                
        except ImportError:
            logger.info("[INFO] Strategy Runtime Engine not available (module not found)")
        except Exception as e:
            logger.error(f"[ERROR] Strategy Runtime Engine initialization failed: {e}", exc_info=True)
        
        logger.info("[OK] Quant modules initialized: RegimeDetector, CostModel, SymbolPerformanceManager, OrchestratorPolicy")
        logger.info(f"[OK] {self.orch_config.get_summary()}")
        
        # [NEW] Register orchestrator in AISystemServices if available
        if self.ai_services and hasattr(self.ai_services, '_services_status'):
            self.ai_services._services_status["orchestrator"] = "HEALTHY"
            logger.info("[OK] Orchestrator registered in AI System Services")
        
        # [NEW] EMERGENCY STOP SYSTEM (ESS) - SPRINT 1 D3
        self.ess = None
        self.ess_listener = None
        if ESS_AVAILABLE and event_bus:
            try:
                # Get PolicyStore from app_state if available
                policy_store = None
                if app_state and hasattr(app_state, 'policy_store'):
                    policy_store = app_state.policy_store
                
                if policy_store:
                    self.ess = EmergencyStopSystem(policy_store, event_bus)
                    self.ess_listener = ESSEventListener(self.ess, event_bus)
                    logger_ess.info("[OK] Emergency Stop System initialized")
                else:
                    logger_ess.warning("[WARNING] ESS not initialized: PolicyStore unavailable")
            except Exception as e:
                logger_ess.error(f"[ERROR] Emergency Stop System initialization failed: {e}", exc_info=True)
                self.ess = None
                self.ess_listener = None
        
        # [NEW] RL VOLATILITY SAFETY ENVELOPE - SPRINT 1 D4
        self.rl_envelope = None
        if RL_ENVELOPE_AVAILABLE:
            try:
                # Get PolicyStore from app_state if available
                policy_store = None
                if app_state and hasattr(app_state, 'policy_store'):
                    policy_store = app_state.policy_store
                
                self.rl_envelope = get_rl_volatility_envelope(policy_store)
                logger_envelope.info("[OK] RL Volatility Safety Envelope initialized")
            except Exception as e:
                logger_envelope.error(f"[ERROR] RL Envelope initialization failed: {e}", exc_info=True)
                self.rl_envelope = None
        
        # [NEW] TRADESTORE - SPRINT 1 D5: Initialize trade persistence
        self.trade_store = None
        if TRADESTORE_AVAILABLE:
            # TradeStore will be initialized async in start() method
            logger_tradestore.info("[OK] TradeStore will be initialized on start")
        
        # [EXIT BRAIN V3] Initialize Exit Router for exit strategy orchestration
        self.exit_router = None
        self.exit_brain_enabled = os.getenv("EXIT_BRAIN_V3_ENABLED", "true").lower() == "true"
        if EXIT_BRAIN_V3_AVAILABLE and self.exit_brain_enabled:
            try:
                self.exit_router = ExitRouter()
                logger_exit_brain.info("[OK] Exit Brain v3 Exit Router initialized")
            except Exception as e:
                logger_exit_brain.error(f"[ERROR] Exit Router initialization failed: {e}", exc_info=True)
                self.exit_router = None
        elif not self.exit_brain_enabled:
            logger_exit_brain.info("[INFO] Exit Brain v3 disabled via EXIT_BRAIN_V3_ENABLED=false")
        else:
            logger_exit_brain.warning("[WARNING] Exit Brain v3 not available - using legacy hybrid_tpsl")
        
        # [TARGET] CRITICAL: Store Dynamic TP/SL per symbol for position_monitor
        # Format: {symbol: {"tp_percent": 0.06, "sl_percent": 0.08, ...}}
        self._symbol_tpsl = {}
        
        logger.info(
            "Event-driven executor initialized: %d symbols, confidence >= %.2f, "
            "check every %ds, cooldown %ds",
            len(symbols), confidence_threshold, check_interval_seconds, cooldown_seconds
        )
        
        # [REGIME FIX] Subscribe to regime.changed events if EventBus available
        if event_bus:
            asyncio.create_task(self._subscribe_to_regime_events())
    
    async def _subscribe_to_regime_events(self) -> None:
        """[REGIME FIX] Subscribe to regime.changed events for coordinated response."""
        try:
            await self.event_bus.subscribe(
                stream_name="regime.changed",
                consumer_group="executor",
                handler=self._handle_regime_changed
            )
            logger.info("[REGIME] ✅ Subscribed to regime.changed events")
        except Exception as e:
            logger.error(f"[REGIME] Failed to subscribe to regime.changed events: {e}")
    
    async def _handle_regime_changed(self, event: Dict[str, Any]) -> None:
        """[REGIME FIX] Handle regime.changed event - trigger immediate policy update."""
        try:
            symbol = event.get("symbol", "GLOBAL")
            change_type = event.get("change_type", "UNKNOWN")
            old_vol_regime = event.get("old_volatility_regime", "UNKNOWN")
            new_vol_regime = event.get("new_volatility_regime", "UNKNOWN")
            old_trend = event.get("old_trend_regime", "UNKNOWN")
            new_trend = event.get("new_trend_regime", "UNKNOWN")
            
            logger.warning(
                f"[REGIME] 🔄 Regime change detected for {symbol}:\n"
                f"   Change Type: {change_type}\n"
                f"   Volatility: {old_vol_regime} → {new_vol_regime}\n"
                f"   Trend: {old_trend} → {new_trend}\n"
                f"   Triggering immediate policy update..."
            )
            
            # Trigger immediate AI-HFOS update if available
            if self.ai_services and hasattr(self.ai_services, 'trigger_immediate_update'):
                await self.ai_services.trigger_immediate_update(reason="regime_change")
                logger.info("[REGIME] ✅ AI-HFOS immediate update triggered")
            
            # Trigger immediate Meta-Strategy re-evaluation
            if hasattr(self, 'meta_strategy') and self.meta_strategy:
                logger.info("[REGIME] 🔄 Meta-Strategy will use new regime on next signal")
            
            # Force policy refresh on next cycle (don't wait for 60s interval)
            self._last_policy_update = None
            
        except Exception as e:
            logger.error(f"[REGIME] Error handling regime.changed event: {e}", exc_info=True)
    
    async def _handle_strategy_switch(self, event_data: dict) -> None:
        """
        Handle strategy.switched event from Meta-Strategy Controller (CRITICAL FIX #6).
        
        Args:
            event_data: {
                "from_strategy": str,
                "to_strategy": str,
                "reason": str,
                "timestamp": str,
                "confidence": float
            }
        """
        old_strategy = event_data.get("from_strategy", "unknown")
        new_strategy = event_data.get("to_strategy", "unknown")
        reason = event_data.get("reason", "unknown")
        confidence = event_data.get("confidence", 0.0)
        
        logger.warning(
            f"🔄 STRATEGY SWITCH DETECTED: {old_strategy} → {new_strategy}\n"
            f"   Reason: {reason}\n"
            f"   Confidence: {confidence:.2%}\n"
            f"   Applying new execution config immediately..."
        )
        
        self.current_strategy = new_strategy
        
        # Apply strategy-specific execution config
        await self._apply_strategy_config(new_strategy)
    
    async def _apply_strategy_config(self, strategy: str) -> None:
        """
        Apply execution configuration for strategy (CRITICAL FIX #6).
        
        🔧 SPRINT 1 - D1: Now reads from PolicyStore for dynamic risk limits.
        
        Args:
            strategy: Strategy name (conservative, moderate, aggressive, etc.)
        """
        # 🔧 SPRINT 1 - D1: Try to get config from PolicyStore first
        if hasattr(self, 'policy_store') and self.policy_store:
            try:
                config_from_policy = await self._get_strategy_from_policy_store(strategy)
                if config_from_policy:
                    await self._apply_config_dict(config_from_policy, strategy)
                    return
            except Exception as e:
                logger.warning(f"Failed to load strategy config from PolicyStore: {e}, using hardcoded")
        
        # Fallback: Strategy configurations
        configs = {
            "conservative": {
                "max_position_size": 0.02,  # 2% of portfolio per position
                "max_leverage": 5,
                "confidence_threshold": 0.65,
                "cooldown_seconds": 600,
                "max_open_positions": 3
            },
            "moderate": {
                "max_position_size": 0.05,  # 5% of portfolio per position
                "max_leverage": 10,
                "confidence_threshold": 0.50,
                "cooldown_seconds": 300,
                "max_open_positions": 5
            },
            "aggressive": {
                "max_position_size": 0.10,  # 10% of portfolio per position
                "max_leverage": 20,
                "confidence_threshold": 0.40,
                "cooldown_seconds": 120,
                "max_open_positions": 8
            },
            "defensive": {
                "max_position_size": 0.01,  # 1% of portfolio per position
                "max_leverage": 3,
                "confidence_threshold": 0.75,
                "cooldown_seconds": 900,
                "max_open_positions": 2
            }
        }
        
        config = configs.get(strategy, configs["moderate"])  # Default to moderate
        await self._apply_config_dict(config, strategy)
    
    async def _get_strategy_from_policy_store(self, strategy: str) -> Optional[dict]:
        """
        Get strategy configuration from PolicyStore based on risk mode.
        
        🔧 SPRINT 1 - D1: Maps strategy names to PolicyStore risk modes.
        
        Args:
            strategy: Strategy name (conservative, moderate, aggressive, defensive)
        
        Returns:
            Config dict or None if not available
        """
        if not self.policy_store:
            return None
        
        risk_config = await self.policy_store.get_active_risk_config()
        
        # Extract relevant fields from PolicyStore
        return {
            "max_position_size": risk_config.max_risk_pct_per_trade,
            "max_leverage": risk_config.max_leverage,
            "confidence_threshold": risk_config.global_min_confidence,
            "cooldown_seconds": 300,  # Default, not in PolicyStore yet
            "max_open_positions": risk_config.max_positions
        }
    
    async def _apply_config_dict(self, config: dict, strategy: str) -> None:
        """
        Apply configuration dictionary to executor.
        
        🔧 SPRINT 1 - D1: Separated for cleaner code.
        
        Args:
            config: Configuration dictionary
            strategy: Strategy name for logging
        """
        # Apply configuration
        self.confidence_threshold = config["confidence_threshold"]
        self.cooldown = config["cooldown_seconds"]
        
        # Update risk config if available
        if hasattr(self, '_risk_config'):
            self._risk_config.max_position_size_pct = config["max_position_size"]
            self._risk_config.max_leverage = config["max_leverage"]
        
        logger.info(
            f"✅ Execution config updated for {strategy} strategy:\n"
            f"   Max Position Size: {config['max_position_size']:.1%}\n"
            f"   Max Leverage: {config['max_leverage']}x\n"
            f"   Confidence Threshold: {config['confidence_threshold']:.1%}\n"
            f"   Cooldown: {config['cooldown_seconds']}s\n"
            f"   Max Open Positions: {config['max_open_positions']}"
        )

    async def start(self):
        """Start the event-driven monitoring loop as a background task."""
        if self._running:
            logger.warning("Event-driven executor already running")
            return
        
        # [NEW] TRADESTORE - SPRINT 1 D5: Initialize async trade persistence
        if TRADESTORE_AVAILABLE and self.trade_store is None:
            try:
                # Get Redis client from app_state if available
                redis_client = None
                if self._app_state and hasattr(self._app_state, 'redis'):
                    redis_client = self._app_state.redis
                
                self.trade_store = await get_trade_store(redis_client=redis_client)
                logger_tradestore.info(
                    f"[OK] TradeStore initialized: {self.trade_store.backend_name} backend"
                )
                
                # Recovery: Load open trades from persistence
                open_trades = await self.trade_store.get_open_trades()
                if open_trades:
                    logger_tradestore.info(
                        f"[RECOVERY] Found {len(open_trades)} open trades from previous session"
                    )
                    for trade in open_trades[:5]:  # Log first 5
                        logger_tradestore.info(
                            f"  - {trade.symbol} {trade.side.value}: "
                            f"Entry=${trade.entry_price:.2f}, Size=${trade.margin_usd:.2f}"
                        )
            except Exception as e:
                logger_tradestore.error(f"[ERROR] TradeStore initialization failed: {e}", exc_info=True)
                self.trade_store = None
        
        # [NEW] Start ESS Listener if available
        if self.ess_listener:
            try:
                await self.ess_listener.start()
                logger_ess.info("[OK] Emergency Stop System listener started")
            except Exception as e:
                logger_ess.error(f"[ERROR] ESS listener start failed: {e}", exc_info=True)
        
        # Warmup time-series models with historical data (async background to avoid blocking startup)
        async def _warmup_models() -> None:
            try:
                logger.info("[STARTUP] Warming up AI models with historical data...")
                symbols = load_universe(universe_name="l1l2-top", max_symbols=100)
                await self.ai_engine.agent.warmup_history_buffers(symbols, lookback=120)
                logger.info("[STARTUP] ✅ Models warmed up - ready for live trading!")
            except Exception as warmup_error:
                logger.warning(
                    f"[STARTUP] Warmup failed (models will use fallback initially): {warmup_error}"
                )

        # Run warmup without blocking the main event loop so health checks respond quickly
        loop = asyncio.get_running_loop()
        loop.create_task(_warmup_models())
        
        self._running = True
        # Create background task and immediately add it to a set to prevent garbage collection
        # This is the recommended asyncio pattern for long-lived background tasks
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self._monitor_loop(), name="event-driven-monitor")
        logger.info("Event-driven trading mode active - monitoring market continuously")

    async def stop(self):
        """Stop the monitoring loop."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("EventDrivenExecutor stopped")

    async def _monitor_loop(self):
        """Main loop: continuously check market and execute on strong signals."""
        logger.info("Monitoring loop started")
        try:
            while self._running:
                try:
                    logger.info("Checking %d symbols for signals >= %.2f threshold", len(self.symbols), self.confidence_threshold)
                    await self._check_and_execute()
                    logger.info("Check complete, sleeping %d seconds", self.check_interval)
                    
                    # [NEW] PERIODIC AI SYSTEM CHECKS
                    if AI_INTEGRATION_AVAILABLE and self.ai_services:
                        try:
                            # Self-Healing check (every 2 minutes)
                            await periodic_self_healing_check()
                            # AI-HFOS coordination (every 60 seconds)
                            await periodic_ai_hfos_coordination()
                        except Exception as e:
                            logger.error(f"[ERROR] Periodic AI check failed: {e}", exc_info=True)
                            # Continue execution - fail-safe behavior
                    
                except asyncio.CancelledError:
                    logger.info("Monitoring loop cancelled")
                    raise  # Re-raise to exit properly
                except Exception as e:
                    logger.error("Error in event-driven monitoring: %s", e, exc_info=True)
                
                # Wait before next check
                try:
                    await asyncio.sleep(self.check_interval)
                except asyncio.CancelledError:
                    logger.info("Sleep cancelled, exiting loop")
                    raise  # Re-raise to exit properly
        except asyncio.CancelledError:
            logger.info("Monitor loop task cancelled")
        except Exception as e:
            logger.error("FATAL error in monitor loop: %s", e, exc_info=True)
        finally:
            logger.info("Monitoring loop ended")

    async def _check_and_execute(self):
        """
        Check AI signals and trigger portfolio rebalancing if strong signals detected.
        
        [FULL DEPLOYMENT MODE - ALL AI-OS SUBSYSTEMS ACTIVE]
        """
        logger.info("[SEARCH] _check_and_execute() started")
        # MICRO HOTFIX: Always define top_signals as safe default
        top_signals = []
        now = datetime.now(timezone.utc)
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 0: MSC AI POLICY CHECK (SUPREME CONTROLLER)
        # ═══════════════════════════════════════════════════════════════════════
        
        # [NEW] MSC AI: Read supreme policy decisions
        msc_policy = None
        if MSC_AI_AVAILABLE and self.msc_policy_store:
            try:
                msc_policy = self.msc_policy_store.read_policy()
                if msc_policy:
                    logger.info(
                        f"[MSC AI] Policy loaded: "
                        f"risk_mode={msc_policy.get('risk_mode')}, "
                        f"strategies={len(msc_policy.get('allowed_strategies', []))}, "
                        f"max_risk={msc_policy.get('max_risk_per_trade', 0)*100:.2f}%"
                    )
                    
                    # Apply MSC AI confidence threshold (overrides all other settings)
                    if msc_policy.get('global_min_confidence'):
                        effective_confidence = msc_policy['global_min_confidence']
                        logger.info(
                            f"[MSC AI] Confidence threshold set by MSC AI: {effective_confidence:.2f}"
                        )
                else:
                    logger.debug("[MSC AI] No policy available yet (waiting for first evaluation)")
            except Exception as e:
                logger.error(f"[ERROR] Failed to read MSC AI policy: {e}", exc_info=True)
                msc_policy = None
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 1: SAFETY & HEALTH CHECKS
        # ═══════════════════════════════════════════════════════════════════════
        
        # CRITICAL FIX #1: INFRASTRUCTURE HEALTH CHECK (Trading Gate)
        # Block new trades if Redis is unavailable (prevents stale policy trading)
        try:
            # Check if event_bus has health check method (EventBus v2 with Redis)
            if self.event_bus and hasattr(self.event_bus, 'redis_health_check'):
                redis_healthy = await self.event_bus.redis_health_check()
            else:
                # Fallback for InMemoryEventBus or no event_bus
                redis_healthy = True  # In-memory bus is always "healthy"
            
            # Check PolicyStore Redis health if available
            if self.policy_store and hasattr(self.policy_store, 'redis_health_check'):
                policy_redis_healthy = await self.policy_store.redis_health_check()
            else:
                policy_redis_healthy = True  # No Redis dependency
            
            if not redis_healthy or not policy_redis_healthy:
                logger.critical(
                    f"🚨 TRADING GATE: Infrastructure unhealthy - BLOCKING new trades\n"
                    f"   EventBus Redis: {'HEALTHY' if redis_healthy else 'UNAVAILABLE'}\n"
                    f"   PolicyStore Redis: {'HEALTHY' if policy_redis_healthy else 'UNAVAILABLE'}\n"
                    f"   [BLOCKED] Cannot open positions with infrastructure failures"
                )
                return
        except Exception as e:
            logger.error(f"Infrastructure health check failed: {e}", exc_info=True)
            # Fail-safe: block trading if health check fails
            logger.critical("🚨 TRADING GATE: Health check error - BLOCKING new trades")
            return
        
        # [NEW] SELF-HEALING: Check system health FIRST (directive override)
        if AI_INTEGRATION_AVAILABLE and self.ai_services:
            try:
                healing_directive = await periodic_self_healing_check()
                if healing_directive:
                    directive_type = healing_directive.get("directive")
                    
                    if directive_type == "NO_NEW_TRADES":
                        logger.warning(
                            f"[SELF-HEALING] 🛑 NO NEW TRADES directive active\n"
                            f"   Reason: {healing_directive.get('reason')}\n"
                            f"   Severity: {healing_directive.get('severity')}\n"
                            f"   [BLOCKED] Skipping signal check"
                        )
                        return
                    
                    elif directive_type == "DEFENSIVE_EXIT":
                        logger.warning(
                            f"[SELF-HEALING] ⚠️ DEFENSIVE EXIT directive\n"
                            f"   Reason: {healing_directive.get('reason')}\n"
                            f"   [ACTION] Existing positions should tighten stops"
                        )
                        # Continue but signal position_monitor to be defensive
                    
                    elif directive_type == "EMERGENCY_SHUTDOWN":
                        logger.critical(
                            f"[SELF-HEALING] 🚨 EMERGENCY SHUTDOWN directive\n"
                            f"   Reason: {healing_directive.get('reason')}\n"
                            f"   [CRITICAL] Stopping executor"
                        )
                        self._running = False
                        return
                        
            except Exception as e:
                logger.error(f"[ERROR] Self-Healing check failed: {e}", exc_info=True)
                # Fail-safe: continue with caution
        
        # [NEW] AI-HFOS: Check trading permissions and risk mode
        ai_hfos_allow_trades = True
        ai_hfos_risk_mode = "NORMAL"
        ai_hfos_confidence_multiplier = 1.0
        
        if AI_INTEGRATION_AVAILABLE and self.ai_services:
            try:
                coordination_result = await periodic_ai_hfos_coordination()
                if coordination_result:
                    ai_hfos_allow_trades = coordination_result.get("allow_new_trades", True)
                    ai_hfos_risk_mode = coordination_result.get("risk_mode", "NORMAL")
                    ai_hfos_confidence_multiplier = coordination_result.get("confidence_multiplier", 1.0)
                    
                    if not ai_hfos_allow_trades:
                        logger.warning(
                            f"[AI-HFOS] 🛑 Trading BLOCKED by supreme coordinator\n"
                            f"   Risk Mode: {ai_hfos_risk_mode}\n"
                            f"   Directive: No new trades allowed\n"
                            f"   [SKIP] Exiting signal check"
                        )
                        return
                    
                    if ai_hfos_risk_mode in ["AGGRESSIVE", "CRITICAL"]:
                        logger.warning(
                            f"[AI-HFOS] ⚠️ Elevated risk mode: {ai_hfos_risk_mode}\n"
                            f"   Confidence multiplier: {ai_hfos_confidence_multiplier:.2f}\n"
                            f"   [CAUTION] Trading with increased scrutiny"
                        )
                        
            except Exception as e:
                logger.error(f"[ERROR] AI-HFOS coordination failed: {e}", exc_info=True)
                # Fail-safe: allow trades but log warning
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 2: COOLDOWN CHECK
        # ═══════════════════════════════════════════════════════════════════════
        
        # Enforce cooldown between rebalances
        if self._last_rebalance_time:
            time_since_last = (now - self._last_rebalance_time).total_seconds()
            if time_since_last < self.cooldown:
                logger.info("⏸️ Still in cooldown (%ds left)", self.cooldown - time_since_last)
                return
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 3: UNIVERSE FILTERING
        # ═══════════════════════════════════════════════════════════════════════
        
        # [NEW] UNIVERSE OS: Filter symbols through dynamic universe and blacklist
        symbols_to_check = self.symbols
        if AI_INTEGRATION_AVAILABLE and self.ai_services:
            try:
                filtered_symbols = await pre_trade_universe_filter(self.symbols)
                if filtered_symbols != self.symbols:
                    logger.info(
                        f"[UNIVERSE OS] Symbol filter: {len(self.symbols)} → {len(filtered_symbols)} symbols\n"
                        f"   [REMOVED] {set(self.symbols) - set(filtered_symbols)}"
                    )
                symbols_to_check = filtered_symbols
                
                # Check if NO symbols passed filter
                if not symbols_to_check:
                    logger.warning(
                        f"[UNIVERSE OS] ⚠️ All symbols filtered out - no trading universe available\n"
                        f"   [SKIP] Exiting signal check"
                    )
                    return
                    
            except Exception as e:
                logger.error(f"[ERROR] Universe filter failed: {e}", exc_info=True)
                # Fail-safe: use all symbols
                symbols_to_check = self.symbols
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 4: AI SIGNAL GENERATION
        # ═══════════════════════════════════════════════════════════════════════
        
        # Get AI signals for all symbols
        logger.info("[SIGNAL] Calling get_trading_signals for %d symbols", len(symbols_to_check))
        try:
            # get_trading_signals expects (symbols, current_positions)
            # For monitoring, we pass empty positions since we check portfolio during rebalance
            signals_list = await self.ai_engine.get_trading_signals(symbols_to_check, {})
            logger.info("Got %d AI signals from engine", len(signals_list))
        except Exception as e:
            logger.error("Failed to get AI signals: %s", e, exc_info=True)
            return
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 4.1: CONSENSUS FILTERING - Require Strong Model Agreement
        # ═══════════════════════════════════════════════════════════════════════
        
        # [NEW] QUALITY FILTER: Only trade signals with "strong" consensus (3/4 models agree)
        filtered_signals = []
        for signal in signals_list:
            model_metadata = signal.get("model", {})
            
            # Check if this is an ensemble signal with consensus info
            if isinstance(model_metadata, dict):
                consensus_type = model_metadata.get("consensus", "")
                consensus_count = model_metadata.get("consensus_count", 0)
                
                # Require "strong" consensus (3+ models agree)
                if consensus_type == "strong" and consensus_count >= 3:
                    filtered_signals.append(signal)
                    logger.debug(
                        f"[OK] {signal.get('symbol')}: Strong consensus "
                        f"({consensus_count}/4 models agree on {signal.get('action')})"
                    )
                else:
                    logger.debug(
                        f"[BLOCKED] {signal.get('symbol')}: Weak consensus "
                        f"(type={consensus_type}, count={consensus_count}) - FILTERED OUT"
                    )
            else:
                # Not an ensemble signal - allow through (strategy signals, etc)
                filtered_signals.append(signal)
        
        if len(signals_list) > len(filtered_signals):
            logger.info(
                f"[CONSENSUS FILTER] Removed {len(signals_list) - len(filtered_signals)} "
                f"split/weak consensus signals ({len(filtered_signals)} remaining)"
            )
        
        signals_list = filtered_signals
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 4.5: STRATEGY RUNTIME ENGINE - SG AI Generated Strategies
        # ═══════════════════════════════════════════════════════════════════════
        
        # [NEW] STRATEGY RUNTIME ENGINE: Get signals from AI-generated strategies
        strategy_signals = []
        try:
            from backend.services.strategy_runtime_integration import generate_strategy_signals
            
            # Get current market regime  
            current_regime = None
            if hasattr(self, 'regime_detector') and self.regime_detector:
                try:
                    # Get regime for first symbol as proxy
                    if symbols_to_check:
                        # symbols_to_check is a list of strings, pass directly
                        regime_info = self.regime_detector.detect_regime(symbols_to_check[0])
                        if isinstance(regime_info, dict):
                            current_regime = regime_info.get('regime_name')
                        elif isinstance(regime_info, str):
                            current_regime = regime_info
                except Exception as e:
                    logger.debug(f"[STRATEGY] Could not determine regime: {e}")
                    # Not critical, continue without regime
            
            # Generate signals from LIVE strategies
            strategy_decisions = generate_strategy_signals(
                symbols=symbols_to_check,
                current_regime=current_regime
            )
            
            if strategy_decisions:
                logger.info(
                    f"[STRATEGY] Generated {len(strategy_decisions)} signals from Strategy Runtime Engine"
                )
                
                # Convert TradeDecision to signal format compatible with AI signals
                for decision in strategy_decisions:
                    # [NEW] QUALITY FILTER: Skip low-confidence strategy signals
                    if decision.confidence < 0.60:
                        logger.debug(
                            f"[BLOCKED] Strategy {decision.strategy_id}: "
                            f"{decision.symbol} {decision.side} - "
                            f"Confidence {decision.confidence:.2f} < 0.60 threshold"
                        )
                        continue
                    
                    signal = {
                        "symbol": decision.symbol,
                        "action": decision.side.lower(),  # "long" or "short"
                        "confidence": decision.confidence,
                        "source": f"strategy:{decision.strategy_id}",
                        "strategy_id": decision.strategy_id,
                        "position_size_usd": decision.size_usd,  # Fixed: TradeDecision uses size_usd
                        "stop_loss": decision.stop_loss,  # Use absolute price, not pct
                        "take_profit": decision.take_profit,  # Use absolute price, not pct
                        "reasoning": decision.reasoning
                    }
                    strategy_signals.append(signal)
                    
                logger.info(
                    f"[STRATEGY] Converted to {len(strategy_signals)} signals:\n" +
                    "\n".join([
                        f"   • {s['symbol']}: {s['action'].upper()} "
                        f"@ {s['confidence']:.0%} confidence "
                        f"(${s['position_size_usd']:.0f}, strategy={s['strategy_id']})"
                        for s in strategy_signals[:5]  # Show first 5
                    ]) + (f"\n   ... and {len(strategy_signals) - 5} more" if len(strategy_signals) > 5 else "")
                )
            else:
                logger.debug("[STRATEGY] No signals from Strategy Runtime Engine")
                
        except ImportError:
            logger.debug("[STRATEGY] Strategy Runtime Engine not available (not imported)")
        except Exception as e:
            logger.error(f"[STRATEGY] Failed to get strategy signals: {e}", exc_info=True)
            # Continue without strategy signals - fail-safe
        
        # Merge AI signals with strategy signals - WITH CONFLICT RESOLUTION
        if strategy_signals:
            # CRITICAL FIX: Prioritize AI ensemble signals with strong consensus over strategy signals
            # If AI ensemble has "strong" consensus (3/4 models agree), it overrides strategy signals
            
            ai_strong_consensus_symbols = set()
            for signal in signals_list:
                model_metadata = signal.get("model", {})
                if isinstance(model_metadata, dict):
                    consensus_type = model_metadata.get("consensus", "")
                    if consensus_type == "strong":
                        ai_strong_consensus_symbols.add(signal.get("symbol"))
            
            # Filter out strategy signals that conflict with AI strong consensus
            filtered_strategy_signals = []
            for strat_sig in strategy_signals:
                symbol = strat_sig.get("symbol")
                
                # Check if AI has strong consensus for this symbol
                if symbol in ai_strong_consensus_symbols:
                    # Find the AI signal
                    ai_signal = next((s for s in signals_list if s.get("symbol") == symbol), None)
                    if ai_signal:
                        ai_action = ai_signal.get("action", "").upper()
                        strat_action = strat_sig.get("action", "").upper()
                        
                        # Check for conflict
                        if ai_action != strat_action:
                            logger.warning(
                                f"[CONFLICT] {symbol}: AI ensemble (strong consensus) says {ai_action}, "
                                f"but strategy {strat_sig.get('strategy_id')} says {strat_action}\n"
                                f"   → BLOCKING strategy signal - AI ensemble takes priority"
                            )
                            continue
                
                # No conflict or no AI strong consensus - allow strategy signal
                filtered_strategy_signals.append(strat_sig)
            
            blocked_count = len(strategy_signals) - len(filtered_strategy_signals)
            if blocked_count > 0:
                logger.warning(
                    f"[CONFLICT] Blocked {blocked_count} strategy signals due to AI ensemble conflicts"
                )
            
            signals_list.extend(filtered_strategy_signals)
            logger.info(
                f"[SIGNAL] Merged signals: {len(signals_list)} total "
                f"({len(signals_list) - len(filtered_strategy_signals)} AI + "
                f"{len(filtered_strategy_signals)} strategy, {blocked_count} conflicts resolved)"
            )
        
        # [TARGET] ORCHESTRATOR: Update trading policy based on all subsystems
        # In OBSERVATION MODE: compute policy but DON'T enforce it
        policy = None
        regime_tag = "TRENDING"  # Will be refined per-symbol later
        vol_level = "NORMAL"     # Will be refined per-symbol later
        risk_state = None
        symbol_perf_list = []
        cost_metrics = None
        
        if self.orch_config.enable_orchestrator:
            try:
                # Collect risk state
                risk_state = create_risk_state(
                    daily_pnl_pct=0.0,  # TODO: Integrate with actual PnL tracker
                    current_drawdown_pct=0.0,  # TODO: Integrate with actual DD tracker
                    losing_streak=0,  # TODO: Integrate with TradeLifecycleManager
                    open_trades_count=0,  # TODO: Track actual positions
                    total_exposure_pct=0.0  # TODO: Calculate from current positions
                )
                
                # Collect symbol performance data
                symbol_perf_list = []
                for sym in self.symbols:
                    stats = self.symbol_perf.get_stats(sym)
                    # Check if stats is valid before accessing attributes
                    if stats and hasattr(stats, 'trade_count') and stats.trade_count >= 3:
                        # Determine performance tag
                        if stats.win_rate < 0.35 or stats.avg_R < 0.5:
                            tag = "BAD"
                        elif stats.win_rate > 0.55 and stats.avg_R > 1.2:
                            tag = "GOOD"
                        else:
                            tag = "NEUTRAL"
                        
                        symbol_perf_list.append(create_symbol_performance(
                            symbol=sym,
                            winrate=stats.win_rate,
                            avg_R=stats.avg_R,
                            cumulative_pnl=stats.total_pnl,
                            performance_tag=tag
                        ))
                
                # Estimate current market costs (simplified for now)
                cost_metrics = create_cost_metrics(
                    spread_level="NORMAL",  # TODO: Calculate from recent spreads
                    slippage_level="NORMAL"  # TODO: Calculate from recent slippage
                )
                
                # Update policy
                policy = self.orchestrator.update_policy(
                    regime_tag=regime_tag,
                    vol_level=vol_level,
                    risk_state=risk_state,
                    symbol_performance=symbol_perf_list,
                    cost_metrics=cost_metrics
                )
                
                # [TARGET] FULL LIVE MODE: Pass policy to TradeLifecycleManager for ALL controls
                if (self.orch_config.use_for_risk_sizing or 
                    self.orch_config.use_for_exit_mode or 
                    self.orch_config.use_for_position_limits):
                    
                    try:
                        self._trade_manager.set_policy(policy)
                        log_parts = []
                        if self.orch_config.use_for_risk_sizing:
                            log_parts.append(f"max_risk_pct={policy.max_risk_pct:.2%}")
                            log_parts.append(f"risk_profile={policy.risk_profile}")
                        if self.orch_config.use_for_exit_mode:
                            log_parts.append(f"exit_mode={policy.exit_mode}")
                        if self.orch_config.use_for_position_limits:
                            log_parts.append(f"position_limits=ENFORCED")
                        logger.info(f"[TARGET] Policy passed to TradeManager: {', '.join(log_parts)}")
                    except Exception as e:
                        logger.error(f"[WARNING] Failed to set policy on TradeManager: {e}", exc_info=True)
                        # Continue with fallback defaults
                
                # [TARGET] FULL LIVE MODE: Enhanced policy logging with all subsystems
                if self.orch_config.is_live_mode():
                    logger.info(f"[RED_CIRCLE] FULL LIVE MODE - Policy ENFORCED: {policy.note}")
                    control_parts = [
                        f"allow_trades={policy.allow_new_trades}",
                        f"min_conf={policy.min_confidence:.2f}",
                        f"blocked_symbols={len(policy.disallowed_symbols)}",
                        f"risk_pct={policy.max_risk_pct:.2%}"
                    ]
                    if self.orch_config.use_for_exit_mode:
                        control_parts.append(f"exit_mode={policy.exit_mode}")
                    if self.orch_config.use_for_position_limits:
                        control_parts.append(f"position_limits=ACTIVE")
                    logger.info(f"[CLIPBOARD] Policy Controls: {', '.join(control_parts)}")
                    
                    # [ALERT] Highlight if trading is PAUSED
                    if not policy.allow_new_trades:
                        logger.warning(
                            f"[WARNING] TRADING PAUSED: {policy.note}\n"
                            f"   Reason: {policy.risk_profile}\n"
                            f"   Regime: {regime_tag} | Vol: {vol_level}"
                        )
                else:
                    # OBSERVE MODE: Log policy but DON'T enforce it
                    logger.info(f"[EYE] OBSERVE MODE - Policy computed but NOT enforced: {policy.note}")
                
            except Exception as e:
                logger.error(f"[WARNING] Orchestrator policy update failed: {e}", exc_info=True)
                policy = None
                
                # [SHIELD] SAFETY FALLBACK: Create safe default policy if orchestrator fails
                logger.warning("[SHIELD] Using SAFE FALLBACK policy due to orchestrator failure")
                from backend.services.governance.orchestrator_policy import TradingPolicy
                policy = TradingPolicy(
                    allow_new_trades=True,  # Allow trading but with conservative settings
                    min_confidence=0.65,     # Higher threshold for safety
                    max_risk_pct=0.01,       # Conservative 1% risk
                    allowed_symbols=[],      # Empty = allow all
                    disallowed_symbols=[],   # Empty = block none
                    exit_mode="DEFENSIVE_TRAIL",  # Most conservative exit
                    risk_profile="FALLBACK",
                    note="FALLBACK: Orchestrator failed, using safe defaults"
                )
        
        # [TARGET] DETERMINE EFFECTIVE SETTINGS based on mode and config
        effective_confidence = self.confidence_threshold  # Default: 0.45
        actual_trading_allowed = True  # Default: allowed
        
        # [SHIELD] SAFETY CHECK: Ensure policy exists before applying
        if policy is None:
            logger.warning(
                "[WARNING] No policy available - using system defaults\n"
                "   confidence_threshold=0.45\n"
                "   trading_allowed=True\n"
                "   risk_controls=DEFAULT"
            )
        
        # Apply policy overrides if in LIVE mode AND policy exists
        elif policy and self.orch_config.is_live_mode():
            # Step 1: Confidence threshold enforcement
            if self.orch_config.use_for_confidence_threshold:
                effective_confidence = policy.min_confidence
                logger.info(
                    f"[OK] Policy confidence active: {effective_confidence:.2f} "
                    f"(default: {self.confidence_threshold:.2f})"
                )
        
        # [NEW] AI-HFOS CONFIDENCE ADJUSTMENT: Further adjust confidence based on system risk mode
        if AI_INTEGRATION_AVAILABLE and self.ai_services and signals_list:
            try:
                adjusted_confidence = await pre_trade_confidence_adjustment(
                    signals_list[0] if signals_list else None,
                    effective_confidence
                )
                if adjusted_confidence != effective_confidence:
                    logger.info(
                        f"[AI-HFOS] Confidence adjusted: {effective_confidence:.2f} → {adjusted_confidence:.2f}"
                    )
                    effective_confidence = adjusted_confidence
            except Exception as e:
                logger.error(f"[ERROR] Confidence adjustment failed: {e}", exc_info=True)
                # Fail-safe: keep existing confidence
            
            # Step 4: Trading gate enforcement - BLOCK NEW TRADES IN DANGEROUS CONDITIONS
            if self.orch_config.use_for_trading_gate:
                actual_trading_allowed = policy.allow_new_trades
                if not actual_trading_allowed:
                    # [ALERT] CRITICAL: Trading is SHUT DOWN
                    logger.warning(
                        f"[ALERT] TRADE SHUTDOWN ACTIVE [ALERT]\n"
                        f"   Reason: {policy.note}\n"
                        f"   Risk Profile: {policy.risk_profile}\n"
                        f"   Regime: {regime_tag} | Vol: {vol_level}\n"
                        f"   🛑 NO NEW TRADES - Exits only\n"
                        f"   ⏳ Will check for recovery in next cycle"
                    )
                else:
                    logger.debug("[OK] Trading gate: OPEN (new trades allowed)")
            
            # Step 5: Position limits enforcement
            if self.orch_config.use_for_position_limits:
                logger.debug("[OK] Position limits: ACTIVE (per-symbol caps enforced)")
        
        # 🛑 EARLY EXIT: If trading gate is closed, skip signal processing entirely
        if not actual_trading_allowed:
            logger.info(
                "[SKIP] Skipping signal processing - trading gate CLOSED\n"
                "   [OK] Existing positions continue to be monitored\n"
                "   [OK] Exits will be processed normally\n"
                "   [BLOCKED] New entries BLOCKED"
            )
            # Continue loop (return will skip to next iteration after cooldown)
            return
        
        # Check for high-confidence signals
        strong_signals = []
        
        # [SPRINT 5 - PATCH #3] Apply signal queue throttling
        signals_to_process = []
        for signal in signals_list:
            symbol = signal.get("symbol", "")
            confidence = abs(signal.get("confidence", 0.0))
            action = signal.get("action", "HOLD")
            
            # Skip HOLD actions early
            if action == "HOLD":
                continue
            
            # Check if queue is full
            if len(self._signal_queue) >= self._signal_queue_max_size:
                # Find lowest confidence signal in queue
                if self._signal_queue:
                    min_confidence_in_queue = min(s.get("confidence", 0.0) for s in self._signal_queue)
                    
                    # Only replace if new signal has higher confidence
                    if confidence > min_confidence_in_queue:
                        # Remove lowest confidence signal
                        min_signal = min(self._signal_queue, key=lambda s: s.get("confidence", 0.0))
                        self._signal_queue.remove(min_signal)
                        self._signal_queue.append(signal)
                        self._dropped_signals_count += 1
                        logger.warning(
                            f"[THROTTLE] Signal queue full ({self._signal_queue_max_size}), "
                            f"replaced low confidence signal (conf={min_confidence_in_queue:.2f}) "
                            f"with {symbol} {action} (conf={confidence:.2f})"
                        )
                    else:
                        self._dropped_signals_count += 1
                        logger.warning(
                            f"[THROTTLE] Signal queue full, dropped {symbol} {action} "
                            f"(conf={confidence:.2f} < queue_min={min_confidence_in_queue:.2f})"
                        )
                        continue
            else:
                # Queue has space, add signal
                self._signal_queue.append(signal)
        
        # Process signals from queue (rate-limited)
        signals_processed_this_cycle = 0
        max_signals_per_cycle = int(os.getenv("QT_MAX_SIGNALS_PER_CYCLE", "10"))
        
        while self._signal_queue and signals_processed_this_cycle < max_signals_per_cycle:
            signal = self._signal_queue.popleft()
            signals_to_process.append(signal)
            signals_processed_this_cycle += 1
        
        if signals_processed_this_cycle >= max_signals_per_cycle and self._signal_queue:
            logger.info(
                f"[THROTTLE] Processed {signals_processed_this_cycle} signals this cycle, "
                f"{len(self._signal_queue)} signals remain in queue"
            )
        
        # Now process the throttled signal list
        for signal in signals_to_process:
            symbol = signal.get("symbol", "")
            confidence = abs(signal.get("confidence", 0.0))
            action = signal.get("action", "HOLD")
            model = signal.get("model", "unknown")
            
            # Skip HOLD actions
            if action == "HOLD":
                continue
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 5: SIGNAL-LEVEL AI-OS PROCESSING
            # ═══════════════════════════════════════════════════════════════════════
            
            # [NEW] DYNAMIC TP/SL: Calculate AI-driven TP/SL based on confidence
            tp_percent = signal.get("tp_percent", 0.06)      # Fallback: 6%
            sl_percent = signal.get("sl_percent", 0.08)      # Fallback: 8%
            trail_percent = signal.get("trail_percent", 0.02) # Fallback: 2%
            partial_tp = signal.get("partial_tp", 0.5)       # Fallback: 50%
            
            if AI_INTEGRATION_AVAILABLE and self.ai_services:
                try:
                    if (self.ai_services.config.dynamic_tpsl_enabled and 
                        self.ai_services.config.dynamic_tpsl_override_legacy):
                        
                        dynamic_calc = self.ai_services.dynamic_tpsl
                        if dynamic_calc:
                            tpsl_result = dynamic_calc.calculate(
                                signal_confidence=confidence,
                                signal_action=action,
                                base_tp_pct=tp_percent,
                                base_sl_pct=sl_percent
                            )
                            
                            # Override legacy TP/SL with AI-calculated values
                            if tpsl_result.success:
                                tp_percent = tpsl_result.tp_pct
                                sl_percent = tpsl_result.sl_pct
                                logger.info(
                                    f"[DYNAMIC TP/SL] {symbol}: "
                                    f"TP={tp_percent:.2%} SL={sl_percent:.2%} "
                                    f"(R:R={tpsl_result.risk_reward_ratio:.2f}, "
                                    f"mode={ai_hfos_risk_mode})"
                                )
                            else:
                                logger.warning(
                                    f"[DYNAMIC TP/SL] {symbol}: Calculation failed, using fallback"
                                )
                                
                except Exception as e:
                    logger.error(f"[ERROR] Dynamic TP/SL calculation failed: {e}", exc_info=True)
                    # Fail-safe: use original TP/SL
            
            # [NEW] AI-HFOS: Apply confidence multiplier from risk mode
            adjusted_confidence = confidence * ai_hfos_confidence_multiplier
            
            # [MSC AI] ENFORCE STRATEGY FILTERING (HIGHEST PRIORITY)
            if MSC_AI_AVAILABLE and msc_policy:
                allowed_strategies = msc_policy.get('allowed_strategies', [])
                strategy_id = signal.get('strategy_id')
                
                # If signal has strategy_id (from Strategy Runtime), check if allowed
                if strategy_id and allowed_strategies:
                    if strategy_id not in allowed_strategies:
                        logger.info(
                            f"[MSC AI] BLOCKED: {symbol} {action} - "
                            f"Strategy {strategy_id} not in MSC AI allowed list: {allowed_strategies}"
                        )
                        if policy and self.orch_config.log_all_signals:
                            self.policy_observer.log_signal_decision(
                                signal=signal,
                                policy=policy,
                                decision="BLOCKED_BY_MSC_AI",
                                reason=f"Strategy {strategy_id} not in MSC AI allowed_strategies"
                            )
                        continue
            
            # [RED_CIRCLE] LIVE MODE: ENFORCE POLICY-BASED SYMBOL FILTERING (Step 1)
            if policy and self.orch_config.is_live_mode() and self.orch_config.use_for_signal_filter:
                # Block disallowed symbols
                if symbol in policy.disallowed_symbols:
                    logger.info(
                        f"[BLOCKED] BLOCKED by policy: {symbol} {action} (conf={adjusted_confidence:.2f}) - "
                        f"Symbol in disallowed list"
                    )
                    if self.orch_config.log_all_signals:
                        self.policy_observer.log_signal_decision(
                            signal=signal,
                            policy=policy,
                            decision="BLOCKED_BY_POLICY_FILTER",
                            reason=f"Symbol in policy.disallowed_symbols"
                        )
                    continue
                
                # Enforce allowed_symbols if specified (non-empty)
                if policy.allowed_symbols and symbol not in policy.allowed_symbols:
                    logger.info(
                        f"[BLOCKED] BLOCKED by policy: {symbol} {action} (conf={adjusted_confidence:.2f}) - "
                        f"Symbol not in allowed list"
                    )
                    if self.orch_config.log_all_signals:
                        self.policy_observer.log_signal_decision(
                            signal=signal,
                            policy=policy,
                            decision="BLOCKED_BY_POLICY_FILTER",
                            reason=f"Symbol not in policy.allowed_symbols"
                        )
                    continue
            
            # [EYE] OBSERVATION MODE: Log what policy WOULD do (but don't enforce)
            elif policy and self.orch_config.is_observe_mode() and self.orch_config.log_all_signals:
                would_block_symbol = symbol in policy.disallowed_symbols
                # Use <= to block when below threshold (inverse of >= allow logic)
                would_block_confidence = confidence < policy.min_confidence
                would_block_trading = not policy.allow_new_trades
                
                if would_block_symbol or would_block_confidence or would_block_trading:
                    # Policy would block, but we're in observe mode so we don't enforce
                    self.policy_observer.log_signal_decision(
                        signal=signal,
                        policy=policy,
                        decision="PROCEEDING_DESPITE_POLICY",
                        reason=f"OBSERVE mode: would block (symbol={would_block_symbol}, conf={would_block_confidence}, gate={would_block_trading})"
                    )
            
            # [TARGET] SYMBOL PERFORMANCE FILTER: Skip disabled symbols (THIS IS STILL ENFORCED)
            if not self.symbol_perf.should_trade_symbol(symbol):
                stats = self.symbol_perf.get_stats(symbol)
                logger.debug(
                    f"[SKIP] Skipping disabled symbol {symbol} "
                    f"(WR={stats.win_rate:.1%}, consecutive_losses={stats.consecutive_losses})"
                )
                if policy and self.orch_config.log_all_signals:
                    self.policy_observer.log_signal_decision(
                        signal=signal,
                        policy=policy,
                        decision="BLOCKED_BY_SYMBOL_PERF",
                        reason=f"SymbolPerformanceManager disabled (WR={stats.win_rate:.1%})"
                    )
                continue
            
            # [NEW] FUNDING RATE FILTER: Block symbols with excessive funding costs
            # Calculate position size for funding check
            try:
                account = await self._adapter.get_account_balance()
                balance_usdt = float(account.get('availableBalance', 0))
                risk_modifier = self.symbol_perf.get_risk_modifier(symbol)
                position_risk = self._risk_config.get('position_risk', 0.01) * risk_modifier
                position_size_usdt = balance_usdt * position_risk * 30  # 30x leverage
                
                is_long = (action == "BUY")
                should_block, reason = self.funding_filter.should_block_trade(
                    symbol=symbol,
                    position_size_usdt=position_size_usdt,
                    is_long=is_long
                )
                
                if should_block:
                    logger.warning(f"[BLOCKED] {symbol}: {reason}")
                    if policy and self.orch_config.log_all_signals:
                        self.policy_observer.log_signal_decision(
                            signal=signal,
                            policy=policy,
                            decision="BLOCKED_BY_FUNDING_RATE",
                            reason=reason
                        )
                    continue
                elif "High funding" in reason or "Warning" in reason:
                    logger.info(f"[FUNDING] {symbol}: {reason}")
            except Exception as e:
                logger.debug(f"Could not check funding rate for {symbol}: {e}")
            
            # [OK] CONFIDENCE FILTER: Apply effective_confidence (policy-controlled in LIVE mode)
            # Using >= comparison: signals AT or ABOVE threshold are allowed
            # Rationale: A signal with exactly threshold confidence should pass
            if adjusted_confidence >= effective_confidence:
                # Store FULL signal with TP/SL data + risk modifier
                risk_modifier = self.symbol_perf.get_risk_modifier(symbol)
                strong_signals.append({
                    "symbol": symbol,
                    "action": action,
                    "confidence": adjusted_confidence,  # Use AI-HFOS adjusted confidence
                    "original_confidence": confidence,  # Store original for reference
                    "model": model,
                    "tp_percent": tp_percent,           # Dynamic TP/SL if enabled
                    "sl_percent": sl_percent,           # Dynamic TP/SL if enabled
                    "trail_percent": trail_percent,
                    "partial_tp": partial_tp,
                    "risk_modifier": risk_modifier      # 0.5 for poor performers, 1.0 for normal
                })
                
                # [NEW] MODEL SUPERVISOR: Observe signal for bias detection
                if AI_INTEGRATION_AVAILABLE:
                    try:
                        ai_services = get_ai_services()
                        if (ai_services._initialized and 
                            ai_services.model_supervisor and
                            ai_services.config.model_supervisor_mode != SubsystemMode.OFF):
                            
                            ai_services.model_supervisor.observe(
                                signal={
                                    "symbol": symbol,
                                    "action": action,
                                    "confidence": adjusted_confidence,
                                    "original_confidence": confidence,
                                    "regime": regime_tag,
                                    "model_predictions": model if isinstance(model, dict) else None
                                }
                            )
                    except Exception as e:
                        logger.debug(f"[MODEL_SUPERVISOR] observe() failed: {e}")
                
                # Log successful signal with regime context
                if policy and self.orch_config.log_all_signals:
                    policy_enforced = self.orch_config.is_live_mode() and self.orch_config.use_for_confidence_threshold
                    logger.info(
                        f"[ALLOWED] {symbol} {action} "
                        f"(conf={adjusted_confidence:.2f} >= {effective_confidence:.2f}) - "
                        f"Regime: {regime_tag} | Vol: {vol_level} | "
                        f"Policy: {'ENFORCED' if policy_enforced else 'NOT enforced'}"
                    )
                    self.policy_observer.log_signal_decision(
                        signal=signal,
                        policy=policy,
                        decision="TRADE_ALLOWED",
                        reason=f"Passed all filters (conf={adjusted_confidence:.2f} >= {effective_confidence:.2f}) [Policy {'ENFORCED' if policy_enforced else 'NOT enforced'}]"
                    )
            else:
                # [RED_CIRCLE] BLOCKED BY CONFIDENCE (policy-controlled in LIVE mode)
                # Using < comparison: signals BELOW threshold are blocked
                policy_enforced = (
                    policy and 
                    self.orch_config.is_live_mode() and 
                    self.orch_config.use_for_confidence_threshold
                )
                
                if policy_enforced:
                    # Log with regime context for better diagnostics
                    logger.info(
                        f"[BLOCKED] BLOCKED by policy: {symbol} {action} (conf={confidence:.2f}) - "
                        f"Below min_confidence={policy.min_confidence:.2f} | "
                        f"Regime: {regime_tag} | Vol: {vol_level}"
                    )
                
                if policy and self.orch_config.log_all_signals:
                    self.policy_observer.log_signal_decision(
                        signal=signal,
                        policy=policy,
                        decision="BLOCKED_BY_POLICY_FILTER" if policy_enforced else "BLOCKED_BY_CONFIDENCE",
                        reason=f"Confidence {confidence:.2f} < threshold {effective_confidence:.2f} [Policy {'ENFORCED' if policy_enforced else 'NOT enforced'}]"
                    )
        
        logger.info("Found %d high-confidence signals (>= %.2f)", len(strong_signals), effective_confidence)
        
        # [EYE] OBSERVATION: Log complete policy observation
        if policy and self.orch_config.enable_orchestrator:
            try:
                self.policy_observer.log_policy_update(
                    policy=policy,
                    regime_tag=regime_tag,
                    vol_level=vol_level,
                    risk_state=risk_state,
                    symbol_performance=symbol_perf_list,
                    cost_metrics=cost_metrics,
                    signals_before_filter=signals_list,
                    actual_confidence_used=effective_confidence,
                    actual_trading_allowed=actual_trading_allowed
                )
            except Exception as e:
                logger.error(f"Failed to log policy observation: {e}", exc_info=True)
        
        if not strong_signals:
            # Provide a quick snapshot of the best BUY/SELL confidences observed
            best_buy = 0.0
            best_sell = 0.0
            for s in signals_list:
                act = s.get("action", "HOLD")
                conf = abs(float(s.get("confidence", 0.0)))
                if act == "BUY":
                    best_buy = max(best_buy, conf)
                elif act == "SELL":
                    best_sell = max(best_sell, conf)
            logger.debug(
                "No strong signals (thr=%.2f). Best BUY=%.2f, SELL=%.2f across %d symbols",
                self.confidence_threshold, best_buy, best_sell, len(signals_list)
            )
            return
        
        # Sort by confidence (highest first) and take top opportunities
        strong_signals.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Use QT_MAX_POSITIONS for signal selection limit (not hardcoded 5)
        max_signal_selection = int(os.getenv("QT_MAX_POSITIONS", "20"))
        top_signals = strong_signals[:max_signal_selection]
        
        if len(strong_signals) > max_signal_selection:
            logger.info(
                "[CHART] Found %d strong signals, selecting top %d by confidence",
                len(strong_signals), max_signal_selection
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # PHASE 6: PORTFOLIO-LEVEL AI-OS PROCESSING
        # ═══════════════════════════════════════════════════════════════════════
        
        # [NEW] PORTFOLIO BALANCER: Pre-trade filtering through PBA
        if AI_INTEGRATION_AVAILABLE and self.ai_services:
            # MICRO HOTFIX: Only call PBA if we have signals
            if not top_signals:
                logger.info("[PBA] No signals available, skipping portfolio balancing")
                return
            try:
                # Get current positions for PBA check
                current_positions = {}
                if self._adapter:
                    raw_positions = await self._adapter.get_positions()
                    current_positions = {sym: qty for sym, qty in raw_positions.items() if float(qty) != 0}
                
                # Filter signals through PBA - check each signal individually
                filtered_signals = []
                for sig in top_signals:
                    allowed, reason = await pre_trade_portfolio_check(
                        sig["symbol"], 
                        sig, 
                        current_positions
                    )
                    if allowed:
                        filtered_signals.append(sig)
                    else:
                        logger.warning(f"[PBA] Blocked {sig['symbol']} {sig['action']}: {reason}")
                
                blocked_count = len(top_signals) - len(filtered_signals)
                if blocked_count > 0:
                    blocked_symbols = set(s["symbol"] for s in top_signals) - set(s["symbol"] for s in filtered_signals)
                    logger.warning(
                        f"[PBA] Portfolio Balancer blocked {blocked_count}/{len(top_signals)} signals: {blocked_symbols}"
                    )
                
                top_signals = filtered_signals
                    
            except Exception as e:
                logger.error(f"[ERROR] Portfolio Balancer check failed: {e}", exc_info=True)
                # Fail-safe: continue with original signals
        
        # [NEW] PORTFOLIO BALANCER: Pre-trade filtering
        if AI_INTEGRATION_AVAILABLE:
            try:
                from fastapi import FastAPI
                app = FastAPI()
                
                # Check if PBA is available in app state
                if hasattr(app.state, "portfolio_balancer"):
                    pba = app.state.portfolio_balancer
                    
                    # Get current positions
                    if self._adapter:
                        raw_positions = await self._adapter.get_positions()
                        positions_data = []
                        
                        for symbol, qty in raw_positions.items():
                            if float(qty) != 0:
                                from backend.services.portfolio_balancer import Position
                                positions_data.append(Position(
                                    symbol=symbol,
                                    side="LONG" if float(qty) > 0 else "SHORT",
                                    size=abs(float(qty)),
                                    entry_price=0.0,  # Would need to fetch
                                    current_price=0.0,  # Would need to fetch
                                    margin=0.0,
                                    leverage=20
                                ))
                        
                        # Convert signals to candidates
                        from backend.services.portfolio_balancer import CandidateTrade
                        candidates = []
                        for sig in top_signals:
                            candidates.append(CandidateTrade(
                                symbol=sig["symbol"],
                                action=sig["action"],
                                confidence=sig["confidence"],
                                size=1.0,
                                margin_required=100.0
                            ))
                        
                        # Analyze portfolio
                        if positions_data or candidates:
                            output = pba.analyze_portfolio(
                                positions=positions_data,
                                candidates=candidates,
                                total_equity=10000,  # Would need to fetch
                                used_margin=1000,
                                free_margin=9000
                            )
                            
                            # Filter out blocked trades
                            if output.dropped_trades:
                                blocked_symbols = {t.symbol for t in output.dropped_trades}
                                top_signals = [s for s in top_signals if s["symbol"] not in blocked_symbols]
                                
                                logger.warning(
                                    f"⚖️ [PBA] Blocked {len(output.dropped_trades)} trades: "
                                    f"{', '.join(blocked_symbols)}"
                                )
                            
                            if output.allowed_trades:
                                logger.info(
                                    f"⚖️ [PBA] Allowed {len(output.allowed_trades)} trades after portfolio analysis"
                                )
                    
            except Exception as e:
                logger.debug(f"[PBA] Portfolio balancer check failed: {e}")
        
        if not top_signals:
            logger.info("⚖️ [PBA] All trades blocked by Portfolio Balancer")
            return
        
        # Strong signals detected - execute orders
        logger.info(
            "[TARGET] Strong signals: %s",
            ", ".join(f"{s['symbol']}={s['action']}({s['confidence']:.2f},{s['model']})" for s in top_signals)
        )
        
        # OPTIMIZATION: Use direct execution for faster order placement
        if self._direct_execute:
            logger.info("⚡ Direct execution mode - placing orders immediately")
            result = await self._execute_signals_direct(top_signals)
        else:
            # Fallback to traditional rebalancing (slower)
            logger.info("🔄 Using portfolio rebalancing (legacy mode)")
            with SessionLocal() as db:
                result = await run_portfolio_rebalance(db)
        
        # Log raw result for debugging
        logger.info("[SEARCH] Execution result: %s", result)
        
        # Check for success (status can be "ok" or "success")
        status = result.get("status", "")
        if status in ("ok", "success"):
            num_orders = result.get("orders_submitted", 0)
            orders_planned = result.get("orders_planned", 0)
            orders_skipped = result.get("orders_skipped", 0)
            orders_failed = result.get("orders_failed", 0)
            
            # [FIXED] Only start cooldown if trades were actually executed
            if num_orders > 0:
                self._last_rebalance_time = now
                logger.info(
                    "[OK] ✅ Execution complete: planned=%d submitted=%d skipped=%d failed=%d | Cooldown started (%ds)",
                    orders_planned, num_orders, orders_skipped, orders_failed, self.cooldown
                )
            elif orders_skipped > 0:
                logger.warning(
                    "[WARNING] All orders skipped (planned=%d skipped=%d) - NO cooldown, will check again in %ds",
                    orders_planned, orders_skipped, self.check_interval
                )
            else:
                logger.info("ℹ️ No orders to execute - NO cooldown, continuing to scan for signals")
        else:
            error_msg = result.get("error", status or "unknown")
            logger.warning("[WARNING] Execution failed: status=%s error=%s", status, error_msg)

    async def _execute_signals_direct(self, signals: List[tuple]) -> Dict[str, any]:
        """
        FAST PATH: Execute orders directly from signals without portfolio rebalancing.
        
        Args:
            signals: List of (symbol, action, confidence) tuples
            
        Returns:
            Dict with execution results
        """
        if self._adapter is None:
            self._adapter = build_execution_adapter(self._execution_config)
        
        try:
            # Get current positions to check available slots
            raw_positions = await self._adapter.get_positions()
            positions = {sym.upper(): abs(float(qty)) for sym, qty in raw_positions.items() if float(qty) != 0}
            open_positions = len(positions)
            
            # [NEW] HEDGEFUND MODE: Dynamic max_positions based on AI-HFOS risk mode
            base_max_positions = int(os.getenv("QT_MAX_POSITIONS", "4"))
            max_positions = base_max_positions
            
            # Check AI-HFOS risk mode for AGGRESSIVE scaling
            if self._app_state and hasattr(self._app_state, "ai_hfos_directives"):
                hfos_directives = self._app_state.ai_hfos_directives
                if hfos_directives:
                    risk_mode = hfos_directives.get("risk_mode", "NORMAL")
                    if risk_mode == "AGGRESSIVE":
                        # Allow 2.5x more positions in AGGRESSIVE mode
                        max_positions = int(base_max_positions * 2.5)
                        logger.info(f"🚀 [HEDGEFUND MODE] AGGRESSIVE: max_positions scaled to {max_positions}")
                    elif risk_mode == "CRITICAL":
                        # Reduce to 50% in CRITICAL (damage control)
                        max_positions = max(1, int(base_max_positions * 0.5))
                        logger.warning(f"⚠️ [CRITICAL MODE] max_positions reduced to {max_positions}")
            
            # SafetyGovernor can override (highest priority)
            if self._app_state and hasattr(self._app_state, "safety_governor_directives"):
                safety_directives = self._app_state.safety_governor_directives
                if safety_directives:
                    if not safety_directives.global_allow_new_trades:
                        max_positions = open_positions  # No new positions allowed
                        logger.warning("🛡️ [SafetyGovernor] New trades blocked - max_positions = current positions")
            
            available_slots = max(0, max_positions - open_positions)
            
            # [SAFETY] Get max positions per symbol limit
            from config.config import get_max_positions_per_symbol
            max_per_symbol = get_max_positions_per_symbol()
            
            # Count positions per symbol for stacking check
            from collections import Counter
            symbol_position_counts = Counter()
            for sym in positions.keys():
                symbol_position_counts[sym] += 1
            
            logger.info(
                f"[BRIEFCASE] Current positions: {open_positions}/{max_positions}, "
                f"available: {available_slots}, max per symbol: {max_per_symbol}"
            )
            if symbol_position_counts:
                logger.info(f"[BRIEFCASE] Per-symbol counts: {dict(symbol_position_counts)}")
            
            if available_slots == 0:
                logger.warning("[WARNING] Max positions reached, skipping new orders")
                return {
                    "status": "ok",
                    "orders_planned": len(signals),
                    "orders_submitted": 0,
                    "orders_skipped": len(signals),
                    "orders_failed": 0,
                    "reason": "max_positions_reached"
                }
            
            # Get account balance for position sizing
            cash = await self._adapter.get_cash_balance()
            # max_notional is MARGIN, not position size
            # With 30x leverage: $5000 margin = $150,000 position
            max_margin_per_trade = self._risk_config.max_notional_per_trade or 5000.0
            
            # [TARGET] TESTNET MODE: Use 10% of balance per trade (not 25%)
            # This allows opening positions even with small balance
            margin_pct = 0.10 if cash < 100 else 0.25  # 10% if balance < $100, else 25%
            margin_from_balance = cash * margin_pct
            actual_margin = min(max_margin_per_trade, margin_from_balance)
            
            logger.info(
                f"[MONEY] Cash: ${cash:.2f}, Margin per trade: ${actual_margin:.2f} "
                f"({margin_pct*100:.0f}% of balance, max ${max_margin_per_trade:.2f})"
            )
            
            # Place orders for top signals (up to available slots)
            orders_to_place = signals[:available_slots]
            orders_submitted = 0
            orders_failed = 0
            orders_skipped = 0
            
            for signal_dict in orders_to_place:
                # Extract signal data
                symbol = signal_dict["symbol"]
                action = signal_dict["action"]
                confidence = signal_dict["confidence"]
                model = signal_dict["model"]
                risk_modifier = signal_dict.get("risk_modifier", 1.0)
                
                # Extract TP/SL/Trail data from signal (needed for checks below)
                tp_percent = signal_dict.get("tp_percent", 0.06)  # Default 6%
                sl_percent = signal_dict.get("sl_percent", 0.03)  # Default 3%
                trail_percent = signal_dict.get("trail_percent", 0.005)  # Default 0.5%
                partial_tp = signal_dict.get("partial_tp", False)
                
                # [SAFETY] Check per-symbol position limit
                if symbol_position_counts.get(symbol, 0) >= max_per_symbol:
                    logger.warning(
                        f"[SAFETY] Skipping {symbol}: Already at max {max_per_symbol} positions for this symbol"
                    )
                    orders_skipped += 1
                    continue
                
                # [SAFETY GOVERNOR] CHECK GLOBAL SAFETY DIRECTIVES BEFORE TRADE
                # This is the ULTIMATE safety layer that wraps all AI decisions
                try:
                    from fastapi import Request
                    # Access app.state through a stored reference
                    if hasattr(self, '_app_state') and hasattr(self._app_state, 'safety_governor'):
                        safety_governor = self._app_state.safety_governor
                        directives = getattr(self._app_state, 'safety_governor_directives', None)
                        
                        if directives:
                            # Check if new trades are globally allowed
                            if not directives.global_allow_new_trades:
                                logger.error(
                                    f"🛡️ [SAFETY GOVERNOR] ❌ BLOCKED: NEW_TRADE {symbol} | "
                                    f"Reason: Global trading disabled | "
                                    f"Safety Level: {directives.safety_level.value} | "
                                    f"Primary Reason: {directives.primary_reason}"
                                )
                                orders_skipped += 1
                                continue
                            
                            # Check if expansion symbols are allowed
                            symbol_category = signal_dict.get("category", "EXPANSION")
                            if symbol_category == "EXPANSION" and not directives.allow_expansion_symbols:
                                logger.error(
                                    f"🛡️ [SAFETY GOVERNOR] ❌ BLOCKED: {symbol} (EXPANSION) | "
                                    f"Reason: Expansion symbols blocked in {directives.safety_level.value} mode"
                                )
                                orders_skipped += 1
                                continue
                            
                            # Evaluate trade with Safety Governor
                            # Get current price for leverage calculation
                            try:
                                price_data = await self._adapter._signed_request('GET', '/fapi/v1/ticker/price', params={'symbol': symbol})
                                current_price = float(price_data['price'])
                            except:
                                current_price = 0.0
                            
                            # Calculate proposed leverage from environment (testnet-compatible)
                            proposed_leverage = float(os.getenv("QT_DEFAULT_LEVERAGE", "30.0"))
                            
                            decision, record = safety_governor.evaluate_trade_request(
                                symbol=symbol,
                                action="NEW_TRADE",
                                size=actual_margin,  # Using margin as size
                                leverage=proposed_leverage,
                                confidence=confidence,
                                metadata={
                                    "category": symbol_category,
                                    "model": model,
                                    "risk_modifier": risk_modifier
                                }
                            )
                            
                            # Log the decision with full transparency
                            safety_governor.log_decision(record)
                            
                            # Handle decision
                            if not record.allowed:
                                logger.error(
                                    f"🛡️ [SAFETY GOVERNOR] ❌ TRADE REJECTED: {symbol} | "
                                    f"Reason: {record.block_reason.value if record.block_reason else 'UNKNOWN'}"
                                )
                                orders_skipped += 1
                                continue
                            
                            elif record.modified:
                                # Apply Safety Governor multipliers
                                original_margin = actual_margin
                                original_leverage = proposed_leverage
                                
                                actual_margin = actual_margin * record.applied_multipliers.get("size", 1.0)
                                proposed_leverage = proposed_leverage * record.applied_multipliers.get("leverage", 1.0)
                                
                                logger.warning(
                                    f"🛡️ [SAFETY GOVERNOR] ⚠️ TRADE MODIFIED: {symbol} | "
                                    f"Margin: ${original_margin:.2f} → ${actual_margin:.2f} | "
                                    f"Leverage: {original_leverage:.1f}x → {proposed_leverage:.1f}x | "
                                    f"Safety Level: {directives.safety_level.value}"
                                )
                            else:
                                logger.info(
                                    f"🛡️ [SAFETY GOVERNOR] ✅ TRADE APPROVED: {symbol} | "
                                    f"Margin: ${actual_margin:.2f}, Leverage: {proposed_leverage:.1f}x"
                                )
                        else:
                            logger.debug("[SAFETY GOVERNOR] No directives available yet - allowing trade")
                    else:
                        logger.debug("[SAFETY GOVERNOR] Not initialized - allowing trade")
                
                except Exception as gov_error:
                    logger.error(f"[SAFETY GOVERNOR] Error during trade evaluation: {gov_error}", exc_info=True)
                    # On error, allow trade to proceed (fail-open for now)
                    logger.warning("[SAFETY GOVERNOR] Allowing trade due to evaluation error")
                
                # 🔥 [RL-UNIFIED] USE REINFORCEMENT LEARNING FOR ALL TRADE PARAMETERS
                # RL Agent decides: Position size, Leverage, TP, SL, Partial TP
                # NO MORE: Exit Policy Engine, DynamicTPSLCalculator, AI-OVERRIDE
                rl_decision = None  # Initialize RL decision
                rl_tp_pct = None
                rl_sl_pct = None
                rl_partial_tp_pct = None
                rl_partial_tp_size = None
                rl_partial_enabled = False
                
                try:
                    # [TARGET] RISK MANAGEMENT: Fetch market conditions
                    market_data = await fetch_market_conditions(symbol, self._adapter, lookback=200)
                    if not market_data:
                        logger.warning(f"[WARNING] Could not fetch market conditions for {symbol}, skipping")
                        orders_skipped += 1
                        continue
                    
                    price = market_data['price']
                    
                    # 💡 [SMART-SIZER] USE SMART POSITION SIZER FOR INTELLIGENT SIZING
                    # Smart rule-based sizing (no training needed!)
                    smart_size_usd = None
                    smart_tp_pct = None
                    smart_sl_pct = None
                    smart_leverage = None
                    try:
                        from backend.services.execution.smart_position_sizer import get_smart_position_sizer
                        
                        smart_sizer = get_smart_position_sizer()
                        
                        # Calculate volatility as ATR percentage
                        volatility = market_data['atr'] / price if price > 0 else 0.03
                        
                        # Calculate trend strength (simplified: distance from EMA200)
                        ema_200 = market_data.get('ema_200', price)
                        trend_strength = abs(price - ema_200) / ema_200 if ema_200 > 0 else 0.5
                        trend_strength = min(1.0, trend_strength * 10)  # Scale to 0-1
                        
                        # Get smart sizing decision
                        smart_result = smart_sizer.calculate_optimal_size(
                            symbol=symbol,
                            side=action,
                            volatility=volatility,
                            trend_strength=trend_strength,
                            regime="UNKNOWN",  # Will detect regime later
                            market_data=market_data
                        )
                        
                        # Check if trade was blocked (win rate too low)
                        if smart_result.size_usd == 0:
                            logger.warning(
                                f"🚫 [SMART-SIZER] {symbol} {action} BLOCKED: {smart_result.reasoning}"
                            )
                            orders_skipped += 1
                            continue
                        
                        # Extract smart sizing values
                        smart_size_usd = smart_result.size_usd
                        smart_tp_pct = smart_result.tp_pct
                        smart_sl_pct = smart_result.sl_pct
                        smart_leverage = smart_result.leverage
                        
                        logger.info(
                            f"💡 [SMART-SIZER] {symbol}: Smart sizing applied - "
                            f"Size=${smart_size_usd:.0f} ({smart_result.size_pct*100:.0f}%), "
                            f"Lev={smart_leverage:.1f}x, "
                            f"TP={smart_tp_pct*100:.1f}%, SL={smart_sl_pct*100:.2f}%, "
                            f"Confidence={smart_result.confidence*100:.0f}%"
                        )
                        
                        # Track open position for correlation checks
                        smart_sizer.add_open_position(symbol, action)
                        
                    except Exception as smart_error:
                        logger.error(f"[SMART-SIZER] Smart sizing failed for {symbol}: {smart_error}")
                    
                    # 🔥 [RL-UNIFIED] CALL RL AGENT FOR ALL TRADE PARAMETERS (FALLBACK)
                    # Get position size, leverage, TP, SL, partial TP from RL agent
                    try:
                        from backend.services.ai.rl_position_sizing_agent import get_rl_sizing_agent
                        
                        rl_agent = get_rl_sizing_agent(enabled=True)
                        if rl_agent:
                            # Calculate current exposure from number of open positions
                            num_positions = open_positions  # open_positions is already an int
                            max_positions = 12  # From Portfolio Balancer
                            current_exposure_pct = num_positions / max_positions if max_positions > 0 else 0.0
                            
                            # Calculate ATR as percentage
                            atr_pct = market_data['atr'] / price if price > 0 else 0.02
                            
                            # 🔍 DEBUG: Log what we're sending to RL agent
                            logger.info(f"[DEBUG] Sending to RL agent: symbol={symbol}, equity_usd=${cash:.2f}, exposure={current_exposure_pct:.2%}")
                            
                            # Get RL decision
                            rl_decision = rl_agent.decide_sizing(
                                symbol=symbol,
                                confidence=confidence,
                                atr_pct=atr_pct,
                                current_exposure_pct=current_exposure_pct,
                                equity_usd=cash,
                                adx=None,  # Will add later
                                trend_strength=None
                            )
                            
                            # Extract RL values
                            rl_tp_pct = rl_decision.tp_percent
                            rl_sl_pct = rl_decision.sl_percent
                            rl_partial_tp_pct = rl_decision.partial_tp_percent
                            rl_partial_tp_size = rl_decision.partial_tp_size
                            rl_partial_enabled = rl_decision.partial_tp_enabled
                            
                            # 🔥 PRIORITY: ALWAYS USE MATH AI/RL (most sophisticated system)
                            # Smart Sizer is only fallback if RL fails
                            if rl_decision.position_size_usd > 0:
                                logger.info(
                                    f"🤖 [RL-PRIORITY] {symbol}: Using Math AI/RL (BEST) - "
                                    f"Size=${rl_decision.position_size_usd:.0f}, "
                                    f"Leverage={rl_decision.leverage:.1f}x, "
                                    f"TP={rl_tp_pct*100:.1f}%, SL={rl_sl_pct*100:.1f}%"
                                )
                                # Smart Sizer values discarded - Math AI is superior
                                if smart_size_usd is not None:
                                    logger.debug(
                                        f"[SMART-DISCARDED] {symbol}: Smart Sizer=${ smart_size_usd:.0f} ignored, using Math AI=${rl_decision.position_size_usd:.0f}"
                                    )
                            elif smart_size_usd is not None:
                                # Fallback to Smart Sizer only if RL failed
                                rl_decision.position_size_usd = smart_size_usd
                                rl_decision.leverage = smart_leverage
                                rl_tp_pct = smart_tp_pct
                                rl_sl_pct = smart_sl_pct
                                logger.warning(
                                    f"⚠️ [SMART-FALLBACK] {symbol}: RL failed, using Smart Sizer - "
                                    f"Size=${smart_size_usd:.0f}, TP={smart_tp_pct*100:.1f}%"
                                )
                            else:
                                logger.info(
                                    f"🤖 [RL-UNIFIED] {symbol}: Using RL parameters - "
                                    f"Size=${rl_decision.position_size_usd:.0f}, Lev={rl_decision.leverage:.1f}x, "
                                    f"TP={rl_tp_pct*100:.1f}%, SL={rl_sl_pct*100:.1f}%"
                                )
                        else:
                            logger.warning(f"[RL-UNIFIED] RL agent not available for {symbol}")
                    except Exception as rl_error:
                        logger.error(f"[RL-UNIFIED] RL decision failed for {symbol}: {rl_error}")
                    
                    # 🛡️ [RL VOLATILITY SAFETY ENVELOPE] - SPRINT 1 D4
                    # Apply volatility-based caps to RL decisions BEFORE Safety Governor
                    if rl_decision and rl_decision.position_size_usd > 0 and self.rl_envelope:
                        try:
                            # Calculate ATR percentage from market data
                            atr_pct = market_data['atr'] / price if price > 0 else 0.02
                            
                            # Calculate proposed risk % from RL decision
                            proposed_risk_pct = rl_decision.position_size_usd / cash if cash > 0 else 0.05
                            
                            # Apply envelope limits
                            envelope_result = self.rl_envelope.apply_limits(
                                symbol=symbol,
                                atr_pct=atr_pct,
                                proposed_leverage=rl_decision.leverage,
                                proposed_risk_pct=proposed_risk_pct,
                                equity_usd=cash
                            )
                            
                            # Update RL decision with capped values if needed
                            if envelope_result.was_capped:
                                original_leverage = rl_decision.leverage
                                original_size = rl_decision.position_size_usd
                                
                                # Apply capped values
                                rl_decision.leverage = envelope_result.capped_leverage
                                rl_decision.position_size_usd = envelope_result.capped_risk_pct * cash
                                
                                logger_envelope.warning(
                                    f"🛡️ [RL-ENVELOPE] {symbol} | {envelope_result.volatility_bucket.value} volatility | "
                                    f"Leverage: {original_leverage:.1f}x → {rl_decision.leverage:.1f}x | "
                                    f"Size: ${original_size:.0f} → ${rl_decision.position_size_usd:.0f}"
                                )
                            else:
                                logger_envelope.debug(
                                    f"✅ [RL-ENVELOPE] {symbol} | {envelope_result.volatility_bucket.value} volatility | "
                                    f"Leverage: {rl_decision.leverage:.1f}x | Size: ${rl_decision.position_size_usd:.0f}"
                                )
                        except Exception as envelope_error:
                            logger_envelope.error(f"[RL-ENVELOPE] Envelope check failed for {symbol}: {envelope_error}")
                            # Continue with original RL decision on envelope error (fail-open)
                    
                    # 🔥 [SAFETY GOVERNOR] RE-CHECK WITH ACTUAL RL SIZING
                    # Now that we have RL decision, validate with Safety Governor again using REAL values
                    if rl_decision and rl_decision.position_size_usd > 0:
                        actual_margin = rl_decision.position_size_usd
                        proposed_leverage = rl_decision.leverage
                        
                        try:
                            if hasattr(self, '_app_state') and hasattr(self._app_state, 'safety_governor'):
                                safety_governor = self._app_state.safety_governor
                                
                                decision_sg, record_sg = safety_governor.evaluate_trade_request(
                                    symbol=symbol,
                                    action="NEW_TRADE",
                                    size=actual_margin,  # RL-calculated margin
                                    leverage=proposed_leverage,  # RL-calculated leverage
                                    confidence=confidence,
                                    metadata={
                                        "category": signal_dict.get("category", "EXPANSION"),
                                        "model": model,
                                        "risk_modifier": risk_modifier,
                                        "rl_optimized": True
                                    }
                                )
                                
                                safety_governor.log_decision(record_sg)
                                
                                if not record_sg.allowed:
                                    logger.error(
                                        f"🛡️ [SAFETY GOVERNOR] ❌ RL TRADE REJECTED: {symbol} | "
                                        f"Size=${actual_margin:.2f}, Leverage={proposed_leverage:.1f}x | "
                                        f"Reason: {record_sg.block_reason.value if record_sg.block_reason else 'UNKNOWN'}"
                                    )
                                    orders_skipped += 1
                                    continue
                                
                                elif record_sg.modified:
                                    original_margin = actual_margin
                                    original_leverage = proposed_leverage
                                    
                                    actual_margin = actual_margin * record_sg.applied_multipliers.get("size", 1.0)
                                    proposed_leverage = proposed_leverage * record_sg.applied_multipliers.get("leverage", 1.0)
                                    
                                    # Update RL decision with Safety Governor adjustments
                                    rl_decision.position_size_usd = actual_margin
                                    rl_decision.leverage = proposed_leverage
                                    
                                    logger.warning(
                                        f"🛡️ [SAFETY GOVERNOR] ⚠️ RL TRADE MODIFIED: {symbol} | "
                                        f"Margin: ${original_margin:.2f} → ${actual_margin:.2f} | "
                                        f"Leverage: {original_leverage:.1f}x → {proposed_leverage:.1f}x"
                                    )
                                else:
                                    logger.info(
                                        f"🛡️ [SAFETY GOVERNOR] ✅ RL TRADE APPROVED: {symbol} | "
                                        f"Margin: ${actual_margin:.2f}, Leverage: {proposed_leverage:.1f}x"
                                    )
                        except Exception as gov_error:
                            logger.error(f"[SAFETY GOVERNOR] RL trade evaluation error: {gov_error}", exc_info=True)
                    
                    # [TARGET] REGIME DETECTION: Detect current market regime
                    # Skip regime detection for now - market_data doesn't contain ADX
                    regime = None
                    # try:
                    #     # Convert market_data dict to MarketConditions object
                    #     market_conditions_obj = MarketConditions(
                    #         price=market_data['price'],
                    #         atr=market_data['atr'],
                    #         ema_200=market_data['ema_200'],
                    #         volume_24h=market_data['volume_24h'],
                    #         spread_bps=market_data['spread_bps'],
                    #         timestamp=datetime.now(timezone.utc)
                    #     )
                    #     regime = self.regime_detector.detect_regime(market_conditions_obj)
                    #     logger.info(
                    #         f"[CHART] {symbol} Market Regime: {regime.regime} "
                    #         f"(VOL={regime.volatility_regime}, TREND={regime.trend_regime}, "
                    #         f"ATR={regime.atr_current:.2f}, ADX={regime.adx:.2f})"
                    #     )
                    # except Exception as e:
                    #     logger.warning(f"[WARNING] Could not detect regime for {symbol}: {e}")
                    #     regime = None
                    
                    # [TARGET] EXIT POLICY: Get regime-specific exit parameters
                    if regime:
                        try:
                            exit_params = get_exit_params(regime.regime)
                            logger.info(
                                f"[TARGET] Using {regime.regime} exit params: "
                                f"k1_SL={exit_params.k1_SL}, k2_TP={exit_params.k2_TP}, "
                                f"R:R={exit_params.k2_TP/exit_params.k1_SL:.2f}, "
                                f"BE@{exit_params.breakeven_R}R, max={exit_params.max_duration_hours}h"
                            )
                        except Exception as e:
                            logger.warning(f"[WARNING] Could not get exit params for {regime.regime}: {e}")
                            exit_params = None
                    else:
                        exit_params = None
                    
                    # Build SignalQuality for risk evaluation
                    # Extract model votes properly - model might be a dict with ensemble data
                    if isinstance(model, dict):
                        # If model is ensemble metadata, extract individual votes
                        model_votes_dict = {}
                        if 'models' in model:
                            for model_name, vote_data in model.get('models', {}).items():
                                if isinstance(vote_data, dict) and 'action' in vote_data:
                                    model_votes_dict[model_name] = normalize_action(vote_data['action'])
                        else:
                            # Use consensus as single vote
                            model_votes_dict['ensemble'] = normalize_action(action)
                    else:
                        # Simple case: single model name
                        model_votes_dict = {str(model): normalize_action(action)}
                    
                    signal_quality = SignalQuality(
                        consensus_type=ConsensusType.STRONG,  # Assume strong since passed confidence threshold
                        confidence=confidence,
                        model_votes=model_votes_dict,
                        signal_strength=confidence
                    )
                    
                    # Build MarketConditions
                    market_conditions = MarketConditions(
                        price=price,
                        atr=market_data['atr'],
                        ema_200=market_data['ema_200'],
                        volume_24h=market_data['volume_24h'],
                        spread_bps=market_data['spread_bps'],
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    # [TARGET] EVALUATE THROUGH RISK MANAGEMENT LAYER
                    decision = self._trade_manager.evaluate_new_signal(
                        symbol=symbol,
                        action=normalize_action(action),
                        signal_quality=signal_quality,
                        market_conditions=market_conditions,
                        current_equity=cash
                    )
                    
                    if not decision.approved:
                        logger.warning(
                            f"❌ {symbol} {action} REJECTED by risk management: {decision.rejection_reason}"
                        )
                        orders_skipped += 1
                        continue
                    
                    baseline_sl_pct = abs(price - decision.stop_loss) / price if price else 0.0
                    baseline_tp_pct = abs(decision.take_profit - price) / price if price else 0.0
                    
                    # 🔥 [RL-UNIFIED] ALWAYS override with RL-calculated TP/SL
                    # RL Agent has already optimized these based on learned patterns
                    if rl_tp_pct and rl_sl_pct:
                        # DEBUG: Log action value
                        logger.info(f"🔍 [DEBUG] {symbol}: action='{action}', type={type(action)}")
                        
                        # Normalize action to handle case sensitivity
                        normalized_action = normalize_action(action)
                        logger.info(f"🔍 [DEBUG] {symbol}: normalized_action='{normalized_action}'")
                        
                        # Calculate RL-driven TP/SL prices
                        if normalized_action == "LONG":
                            rl_take_profit = price * (1 + rl_tp_pct)
                            rl_stop_loss = price * (1 - rl_sl_pct)
                        else:  # SELL or SHORT
                            rl_take_profit = price * (1 - rl_tp_pct)
                            rl_stop_loss = price * (1 + rl_sl_pct)
                        
                        # Override decision with RL values
                        logger.info(
                            f"🤖 [RL-TPSL] {symbol}: Ignoring Exit Policy ({baseline_tp_pct*100:.2f}% TP) → "
                            f"Using RL: TP={rl_tp_pct*100:.1f}% (${rl_take_profit:.4f}), "
                            f"SL={rl_sl_pct*100:.1f}% (${rl_stop_loss:.4f})"
                        )
                        if rl_partial_enabled:
                            logger.info(
                                f"   ↳ Partial TP: {rl_partial_tp_size*100:.0f}% @ {rl_partial_tp_pct*100:.1f}%"
                            )
                        
                        decision.take_profit = rl_take_profit
                        decision.stop_loss = rl_stop_loss
                        
                        # Update baseline percentages
                        baseline_sl_pct = rl_sl_pct
                        baseline_tp_pct = rl_tp_pct
                    else:
                        # No RL TP/SL - this shouldn't happen, log critical warning
                        logger.error(
                            f"❌ [RL-TPSL] {symbol}: NO RL TP/SL CALCULATED! "
                            f"Using Exit Policy fallback (TP={baseline_tp_pct*100:.2f}%, SL={baseline_sl_pct*100:.2f}%)"
                        )
                    
                    # 🔥 [RL-UNIFIED] USE RL POSITION SIZE INSTEAD OF DECISION.QUANTITY
                    # Convert RL position_size_usd (margin) to quantity using leverage
                    if rl_decision and rl_decision.position_size_usd > 0:
                        # RL gives us MARGIN in USD
                        # Notional = Margin × Leverage
                        # Quantity = Notional / Price
                        notional = rl_decision.position_size_usd * rl_decision.leverage
                        quantity = notional / price
                        
                        logger.info(
                            f"🤖 [RL-SIZE] {symbol}: Using RL position size - "
                            f"Margin=${rl_decision.position_size_usd:.2f}, "
                            f"Leverage={rl_decision.leverage:.1f}x, "
                            f"Notional=${notional:.2f}, "
                            f"Quantity={quantity:.4f}"
                        )
                    else:
                        # Fallback to TradeManager quantity (shouldn't happen)
                        quantity = decision.quantity * risk_modifier
                        logger.warning(
                            f"⚠️ [FALLBACK-SIZE] {symbol}: No RL size, using TradeManager quantity={quantity:.4f}"
                        )
                    
                    if risk_modifier < 1.0:
                        logger.info(
                            f"[WARNING] {symbol} risk modifier={risk_modifier:.1%} (poor performance)"
                        )
                    
                    # [TARGET] COST MODEL: Estimate realistic costs
                    try:
                        cost_estimate = estimate_trade_cost(
                            entry_price=price,
                            exit_price=decision.take_profit,  # Optimistic case
                            size=quantity,
                            atr=market_data['atr'],
                            is_maker=False  # Conservative estimate (taker fees)
                        )
                        logger.info(
                            f"[MONEY] {symbol} Cost estimate: "
                            f"Total=${cost_estimate.total_cost:.2f} "
                            f"({cost_estimate.total_cost_pct:.2%}), "
                            f"Cost in R: {cost_estimate.cost_in_R:.3f}R"
                        )
                    except Exception as e:
                        logger.warning(f"[WARNING] Could not estimate costs: {e}")
                        cost_estimate = None
                    
                    logger.info(
                        f"[OK] {symbol} {action} APPROVED by risk management: "
                        f"Quantity={quantity:.4f} @ ${price:.4f}, "
                        f"SL=${decision.stop_loss:.4f}, TP=${decision.take_profit:.4f}"
                    )
                    
                    # Note: Liquidity check now done by TradeOpportunityFilter
                    # Check if we already have this position
                    if symbol.upper() in positions:
                        logger.info(f"[SKIP] Already have {symbol} position, skipping")
                        orders_skipped += 1
                        continue
                    
                    # [TARGET] Round quantity to exchange step size
                    try:
                        exchange_info = await self._adapter.get_exchange_info()
                        symbol_info = next((s for s in exchange_info.get('symbols', []) if s['symbol'] == symbol), None)
                        
                        if symbol_info:
                            # Get quantity precision (stepSize)
                            step_size = 0.001  # Default
                            for filter_item in symbol_info.get('filters', []):
                                if filter_item['filterType'] == 'LOT_SIZE':
                                    step_size = float(filter_item.get('stepSize', 0.001))
                                    break
                            
                            # Round down to step size
                            from decimal import Decimal, ROUND_DOWN
                            quantity = float(Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN))
                            
                            logger.debug(f"[CHART] {symbol}: step_size={step_size}, rounded_qty={quantity}")
                        
                    except Exception as e:
                        logger.warning(f"[WARNING] Could not round quantity for {symbol}: {e}")
                    
                    # [NEW] AI SYSTEM PRE-EXECUTION HOOKS
                    if AI_INTEGRATION_AVAILABLE and self.ai_services:
                        try:
                            # Risk check (AI-HFOS + Risk OS)
                            allowed, reason = await pre_trade_risk_check(
                                symbol, signal_dict, positions
                            )
                            if not allowed:
                                logger.warning(f"[AI-HFOS] Trade blocked: {reason}")
                                orders_skipped += 1
                                continue
                            
                            # Portfolio check (PBA)
                            allowed, reason = await pre_trade_portfolio_check(
                                symbol, signal_dict, positions
                            )
                            if not allowed:
                                logger.warning(f"[PBA] Trade blocked: {reason}")
                                orders_skipped += 1
                                continue
                            
                            # Position sizing (AI-HFOS)
                            adjusted_quantity = await pre_trade_position_sizing(
                                symbol, signal_dict, quantity
                            )
                            if adjusted_quantity != quantity:
                                logger.info(
                                    f"[AI-HFOS] Position size adjusted: "
                                    f"{quantity:.4f} → {adjusted_quantity:.4f}"
                                )
                                quantity = adjusted_quantity
                            
                        except Exception as e:
                            logger.error(f"[ERROR] AI pre-execution hooks failed: {e}", exc_info=True)
                            # Fail-safe: continue with original values
                    
                    # Determine side
                    side = "buy" if normalize_action(action) == "LONG" else "sell"
                    
                    # 🛑 [MODEL_SUPERVISOR] Check for directional bias before placing trade
                    if self.model_supervisor:
                        try:
                            min_samples = int(os.getenv("QT_MODEL_SUPERVISOR_MIN_SAMPLES", "20"))
                            bias_threshold = float(os.getenv("QT_MODEL_SUPERVISOR_BIAS_THRESHOLD", "0.70"))
                            
                            should_block, reason = self.model_supervisor.check_bias_and_block(
                                action=action,
                                min_samples=min_samples,
                                bias_threshold=bias_threshold
                            )
                            
                            if should_block:
                                logger.warning(
                                    f"🛑 [MODEL_SUPERVISOR] TRADE BLOCKED: {symbol} {action} - {reason}"
                                )
                                # Skip this trade to prevent excessive directional bias
                                continue
                        except Exception as bias_error:
                            logger.error(
                                f"[MODEL_SUPERVISOR] Bias check failed for {symbol}: {bias_error}",
                                exc_info=True
                            )
                            # Fail-safe: allow trade if bias check fails
                    
                    # [NEW] Order type selection (AELM)
                    order_type = "market"
                    if AI_INTEGRATION_AVAILABLE and self.ai_services:
                        try:
                            order_type = await execution_order_type_selection(
                                symbol, signal_dict, "market"
                            )
                        except Exception as e:
                            logger.error(f"[ERROR] Order type selection failed: {e}", exc_info=True)
                            # Fail-safe: use market order
                    
                    position_size_usd = quantity * price
                    
                    # [NEW] META-STRATEGY SELECTOR: Dynamically select optimal strategy
                    meta_strategy_result = None  # Store for later use
                    if self.meta_strategy and self.meta_strategy.enabled:
                        try:
                            # Build comprehensive market data for strategy selection
                            meta_market_data = {
                                "price": price,
                                "atr": market_data.get('atr', 0),
                                "atr_pct": market_data.get('atr', 0) / price if price > 0 else 0,
                                "volume_24h": market_data.get('volume_24h', 0),
                                "depth_5bps": market_data.get('depth', 0),
                                "spread_bps": market_data.get('spread_bps', 0),
                                "adx": market_data.get('adx', 0),
                                "trend_strength": market_data.get('trend_strength', 0),
                            }
                            
                            meta_strategy_result = await self.meta_strategy.select_strategy_for_signal(
                                symbol=symbol,
                                signal=signal_dict,
                                market_data=meta_market_data
                            )
                            
                            # Apply selected strategy's TP/SL configuration
                            tpsl_config = meta_strategy_result.tpsl_config
                            
                            # Override TP/SL percentages with Meta-Strategy selection
                            # Convert R-multiples to percentages using ATR
                            atr = market_data.get('atr', price * 0.01)  # Fallback: 1% of price
                            atr_pct = atr / price
                            
                            # Update TP/SL based on selected strategy
                            tp_percent = tpsl_config['atr_mult_tp1'] * atr_pct
                            sl_percent = tpsl_config['atr_mult_sl'] * atr_pct
                            trail_percent = tpsl_config.get('trail_dist_mult', 1.5) * atr_pct
                            
                            logger.info(
                                f"[META-STRATEGY] {symbol}: {meta_strategy_result.strategy.name} "
                                f"(regime={meta_strategy_result.regime.value}, "
                                f"explore={meta_strategy_result.decision.is_exploration}, "
                                f"conf={meta_strategy_result.decision.confidence:.0%}) | "
                                f"TP={tp_percent*100:.1f}% SL={sl_percent*100:.1f}%"
                            )
                            logger.info(
                                f"[META-STRATEGY] Reasoning: {meta_strategy_result.decision.reasoning}"
                            )
                        except Exception as meta_error:
                            logger.error(
                                f"[ERROR] Meta-Strategy selection failed for {symbol}: {meta_error}",
                                exc_info=True
                            )
                            # Continue with default TP/SL if meta-strategy fails
                    
                    # [TARGET] LOGGING EXTENSIONS: Enrich trade entry data
                    try:
                        enriched_entry = enrich_trade_entry(
                            symbol=symbol,
                            action=action,
                            entry_price=price,
                            quantity=quantity,
                            stop_loss=decision.stop_loss,
                            take_profit=decision.take_profit,
                            atr=market_data['atr'],
                            confidence=confidence,
                            consensus="STRONG" if confidence >= 0.7 else "MODERATE",
                            regime=regime.regime if regime else None
                        )
                        log_message = format_trade_log_message(enriched_entry, "ENTRY")
                        logger.info(log_message)
                    except Exception as e:
                        logger.warning(f"[WARNING] Could not enrich trade entry: {e}")
                        logger.info(
                            f"📤 Placing {order_type.upper()} order: {symbol} qty={quantity:.4f} @ ${price:.4f} "
                            f"(position=${position_size_usd:.2f}, conf={confidence:.2%})"
                        )
                    
                    # [TARGET] STORE DYNAMIC TP/SL + Risk-adjusted levels
                    self._symbol_tpsl[symbol] = {
                        "tp_percent": tp_percent,
                        "sl_percent": sl_percent,
                        "trail_percent": trail_percent,
                        "partial_tp": partial_tp,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "confidence": confidence,
                        # Add risk-adjusted TP/SL prices from ExitPolicyEngine
                        "entry_price": price,
                        "stop_loss_price": decision.stop_loss,
                        "take_profit_price": decision.take_profit,
                    }
                    
                    # Save to file for position_monitor to read
                    try:
                        with open(TPSL_STORAGE_PATH, 'w') as f:
                            json.dump(self._symbol_tpsl, f)
                    except Exception as e:
                        logger.warning(f"Could not save TP/SL to file: {e}")
                    
                    logger.info(
                        f"[TARGET] Stored Dynamic TP/SL for {symbol}: "
                        f"Entry=${price:.4f}, "
                        f"SL=${decision.stop_loss:.4f} (-{((price - decision.stop_loss) / price * 100):.1f}%), "
                        f"TP=${decision.take_profit:.4f} (+{((decision.take_profit - price) / price * 100):.1f}%)"
                    )
                    
                    hybrid_orders_placed = False
                    # 🔥 [RL-UNIFIED] USE RL-DECIDED LEVERAGE
                    leverage = None
                    if rl_decision and rl_decision.leverage:
                        leverage = rl_decision.leverage
                        logger.info(f"🤖 [RL-LEVERAGE] Using RL-decided leverage {leverage:.1f}x for {symbol}")
                    elif decision and decision.position_size and hasattr(decision.position_size, 'leverage_used'):
                        leverage = decision.position_size.leverage_used
                        logger.warning(f"⚠️ [FALLBACK-LEVERAGE] Using TradeManager leverage {leverage:.1f}x for {symbol}")
                    else:
                        logger.warning(f"❌ [NO-LEVERAGE] No leverage set for {symbol}, will use exchange default")
                    
                    # [NEW] EMERGENCY STOP CHECK - SPRINT 1 D3
                    if self.ess:
                        try:
                            can_execute = await self.ess.can_execute_orders()
                            if not can_execute:
                                ess_status = self.ess.get_status()
                                logger_ess.error(
                                    f"🛑 [ESS BLOCK] Order blocked by Emergency Stop System: {symbol} {side.upper()}\n"
                                    f"   ESS State: {ess_status['state']}\n"
                                    f"   Reason: {ess_status.get('trip_reason', 'N/A')}\n"
                                    f"   Can Execute: {ess_status['can_execute']}"
                                )
                                
                                # Publish order.blocked_by_ess event
                                if self.event_bus:
                                    await self.event_bus.publish("order.blocked_by_ess", {
                                        "symbol": symbol,
                                        "side": side,
                                        "quantity": quantity,
                                        "price": price,
                                        "ess_state": ess_status['state'],
                                        "trip_reason": ess_status.get('trip_reason'),
                                        "timestamp": datetime.now(timezone.utc).isoformat()
                                    })
                                
                                # Skip order submission
                                continue
                        except Exception as e:
                            logger_ess.error(f"[ESS ERROR] ESS check failed: {e}", exc_info=True)
                            # Continue with order on ESS error (fail-open for safety)
                    
                    # 🛑 [POSITION INVARIANT] Critical fix: Check for conflicting positions
                    # Prevents opening BOTH long AND short on same symbol (CRITICAL BUG FIX)
                    try:
                        enforcer = get_position_invariant_enforcer()
                        current_positions = await self._get_current_positions()
                        
                        enforcer.enforce_before_order(
                            symbol=symbol,
                            side=side,
                            quantity=quantity,
                            current_positions=current_positions,
                            account="main",  # TODO: Make configurable
                            exchange="binance"  # TODO: Make configurable
                        )
                    except PositionInvariantViolation as e:
                        logger.error(
                            f"🛑 [POSITION INVARIANT] Order BLOCKED - Would violate position rules: {e}\n"
                            f"   Symbol: {symbol}\n"
                            f"   Attempted: {side.upper()} {quantity}\n"
                            f"   Reason: {str(e)}"
                        )
                        
                        # Publish blocked order event for metrics
                        if self.event_bus:
                            await self.event_bus.publish("order.blocked_by_invariant", {
                                "symbol": symbol,
                                "side": side,
                                "quantity": quantity,
                                "price": price,
                                "reason": str(e),
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            })
                        
                        # Skip this order - critical invariant violation
                        orders_skipped += 1
                        continue
                    except Exception as e:
                        logger.error(f"[ERROR] Position invariant check failed: {e}", exc_info=True)
                        # Fail-safe: block order on check failure for safety
                        logger.error(f"🛑 [FAIL-SAFE] Blocking order due to invariant check error")
                        orders_skipped += 1
                        continue
                    
                    # 🎯 FIX: Get order result with ACTUAL fill price
                    order_result = await self._adapter.submit_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,  # Signal price (for reference)
                        leverage=leverage  # RL-decided leverage or None for default
                    )
                    
                    # Extract actual fill data
                    order_id = order_result['order_id']
                    actual_entry_price = order_result['avg_price']  # ✅ ACTUAL FILL PRICE
                    filled_qty = order_result['filled_qty']
                    
                    # 🎯 FALLBACK: If avgPrice is 0 (testnet quirk), fetch from position
                    if actual_entry_price == 0 or actual_entry_price == price:
                        try:
                            await asyncio.sleep(0.5)  # Wait for position to update
                            positions = await self._adapter.fetch_positions()
                            if symbol in positions:
                                pos = positions[symbol]
                                if 'entryPrice' in pos and pos['entryPrice'] > 0:
                                    actual_entry_price = float(pos['entryPrice'])
                                    logger.info(
                                        f"✅ [PRICE-FETCH] {symbol}: Got actual entry from position: ${actual_entry_price:.5f}"
                                    )
                        except Exception as e:
                            logger.warning(f"⚠️ Could not fetch entry price from position: {e}")
                    
                    # Log slippage if significant
                    slippage_pct = abs(actual_entry_price - price) / price * 100
                    if slippage_pct > 0.1:  # More than 0.1% slippage
                        logger.warning(
                            f"⚠️ [SLIPPAGE] {symbol}: Signal ${price:.5f} → Fill ${actual_entry_price:.5f} "
                            f"({slippage_pct:+.2f}%)"
                        )
                    
                    logger.info(
                        f"[OK] Order placed: {symbol} {side.upper()} - ID: {order_id} "
                        f"@ ${actual_entry_price:.5f} (qty: {filled_qty})"
                    )
                    
                    # 🎯 [EXIT BRAIN V3] Create exit plan IMMEDIATELY after position opens
                    exit_plan_created = False
                    if self.exit_brain_enabled and self.exit_router:
                        try:
                            # Build position dict from filled order
                            # ✅ CRITICAL: Pass AI-determined leverage so Exit Brain uses correct leverage for SL calculation
                            position_for_brain = {
                                'symbol': symbol,
                                'side': 'LONG' if side == 'buy' else 'SHORT',
                                'quantity': filled_qty,
                                'entry_price': actual_entry_price,
                                'leverage': leverage or 1,  # Binance API leverage (often wrong for cross margin)
                                'actual_leverage': leverage or 1,  # ✅ AI-determined leverage for Exit Brain SL calculation
                                'unrealized_pnl': 0.0,
                                'mark_price': actual_entry_price
                            }
                            
                            # Build context
                            from backend.domains.exits.exit_brain_v3.integration import build_context_from_position
                            
                            # Get RL hints
                            rl_hints = None
                            if rl_decision:
                                rl_hints = {
                                    'tp_pct': rl_decision.tp_pct,
                                    'sl_pct': rl_decision.sl_pct,
                                    'confidence': confidence
                                }
                            
                            # Build context
                            context = build_context_from_position(
                                position=position_for_brain,
                                rl_hints=rl_hints,
                                risk_context={'max_leverage': leverage or 1},
                                market_data={'atr': market_data.get('atr', actual_entry_price * 0.01) if market_data else actual_entry_price * 0.01}
                            )
                            
                            # Create plan
                            plan = await self.exit_router.get_or_create_plan(symbol, context)
                            
                            if plan and len(plan.legs) > 0:
                                logger_exit_brain.info(
                                    f"[EXIT BRAIN V3] {symbol}: Created exit plan with {len(plan.legs)} legs"
                                )
                                exit_plan_created = True
                                
                                # Log leg details
                                for leg in plan.legs:
                                    logger_exit_brain.debug(
                                        f"  ↳ {leg.kind.name}: size={leg.size_pct*100:.0f}%, "
                                        f"trigger={leg.trigger_pct*100:+.2f}%, priority={leg.priority}"
                                    )
                            else:
                                logger_exit_brain.warning(
                                    f"[EXIT BRAIN V3] {symbol}: Plan has no legs - position monitor will protect"
                                )
                        
                        except Exception as brain_exc:
                            logger_exit_brain.error(
                                f"[EXIT BRAIN V3] {symbol}: Plan creation failed: {brain_exc}",
                                exc_info=True
                            )
                            logger_exit_brain.warning(f"[EXIT BRAIN V3] {symbol}: Position monitor will provide fallback protection")
                    else:
                        logger_exit_brain.debug(f"[EXIT BRAIN V3] Disabled or router not available - position monitor will protect {symbol}")
                    
                    # 🎯 CRITICAL FIX: Recalculate TP/SL based on ACTUAL entry price
                    if rl_decision and actual_entry_price != price:
                        # Recalculate TP/SL with actual entry
                        if side == "buy":
                            actual_tp = actual_entry_price * (1 + rl_decision.tp_pct / 100)
                            actual_sl = actual_entry_price * (1 - rl_decision.sl_pct / 100)
                        else:  # short
                            actual_tp = actual_entry_price * (1 - rl_decision.tp_pct / 100)
                            actual_sl = actual_entry_price * (1 + rl_decision.sl_pct / 100)
                        
                        # Update decision with recalculated prices
                        decision.take_profit = actual_tp
                        decision.stop_loss = actual_sl
                        
                        logger.info(
                            f"[RECALC] {symbol} TP/SL adjusted for actual entry: "
                            f"TP=${actual_tp:.5f} ({rl_decision.tp_pct:+.1f}%), "
                            f"SL=${actual_sl:.5f} ({-rl_decision.sl_pct:.1f}%)"
                        )
                    
                    # Update stored TP/SL with actual entry price
                    if symbol in self._symbol_tpsl:
                        self._symbol_tpsl[symbol]["entry_price"] = actual_entry_price
                        self._symbol_tpsl[symbol]["stop_loss_price"] = decision.stop_loss
                        self._symbol_tpsl[symbol]["take_profit_price"] = decision.take_profit
                        
                        # Save updated TP/SL to file
                        try:
                            with open(TPSL_STORAGE_PATH, 'w') as f:
                                json.dump(self._symbol_tpsl, f)
                        except Exception as e:
                            logger.warning(f"Could not update TP/SL file: {e}")
                    
                    # [NEW] TRADESTORE - SPRINT 1 D5: Persist trade to storage
                    if self.trade_store and TRADESTORE_AVAILABLE:
                        try:
                            trade_obj = Trade(
                                trade_id=order_id,
                                symbol=symbol,
                                side=TradeSide.LONG if side == "buy" else TradeSide.SHORT,
                                status=TradeStatus.OPEN,
                                quantity=filled_qty,  # ✅ Use actual filled quantity
                                leverage=leverage or 1,
                                margin_usd=actual_margin,
                                entry_price=actual_entry_price,  # ✅ Use ACTUAL fill price
                                entry_time=datetime.utcnow(),
                                sl_price=decision.stop_loss if decision else None,
                                tp_price=decision.take_profit if decision else None,
                                trail_percent=trail_percent,
                                model=model,
                                confidence=confidence,
                                meta_strategy_id=meta_strategy_result.strategy.strategy_id.value if (self.meta_strategy and meta_strategy_result) else None,
                                regime=meta_strategy_result.regime.value if (self.meta_strategy and meta_strategy_result) else None,
                                rl_state_key=None,  # Will be set below if RL sizing was used
                                rl_action_key=None,
                                rl_leverage_original=leverage if (rl_decision and rl_decision.position_size_usd > 0) else None,
                                entry_fee_usd=actual_margin * 0.0004,  # 🔧 Use 0.04% fee estimate (estimated_fee not in scope)
                                exchange_order_id=order_id,
                                metadata={
                                    "signal_category": signal_dict.get("category") if signal_dict else None,
                                    "risk_modifier": risk_modifier,
                                    "baseline_sl_pct": baseline_sl_pct,
                                    "baseline_tp_pct": baseline_tp_pct,
                                    "partial_tp": partial_tp,
                                    "signal_price": price,  # Store original signal price for reference
                                    "slippage_pct": slippage_pct
                                }
                            )
                            await self.trade_store.save_new_trade(trade_obj)
                            logger_tradestore.info(
                                f"[OK] Trade saved: {symbol} {side.upper()} {filled_qty}@${actual_entry_price:.5f} "
                                f"Margin=${actual_margin:.2f} Leverage={leverage}x"
                            )
                        except Exception as e:
                            logger_tradestore.error(f"[ERROR] Failed to save trade {order_id}: {e}", exc_info=True)
                    
                    # [EXIT BRAIN V3] Create exit plan for immediate position protection
                    exit_plan_created = False
                    if self.exit_router and EXIT_BRAIN_V3_AVAILABLE:
                        try:
                            # 🔧 Build position dict from filled order (Binance format expects floats, not strings)
                            position_dict = {
                                "symbol": symbol,
                                "positionAmt": filled_qty if side in ["BUY", "LONG"] else -filled_qty,
                                "entryPrice": actual_entry_price,
                                "markPrice": actual_entry_price,
                                "leverage": leverage,
                                "unrealizedProfit": 0.0,
                                "notional": abs(filled_qty * actual_entry_price),
                            }
                            
                            # Build RL hints from decision
                            rl_hints = None
                            if decision and hasattr(decision, 'take_profit') and decision.take_profit:
                                rl_hints = {
                                    "tp_target_pct": tp_percent,
                                    "sl_target_pct": sl_percent,
                                    "trail_callback_pct": trail_percent,
                                    "confidence": confidence,
                                }
                            
                            # Build risk context from current state
                            risk_context = {
                                "risk_mode": "NORMAL",  # TODO: Get from ESS or risk management
                                "daily_pnl_pct": 0.0,  # TODO: Get from portfolio tracking
                                "position_count": len([p for p in positions if float(p.get('positionAmt', 0)) != 0]),
                                "max_positions": self.orchestrator.config.max_open_positions,
                            }
                            
                            # Build market data from signal
                            market_data = {
                                "current_price": actual_entry_price,
                                "volatility": market_conditions.volatility if market_conditions else 0.02,
                                "atr": signal_dict.get('atr', actual_entry_price * 0.01) if signal_dict else actual_entry_price * 0.01,
                                "regime": market_conditions.regime.value if market_conditions else "NORMAL",
                            }
                            
                            # Create Exit Brain plan
                            plan = await self.exit_router.get_or_create_plan(
                                position=position_dict,
                                rl_hints=rl_hints,
                                risk_context=risk_context,
                                market_data=market_data
                            )
                            
                            if plan and len(plan.legs) > 0:
                                logger_exit_brain.info(
                                    f"[EXIT BRAIN V3] {symbol}: Created exit plan with {len(plan.legs)} legs "
                                    f"(strategy={plan.strategy_id}, source={plan.source})"
                                )
                                exit_plan_created = True
                                
                                # Log leg details
                                for leg in plan.legs:
                                    logger_exit_brain.debug(
                                        f"  ↳ {leg.kind.name}: size={leg.size_pct*100:.0f}%, "
                                        f"trigger={leg.trigger_pct*100:+.2f}%, priority={leg.priority}"
                                    )
                            else:
                                logger_exit_brain.warning(
                                    f"[EXIT BRAIN V3] {symbol}: Plan created but has no legs - falling back to hybrid_tpsl"
                                )
                        
                        except Exception as brain_exc:
                            logger_exit_brain.error(
                                f"[EXIT BRAIN V3] {symbol}: Plan creation failed: {brain_exc}",
                                exc_info=True
                            )
                    
                    # [HYBRID-TPSL] Fallback to legacy hybrid orders if Exit Brain disabled or failed
                    if not exit_plan_created:
                        try:
                            logger.info(
                                f"[HYBRID-TPSL] {symbol}: Using legacy hybrid orders "
                                f"(Exit Brain {'failed' if self.exit_router else 'disabled'})"
                            )
                            hybrid_orders_placed = await place_hybrid_orders(
                                client=self._adapter,
                                symbol=symbol,
                                side=side,
                                entry_price=price,
                                qty=quantity,
                                risk_sl_percent=baseline_sl_pct,
                                base_tp_percent=baseline_tp_pct,
                                ai_tp_percent=tp_percent,
                                ai_trail_percent=trail_percent,
                                confidence=confidence,
                                policy_store=None,  # D7: PolicyStore for retry/slippage limits (TODO: pass actual instance)
                            )
                        except Exception as hybrid_exc:
                            logger.error(
                                f"[HYBRID-TPSL] Hybrid placement failed for {symbol}: {hybrid_exc}",
                                exc_info=True,
                            )
                    
                    # [NEW] POST-TRADE HOOKS: Slippage check
                    if AI_INTEGRATION_AVAILABLE and self.ai_services:
                        try:
                            acceptable, reason = await execution_slippage_check(
                                symbol, price, actual_entry_price  # ✅ Use actual fill price
                            )
                            if not acceptable:
                                logger.warning(f"[AELM] Slippage warning: {reason}")
                        except Exception as e:
                            logger.error(f"[ERROR] Slippage check failed: {e}", exc_info=True)
                    
                    # [TARGET] REGISTER TRADE WITH LIFECYCLE MANAGER
                    try:
                        trade = self._trade_manager.open_trade(
                            trade_id=order_id,
                            decision=decision,
                            signal_quality=signal_quality,
                            market_conditions=market_conditions,
                            actual_entry_price=actual_entry_price  # ✅ Use actual fill price
                        )
                        logger.info(f"[MEMO] Trade registered with lifecycle manager: {order_id}")
                        
                        # [FIX] CRITICAL: Store trail_pct in trade state for Trailing Stop Manager
                        try:
                            # DISABLED: Legacy utils trade_store no longer exists
                            # from backend.utils.trade_store import get_trade_store
                            trade_store = None  # Disabled - use self.trade_store (TradeStore) instead
                            state = trade_store.get(symbol)
                            if state:
                                state["ai_trail_pct"] = trail_percent
                                state["ai_tp_pct"] = tp_percent
                                state["ai_sl_pct"] = sl_percent
                                state["ai_partial_tp"] = partial_tp
                                
                                # [NEW] Store meta-strategy info for reward update on close
                                if self.meta_strategy and meta_strategy_result:
                                    state["meta_strategy"] = {
                                        "strategy_id": meta_strategy_result.strategy.strategy_id.value,
                                        "regime": meta_strategy_result.regime.value,
                                        "entry_price": price,
                                        "atr": market_data.get('atr', price * 0.01)
                                    }
                                
                                # [NEW] Store RL sizing state/action for learning on close
                                if hasattr(position_size, 'adjustment_reason') and 'RL:' in position_size.adjustment_reason:
                                    try:
                                        from backend.services.ai.rl_position_sizing_agent import get_rl_sizing_agent
                                        rl_agent = get_rl_sizing_agent(enabled=True)
                                        
                                        if rl_agent and hasattr(rl_agent, '_last_state_key') and hasattr(rl_agent, '_last_action_key'):
                                            state["rl_state_key"] = rl_agent._last_state_key
                                            state["rl_action_key"] = rl_agent._last_action_key
                                            logger.info(f"[RL-SIZING] Stored state/action for {symbol} learning")
                                    except Exception as rl_err:
                                        logger.debug(f"[RL-SIZING] Could not store state/action: {rl_err}")
                                
                                trade_store.set(symbol, state)
                                logger.info(
                                    f"[TARGET] Trail config stored for {symbol}: "
                                    f"Trail={trail_percent*100:.1f}% TP={tp_percent*100:.1f}% SL={sl_percent*100:.1f}%"
                                )
                            else:
                                logger.warning(f"[WARNING] No trade state found for {symbol} - trail config not stored")
                        except Exception as trail_error:
                            logger.error(f"[ERROR] Failed to store trail config: {trail_error}")
                        
                        # [TODO] META-STRATEGY REWARD UPDATE: Add this to position close handler
                        # When trade closes, calculate realized R and update Meta-Strategy:
                        # 
                        # if self.meta_strategy and symbol in trade_store:
                        #     meta_info = trade_store.get(symbol).get("meta_strategy")
                        #     if meta_info:
                        #         entry_price = meta_info["entry_price"]
                        #         atr = meta_info["atr"]
                        #         exit_price = <get from position close>
                        #         
                        #         # Calculate realized R
                        #         if side == "LONG":
                        #             realized_r = (exit_price - entry_price) / atr
                        #         else:  # SHORT
                        #             realized_r = (entry_price - exit_price) / atr
                        #         
                        #         # Update RL reward
                        #         await self.meta_strategy.update_strategy_reward(
                        #             symbol=symbol,
                        #             realized_r=realized_r,
                        #             trade_meta={"pnl": pnl, "duration_hours": duration}
                        #         )
                        #         logger.info(f"[RL UPDATE] {symbol}: R={realized_r:+.2f}")
                        
                        # [FIX] CRITICAL: Place SL IMMEDIATELY after trade opens (don't wait for position_monitor)
                        # This prevents "order would immediately trigger" errors with high leverage
                        if not hybrid_orders_placed:
                            try:
                                await self._place_immediate_stop_loss(
                                    symbol=symbol,
                                    side=side,
                                    entry_price=actual_entry_price,  # ✅ Use actual fill price
                                    stop_loss_price=decision.stop_loss,
                                    quantity=filled_qty  # ✅ Use actual filled quantity
                                )
                            except Exception as sl_error:
                                logger.error(f"[CRITICAL] Failed to place immediate SL for {symbol}: {sl_error}")
                                # Emergency close if SL placement fails
                                try:
                                    logger.error(f"[ALERT] EMERGENCY CLOSE: {symbol} - SL placement failed, closing position")
                                    close_side = 'SELL' if side == 'buy' else 'BUY'
                                    await self._adapter.submit_order(
                                        symbol=symbol,
                                        side=close_side,
                                        quantity=quantity,
                                        price=price  # Close at market
                                    )
                                    logger.error(f"[OK] Emergency close executed for {symbol}")
                                    orders_failed += 1
                                    continue  # Skip to next order
                                except Exception as close_error:
                                    logger.error(f"[CRITICAL] Emergency close FAILED for {symbol}: {close_error}")
                        else:
                            logger.info("[HYBRID-TPSL] Hybrid TP/SL already placed; skipping legacy immediate SL")
                        
                        # [NEW] POST-TRADE HOOKS: Position classification and amplification
                        if AI_INTEGRATION_AVAILABLE and self.ai_services:
                            try:
                                # Build position dict for hooks (use ACTUAL entry price)
                                position_data = {
                                    "symbol": symbol,
                                    "side": side,
                                    "entry_price": actual_entry_price,  # ✅ Use actual fill price
                                    "quantity": filled_qty,  # ✅ Use actual filled quantity
                                    "stop_loss": decision.stop_loss,
                                    "take_profit": decision.take_profit,
                                    "confidence": confidence,
                                    "trade_id": order_id
                                }
                                
                                # Classify position (PIL)
                                classified_position = await post_trade_position_classification(
                                    position_data
                                )
                                logger.info(f"[PIL] Position classified: {classified_position}")
                                
                                # Check amplification opportunity (PAL)
                                recommendation = await post_trade_amplification_check(
                                    position_data
                                )
                                if recommendation:
                                    logger.info(f"[PAL] Amplification opportunity: {recommendation}")
                                    
                            except Exception as e:
                                logger.error(f"[ERROR] Post-trade hooks failed: {e}", exc_info=True)
                        
                        # [NEW] Log final entry confirmation with ACTUAL prices
                        logger.info(
                            f"📈 [ENTRY CONFIRMED] {symbol} {side.upper()}\n"
                            f"   Entry: ${actual_entry_price:.5f} (Signal: ${price:.5f}, Slippage: {slippage_pct:+.2f}%)\n"
                            f"   Quantity: {filled_qty:.4f}\n"
                            f"   TP: ${decision.take_profit:.5f} (+{((decision.take_profit/actual_entry_price - 1)*100 if side == 'buy' else (1 - decision.take_profit/actual_entry_price)*100):.2f}%)\n"
                            f"   SL: ${decision.stop_loss:.5f} ({((decision.stop_loss/actual_entry_price - 1)*100 if side == 'buy' else (1 - decision.stop_loss/actual_entry_price)*100):+.2f}%)\n"
                            f"   Confidence: {confidence:.1%}"
                        )
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to register trade with lifecycle manager: {e}")
                    
                    orders_submitted += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to place order for {symbol}: {e}", exc_info=True)
                    orders_failed += 1
            
            return {
                "status": "ok",
                "orders_planned": len(orders_to_place),
                "orders_submitted": orders_submitted,
                "orders_skipped": orders_skipped,
                "orders_failed": orders_failed
            }
            
        except Exception as e:
            logger.error(f"❌ Direct execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "orders_planned": 0,
                "orders_submitted": 0,
                "orders_skipped": 0,
                "orders_failed": 0
            }
    
    async def _get_current_positions(self) -> dict[str, float]:
        """
        Get current portfolio positions as {symbol: net_qty}.
        
        Used by position invariant enforcer to check for conflicts.
        Positive quantity = LONG position, negative = SHORT position.
        
        CRITICAL: In hedge mode, Binance returns BOTH long and short positions separately.
        We need to detect this and sum them properly for invariant checking.
        
        Returns:
            dict[str, float]: {symbol: net_quantity}
        """
        try:
            if hasattr(self, 'portfolio_position_service') and self.portfolio_position_service:
                positions = self.portfolio_position_service.all()
                return {p.symbol: p.quantity for p in positions}
            else:
                # Fallback: Try to get positions from adapter
                if hasattr(self, '_adapter') and self._adapter:
                    # Get raw positions from exchange
                    positions_list = await self._adapter.get_open_positions()
                    
                    # CRITICAL FIX: Handle hedge mode properly
                    # In hedge mode, same symbol can have BOTH long and short positions
                    result = {}
                    hedge_mode_detected = False
                    
                    for pos in positions_list:
                        symbol = pos.symbol
                        qty = float(pos.quantity)  # Always positive
                        side = pos.side  # 'long' or 'short'
                        position_side = getattr(pos, 'position_side', None)
                        
                        # Detect hedge mode: If we see position_side = 'LONG' or 'SHORT'
                        if position_side and position_side.value in ['LONG', 'SHORT']:
                            hedge_mode_detected = True
                            logger.warning(
                                f"⚠️ [HEDGE MODE DETECTED] Symbol {symbol} has {position_side.value} position. "
                                f"System will track each side separately."
                            )
                        
                        # Convert to signed quantity
                        if side.lower() == 'long':
                            signed_qty = qty
                        else:  # short
                            signed_qty = -qty
                        
                        # In hedge mode, track each position side separately
                        # In one-way mode, sum them (should only be one anyway)
                        if symbol in result:
                            # Symbol already exists - this indicates hedge mode
                            logger.warning(
                                f"🚨 [HEDGE MODE CONFLICT] Symbol {symbol} has MULTIPLE positions: "
                                f"existing={result[symbol]:.8f}, new={signed_qty:.8f}. "
                                f"This violates trading invariants!"
                            )
                            result[symbol] += signed_qty
                        else:
                            result[symbol] = signed_qty
                    
                    if hedge_mode_detected:
                        logger.error(
                            "🛑 [CRITICAL] Hedge mode detected on exchange! "
                            "System is designed for ONE-WAY mode only. "
                            "Please disable hedge mode on exchange or enable QT_ALLOW_HEDGING=true"
                        )
                    
                    return result
                else:
                    logger.warning("[WARN] No position service or adapter available, returning empty positions")
                    return {}
        except Exception as e:
            logger.error(f"[ERROR] Failed to get current positions: {e}", exc_info=True)
            return {}
    
    async def _place_immediate_stop_loss(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell' (entry side)
        entry_price: float,
        stop_loss_price: float,
        quantity: float
    ) -> bool:
        """
        [FIX] CRITICAL: Place stop loss IMMEDIATELY after trade entry.
        
        This prevents "order would immediately trigger" errors by placing SL
        within milliseconds of entry, before price moves significantly with leverage.
        
        Args:
            symbol: Trading symbol
            side: Entry side ('buy' or 'sell')
            entry_price: Actual entry price
            stop_loss_price: Calculated SL price
            quantity: Position quantity
            
        Returns:
            True if SL placed successfully, False otherwise
            
        Raises:
            Exception if SL cannot be placed and position must be emergency closed
        """
        from binance.client import Client
        from binance.exceptions import BinanceAPIException
        
        # Initialize Binance client (use same settings as position_monitor)
        use_testnet = os.getenv("USE_BINANCE_TESTNET", "false").lower() == "true"
        
        if use_testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")
            client = Client(api_key, api_secret, testnet=True)
            client.API_URL = 'https://testnet.binancefuture.com'
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            client = Client(api_key, api_secret)
        
        # Determine SL side (opposite of entry)
        sl_side = 'SELL' if side == 'buy' else 'BUY'
        
        # Get price precision
        try:
            exchange_info = client.futures_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            if symbol_info:
                price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if price_filter:
                    tick_size = float(price_filter['tickSize'])
                    tick_str = f"{tick_size:.10f}".rstrip('0')
                    if '.' in tick_str:
                        price_precision = len(tick_str.split('.')[-1])
                    else:
                        price_precision = 0
                else:
                    price_precision = 5
            else:
                price_precision = 5
        except Exception as e:
            logger.warning(f"Could not get price precision for {symbol}: {e}")
            price_precision = 5
        
        # Round SL price
        sl_price = round(stop_loss_price, price_precision)
        
        logger.info(f"[SHIELD] Placing IMMEDIATE SL for {symbol}: {sl_side} @ ${sl_price:.{price_precision}f}")
        
        # Attempt 1: Place SL at calculated price
        try:
            # [PHASE 1] Route through exit gateway for observability
            sl_order_params = {
                'symbol': symbol,
                'side': sl_side,
                'type': 'STOP_MARKET',
                'stopPrice': sl_price,
                'closePosition': True,
                'workingType': 'MARK_PRICE'
            }
            
            if EXIT_GATEWAY_AVAILABLE:
                sl_order = await submit_exit_order(
                    module_name="event_driven_executor",
                    symbol=symbol,
                    order_params=sl_order_params,
                    order_kind="sl",
                    client=client,
                    explanation="Emergency SL shield (initial attempt)"
                )
            else:
                sl_order = client.futures_create_order(**sl_order_params)
            
            logger.info(f"   [OK] SL placed successfully: order ID {sl_order.get('orderId')}")
            return True
            
        except BinanceAPIException as e:
            if e.code == -2021:  # Order would immediately trigger
                logger.warning(f"   [WARNING] SL would immediately trigger @ ${sl_price:.{price_precision}f}")
                
                # Attempt 2: Adjust SL price with buffer (0.05% away from current price)
                try:
                    # Get current mark price
                    ticker = client.futures_mark_price(symbol=symbol)
                    current_price = float(ticker['markPrice'])
                    
                    # Calculate new SL with buffer
                    if side == 'buy':  # LONG position
                        # SL should be below current price
                        buffer_price = current_price * 0.9995  # 0.05% below
                        adjusted_sl = min(sl_price, buffer_price)
                    else:  # SHORT position
                        # SL should be above current price
                        buffer_price = current_price * 1.0005  # 0.05% above
                        adjusted_sl = max(sl_price, buffer_price)
                    
                    adjusted_sl = round(adjusted_sl, price_precision)
                    
                    logger.warning(f"   [RETRY] Adjusting SL: ${sl_price:.{price_precision}f} → ${adjusted_sl:.{price_precision}f} (current: ${current_price:.{price_precision}f})")
                    
                    # [PHASE 1] Route through exit gateway for observability
                    adjusted_sl_params = {
                        'symbol': symbol,
                        'side': sl_side,
                        'type': 'STOP_MARKET',
                        'stopPrice': adjusted_sl,
                        'closePosition': True,
                        'workingType': 'MARK_PRICE'
                    }
                    
                    if EXIT_GATEWAY_AVAILABLE:
                        sl_order = await submit_exit_order(
                            module_name="event_driven_executor",
                            symbol=symbol,
                            order_params=adjusted_sl_params,
                            order_kind="sl",
                            client=client,
                            explanation=f"Emergency SL shield (adjusted retry: ${adjusted_sl:.{price_precision}f})"
                        )
                    else:
                        sl_order = client.futures_create_order(**adjusted_sl_params)
                    
                    logger.warning(f"   [OK] Adjusted SL placed: order ID {sl_order.get('orderId')}")
                    return True
                    
                except Exception as retry_error:
                    logger.error(f"   [CRITICAL] SL retry failed: {retry_error}")
                    raise Exception(f"Cannot place SL for {symbol} - position MUST be closed")
            else:
                # Other Binance API error
                logger.error(f"   [CRITICAL] Binance API error placing SL: {e}")
                raise Exception(f"SL placement failed for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"   [CRITICAL] Unexpected error placing SL: {e}")
            raise Exception(f"SL placement failed for {symbol}: {e}")


# Global instance (initialized at startup)
_executor: Optional[EventDrivenExecutor] = None


async def start_event_driven_executor(
    ai_engine: AITradingEngine,
    symbols: List[str],
    confidence_threshold: float = 0.65,
    check_interval: int = 30,
    cooldown: int = 300,
    app_state: Optional[any] = None,  # [NEW] App state for Safety Governor access
    ai_services: Optional['AISystemServices'] = None,  # [NEW] AI System Services
    policy_store = None,  # [NEW] PolicyStore for centralized config
    event_bus = None,  # [CRITICAL FIX #1] EventBus for infrastructure health checks
) -> "EventDrivenExecutor":
    """Start the global event-driven executor. Returns executor instance.
    
    IMPORTANT: The executor starts a background task. The caller MUST keep
    a reference to executor._task to prevent garbage collection!
    """
    global _executor
    
    if _executor is not None:
        logger.warning("Event-driven executor already running")
        return _executor
    
    # Store policy_store reference for signal filtering
    if policy_store:
        logger.info("[EventDrivenExecutor] PolicyStore connected - will read confidence threshold dynamically")
    
    _executor = EventDrivenExecutor(
        ai_engine=ai_engine,
        symbols=symbols,
        confidence_threshold=confidence_threshold,
        check_interval_seconds=check_interval,
        cooldown_seconds=cooldown,
        app_state=app_state,  # Pass app_state for Safety Governor access
        ai_services=ai_services,  # [FIX] Pass AI services for PAL integration
        event_bus=event_bus,  # [CRITICAL FIX #1] Pass EventBus for infrastructure health checks
    )
    
    # Attach policy_store to executor for runtime access
    _executor.policy_store = policy_store
    
    await _executor.start()
    
    # [WARNING] CRITICAL: Verify task was created and is running
    if _executor._task is None or _executor._task.done():
        logger.error("❌ CRITICAL: Event-driven executor task failed to start!")
        raise RuntimeError("Event-driven executor task not running")
    else:
        logger.info("[OK] Event-driven executor task confirmed running: %s", _executor._task.get_name())
    
    return _executor


async def stop_event_driven_executor():
    """Stop the global event-driven executor."""
    global _executor
    
    if _executor is None:
        return
    
    await _executor.stop()
    _executor = None


def is_event_driven_active() -> bool:
    """Check if event-driven executor is running."""
    return _executor is not None and _executor._running
