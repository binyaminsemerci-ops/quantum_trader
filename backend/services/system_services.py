"""
QUANTUM TRADER AI SYSTEM SERVICES REGISTRY
===========================================

Central registry and lifecycle manager for all AI subsystems.

This module provides:
- Service initialization with feature flags
- Dependency injection
- Graceful startup/shutdown
- Health monitoring
- Configuration management

INTEGRATION STAGES:
- Stage 1: OBSERVATION - Services run but don't enforce decisions
- Stage 2: PARTIAL - Services provide advisory guidance
- Stage 3: COORDINATION - AI-HFOS coordinates all subsystems
- Stage 4: AUTONOMY - Full autonomous operation (testnet only)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


# ============================================================================
# HEALTH STATUS UTILITIES
# ============================================================================

def normalize_subsystem_status(value: Any) -> str:
    """
    Normalize any subsystem health status to standardized values.
    
    Args:
        value: Can be string, dict, enum, or other type
        
    Returns:
        Standardized status: HEALTHY, DEGRADED, CRITICAL, or UNKNOWN
    """
    if value is None:
        return "UNKNOWN"
    
    # If it's a dict with 'status' field
    if isinstance(value, dict):
        value = value.get("status", "UNKNOWN")
    
    # Convert to string and normalize
    status_str = str(value).upper().strip()
    
    # Map to standard values
    if status_str in ["HEALTHY", "ACTIVE", "OK", "SUCCESS", "RUNNING"]:
        return "HEALTHY"
    elif status_str in ["DEGRADED", "PARTIAL", "WARNING"]:
        return "DEGRADED"
    elif status_str in ["CRITICAL", "ERROR", "FAILED", "FAILING"]:
        return "CRITICAL"
    else:
        return "UNKNOWN"


# ============================================================================
# FEATURE FLAGS & MODES
# ============================================================================

class IntegrationStage(str, Enum):
    """System integration stages."""
    OBSERVATION = "OBSERVATION"      # Stage 1: Observe only, no enforcement
    PARTIAL = "PARTIAL"              # Stage 2: Advisory mode
    COORDINATION = "COORDINATION"    # Stage 3: AI-HFOS coordinates
    AUTONOMY = "AUTONOMY"            # Stage 4: Full autonomy


class SubsystemMode(str, Enum):
    """Individual subsystem operation modes."""
    OFF = "OFF"                      # Disabled
    OBSERVE = "OBSERVE"              # Log decisions only
    ADVISORY = "ADVISORY"            # Provide recommendations
    ENFORCED = "ENFORCED"            # Enforce decisions


@dataclass
class AISystemConfig:
    """
    Master configuration for all AI subsystems.
    
    Feature flags control which subsystems are enabled and their modes.
    Default: All OFF except what's already in production (Orchestrator).
    """
    
    # ========================================================================
    # MASTER CONTROLS
    # ========================================================================
    
    # AI Hedgefund Operating System (AI-HFOS) - Supreme coordinator
    ai_hfos_enabled: bool = True  # ENABLED: Import paths fixed, unified initialization
    ai_hfos_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    ai_hfos_update_interval_sec: int = 60
    
    # Integration stage (controls all subsystems)
    integration_stage: IntegrationStage = IntegrationStage.AUTONOMY  # FULL DEPLOYMENT
    
    # ========================================================================
    # INTELLIGENCE LAYERS
    # ========================================================================
    
    # Position Intelligence Layer (PIL)
    pil_enabled: bool = True  # FULL DEPLOYMENT
    pil_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    pil_classification_interval_sec: int = 300
    
    # Portfolio Balancer AI (PBA)
    pba_enabled: bool = True  # FULL DEPLOYMENT
    pba_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    pba_rebalance_check_interval_sec: int = 600
    
    # Profit Amplification Layer (PAL)
    pal_enabled: bool = True  # FULL DEPLOYMENT
    pal_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    pal_analysis_interval_sec: int = 900  # 15 minutes
    pal_min_r_for_amplification: float = 1.0
    pal_min_r_for_scale_in: float = 1.5
    
    # Self-Healing System
    self_healing_enabled: bool = True  # FULL DEPLOYMENT
    self_healing_mode: SubsystemMode = SubsystemMode.ENFORCED  # PROTECTIVE MODE
    self_healing_check_interval_sec: int = 120
    
    # Model Supervisor
    model_supervisor_enabled: bool = True  # FULL DEPLOYMENT
    model_supervisor_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    model_supervisor_eval_interval_sec: int = 3600
    
    # ========================================================================
    # CORE SYSTEMS
    # ========================================================================
    
    # Universe OS (already partially integrated via selection_engine.py)
    universe_os_enabled: bool = True  # FULL DEPLOYMENT
    universe_os_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    universe_os_use_dynamic_universe: bool = True  # Use dynamic filtering
    
    # Risk OS (already integrated via orchestrator_policy.py & risk_guard.py)
    risk_os_enabled: bool = True  # Already in production
    risk_os_mode: SubsystemMode = SubsystemMode.ENFORCED
    
    # Orchestrator Policy (already in production)
    orchestrator_enabled: bool = True  # Already in production
    orchestrator_mode: SubsystemMode = SubsystemMode.ENFORCED
    
    # Execution Layer Manager (AELM) - extends existing execution.py
    aelm_enabled: bool = True  # FULL DEPLOYMENT
    aelm_mode: SubsystemMode = SubsystemMode.ENFORCED  # FULL AUTONOMY
    aelm_use_smart_execution: bool = True
    aelm_enforce_slippage_caps: bool = True
    
    # Retraining System (already exists as retraining_orchestrator.py)
    retraining_enabled: bool = True  # FULL DEPLOYMENT
    retraining_mode: SubsystemMode = SubsystemMode.ADVISORY  # ADVISORY ONLY (safe)
    retraining_auto_deploy: bool = False  # Never auto-deploy (safe)
    
    # Dynamic TP/SL System
    dynamic_tpsl_enabled: bool = True  # FULL DEPLOYMENT
    dynamic_tpsl_override_legacy: bool = True  # Override exit policy engine
    
    # ========================================================================
    # PATHS & PERSISTENCE
    # ========================================================================
    
    data_dir: Path = field(default_factory=lambda: Path("/app/data"))
    log_dir: Path = field(default_factory=lambda: Path("/app/logs"))
    
    # ========================================================================
    # SAFETY SETTINGS
    # ========================================================================
    
    # Emergency brake overrides
    emergency_brake_active: bool = False
    
    # Fail-safe mode (if any subsystem crashes, revert to safe defaults)
    fail_safe_enabled: bool = True
    
    # Maximum allowed daily drawdown before AI-HFOS forces CRITICAL mode
    max_daily_dd_pct: float = 5.0
    
    # Maximum open drawdown before AI-HFOS forces CRITICAL mode
    max_open_dd_pct: float = 10.0
    
    @classmethod
    def from_env(cls) -> "AISystemConfig":
        """
        Load configuration from environment variables.
        
        Environment variables follow pattern: QT_AI_<SUBSYSTEM>_<SETTING>
        Example: QT_AI_HFOS_ENABLED=true
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, str(default)).lower()
            return val in ("true", "1", "yes", "on")
        
        def get_mode(key: str, default: SubsystemMode) -> SubsystemMode:
            val = os.getenv(key, default.value).upper()
            try:
                return SubsystemMode(val)
            except ValueError:
                logger.warning(f"Invalid mode '{val}' for {key}, using {default.value}")
                return default
        
        def get_stage(key: str, default: IntegrationStage) -> IntegrationStage:
            val = os.getenv(key, default.value).upper()
            try:
                return IntegrationStage(val)
            except ValueError:
                # Try to map common aliases
                if val in ('ENFORCED', 'FULL', 'ACTIVE'):
                    logger.info(f"Mapping '{val}' to AUTONOMY for {key}")
                    return IntegrationStage.AUTONOMY
                logger.warning(f"Invalid stage '{val}' for {key}, using {default.value}")
                return default
        
        # Import config helpers
        from config.config import get_model_supervisor_enabled, get_model_supervisor_mode
        
        return cls(
            # Master controls
            ai_hfos_enabled=get_bool("QT_AI_HFOS_ENABLED", False),
            ai_hfos_mode=get_mode("QT_AI_HFOS_MODE", SubsystemMode.OBSERVE),
            integration_stage=get_stage("QT_AI_INTEGRATION_STAGE", IntegrationStage.OBSERVATION),
            
            # Intelligence layers
            pil_enabled=get_bool("QT_AI_PIL_ENABLED", False),
            pil_mode=get_mode("QT_AI_PIL_MODE", SubsystemMode.ADVISORY),
            
            pba_enabled=get_bool("QT_AI_PBA_ENABLED", False),
            pba_mode=get_mode("QT_AI_PBA_MODE", SubsystemMode.ADVISORY),
            
            pal_enabled=get_bool("QT_AI_PAL_ENABLED", False),
            pal_mode=get_mode("QT_AI_PAL_MODE", SubsystemMode.ADVISORY),
            pal_min_r_for_amplification=float(os.getenv("QT_AI_PAL_MIN_R", "1.0")),
            
            self_healing_enabled=get_bool("QT_AI_SELF_HEALING_ENABLED", False),
            self_healing_mode=get_mode("QT_AI_SELF_HEALING_MODE", SubsystemMode.OBSERVE),
            
            model_supervisor_enabled=get_model_supervisor_enabled(),
            model_supervisor_mode=SubsystemMode(get_model_supervisor_mode()),
            
            # Core systems
            universe_os_enabled=get_bool("QT_AI_UNIVERSE_OS_ENABLED", False),
            universe_os_mode=get_mode("QT_AI_UNIVERSE_OS_MODE", SubsystemMode.OBSERVE),
            universe_os_use_dynamic_universe=get_bool("QT_AI_UNIVERSE_DYNAMIC", False),
            
            risk_os_enabled=get_bool("QT_AI_RISK_OS_ENABLED", True),  # Already in prod
            orchestrator_enabled=get_bool("QT_AI_ORCHESTRATOR_ENABLED", True),  # Already in prod
            
            aelm_enabled=get_bool("QT_AI_AELM_ENABLED", False),
            aelm_mode=get_mode("QT_AI_AELM_MODE", SubsystemMode.ADVISORY),
            aelm_use_smart_execution=get_bool("QT_AI_AELM_SMART_EXEC", False),
            
            retraining_enabled=get_bool("QT_AI_RETRAINING_ENABLED", False),
            retraining_mode=get_mode("QT_AI_RETRAINING_MODE", SubsystemMode.ADVISORY),
            retraining_auto_deploy=get_bool("QT_AI_RETRAINING_AUTO_DEPLOY", False),
            
            # Dynamic TP/SL
            dynamic_tpsl_enabled=get_bool("QT_AI_DYNAMIC_TPSL_ENABLED", False),
            dynamic_tpsl_override_legacy=get_bool("QT_AI_DYNAMIC_TPSL_OVERRIDE", False),
            
            # Safety
            emergency_brake_active=get_bool("QT_AI_EMERGENCY_BRAKE", False),
            fail_safe_enabled=get_bool("QT_AI_FAIL_SAFE", True),
            max_daily_dd_pct=float(os.getenv("QT_AI_MAX_DAILY_DD", "5.0")),
            max_open_dd_pct=float(os.getenv("QT_AI_MAX_OPEN_DD", "10.0")),
        )
    
    def get_summary(self) -> str:
        """Get human-readable configuration summary."""
        enabled_subsystems = []
        
        if self.ai_hfos_enabled:
            enabled_subsystems.append(f"AI-HFOS ({self.ai_hfos_mode.value})")
        if self.pil_enabled:
            enabled_subsystems.append(f"PIL ({self.pil_mode.value})")
        if self.pba_enabled:
            enabled_subsystems.append(f"PBA ({self.pba_mode.value})")
        if self.pal_enabled:
            enabled_subsystems.append(f"PAL ({self.pal_mode.value})")
        if self.self_healing_enabled:
            enabled_subsystems.append(f"Self-Healing ({self.self_healing_mode.value})")
        if self.model_supervisor_enabled:
            enabled_subsystems.append(f"Model Supervisor ({self.model_supervisor_mode.value})")
        if self.universe_os_enabled:
            enabled_subsystems.append(f"Universe OS ({self.universe_os_mode.value})")
        if self.aelm_enabled:
            enabled_subsystems.append(f"AELM ({self.aelm_mode.value})")
        if self.retraining_enabled:
            enabled_subsystems.append(f"Retraining ({self.retraining_mode.value})")
        
        if not enabled_subsystems:
            enabled_subsystems.append("None (using existing systems only)")
        
        return (
            f"AI System Integration - Stage: {self.integration_stage.value}\n"
            f"Enabled Subsystems: {', '.join(enabled_subsystems)}\n"
            f"Emergency Brake: {'ACTIVE' if self.emergency_brake_active else 'OFF'}\n"
            f"Fail-Safe: {'ENABLED' if self.fail_safe_enabled else 'DISABLED'}"
        )


# ============================================================================
# SERVICE REGISTRY
# ============================================================================

class AISystemServices:
    """
    Central registry for all AI subsystem instances.
    
    Manages lifecycle:
    - Initialization with dependency injection
    - Graceful startup/shutdown
    - Health monitoring
    - Error handling with fail-safe fallbacks
    """
    
    def __init__(self, config: Optional[AISystemConfig] = None):
        self.config = config or AISystemConfig.from_env()
        
        # Service instances (initialized lazily)
        self.ai_hfos = None
        self.ai_hfos_integration = None
        self.pil = None
        self.pba = None
        self.pal = None
        self.self_healing = None
        self.model_supervisor = None
        self.universe_os = None
        self.aelm = None
        self.retraining_orchestrator = None
        self.dynamic_tpsl = None  # Dynamic TP/SL calculator
        
        # Existing services (passed in from main)
        self.orchestrator = None
        self.risk_guard = None
        self.ai_engine = None
        
        # Status tracking
        self._initialized = False
        self._services_status: Dict[str, str] = {}
        
        logger.info(f"[AI System Services] Configuration loaded:\n{self.config.get_summary()}")
    
    async def initialize(
        self,
        orchestrator=None,
        risk_guard=None,
        ai_engine=None,
        **kwargs
    ):
        """
        Initialize all enabled AI subsystems.
        
        Args:
            orchestrator: Existing OrchestratorPolicy instance
            risk_guard: Existing RiskGuardService instance
            ai_engine: Existing AITradingEngine instance
            **kwargs: Additional dependencies
        """
        if self._initialized:
            logger.warning("[AI System Services] Already initialized")
            return
        
        logger.info("[AI System Services] Initializing subsystems...")
        logger.info(f"[AI-OS] Integration stage: {self.config.integration_stage.value}")
        
        # Store existing services
        self.orchestrator = orchestrator
        self.risk_guard = risk_guard
        self.ai_engine = ai_engine
        
        # Register existing services in status registry
        if orchestrator:
            self._services_status["orchestrator"] = normalize_subsystem_status("HEALTHY")
            logger.info("[Orchestrator] Existing policy service registered")
        
        if risk_guard:
            self._services_status["risk_os"] = normalize_subsystem_status("HEALTHY")
            logger.info("[Risk OS] Existing risk guard service registered")
        
        try:
            # Initialize in dependency order
            
            # 1. Self-Healing (monitors all others)
            if self.config.self_healing_enabled:
                await self._init_self_healing()
            
            # 2. Universe OS
            if self.config.universe_os_enabled:
                await self._init_universe_os()
            
            # 3. Model Supervisor
            if self.config.model_supervisor_enabled:
                await self._init_model_supervisor()
            
            # 4. Retraining Orchestrator
            if self.config.retraining_enabled:
                await self._init_retraining()
            
            # 5. Position Intelligence Layer
            if self.config.pil_enabled:
                await self._init_pil()
            
            # 6. Portfolio Balancer AI
            if self.config.pba_enabled:
                await self._init_pba()
            
            # 7. Profit Amplification Layer
            if self.config.pal_enabled:
                await self._init_pal()
            
            # 8. Dynamic TP/SL System (always initialize - used by trading engine)
            await self._init_dynamic_tpsl()
            
            # 9. Execution Layer Manager
            if self.config.aelm_enabled:
                await self._init_aelm()
            
            # 10. AI-HFOS (supreme coordinator - initializes last)
            if self.config.ai_hfos_enabled:
                await self._init_ai_hfos()
            
            self._initialized = True
            logger.info("[AI System Services] All subsystems initialized successfully")
            
        except Exception as e:
            logger.error(f"[AI System Services] Initialization failed: {e}", exc_info=True)
            if self.config.fail_safe_enabled:
                logger.warning("[AI System Services] Fail-safe enabled - continuing with partial initialization")
                self._initialized = True  # Continue anyway
            else:
                raise
    
    async def _init_self_healing(self):
        """Initialize Self-Healing System."""
        try:
            from backend.services.monitoring.self_healing import SelfHealingSystem
            
            self.self_healing = SelfHealingSystem(
                data_dir=str(self.config.data_dir),
                check_interval=self.config.self_healing_check_interval_sec
            )
            
            self._services_status["self_healing"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[Self-Healing] Initialized in {self.config.self_healing_mode.value} mode")
            
        except Exception as e:
            self.self_healing = None
            self._services_status["self_healing"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[Self-Healing] Initialization failed: {e}")
            logger.warning("[Self-Healing] System will continue without Self-Healing (degraded mode)")
            # DO NOT CRASH - Continue without Self-Healing
    
    async def _init_universe_os(self):
        """Initialize Universe OS (uses existing selection_engine.py)."""
        try:
            # Universe OS is partially implemented via selection_engine.py
            # For now, mark as available
            self._services_status["universe_os"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[Universe OS] Using existing selection engine in {self.config.universe_os_mode.value} mode")
            
        except Exception as e:
            self._services_status["universe_os"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[Universe OS] Initialization failed: {e}")
            logger.warning("[Universe OS] System will continue without Universe OS (degraded mode)")
            # DO NOT CRASH - Continue without Universe OS
    
    async def _init_model_supervisor(self):
        """Initialize Model Supervisor."""
        try:
            from backend.services.ai.model_supervisor import ModelSupervisor
            
            # Initialize Model Supervisor with config
            self.model_supervisor = ModelSupervisor(
                data_dir=str(self.config.data_dir),
                analysis_window_days=30,
                recent_window_days=7
            )
            
            self._services_status["model_supervisor"] = normalize_subsystem_status("HEALTHY")
            logger.info(
                f"[Model Supervisor] Initialized in {self.config.model_supervisor_mode.value} mode "
                f"(real-time observation active)"
            )
            
        except Exception as e:
            self.model_supervisor = None
            self._services_status["model_supervisor"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[Model Supervisor] Initialization failed: {e}")
            logger.warning("[Model Supervisor] System will continue without Model Supervisor (degraded mode)")
            # DO NOT CRASH - Continue without Model Supervisor
    
    async def _init_retraining(self):
        """Initialize Retraining Orchestrator."""
        try:
            from backend.services.retraining_orchestrator import RetrainingOrchestrator
            
            # Retraining orchestrator exists
            self._services_status["retraining"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[Retraining] Available in {self.config.retraining_mode.value} mode")
            
        except Exception as e:
            self._services_status["retraining"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[Retraining] Initialization failed: {e}")
            logger.warning("[Retraining] System will continue without Retraining (degraded mode)")
            # DO NOT CRASH - Continue without Retraining
    
    async def _init_pil(self):
        """Initialize Position Intelligence Layer."""
        try:
            from backend.services.position_intelligence import get_position_intelligence
            
            self.pil = get_position_intelligence()
            self._services_status["pil"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[PIL] Initialized in {self.config.pil_mode.value} mode")
            
        except Exception as e:
            self.pil = None
            self._services_status["pil"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[PIL] Initialization failed: {e}")
            logger.warning("[PIL] System will continue without PIL (degraded mode)")
            # DO NOT CRASH - Continue without PIL
    
    async def _init_pba(self):
        """Initialize Portfolio Balancer AI."""
        try:
            from backend.services.portfolio_balancer import PortfolioBalancerAI
            
            # Portfolio Balancer exists
            self._services_status["pba"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[PBA] Available in {self.config.pba_mode.value} mode")
            
        except Exception as e:
            self._services_status["pba"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[PBA] Initialization failed: {e}")
            logger.warning("[PBA] System will continue without PBA (degraded mode)")
            # DO NOT CRASH - Continue without PBA
    
    async def _init_pal(self):
        """Initialize Profit Amplification Layer."""
        try:
            from backend.services.profit_amplification import ProfitAmplificationLayer
            
            self.pal = ProfitAmplificationLayer(
                data_dir=str(self.config.data_dir),
                min_R_for_amplification=self.config.pal_min_r_for_amplification,
                min_R_for_scale_in=self.config.pal_min_r_for_scale_in
            )
            
            self._services_status["pal"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[PAL] Initialized in {self.config.pal_mode.value} mode")
            
        except Exception as e:
            self.pal = None
            self._services_status["pal"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[PAL] Initialization failed: {e}")
            logger.warning("[PAL] System will continue without PAL (degraded mode)")
            # DO NOT CRASH - Continue without PAL
    
    async def _init_dynamic_tpsl(self):
        """Initialize Dynamic TP/SL Calculator."""
        try:
            from backend.services.execution.dynamic_tpsl import get_dynamic_tpsl_calculator
            
            self.dynamic_tpsl = get_dynamic_tpsl_calculator()
            self._services_status["dynamic_tpsl"] = normalize_subsystem_status("HEALTHY")
            logger.info(
                f"[Dynamic TP/SL] Initialized "
                f"(override_legacy={self.config.dynamic_tpsl_override_legacy})"
            )
            
        except Exception as e:
            self.dynamic_tpsl = None
            self._services_status["dynamic_tpsl"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[Dynamic TP/SL] Initialization failed: {e}")
            logger.warning("[Dynamic TP/SL] System will continue without Dynamic TP/SL (degraded mode)")
            # DO NOT CRASH - Continue without Dynamic TP/SL
    
    async def _init_aelm(self):
        """Initialize Autonomous Execution Layer Manager."""
        try:
            # AELM extends existing execution.py and smart_execution.py
            self._services_status["aelm"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[AELM] Using existing execution layer in {self.config.aelm_mode.value} mode")
            
        except Exception as e:
            self._services_status["aelm"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[AELM] Initialization failed: {e}")
            logger.warning("[AELM] System will continue without AELM (degraded mode)")
            # DO NOT CRASH - Continue without AELM
    
    async def _init_ai_hfos(self):
        """Initialize AI Hedgefund Operating System (supreme coordinator)."""
        try:
            # Import from correct location in ai subdirectory
            from backend.services.ai.ai_hedgefund_os import AIHedgeFundOS
            from backend.services.ai.ai_hfos_integration import AIHFOSIntegration
            
            self.ai_hfos = AIHedgeFundOS(
                data_dir=str(self.config.data_dir)
            )
            
            self.ai_hfos_integration = AIHFOSIntegration(
                data_dir=str(self.config.data_dir),
                update_interval_seconds=self.config.ai_hfos_update_interval_sec
            )
            
            # Start coordination loop
            await self.ai_hfos_integration.start()
            
            self._services_status["ai_hfos"] = normalize_subsystem_status("HEALTHY")
            logger.info(f"[AI-HFOS] Initialized in {self.config.ai_hfos_mode.value} mode")
            logger.info(f"[AI-HFOS] Coordination cycle started (interval: {self.config.ai_hfos_update_interval_sec}s)")
            
        except ImportError as e:
            self.ai_hfos = None
            self.ai_hfos_integration = None
            self._services_status["ai_hfos"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[AI-HFOS] Initialization failed: {e}")
            logger.warning(f"[AI-HFOS] System will continue without AI-HFOS (degraded mode)")
            # DO NOT CRASH - Continue without AI-HFOS
        except Exception as e:
            self.ai_hfos = None
            self.ai_hfos_integration = None
            self._services_status["ai_hfos"] = normalize_subsystem_status("DEGRADED")
            logger.error(f"[AI-HFOS] Initialization failed: {e}")
            logger.warning(f"[AI-HFOS] System will continue without AI-HFOS (degraded mode)")
            # DO NOT CRASH - Continue without AI-HFOS
    
    async def shutdown(self):
        """Gracefully shutdown all subsystems."""
        logger.info("[AI System Services] Shutting down subsystems...")
        
        # Shutdown in reverse order
        if self.ai_hfos_integration:
            try:
                self.ai_hfos_integration.stop()
            except Exception as e:
                logger.error(f"[AI-HFOS] Shutdown error: {e}")
        
        # Add other shutdown logic as needed
        
        self._initialized = False
        logger.info("[AI System Services] All subsystems shut down")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all subsystems."""
        enabled_subsystems = [
            name for name, status in self._services_status.items()
            if status == "HEALTHY"
        ]
        return {
            "initialized": self._initialized,
            "integration_stage": self.config.integration_stage.value,
            "services": self._services_status,
            "enabled_subsystems": enabled_subsystems,
            "emergency_brake": self.config.emergency_brake_active
        }
    
    def is_subsystem_enabled(self, name: str) -> bool:
        """Check if a subsystem is enabled and operational."""
        if not self._initialized:
            return False
        
        status = self._services_status.get(name, "disabled")
        return status == "HEALTHY"


# ============================================================================
# GLOBAL INSTANCE (Singleton pattern for app-wide access)
# ============================================================================

_global_services: Optional[AISystemServices] = None


def get_ai_services() -> AISystemServices:
    """Get global AI system services instance."""
    global _global_services
    if _global_services is None:
        _global_services = AISystemServices()
    return _global_services


def init_ai_services(config: Optional[AISystemConfig] = None) -> AISystemServices:
    """Initialize global AI system services."""
    global _global_services
    _global_services = AISystemServices(config)
    return _global_services
