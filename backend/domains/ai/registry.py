"""
AI Module Registry
==================
Central registry of all AI modules in Quantum Trader v2.0

Tracks: name, type, path, enabled status, runtime usage
"""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel


class AIModuleKind(str, Enum):
    """AI module classification."""
    PREDICTOR = "predictor"           # Price/signal prediction
    RL_AGENT = "rl_agent"             # Reinforcement learning agent
    DETECTOR = "detector"             # Regime/drift/anomaly detection
    MANAGER = "manager"               # Lifecycle/state management
    ORCHESTRATOR = "orchestrator"     # High-level coordination
    ENSEMBLE = "ensemble"             # Model ensemble/fusion
    STRATEGY = "strategy"             # Strategy generation/selection
    MONITOR = "monitor"               # Model monitoring/supervision
    OPTIMIZER = "optimizer"           # Hyperparameter/portfolio optimization


class AIModuleInfo(BaseModel):
    """Metadata for a single AI module."""
    name: str
    kind: AIModuleKind
    path: str                         # Relative path from backend/
    enabled: bool = True
    in_runtime: bool = False          # Currently used in production
    has_tests: bool = False
    description: str = ""


# ============================================================================
# AI MODULE REGISTRY - Complete inventory (24 modules)
# ============================================================================

AI_MODULES: Dict[str, AIModuleInfo] = {
    # ---- RL Agents (5) ----
    "rl_v3_live_orchestrator": AIModuleInfo(
        name="rl_v3_live_orchestrator",
        kind=AIModuleKind.RL_AGENT,
        path="services/ai/rl_v3_live_orchestrator.py",
        enabled=False,  # Needs event_bus, policy_store
        in_runtime=True,
        has_tests=True,
        description="RL v3 live decision orchestrator for position sizing"
    ),
    "rl_position_sizing_agent": AIModuleInfo(
        name="rl_position_sizing_agent",
        kind=AIModuleKind.RL_AGENT,
        path="services/ai/rl_position_sizing_agent.py",
        enabled=False,  # Legacy module
        in_runtime=True,
        has_tests=False,
        description="Legacy RL position sizing agent"
    ),
    "rl_position_sizing_agent_v2": AIModuleInfo(
        name="rl_position_sizing_agent_v2",
        kind=AIModuleKind.RL_AGENT,
        path="agents/rl_position_sizing_agent_v2.py",
        enabled=False,  # Requires constructor args
        in_runtime=False,
        has_tests=False,
        description="RL position sizing agent v2 with PPO"
    ),
    "rl_meta_strategy_agent_v2": AIModuleInfo(
        name="rl_meta_strategy_agent_v2",
        kind=AIModuleKind.RL_AGENT,
        path="agents/rl_meta_strategy_agent_v2.py",
        enabled=False,  # Requires constructor args
        in_runtime=False,
        has_tests=False,
        description="RL meta-strategy selection agent"
    ),
    "rl_v3_training_daemon": AIModuleInfo(
        name="rl_v3_training_daemon",
        kind=AIModuleKind.RL_AGENT,
        path="services/ai/rl_v3_training_daemon.py",
        enabled=False,  # Background service, not testable
        in_runtime=True,
        has_tests=False,
        description="Background training daemon for RL v3"
    ),
    
    # ---- Detectors (4) ----
    "regime_detector": AIModuleInfo(
        name="regime_detector",
        kind=AIModuleKind.DETECTOR,
        path="services/ai/regime_detector.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Market regime detection (trend/range/volatility)"
    ),
    "drift_detection_manager": AIModuleInfo(
        name="drift_detection_manager",
        kind=AIModuleKind.DETECTOR,
        path="services/ai/drift_detection_manager.py",
        enabled=False,  # detect_drift() requires 4 specific args
        in_runtime=True,
        has_tests=False,
        description="Concept drift detection for models"
    ),
    "covariate_shift_handler": AIModuleInfo(
        name="covariate_shift_handler",
        kind=AIModuleKind.DETECTOR,
        path="services/ai/covariate_shift_handler.py",
        enabled=True,
        in_runtime=True,
        has_tests=True,
        description="Covariate shift detection and handling"
    ),
    "covariate_shift_manager": AIModuleInfo(
        name="covariate_shift_manager",
        kind=AIModuleKind.DETECTOR,
        path="services/ai/covariate_shift_manager.py",
        enabled=True,
        in_runtime=False,
        has_tests=False,
        description="Alternative covariate shift manager"
    ),
    
    # ---- Managers (6) ----
    "continuous_learning_manager": AIModuleInfo(
        name="continuous_learning_manager",
        kind=AIModuleKind.MANAGER,
        path="services/ai/continuous_learning_manager.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Continuous model retraining and lifecycle"
    ),
    "shadow_model_manager": AIModuleInfo(
        name="shadow_model_manager",
        kind=AIModuleKind.MANAGER,
        path="services/ai/shadow_model_manager.py",
        enabled=True,
        in_runtime=True,
        has_tests=True,
        description="Shadow model testing and A/B comparison"
    ),
    "memory_state_manager": AIModuleInfo(
        name="memory_state_manager",
        kind=AIModuleKind.MANAGER,
        path="services/ai/memory_state_manager.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Memory and state management for world model"
    ),
    "rl_state_manager_v2": AIModuleInfo(
        name="rl_state_manager_v2",
        kind=AIModuleKind.MANAGER,
        path="services/ai/rl_state_manager_v2.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="RL state construction and management v2"
    ),
    "reinforcement_signal_manager": AIModuleInfo(
        name="reinforcement_signal_manager",
        kind=AIModuleKind.MANAGER,
        path="services/ai/reinforcement_signal_manager.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Reinforcement learning signal aggregation"
    ),
    "model_supervisor": AIModuleInfo(
        name="model_supervisor",
        kind=AIModuleKind.MONITOR,
        path="services/ai/model_supervisor.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Model performance monitoring and supervision"
    ),
    
    # ---- Orchestrators (2) ----
    "ai_hedgefund_os": AIModuleInfo(
        name="ai_hedgefund_os",
        kind=AIModuleKind.ORCHESTRATOR,
        path="services/ai/ai_hedgefund_os.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Top-level AI HedgeFund OS orchestrator"
    ),
    "ai_hfos_integration": AIModuleInfo(
        name="ai_hfos_integration",
        kind=AIModuleKind.ORCHESTRATOR,
        path="services/ai/ai_hfos_integration.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="AI HFOS system integration layer"
    ),
    
    # ---- Strategy & Selection (3) ----
    "meta_strategy_selector": AIModuleInfo(
        name="meta_strategy_selector",
        kind=AIModuleKind.STRATEGY,
        path="services/ai/meta_strategy_selector.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Meta-strategy selection based on regime"
    ),
    "strategy_profiles": AIModuleInfo(
        name="strategy_profiles",
        kind=AIModuleKind.STRATEGY,
        path="services/ai/strategy_profiles.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Strategy profile definitions and configs"
    ),
    "trading_mathematician": AIModuleInfo(
        name="trading_mathematician",
        kind=AIModuleKind.OPTIMIZER,
        path="services/ai/trading_mathematician.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Advanced trading math and optimization"
    ),
    
    # ---- Ensemble & Engine (2) ----
    "ai_trading_engine": AIModuleInfo(
        name="ai_trading_engine",
        kind=AIModuleKind.ENSEMBLE,
        path="services/ai/ai_trading_engine.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="Main AI trading engine with ensemble logic"
    ),
    "shadow_model_integration": AIModuleInfo(
        name="shadow_model_integration",
        kind=AIModuleKind.ENSEMBLE,
        path="services/ai/shadow_model_integration.py",
        enabled=False,  # Has encoding issues in source
        in_runtime=True,
        has_tests=False,
        description="Shadow model integration into ensemble"
    ),
    
    # ---- Supporting Modules (2) ----
    "rl_reward_engine_v2": AIModuleInfo(
        name="rl_reward_engine_v2",
        kind=AIModuleKind.MANAGER,
        path="services/ai/rl_reward_engine_v2.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="RL reward computation engine v2"
    ),
    "rl_episode_tracker_v2": AIModuleInfo(
        name="rl_episode_tracker_v2",
        kind=AIModuleKind.MANAGER,
        path="services/ai/rl_episode_tracker_v2.py",
        enabled=True,
        in_runtime=True,
        has_tests=False,
        description="RL episode tracking and logging"
    ),
}


# ============================================================================
# REGISTRY API
# ============================================================================

def get_module(name: str) -> Optional[AIModuleInfo]:
    """Get module info by name."""
    return AI_MODULES.get(name)


def get_modules_by_kind(kind: AIModuleKind) -> list[AIModuleInfo]:
    """Get all modules of a specific kind."""
    return [m for m in AI_MODULES.values() if m.kind == kind]


def get_runtime_modules() -> list[AIModuleInfo]:
    """Get all modules currently in runtime."""
    return [m for m in AI_MODULES.values() if m.in_runtime]


def get_enabled_modules() -> list[AIModuleInfo]:
    """Get all enabled modules."""
    return [m for m in AI_MODULES.values() if m.enabled]


def module_count() -> int:
    """Total number of registered modules."""
    return len(AI_MODULES)


def runtime_coverage() -> float:
    """Percentage of modules in runtime."""
    total = len(AI_MODULES)
    if total == 0:
        return 0.0
    runtime = sum(1 for m in AI_MODULES.values() if m.in_runtime)
    return (runtime / total) * 100.0


def test_coverage() -> float:
    """Percentage of modules with tests."""
    total = len(AI_MODULES)
    if total == 0:
        return 0.0
    tested = sum(1 for m in AI_MODULES.values() if m.has_tests)
    return (tested / total) * 100.0
