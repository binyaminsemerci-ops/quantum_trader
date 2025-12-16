"""AI Module Domain - Quantum Trader v2.0"""

from backend.domains.ai.registry import (
    AIModuleInfo,
    AIModuleKind,
    AI_MODULES,
    get_module,
    get_modules_by_kind,
    get_runtime_modules,
    get_enabled_modules,
    module_count,
    runtime_coverage,
    test_coverage
)

from backend.domains.ai.interface import (
    AIInput,
    AIOutput,
    run_predictor,
    run_rl_agent,
    run_detector,
    run_generic,
    normalize_features,
    clamp_confidence
)

from backend.domains.ai.health import (
    AIModuleHealth,
    check_module_health,
    run_full_ai_healthcheck,
    run_full_ai_healthcheck_sync,
    calculate_health_score
)

__all__ = [
    # Registry
    "AIModuleInfo",
    "AIModuleKind",
    "AI_MODULES",
    "get_module",
    "get_modules_by_kind",
    "get_runtime_modules",
    "get_enabled_modules",
    "module_count",
    "runtime_coverage",
    "test_coverage",
    # Interface
    "AIInput",
    "AIOutput",
    "run_predictor",
    "run_rl_agent",
    "run_detector",
    "run_generic",
    "normalize_features",
    "clamp_confidence",
    # Health
    "AIModuleHealth",
    "check_module_health",
    "run_full_ai_healthcheck",
    "run_full_ai_healthcheck_sync",
    "calculate_health_score",
]
