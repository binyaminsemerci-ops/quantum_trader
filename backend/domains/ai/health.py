"""
AI Module Health Checks
========================
Health check infrastructure for all AI modules

Provides: per-module health check, full system health scan
"""

from __future__ import annotations
from typing import Optional, Any, Dict
from pydantic import BaseModel
import logging
import asyncio
import numpy as np
from datetime import datetime, timezone

from backend.domains.ai.registry import AI_MODULES, AIModuleInfo, get_enabled_modules
from backend.domains.ai.interface import AIInput, AIOutput, run_predictor, run_rl_agent, run_detector, run_generic

logger = logging.getLogger(__name__)


class AIModuleHealth(BaseModel):
    """Health check result for a single AI module."""
    name: str
    ok: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc)
        super().__init__(**data)


# ============================================================================
# MODULE LOADER
# ============================================================================

def load_module_instance(info: AIModuleInfo) -> tuple[Any, Optional[str]]:
    """
    Dynamically load module instance.
    Returns: (module_instance, error_message)
    """
    try:
        # Skip modules with known issues
        if 'shadow_model_integration' in info.path:
            return None, "Skipped: encoding issues"
        
        # Convert path to import path
        import_path = info.path.replace('/', '.').replace('.py', '')
        module_path = f"backend.{import_path}"
        
        # Dynamic import
        parts = module_path.split('.')
        module_name = parts[-1]
        
        module = __import__(module_path, fromlist=[module_name])
        
        # Pattern 1: Singleton getter (get_X functions)
        getter_names = [
            f"get_{module_name}",
            f"get_{module_name}_singleton",
            "get_instance",
            "instance",  # class method
        ]
        for getter_name in getter_names:
            if hasattr(module, getter_name):
                try:
                    attr = getattr(module, getter_name)
                    # Could be function or classmethod
                    if callable(attr):
                        instance = attr()
                    else:
                        # Could be a property
                        instance = attr
                    if instance is not None:
                        return instance, None
                except Exception as e:
                    logger.debug(f"Getter {getter_name} failed: {e}")
        
        # Pattern 2: Main class - find actual class (not Enum, not typing._AnyMeta)
        from enum import Enum as EnumBase
        import typing
        
        candidate_classes = []
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue
            
            attr = getattr(module, attr_name)
            
            # Skip non-classes
            if not isinstance(attr, type):
                continue
            
            # Skip Enums
            if issubclass(attr, EnumBase):
                continue
            
            # Skip typing constructs
            if hasattr(typing, '_AnyMeta') and isinstance(attr, typing._AnyMeta):
                continue
            
            # Skip BaseModel/Pydantic models (not the main class)
            try:
                from pydantic import BaseModel
                if issubclass(attr, BaseModel):
                    continue
            except:
                pass
            
            # Match name pattern (CamelCase version of module name)
            expected_names = [
                ''.join(word.capitalize() for word in module_name.split('_')),
                module_name.upper(),
            ]
            if attr_name in expected_names or module_name.lower() in attr_name.lower():
                candidate_classes.append((attr_name, attr))
        
        # Try to instantiate candidates
        for class_name, cls in candidate_classes:
            try:
                instance = cls()
                return instance, None
            except TypeError:
                # Constructor needs args - return class for manual testing
                logger.debug(f"{class_name} needs constructor args")
                return cls, None
            except Exception as e:
                logger.debug(f"Failed to instantiate {class_name}: {e}")
        
        # Pattern 3: Return module itself (for function-based modules)
        return module, None
        
    except Exception as e:
        return None, f"Load failed: {str(e)}"


# ============================================================================
# HEALTH CHECK EXECUTOR
# ============================================================================

async def check_module_health(name: str) -> AIModuleHealth:
    """
    Health check for a single AI module.
    
    Process:
    1. Load module from registry
    2. Create synthetic AIInput
    3. Run through appropriate adapter
    4. Validate AIOutput
    """
    info = AI_MODULES.get(name)
    if not info:
        return AIModuleHealth(
            name=name,
            ok=False,
            error="Module not in registry"
        )
    
    if not info.enabled:
        return AIModuleHealth(
            name=name,
            ok=False,
            error="Module disabled"
        )
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Load module
        module, load_error = load_module_instance(info)
        if load_error:
            return AIModuleHealth(
                name=name,
                ok=False,
                error=load_error
            )
        
        # Create synthetic input
        synthetic_input = create_synthetic_input(info)
        
        # Run through adapter based on module kind
        output = run_module_adapter(module, synthetic_input, info)
        
        # Validate output
        if not output.success:
            return AIModuleHealth(
                name=name,
                ok=False,
                error=output.error or "Unknown error"
            )
        
        # Validate confidence range
        if output.confidence is not None and not (0.0 <= output.confidence <= 1.0):
            return AIModuleHealth(
                name=name,
                ok=False,
                error=f"Invalid confidence: {output.confidence}"
            )
        
        end_time = asyncio.get_event_loop().time()
        latency_ms = (end_time - start_time) * 1000
        
        return AIModuleHealth(
            name=name,
            ok=True,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Health check failed for {name}: {e}", exc_info=True)
        return AIModuleHealth(
            name=name,
            ok=False,
            error=f"Health check exception: {str(e)}"
        )


def create_synthetic_input(info: AIModuleInfo) -> AIInput:
    """Create synthetic input for module testing."""
    
    # Default: simple feature vector
    if info.kind in ["predictor", "rl_agent"]:
        features = np.random.randn(10).astype(np.float32)
    elif info.kind == "detector":
        # Detectors might need context
        features = np.random.randn(20).astype(np.float32)
    else:
        features = np.random.randn(5).astype(np.float32)
    
    return AIInput(
        features=features,
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        metadata={
            "test": True,
            "health_check": True
        }
    )


def run_module_adapter(module: Any, ai_input: AIInput, info: AIModuleInfo) -> AIOutput:
    """Run module through appropriate adapter."""
    
    from backend.domains.ai.registry import AIModuleKind
    
    try:
        if info.kind == AIModuleKind.PREDICTOR:
            return run_predictor(module, ai_input)
        elif info.kind == AIModuleKind.RL_AGENT:
            return run_rl_agent(module, ai_input)
        elif info.kind == AIModuleKind.DETECTOR:
            return run_detector(module, ai_input)
        else:
            # Generic adapter for managers/orchestrators/etc
            return run_generic(module, ai_input)
    except Exception as e:
        return AIOutput(
            success=False,
            error=f"Adapter error: {str(e)}"
        )


# ============================================================================
# FULL SYSTEM HEALTH CHECK
# ============================================================================

async def run_full_ai_healthcheck() -> list[AIModuleHealth]:
    """
    Run health checks on all enabled AI modules.
    Returns list of health results.
    """
    enabled = get_enabled_modules()
    logger.info(f"Running health checks on {len(enabled)} enabled modules")
    
    # Run all checks in parallel
    tasks = [check_module_health(m.name) for m in enabled]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    health_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            health_results.append(AIModuleHealth(
                name=enabled[i].name,
                ok=False,
                error=f"Exception: {str(result)}"
            ))
        else:
            health_results.append(result)
    
    return health_results


def calculate_health_score(results: list[AIModuleHealth]) -> Dict[str, Any]:
    """Calculate aggregate health metrics."""
    total = len(results)
    if total == 0:
        return {"score": 0.0, "pass": 0, "fail": 0, "pass_rate": 0.0}
    
    passed = sum(1 for r in results if r.ok)
    failed = total - passed
    pass_rate = (passed / total) * 100.0
    
    avg_latency = None
    latencies = [r.latency_ms for r in results if r.ok and r.latency_ms]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
    
    return {
        "score": pass_rate,
        "pass": passed,
        "fail": failed,
        "total": total,
        "pass_rate": pass_rate,
        "avg_latency_ms": avg_latency,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================================
# SYNC WRAPPER (for non-async contexts)
# ============================================================================

def run_full_ai_healthcheck_sync() -> list[AIModuleHealth]:
    """Synchronous wrapper for health check."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run_full_ai_healthcheck())
    finally:
        loop.close()
