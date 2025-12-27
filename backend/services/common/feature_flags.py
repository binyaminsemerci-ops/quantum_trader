"""
Feature Flag System for AI Modules
Centralized control for enabling/disabling AI modules at runtime
"""
import os
import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class ModuleMode(str, Enum):
    """Operating modes for AI modules"""
    DISABLED = "DISABLED"
    OBSERVE = "OBSERVE"      # Logging only, no actions
    ADVISORY = "ADVISORY"    # Recommendations only
    ENFORCE = "ENFORCE"      # Full enforcement
    AGGRESSIVE = "AGGRESSIVE"  # Maximum optimization


class FeatureFlags:
    """
    Feature flag manager for AI modules.
    Reads from environment variables and provides runtime checks.
    """
    
    def __init__(self):
        """Initialize feature flags from environment"""
        self._flags = self._load_flags()
        logger.info(f"Feature flags loaded: {sum(1 for v in self._flags.values() if v)} modules enabled")
    
    def _load_flags(self) -> Dict[str, bool]:
        """Load all feature flags from environment"""
        return {
            # Phase 1: Observation
            "universe_os": self._get_bool("ENABLE_UNIVERSE_OS", False),
            "pil": self._get_bool("ENABLE_PIL", False),
            "model_supervisor": self._get_bool("ENABLE_MODEL_SUPERVISOR", False),
            
            # Phase 2: Portfolio & Risk
            "pba": self._get_bool("ENABLE_PBA", False),
            "self_healing": self._get_bool("ENABLE_SELF_HEALING", False),
            "orchestrator_policy": self._get_bool("ENABLE_ORCHESTRATOR_POLICY", False),
            
            # Phase 3: Amplification
            "pal": self._get_bool("ENABLE_PAL", False),
            "trading_mathematician": self._get_bool("ENABLE_TRADING_MATHEMATICIAN", False),
            
            # Phase 4: Coordination
            "ai_hfos": self._get_bool("ENABLE_AI_HFOS", False),
            
            # Phase 5: Advanced
            "aelm": self._get_bool("ENABLE_AELM", False),
            "msc_ai": self._get_bool("ENABLE_MSC_AI", False),
            "opportunity_ranker": self._get_bool("ENABLE_OPPORTUNITY_RANKER", False),
            "ess": self._get_bool("ENABLE_ESS", False),
            "retraining_orchestrator": self._get_bool("ENABLE_RETRAINING_ORCHESTRATOR", False),
        }
    
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def is_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled"""
        return self._flags.get(module_name, False)
    
    def get_mode(self, module_name: str) -> ModuleMode:
        """Get operating mode for a module"""
        mode_key = f"{module_name.upper()}_MODE"
        mode_str = os.getenv(mode_key, "DISABLED").upper()
        
        try:
            return ModuleMode(mode_str)
        except ValueError:
            logger.warning(f"Invalid mode '{mode_str}' for {module_name}, defaulting to DISABLED")
            return ModuleMode.DISABLED
    
    def enable(self, module_name: str):
        """Enable a module at runtime"""
        if module_name in self._flags:
            self._flags[module_name] = True
            logger.info(f"✅ Module '{module_name}' enabled")
        else:
            logger.error(f"Unknown module: {module_name}")
    
    def disable(self, module_name: str):
        """Disable a module at runtime"""
        if module_name in self._flags:
            self._flags[module_name] = False
            logger.warning(f"⏸️ Module '{module_name}' disabled")
        else:
            logger.error(f"Unknown module: {module_name}")
    
    def get_enabled_modules(self) -> list:
        """Get list of enabled modules"""
        return [name for name, enabled in self._flags.items() if enabled]
    
    def get_status(self) -> Dict[str, dict]:
        """Get full status of all modules"""
        return {
            name: {
                "enabled": enabled,
                "mode": self.get_mode(name).value if enabled else "DISABLED"
            }
            for name, enabled in self._flags.items()
        }


# Global instance
_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Get global feature flags instance"""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags


def is_enabled(module_name: str) -> bool:
    """Quick check if module is enabled"""
    return get_feature_flags().is_enabled(module_name)


def get_mode(module_name: str) -> ModuleMode:
    """Quick get mode for module"""
    return get_feature_flags().get_mode(module_name)
