"""
Exit Brain v3 - Unified orchestrator for all exit logic.
"""

from .models import (
    ExitContext,
    ExitLeg,
    ExitPlan,
    ExitKind
)
from .planner import ExitBrainV3
from .integration import (
    to_dynamic_tpsl,
    to_trailing_config,
    to_partial_exit_config
)
from .tp_profiles_v3 import (
    TPProfile,
    TPProfileLeg,
    TrailingProfile,
    TPProfileMapping,
    MarketRegime,
    TPKind,
    get_tp_and_trailing_profile,
    register_custom_profile,
    get_profile_by_name,
    list_available_profiles
)

__all__ = [
    "ExitContext",
    "ExitLeg",
    "ExitPlan", 
    "ExitKind",
    "ExitBrainV3",
    "to_dynamic_tpsl",
    "to_trailing_config",
    "to_partial_exit_config",
    "TPProfile",
    "TPProfileLeg",
    "TrailingProfile",
    "TPProfileMapping",
    "MarketRegime",
    "TPKind",
    "get_tp_and_trailing_profile",
    "register_custom_profile",
    "get_profile_by_name",
    "list_available_profiles"
]
