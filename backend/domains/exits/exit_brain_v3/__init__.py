"""
Exit Brain v3 - Unified orchestrator for all exit logic.
"""

from backend.domains.exits.exit_brain_v3.models import (
    ExitContext,
    ExitLeg,
    ExitPlan,
    ExitKind
)
from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
from backend.domains.exits.exit_brain_v3.integration import (
    to_dynamic_tpsl,
    to_trailing_config,
    to_partial_exit_config
)
from backend.domains.exits.exit_brain_v3.tp_profiles_v3 import (
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
