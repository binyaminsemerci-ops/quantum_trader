"""Federation Layer - Integration layer for AI Orchestration components.

This module provides a unified interface that combines outputs from
AI CEO, AI Risk Officer, and AI Strategy Officer into a cohesive
global decision snapshot.

Components:
- FederatedEngine: Main integration engine (v1)
- FederatedEngineV2: Advanced integration with Global Action Plan (v2)
- IntegrationLayer: API and event aggregation
"""

from backend.federation.federated_engine import FederatedEngine, GlobalState
from backend.federation.integration_layer import IntegrationLayer

# V2 imports (with fallback if not yet created)
try:
    from backend.federation.federated_engine_v2 import (
        FederatedEngineV2,
        FederationConfig,
        GlobalActionPlan,
        ActionPriority,
        ActionType,
        Action,
    )
    __all__ = [
        "FederatedEngine",
        "GlobalState",
        "IntegrationLayer",
        "FederatedEngineV2",
        "FederationConfig",
        "GlobalActionPlan",
        "ActionPriority",
        "ActionType",
        "Action",
    ]
except ImportError:
    __all__ = [
        "FederatedEngine",
        "GlobalState",
        "IntegrationLayer",
    ]
