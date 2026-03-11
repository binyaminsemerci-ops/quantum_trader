"""
Model Registry — Maps model names to agent classes and metadata.

This is the single configuration point for which models the exit brain
ensemble adapter uses, their horizon labels, feature versions, and
how to instantiate them.

Does NOT own agent instances — the adapter owns the lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

logger = logging.getLogger(__name__)

ALL_MODEL_NAMES = ["xgboost", "lightgbm", "nhits", "patchtst", "tft", "dlinear"]


@dataclass(frozen=True)
class ModelSpec:
    """Static specification for one ensemble model."""
    name: str               # Canonical name, e.g. "xgboost"
    agent_class_name: str   # Class name in unified_agents, e.g. "XGBoostAgent"
    agent_module: str       # Import path, e.g. "ai_engine.agents.unified_agents"
    horizon_label: str      # Default horizon for this model
    feature_version: str    # Feature set version the model was trained on
    model_type: str         # "tree" or "torch"


# ASSUMPTION: All models are trained on the same 49-feature set (feature_version="v5").
# ASSUMPTION: All models have horizon_label="short" (1-5 min forecasting window).
# Verify these against actual model training configs on VPS.

MODEL_SPECS: Dict[str, ModelSpec] = {
    "xgboost": ModelSpec(
        name="xgboost",
        agent_class_name="XGBoostAgent",
        agent_module="ai_engine.agents.unified_agents",
        horizon_label="short",
        feature_version="v5",
        model_type="tree",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        agent_class_name="LightGBMAgent",
        agent_module="ai_engine.agents.unified_agents",
        horizon_label="short",
        feature_version="v5",
        model_type="tree",
    ),
    "nhits": ModelSpec(
        name="nhits",
        agent_class_name="NHiTSAgent",
        agent_module="ai_engine.agents.unified_agents",
        horizon_label="short",
        feature_version="v5",
        model_type="torch",
    ),
    "patchtst": ModelSpec(
        name="patchtst",
        agent_class_name="PatchTSTAgent",
        agent_module="ai_engine.agents.unified_agents",
        horizon_label="short",
        feature_version="v5",
        model_type="torch",
    ),
    "tft": ModelSpec(
        name="tft",
        agent_class_name="TFTAgent",
        agent_module="ai_engine.agents.unified_agents",
        horizon_label="short",
        feature_version="v5",
        model_type="torch",
    ),
    "dlinear": ModelSpec(
        name="dlinear",
        agent_class_name="DLinearAgent",
        agent_module="ai_engine.agents.unified_agents",
        horizon_label="short",
        feature_version="v5",
        model_type="torch",
    ),
}


def get_model_spec(model_name: str) -> Optional[ModelSpec]:
    """Get spec for a model by canonical name."""
    return MODEL_SPECS.get(model_name)


def get_all_model_names() -> List[str]:
    """Return all registered model names."""
    return list(MODEL_SPECS.keys())


def get_agent_class(model_name: str) -> Optional[Type]:
    """
    Dynamically import and return the agent class for a model.

    Returns None if import fails (fail-closed).
    """
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        logger.warning("[ModelRegistry] Unknown model: %s", model_name)
        return None

    try:
        import importlib
        module = importlib.import_module(spec.agent_module)
        cls = getattr(module, spec.agent_class_name)
        return cls
    except (ImportError, AttributeError) as e:
        logger.error(
            "[ModelRegistry] Failed to import %s.%s: %s",
            spec.agent_module, spec.agent_class_name, e,
        )
        return None
