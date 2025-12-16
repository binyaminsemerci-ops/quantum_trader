"""World Model for Quantum Trader v5.

This package provides world modeling and scenario simulation:
- World Model: Probabilistic market state projections
- Scenario Simulator: Policy simulation and evaluation
"""

from backend.world_model.world_model import WorldModel, Scenario, MarketState
from backend.world_model.scenario_simulator import (
    ScenarioSimulator,
    SimulationResult,
    SimulationConfig,
)

__all__ = [
    "WorldModel",
    "Scenario",
    "MarketState",
    "ScenarioSimulator",
    "SimulationResult",
    "SimulationConfig",
]
