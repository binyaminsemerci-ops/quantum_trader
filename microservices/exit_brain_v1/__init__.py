"""
Exit Brain v1 — Phase 1: Foundation Layer

Shadow-only module that builds enriched position state, computes exit geometry,
and detects regime drift. Produces features and scores — never executes trades.

Architecture:
    Redis state → position_state_builder → PositionExitState
    PositionExitState → geometry_engine → MFE/MAE/drawdown/momentum scores
    PositionExitState → regime_drift_engine → trend alignment/reversal/chop scores
    All outputs → shadow_publisher → quantum:stream:exit.*.shadow

Version: 1.0.0-shadow
"""

__version__ = "1.0.0-shadow"
