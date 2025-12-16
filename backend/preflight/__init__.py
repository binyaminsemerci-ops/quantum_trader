"""
Pre-Flight Check Module

EPIC-PREFLIGHT-001: Pre-flight validation system for GO-LIVE readiness.

This module provides automated checks to verify system health before
enabling real trading.
"""

from .types import PreflightResult
from .checks import (
    register_check,
    run_all_preflight_checks,
    CHECKS,
)

__all__ = [
    "PreflightResult",
    "register_check",
    "run_all_preflight_checks",
    "CHECKS",
]
