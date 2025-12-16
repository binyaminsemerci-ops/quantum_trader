"""
GO-LIVE Module

Provides activation/deactivation logic for enabling REAL TRADING.
"""

from backend.go_live.activation import (
    GO_LIVE_MARKER_FILE,
    go_live_activate,
    go_live_deactivate,
    is_go_live_active,
)

__all__ = [
    "go_live_activate",
    "go_live_deactivate",
    "is_go_live_active",
    "GO_LIVE_MARKER_FILE",
]
