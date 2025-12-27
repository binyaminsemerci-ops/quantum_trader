"""
Pre-Flight Check Types

EPIC-PREFLIGHT-001: Type definitions for pre-flight validation system.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PreflightResult:
    """
    Result of a single pre-flight check.
    
    Attributes:
        name: Check identifier (e.g., "check_health_endpoints")
        success: True if check passed, False if failed
        reason: Human-readable reason for success/failure
        details: Optional dict with additional context
    
    Example:
        PreflightResult(
            name="check_exchanges",
            success=True,
            reason="all_exchanges_healthy",
            details={"binance": "ok", "bybit": "ok"}
        )
    """
    name: str
    success: bool
    reason: str
    details: Optional[Dict[str, str]] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Format result for display."""
        status = "✅" if self.success else "❌"
        return f"{status} {self.name}: {self.reason}"
