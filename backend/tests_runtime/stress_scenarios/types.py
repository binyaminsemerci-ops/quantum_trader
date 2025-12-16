"""
Stress Scenario Types

EPIC-STRESS-001: Type definitions for stress testing framework.
"""

from dataclasses import dataclass, field
from typing import Literal, Callable, Awaitable

ScenarioName = str
ScenarioStatus = Literal["not_started", "running", "completed", "failed"]


@dataclass
class ScenarioResult:
    """
    Result of a stress scenario execution.
    
    Attributes:
        name: Scenario identifier
        success: Whether scenario passed validation checks
        reason: Human-readable explanation of outcome
        metrics: Key performance/behavior metrics collected during scenario
        duration_sec: Execution time in seconds
        
    Example:
        result = ScenarioResult(
            name="flash_crash",
            success=True,
            reason="all_checks_passed",
            metrics={
                "max_latency_ms": 123.0,
                "orders_blocked": 5,
                "ess_triggers": 1,
            },
            duration_sec=2.5,
        )
    """
    name: ScenarioName
    success: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    duration_sec: float = 0.0
    
    def __str__(self) -> str:
        status = "✅ PASS" if self.success else "❌ FAIL"
        return f"{status} {self.name}: {self.reason} (metrics={len(self.metrics)}, duration={self.duration_sec:.2f}s)"


# Type alias for scenario functions
ScenarioFn = Callable[[], Awaitable[ScenarioResult]]
