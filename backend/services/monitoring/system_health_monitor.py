"""
System Health Monitor (SHM)

Continuously evaluates operational health of all Quantum Trader subsystems,
detects failures/anomalies, computes global health status, and writes results
to PolicyStore for system-wide coordination and alerting.

Core Responsibilities:
- Run health checks for all critical modules
- Aggregate module statuses into system-wide health
- Detect stale/dying modules via heartbeat timestamps
- Write health summary to PolicyStore
- Enable self-healing and emergency shutdown capabilities

Author: Quantum Trader AI Team
Date: 2025-11-30
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol, Literal, Any
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class HealthStatus(str, Enum):
    """Health status levels for individual modules and system-wide state."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class HealthCheckResult:
    """
    Result from a single module health check.
    
    Attributes:
        module: Name of the module (e.g., "market_data", "policy_store")
        status: Current health status
        details: Additional diagnostic information (latency, error count, etc.)
        timestamp: When this check was performed
        message: Optional human-readable status message
    """
    module: str
    status: HealthStatus
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "module": self.module,
            "status": self.status.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
        }


@dataclass
class SystemHealthSummary:
    """
    Aggregated system-wide health summary.
    
    Attributes:
        status: Overall system health (HEALTHY/WARNING/CRITICAL)
        failed_modules: List of modules in CRITICAL state
        warning_modules: List of modules in WARNING state
        healthy_modules: List of modules in HEALTHY state
        total_checks: Total number of health checks run
        timestamp: When this summary was computed
        details: Additional system-level metrics
    """
    status: HealthStatus
    failed_modules: list[str] = field(default_factory=list)
    warning_modules: list[str] = field(default_factory=list)
    healthy_modules: list[str] = field(default_factory=list)
    total_checks: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for PolicyStore serialization."""
        return {
            "status": self.status.value,
            "failed_modules": self.failed_modules,
            "warning_modules": self.warning_modules,
            "healthy_modules": self.healthy_modules,
            "total_checks": self.total_checks,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }

    def is_healthy(self) -> bool:
        """Check if system is fully healthy."""
        return self.status == HealthStatus.HEALTHY

    def is_degraded(self) -> bool:
        """Check if system is degraded (warnings but no critical failures)."""
        return self.status == HealthStatus.WARNING

    def is_critical(self) -> bool:
        """Check if system has critical failures."""
        return self.status == HealthStatus.CRITICAL


# ============================================================================
# Protocol Interface
# ============================================================================

class HealthMonitor(Protocol):
    """
    Protocol for module-specific health monitors.
    
    Each subsystem (market data, execution, AI models, etc.) should implement
    this interface to provide health check capabilities.
    """
    
    def run_check(self) -> HealthCheckResult:
        """
        Execute health check for this module.
        
        Returns:
            HealthCheckResult with current status and diagnostics
        """
        ...


# ============================================================================
# System Health Monitor (Main Aggregator)
# ============================================================================

class SystemHealthMonitor:
    """
    Central health monitoring service that orchestrates all subsystem checks,
    aggregates results, determines global health status, and writes to PolicyStore.
    
    Usage:
        monitors = [DataFeedMonitor(), PolicyStoreMonitor(), ...]
        shm = SystemHealthMonitor(monitors, policy_store)
        summary = shm.run()  # Run every 30-60 seconds
    """
    
    def __init__(
        self,
        monitors: list[HealthMonitor],
        policy_store: Any,  # PolicyStore protocol
        *,
        critical_threshold: int = 1,  # >= N critical → system CRITICAL
        warning_threshold: int = 3,   # >= N warnings → system WARNING
        enable_auto_write: bool = True,
    ):
        """
        Initialize System Health Monitor.
        
        Args:
            monitors: List of module-specific health monitors
            policy_store: PolicyStore instance for writing health state
            critical_threshold: Number of critical failures to mark system CRITICAL
            warning_threshold: Number of warnings to mark system WARNING
            enable_auto_write: Whether to auto-write to PolicyStore on each run
        """
        self.monitors = monitors
        self.policy_store = policy_store
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
        self.enable_auto_write = enable_auto_write
        
        self._last_summary: SystemHealthSummary | None = None
        self._check_history: list[SystemHealthSummary] = []
        
        logger.info(
            f"[SHM] Initialized with {len(monitors)} monitors "
            f"(critical_threshold={critical_threshold}, "
            f"warning_threshold={warning_threshold})"
        )

    def run(self) -> SystemHealthSummary:
        """
        Execute all health checks, aggregate results, and write to PolicyStore.
        
        Returns:
            SystemHealthSummary with aggregated health status
        """
        logger.debug(f"[SHM] Running {len(self.monitors)} health checks...")
        
        # Run all health checks
        results = self._run_all_checks()
        
        # Aggregate into system-wide summary
        summary = self._aggregate_results(results)
        
        # Write to PolicyStore if enabled
        if self.enable_auto_write:
            self._write_to_policy_store(summary)
        
        # Store for history
        self._last_summary = summary
        self._check_history.append(summary)
        
        # Keep only last 100 summaries
        if len(self._check_history) > 100:
            self._check_history = self._check_history[-100:]
        
        # Log summary
        self._log_summary(summary)
        
        return summary

    def _run_all_checks(self) -> list[HealthCheckResult]:
        """
        Execute all registered health monitors.
        
        Returns:
            List of health check results
        """
        results = []
        
        for monitor in self.monitors:
            try:
                result = monitor.run_check()
                results.append(result)
            except Exception as e:
                logger.error(f"[SHM] Health check failed for monitor {monitor}: {e}")
                # Create a CRITICAL result for failed monitor
                results.append(HealthCheckResult(
                    module=getattr(monitor, '__class__', 'unknown').__name__,
                    status=HealthStatus.CRITICAL,
                    details={"error": str(e)},
                    message=f"Health check crashed: {e}",
                ))
        
        return results

    def _aggregate_results(
        self,
        results: list[HealthCheckResult]
    ) -> SystemHealthSummary:
        """
        Aggregate individual module results into system-wide health summary.
        
        Args:
            results: List of individual health check results
            
        Returns:
            SystemHealthSummary with overall status
        """
        # Categorize modules by status
        failed_modules = []
        warning_modules = []
        healthy_modules = []
        
        for result in results:
            if result.status == HealthStatus.CRITICAL:
                failed_modules.append(result.module)
            elif result.status == HealthStatus.WARNING:
                warning_modules.append(result.module)
            elif result.status == HealthStatus.HEALTHY:
                healthy_modules.append(result.module)
        
        # Determine global system status
        num_critical = len(failed_modules)
        num_warnings = len(warning_modules)
        
        if num_critical >= self.critical_threshold:
            global_status = HealthStatus.CRITICAL
        elif num_warnings >= self.warning_threshold:
            global_status = HealthStatus.WARNING
        else:
            global_status = HealthStatus.HEALTHY
        
        # Build summary
        summary = SystemHealthSummary(
            status=global_status,
            failed_modules=failed_modules,
            warning_modules=warning_modules,
            healthy_modules=healthy_modules,
            total_checks=len(results),
            timestamp=datetime.utcnow(),
            details={
                "critical_count": num_critical,
                "warning_count": num_warnings,
                "healthy_count": len(healthy_modules),
                "check_results": [r.to_dict() for r in results],
            }
        )
        
        return summary

    def _write_to_policy_store(self, summary: SystemHealthSummary) -> None:
        """
        Write health summary to PolicyStore for system-wide coordination.
        
        Args:
            summary: SystemHealthSummary to write
        """
        try:
            self.policy_store.patch({
                "system_health": summary.to_dict()
            })
            logger.debug("[SHM] ✅ Wrote health summary to PolicyStore")
        except Exception as e:
            logger.error(f"[SHM] ❌ Failed to write to PolicyStore: {e}")

    def _log_summary(self, summary: SystemHealthSummary) -> None:
        """
        Log human-readable summary of system health.
        
        Args:
            summary: SystemHealthSummary to log
        """
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🚨",
        }
        
        emoji = status_emoji.get(summary.status, "❓")
        
        logger.info(
            f"[SHM] {emoji} System Health: {summary.status.value} "
            f"({summary.total_checks} checks: "
            f"{len(summary.healthy_modules)} healthy, "
            f"{len(summary.warning_modules)} warnings, "
            f"{len(summary.failed_modules)} critical)"
        )
        
        if summary.failed_modules:
            logger.warning(f"[SHM] 🚨 Critical modules: {', '.join(summary.failed_modules)}")
        
        if summary.warning_modules:
            logger.warning(f"[SHM] ⚠️ Warning modules: {', '.join(summary.warning_modules)}")

    def get_last_summary(self) -> SystemHealthSummary | None:
        """Get the most recent health summary."""
        return self._last_summary

    def get_history(self, limit: int = 10) -> list[SystemHealthSummary]:
        """Get recent health check history."""
        return self._check_history[-limit:]

    def get_module_status(self, module_name: str) -> HealthStatus | None:
        """
        Get current status of a specific module.
        
        Args:
            module_name: Name of module to query
            
        Returns:
            HealthStatus or None if module not found
        """
        if not self._last_summary:
            return None
        
        if module_name in self._last_summary.failed_modules:
            return HealthStatus.CRITICAL
        elif module_name in self._last_summary.warning_modules:
            return HealthStatus.WARNING
        elif module_name in self._last_summary.healthy_modules:
            return HealthStatus.HEALTHY
        
        return None


# ============================================================================
# Base Health Monitor Implementation
# ============================================================================

class BaseHealthMonitor:
    """
    Base class for health monitors with common functionality.
    
    Subclasses should override _perform_check() to implement specific logic.
    """
    
    def __init__(
        self,
        module_name: str,
        *,
        heartbeat_timeout: timedelta = timedelta(minutes=5),
    ):
        """
        Initialize base health monitor.
        
        Args:
            module_name: Name of the module being monitored
            heartbeat_timeout: Max time since last heartbeat before marking stale
        """
        self.module_name = module_name
        self.heartbeat_timeout = heartbeat_timeout
        self._last_check_time: datetime | None = None

    def run_check(self) -> HealthCheckResult:
        """Execute health check (template method pattern)."""
        self._last_check_time = datetime.utcnow()
        return self._perform_check()

    def _perform_check(self) -> HealthCheckResult:
        """
        Override this method to implement specific health check logic.
        
        Returns:
            HealthCheckResult with status and details
        """
        raise NotImplementedError("Subclasses must implement _perform_check()")

    def _check_heartbeat(
        self,
        last_heartbeat: datetime | None,
        critical_timeout: timedelta | None = None
    ) -> tuple[HealthStatus, str]:
        """
        Check if heartbeat is fresh.
        
        Args:
            last_heartbeat: Last heartbeat timestamp
            critical_timeout: Optional override for critical timeout
            
        Returns:
            Tuple of (status, message)
        """
        if last_heartbeat is None:
            return HealthStatus.CRITICAL, "No heartbeat recorded"
        
        age = datetime.utcnow() - last_heartbeat
        timeout = critical_timeout or self.heartbeat_timeout
        
        if age > timeout:
            return HealthStatus.CRITICAL, f"Heartbeat stale ({age.total_seconds():.0f}s)"
        elif age > timeout / 2:
            return HealthStatus.WARNING, f"Heartbeat aging ({age.total_seconds():.0f}s)"
        else:
            return HealthStatus.HEALTHY, f"Heartbeat fresh ({age.total_seconds():.0f}s)"


# ============================================================================
# Example Health Monitors (for testing/reference)
# ============================================================================

class FakeMarketDataHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for market data feed.
    
    Checks:
    - Feed latency
    - Missing candles
    - API connection status
    """
    
    def __init__(self, latency_ms: float = 15.0, missing_candles: int = 0):
        super().__init__("market_data")
        self.latency_ms = latency_ms
        self.missing_candles = missing_candles

    def _perform_check(self) -> HealthCheckResult:
        """Simulate market data health check."""
        # Simulate health logic
        if self.latency_ms > 500:
            status = HealthStatus.CRITICAL
            message = "Feed latency too high"
        elif self.latency_ms > 200:
            status = HealthStatus.WARNING
            message = "Feed latency elevated"
        elif self.missing_candles > 10:
            status = HealthStatus.CRITICAL
            message = "Too many missing candles"
        elif self.missing_candles > 0:
            status = HealthStatus.WARNING
            message = "Some candles missing"
        else:
            status = HealthStatus.HEALTHY
            message = "Feed operating normally"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "latency_ms": self.latency_ms,
                "missing_candles": self.missing_candles,
            },
            message=message,
        )


class FakePolicyStoreHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for PolicyStore.
    
    Checks:
    - Last update timestamp
    - Data integrity
    - Connection status
    """
    
    def __init__(self, last_update_age_seconds: float = 30.0):
        super().__init__("policy_store")
        self.last_update_age_seconds = last_update_age_seconds

    def _perform_check(self) -> HealthCheckResult:
        """Simulate PolicyStore health check."""
        # Simulate health logic
        if self.last_update_age_seconds > 600:  # 10 minutes
            status = HealthStatus.CRITICAL
            message = "PolicyStore stale (no updates)"
        elif self.last_update_age_seconds > 300:  # 5 minutes
            status = HealthStatus.WARNING
            message = "PolicyStore aging"
        else:
            status = HealthStatus.HEALTHY
            message = "PolicyStore up-to-date"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "last_update_age_seconds": self.last_update_age_seconds,
            },
            message=message,
        )


class FakeExecutionHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for execution layer.
    
    Checks:
    - Order execution latency
    - Failed orders rate
    - Position sync status
    """
    
    def __init__(
        self,
        execution_latency_ms: float = 150.0,
        failed_orders_rate: float = 0.01
    ):
        super().__init__("execution")
        self.execution_latency_ms = execution_latency_ms
        self.failed_orders_rate = failed_orders_rate

    def _perform_check(self) -> HealthCheckResult:
        """Simulate execution health check."""
        # Simulate health logic
        if self.failed_orders_rate > 0.10:  # 10% failure rate
            status = HealthStatus.CRITICAL
            message = "High order failure rate"
        elif self.failed_orders_rate > 0.05:  # 5% failure rate
            status = HealthStatus.WARNING
            message = "Elevated order failures"
        elif self.execution_latency_ms > 1000:
            status = HealthStatus.CRITICAL
            message = "Execution latency critical"
        elif self.execution_latency_ms > 500:
            status = HealthStatus.WARNING
            message = "Execution latency elevated"
        else:
            status = HealthStatus.HEALTHY
            message = "Execution operating normally"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "execution_latency_ms": self.execution_latency_ms,
                "failed_orders_rate": self.failed_orders_rate,
            },
            message=message,
        )


class FakeStrategyRuntimeHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for strategy runtime engine.
    
    Checks:
    - Active strategies count
    - Strategy execution rate
    - Signal generation health
    """
    
    def __init__(
        self,
        active_strategies: int = 5,
        signals_per_hour: float = 12.0
    ):
        super().__init__("strategy_runtime")
        self.active_strategies = active_strategies
        self.signals_per_hour = signals_per_hour

    def _perform_check(self) -> HealthCheckResult:
        """Simulate strategy runtime health check."""
        # Simulate health logic
        if self.active_strategies == 0:
            status = HealthStatus.CRITICAL
            message = "No active strategies"
        elif self.active_strategies < 3:
            status = HealthStatus.WARNING
            message = "Low strategy count"
        elif self.signals_per_hour < 1:
            status = HealthStatus.WARNING
            message = "Low signal generation rate"
        else:
            status = HealthStatus.HEALTHY
            message = "Strategies operating normally"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "active_strategies": self.active_strategies,
                "signals_per_hour": self.signals_per_hour,
            },
            message=message,
        )


class FakeMSCHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for Meta Strategy Controller (MSC AI).
    
    Checks:
    - Last policy update age
    - Decision-making health
    - Risk mode appropriateness
    """
    
    def __init__(self, last_update_minutes_ago: float = 25.0):
        super().__init__("msc_ai")
        self.last_update_minutes_ago = last_update_minutes_ago

    def _perform_check(self) -> HealthCheckResult:
        """Simulate MSC AI health check."""
        # Simulate health logic
        if self.last_update_minutes_ago > 60:  # 1 hour
            status = HealthStatus.CRITICAL
            message = "MSC AI not updating policy"
        elif self.last_update_minutes_ago > 40:  # 40 minutes
            status = HealthStatus.WARNING
            message = "MSC AI updates delayed"
        else:
            status = HealthStatus.HEALTHY
            message = "MSC AI operating normally"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "last_update_minutes_ago": self.last_update_minutes_ago,
            },
            message=message,
        )


class FakeCLMHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for Continuous Learning Manager.
    
    Checks:
    - Last retraining timestamp
    - Model drift metrics
    - Training job status
    """
    
    def __init__(self, last_retrain_hours_ago: float = 48.0):
        super().__init__("continuous_learning")
        self.last_retrain_hours_ago = last_retrain_hours_ago

    def _perform_check(self) -> HealthCheckResult:
        """Simulate CLM health check."""
        # Simulate health logic
        if self.last_retrain_hours_ago > 168:  # 7 days
            status = HealthStatus.CRITICAL
            message = "Models not retrained in over a week"
        elif self.last_retrain_hours_ago > 72:  # 3 days
            status = HealthStatus.WARNING
            message = "Models overdue for retraining"
        else:
            status = HealthStatus.HEALTHY
            message = "CLM operating normally"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "last_retrain_hours_ago": self.last_retrain_hours_ago,
            },
            message=message,
        )


class FakeOpportunityRankerHealthMonitor(BaseHealthMonitor):
    """
    Example health monitor for Opportunity Ranker.
    
    Checks:
    - Ranking freshness
    - Symbol coverage
    - Score distribution health
    """
    
    def __init__(self, ranking_age_minutes: float = 4.0, symbols_ranked: int = 15):
        super().__init__("opportunity_ranker")
        self.ranking_age_minutes = ranking_age_minutes
        self.symbols_ranked = symbols_ranked

    def _perform_check(self) -> HealthCheckResult:
        """Simulate OpportunityRanker health check."""
        # Simulate health logic
        if self.ranking_age_minutes > 15:
            status = HealthStatus.CRITICAL
            message = "Rankings stale"
        elif self.ranking_age_minutes > 8:
            status = HealthStatus.WARNING
            message = "Rankings aging"
        elif self.symbols_ranked < 5:
            status = HealthStatus.WARNING
            message = "Low symbol coverage"
        else:
            status = HealthStatus.HEALTHY
            message = "Ranker operating normally"
        
        return HealthCheckResult(
            module=self.module_name,
            status=status,
            details={
                "ranking_age_minutes": self.ranking_age_minutes,
                "symbols_ranked": self.symbols_ranked,
            },
            message=message,
        )
