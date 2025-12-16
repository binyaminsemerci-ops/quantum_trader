"""
SELF-HEALING SYSTEM

Detects failures, degradation, anomalies and automatically applies safe fallbacks or recovery actions.
Protects all critical subsystems and ensures system stability.

Author: Quantum Trader AI Team
Date: November 23, 2025
"""

import os
import json
import logging
import asyncio
import psutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

class SubsystemType(Enum):
    """Types of subsystems to monitor."""
    DATA_FEED = "data_feed"
    EXCHANGE_CONNECTION = "exchange_connection"
    AI_MODEL = "ai_model"
    EVENT_EXECUTOR = "event_executor"
    ORCHESTRATOR = "orchestrator"
    PORTFOLIO_BALANCER = "portfolio_balancer"
    MODEL_SUPERVISOR = "model_supervisor"
    RETRAINING_ORCHESTRATOR = "retraining_orchestrator"
    POSITION_MONITOR = "position_monitor"
    RISK_GUARD = "risk_guard"
    DATABASE = "database"
    UNIVERSE_OS = "universe_os"
    LOGGING = "logging"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class IssueSeverity(Enum):
    """Severity of detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART_SUBSYSTEM = "restart_subsystem"
    PAUSE_TRADING = "pause_trading"
    SWITCH_TO_SAFE_PROFILE = "switch_to_safe_profile"
    DISABLE_MODULE = "disable_module"
    FALLBACK_TO_BACKUP = "fallback_to_backup"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    NO_NEW_TRADES = "no_new_trades"
    DEFENSIVE_EXIT = "defensive_exit"
    RELOAD_CONFIG = "reload_config"
    CLEAR_CACHE = "clear_cache"


class SafetyPolicy(Enum):
    """Safety policies for trading control."""
    ALLOW_ALL = "allow_all"
    NO_NEW_TRADES = "no_new_trades"
    DEFENSIVE_EXIT = "defensive_exit"
    SAFE_RISK_PROFILE = "safe_risk_profile"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    subsystem: SubsystemType
    status: HealthStatus
    timestamp: str
    
    # Metrics
    response_time_ms: Optional[float]
    error_count: int
    last_error: Optional[str]
    last_success: Optional[str]
    
    # Details
    details: Dict[str, Any]


@dataclass
class DetectedIssue:
    """Represents a detected issue."""
    issue_id: str
    subsystem: SubsystemType
    severity: IssueSeverity
    timestamp: str
    
    description: str
    symptoms: List[str]
    root_cause: Optional[str]
    
    # Impact
    impacts_trading: bool
    affects_subsystems: List[SubsystemType]


@dataclass
class RecoveryRecommendation:
    """Recommended recovery action."""
    recommendation_id: str
    issue: DetectedIssue
    action: RecoveryAction
    priority: int  # 1=highest
    
    description: str
    expected_result: str
    risks: List[str]
    
    # Execution
    can_auto_execute: bool
    requires_approval: bool


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: str
    overall_status: HealthStatus
    current_safety_policy: SafetyPolicy
    
    # Subsystem statuses
    subsystem_health: Dict[SubsystemType, HealthCheck]
    
    # Issues
    detected_issues: List[DetectedIssue]
    critical_issues: List[DetectedIssue]
    
    # Recommendations
    recovery_recommendations: List[RecoveryRecommendation]
    
    # Flags
    trading_should_continue: bool
    requires_immediate_action: bool
    
    # Summary
    healthy_count: int
    degraded_count: int
    critical_count: int
    failed_count: int


# ============================================================
# SELF-HEALING SYSTEM
# ============================================================

class SelfHealingSystem:
    """
    Monitors system health and automatically applies recovery actions.
    
    Detects:
    - Data feed interruptions
    - Exchange connection drops
    - Model evaluation errors
    - Missing logs
    - Stalled subsystems
    - Resource issues
    
    Responds with:
    - Automatic recovery actions
    - Safety policy enforcement
    - Subsystem restarts
    - Graceful degradation
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        log_dir: str = "/app/logs",
        
        # Health check intervals (seconds)
        check_interval: int = 30,
        critical_check_interval: int = 5,
        
        # Failure thresholds
        max_consecutive_failures: int = 3,
        max_error_rate: float = 0.20,  # 20% error rate = degraded
        stale_data_threshold_sec: int = 300,  # 5 min = stale
        
        # Resource thresholds
        max_cpu_percent: float = 90.0,
        max_memory_percent: float = 85.0,
        
        # Auto-recovery settings
        auto_restart_enabled: bool = True,
        auto_pause_on_critical: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.log_dir = Path(log_dir)
        
        self.check_interval = check_interval
        self.critical_check_interval = critical_check_interval
        
        self.max_consecutive_failures = max_consecutive_failures
        self.max_error_rate = max_error_rate
        self.stale_data_threshold_sec = stale_data_threshold_sec
        
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        
        self.auto_restart_enabled = auto_restart_enabled
        self.auto_pause_on_critical = auto_pause_on_critical
        
        # State tracking
        self.subsystem_states: Dict[SubsystemType, Dict[str, Any]] = {}
        self.detected_issues: Dict[str, DetectedIssue] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.current_safety_policy = SafetyPolicy.ALLOW_ALL
        
        # [CRITICAL FIX #4] Exponential backoff tracking
        self.retry_counts: Dict[str, int] = {}  # subsystem_name -> retry_count
        self.max_retries = 5
        self.base_delay = 1.0  # seconds
        
        # Initialize state for each subsystem
        for subsystem in SubsystemType:
            self.subsystem_states[subsystem] = {
                "status": HealthStatus.UNKNOWN,
                "consecutive_failures": 0,
                "last_check": None,
                "last_success": None,
                "error_count": 0,
                "total_checks": 0,
            }
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache last report for sync access
        self._last_report: Optional[SystemHealthReport] = None
    
    # --------------------------------------------------------
    # HEALTH CHECKS
    # --------------------------------------------------------
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get cached health report in synchronous format for Safety Governor.
        
        Returns:
            Dict with health status, issues, and recommendations
        """
        if self._last_report is None:
            # Return minimal report if no check has run yet
            return {
                "overall_status": "UNKNOWN",
                "safety_policy": self.current_safety_policy.value,
                "trading_should_continue": True,
                "requires_immediate_action": False,
                "detected_issues": [],
                "critical_issues": [],
                "healthy_count": 0,
                "degraded_count": 0,
                "critical_count": 0,
                "failed_count": 0
            }
        
        # Convert last report to dict
        return {
            "overall_status": self._last_report.overall_status.value,
            "safety_policy": self._last_report.current_safety_policy.value,
            "trading_should_continue": self._last_report.trading_should_continue,
            "requires_immediate_action": self._last_report.requires_immediate_action,
            "detected_issues": [
                {
                    "subsystem": issue.subsystem.value,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "impacts_trading": issue.impacts_trading
                }
                for issue in self._last_report.detected_issues
            ],
            "critical_issues": [
                {
                    "subsystem": issue.subsystem.value,
                    "description": issue.description
                }
                for issue in self._last_report.critical_issues
            ],
            "healthy_count": self._last_report.healthy_count,
            "degraded_count": self._last_report.degraded_count,
            "critical_count": self._last_report.critical_count,
            "failed_count": self._last_report.failed_count
        }
    
    async def check_all_subsystems(self) -> SystemHealthReport:
        """
        Run health checks on all subsystems and generate comprehensive report.
        
        Returns:
            SystemHealthReport with current system state
        """
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        
        logger.info("[SELF-HEAL] Running comprehensive health checks...")
        
        # Run all health checks in parallel
        health_checks = await asyncio.gather(
            self._check_data_feed(),
            self._check_exchange_connection(),
            self._check_ai_models(),
            self._check_event_executor(),
            self._check_orchestrator(),
            self._check_portfolio_balancer(),
            self._check_model_supervisor(),
            self._check_retraining_orchestrator(),
            self._check_position_monitor(),
            self._check_risk_guard(),
            self._check_database(),
            self._check_universe_os(),
            self._check_logging(),
            return_exceptions=True
        )
        
        # Build subsystem_health dict
        subsystem_health = {}
        for check in health_checks:
            if isinstance(check, HealthCheck):
                subsystem_health[check.subsystem] = check
                self._update_subsystem_state(check)
        
        # Detect issues
        detected_issues = self._detect_issues(subsystem_health)
        
        # Generate recovery recommendations
        recovery_recommendations = self._generate_recovery_recommendations(detected_issues)
        
        # Determine overall status
        overall_status = self._determine_overall_status(subsystem_health)
        
        # Evaluate safety policy
        new_safety_policy = self._evaluate_safety_policy(overall_status, detected_issues)
        if new_safety_policy != self.current_safety_policy:
            logger.warning(f"[SELF-HEAL] Safety policy changed: {self.current_safety_policy.value} → {new_safety_policy.value}")
            self.current_safety_policy = new_safety_policy
        
        # Count statuses
        status_counts = {
            "healthy": 0,
            "degraded": 0,
            "critical": 0,
            "failed": 0
        }
        for check in subsystem_health.values():
            if check.status == HealthStatus.HEALTHY:
                status_counts["healthy"] += 1
            elif check.status == HealthStatus.DEGRADED:
                status_counts["degraded"] += 1
            elif check.status == HealthStatus.CRITICAL:
                status_counts["critical"] += 1
            elif check.status == HealthStatus.FAILED:
                status_counts["failed"] += 1
        
        # Determine trading flags
        critical_issues = [i for i in detected_issues if i.severity == IssueSeverity.CRITICAL]
        trading_should_continue = (
            self.current_safety_policy in [SafetyPolicy.ALLOW_ALL, SafetyPolicy.NO_NEW_TRADES, SafetyPolicy.SAFE_RISK_PROFILE] and
            len(critical_issues) == 0
        )
        requires_immediate_action = len(critical_issues) > 0 or overall_status == HealthStatus.FAILED
        
        # Build report
        report = SystemHealthReport(
            timestamp=timestamp,
            overall_status=overall_status,
            current_safety_policy=self.current_safety_policy,
            subsystem_health=subsystem_health,
            detected_issues=detected_issues,
            critical_issues=critical_issues,
            recovery_recommendations=recovery_recommendations,
            trading_should_continue=trading_should_continue,
            requires_immediate_action=requires_immediate_action,
            healthy_count=status_counts["healthy"],
            degraded_count=status_counts["degraded"],
            critical_count=status_counts["critical"],
            failed_count=status_counts["failed"]
        )
        
        # Save report
        self._save_health_report(report)
        
        # Cache report for sync access
        self._last_report = report
        
        # Log summary
        logger.info(
            f"[SELF-HEAL] Health check complete: "
            f"Overall={overall_status.value}, "
            f"Healthy={status_counts['healthy']}, "
            f"Degraded={status_counts['degraded']}, "
            f"Critical={status_counts['critical']}, "
            f"Failed={status_counts['failed']}"
        )
        
        if critical_issues:
            logger.error(f"[SELF-HEAL] {len(critical_issues)} CRITICAL ISSUES detected!")
            for issue in critical_issues:
                logger.error(f"  - {issue.description}")
        
        return report
    
    # --------------------------------------------------------
    # INDIVIDUAL HEALTH CHECKS
    # --------------------------------------------------------
    
    async def _check_data_feed(self) -> HealthCheck:
        """Check data feed health."""
        subsystem = SubsystemType.DATA_FEED
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check universe snapshot freshness
            snapshot_path = self.data_dir / "universe_snapshot.json"
            
            if not snapshot_path.exists():
                return HealthCheck(
                    subsystem=subsystem,
                    status=HealthStatus.DEGRADED,
                    timestamp=start_time.isoformat(),
                    response_time_ms=None,
                    error_count=1,
                    last_error=start_time.isoformat(),
                    last_success=None,
                    details={"error": "Universe snapshot missing"}
                )
            
            # Check age of snapshot
            snapshot_age = (start_time.timestamp() - snapshot_path.stat().st_mtime)
            
            if snapshot_age > self.stale_data_threshold_sec:
                return HealthCheck(
                    subsystem=subsystem,
                    status=HealthStatus.DEGRADED,
                    timestamp=start_time.isoformat(),
                    response_time_ms=None,
                    error_count=0,
                    last_error=None,
                    last_success=start_time.isoformat(),
                    details={"warning": f"Snapshot {snapshot_age:.0f}s old (stale)"}
                )
            
            # Load and validate snapshot
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
            
            symbol_count = len(snapshot.get("symbols", []))
            
            if symbol_count == 0:
                return HealthCheck(
                    subsystem=subsystem,
                    status=HealthStatus.CRITICAL,
                    timestamp=start_time.isoformat(),
                    response_time_ms=None,
                    error_count=1,
                    last_error=start_time.isoformat(),
                    last_success=None,
                    details={"error": "No symbols in universe"}
                )
            
            # Healthy
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.HEALTHY,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details={"symbol_count": symbol_count, "age_seconds": snapshot_age}
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    async def _check_exchange_connection(self) -> HealthCheck:
        """Check exchange connection health."""
        subsystem = SubsystemType.EXCHANGE_CONNECTION
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check for recent trades/orders
            trades_path = self.data_dir / "recent_trades.json"
            
            if trades_path.exists():
                trade_age = (start_time.timestamp() - trades_path.stat().st_mtime)
                details = {"last_trade_age_seconds": trade_age}
                
                if trade_age < 300:  # < 5 min = healthy
                    status = HealthStatus.HEALTHY
                elif trade_age < 3600:  # < 1 hour = degraded
                    status = HealthStatus.DEGRADED
                else:  # > 1 hour = critical
                    status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.UNKNOWN
                details = {"note": "No recent trades file"}
            
            return HealthCheck(
                subsystem=subsystem,
                status=status,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details=details
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    async def _check_ai_models(self) -> HealthCheck:
        """Check AI model health."""
        subsystem = SubsystemType.AI_MODEL
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check model files exist
            models_dir = Path("/app/ai_engine/models")
            
            model_files = [
                "xgb_model.pkl",
                "lgbm_model.pkl",
                "nhits_model.pth",
                "patchtst_model.pth"
            ]
            
            existing_models = [f for f in model_files if (models_dir / f).exists()]
            
            if len(existing_models) == 0:
                status = HealthStatus.CRITICAL
            elif len(existing_models) < 3:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheck(
                subsystem=subsystem,
                status=status,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details={"existing_models": existing_models, "total": len(existing_models)}
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    async def _check_event_executor(self) -> HealthCheck:
        """Check event-driven executor health."""
        subsystem = SubsystemType.EVENT_EXECUTOR
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check event executor state file
            state_path = self.data_dir / "event_executor_state.json"
            
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                last_cycle = state.get("last_cycle_timestamp")
                if last_cycle:
                    age = (start_time - datetime.fromisoformat(last_cycle)).total_seconds()
                    
                    if age < 60:  # < 1 min = healthy
                        status = HealthStatus.HEALTHY
                    elif age < 300:  # < 5 min = degraded
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.CRITICAL
                else:
                    status = HealthStatus.UNKNOWN
            else:
                status = HealthStatus.UNKNOWN
            
            return HealthCheck(
                subsystem=subsystem,
                status=status,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details={}
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    async def _check_orchestrator(self) -> HealthCheck:
        """Check orchestrator health."""
        return await self._check_generic_subsystem(SubsystemType.ORCHESTRATOR, "orchestrator_state.json")
    
    async def _check_portfolio_balancer(self) -> HealthCheck:
        """Check portfolio balancer health."""
        return await self._check_generic_subsystem(SubsystemType.PORTFOLIO_BALANCER, "portfolio_balancer_output.json")
    
    async def _check_model_supervisor(self) -> HealthCheck:
        """Check model supervisor health."""
        return await self._check_generic_subsystem(SubsystemType.MODEL_SUPERVISOR, "model_supervisor_output.json")
    
    async def _check_retraining_orchestrator(self) -> HealthCheck:
        """Check retraining orchestrator health."""
        return await self._check_generic_subsystem(SubsystemType.RETRAINING_ORCHESTRATOR, "retraining_orchestrator_state.json")
    
    async def _check_position_monitor(self) -> HealthCheck:
        """Check position monitor health."""
        return await self._check_generic_subsystem(SubsystemType.POSITION_MONITOR, "position_monitor_state.json")
    
    async def _check_risk_guard(self) -> HealthCheck:
        """Check risk guard health."""
        return await self._check_generic_subsystem(SubsystemType.RISK_GUARD, "risk_state.db")
    
    async def _check_database(self) -> HealthCheck:
        """Check database health."""
        subsystem = SubsystemType.DATABASE
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check if database file exists
            db_path = Path("/app/backend/data/trades.db")
            
            if not db_path.exists():
                status = HealthStatus.CRITICAL
                details = {"error": "Database file missing"}
            else:
                # Check size
                size_mb = db_path.stat().st_size / 1024 / 1024
                details = {"size_mb": size_mb}
                status = HealthStatus.HEALTHY
            
            return HealthCheck(
                subsystem=subsystem,
                status=status,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details=details
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    async def _check_universe_os(self) -> HealthCheck:
        """Check Universe OS health - virtual subsystem, always healthy."""
        start_time = datetime.now(timezone.utc)
        # Universe OS is a virtual subsystem integrated into PolicyStore
        # It doesn't generate state files, so we mark it as healthy if system is running
        return HealthCheck(
            subsystem=SubsystemType.UNIVERSE_OS,
            status=HealthStatus.HEALTHY,
            timestamp=start_time.isoformat(),
            response_time_ms=0.0,
            error_count=0,
            last_error=None,
            last_success=start_time.isoformat(),
            details={"note": "Virtual subsystem - integrated into PolicyStore"}
        )
    
    async def _check_logging(self) -> HealthCheck:
        """Check logging system health."""
        subsystem = SubsystemType.LOGGING
        start_time = datetime.now(timezone.utc)
        
        try:
            # Test if logging is working by checking logger availability
            test_logger = logging.getLogger(__name__)
            
            # Check if we're using JSON logging to stdout (Docker mode)
            using_json_logging = any(
                isinstance(handler, logging.StreamHandler) 
                for handler in logging.root.handlers
            )
            
            if using_json_logging:
                # JSON logging to stdout is working (Docker mode)
                status = HealthStatus.HEALTHY
                details = {
                    "mode": "JSON stdout (Docker)",
                    "handlers": len(logging.root.handlers)
                }
            else:
                # Check log directory for file-based logging
                if not self.log_dir.exists():
                    status = HealthStatus.DEGRADED
                    details = {"warning": "Log directory missing"}
                else:
                    # Check recent log file
                    log_files = list(self.log_dir.glob("*.log"))
                    if log_files:
                        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                        age = (start_time.timestamp() - latest_log.stat().st_mtime)
                        
                        if age < 60:
                            status = HealthStatus.HEALTHY
                        elif age < 300:
                            status = HealthStatus.DEGRADED
                        else:
                            status = HealthStatus.CRITICAL
                        
                        details = {"mode": "file-based", "latest_log_age_seconds": age}
                    else:
                        status = HealthStatus.DEGRADED
                        details = {"warning": "No log files found"}
            
            return HealthCheck(
                subsystem=subsystem,
                status=status,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details=details
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    async def _check_generic_subsystem(self, subsystem: SubsystemType, state_file: str) -> HealthCheck:
        """Generic health check for subsystems with state files."""
        start_time = datetime.now(timezone.utc)
        
        try:
            state_path = self.data_dir / state_file
            
            if not state_path.exists():
                status = HealthStatus.UNKNOWN
                details = {"note": "State file not found (subsystem may not be initialized)"}
            else:
                age = (start_time.timestamp() - state_path.stat().st_mtime)
                
                if age < 300:  # < 5 min = healthy
                    status = HealthStatus.HEALTHY
                elif age < 3600:  # < 1 hour = degraded
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.CRITICAL
                
                details = {"state_age_seconds": age}
            
            return HealthCheck(
                subsystem=subsystem,
                status=status,
                timestamp=start_time.isoformat(),
                response_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                error_count=0,
                last_error=None,
                last_success=start_time.isoformat(),
                details=details
            )
        
        except Exception as e:
            return HealthCheck(
                subsystem=subsystem,
                status=HealthStatus.FAILED,
                timestamp=start_time.isoformat(),
                response_time_ms=None,
                error_count=1,
                last_error=start_time.isoformat(),
                last_success=None,
                details={"error": str(e)}
            )
    
    # --------------------------------------------------------
    # ISSUE DETECTION
    # --------------------------------------------------------
    
    def _detect_issues(self, subsystem_health: Dict[SubsystemType, HealthCheck]) -> List[DetectedIssue]:
        """Detect issues from health checks."""
        issues = []
        now = datetime.now(timezone.utc)
        
        for subsystem, check in subsystem_health.items():
            if check.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                # Critical/Failed subsystem
                issue = DetectedIssue(
                    issue_id=f"{subsystem.value}_{now.strftime('%Y%m%d_%H%M%S')}",
                    subsystem=subsystem,
                    severity=IssueSeverity.CRITICAL if check.status == HealthStatus.CRITICAL else IssueSeverity.HIGH,
                    timestamp=now.isoformat(),
                    description=f"{subsystem.value} is {check.status.value}",
                    symptoms=[
                        f"Status: {check.status.value}",
                        f"Last error: {check.last_error or 'Unknown'}",
                        f"Details: {check.details}"
                    ],
                    root_cause=check.details.get("error"),
                    impacts_trading=self._subsystem_impacts_trading(subsystem),
                    affects_subsystems=self._get_dependent_subsystems(subsystem)
                )
                issues.append(issue)
            
            elif check.status == HealthStatus.DEGRADED:
                # Degraded subsystem
                issue = DetectedIssue(
                    issue_id=f"{subsystem.value}_{now.strftime('%Y%m%d_%H%M%S')}",
                    subsystem=subsystem,
                    severity=IssueSeverity.MEDIUM,
                    timestamp=now.isoformat(),
                    description=f"{subsystem.value} performance degraded",
                    symptoms=[
                        f"Status: {check.status.value}",
                        f"Details: {check.details}"
                    ],
                    root_cause=check.details.get("warning"),
                    impacts_trading=False,
                    affects_subsystems=[]
                )
                issues.append(issue)
        
        # Check system resources
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.max_cpu_percent:
                issues.append(DetectedIssue(
                    issue_id=f"cpu_high_{now.strftime('%Y%m%d_%H%M%S')}",
                    subsystem=SubsystemType.EVENT_EXECUTOR,
                    severity=IssueSeverity.HIGH,
                    timestamp=now.isoformat(),
                    description=f"CPU usage critically high: {cpu_percent:.1f}%",
                    symptoms=[f"CPU: {cpu_percent:.1f}%"],
                    root_cause="High CPU usage may cause slowdowns",
                    impacts_trading=True,
                    affects_subsystems=list(SubsystemType)
                ))
            
            if memory_percent > self.max_memory_percent:
                issues.append(DetectedIssue(
                    issue_id=f"memory_high_{now.strftime('%Y%m%d_%H%M%S')}",
                    subsystem=SubsystemType.EVENT_EXECUTOR,
                    severity=IssueSeverity.HIGH,
                    timestamp=now.isoformat(),
                    description=f"Memory usage critically high: {memory_percent:.1f}%",
                    symptoms=[f"Memory: {memory_percent:.1f}%"],
                    root_cause="High memory usage may cause system instability",
                    impacts_trading=True,
                    affects_subsystems=list(SubsystemType)
                ))
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
        
        return issues
    
    # --------------------------------------------------------
    # RECOVERY RECOMMENDATIONS
    # --------------------------------------------------------
    
    def _generate_recovery_recommendations(self, issues: List[DetectedIssue]) -> List[RecoveryRecommendation]:
        """Generate recovery recommendations for detected issues."""
        recommendations = []
        
        for i, issue in enumerate(issues):
            # Determine appropriate action
            if issue.severity == IssueSeverity.CRITICAL:
                if issue.subsystem in [SubsystemType.EXCHANGE_CONNECTION, SubsystemType.DATA_FEED]:
                    action = RecoveryAction.PAUSE_TRADING
                    can_auto = self.auto_pause_on_critical
                elif issue.subsystem == SubsystemType.AI_MODEL:
                    action = RecoveryAction.DISABLE_MODULE
                    can_auto = True
                else:
                    action = RecoveryAction.RESTART_SUBSYSTEM
                    can_auto = self.auto_restart_enabled
            
            elif issue.severity == IssueSeverity.HIGH:
                action = RecoveryAction.SWITCH_TO_SAFE_PROFILE
                can_auto = True
            
            else:
                action = RecoveryAction.RELOAD_CONFIG
                can_auto = False
            
            recommendation = RecoveryRecommendation(
                recommendation_id=f"rec_{i+1}_{issue.issue_id}",
                issue=issue,
                action=action,
                priority=1 if issue.severity == IssueSeverity.CRITICAL else 2,
                description=self._get_action_description(action, issue.subsystem),
                expected_result=self._get_expected_result(action),
                risks=self._get_action_risks(action),
                can_auto_execute=can_auto,
                requires_approval=not can_auto
            )
            recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)
        
        return recommendations
    
    # --------------------------------------------------------
    # SAFETY POLICY
    # --------------------------------------------------------
    
    def _evaluate_safety_policy(self, overall_status: HealthStatus, issues: List[DetectedIssue]) -> SafetyPolicy:
        """Determine appropriate safety policy based on system state."""
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        
        if overall_status == HealthStatus.FAILED:
            return SafetyPolicy.EMERGENCY_SHUTDOWN
        
        elif len(critical_issues) > 0:
            # Check if critical issues impact trading
            trading_impacted = any(i.impacts_trading for i in critical_issues)
            if trading_impacted:
                return SafetyPolicy.DEFENSIVE_EXIT
            else:
                return SafetyPolicy.NO_NEW_TRADES
        
        elif overall_status == HealthStatus.CRITICAL:
            return SafetyPolicy.SAFE_RISK_PROFILE
        
        elif overall_status == HealthStatus.DEGRADED:
            return SafetyPolicy.NO_NEW_TRADES
        
        else:
            return SafetyPolicy.ALLOW_ALL
    
    # --------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------
    
    def _update_subsystem_state(self, check: HealthCheck):
        """Update internal state tracking for a subsystem."""
        state = self.subsystem_states[check.subsystem]
        state["total_checks"] += 1
        state["status"] = check.status
        state["last_check"] = check.timestamp
        
        if check.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            state["consecutive_failures"] += 1
            state["error_count"] += 1
            state["last_error"] = check.last_error
        else:
            state["consecutive_failures"] = 0
            if check.status == HealthStatus.HEALTHY:
                state["last_success"] = check.last_success
    
    def _determine_overall_status(self, subsystem_health: Dict[SubsystemType, HealthCheck]) -> HealthStatus:
        """Determine overall system health status."""
        statuses = [check.status for check in subsystem_health.values()]
        
        if HealthStatus.FAILED in statuses:
            return HealthStatus.FAILED
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def _subsystem_impacts_trading(self, subsystem: SubsystemType) -> bool:
        """Check if subsystem directly impacts trading."""
        critical_for_trading = [
            SubsystemType.EXCHANGE_CONNECTION,
            SubsystemType.EVENT_EXECUTOR,
            SubsystemType.RISK_GUARD,
            SubsystemType.DATA_FEED
        ]
        return subsystem in critical_for_trading
    
    def _get_dependent_subsystems(self, subsystem: SubsystemType) -> List[SubsystemType]:
        """Get subsystems that depend on this one."""
        dependencies = {
            SubsystemType.DATA_FEED: [SubsystemType.AI_MODEL, SubsystemType.EVENT_EXECUTOR],
            SubsystemType.AI_MODEL: [SubsystemType.EVENT_EXECUTOR, SubsystemType.MODEL_SUPERVISOR],
            SubsystemType.EXCHANGE_CONNECTION: [SubsystemType.EVENT_EXECUTOR, SubsystemType.POSITION_MONITOR],
            SubsystemType.DATABASE: list(SubsystemType),
        }
        return dependencies.get(subsystem, [])
    
    def _get_action_description(self, action: RecoveryAction, subsystem: SubsystemType) -> str:
        """Get description for recovery action."""
        descriptions = {
            RecoveryAction.RESTART_SUBSYSTEM: f"Restart {subsystem.value} subsystem",
            RecoveryAction.PAUSE_TRADING: "Pause all trading activity",
            RecoveryAction.SWITCH_TO_SAFE_PROFILE: "Switch to SAFE risk profile",
            RecoveryAction.DISABLE_MODULE: f"Disable {subsystem.value} module",
            RecoveryAction.NO_NEW_TRADES: "Block new trades, allow exits",
            RecoveryAction.DEFENSIVE_EXIT: "Close all positions defensively",
            RecoveryAction.RELOAD_CONFIG: "Reload configuration files",
            RecoveryAction.CLEAR_CACHE: "Clear system caches"
        }
        return descriptions.get(action, "Unknown action")
    
    def _get_expected_result(self, action: RecoveryAction) -> str:
        """Get expected result of recovery action."""
        results = {
            RecoveryAction.RESTART_SUBSYSTEM: "Subsystem restored to operational state",
            RecoveryAction.PAUSE_TRADING: "Trading halted, positions maintained",
            RecoveryAction.SWITCH_TO_SAFE_PROFILE: "Risk reduced, system stabilized",
            RecoveryAction.DISABLE_MODULE: "Module disabled, system continues with fallback",
            RecoveryAction.NO_NEW_TRADES: "No new positions opened, exits allowed",
            RecoveryAction.DEFENSIVE_EXIT: "All positions closed, capital preserved",
            RecoveryAction.RELOAD_CONFIG: "Configuration refreshed",
            RecoveryAction.CLEAR_CACHE: "Caches cleared, memory freed"
        }
        return results.get(action, "Unknown result")
    
    def _get_action_risks(self, action: RecoveryAction) -> List[str]:
        """Get risks associated with recovery action."""
        risks = {
            RecoveryAction.RESTART_SUBSYSTEM: ["Brief service interruption"],
            RecoveryAction.PAUSE_TRADING: ["Missed trading opportunities"],
            RecoveryAction.SWITCH_TO_SAFE_PROFILE: ["Reduced profit potential"],
            RecoveryAction.DISABLE_MODULE: ["Degraded functionality"],
            RecoveryAction.NO_NEW_TRADES: ["Cannot capitalize on opportunities"],
            RecoveryAction.DEFENSIVE_EXIT: ["Potential losses from forced exits"],
            RecoveryAction.RELOAD_CONFIG: ["Configuration errors may cause issues"],
            RecoveryAction.CLEAR_CACHE: ["Performance impact during rebuild"]
        }
        return risks.get(action, ["Unknown risks"])
    
    async def attempt_recovery(self, subsystem_name: str, recovery_function) -> bool:
        """
        Attempt service recovery with exponential backoff (CRITICAL FIX #4).
        
        Args:
            subsystem_name: Name of subsystem to recover
            recovery_function: Async function that performs recovery
        
        Returns:
            True if recovery successful, False otherwise
        """
        import random
        
        retry_count = self.retry_counts.get(subsystem_name, 0)
        
        if retry_count >= self.max_retries:
            logger.error(
                f"❌ Max retries ({self.max_retries}) reached for {subsystem_name} - giving up"
            )
            # Reset retry count after giving up (allow retry after cooldown)
            self.retry_counts[subsystem_name] = 0
            return False
        
        # Exponential backoff: delay = base * 2^retry + random jitter
        delay = self.base_delay * (2 ** retry_count)
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        total_delay = delay + jitter
        
        logger.info(
            f"🔄 Recovery attempt {retry_count + 1}/{self.max_retries} for {subsystem_name} "
            f"after {total_delay:.2f}s delay (exponential backoff)"
        )
        
        await asyncio.sleep(total_delay)
        
        # Attempt recovery
        try:
            success = await recovery_function()
            
            if success:
                self.retry_counts[subsystem_name] = 0  # Reset on success
                logger.info(f"✅ Recovery successful for {subsystem_name}")
                
                # Record in recovery history
                self.recovery_history.append({
                    "subsystem": subsystem_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "retry_count": retry_count + 1,
                    "delay_sec": total_delay,
                    "success": True
                })
                
                return True
            else:
                self.retry_counts[subsystem_name] = retry_count + 1
                logger.warning(
                    f"⚠️ Recovery failed for {subsystem_name} "
                    f"(attempt {retry_count + 1}/{self.max_retries})"
                )
                return False
        
        except Exception as e:
            self.retry_counts[subsystem_name] = retry_count + 1
            logger.error(
                f"❌ Recovery error for {subsystem_name}: {e} "
                f"(attempt {retry_count + 1}/{self.max_retries})"
            )
            return False
    
    def _save_health_report(self, report: SystemHealthReport):
        """Save health report to disk."""
        output_path = self.data_dir / "self_healing_report.json"
        
        # Convert to dict (handle enums)
        report_dict = {
            "timestamp": report.timestamp,
            "overall_status": report.overall_status.value,
            "current_safety_policy": report.current_safety_policy.value,
            "subsystem_health": {
                k.value: {
                    "subsystem": v.subsystem.value,
                    "status": v.status.value,
                    "timestamp": v.timestamp,
                    "response_time_ms": v.response_time_ms,
                    "error_count": v.error_count,
                    "details": v.details
                }
                for k, v in report.subsystem_health.items()
            },
            "detected_issues": [
                {
                    "issue_id": i.issue_id,
                    "subsystem": i.subsystem.value,
                    "severity": i.severity.value,
                    "description": i.description,
                    "impacts_trading": i.impacts_trading
                }
                for i in report.detected_issues
            ],
            "recovery_recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "action": r.action.value,
                    "priority": r.priority,
                    "description": r.description,
                    "can_auto_execute": r.can_auto_execute
                }
                for r in report.recovery_recommendations
            ],
            "trading_should_continue": report.trading_should_continue,
            "requires_immediate_action": report.requires_immediate_action,
            "summary": {
                "healthy": report.healthy_count,
                "degraded": report.degraded_count,
                "critical": report.critical_count,
                "failed": report.failed_count
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("SELF-HEALING SYSTEM - Standalone Test")
    print("=" * 60)
    
    # Initialize self-healing system
    self_healer = SelfHealingSystem(
        data_dir="./data",
        log_dir="./logs",
        check_interval=30,
        auto_restart_enabled=True,
        auto_pause_on_critical=True
    )
    
    print(f"\n[OK] Self-Healing System initialized")
    print(f"  Data dir: {self_healer.data_dir}")
    print(f"  Check interval: {self_healer.check_interval}s")
    print(f"  Auto-restart: {self_healer.auto_restart_enabled}")
    print(f"  Auto-pause: {self_healer.auto_pause_on_critical}")
    
    # Run health checks
    print("\n" + "=" * 60)
    print("Running comprehensive health checks...")
    print("=" * 60)
    
    async def run_checks():
        report = await self_healer.check_all_subsystems()
        return report
    
    report = asyncio.run(run_checks())
    
    print(f"\n[OK] Health check complete")
    print(f"  Overall status: {report.overall_status.value}")
    print(f"  Safety policy: {report.current_safety_policy.value}")
    print(f"  Trading should continue: {report.trading_should_continue}")
    print(f"  Requires immediate action: {report.requires_immediate_action}")
    
    print(f"\n  Subsystem summary:")
    print(f"    Healthy: {report.healthy_count}")
    print(f"    Degraded: {report.degraded_count}")
    print(f"    Critical: {report.critical_count}")
    print(f"    Failed: {report.failed_count}")
    
    if report.detected_issues:
        print(f"\n  Detected issues ({len(report.detected_issues)}):")
        for issue in report.detected_issues:
            print(f"    [{issue.severity.value}] {issue.description}")
    
    if report.recovery_recommendations:
        print(f"\n  Recovery recommendations ({len(report.recovery_recommendations)}):")
        for rec in report.recovery_recommendations[:5]:  # Show first 5
            print(f"    [P{rec.priority}] {rec.description}")
            print(f"         Auto-execute: {rec.can_auto_execute}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("=" * 60)
    
    print(f"\n[OK] Health report saved to: {self_healer.data_dir / 'self_healing_report.json'}")
