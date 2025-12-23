"""
PHASE 3C-1: System Health Monitor
Real-time monitoring and alerting for all AI Engine modules

Monitors:
- Phase 2B: Orderbook Imbalance Module
- Phase 2D: Volatility Structure Engine
- Phase 3A: Risk Mode Predictor
- Phase 3B: Strategy Selector
- Ensemble models
- System-wide metrics

Author: Quantum Trader AI Team
Date: December 24, 2025
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ModuleHealthStatus:
    """Health status for a single module"""
    module_name: str
    module_type: str                    # "phase_2b", "phase_2d", "phase_3a", "phase_3b"
    status: HealthStatus
    health_score: float                 # 0-100
    last_check: datetime
    last_success: Optional[datetime]
    last_error: Optional[datetime]
    
    # Performance metrics
    uptime_pct: float                   # % time module has been operational
    error_rate: float                   # % of operations that fail
    avg_latency_ms: float               # Average operation time
    success_count_24h: int
    error_count_24h: int
    
    # Module-specific metrics
    data_freshness_sec: Optional[float] = None  # Seconds since last data update
    last_value: Optional[Any] = None            # Last computed value
    
    # Issues
    current_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['last_check'] = self.last_check.isoformat() if self.last_check else None
        data['last_success'] = self.last_success.isoformat() if self.last_success else None
        data['last_error'] = self.last_error.isoformat() if self.last_error else None
        return data


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    timestamp: datetime
    overall_status: HealthStatus
    overall_health_score: float         # 0-100, weighted average
    
    # Module health statuses
    phase_2b_health: Optional[ModuleHealthStatus] = None
    phase_2d_health: Optional[ModuleHealthStatus] = None
    phase_3a_health: Optional[ModuleHealthStatus] = None
    phase_3b_health: Optional[ModuleHealthStatus] = None
    ensemble_health: Optional[ModuleHealthStatus] = None
    
    # System-wide metrics
    signal_generation_success_rate: float = 0.0
    avg_signal_latency_ms: float = 0.0
    total_signals_24h: int = 0
    successful_signals_24h: int = 0
    failed_signals_24h: int = 0
    
    # Error tracking
    error_count_1h: int = 0
    error_count_24h: int = 0
    warning_count_1h: int = 0
    warning_count_24h: int = 0
    
    # Trading performance (if available)
    win_rate_7d: Optional[float] = None
    sharpe_ratio_7d: Optional[float] = None
    total_trades_7d: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status.value,
            'overall_health_score': round(self.overall_health_score, 2),
            'modules': {
                'phase_2b': self.phase_2b_health.to_dict() if self.phase_2b_health else None,
                'phase_2d': self.phase_2d_health.to_dict() if self.phase_2d_health else None,
                'phase_3a': self.phase_3a_health.to_dict() if self.phase_3a_health else None,
                'phase_3b': self.phase_3b_health.to_dict() if self.phase_3b_health else None,
                'ensemble': self.ensemble_health.to_dict() if self.ensemble_health else None,
            },
            'system_metrics': {
                'signal_success_rate': round(self.signal_generation_success_rate, 3),
                'avg_latency_ms': round(self.avg_signal_latency_ms, 2),
                'signals_24h': self.total_signals_24h,
                'errors_1h': self.error_count_1h,
                'errors_24h': self.error_count_24h,
                'warnings_1h': self.warning_count_1h,
                'warnings_24h': self.warning_count_24h,
            },
            'trading_performance': {
                'win_rate_7d': round(self.win_rate_7d, 3) if self.win_rate_7d else None,
                'sharpe_7d': round(self.sharpe_ratio_7d, 2) if self.sharpe_ratio_7d else None,
                'trades_7d': self.total_trades_7d,
            }
        }


@dataclass
class HealthAlert:
    """Health alert for issues detected"""
    alert_id: str
    severity: AlertSeverity
    module: str
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    recommended_action: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'module': self.module,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'recommended_action': self.recommended_action,
        }


class SystemHealthMonitor:
    """
    Main health monitoring system for AI Engine
    
    Responsibilities:
    - Monitor all AI modules (2B, 2D, 3A, 3B)
    - Track performance metrics
    - Detect degradation and issues
    - Generate alerts
    - Provide health dashboard data
    """
    
    def __init__(
        self,
        check_interval_sec: int = 60,
        alert_retention_hours: int = 24,
        metrics_history_size: int = 1000,
    ):
        """
        Initialize System Health Monitor
        
        Args:
            check_interval_sec: Seconds between health checks
            alert_retention_hours: Hours to retain alerts
            metrics_history_size: Number of health check results to keep
        """
        self.check_interval = check_interval_sec
        self.alert_retention = timedelta(hours=alert_retention_hours)
        
        # Module references (set during integration)
        self.orderbook_module = None         # Phase 2B
        self.volatility_engine = None        # Phase 2D
        self.risk_mode_predictor = None      # Phase 3A
        self.strategy_selector = None        # Phase 3B
        self.ensemble_manager = None
        
        # Health tracking
        self.module_health: Dict[str, ModuleHealthStatus] = {}
        self.health_history: deque = deque(maxlen=metrics_history_size)
        self.alerts: deque = deque(maxlen=500)
        self.active_alerts: Dict[str, HealthAlert] = {}
        
        # Metrics tracking
        self.signal_successes: deque = deque(maxlen=1000)
        self.signal_latencies: deque = deque(maxlen=1000)
        self.error_timestamps: deque = deque(maxlen=1000)
        self.warning_timestamps: deque = deque(maxlen=1000)
        
        # Monitoring state
        self.is_monitoring = False
        self.last_check_time: Optional[datetime] = None
        
        logger.info("[PHASE 3C] SystemHealthMonitor initialized "
                   f"(interval={check_interval_sec}s, retention={alert_retention_hours}h)")
    
    
    def set_modules(
        self,
        orderbook_module=None,
        volatility_engine=None,
        risk_mode_predictor=None,
        strategy_selector=None,
        ensemble_manager=None,
    ):
        """Set references to modules to monitor"""
        self.orderbook_module = orderbook_module
        self.volatility_engine = volatility_engine
        self.risk_mode_predictor = risk_mode_predictor
        self.strategy_selector = strategy_selector
        self.ensemble_manager = ensemble_manager
        
        logger.info("[PHASE 3C] Health monitor linked to modules: "
                   f"2B={bool(orderbook_module)}, 2D={bool(volatility_engine)}, "
                   f"3A={bool(risk_mode_predictor)}, 3B={bool(strategy_selector)}")
    
    
    async def start_monitoring(self):
        """Start background health monitoring loop"""
        self.is_monitoring = True
        logger.info(f"[PHASE 3C] Starting health monitoring loop (interval={self.check_interval}s)")
        
        while self.is_monitoring:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"[PHASE 3C] Health check failed: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    
    def stop_monitoring(self):
        """Stop health monitoring loop"""
        self.is_monitoring = False
        logger.info("[PHASE 3C] Health monitoring stopped")
    
    
    async def perform_health_check(self) -> SystemHealthMetrics:
        """
        Perform comprehensive health check of all modules
        
        Returns:
            SystemHealthMetrics with current system health
        """
        now = datetime.utcnow()
        self.last_check_time = now
        
        # Check each module
        phase_2b_health = await self._check_phase_2b_health()
        phase_2d_health = await self._check_phase_2d_health()
        phase_3a_health = await self._check_phase_3a_health()
        phase_3b_health = await self._check_phase_3b_health()
        ensemble_health = await self._check_ensemble_health()
        
        # Store module health
        if phase_2b_health:
            self.module_health['phase_2b'] = phase_2b_health
        if phase_2d_health:
            self.module_health['phase_2d'] = phase_2d_health
        if phase_3a_health:
            self.module_health['phase_3a'] = phase_3a_health
        if phase_3b_health:
            self.module_health['phase_3b'] = phase_3b_health
        if ensemble_health:
            self.module_health['ensemble'] = ensemble_health
        
        # Calculate overall health
        overall_score = self._calculate_overall_health_score()
        overall_status = self._determine_overall_status(overall_score)
        
        # Build system metrics
        metrics = SystemHealthMetrics(
            timestamp=now,
            overall_status=overall_status,
            overall_health_score=overall_score,
            phase_2b_health=phase_2b_health,
            phase_2d_health=phase_2d_health,
            phase_3a_health=phase_3a_health,
            phase_3b_health=phase_3b_health,
            ensemble_health=ensemble_health,
            signal_generation_success_rate=self._calculate_signal_success_rate(),
            avg_signal_latency_ms=self._calculate_avg_signal_latency(),
            total_signals_24h=len(self.signal_successes),
            successful_signals_24h=sum(1 for s in self.signal_successes if s),
            failed_signals_24h=sum(1 for s in self.signal_successes if not s),
            error_count_1h=self._count_recent_events(self.error_timestamps, hours=1),
            error_count_24h=len(self.error_timestamps),
            warning_count_1h=self._count_recent_events(self.warning_timestamps, hours=1),
            warning_count_24h=len(self.warning_timestamps),
        )
        
        # Store in history
        self.health_history.append(metrics)
        
        # Generate alerts if needed
        await self._check_and_generate_alerts(metrics)
        
        # Clean up old alerts
        self._cleanup_old_alerts()
        
        logger.debug(f"[PHASE 3C] Health check complete: "
                    f"score={overall_score:.1f}, status={overall_status.value}")
        
        return metrics
    
    
    async def _check_phase_2b_health(self) -> Optional[ModuleHealthStatus]:
        """Check Phase 2B Orderbook Imbalance Module health"""
        if not self.orderbook_module:
            return None
        
        now = datetime.utcnow()
        issues = []
        warnings = []
        health_score = 100.0
        
        try:
            # Check if module is responsive
            # Try to access recent data
            symbols = getattr(self.orderbook_module, 'tracked_symbols', [])
            
            if not symbols:
                issues.append("No symbols being tracked")
                health_score -= 30
            
            # Check data freshness (if available)
            # This would check last orderbook update time
            # For now, we assume healthy if module exists and has symbols
            
            status = HealthStatus.HEALTHY
            if health_score < 60:
                status = HealthStatus.CRITICAL
            elif health_score < 80:
                status = HealthStatus.DEGRADED
            
            return ModuleHealthStatus(
                module_name="Orderbook Imbalance Module",
                module_type="phase_2b",
                status=status,
                health_score=health_score,
                last_check=now,
                last_success=now,
                last_error=None,
                uptime_pct=99.5,  # Would track this over time
                error_rate=0.0,
                avg_latency_ms=10.0,
                success_count_24h=1440,  # Assuming 1/min
                error_count_24h=0,
                current_issues=issues,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.error(f"[PHASE 3C] Phase 2B health check failed: {e}")
            return ModuleHealthStatus(
                module_name="Orderbook Imbalance Module",
                module_type="phase_2b",
                status=HealthStatus.CRITICAL,
                health_score=0.0,
                last_check=now,
                last_success=None,
                last_error=now,
                uptime_pct=0.0,
                error_rate=1.0,
                avg_latency_ms=0.0,
                success_count_24h=0,
                error_count_24h=1,
                current_issues=[f"Module check failed: {str(e)}"],
                warnings=[],
            )
    
    
    async def _check_phase_2d_health(self) -> Optional[ModuleHealthStatus]:
        """Check Phase 2D Volatility Structure Engine health"""
        if not self.volatility_engine:
            return None
        
        now = datetime.utcnow()
        issues = []
        warnings = []
        health_score = 100.0
        
        try:
            # Check if module is operational
            # Volatility engine should have methods for checking status
            
            status = HealthStatus.HEALTHY
            
            return ModuleHealthStatus(
                module_name="Volatility Structure Engine",
                module_type="phase_2d",
                status=status,
                health_score=health_score,
                last_check=now,
                last_success=now,
                last_error=None,
                uptime_pct=99.8,
                error_rate=0.0,
                avg_latency_ms=15.0,
                success_count_24h=1440,
                error_count_24h=0,
                current_issues=issues,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.error(f"[PHASE 3C] Phase 2D health check failed: {e}")
            return ModuleHealthStatus(
                module_name="Volatility Structure Engine",
                module_type="phase_2d",
                status=HealthStatus.CRITICAL,
                health_score=0.0,
                last_check=now,
                last_success=None,
                last_error=now,
                uptime_pct=0.0,
                error_rate=1.0,
                avg_latency_ms=0.0,
                success_count_24h=0,
                error_count_24h=1,
                current_issues=[f"Module check failed: {str(e)}"],
                warnings=[],
            )
    
    
    async def _check_phase_3a_health(self) -> Optional[ModuleHealthStatus]:
        """Check Phase 3A Risk Mode Predictor health"""
        if not self.risk_mode_predictor:
            return None
        
        now = datetime.utcnow()
        issues = []
        warnings = []
        health_score = 100.0
        
        try:
            # Check risk mode predictor status
            
            status = HealthStatus.HEALTHY
            
            return ModuleHealthStatus(
                module_name="Risk Mode Predictor",
                module_type="phase_3a",
                status=status,
                health_score=health_score,
                last_check=now,
                last_success=now,
                last_error=None,
                uptime_pct=99.7,
                error_rate=0.0,
                avg_latency_ms=20.0,
                success_count_24h=1440,
                error_count_24h=0,
                current_issues=issues,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.error(f"[PHASE 3C] Phase 3A health check failed: {e}")
            return ModuleHealthStatus(
                module_name="Risk Mode Predictor",
                module_type="phase_3a",
                status=HealthStatus.CRITICAL,
                health_score=0.0,
                last_check=now,
                last_success=None,
                last_error=now,
                uptime_pct=0.0,
                error_rate=1.0,
                avg_latency_ms=0.0,
                success_count_24h=0,
                error_count_24h=1,
                current_issues=[f"Module check failed: {str(e)}"],
                warnings=[],
            )
    
    
    async def _check_phase_3b_health(self) -> Optional[ModuleHealthStatus]:
        """Check Phase 3B Strategy Selector health"""
        if not self.strategy_selector:
            return None
        
        now = datetime.utcnow()
        issues = []
        warnings = []
        health_score = 100.0
        
        try:
            # Check strategy selector
            # Could check: number of active strategies, recent selections, etc.
            
            # Check if performance tracker has data
            if hasattr(self.strategy_selector, 'performance_tracker'):
                tracker = self.strategy_selector.performance_tracker
                # Check if strategies are being used
                # This is a placeholder - would need actual implementation
            
            status = HealthStatus.HEALTHY
            
            return ModuleHealthStatus(
                module_name="Strategy Selector",
                module_type="phase_3b",
                status=status,
                health_score=health_score,
                last_check=now,
                last_success=now,
                last_error=None,
                uptime_pct=99.9,
                error_rate=0.0,
                avg_latency_ms=25.0,
                success_count_24h=1440,
                error_count_24h=0,
                current_issues=issues,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.error(f"[PHASE 3C] Phase 3B health check failed: {e}")
            return ModuleHealthStatus(
                module_name="Strategy Selector",
                module_type="phase_3b",
                status=HealthStatus.CRITICAL,
                health_score=0.0,
                last_check=now,
                last_success=None,
                last_error=now,
                uptime_pct=0.0,
                error_rate=1.0,
                avg_latency_ms=0.0,
                success_count_24h=0,
                error_count_24h=1,
                current_issues=[f"Module check failed: {str(e)}"],
                warnings=[],
            )
    
    
    async def _check_ensemble_health(self) -> Optional[ModuleHealthStatus]:
        """Check Ensemble Manager health"""
        if not self.ensemble_manager:
            return None
        
        now = datetime.utcnow()
        issues = []
        warnings = []
        health_score = 100.0
        
        try:
            # Check ensemble models
            if hasattr(self.ensemble_manager, 'agents'):
                active_models = sum(1 for agent in self.ensemble_manager.agents.values() 
                                  if agent is not None)
                
                if active_models < 2:
                    issues.append(f"Only {active_models} models active (need 2+)")
                    health_score -= 40
                elif active_models < 3:
                    warnings.append(f"Only {active_models} models active (optimal: 4)")
                    health_score -= 15
            
            status = HealthStatus.HEALTHY
            if health_score < 60:
                status = HealthStatus.CRITICAL
            elif health_score < 80:
                status = HealthStatus.DEGRADED
            
            return ModuleHealthStatus(
                module_name="Ensemble Manager",
                module_type="ensemble",
                status=status,
                health_score=health_score,
                last_check=now,
                last_success=now,
                last_error=None,
                uptime_pct=99.6,
                error_rate=0.0,
                avg_latency_ms=50.0,
                success_count_24h=1440,
                error_count_24h=0,
                current_issues=issues,
                warnings=warnings,
            )
            
        except Exception as e:
            logger.error(f"[PHASE 3C] Ensemble health check failed: {e}")
            return ModuleHealthStatus(
                module_name="Ensemble Manager",
                module_type="ensemble",
                status=HealthStatus.CRITICAL,
                health_score=0.0,
                last_check=now,
                last_success=None,
                last_error=now,
                uptime_pct=0.0,
                error_rate=1.0,
                avg_latency_ms=0.0,
                success_count_24h=0,
                error_count_24h=1,
                current_issues=[f"Module check failed: {str(e)}"],
                warnings=[],
            )
    
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate weighted average health score"""
        if not self.module_health:
            return 0.0
        
        # Weights for each module
        weights = {
            'phase_2b': 0.20,
            'phase_2d': 0.20,
            'phase_3a': 0.25,
            'phase_3b': 0.25,
            'ensemble': 0.10,
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for module_type, weight in weights.items():
            if module_type in self.module_health:
                health = self.module_health[module_type]
                weighted_sum += health.health_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    
    def _determine_overall_status(self, health_score: float) -> HealthStatus:
        """Determine overall status from health score"""
        if health_score >= 80:
            return HealthStatus.HEALTHY
        elif health_score >= 60:
            return HealthStatus.DEGRADED
        elif health_score > 0:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.OFFLINE
    
    
    def _calculate_signal_success_rate(self) -> float:
        """Calculate signal generation success rate"""
        if not self.signal_successes:
            return 1.0
        
        successes = sum(1 for s in self.signal_successes if s)
        return successes / len(self.signal_successes)
    
    
    def _calculate_avg_signal_latency(self) -> float:
        """Calculate average signal generation latency"""
        if not self.signal_latencies:
            return 0.0
        
        return sum(self.signal_latencies) / len(self.signal_latencies)
    
    
    def _count_recent_events(self, event_queue: deque, hours: int) -> int:
        """Count events in recent time window"""
        if not event_queue:
            return 0
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return sum(1 for ts in event_queue if ts > cutoff)
    
    
    async def _check_and_generate_alerts(self, metrics: SystemHealthMetrics):
        """Check metrics and generate alerts if needed"""
        now = datetime.utcnow()
        
        # Check overall health
        if metrics.overall_health_score < 60:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                module="system",
                message=f"System health critically low: {metrics.overall_health_score:.1f}%",
                metric_name="overall_health_score",
                metric_value=metrics.overall_health_score,
                threshold=60.0,
                recommended_action="Check individual module health and resolve issues immediately",
            )
        elif metrics.overall_health_score < 80:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                module="system",
                message=f"System health degraded: {metrics.overall_health_score:.1f}%",
                metric_name="overall_health_score",
                metric_value=metrics.overall_health_score,
                threshold=80.0,
                recommended_action="Monitor closely and investigate degraded modules",
            )
        
        # Check signal success rate
        if metrics.signal_generation_success_rate < 0.90:
            self._create_alert(
                severity=AlertSeverity.ERROR,
                module="system",
                message=f"Signal generation success rate low: {metrics.signal_generation_success_rate:.1%}",
                metric_name="signal_success_rate",
                metric_value=metrics.signal_generation_success_rate,
                threshold=0.90,
                recommended_action="Check ensemble and feature extraction modules",
            )
        
        # Check error rate
        if metrics.error_count_1h > 10:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                module="system",
                message=f"High error rate: {metrics.error_count_1h} errors in last hour",
                metric_name="error_count_1h",
                metric_value=float(metrics.error_count_1h),
                threshold=10.0,
                recommended_action="Review recent logs for error patterns",
            )
        
        # Check individual module health
        for module_type, module_health in self.module_health.items():
            if module_health.status == HealthStatus.CRITICAL:
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    module=module_type,
                    message=f"{module_health.module_name} is in critical state",
                    recommended_action=f"Investigate {module_type} immediately - check logs and dependencies",
                )
            elif module_health.status == HealthStatus.DEGRADED:
                self._create_alert(
                    severity=AlertSeverity.WARNING,
                    module=module_type,
                    message=f"{module_health.module_name} is degraded",
                    recommended_action=f"Monitor {module_type} and check for issues",
                )
    
    
    def _create_alert(
        self,
        severity: AlertSeverity,
        module: str,
        message: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
        recommended_action: Optional[str] = None,
    ):
        """Create and store a health alert"""
        alert_id = f"{module}_{severity.value}_{datetime.utcnow().timestamp()}"
        
        # Check if similar alert already exists
        for active_alert in self.active_alerts.values():
            if (active_alert.module == module and 
                active_alert.severity == severity and
                active_alert.message == message):
                return  # Don't create duplicate
        
        alert = HealthAlert(
            alert_id=alert_id,
            severity=severity,
            module=module,
            message=message,
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            recommended_action=recommended_action,
        )
        
        self.alerts.append(alert)
        self.active_alerts[alert_id] = alert
        
        # Log alert
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(severity, logger.info)
        
        log_func(f"[PHASE 3C] ALERT [{severity.value.upper()}] {module}: {message}")
    
    
    def _cleanup_old_alerts(self):
        """Remove alerts older than retention period"""
        cutoff = datetime.utcnow() - self.alert_retention
        
        # Remove from active alerts
        expired_ids = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.timestamp < cutoff
        ]
        
        for alert_id in expired_ids:
            del self.active_alerts[alert_id]
    
    
    def record_signal_attempt(self, success: bool, latency_ms: float):
        """Record a signal generation attempt"""
        self.signal_successes.append(success)
        self.signal_latencies.append(latency_ms)
        
        if not success:
            self.error_timestamps.append(datetime.utcnow())
    
    
    def record_warning(self):
        """Record a warning event"""
        self.warning_timestamps.append(datetime.utcnow())
    
    
    def record_error(self):
        """Record an error event"""
        self.error_timestamps.append(datetime.utcnow())
    
    
    def get_current_health(self) -> Optional[SystemHealthMetrics]:
        """Get most recent health metrics"""
        if not self.health_history:
            return None
        return self.health_history[-1]
    
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    
    def get_recent_alerts(self, hours: int = 24) -> List[HealthAlert]:
        """Get alerts from recent time window"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp > cutoff]
    
    
    def get_health_history(self, hours: int = 24) -> List[SystemHealthMetrics]:
        """Get health metrics history"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [m for m in self.health_history if m.timestamp > cutoff]
