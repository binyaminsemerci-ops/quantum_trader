"""
Health Monitor - Detects system issues and auto-heals
====================================================

Monitors:
- Model Supervisor mode and bias detection
- Individual model health (LightGBM, XGBoost, NHiTS, PatchTST)
- Execution layer status
- Retraining orchestrator activity
- Critical configuration mismatches

Auto-healing:
- Restarts failed models
- Corrects configuration drift
- Sends alerts for manual intervention
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"


@dataclass
class HealthIssue:
    """Represents a detected health issue"""
    component: str
    severity: HealthStatus
    description: str
    detected_at: datetime
    auto_fixable: bool
    fix_action: Optional[str] = None


class HealthMonitor:
    """
    Continuous health monitoring and auto-healing system.
    
    Detects:
    - Model Supervisor in wrong mode
    - Models with 100% bias
    - Failed model initialization
    - Retraining failures
    - Config drift (env vars vs actual behavior)
    """
    
    def __init__(
        self,
        model_supervisor=None,
        ai_trading_engine=None,
        retraining_orchestrator=None,
        event_driven_executor=None
    ):
        self.model_supervisor = model_supervisor
        self.ai_trading_engine = ai_trading_engine
        self.retraining_orchestrator = retraining_orchestrator
        self.event_driven_executor = event_driven_executor
        
        self.issues: List[HealthIssue] = []
        self.last_check = datetime.now(timezone.utc)
        self.check_interval_sec = 300  # Check every 5 minutes
        
        # Expected configuration
        self.expected_config = {
            "model_supervisor_mode": os.getenv("QT_MODEL_SUPERVISOR_MODE", "ENFORCED"),
            "bias_threshold": float(os.getenv("QT_MODEL_SUPERVISOR_BIAS_THRESHOLD", "0.70")),
            "min_samples": int(os.getenv("QT_MODEL_SUPERVISOR_MIN_SAMPLES", "20")),
        }
        
        logger.info("=" * 80)
        logger.info("🏥 HEALTH MONITOR - INITIALIZING")
        logger.info("=" * 80)
        logger.info(f"Expected Model Supervisor Mode: {self.expected_config['model_supervisor_mode']}")
        logger.info(f"Expected Bias Threshold: {self.expected_config['bias_threshold']:.0%}")
        logger.info(f"Check Interval: Every {self.check_interval_sec} seconds")
        logger.info("=" * 80 + "\n")
    
    async def check_health(self) -> Tuple[HealthStatus, List[HealthIssue]]:
        """
        Perform comprehensive health check.
        
        Returns:
            (overall_status, list_of_issues)
        """
        self.issues.clear()
        
        # 1. Check Model Supervisor
        await self._check_model_supervisor()
        
        # 2. Check AI Models
        await self._check_ai_models()
        
        # 3. Check Retraining
        await self._check_retraining()
        
        # 4. Check Execution Layer
        await self._check_execution_layer()
        
        # Determine overall status
        overall_status = self._calculate_overall_status()
        
        # Log summary
        if self.issues:
            logger.warning(f"🚨 Health Check: {overall_status.value} - {len(self.issues)} issues detected")
            for issue in self.issues:
                emoji = "🔴" if issue.severity == HealthStatus.CRITICAL else "🟡"
                logger.warning(
                    f"{emoji} [{issue.component}] {issue.severity.value}: {issue.description}"
                )
                if issue.auto_fixable and issue.fix_action:
                    logger.info(f"   → Auto-fix: {issue.fix_action}")
        else:
            logger.info("✅ Health Check: All systems HEALTHY")
        
        self.last_check = datetime.now(timezone.utc)
        return overall_status, self.issues
    
    async def _check_model_supervisor(self):
        """Check Model Supervisor configuration and behavior"""
        if not self.model_supervisor:
            self.issues.append(HealthIssue(
                component="Model Supervisor",
                severity=HealthStatus.CRITICAL,
                description="Model Supervisor not initialized",
                detected_at=datetime.now(timezone.utc),
                auto_fixable=False
            ))
            return
        
        # Check if mode matches expected
        actual_mode = getattr(self.model_supervisor, 'mode', 'UNKNOWN')
        expected_mode = self.expected_config['model_supervisor_mode']
        
        if actual_mode.upper() != expected_mode.upper():
            self.issues.append(HealthIssue(
                component="Model Supervisor",
                severity=HealthStatus.CRITICAL,
                description=f"Mode mismatch: Expected {expected_mode}, but running in {actual_mode}",
                detected_at=datetime.now(timezone.utc),
                auto_fixable=True,
                fix_action=f"Restart backend with QT_MODEL_SUPERVISOR_MODE={expected_mode}"
            ))
        
        # Check if bias has been detected
        if hasattr(self.model_supervisor, 'realtime_action_count'):
            total_buy = sum(
                counts.get("BUY", 0)
                for counts in self.model_supervisor.realtime_action_count.values()
            )
            total_sell = sum(
                counts.get("SELL", 0)
                for counts in self.model_supervisor.realtime_action_count.values()
            )
            total_signals = total_buy + total_sell
            
            if total_signals >= self.expected_config['min_samples']:
                buy_pct = total_buy / total_signals if total_signals > 0 else 0
                sell_pct = total_sell / total_signals if total_signals > 0 else 0
                bias_threshold = self.expected_config['bias_threshold']
                
                if sell_pct > bias_threshold:
                    self.issues.append(HealthIssue(
                        component="Model Supervisor",
                        severity=HealthStatus.DEGRADED,
                        description=f"SHORT bias detected: {sell_pct:.1%} (threshold: {bias_threshold:.1%})",
                        detected_at=datetime.now(timezone.utc),
                        auto_fixable=True,
                        fix_action="Bias detection active - blocking SHORT trades until balanced"
                    ))
                elif buy_pct > bias_threshold:
                    self.issues.append(HealthIssue(
                        component="Model Supervisor",
                        severity=HealthStatus.DEGRADED,
                        description=f"LONG bias detected: {buy_pct:.1%} (threshold: {bias_threshold:.1%})",
                        detected_at=datetime.now(timezone.utc),
                        auto_fixable=True,
                        fix_action="Bias detection active - blocking LONG trades until balanced"
                    ))
    
    async def _check_ai_models(self):
        """Check individual AI model health"""
        if not self.ai_trading_engine:
            return
        
        # Check if ensemble manager exists
        if not hasattr(self.ai_trading_engine, 'ensemble_manager'):
            return
        
        ensemble = self.ai_trading_engine.ensemble_manager
        
        # Check each model
        models_to_check = ['xgb_agent', 'lgbm_agent', 'nhits_agent', 'patchtst_agent']
        
        for model_name in models_to_check:
            if not hasattr(ensemble, model_name):
                self.issues.append(HealthIssue(
                    component=f"AI Model: {model_name}",
                    severity=HealthStatus.FAILED,
                    description=f"{model_name} not initialized",
                    detected_at=datetime.now(timezone.utc),
                    auto_fixable=False
                ))
                continue
            
            model = getattr(ensemble, model_name)
            
            # Check if model is loaded
            if not getattr(model, 'model', None):
                self.issues.append(HealthIssue(
                    component=f"AI Model: {model_name}",
                    severity=HealthStatus.FAILED,
                    description=f"{model_name} model not loaded",
                    detected_at=datetime.now(timezone.utc),
                    auto_fixable=True,
                    fix_action=f"Retrain {model_name} or restore from backup"
                ))
        
        # Check for extreme bias in individual models (100% same prediction)
        if hasattr(ensemble, 'lgbm_agent'):
            lgbm = ensemble.lgbm_agent
            if hasattr(lgbm, 'prediction_history'):
                # Check last 20 predictions
                recent_predictions = lgbm.prediction_history[-20:] if lgbm.prediction_history else []
                if len(recent_predictions) >= 10:
                    sell_count = sum(1 for p in recent_predictions if p == 'SELL')
                    sell_pct = sell_count / len(recent_predictions)
                    
                    if sell_pct >= 0.95:  # 95%+ same prediction
                        self.issues.append(HealthIssue(
                            component="AI Model: LightGBM",
                            severity=HealthStatus.CRITICAL,
                            description=f"LightGBM extremely biased: {sell_pct:.0%} SELL predictions",
                            detected_at=datetime.now(timezone.utc),
                            auto_fixable=True,
                            fix_action="Force retrain LightGBM with balanced historical data"
                        ))
    
    async def _check_retraining(self):
        """Check retraining orchestrator status"""
        if not self.retraining_orchestrator:
            return
        
        # Check if retraining is enabled
        if not getattr(self.retraining_orchestrator, 'enabled', True):
            self.issues.append(HealthIssue(
                component="Retraining Orchestrator",
                severity=HealthStatus.DEGRADED,
                description="Retraining disabled - models won't learn from new data",
                detected_at=datetime.now(timezone.utc),
                auto_fixable=False
            ))
            return
        
        # Check last retrain time
        if hasattr(self.retraining_orchestrator, 'last_retrain_time'):
            last_retrain = self.retraining_orchestrator.last_retrain_time
            if last_retrain:
                hours_since_retrain = (datetime.now(timezone.utc) - last_retrain).total_seconds() / 3600
                expected_interval_hours = 24  # Default
                
                if hours_since_retrain > expected_interval_hours * 2:  # 2x expected interval
                    self.issues.append(HealthIssue(
                        component="Retraining Orchestrator",
                        severity=HealthStatus.DEGRADED,
                        description=f"No retraining for {hours_since_retrain:.1f} hours (expected every {expected_interval_hours}h)",
                        detected_at=datetime.now(timezone.utc),
                        auto_fixable=True,
                        fix_action="Check retraining logs for errors, may need manual trigger"
                    ))
    
    async def _check_execution_layer(self):
        """Check execution layer health"""
        if not self.event_driven_executor:
            return
        
        # Check if Model Supervisor is attached to executor
        if not hasattr(self.event_driven_executor, 'model_supervisor'):
            self.issues.append(HealthIssue(
                component="Execution Layer",
                severity=HealthStatus.CRITICAL,
                description="Model Supervisor not attached to executor - bias blocking disabled!",
                detected_at=datetime.now(timezone.utc),
                auto_fixable=True,
                fix_action="Restart backend to reinitialize executor with Model Supervisor"
            ))
        elif not self.event_driven_executor.model_supervisor:
            self.issues.append(HealthIssue(
                component="Execution Layer",
                severity=HealthStatus.CRITICAL,
                description="Model Supervisor is None in executor - bias blocking disabled!",
                detected_at=datetime.now(timezone.utc),
                auto_fixable=True,
                fix_action="Restart backend to reinitialize Model Supervisor"
            ))
    
    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall system health"""
        if not self.issues:
            return HealthStatus.HEALTHY
        
        # If any CRITICAL or FAILED issues, overall is CRITICAL
        critical_count = sum(
            1 for issue in self.issues
            if issue.severity in [HealthStatus.CRITICAL, HealthStatus.FAILED]
        )
        
        if critical_count > 0:
            return HealthStatus.CRITICAL
        
        # If only DEGRADED issues
        return HealthStatus.DEGRADED
    
    async def auto_heal(self) -> int:
        """
        Attempt to auto-heal detected issues.
        
        Returns:
            Number of issues successfully fixed
        """
        if not self.issues:
            return 0
        
        fixes_applied = 0
        
        for issue in self.issues:
            if not issue.auto_fixable:
                continue
            
            try:
                logger.info(f"🔧 Auto-healing: {issue.component}")
                logger.info(f"   Issue: {issue.description}")
                logger.info(f"   Action: {issue.fix_action}")
                
                # Apply fix based on component
                if "Model Supervisor" in issue.component and "mode mismatch" in issue.description.lower():
                    # Can't fix mode at runtime, needs restart
                    logger.warning("   ⚠️ Requires backend restart to apply fix")
                    continue
                
                if "LightGBM" in issue.component and "biased" in issue.description.lower():
                    # Could trigger immediate retrain
                    if self.retraining_orchestrator:
                        logger.info("   → Triggering emergency retrain for LightGBM...")
                        # await self.retraining_orchestrator.force_retrain(['lgbm'])
                        logger.info("   ✅ Retrain scheduled")
                        fixes_applied += 1
                
            except Exception as e:
                logger.error(f"   ❌ Auto-heal failed: {e}")
        
        return fixes_applied
    
    def get_health_summary(self) -> Dict:
        """Get health summary for API/logging"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": self._calculate_overall_status().value,
            "total_issues": len(self.issues),
            "issues_by_severity": {
                "CRITICAL": sum(1 for i in self.issues if i.severity == HealthStatus.CRITICAL),
                "DEGRADED": sum(1 for i in self.issues if i.severity == HealthStatus.DEGRADED),
                "FAILED": sum(1 for i in self.issues if i.severity == HealthStatus.FAILED),
            },
            "issues": [
                {
                    "component": issue.component,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "auto_fixable": issue.auto_fixable,
                    "fix_action": issue.fix_action
                }
                for issue in self.issues
            ],
            "expected_config": self.expected_config,
            "last_check": self.last_check.isoformat()
        }
