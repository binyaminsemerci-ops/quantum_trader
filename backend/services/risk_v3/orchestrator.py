"""
Risk Orchestrator - Core Risk Evaluation Engine

EPIC-RISK3-001: Central orchestrator for all risk evaluation

Coordinates:
- Exposure Matrix Engine
- VaR/ES Engine
- Systemic Risk Detector
- Risk Rules Engine
- ESS v3 Integration
- Federation AI CRO Integration

Event Publishing:
- risk.global_snapshot
- risk.var_es_updated
- risk.systemic_alert
- risk.exposure_matrix_updated
- risk.threshold_breach
"""

import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime

from .models import (
    RiskSnapshot,
    ExposureMatrix,
    VaRResult,
    ESResult,
    SystemicRiskSignal,
    GlobalRiskSignal,
    RiskLevel,
    RiskLimits,
)
from .exposure_matrix import ExposureMatrixEngine
from .var_es import VaRESEngine
from .systemic import SystemicRiskDetector
from .rules import RiskRulesEngine, ESSTier
from .adapters import AdapterFactory

logger = logging.getLogger(__name__)


class RiskOrchestrator:
    """
    Central orchestrator for global risk evaluation
    
    This is the main entry point for Risk v3 system.
    Coordinates all risk engines and publishes events.
    """
    
    def __init__(
        self,
        exposure_engine: Optional[ExposureMatrixEngine] = None,
        var_es_engine: Optional[VaRESEngine] = None,
        systemic_detector: Optional[SystemicRiskDetector] = None,
        risk_limits: Optional[RiskLimits] = None,
        event_bus = None,  # EventBus for publishing events
    ):
        """
        Initialize Risk Orchestrator
        
        Args:
            exposure_engine: Exposure matrix computation engine
            var_es_engine: VaR/ES calculation engine
            systemic_detector: Systemic risk detector
            risk_limits: Risk limits configuration
            event_bus: EventBus for publishing risk events
        """
        # Initialize engines
        self.exposure_engine = exposure_engine or ExposureMatrixEngine()
        self.var_es_engine = var_es_engine or VaRESEngine()
        self.systemic_detector = systemic_detector or SystemicRiskDetector()
        
        # Initialize adapters
        self.adapter_factory = AdapterFactory()
        self.portfolio_adapter = self.adapter_factory.get_portfolio_adapter()
        self.market_data_adapter = self.adapter_factory.get_market_data_adapter()
        self.policy_store_adapter = self.adapter_factory.get_policy_store_adapter()
        self.federation_ai_adapter = self.adapter_factory.get_federation_ai_adapter()
        
        # Initialize rules engine (will load limits from PolicyStore on first evaluation)
        self.risk_limits = risk_limits
        self.rules_engine = None  # Initialized on first evaluation
        
        # Event bus for publishing
        self.event_bus = event_bus
        
        # State tracking
        self.last_evaluation: Optional[GlobalRiskSignal] = None
        self.evaluation_count = 0
        
        logger.info("[RISK-V3] 🎯 RiskOrchestrator initialized")
        logger.info("  ✅ Exposure Matrix Engine")
        logger.info("  ✅ VaR/ES Engine")
        logger.info("  ✅ Systemic Risk Detector")
        logger.info("  ✅ Adapters (Portfolio, Market Data, PolicyStore, Federation AI)")
    
    async def evaluate_risk(
        self,
        force_refresh: bool = False,
    ) -> GlobalRiskSignal:
        """
        Perform complete risk evaluation
        
        This is the main entry point for risk evaluation.
        Executes the full risk pipeline:
        1. Fetch portfolio snapshot
        2. Compute exposure matrix
        3. Calculate VaR/ES
        4. Detect systemic risks
        5. Evaluate rules and thresholds
        6. Generate global risk signal
        7. Publish events
        8. Integrate with ESS v3 and Federation AI
        
        Args:
            force_refresh: Force refresh of all data sources
        
        Returns:
            GlobalRiskSignal with complete risk assessment
        """
        logger.info(f"[RISK-V3] 🎯 Starting global risk evaluation (iteration {self.evaluation_count + 1})")
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Load risk limits from PolicyStore
            if not self.risk_limits or force_refresh:
                self.risk_limits = await self.policy_store_adapter.get_risk_limits()
                self.rules_engine = RiskRulesEngine(self.risk_limits)
                logger.info("[RISK-V3] ✅ Risk limits loaded from PolicyStore")
            
            # Step 2: Fetch portfolio snapshot
            logger.info("[RISK-V3] 📊 Fetching portfolio snapshot...")
            snapshot = await self.portfolio_adapter.get_snapshot()
            logger.info(
                f"[RISK-V3] ✅ Snapshot: {len(snapshot.positions)} positions, "
                f"${snapshot.total_notional:,.2f} notional, {snapshot.total_leverage:.1f}x leverage"
            )
            
            # Step 3: Fetch market data for returns (for VaR/ES and correlation)
            logger.info("[RISK-V3] 📈 Fetching market data...")
            symbols = list(snapshot.symbol_exposure.keys()) if snapshot.symbol_exposure else []
            returns_data = await self.market_data_adapter.get_returns_data(
                symbols=symbols,
                lookback_periods=30,
            )
            market_state = await self.market_data_adapter.get_market_state()
            
            # Step 4: Compute exposure matrix
            logger.info("[RISK-V3] 🔍 Computing exposure matrix...")
            exposure_matrix = self.exposure_engine.compute_exposure_matrix(
                snapshot=snapshot,
                returns_data=returns_data,
            )
            logger.info(
                f"[RISK-V3] ✅ Exposure matrix: HHI={exposure_matrix.symbol_concentration_hhi:.3f}, "
                f"Corr={exposure_matrix.avg_correlation:.3f}, Hotspots={len(exposure_matrix.risk_hotspots)}"
            )
            
            # Step 5: Calculate VaR/ES
            logger.info("[RISK-V3] 📊 Calculating VaR/ES...")
            var_result = self.var_es_engine.compute_var_result(
                snapshot=snapshot,
                returns_data=returns_data,
                method="delta_normal",
                threshold_95=self.risk_limits.var_95_limit,
                threshold_99=self.risk_limits.var_99_limit,
            )
            
            es_result = self.var_es_engine.compute_es_result(
                snapshot=snapshot,
                returns_data=returns_data,
                method="historical",
                threshold_975=self.risk_limits.es_975_limit,
            )
            logger.info(
                f"[RISK-V3] ✅ VaR/ES: 95%=${var_result.var_95:,.2f}, "
                f"99%=${var_result.var_99:,.2f}, ES=${es_result.es_975:,.2f}"
            )
            
            # Step 6: Detect systemic risks
            logger.info("[RISK-V3] 🚨 Detecting systemic risks...")
            systemic_signals = self.systemic_detector.detect(
                snapshot=snapshot,
                exposure_matrix=exposure_matrix,
                var_result=var_result,
                market_state=market_state,
            )
            logger.info(f"[RISK-V3] ✅ Systemic risks: {len(systemic_signals)} signals")
            
            # Step 7: Evaluate rules and thresholds
            logger.info("[RISK-V3] ⚖️ Evaluating risk rules...")
            risk_level, breached_thresholds, critical_issues, warnings = self.rules_engine.evaluate_all_rules(
                snapshot=snapshot,
                exposure_matrix=exposure_matrix,
                var_result=var_result,
                es_result=es_result,
                systemic_signals=systemic_signals,
            )
            logger.info(
                f"[RISK-V3] ✅ Rules: {risk_level}, "
                f"{len(critical_issues)} critical, {len(warnings)} warnings"
            )
            
            # Step 8: Recommend ESS tier
            ess_tier = self.rules_engine.recommend_ess_tier(
                risk_level=risk_level,
                critical_issues=critical_issues,
                systemic_signals=systemic_signals,
            )
            ess_action_required = ess_tier != ESSTier.NORMAL
            logger.info(f"[RISK-V3] 🛡️ ESS Recommendation: {ess_tier.value}")
            
            # Step 9: Calculate overall risk score
            overall_risk_score = self._calculate_overall_risk_score(
                snapshot=snapshot,
                exposure_matrix=exposure_matrix,
                var_result=var_result,
                es_result=es_result,
                systemic_signals=systemic_signals,
                breached_thresholds=breached_thresholds,
            )
            
            # Step 10: Generate risk summary
            risk_summary = self._generate_risk_summary(
                snapshot=snapshot,
                risk_level=risk_level,
                critical_issues=critical_issues,
                warnings=warnings,
                systemic_signals=systemic_signals,
            )
            
            # Step 11: Create global risk signal
            global_signal = GlobalRiskSignal(
                timestamp=datetime.utcnow(),
                risk_level=risk_level,
                snapshot=snapshot,
                exposure_matrix=exposure_matrix,
                var_result=var_result,
                es_result=es_result,
                systemic_signals=systemic_signals,
                overall_risk_score=overall_risk_score,
                ess_tier_recommendation=ess_tier.value,
                ess_action_required=ess_action_required,
                cro_alert_sent=False,
                cro_approval_required=False,
                risk_summary=risk_summary,
                critical_issues=critical_issues,
                warnings=warnings,
            )
            
            # Step 12: Publish events
            await self._publish_events(global_signal)
            
            # Step 13: Federation AI CRO integration
            if risk_level == RiskLevel.CRITICAL or len(critical_issues) >= 2:
                await self._notify_federation_ai_cro(global_signal)
                global_signal.cro_alert_sent = True
            
            # Step 14: Update state
            self.last_evaluation = global_signal
            self.evaluation_count += 1
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"[RISK-V3] ✅ Risk evaluation complete in {duration:.2f}s\n"
                f"  Risk Level: {risk_level.value}\n"
                f"  Risk Score: {overall_risk_score:.2f}\n"
                f"  ESS Tier: {ess_tier.value}\n"
                f"  Critical Issues: {len(critical_issues)}\n"
                f"  Warnings: {len(warnings)}\n"
                f"  Systemic Signals: {len(systemic_signals)}"
            )
            
            return global_signal
        
        except Exception as e:
            logger.error(f"[RISK-V3] ❌ Risk evaluation failed: {e}", exc_info=True)
            raise
    
    def _calculate_overall_risk_score(
        self,
        snapshot: RiskSnapshot,
        exposure_matrix: ExposureMatrix,
        var_result: VaRResult,
        es_result: ESResult,
        systemic_signals: List[SystemicRiskSignal],
        breached_thresholds: List,
    ) -> float:
        """
        Calculate overall risk score (0-1)
        
        Combines multiple risk factors into single score:
        - Leverage utilization
        - Concentration risk
        - VaR/ES violations
        - Systemic risk severity
        - Threshold breaches
        """
        scores = []
        
        # Leverage score
        if self.risk_limits:
            leverage_util = snapshot.total_leverage / self.risk_limits.max_leverage
            scores.append(min(leverage_util, 1.0))
        
        # Concentration score (HHI)
        scores.append(exposure_matrix.symbol_concentration_hhi)
        
        # Correlation score
        scores.append(exposure_matrix.avg_correlation)
        
        # VaR breach score
        if var_result:
            var_score = 0.0
            if not var_result.pass_95:
                var_score = 0.5
            if not var_result.pass_99:
                var_score = 0.75
            scores.append(var_score)
        
        # ES breach score
        if es_result and not es_result.pass_975:
            scores.append(0.80)
        
        # Systemic risk score (max severity)
        if systemic_signals:
            max_systemic = max(s.severity_score for s in systemic_signals)
            scores.append(max_systemic)
        
        # Threshold breach score
        if breached_thresholds:
            critical_count = sum(1 for t in breached_thresholds if t.severity == RiskLevel.CRITICAL)
            breach_score = min(critical_count * 0.25, 1.0)
            scores.append(breach_score)
        
        # Overall score is weighted average
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        return min(round(overall_score, 3), 1.0)
    
    def _generate_risk_summary(
        self,
        snapshot: RiskSnapshot,
        risk_level: RiskLevel,
        critical_issues: List[str],
        warnings: List[str],
        systemic_signals: List[SystemicRiskSignal],
    ) -> str:
        """Generate human-readable risk summary"""
        if risk_level == RiskLevel.INFO:
            return f"Risk profile normal. Portfolio: {len(snapshot.positions)} positions, ${snapshot.total_notional:,.0f} notional, {snapshot.total_leverage:.1f}x leverage."
        
        parts = [f"Risk level: {risk_level.value}."]
        
        if critical_issues:
            parts.append(f"Critical: {', '.join(critical_issues[:2])}")
        
        if warnings:
            parts.append(f"Warnings: {len(warnings)} active")
        
        if systemic_signals:
            critical_systemic = [s for s in systemic_signals if s.level == RiskLevel.CRITICAL]
            if critical_systemic:
                parts.append(f"Systemic: {critical_systemic[0].description}")
        
        return " ".join(parts)
    
    async def _publish_events(self, signal: GlobalRiskSignal):
        """Publish risk events to EventBus"""
        if not self.event_bus:
            logger.debug("[RISK-V3] EventBus not configured, skipping event publishing")
            return
        
        try:
            # Publish global snapshot event
            await self.event_bus.publish(
                "risk.global_snapshot",
                {
                    "timestamp": signal.timestamp.isoformat(),
                    "risk_level": signal.risk_level.value,
                    "risk_score": signal.overall_risk_score,
                    "positions": len(signal.snapshot.positions),
                    "total_notional": signal.snapshot.total_notional,
                    "leverage": signal.snapshot.total_leverage,
                }
            )
            
            # Publish VaR/ES event
            if signal.var_result:
                await self.event_bus.publish(
                    "risk.var_es_updated",
                    {
                        "timestamp": signal.timestamp.isoformat(),
                        "var_95": signal.var_result.var_95,
                        "var_99": signal.var_result.var_99,
                        "es_975": signal.es_result.es_975 if signal.es_result else 0.0,
                        "pass_95": signal.var_result.pass_95,
                        "pass_99": signal.var_result.pass_99,
                    }
                )
            
            # Publish exposure matrix event
            await self.event_bus.publish(
                "risk.exposure_matrix_updated",
                {
                    "timestamp": signal.timestamp.isoformat(),
                    "symbol_hhi": signal.exposure_matrix.symbol_concentration_hhi,
                    "avg_correlation": signal.exposure_matrix.avg_correlation,
                    "hotspots": len(signal.exposure_matrix.risk_hotspots),
                }
            )
            
            # Publish systemic alerts
            for systemic_signal in signal.systemic_signals:
                if systemic_signal.level in [RiskLevel.WARNING, RiskLevel.CRITICAL]:
                    await self.event_bus.publish(
                        "risk.systemic_alert",
                        {
                            "timestamp": systemic_signal.timestamp.isoformat(),
                            "level": systemic_signal.level.value,
                            "risk_type": systemic_signal.risk_type.value,
                            "description": systemic_signal.description,
                            "severity": systemic_signal.severity_score,
                        }
                    )
            
            # Publish threshold breach events
            if signal.critical_issues:
                await self.event_bus.publish(
                    "risk.threshold_breach",
                    {
                        "timestamp": signal.timestamp.isoformat(),
                        "critical_count": len(signal.critical_issues),
                        "warning_count": len(signal.warnings),
                        "issues": signal.critical_issues,
                    }
                )
            
            logger.info("[RISK-V3] ✅ Events published to EventBus")
        
        except Exception as e:
            logger.error(f"[RISK-V3] ❌ Event publishing failed: {e}")
    
    async def _notify_federation_ai_cro(self, signal: GlobalRiskSignal):
        """Notify Federation AI CRO of critical risk"""
        try:
            await self.federation_ai_adapter.send_cro_alert(
                risk_level=signal.risk_level.value,
                description=signal.risk_summary,
                metrics={
                    "risk_score": signal.overall_risk_score,
                    "leverage": signal.snapshot.total_leverage,
                    "critical_issues": signal.critical_issues,
                    "systemic_signals": len(signal.systemic_signals),
                }
            )
            logger.info("[RISK-V3] ✅ Federation AI CRO notified")
        except Exception as e:
            logger.error(f"[RISK-V3] ❌ CRO notification failed: {e}")
    
    async def get_status(self) -> Dict:
        """Get orchestrator status"""
        return {
            "evaluation_count": self.evaluation_count,
            "last_evaluation": self.last_evaluation.timestamp.isoformat() if self.last_evaluation else None,
            "last_risk_level": self.last_evaluation.risk_level.value if self.last_evaluation else None,
            "engines": {
                "exposure_matrix": "active",
                "var_es": "active",
                "systemic_detector": "active",
                "rules_engine": "active" if self.rules_engine else "pending",
            }
        }


__all__ = [
    "RiskOrchestrator",
]
