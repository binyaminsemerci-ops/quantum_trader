"""
Risk Rules Engine

EPIC-RISK3-001: Rule-based risk evaluation and threshold checking

Evaluates:
- Risk limit breaches
- Threshold violations
- Policy compliance
- ESS tier recommendations

TODO (RISK3-002):
- Add configurable rule sets
- Add rule priority system
- Add custom rule definitions
"""

import logging
from typing import List, Tuple, Optional
from enum import Enum

from .models import (
    RiskSnapshot,
    ExposureMatrix,
    VaRResult,
    ESResult,
    SystemicRiskSignal,
    RiskLimits,
    RiskThreshold,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class ESSTier(str, Enum):
    """Emergency Stop System tiers"""
    NORMAL = "NORMAL"
    REDUCED = "REDUCED"
    EMERGENCY = "EMERGENCY"


class RiskRulesEngine:
    """Evaluate risk conditions against rules and recommend actions"""
    
    def __init__(self, risk_limits: RiskLimits):
        """
        Initialize risk rules engine
        
        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits
        logger.info("[RISK-V3] RiskRulesEngine initialized")
    
    def evaluate_all_rules(
        self,
        snapshot: RiskSnapshot,
        exposure_matrix: ExposureMatrix,
        var_result: Optional[VaRResult],
        es_result: Optional[ESResult],
        systemic_signals: List[SystemicRiskSignal],
    ) -> Tuple[RiskLevel, List[RiskThreshold], List[str], List[str]]:
        """
        Evaluate all risk rules and return comprehensive assessment
        
        Args:
            snapshot: Current risk snapshot
            exposure_matrix: Exposure matrix analysis
            var_result: VaR calculation result
            es_result: ES calculation result
            systemic_signals: Systemic risk signals
        
        Returns:
            (overall_risk_level, breached_thresholds, critical_issues, warnings)
        """
        breached_thresholds = []
        critical_issues = []
        warnings = []
        
        # 1. Leverage rules
        leverage_threshold = self._check_leverage(snapshot)
        if leverage_threshold:
            breached_thresholds.append(leverage_threshold)
            if leverage_threshold.severity == RiskLevel.CRITICAL:
                critical_issues.append(f"Leverage {snapshot.total_leverage:.1f}x exceeds limit {self.risk_limits.max_leverage}x")
            else:
                warnings.append(f"Leverage approaching limit: {snapshot.total_leverage:.1f}x / {self.risk_limits.max_leverage}x")
        
        # 2. Drawdown rules
        drawdown_threshold = self._check_drawdown(snapshot)
        if drawdown_threshold:
            breached_thresholds.append(drawdown_threshold)
            if drawdown_threshold.severity == RiskLevel.CRITICAL:
                critical_issues.append(f"Drawdown {snapshot.drawdown_pct:.2%} exceeds limit {self.risk_limits.max_daily_drawdown_pct:.2%}")
            else:
                warnings.append(f"Drawdown approaching limit: {snapshot.drawdown_pct:.2%}")
        
        # 3. Concentration rules
        conc_thresholds = self._check_concentration(exposure_matrix)
        breached_thresholds.extend(conc_thresholds)
        for threshold in conc_thresholds:
            if threshold.severity == RiskLevel.CRITICAL:
                critical_issues.append(f"{threshold.name} breach: {threshold.current_value:.2%}")
            else:
                warnings.append(f"{threshold.name}: {threshold.current_value:.2%}")
        
        # 4. VaR rules
        if var_result:
            var_thresholds = self._check_var(var_result)
            breached_thresholds.extend(var_thresholds)
            for threshold in var_thresholds:
                if threshold.severity == RiskLevel.CRITICAL:
                    critical_issues.append(f"{threshold.name} breach: ${threshold.current_value:,.2f}")
                else:
                    warnings.append(f"{threshold.name}: ${threshold.current_value:,.2f}")
        
        # 5. ES rules
        if es_result:
            es_threshold = self._check_es(es_result)
            if es_threshold:
                breached_thresholds.append(es_threshold)
                if es_threshold.severity == RiskLevel.CRITICAL:
                    critical_issues.append(f"Expected Shortfall ${es_threshold.current_value:,.2f} exceeds limit")
                else:
                    warnings.append(f"Expected Shortfall elevated: ${es_threshold.current_value:,.2f}")
        
        # 6. Correlation rules
        corr_threshold = self._check_correlation(exposure_matrix)
        if corr_threshold:
            breached_thresholds.append(corr_threshold)
            if corr_threshold.severity == RiskLevel.CRITICAL:
                critical_issues.append(f"Correlation {corr_threshold.current_value:.2%} too high")
            else:
                warnings.append(f"Elevated correlation: {corr_threshold.current_value:.2%}")
        
        # 7. Systemic risk rules
        for signal in systemic_signals:
            if signal.level == RiskLevel.CRITICAL:
                critical_issues.append(f"Systemic: {signal.description}")
            elif signal.level == RiskLevel.WARNING:
                warnings.append(f"Systemic: {signal.description}")
        
        # Determine overall risk level
        if critical_issues or any(s.level == RiskLevel.CRITICAL for s in systemic_signals):
            overall_risk_level = RiskLevel.CRITICAL
        elif warnings or any(s.level == RiskLevel.WARNING for s in systemic_signals):
            overall_risk_level = RiskLevel.WARNING
        else:
            overall_risk_level = RiskLevel.INFO
        
        logger.info(
            f"[RISK-V3] Risk rules evaluation: {overall_risk_level}\n"
            f"  Breached thresholds: {len(breached_thresholds)}\n"
            f"  Critical issues: {len(critical_issues)}\n"
            f"  Warnings: {len(warnings)}"
        )
        
        return overall_risk_level, breached_thresholds, critical_issues, warnings
    
    def recommend_ess_tier(
        self,
        risk_level: RiskLevel,
        critical_issues: List[str],
        systemic_signals: List[SystemicRiskSignal],
    ) -> ESSTier:
        """
        Recommend ESS tier based on risk assessment
        
        Args:
            risk_level: Overall risk level
            critical_issues: List of critical issues
            systemic_signals: Systemic risk signals
        
        Returns:
            Recommended ESS tier
        """
        # EMERGENCY: Multiple critical issues or cascading risk
        if risk_level == RiskLevel.CRITICAL and len(critical_issues) >= 2:
            return ESSTier.EMERGENCY
        
        # EMERGENCY: Cascading risk detected
        if any(s.risk_type.value == "cascading_risk" for s in systemic_signals):
            return ESSTier.EMERGENCY
        
        # REDUCED: Single critical issue or multiple warnings
        if risk_level == RiskLevel.CRITICAL or len(critical_issues) >= 1:
            return ESSTier.REDUCED
        
        # REDUCED: High severity systemic signals
        if any(s.severity_score >= 0.75 for s in systemic_signals):
            return ESSTier.REDUCED
        
        # NORMAL: Everything else
        return ESSTier.NORMAL
    
    def _check_leverage(self, snapshot: RiskSnapshot) -> Optional[RiskThreshold]:
        """Check leverage threshold"""
        if snapshot.total_leverage > self.risk_limits.max_leverage:
            severity = RiskLevel.CRITICAL if snapshot.total_leverage > self.risk_limits.max_leverage * 1.2 else RiskLevel.WARNING
            return RiskThreshold(
                name="Max Leverage",
                value=self.risk_limits.max_leverage,
                breached=True,
                current_value=snapshot.total_leverage,
                severity=severity,
            )
        return None
    
    def _check_drawdown(self, snapshot: RiskSnapshot) -> Optional[RiskThreshold]:
        """Check drawdown threshold"""
        if snapshot.drawdown_pct > self.risk_limits.max_daily_drawdown_pct / 100:
            severity = RiskLevel.CRITICAL if snapshot.drawdown_pct > self.risk_limits.max_daily_drawdown_pct / 100 * 1.2 else RiskLevel.WARNING
            return RiskThreshold(
                name="Max Daily Drawdown",
                value=self.risk_limits.max_daily_drawdown_pct / 100,
                breached=True,
                current_value=snapshot.drawdown_pct,
                severity=severity,
            )
        return None
    
    def _check_concentration(self, exposure_matrix: ExposureMatrix) -> List[RiskThreshold]:
        """Check concentration thresholds"""
        thresholds = []
        
        # Symbol concentration
        if exposure_matrix.normalized_symbol_exposure:
            max_symbol_exp = max(exposure_matrix.normalized_symbol_exposure.values())
            if max_symbol_exp > self.risk_limits.max_symbol_concentration:
                severity = RiskLevel.CRITICAL if max_symbol_exp > self.risk_limits.max_symbol_concentration * 1.2 else RiskLevel.WARNING
                thresholds.append(RiskThreshold(
                    name="Max Symbol Concentration",
                    value=self.risk_limits.max_symbol_concentration,
                    breached=True,
                    current_value=max_symbol_exp,
                    severity=severity,
                ))
        
        # Exchange concentration
        if exposure_matrix.normalized_exchange_exposure:
            max_exchange_exp = max(exposure_matrix.normalized_exchange_exposure.values())
            if max_exchange_exp > self.risk_limits.max_exchange_concentration:
                severity = RiskLevel.CRITICAL if max_exchange_exp > self.risk_limits.max_exchange_concentration * 1.2 else RiskLevel.WARNING
                thresholds.append(RiskThreshold(
                    name="Max Exchange Concentration",
                    value=self.risk_limits.max_exchange_concentration,
                    breached=True,
                    current_value=max_exchange_exp,
                    severity=severity,
                ))
        
        return thresholds
    
    def _check_var(self, var_result: VaRResult) -> List[RiskThreshold]:
        """Check VaR thresholds"""
        thresholds = []
        
        if not var_result.pass_95:
            severity = RiskLevel.CRITICAL if var_result.var_95 > var_result.threshold_95 * 1.5 else RiskLevel.WARNING
            thresholds.append(RiskThreshold(
                name="VaR 95%",
                value=var_result.threshold_95,
                breached=True,
                current_value=var_result.var_95,
                severity=severity,
            ))
        
        if not var_result.pass_99:
            severity = RiskLevel.CRITICAL if var_result.var_99 > var_result.threshold_99 * 1.5 else RiskLevel.WARNING
            thresholds.append(RiskThreshold(
                name="VaR 99%",
                value=var_result.threshold_99,
                breached=True,
                current_value=var_result.var_99,
                severity=severity,
            ))
        
        return thresholds
    
    def _check_es(self, es_result: ESResult) -> Optional[RiskThreshold]:
        """Check Expected Shortfall threshold"""
        if not es_result.pass_975:
            severity = RiskLevel.CRITICAL if es_result.es_975 > es_result.threshold_975 * 1.5 else RiskLevel.WARNING
            return RiskThreshold(
                name="Expected Shortfall 97.5%",
                value=es_result.threshold_975,
                breached=True,
                current_value=es_result.es_975,
                severity=severity,
            )
        return None
    
    def _check_correlation(self, exposure_matrix: ExposureMatrix) -> Optional[RiskThreshold]:
        """Check correlation threshold"""
        if exposure_matrix.avg_correlation > self.risk_limits.max_correlation:
            severity = RiskLevel.CRITICAL if exposure_matrix.avg_correlation > self.risk_limits.max_correlation * 1.2 else RiskLevel.WARNING
            return RiskThreshold(
                name="Max Portfolio Correlation",
                value=self.risk_limits.max_correlation,
                breached=True,
                current_value=exposure_matrix.avg_correlation,
                severity=severity,
            )
        return None


__all__ = [
    "RiskRulesEngine",
    "ESSTier",
]
