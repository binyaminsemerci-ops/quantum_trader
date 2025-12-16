"""
Systemic Risk Detector

EPIC-RISK3-001: Detection of market-wide and portfolio-wide risk conditions

Detects:
- Liquidity stress (market depth deterioration)
- Correlation spikes (crisis correlation)
- Multi-exchange failure correlation
- Volatility regime shifts
- Cascading risk (contagion effects)
- Concentration risk

TODO (RISK3-002):
- Add liquidity depth tracking from order books
- Implement real-time correlation monitoring
- Add exchange connectivity health checks
- Enhance with machine learning anomaly detection
- Add credit/counterparty risk monitoring
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

from .models import (
    RiskSnapshot,
    ExposureMatrix,
    VaRResult,
    SystemicRiskSignal,
    RiskLevel,
    SystemicRiskType,
)

logger = logging.getLogger(__name__)


class SystemicRiskDetector:
    """Detect systemic risk events that could trigger portfolio-wide losses"""
    
    def __init__(
        self,
        correlation_spike_threshold: float = 0.20,
        volatility_spike_threshold: float = 2.0,
        liquidity_stress_threshold: float = 0.50,
        concentration_threshold: float = 0.60,
    ):
        """
        Initialize systemic risk detector
        
        Args:
            correlation_spike_threshold: Correlation increase triggering warning (e.g., 0.20 = 20%)
            volatility_spike_threshold: Volatility multiplier triggering warning (e.g., 2.0 = 2x)
            liquidity_stress_threshold: Liquidity score below which stress is flagged
            concentration_threshold: Exposure concentration triggering warning
        """
        self.correlation_spike_threshold = correlation_spike_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.liquidity_stress_threshold = liquidity_stress_threshold
        self.concentration_threshold = concentration_threshold
        
        # State tracking for regime shifts
        self._historical_correlation: List[float] = []
        self._historical_volatility: List[float] = []
        self._correlation_baseline: Optional[float] = None
        self._volatility_baseline: Optional[float] = None
        
        logger.info(
            f"[RISK-V3] SystemicRiskDetector initialized\n"
            f"  Correlation spike threshold: {correlation_spike_threshold:.2%}\n"
            f"  Volatility spike threshold: {volatility_spike_threshold}x\n"
            f"  Liquidity stress threshold: {liquidity_stress_threshold}\n"
            f"  Concentration threshold: {concentration_threshold:.2%}"
        )
    
    def detect(
        self,
        snapshot: RiskSnapshot,
        exposure_matrix: ExposureMatrix,
        var_result: Optional[VaRResult] = None,
        market_state: Optional[Dict] = None,
    ) -> List[SystemicRiskSignal]:
        """
        Detect all systemic risk conditions
        
        Args:
            snapshot: Current risk snapshot
            exposure_matrix: Exposure matrix with correlation data
            var_result: Optional VaR calculation result
            market_state: Optional market state data (volatility, liquidity, etc.)
        
        Returns:
            List of systemic risk signals (empty if no risks detected)
        """
        signals = []
        
        # 1. Correlation spike detection
        corr_signal = self._detect_correlation_spike(exposure_matrix)
        if corr_signal:
            signals.append(corr_signal)
        
        # 2. Concentration risk
        conc_signal = self._detect_concentration_risk(exposure_matrix)
        if conc_signal:
            signals.append(conc_signal)
        
        # 3. Liquidity stress (if market data available)
        if market_state:
            liq_signal = self._detect_liquidity_stress(market_state)
            if liq_signal:
                signals.append(liq_signal)
        
        # 4. Volatility regime shift
        vol_signal = self._detect_volatility_regime_shift(snapshot, var_result)
        if vol_signal:
            signals.append(vol_signal)
        
        # 5. Multi-exchange failure correlation (if multiple exchanges)
        if len(snapshot.exchange_exposure) > 1:
            exchange_signal = self._detect_multi_exchange_risk(snapshot)
            if exchange_signal:
                signals.append(exchange_signal)
        
        # 6. Cascading risk (if high correlation + high leverage)
        cascade_signal = self._detect_cascading_risk(snapshot, exposure_matrix)
        if cascade_signal:
            signals.append(cascade_signal)
        
        if signals:
            logger.warning(
                f"[RISK-V3] 🚨 SYSTEMIC RISK DETECTED: {len(signals)} signals\n" +
                "\n".join(f"  - {s.risk_type}: {s.description}" for s in signals)
            )
        else:
            logger.info("[RISK-V3] ✅ No systemic risk detected")
        
        return signals
    
    def _detect_correlation_spike(
        self,
        exposure_matrix: ExposureMatrix,
    ) -> Optional[SystemicRiskSignal]:
        """
        Detect sudden increase in portfolio correlation
        
        High correlation during stress = all positions move together = poor diversification
        """
        current_corr = exposure_matrix.avg_correlation
        
        # Update baseline if we don't have one
        if self._correlation_baseline is None:
            self._correlation_baseline = current_corr
            self._historical_correlation.append(current_corr)
            return None
        
        # Track historical correlation
        self._historical_correlation.append(current_corr)
        if len(self._historical_correlation) > 100:
            self._historical_correlation.pop(0)
        
        # Update baseline (moving average)
        if len(self._historical_correlation) > 10:
            self._correlation_baseline = np.mean(self._historical_correlation[-30:])
        
        # Check for spike
        correlation_increase = current_corr - self._correlation_baseline
        
        if correlation_increase >= self.correlation_spike_threshold:
            severity = min(correlation_increase / 0.30, 1.0)  # Max out at 30% increase
            
            return SystemicRiskSignal(
                level=RiskLevel.WARNING if severity < 0.75 else RiskLevel.CRITICAL,
                risk_type=SystemicRiskType.CORRELATION_SPIKE,
                description=f"Portfolio correlation spiked from {self._correlation_baseline:.2%} to {current_corr:.2%}",
                severity_score=round(severity, 3),
                factors={
                    "baseline_correlation": round(self._correlation_baseline, 3),
                    "current_correlation": round(current_corr, 3),
                    "increase": round(correlation_increase, 3),
                    "max_correlation": round(exposure_matrix.max_correlation, 3),
                },
                recommended_action="reduce_exposure" if severity >= 0.75 else "monitor",
            )
        
        return None
    
    def _detect_concentration_risk(
        self,
        exposure_matrix: ExposureMatrix,
    ) -> Optional[SystemicRiskSignal]:
        """
        Detect excessive concentration in symbol/exchange/strategy
        
        High concentration = lack of diversification = single point of failure
        """
        # Check HHI concentration metrics
        max_hhi = max(
            exposure_matrix.symbol_concentration_hhi,
            exposure_matrix.exchange_concentration_hhi,
            exposure_matrix.strategy_concentration_hhi,
        )
        
        # Check normalized exposures
        max_symbol_exposure = max(exposure_matrix.normalized_symbol_exposure.values()) if exposure_matrix.normalized_symbol_exposure else 0
        max_exchange_exposure = max(exposure_matrix.normalized_exchange_exposure.values()) if exposure_matrix.normalized_exchange_exposure else 0
        
        # Concentration warning if:
        # - HHI > 0.40 (moderate concentration)
        # - Single symbol > 60%
        # - Single exchange > 80%
        
        if max_hhi > 0.40 or max_symbol_exposure > self.concentration_threshold or max_exchange_exposure > 0.80:
            severity = max(
                max_hhi / 0.50,  # HHI max at 0.50
                max_symbol_exposure / 0.70,  # Symbol max at 70%
                max_exchange_exposure / 0.90,  # Exchange max at 90%
            )
            severity = min(severity, 1.0)
            
            # Identify concentration type
            conc_type = "symbol" if max_symbol_exposure > max_exchange_exposure else "exchange"
            
            return SystemicRiskSignal(
                level=RiskLevel.WARNING if severity < 0.75 else RiskLevel.CRITICAL,
                risk_type=SystemicRiskType.CONCENTRATION_RISK,
                description=f"Excessive {conc_type} concentration detected (HHI: {max_hhi:.3f})",
                severity_score=round(severity, 3),
                factors={
                    "symbol_hhi": round(exposure_matrix.symbol_concentration_hhi, 3),
                    "exchange_hhi": round(exposure_matrix.exchange_concentration_hhi, 3),
                    "max_symbol_exposure": round(max_symbol_exposure, 3),
                    "max_exchange_exposure": round(max_exchange_exposure, 3),
                    "hotspots": len(exposure_matrix.risk_hotspots),
                },
                recommended_action="diversify" if severity >= 0.75 else "monitor",
                affected_symbols=[h["name"] for h in exposure_matrix.risk_hotspots if h["type"] == "symbol"][:3],
                affected_exchanges=[h["name"] for h in exposure_matrix.risk_hotspots if h["type"] == "exchange"][:2],
            )
        
        return None
    
    def _detect_liquidity_stress(
        self,
        market_state: Dict,
    ) -> Optional[SystemicRiskSignal]:
        """
        Detect market liquidity stress
        
        TODO (RISK3-002): Implement real liquidity monitoring from order book depth
        
        Args:
            market_state: Dict with keys like 'liquidity_score', 'bid_ask_spread', etc.
        """
        liquidity_score = market_state.get("liquidity_score", 1.0)
        
        if liquidity_score < self.liquidity_stress_threshold:
            severity = 1.0 - (liquidity_score / self.liquidity_stress_threshold)
            
            return SystemicRiskSignal(
                level=RiskLevel.WARNING if severity < 0.75 else RiskLevel.CRITICAL,
                risk_type=SystemicRiskType.LIQUIDITY_STRESS,
                description=f"Market liquidity stress detected (score: {liquidity_score:.2f})",
                severity_score=round(severity, 3),
                factors={
                    "liquidity_score": round(liquidity_score, 3),
                    "threshold": self.liquidity_stress_threshold,
                    "bid_ask_spread": market_state.get("bid_ask_spread", 0.0),
                },
                recommended_action="reduce_exposure" if severity >= 0.75 else "monitor",
            )
        
        return None
    
    def _detect_volatility_regime_shift(
        self,
        snapshot: RiskSnapshot,
        var_result: Optional[VaRResult],
    ) -> Optional[SystemicRiskSignal]:
        """
        Detect sudden volatility regime shift
        
        Volatility regime shift = market uncertainty = higher risk
        """
        if not var_result:
            return None
        
        current_vol = var_result.portfolio_volatility
        
        # Initialize baseline
        if self._volatility_baseline is None:
            self._volatility_baseline = current_vol
            self._historical_volatility.append(current_vol)
            return None
        
        # Track historical volatility
        self._historical_volatility.append(current_vol)
        if len(self._historical_volatility) > 100:
            self._historical_volatility.pop(0)
        
        # Update baseline
        if len(self._historical_volatility) > 10:
            self._volatility_baseline = np.mean(self._historical_volatility[-30:])
        
        # Check for spike
        if self._volatility_baseline > 0:
            vol_multiplier = current_vol / self._volatility_baseline
            
            if vol_multiplier >= self.volatility_spike_threshold:
                severity = min((vol_multiplier - 1.0) / 2.0, 1.0)  # Max at 3x baseline
                
                return SystemicRiskSignal(
                    level=RiskLevel.WARNING if severity < 0.75 else RiskLevel.CRITICAL,
                    risk_type=SystemicRiskType.VOLATILITY_REGIME_SHIFT,
                    description=f"Volatility spiked {vol_multiplier:.1f}x baseline (from {self._volatility_baseline:.2%} to {current_vol:.2%})",
                    severity_score=round(severity, 3),
                    factors={
                        "baseline_volatility": round(self._volatility_baseline, 4),
                        "current_volatility": round(current_vol, 4),
                        "multiplier": round(vol_multiplier, 2),
                        "regime": snapshot.regime or "unknown",
                    },
                    recommended_action="reduce_leverage" if severity >= 0.75 else "monitor",
                )
        
        return None
    
    def _detect_multi_exchange_risk(
        self,
        snapshot: RiskSnapshot,
    ) -> Optional[SystemicRiskSignal]:
        """
        Detect correlated failures or issues across multiple exchanges
        
        TODO (RISK3-002): Implement exchange health monitoring
        
        For now, just check if we have high exposure on multiple exchanges
        """
        exchanges = list(snapshot.exchange_exposure.keys())
        
        if len(exchanges) >= 2:
            # Check if exposure is relatively balanced (good) or unbalanced (bad)
            exposures = list(snapshot.exchange_exposure.values())
            max_exposure = max(exposures)
            min_exposure = min(exposures)
            
            imbalance = (max_exposure - min_exposure) / snapshot.total_notional if snapshot.total_notional > 0 else 0
            
            # High imbalance across multiple exchanges = multi-exchange risk
            if imbalance > 0.50:
                return SystemicRiskSignal(
                    level=RiskLevel.INFO,
                    risk_type=SystemicRiskType.MULTI_EXCHANGE_FAILURE,
                    description=f"Imbalanced exposure across {len(exchanges)} exchanges",
                    severity_score=min(imbalance, 1.0),
                    factors={
                        "exchange_count": len(exchanges),
                        "imbalance": round(imbalance, 3),
                        "exposure_distribution": {
                            ex: round(exp / snapshot.total_notional, 3)
                            for ex, exp in snapshot.exchange_exposure.items()
                        },
                    },
                    recommended_action="monitor",
                    affected_exchanges=exchanges,
                )
        
        return None
    
    def _detect_cascading_risk(
        self,
        snapshot: RiskSnapshot,
        exposure_matrix: ExposureMatrix,
    ) -> Optional[SystemicRiskSignal]:
        """
        Detect potential for cascading liquidations
        
        Cascading risk = High correlation + High leverage + Low liquidity
        One position liquidation triggers others
        """
        # Conditions for cascading risk:
        # 1. High average correlation (>0.70)
        # 2. High portfolio leverage (>3x)
        # 3. Multiple positions with high leverage
        
        high_corr = exposure_matrix.avg_correlation > 0.70
        high_leverage = snapshot.total_leverage > 3.0
        
        # Count high-leverage positions
        high_lev_positions = sum(
            1 for pos in snapshot.positions
            if pos.leverage > 3.0
        )
        
        if high_corr and high_leverage and high_lev_positions >= 2:
            severity = min(
                exposure_matrix.avg_correlation * snapshot.total_leverage / 5.0,
                1.0
            )
            
            return SystemicRiskSignal(
                level=RiskLevel.CRITICAL,
                risk_type=SystemicRiskType.CASCADING_RISK,
                description=f"Cascading liquidation risk: {high_lev_positions} high-leverage positions with {exposure_matrix.avg_correlation:.2%} correlation",
                severity_score=round(severity, 3),
                factors={
                    "avg_correlation": round(exposure_matrix.avg_correlation, 3),
                    "portfolio_leverage": round(snapshot.total_leverage, 2),
                    "high_leverage_positions": high_lev_positions,
                    "total_positions": len(snapshot.positions),
                },
                recommended_action="reduce_leverage",
                affected_symbols=[pos.symbol for pos in snapshot.positions if pos.leverage > 3.0],
            )
        
        return None
    
    def reset_baselines(self):
        """Reset historical baselines (useful for regime changes or after configuration updates)"""
        self._correlation_baseline = None
        self._volatility_baseline = None
        self._historical_correlation.clear()
        self._historical_volatility.clear()
        logger.info("[RISK-V3] Systemic risk baselines reset")


__all__ = [
    "SystemicRiskDetector",
]
