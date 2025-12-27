"""
Exposure Matrix Engine

EPIC-RISK3-001: Multi-dimensional exposure analysis

Computes:
- Symbol-level exposure and concentration
- Exchange-level exposure distribution
- Strategy-level exposure allocation
- Cross-asset correlation matrix
- Beta-weighted exposure (placeholder for future enhancement)
- Risk hotspot identification

TODO (RISK3-002):
- Implement real correlation computation from historical returns
- Add beta calculation vs benchmark (BTC)
- Enhance with rolling window correlation tracking
- Add correlation regime detection (normal vs crisis)
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict

from .models import (
    RiskSnapshot,
    ExposureMatrix,
    CorrelationMatrix,
    PositionExposure,
)

logger = logging.getLogger(__name__)


class ExposureMatrixEngine:
    """Compute multi-dimensional exposure matrix and concentration metrics"""
    
    def __init__(
        self,
        correlation_lookback: int = 30,
        correlation_method: str = "pearson",
    ):
        """
        Initialize exposure matrix engine
        
        Args:
            correlation_lookback: Number of periods for correlation calculation
            correlation_method: Correlation method (pearson, spearman, kendall)
        """
        self.correlation_lookback = correlation_lookback
        self.correlation_method = correlation_method
        
        logger.info(f"[RISK-V3] ExposureMatrixEngine initialized (lookback={correlation_lookback})")
    
    def compute_exposure_matrix(
        self,
        snapshot: RiskSnapshot,
        returns_data: Optional[Dict[str, List[float]]] = None,
    ) -> ExposureMatrix:
        """
        Compute complete exposure matrix from risk snapshot
        
        Args:
            snapshot: Current risk snapshot with positions
            returns_data: Optional returns data for correlation calc
                         Format: {"BTCUSDT": [0.02, -0.01, 0.03, ...], ...}
        
        Returns:
            ExposureMatrix with all exposure metrics
        """
        logger.info(f"[RISK-V3] Computing exposure matrix for {len(snapshot.positions)} positions")
        
        # 1. Compute normalized exposures
        normalized_symbol = self._normalize_exposure(snapshot.symbol_exposure, snapshot.total_notional)
        normalized_exchange = self._normalize_exposure(snapshot.exchange_exposure, snapshot.total_notional)
        normalized_strategy = self._normalize_exposure(snapshot.strategy_exposure, snapshot.total_notional)
        
        # 2. Compute concentration metrics (HHI)
        symbol_hhi = self._compute_hhi(list(normalized_symbol.values()))
        exchange_hhi = self._compute_hhi(list(normalized_exchange.values()))
        strategy_hhi = self._compute_hhi(list(normalized_strategy.values()))
        
        # 3. Identify risk hotspots
        hotspots = self._identify_hotspots(
            snapshot,
            normalized_symbol,
            normalized_exchange,
            normalized_strategy,
        )
        
        # 4. Compute correlation matrix (if returns data available)
        correlation_matrix = None
        avg_correlation = 0.0
        max_correlation = 0.0
        
        if returns_data and len(returns_data) > 1:
            correlation_matrix = self._compute_correlation_matrix(returns_data)
            if correlation_matrix:
                avg_correlation, max_correlation = self._extract_correlation_stats(
                    correlation_matrix.matrix
                )
        
        # 5. Beta weights (placeholder - would require benchmark data)
        beta_weights = self._compute_beta_weights_placeholder(snapshot.symbol_exposure)
        
        result = ExposureMatrix(
            timestamp=datetime.utcnow(),
            correlation_matrix=correlation_matrix,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            beta_weights=beta_weights,
            normalized_symbol_exposure=normalized_symbol,
            normalized_exchange_exposure=normalized_exchange,
            normalized_strategy_exposure=normalized_strategy,
            risk_hotspots=hotspots,
            symbol_concentration_hhi=symbol_hhi,
            exchange_concentration_hhi=exchange_hhi,
            strategy_concentration_hhi=strategy_hhi,
        )
        
        logger.info(
            f"[RISK-V3] Exposure matrix computed:\n"
            f"  Symbol HHI: {symbol_hhi:.3f}\n"
            f"  Exchange HHI: {exchange_hhi:.3f}\n"
            f"  Avg Correlation: {avg_correlation:.3f}\n"
            f"  Hotspots: {len(hotspots)}"
        )
        
        return result
    
    def _normalize_exposure(
        self,
        exposure_dict: Dict[str, float],
        total: float,
    ) -> Dict[str, float]:
        """Normalize exposure to 0-1 scale"""
        if total <= 0:
            return {k: 0.0 for k in exposure_dict}
        
        return {k: v / total for k, v in exposure_dict.items()}
    
    def _compute_hhi(self, shares: List[float]) -> float:
        """
        Compute Herfindahl-Hirschman Index (HHI) for concentration
        
        HHI = sum(share_i^2)
        
        - HHI close to 0: Highly diversified
        - HHI close to 1: Highly concentrated
        
        Args:
            shares: List of normalized shares (0-1)
        
        Returns:
            HHI value (0-1)
        """
        if not shares:
            return 0.0
        
        return sum(s**2 for s in shares)
    
    def _identify_hotspots(
        self,
        snapshot: RiskSnapshot,
        normalized_symbol: Dict[str, float],
        normalized_exchange: Dict[str, float],
        normalized_strategy: Dict[str, float],
        threshold: float = 0.30,  # 30% exposure threshold
    ) -> List[Dict]:
        """
        Identify risk hotspots (concentrated exposures)
        
        A hotspot is any exposure > threshold with high risk score
        """
        hotspots = []
        
        # Symbol hotspots
        for symbol, exposure_pct in normalized_symbol.items():
            if exposure_pct >= threshold:
                risk_score = self._calculate_hotspot_risk_score(
                    exposure_pct,
                    snapshot.symbol_leverage.get(symbol, 1.0),
                )
                hotspots.append({
                    "type": "symbol",
                    "name": symbol,
                    "exposure_pct": round(exposure_pct, 4),
                    "risk_score": round(risk_score, 4),
                    "leverage": round(snapshot.symbol_leverage.get(symbol, 1.0), 2),
                })
        
        # Exchange hotspots
        for exchange, exposure_pct in normalized_exchange.items():
            if exposure_pct >= threshold:
                position_count = snapshot.exchange_position_count.get(exchange, 0)
                risk_score = exposure_pct * (1.0 + 0.1 * position_count)  # More positions = more risk
                hotspots.append({
                    "type": "exchange",
                    "name": exchange,
                    "exposure_pct": round(exposure_pct, 4),
                    "risk_score": min(round(risk_score, 4), 1.0),
                    "position_count": position_count,
                })
        
        # Strategy hotspots
        for strategy, exposure_pct in normalized_strategy.items():
            if exposure_pct >= threshold:
                risk_score = exposure_pct  # Simple risk score for now
                hotspots.append({
                    "type": "strategy",
                    "name": strategy,
                    "exposure_pct": round(exposure_pct, 4),
                    "risk_score": round(risk_score, 4),
                })
        
        # Sort by risk score descending
        hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return hotspots
    
    def _calculate_hotspot_risk_score(
        self,
        exposure_pct: float,
        leverage: float,
    ) -> float:
        """
        Calculate risk score for a hotspot
        
        Risk score considers:
        - Exposure percentage (higher = more risk)
        - Leverage (higher = more risk)
        
        Returns value 0-1
        """
        # Base risk from exposure
        base_risk = exposure_pct
        
        # Leverage multiplier (leverage > 1 increases risk exponentially)
        leverage_factor = 1.0 + (leverage - 1.0) * 0.2  # 20% increase per 1x leverage
        
        risk_score = base_risk * leverage_factor
        
        return min(risk_score, 1.0)
    
    def _compute_correlation_matrix(
        self,
        returns_data: Dict[str, List[float]],
    ) -> Optional[CorrelationMatrix]:
        """
        Compute correlation matrix from returns data
        
        TODO (RISK3-002): Replace placeholder with real correlation calculation
        using pandas or numpy with proper handling of:
        - Missing data
        - Different lengths
        - Rolling windows
        - Correlation regimes
        
        Args:
            returns_data: Dict of symbol -> returns list
        
        Returns:
            CorrelationMatrix or None if insufficient data
        """
        if len(returns_data) < 2:
            return None
        
        symbols = sorted(returns_data.keys())
        n = len(symbols)
        
        # Placeholder: Generate synthetic correlation matrix
        # In RISK3-002, this will use actual return data
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    corr = 1.0  # Perfect self-correlation
                else:
                    # Placeholder: Assume moderate correlation
                    # Real implementation will compute from returns
                    corr = 0.45 + np.random.uniform(-0.15, 0.15)
                    corr = max(-1.0, min(1.0, corr))
                row.append(round(corr, 4))
            matrix.append(row)
        
        return CorrelationMatrix(
            symbols=symbols,
            matrix=matrix,
            lookback_periods=self.correlation_lookback,
            method=self.correlation_method,
        )
    
    def _extract_correlation_stats(
        self,
        matrix: List[List[float]],
    ) -> Tuple[float, float]:
        """
        Extract average and maximum correlation from matrix
        
        Args:
            matrix: NxN correlation matrix
        
        Returns:
            (average_correlation, max_correlation)
        """
        n = len(matrix)
        if n == 0:
            return 0.0, 0.0
        
        correlations = []
        max_corr = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle only (avoid double counting)
                corr = abs(matrix[i][j])  # Use absolute correlation
                correlations.append(corr)
                max_corr = max(max_corr, corr)
        
        avg_corr = sum(correlations) / len(correlations) if correlations else 0.0
        
        return round(avg_corr, 4), round(max_corr, 4)
    
    def _compute_beta_weights_placeholder(
        self,
        symbol_exposure: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Placeholder for beta weight calculation
        
        TODO (RISK3-002): Implement real beta calculation:
        - Fetch benchmark returns (e.g., BTCUSDT)
        - Compute beta for each symbol vs benchmark
        - Use rolling window regression
        
        For now, return neutral betas (1.0)
        """
        return {symbol: 1.0 for symbol in symbol_exposure.keys()}


# Helper functions for external use
def compute_symbol_exposure(positions: List[PositionExposure]) -> Dict[str, float]:
    """
    Compute total exposure per symbol
    
    Args:
        positions: List of position exposures
    
    Returns:
        Dict of symbol -> total notional USD
    """
    exposure = defaultdict(float)
    
    for pos in positions:
        exposure[pos.symbol] += abs(pos.notional_usd)
    
    return dict(exposure)


def compute_exchange_exposure(positions: List[PositionExposure]) -> Dict[str, float]:
    """
    Compute total exposure per exchange
    
    Args:
        positions: List of position exposures
    
    Returns:
        Dict of exchange -> total notional USD
    """
    exposure = defaultdict(float)
    
    for pos in positions:
        exposure[pos.exchange] += abs(pos.notional_usd)
    
    return dict(exposure)


def compute_strategy_exposure(positions: List[PositionExposure]) -> Dict[str, float]:
    """
    Compute total exposure per strategy
    
    Args:
        positions: List of position exposures
    
    Returns:
        Dict of strategy -> total notional USD
    """
    exposure = defaultdict(float)
    
    for pos in positions:
        exposure[pos.strategy] += abs(pos.notional_usd)
    
    return dict(exposure)


__all__ = [
    "ExposureMatrixEngine",
    "compute_symbol_exposure",
    "compute_exchange_exposure",
    "compute_strategy_exposure",
]
