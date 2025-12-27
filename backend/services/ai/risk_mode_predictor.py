"""
PHASE 3A: Risk Mode Predictor - ML-Based Dynamic Risk Management
==================================================================

Combines Phase 2 data to predict optimal risk mode:
- Phase 2D: Volatility regime data
- Phase 2B: Orderbook imbalance data
- Market conditions: BTC dominance, funding rates, fear index

Author: AI System
Date: 2025-12-23
"""

import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class RiskMode(Enum):
    """Risk operating modes."""
    ULTRA_CONSERVATIVE = "ultra_conservative"  # 0.5x normal risk
    CONSERVATIVE = "conservative"              # 0.75x normal risk
    NORMAL = "normal"                          # 1.0x normal risk
    AGGRESSIVE = "aggressive"                  # 1.5x normal risk
    ULTRA_AGGRESSIVE = "ultra_aggressive"      # 2.0x normal risk


class MarketRegime(Enum):
    """Market regime classification."""
    BULL_STRONG = "bull_strong"        # Strong uptrend, low vol
    BULL_WEAK = "bull_weak"            # Weak uptrend, rising vol
    BEAR_STRONG = "bear_strong"        # Strong downtrend, rising vol
    BEAR_WEAK = "bear_weak"            # Weak downtrend, low vol
    SIDEWAYS_TIGHT = "sideways_tight"  # Range-bound, low vol
    SIDEWAYS_WIDE = "sideways_wide"    # Range-bound, high vol
    VOLATILE = "volatile"              # High volatility, no trend
    CHOPPY = "choppy"                  # Erratic price action


@dataclass
class RiskModeSignal:
    """Risk mode prediction with confidence."""
    mode: RiskMode
    confidence: float  # 0-1
    regime: MarketRegime
    volatility_score: float
    orderflow_score: float
    market_condition_score: float
    reason: str


class RiskModePredictor:
    """
    Phase 3A: ML-based risk mode prediction.
    
    Responsibilities:
    - Analyze volatility regime (Phase 2D)
    - Analyze orderflow pressure (Phase 2B)
    - Analyze market conditions (BTC dominance, funding, fear)
    - Predict optimal risk mode with confidence
    - Provide actionable recommendations
    """
    
    def __init__(
        self,
        volatility_engine=None,
        orderbook_module=None,
        vol_threshold_high: float = 0.7,
        vol_threshold_low: float = 0.3,
        imbalance_threshold: float = 0.3,
    ):
        """
        Initialize Risk Mode Predictor.
        
        Args:
            volatility_engine: Phase 2D VolatilityStructureEngine
            orderbook_module: Phase 2B OrderbookImbalanceModule
            vol_threshold_high: Volatility score above this = high risk
            vol_threshold_low: Volatility score below this = low risk
            imbalance_threshold: Orderflow imbalance above this = strong directional flow
        """
        self.volatility_engine = volatility_engine
        self.orderbook_module = orderbook_module
        
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_low = vol_threshold_low
        self.imbalance_threshold = imbalance_threshold
        
        # Feature weights for risk scoring
        self.weights = {
            "volatility": 0.40,      # 40% weight on volatility
            "orderflow": 0.30,       # 30% weight on orderflow
            "market_condition": 0.30  # 30% weight on market conditions
        }
        
        # Regime history for stability
        self.regime_history = []
        self.max_history = 10
        
        logger.info(f"[PHASE 3A] Risk Mode Predictor initialized "
                   f"(vol_thresholds=[{vol_threshold_low:.2f}, {vol_threshold_high:.2f}], "
                   f"imbalance_threshold={imbalance_threshold:.2f})")
    
    def predict_risk_mode(
        self,
        symbol: str,
        current_price: float,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> RiskModeSignal:
        """
        Predict optimal risk mode based on all available data.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            market_conditions: Optional dict with BTC dominance, funding rate, fear index
            
        Returns:
            RiskModeSignal with mode, confidence, and reasoning
        """
        # 1. Analyze Volatility Regime (Phase 2D)
        volatility_score, vol_regime = self._analyze_volatility(symbol)
        
        # 2. Analyze Orderflow Pressure (Phase 2B)
        orderflow_score, flow_direction = self._analyze_orderflow(symbol)
        
        # 3. Analyze Market Conditions (external data)
        market_score, market_state = self._analyze_market_conditions(market_conditions or {})
        
        # 4. Classify Market Regime
        regime = self._classify_regime(
            vol_regime, flow_direction, market_state, volatility_score
        )
        
        # 5. Calculate Risk Score (0-1, higher = more aggressive)
        risk_score = self._calculate_risk_score(
            volatility_score, orderflow_score, market_score
        )
        
        # 6. Map Risk Score to Risk Mode
        mode = self._map_score_to_mode(risk_score, regime)
        
        # 7. Calculate Confidence
        confidence = self._calculate_confidence(
            volatility_score, orderflow_score, market_score
        )
        
        # 8. Generate Reason
        reason = self._generate_reason(
            mode, regime, volatility_score, orderflow_score, market_score
        )
        
        signal = RiskModeSignal(
            mode=mode,
            confidence=confidence,
            regime=regime,
            volatility_score=volatility_score,
            orderflow_score=orderflow_score,
            market_condition_score=market_score,
            reason=reason
        )
        
        # Update regime history for stability
        self.regime_history.append(regime)
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
        
        logger.info(f"[PHASE 3A] {symbol} Risk Prediction: "
                   f"mode={mode.value}, confidence={confidence:.1%}, "
                   f"regime={regime.value}, risk_score={risk_score:.2f}")
        
        return signal
    
    def _analyze_volatility(self, symbol: str) -> Tuple[float, str]:
        """
        Analyze volatility regime from Phase 2D.
        
        Returns:
            (volatility_score 0-1, regime_name)
        """
        if not self.volatility_engine:
            return 0.5, "normal"  # Neutral if no data
        
        try:
            vol_analysis = self.volatility_engine.get_complete_volatility_analysis(symbol)
            
            if not vol_analysis:
                return 0.5, "normal"
            
            # Extract volatility score (0-1 scale)
            vol_score = vol_analysis.get("combined_volatility_score", 0.5)
            regime = vol_analysis.get("overall_regime", "normal").lower()
            
            return vol_score, regime
            
        except Exception as e:
            logger.warning(f"[PHASE 3A] Volatility analysis failed for {symbol}: {e}")
            return 0.5, "normal"
    
    def _analyze_orderflow(self, symbol: str) -> Tuple[float, str]:
        """
        Analyze orderflow pressure from Phase 2B.
        
        Returns:
            (orderflow_score 0-1, direction "bullish"/"bearish"/"neutral")
        """
        if not self.orderbook_module:
            return 0.5, "neutral"
        
        try:
            metrics = self.orderbook_module.get_metrics(symbol)
            
            if not metrics:
                return 0.5, "neutral"
            
            # orderflow_imbalance: -1 to +1 (negative = sell pressure)
            imbalance = metrics.orderflow_imbalance
            
            # Convert to 0-1 score (0 = strong sell, 0.5 = neutral, 1 = strong buy)
            orderflow_score = (imbalance + 1.0) / 2.0
            
            # Determine direction
            if imbalance > self.imbalance_threshold:
                direction = "bullish"
            elif imbalance < -self.imbalance_threshold:
                direction = "bearish"
            else:
                direction = "neutral"
            
            return orderflow_score, direction
            
        except Exception as e:
            logger.warning(f"[PHASE 3A] Orderflow analysis failed for {symbol}: {e}")
            return 0.5, "neutral"
    
    def _analyze_market_conditions(self, conditions: Dict[str, Any]) -> Tuple[float, str]:
        """
        Analyze global market conditions.
        
        Args:
            conditions: Dict with optional keys:
                - btc_dominance: 0-100
                - funding_rate: -0.01 to 0.01 (typically)
                - fear_greed_index: 0-100
                
        Returns:
            (market_score 0-1, market_state "bullish"/"bearish"/"neutral")
        """
        scores = []
        
        # BTC Dominance (40-50% = neutral, >60% = bearish alts, <40% = bullish alts)
        btc_dom = conditions.get("btc_dominance")
        if btc_dom is not None:
            if btc_dom > 60:
                scores.append(0.3)  # Bearish for alts
            elif btc_dom < 40:
                scores.append(0.7)  # Bullish for alts
            else:
                scores.append(0.5)  # Neutral
        
        # Funding Rate (positive = longs paying shorts = bullish sentiment)
        funding = conditions.get("funding_rate")
        if funding is not None:
            # Normalize funding rate (-0.01 to 0.01 → 0 to 1)
            funding_score = np.clip((funding + 0.01) / 0.02, 0, 1)
            scores.append(funding_score)
        
        # Fear & Greed Index (0-100, higher = more greed)
        fear_greed = conditions.get("fear_greed_index")
        if fear_greed is not None:
            # Normalize 0-100 to 0-1
            fg_score = fear_greed / 100.0
            scores.append(fg_score)
        
        # Average all available scores
        if scores:
            market_score = np.mean(scores)
        else:
            market_score = 0.5  # Neutral if no data
        
        # Determine market state
        if market_score > 0.65:
            state = "bullish"
        elif market_score < 0.35:
            state = "bearish"
        else:
            state = "neutral"
        
        return market_score, state
    
    def _classify_regime(
        self,
        vol_regime: str,
        flow_direction: str,
        market_state: str,
        vol_score: float
    ) -> MarketRegime:
        """
        Classify overall market regime.
        
        Combines volatility, orderflow, and market conditions.
        """
        # High volatility regimes
        if vol_score > self.vol_threshold_high:
            if flow_direction == "bullish" and market_state == "bullish":
                return MarketRegime.BULL_WEAK  # Bullish but volatile
            elif flow_direction == "bearish" and market_state == "bearish":
                return MarketRegime.BEAR_STRONG  # Strong selling
            else:
                return MarketRegime.VOLATILE  # High vol, mixed signals
        
        # Low volatility regimes
        elif vol_score < self.vol_threshold_low:
            if flow_direction == "bullish" and market_state == "bullish":
                return MarketRegime.BULL_STRONG  # Stable uptrend
            elif flow_direction == "bearish" and market_state == "bearish":
                return MarketRegime.BEAR_WEAK  # Weak downtrend
            else:
                return MarketRegime.SIDEWAYS_TIGHT  # Range-bound, low vol
        
        # Medium volatility regimes
        else:
            if flow_direction == "neutral" and market_state == "neutral":
                return MarketRegime.SIDEWAYS_WIDE  # Range-bound, medium vol
            elif (flow_direction == "bullish" and market_state == "bearish") or \
                 (flow_direction == "bearish" and market_state == "bullish"):
                return MarketRegime.CHOPPY  # Conflicting signals
            else:
                return MarketRegime.SIDEWAYS_WIDE  # Default medium volatility
    
    def _calculate_risk_score(
        self,
        vol_score: float,
        orderflow_score: float,
        market_score: float
    ) -> float:
        """
        Calculate overall risk score (0-1).
        
        Higher score = more favorable conditions = more aggressive risk mode.
        """
        # Invert volatility score (high vol = lower risk score)
        vol_component = 1.0 - vol_score
        
        # Orderflow and market scores are already 0-1
        orderflow_component = orderflow_score
        market_component = market_score
        
        # Weighted average
        risk_score = (
            self.weights["volatility"] * vol_component +
            self.weights["orderflow"] * orderflow_component +
            self.weights["market_condition"] * market_component
        )
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def _map_score_to_mode(self, risk_score: float, regime: MarketRegime) -> RiskMode:
        """
        Map risk score to risk mode.
        
        Adjusts thresholds based on market regime.
        """
        # Regime-based threshold adjustments
        if regime in [MarketRegime.VOLATILE, MarketRegime.CHOPPY, MarketRegime.BEAR_STRONG]:
            # More conservative in dangerous regimes
            thresholds = [0.2, 0.4, 0.6, 0.75]
        elif regime in [MarketRegime.BULL_STRONG, MarketRegime.SIDEWAYS_TIGHT]:
            # More aggressive in favorable regimes
            thresholds = [0.15, 0.35, 0.55, 0.70]
        else:
            # Default thresholds
            thresholds = [0.2, 0.4, 0.6, 0.75]
        
        # Map score to mode
        if risk_score < thresholds[0]:
            return RiskMode.ULTRA_CONSERVATIVE
        elif risk_score < thresholds[1]:
            return RiskMode.CONSERVATIVE
        elif risk_score < thresholds[2]:
            return RiskMode.NORMAL
        elif risk_score < thresholds[3]:
            return RiskMode.AGGRESSIVE
        else:
            return RiskMode.ULTRA_AGGRESSIVE
    
    def _calculate_confidence(
        self,
        vol_score: float,
        orderflow_score: float,
        market_score: float
    ) -> float:
        """
        Calculate prediction confidence (0-1).
        
        Higher confidence when signals are aligned and strong.
        """
        # Check signal alignment (all bullish or all bearish = high confidence)
        scores = [vol_score, orderflow_score, market_score]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Low std = aligned signals = high confidence
        alignment_confidence = 1.0 - np.clip(std_score * 2, 0, 1)
        
        # Strong signals (far from 0.5) = high confidence
        signal_strength = abs(mean_score - 0.5) * 2  # 0 to 1
        
        # Combined confidence
        confidence = (alignment_confidence + signal_strength) / 2.0
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _generate_reason(
        self,
        mode: RiskMode,
        regime: MarketRegime,
        vol_score: float,
        orderflow_score: float,
        market_score: float
    ) -> str:
        """Generate human-readable reason for risk mode."""
        reasons = []
        
        # Volatility component
        if vol_score > 0.7:
            reasons.append("high volatility")
        elif vol_score < 0.3:
            reasons.append("low volatility")
        
        # Orderflow component
        if orderflow_score > 0.65:
            reasons.append("strong buying pressure")
        elif orderflow_score < 0.35:
            reasons.append("strong selling pressure")
        
        # Market component
        if market_score > 0.65:
            reasons.append("bullish market conditions")
        elif market_score < 0.35:
            reasons.append("bearish market conditions")
        
        # Regime context
        regime_str = regime.value.replace("_", " ")
        
        if reasons:
            return f"{regime_str} regime with {', '.join(reasons)}"
        else:
            return f"{regime_str} regime with neutral conditions"
    
    def get_risk_multiplier(self, mode: RiskMode) -> float:
        """
        Get risk multiplier for a given mode.
        
        Used to scale position sizes and stop losses.
        """
        multipliers = {
            RiskMode.ULTRA_CONSERVATIVE: 0.5,
            RiskMode.CONSERVATIVE: 0.75,
            RiskMode.NORMAL: 1.0,
            RiskMode.AGGRESSIVE: 1.5,
            RiskMode.ULTRA_AGGRESSIVE: 2.0,
        }
        return multipliers[mode]
    
    def get_regime_stability(self) -> float:
        """
        Calculate regime stability (0-1).
        
        Higher = more stable (regime hasn't changed much).
        """
        if len(self.regime_history) < 2:
            return 1.0
        
        # Count regime changes
        changes = sum(
            1 for i in range(1, len(self.regime_history))
            if self.regime_history[i] != self.regime_history[i-1]
        )
        
        # Stability = 1 - (changes / possible_changes)
        possible_changes = len(self.regime_history) - 1
        stability = 1.0 - (changes / possible_changes)
        
        return stability
