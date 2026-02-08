"""
Arbiter Agent #5: Market Understanding & Final Decision Authority

This agent is ONLY invoked when Meta-Agent v2 explicitly escalates a decision.
It has deep market understanding and makes final trading decisions when the
base ensemble is insufficient.

CRITICAL ROLE:
- Called only after Meta-Agent v2 says "market undecided"
- Uses raw market data + technical indicators
- Must have high confidence to override
- Acts as tie-breaker and edge-case handler

DESIGN PRINCIPLES:
1. High confidence threshold (0.70+) - only speak when certain
2. HOLD is acceptable - no forced action
3. Never contradicts strong consensus (Meta v2 blocks that upstream)
4. Provides market understanding when base models disagree
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from collections import Counter
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class ArbiterAgent:
    """
    Arbiter Agent #5: Final decision authority for escalated cases.
    
    This agent is invoked ONLY when Meta-Agent v2 escalates a decision
    due to market uncertainty or disagreement between base models.
    
    Features:
    - Deep market understanding through technical indicators
    - High confidence threshold (default 0.70)
    - Can return HOLD if market is unclear
    - Provides explainable reasoning
    
    Architecture:
    - Input: Market data (OHLCV + indicators), optional base signals
    - Processing: Technical analysis + pattern recognition
    - Output: {action, confidence, reason, indicators_used}
    """
    
    VERSION = "1.0.0"
    ACTIONS = ["SELL", "HOLD", "BUY"]
    
    # Thresholds
    DEFAULT_CONFIDENCE_THRESHOLD = 0.70  # Must be very confident
    STRONG_SIGNAL_THRESHOLD = 0.75
    EXTREME_SIGNAL_THRESHOLD = 0.85
    
    # Technical indicator weights
    INDICATOR_WEIGHTS = {
        'rsi': 0.20,
        'macd': 0.25,
        'bb_position': 0.15,
        'volume_trend': 0.15,
        'price_momentum': 0.25
    }
    
    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        enable_conservative_mode: bool = True
    ):
        """
        Initialize Arbiter Agent.
        
        Args:
            confidence_threshold: Minimum confidence to override (0.70)
            enable_conservative_mode: If True, requires stronger signals
        """
        self.confidence_threshold = confidence_threshold
        self.enable_conservative_mode = enable_conservative_mode
        
        # Statistics
        self.total_calls = 0
        self.decisions_made = 0  # Non-HOLD
        self.holds_returned = 0
        self.low_confidence_rejects = 0
        
        logger.info(f"[Arbiter] Initialized v{self.VERSION}")
        logger.info(f"[Arbiter]   Confidence threshold: {confidence_threshold:.2f}")
        logger.info(f"[Arbiter]   Conservative mode: {enable_conservative_mode}")
    
    def predict(
        self,
        market_data: Dict[str, Any],
        base_signals: Optional[Dict[str, Dict[str, Any]]] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make trading decision based on market data.
        
        This is called ONLY when Meta-Agent v2 escalates.
        
        Args:
            market_data: {
                'price': current price,
                'ohlcv': recent OHLCV data,
                'indicators': technical indicators (RSI, MACD, etc.),
                'regime': market regime info
            }
            base_signals: Optional context from base models (NOT for voting)
            symbol: Trading symbol
        
        Returns:
            {
                "action": "BUY" | "SELL" | "HOLD",
                "confidence": float,
                "reason": str,
                "indicators_used": dict,
                "arbiter_invoked": true
            }
        """
        self.total_calls += 1
        symbol_str = f" [{symbol}]" if symbol else ""
        
        logger.info(f"[Arbiter]{symbol_str} ⚖️  INVOKED (escalated from Meta-V2)")
        
        # Default: HOLD with low confidence
        result = {
            "action": "HOLD",
            "confidence": 0.5,
            "reason": "insufficient_data",
            "indicators_used": {},
            "arbiter_invoked": True
        }
        
        # Validate input
        if not market_data:
            result['reason'] = "no_market_data"
            logger.warning(f"[Arbiter]{symbol_str} No market data provided")
            return result
        
        # Extract and analyze indicators
        try:
            indicators = market_data.get('indicators', {})
            regime = market_data.get('regime', {})
            
            # Analyze technical indicators
            analysis = self._analyze_market(indicators, regime)
            
            # Make decision
            action, confidence, reason = self._decide(analysis, base_signals)
            
            result['action'] = action
            result['confidence'] = confidence
            result['reason'] = reason
            result['indicators_used'] = analysis
            
            # Update statistics
            if action == "HOLD":
                self.holds_returned += 1
            else:
                self.decisions_made += 1
            
            # Check confidence threshold (gating)
            if confidence < self.confidence_threshold:
                self.low_confidence_rejects += 1
                logger.info(
                    f"[Arbiter]{symbol_str} DECLINED: confidence {confidence:.3f} "
                    f"< threshold {self.confidence_threshold:.2f} → Will use ensemble"
                )
            else:
                logger.info(
                    f"[Arbiter]{symbol_str} DECISION: {action} @ {confidence:.3f} "
                    f"(Reason: {reason})"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"[Arbiter]{symbol_str} Analysis failed: {e}")
            result['reason'] = f"analysis_error: {e}"
            return result
    
    def _analyze_market(
        self,
        indicators: Dict[str, Any],
        regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market using technical indicators.
        
        Returns:
            {
                'rsi_signal': -1/0/+1,
                'macd_signal': -1/0/+1,
                'bb_signal': -1/0/+1,
                'volume_signal': -1/0/+1,
                'momentum_signal': -1/0/+1,
                'regime_signal': -1/0/+1,
                'total_score': float
            }
        """
        analysis = {}
        
        # RSI Analysis (oversold/overbought)
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            analysis['rsi_signal'] = 1  # Oversold → BUY
            analysis['rsi_strength'] = (30 - rsi) / 30
        elif rsi > 70:
            analysis['rsi_signal'] = -1  # Overbought → SELL
            analysis['rsi_strength'] = (rsi - 70) / 30
        else:
            analysis['rsi_signal'] = 0  # Neutral
            analysis['rsi_strength'] = 0.0
        
        # MACD Analysis (momentum)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_diff = macd - macd_signal
        
        if macd_diff > 0:
            analysis['macd_signal'] = 1  # Bullish
            analysis['macd_strength'] = min(abs(macd_diff) * 10, 1.0)
        elif macd_diff < 0:
            analysis['macd_signal'] = -1  # Bearish
            analysis['macd_strength'] = min(abs(macd_diff) * 10, 1.0)
        else:
            analysis['macd_signal'] = 0
            analysis['macd_strength'] = 0.0
        
        # Bollinger Bands Position
        bb_position = indicators.get('bb_position', 0.5)  # 0=lower, 0.5=middle, 1=upper
        
        if bb_position < 0.2:
            analysis['bb_signal'] = 1  # Near lower band → BUY
            analysis['bb_strength'] = 0.2 - bb_position
        elif bb_position > 0.8:
            analysis['bb_signal'] = -1  # Near upper band → SELL
            analysis['bb_strength'] = bb_position - 0.8
        else:
            analysis['bb_signal'] = 0
            analysis['bb_strength'] = 0.0
        
        # Volume Trend (confirmation)
        volume_trend = indicators.get('volume_trend', 1.0)  # ratio vs average
        
        if volume_trend > 1.5:
            # High volume: amplifies signal strength
            analysis['volume_signal'] = 1 if macd_diff > 0 else -1
            analysis['volume_strength'] = min((volume_trend - 1.0) / 2.0, 1.0)
        else:
            analysis['volume_signal'] = 0
            analysis['volume_strength'] = 0.0
        
        # Price Momentum (rate of change)
        momentum = indicators.get('momentum_1h', 0.0)  # % change last hour
        
        if momentum > 0.02:  # +2% momentum
            analysis['momentum_signal'] = 1
            analysis['momentum_strength'] = min(momentum * 10, 1.0)
        elif momentum < -0.02:  # -2% momentum
            analysis['momentum_signal'] = -1
            analysis['momentum_strength'] = min(abs(momentum) * 10, 1.0)
        else:
            analysis['momentum_signal'] = 0
            analysis['momentum_strength'] = 0.0
        
        # Regime context (from ensemble)
        volatility = regime.get('volatility', 0.5)
        trend_strength = regime.get('trend_strength', 0.0)
        
        if abs(trend_strength) > 0.3:
            analysis['regime_signal'] = 1 if trend_strength > 0 else -1
            analysis['regime_strength'] = abs(trend_strength)
        else:
            analysis['regime_signal'] = 0
            analysis['regime_strength'] = 0.0
        
        # Compute weighted total score
        total_score = (
            analysis['rsi_signal'] * analysis['rsi_strength'] * self.INDICATOR_WEIGHTS['rsi'] +
            analysis['macd_signal'] * analysis['macd_strength'] * self.INDICATOR_WEIGHTS['macd'] +
            analysis['bb_signal'] * analysis['bb_strength'] * self.INDICATOR_WEIGHTS['bb_position'] +
            analysis['volume_signal'] * analysis['volume_strength'] * self.INDICATOR_WEIGHTS['volume_trend'] +
            analysis['momentum_signal'] * analysis['momentum_strength'] * self.INDICATOR_WEIGHTS['price_momentum']
        )
        
        analysis['total_score'] = total_score
        
        return analysis
    
    def _decide(
        self,
        analysis: Dict[str, Any],
        base_signals: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[str, float, str]:
        """
        Make final decision based on analysis.
        
        Args:
            analysis: Technical indicator analysis
            base_signals: Optional base model signals (context only)
        
        Returns:
            (action, confidence, reason)
        """
        total_score = analysis['total_score']
        
        # Conservative mode: require stronger signals
        if self.enable_conservative_mode:
            buy_threshold = 0.15
            sell_threshold = -0.15
        else:
            buy_threshold = 0.10
            sell_threshold = -0.10
        
        # Decision logic
        if total_score >= buy_threshold:
            action = "BUY"
            # Confidence scales with signal strength
            confidence = min(0.70 + (total_score * 0.5), 0.95)
            
            # Build reason
            strong_indicators = []
            if analysis['rsi_signal'] == 1 and analysis['rsi_strength'] > 0.5:
                strong_indicators.append("oversold_rsi")
            if analysis['macd_signal'] == 1 and analysis['macd_strength'] > 0.5:
                strong_indicators.append("bullish_macd")
            if analysis['momentum_signal'] == 1:
                strong_indicators.append("strong_momentum")
            
            reason = f"buy_signal: {', '.join(strong_indicators) if strong_indicators else 'technical_alignment'}"
            
        elif total_score <= sell_threshold:
            action = "SELL"
            confidence = min(0.70 + (abs(total_score) * 0.5), 0.95)
            
            # Build reason
            strong_indicators = []
            if analysis['rsi_signal'] == -1 and analysis['rsi_strength'] > 0.5:
                strong_indicators.append("overbought_rsi")
            if analysis['macd_signal'] == -1 and analysis['macd_strength'] > 0.5:
                strong_indicators.append("bearish_macd")
            if analysis['momentum_signal'] == -1:
                strong_indicators.append("negative_momentum")
            
            reason = f"sell_signal: {', '.join(strong_indicators) if strong_indicators else 'technical_alignment'}"
            
        else:
            # Neutral zone → HOLD
            action = "HOLD"
            confidence = 0.50 + (0.1 - abs(total_score)) * 2  # Higher confidence for staying neutral
            reason = f"neutral_market: score={total_score:.3f} in neutral zone"
        
        return action, confidence, reason
    
    def should_override_ensemble(
        self,
        action: str,
        confidence: float
    ) -> bool:
        """
        Gating function: Check if Arbiter decision should override ensemble.
        
        Called by ensemble_manager after getting Arbiter prediction.
        
        Args:
            action: Arbiter action
            confidence: Arbiter confidence
        
        Returns:
            True if should override (high confidence + active action)
            False if should use ensemble (low confidence or HOLD)
        """
        # Gate 1: Confidence must meet threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Gate 2: Action must be active (not HOLD)
        if action == "HOLD":
            return False
        
        # Both gates passed
        return True
    
    def get_context_from_base_signals(
        self,
        base_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract useful context from base model signals (NOT for voting).
        
        This provides additional market insight, but does NOT directly
        influence the decision.
        
        Args:
            base_signals: Base model predictions
        
        Returns:
            Context dict with disagreement metrics, avg confidence, etc.
        """
        if not base_signals:
            return {}
        
        actions = [s.get('action', 'HOLD') for s in base_signals.values()]
        confidences = [s.get('confidence', 0.5) for s in base_signals.values()]
        
        action_counts = Counter(actions)
        
        context = {
            'num_base_models': len(base_signals),
            'avg_base_confidence': np.mean(confidences),
            'base_disagreement': len(action_counts) / len(base_signals) if base_signals else 0,
            'base_action_distribution': dict(action_counts)
        }
        
        return context
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics."""
        decision_rate = (
            self.decisions_made / self.total_calls
            if self.total_calls > 0
            else 0.0
        )
        
        return {
            "total_calls": self.total_calls,
            "decisions_made": self.decisions_made,
            "holds_returned": self.holds_returned,
            "low_confidence_rejects": self.low_confidence_rejects,
            "decision_rate": decision_rate,
            "confidence_threshold": self.confidence_threshold
        }
    
    def reset_statistics(self) -> None:
        """Reset runtime statistics."""
        self.total_calls = 0
        self.decisions_made = 0
        self.holds_returned = 0
        self.low_confidence_rejects = 0
        logger.info("[Arbiter] Statistics reset")
