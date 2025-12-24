"""Trade Opportunity Filter - Quality-based trade filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from backend.config.risk_management import ConsensusType, TradeFilterConfig
from backend.services.risk_management.global_regime_detector import (
    get_global_regime_detector,
    GlobalRegimeDetectionResult,
    GlobalRegime
)

# [PHASE 3C] HEALTH-SCORE GATING
try:
    from backend.services.ai.system_health_monitor import SystemHealthMonitor
    PHASE_3C_AVAILABLE = True
except ImportError:
    PHASE_3C_AVAILABLE = False
    SystemHealthMonitor = None
    logger.warning("[WARNING] Phase 3C System Health Monitor not available")

logger = logging.getLogger(__name__)


@dataclass
class MarketConditions:
    """Current market conditions for a symbol."""
    price: float
    atr: float
    ema_200: float
    volume_24h: float
    spread_bps: int
    timestamp: datetime


@dataclass
class SignalQuality:
    """Signal quality assessment."""
    consensus_type: ConsensusType
    confidence: float
    model_votes: Dict[str, str]  # {model_name: action (LONG/SHORT/HOLD)}
    signal_strength: float        # 0.0 to 1.0


@dataclass
class FilterResult:
    """Result of trade opportunity filtering."""
    passed: bool
    rejection_reason: Optional[str] = None
    warnings: list[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TradeOpportunityFilter:
    """
    Filter trade opportunities based on consensus quality, confidence, trend alignment,
    and market conditions.
    
    This is the first line of defense - only high-quality setups pass through.
    """
    
    def __init__(self, config: TradeFilterConfig):
        self.config = config
        self.global_regime_detector = get_global_regime_detector()
        
        # [PHASE 3C] Initialize health monitor
        self.system_health_monitor = None
        
        logger.info("[OK] TradeOpportunityFilter initialized")
        logger.info(f"   Min consensus: {[ct.value for ct in config.min_consensus_types]}")
        logger.info(f"   Min confidence: {config.min_confidence:.1%}")
        logger.info(f"   Trend alignment: {config.require_trend_alignment}")
        logger.info(f"   Volatility gate: {config.enable_volatility_gate}")
        logger.info(f"   Global regime safety: ENABLED")
    
    def set_health_monitor(self, health_monitor: Optional['SystemHealthMonitor']) -> None:
        """
        Inject Phase 3C health monitor for health-score gating.
        
        Args:
            health_monitor: System health monitor instance
        """
        self.system_health_monitor = health_monitor
        if health_monitor:
            logger.info("[PHASE3C] ✅ Health monitor injected for trade gating")
    
    def evaluate_signal(
        self,
        symbol: str,
        signal_quality: SignalQuality,
        market_conditions: MarketConditions,
        action: str,  # "LONG" or "SHORT"
        global_regime: Optional[GlobalRegimeDetectionResult] = None,
        btc_price: Optional[float] = None,
        btc_ema200: Optional[float] = None,
        signal_source: Optional[str] = None,  # For Phase 3C health tracking
    ) -> FilterResult:
        """
        Evaluate if a trading signal meets quality criteria.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            signal_quality: AI model consensus and confidence
            market_conditions: Current market data (price, ATR, EMA, volume)
            action: Intended action ("LONG" or "SHORT")
            global_regime: Optional pre-computed global regime (for efficiency)
            btc_price: Optional BTCUSDT price (for regime detection)
            btc_ema200: Optional BTCUSDT EMA200 (for regime detection)
            signal_source: Module that generated signal (for Phase 3C health tracking)
        
        Returns:
            FilterResult with pass/fail and rejection reason
        """
        warnings = []
        
        # [PHASE 3C] CHECK 0: Health-Score Gating
        if self.system_health_monitor:
            try:
                health_report = self.system_health_monitor.get_health_report()
                overall_health = health_report.get('overall_health_score', 100.0)
                
                # Reject if overall AI Engine health < 80
                if overall_health < 80.0:
                    logger.warning(
                        f"[🚨 PHASE3C-HEALTH] {symbol} {action} REJECTED: "
                        f"AI Engine health {overall_health:.0f}/100 < 80 (degraded performance)"
                    )
                    return FilterResult(
                        passed=False,
                        rejection_reason=f"AI Engine health degraded ({overall_health:.0f}/100 < 80)",
                        warnings=warnings
                    )
                
                # Check signal source module health if provided
                if signal_source:
                    module_health = health_report.get('modules', {}).get(signal_source, {})
                    
                    if module_health and 'health_score' in module_health:
                        module_score = module_health['health_score']
                        
                        # Reject if source module health < 70
                        if module_score < 70.0:
                            logger.warning(
                                f"[🚨 PHASE3C-HEALTH] {symbol} {action} REJECTED: "
                                f"Signal source '{signal_source}' health {module_score:.0f}/100 < 70"
                            )
                            return FilterResult(
                                passed=False,
                                rejection_reason=f"Signal source '{signal_source}' unhealthy ({module_score:.0f}/100 < 70)",
                                warnings=warnings
                            )
                
                # Log health status if all passed
                logger.info(
                    f"[PHASE3C-HEALTH] {symbol}: Overall health={overall_health:.0f}/100 ✅"
                )
            
            except Exception as e:
                logger.warning(f"[PHASE3C-HEALTH] Failed to check health for {symbol}: {e}")
                # Don't block trade on health check failure
        
        # [SAFETY] CRITICAL: Check global regime for SHORT blocking
        if action == "SHORT":
            # Detect global regime if not provided
            if global_regime is None and btc_price and btc_ema200:
                global_regime = self.global_regime_detector.detect_global_regime(
                    btc_price=btc_price,
                    btc_ema200=btc_ema200
                )
            
            # If we have global regime data, enforce safety rules
            if global_regime and global_regime.regime == GlobalRegime.UPTREND:
                # Check if short should be blocked
                if self.global_regime_detector.should_block_shorts(global_regime):
                    # Check for rare exception
                    allow_exception, exception_reason = self.global_regime_detector.check_short_exception(
                        global_regime_result=global_regime,
                        symbol=symbol,
                        symbol_price=market_conditions.price,
                        symbol_ema200=market_conditions.ema_200,
                        short_confidence=signal_quality.confidence
                    )
                    
                    if not allow_exception:
                        logger.warning(
                            f"[SAFETY] SHORT BLOCKED by global uptrend rule | "
                            f"symbol={symbol} | regime={global_regime.regime.value} | "
                            f"conf={signal_quality.confidence:.1%} | reason={exception_reason}"
                        )
                        return FilterResult(
                            passed=False,
                            rejection_reason=f"SHORT blocked in global UPTREND: {exception_reason}",
                            warnings=warnings
                        )
                    else:
                        # Exception allowed - log it prominently
                        logger.warning(exception_reason)
                        warnings.append(exception_reason)
        
        # Check 1: Consensus Type
        if signal_quality.consensus_type not in self.config.min_consensus_types:
            logger.info(
                f"❌ {symbol} {action} REJECTED: Consensus {signal_quality.consensus_type.value} "
                f"not in {[ct.value for ct in self.config.min_consensus_types]}"
            )
            return FilterResult(
                passed=False,
                rejection_reason=f"Insufficient consensus: {signal_quality.consensus_type.value}",
            )
        
        # Check 2: Confidence Threshold
        min_confidence = self.config.min_confidence
        
        # Apply volatility boost if needed
        if self.config.enable_volatility_gate:
            atr_ratio = market_conditions.atr / market_conditions.price
            if atr_ratio > self.config.max_atr_ratio:
                min_confidence = self.config.high_volatility_confidence_boost
                warnings.append(
                    f"High volatility (ATR ratio {atr_ratio:.2%}), "
                    f"raised confidence requirement to {min_confidence:.1%}"
                )
        
        if signal_quality.confidence < min_confidence:
            logger.info(
                f"❌ {symbol} {action} REJECTED: Confidence {signal_quality.confidence:.1%} "
                f"< {min_confidence:.1%}"
            )
            return FilterResult(
                passed=False,
                rejection_reason=f"Low confidence: {signal_quality.confidence:.1%} < {min_confidence:.1%}",
                warnings=warnings,
            )
        
        # Check 3: Trend Alignment
        if self.config.require_trend_alignment:
            price_vs_ema = market_conditions.price / market_conditions.ema_200
            
            # For LONG: Allow if price is within 2% of EMA (0.98-1.02) OR above EMA
            # For SHORT: Allow if price is within 2% of EMA (0.98-1.02) OR below EMA
            if action == "LONG" and price_vs_ema < 0.98:
                # Price more than 2% below EMA - strong downtrend, block LONG
                logger.info(
                    f"❌ {symbol} LONG REJECTED: Price ${market_conditions.price:.2f} "
                    f"more than 2% below EMA200 ${market_conditions.ema_200:.2f} ({price_vs_ema:.2%})"
                )
                return FilterResult(
                    passed=False,
                    rejection_reason=f"LONG against strong downtrend: price {price_vs_ema:.2%}, need >= 98%",
                    warnings=warnings,
                )
            
            if action == "SHORT" and price_vs_ema > 1.02:
                # Price more than 2% above EMA - strong uptrend
                # [NEW] HIGH-CONFIDENCE OVERRIDE for counter-trend shorts
                # Load threshold from config
                try:
                    from config.config import get_qt_countertrend_min_conf
                    min_conf_threshold = get_qt_countertrend_min_conf()
                except ImportError:
                    min_conf_threshold = 0.50  # Fallback default
                
                # If confidence meets threshold, ALLOW the trade
                if signal_quality.confidence >= min_conf_threshold:
                    logger.warning(
                        f"⚠️  {symbol} SHORT_ALLOWED_AGAINST_TREND_HIGH_CONF: "
                        f"Price ${market_conditions.price:.2f} more than 2% above EMA200 ${market_conditions.ema_200:.2f} "
                        f"({price_vs_ema:.2%}), BUT confidence {signal_quality.confidence:.1%} >= "
                        f"threshold {min_conf_threshold:.1%} → APPROVED"
                    )
                    warnings.append(
                        f"Counter-trend SHORT allowed due to high confidence "
                        f"({signal_quality.confidence:.1%} >= {min_conf_threshold:.1%})"
                    )
                    # Continue to next checks (DO NOT reject)
                else:
                    # Confidence too low - BLOCK as before
                    logger.warning(
                        f"❌ {symbol} SHORT_BLOCKED_AGAINST_TREND_LOW_CONF: "
                        f"Price ${market_conditions.price:.2f} more than 2% above EMA200 ${market_conditions.ema_200:.2f} "
                        f"({price_vs_ema:.2%}), confidence {signal_quality.confidence:.1%} < "
                        f"threshold {min_conf_threshold:.1%} → REJECTED"
                    )
                    return FilterResult(
                        passed=False,
                        rejection_reason=(
                            f"SHORT against strong uptrend: price {price_vs_ema:.2%}, need <= 102%, "
                            f"confidence {signal_quality.confidence:.1%} < {min_conf_threshold:.1%}"
                        ),
                        warnings=warnings,
                    )
        
        # Check 4: Volume Filter
        if market_conditions.volume_24h < self.config.min_volume_24h:
            logger.info(
                f"❌ {symbol} {action} REJECTED: Volume ${market_conditions.volume_24h:,.0f} "
                f"< ${self.config.min_volume_24h:,.0f}"
            )
            return FilterResult(
                passed=False,
                rejection_reason=f"Low volume: ${market_conditions.volume_24h:,.0f}",
                warnings=warnings,
            )
        
        # Check 5: Spread Filter
        if market_conditions.spread_bps > self.config.max_spread_bps:
            logger.info(
                f"❌ {symbol} {action} REJECTED: Spread {market_conditions.spread_bps}bps "
                f"> {self.config.max_spread_bps}bps"
            )
            return FilterResult(
                passed=False,
                rejection_reason=f"Wide spread: {market_conditions.spread_bps}bps",
                warnings=warnings,
            )
        
        # All checks passed
        logger.info(
            f"[OK] {symbol} {action} APPROVED: "
            f"Consensus={signal_quality.consensus_type.value}, "
            f"Confidence={signal_quality.confidence:.1%}, "
            f"Trend aligned, ATR ratio={(market_conditions.atr/market_conditions.price):.2%}"
        )
        
        return FilterResult(passed=True, warnings=warnings)
    
    def calculate_consensus_type(self, model_votes: Dict[str, str]) -> ConsensusType:
        """
        Calculate consensus type from model votes.
        
        Args:
            model_votes: {model_name: action} e.g., {"XGBoost": "LONG", "LightGBM": "LONG", ...}
        
        Returns:
            ConsensusType (UNANIMOUS, STRONG, WEAK, or SPLIT)
        """
        actions = list(model_votes.values())
        
        # Count votes for each action
        long_votes = actions.count("LONG")
        short_votes = actions.count("SHORT")
        hold_votes = actions.count("HOLD")
        
        total_votes = len(actions)
        max_votes = max(long_votes, short_votes, hold_votes)
        
        if max_votes == total_votes:
            return ConsensusType.UNANIMOUS
        
        if max_votes >= 3 and total_votes == 4:
            return ConsensusType.STRONG
        
        # Check for 2-2 split
        if total_votes == 4 and max_votes == 2:
            # Count distinct actions with 2+ votes
            high_vote_actions = sum(1 for count in [long_votes, short_votes, hold_votes] if count >= 2)
            if high_vote_actions == 2:
                return ConsensusType.SPLIT
        
        return ConsensusType.WEAK
    
    def get_dominant_action(self, model_votes: Dict[str, str]) -> Optional[str]:
        """
        Get the dominant action from model votes.
        
        Returns:
            "LONG", "SHORT", or None if split/no consensus
        """
        actions = list(model_votes.values())
        
        long_votes = actions.count("LONG")
        short_votes = actions.count("SHORT")
        
        # Majority needed
        if long_votes > short_votes and long_votes >= 3:
            return "LONG"
        if short_votes > long_votes and short_votes >= 3:
            return "SHORT"
        
        return None  # No clear consensus
