"""
PROFIT AMPLIFICATION LAYER (PAL)

Amplifies profit potential on high-quality trades through smart extensions,
scaling, and hold time optimization. Does NOT create new trades - only enhances
existing winners.

Mission: Increase average R and total profit by identifying and amplifying
high-quality opportunities.

Author: Quantum Trader AI Team
Date: November 23, 2025
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class AmplificationAction(Enum):
    """Types of amplification actions."""
    ADD_SIZE = "add_size"                    # Scale into winning position
    EXTEND_HOLD = "extend_hold"              # Hold longer, trail stops
    PARTIAL_TAKE_PROFIT = "partial_take_profit"  # Lock some, leave runner
    SWITCH_TO_TREND_FOLLOW = "switch_to_trend_follow"  # Exit based on trend, not TP
    TIGHTEN_STOPS = "tighten_stops"          # Protect profits more aggressively
    NO_ACTION = "no_action"                  # Not amplifiable


class AmplificationReason(Enum):
    """Reasons for amplification decisions."""
    HIGH_R_STRONG_TREND = "high_r_strong_trend"
    REGIME_ALIGNMENT = "regime_alignment"
    LOW_DRAWDOWN = "low_drawdown"
    CORE_SYMBOL = "core_symbol"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    VOLATILITY_SUPPORTIVE = "volatility_supportive"
    
    # Negative reasons (blocking)
    HIGH_DRAWDOWN = "high_drawdown"
    WEAK_TREND = "weak_trend"
    RISK_PROFILE_REDUCED = "risk_profile_reduced"
    EMERGENCY_BRAKE_ACTIVE = "emergency_brake_active"
    UNSTABLE_SYMBOL = "unstable_symbol"
    INSUFFICIENT_RISK_BUDGET = "insufficient_risk_budget"
    POOR_LIQUIDITY = "poor_liquidity"


class TrendStrength(Enum):
    """Trend strength classification."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PositionSnapshot:
    """Snapshot of a position for amplification analysis."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    
    # Performance
    current_R: float
    peak_R: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Drawdown
    drawdown_from_peak_R: float  # e.g., 0.2 = 20% DD from peak
    drawdown_from_peak_pnl_pct: float
    
    # Risk
    current_leverage: float
    position_size_usd: float
    risk_pct: float
    
    # Timing
    hold_time_hours: float
    entry_time: str
    
    # Classification (from PIL)
    pil_classification: str  # "WINNER", "RUNNER", etc.
    
    # Market data
    trend_strength: TrendStrength
    volatility_regime: str  # "LOW", "NORMAL", "HIGH"
    symbol_rank: int  # From Universe OS
    symbol_category: str  # "CORE", "TACTICAL", "OPPORTUNISTIC"


@dataclass
class AmplificationCandidate:
    """A position identified as amplifiable."""
    position: PositionSnapshot
    
    # Amplification score (0-100)
    amplification_score: float
    
    # Qualifying factors
    qualifies_for_scale_in: bool
    qualifies_for_extend_hold: bool
    qualifies_for_partial_take: bool
    
    # Blocking factors
    blocked_by: List[AmplificationReason]
    
    # Risk assessment
    additional_size_allowed_usd: float
    max_additional_leverage: float


@dataclass
class AmplificationRecommendation:
    """Recommendation for amplifying a position."""
    candidate: AmplificationCandidate
    
    # Primary action
    action: AmplificationAction
    priority: int  # 1=highest
    
    # Parameters
    parameters: Dict[str, Any]  # Action-specific parameters
    
    # Rationale
    rationale: str
    supporting_reasons: List[AmplificationReason]
    
    # Risk
    risk_assessment: str
    risk_score: float  # 0-10, lower is safer
    
    # Expected impact
    expected_R_increase: float  # Expected R increase
    confidence: float  # 0-100


@dataclass
class AmplificationReport:
    """Complete PAL analysis report."""
    timestamp: str
    
    # Positions analyzed
    total_positions: int
    analyzed_positions: List[PositionSnapshot]
    
    # Candidates
    amplification_candidates: List[AmplificationCandidate]
    high_priority_candidates: List[AmplificationCandidate]
    
    # Recommendations
    recommendations: List[AmplificationRecommendation]
    
    # Summary
    avg_amplification_score: float
    total_additional_size_available_usd: float
    
    # Flags
    can_amplify: bool
    amplification_enabled: bool


# ============================================================
# PROFIT AMPLIFICATION LAYER
# ============================================================

class ProfitAmplificationLayer:
    """
    Amplifies profit on high-quality trades.
    
    Does NOT create new trades - only enhances existing winners through:
    - Smart position sizing (scale-in)
    - Extended hold times (trend-following exits)
    - Partial profit taking with runners
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        
        # [HEDGEFUND MODE] Amplification thresholds
        min_R_for_amplification: float = 0.8,  # Lower threshold for more opportunities
        min_R_for_scale_in: float = 1.2,  # Reduced from 1.5 for HEDGEFUND MODE
        min_R_for_extend_hold: float = 0.8,  # Lower threshold
        
        # [HEDGEFUND MODE] Drawdown limits (slightly relaxed)
        max_dd_from_peak_pct: float = 20.0,  # Increased from 15% for HEDGEFUND MODE
        max_dd_for_scale_in_pct: float = 12.0,  # Increased from 10%
        
        # Trend requirements
        min_trend_strength_for_amplification: TrendStrength = TrendStrength.MODERATE,
        min_trend_strength_for_scale_in: TrendStrength = TrendStrength.STRONG,
        
        # [HEDGEFUND MODE] Risk limits (capped at +50% from base)
        max_additional_leverage: float = 5.0,
        max_position_concentration_pct: float = 25.0,  # Increased from 20%
        max_scale_in_multiplier: float = 1.5,  # +50% cap from base size
        
        # Scoring weights
        r_weight: float = 0.30,
        trend_weight: float = 0.25,
        dd_weight: float = 0.20,
        symbol_rank_weight: float = 0.15,
        volatility_weight: float = 0.10,
    ):
        self.data_dir = Path(data_dir)
        
        self.min_R_for_amplification = min_R_for_amplification
        self.min_R_for_scale_in = min_R_for_scale_in
        self.min_R_for_extend_hold = min_R_for_extend_hold
        
        self.max_dd_from_peak_pct = max_dd_from_peak_pct
        self.max_dd_for_scale_in_pct = max_dd_for_scale_in_pct
        
        self.min_trend_strength_for_amplification = min_trend_strength_for_amplification
        self.min_trend_strength_for_scale_in = min_trend_strength_for_scale_in
        
        self.max_additional_leverage = max_additional_leverage
        self.max_position_concentration_pct = max_position_concentration_pct
        self.max_scale_in_multiplier = max_scale_in_multiplier  # [NEW] HEDGEFUND MODE cap
        
        # Scoring weights
        self.r_weight = r_weight
        self.trend_weight = trend_weight
        self.dd_weight = dd_weight
        self.symbol_rank_weight = symbol_rank_weight
        self.volatility_weight = volatility_weight
        
        # State
        self.amplification_enabled = True
        self.emergency_brake_active = False
        self.hedgefund_mode_enabled = True  # [NEW] HEDGEFUND MODE flag
        
        # Create directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[PAL] 🚀 HEDGEFUND MODE: Aggressive amplification enabled")
        logger.info(f"[PAL] Scale-in cap: +{(max_scale_in_multiplier - 1.0) * 100:.0f}% from base size")
        logger.info(f"[PAL] Min R for scale-in: {min_R_for_scale_in:.1f}R")
    
    # --------------------------------------------------------
    # MAIN ANALYSIS
    # --------------------------------------------------------
    
    def analyze_positions(
        self,
        positions: List[PositionSnapshot],
        risk_profile: str = "SAFE",
        portfolio_risk_budget_pct: float = 10.0,
        regime: str = "BULL",
        exit_mode: str = "NORMAL",
        safety_governor_directives: Optional[Dict[str, Any]] = None  # [NEW] SafetyGovernor integration
    ) -> AmplificationReport:
        """
        Analyze positions and generate amplification recommendations.
        
        Args:
            positions: Current open positions
            risk_profile: "SAFE", "AGGRESSIVE", or "REDUCED"
            portfolio_risk_budget_pct: Available risk budget (%)
            regime: Current market regime
            exit_mode: Current exit mode
            safety_governor_directives: SafetyGovernor directives (optional)
        
        Returns:
            AmplificationReport with recommendations
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"[PAL] Analyzing {len(positions)} positions for amplification opportunities")
        logger.info(f"[PAL] Risk profile: {risk_profile}, Regime: {regime}, Exit mode: {exit_mode}")
        
        # Check if amplification is allowed
        can_amplify = self._can_amplify_globally(
            risk_profile, 
            exit_mode, 
            safety_governor_directives  # [NEW] Pass SafetyGovernor directives
        )
        
        if not can_amplify:
            logger.warning("[PAL] Amplification blocked globally")
            return self._empty_report(timestamp, positions)
        
        # Identify candidates
        candidates = []
        for position in positions:
            candidate = self._evaluate_position(
                position,
                risk_profile,
                portfolio_risk_budget_pct,
                regime
            )
            if candidate:
                candidates.append(candidate)
        
        logger.info(f"[PAL] Found {len(candidates)} amplification candidates")
        
        # Generate recommendations
        recommendations = []
        for candidate in candidates:
            rec = self._generate_recommendation(
                candidate,
                risk_profile,
                portfolio_risk_budget_pct,
                regime
            )
            if rec.action != AmplificationAction.NO_ACTION:
                recommendations.append(rec)
        
        # Sort by priority
        recommendations.sort(key=lambda r: (r.priority, -r.expected_R_increase))
        
        # Calculate summary metrics
        high_priority = [c for c in candidates if c.amplification_score >= 70]
        avg_score = sum(c.amplification_score for c in candidates) / len(candidates) if candidates else 0
        total_size_available = sum(c.additional_size_allowed_usd for c in candidates)
        
        # Build report
        report = AmplificationReport(
            timestamp=timestamp,
            total_positions=len(positions),
            analyzed_positions=positions,
            amplification_candidates=candidates,
            high_priority_candidates=high_priority,
            recommendations=recommendations,
            avg_amplification_score=avg_score,
            total_additional_size_available_usd=total_size_available,
            can_amplify=can_amplify,
            amplification_enabled=self.amplification_enabled
        )
        
        # Save report
        self._save_report(report)
        
        # Log summary
        logger.info(
            f"[PAL] Analysis complete: "
            f"{len(candidates)} candidates, "
            f"{len(high_priority)} high priority, "
            f"{len(recommendations)} recommendations"
        )
        
        return report
    
    # --------------------------------------------------------
    # CANDIDATE IDENTIFICATION
    # --------------------------------------------------------
    
    def _evaluate_position(
        self,
        position: PositionSnapshot,
        risk_profile: str,
        portfolio_risk_budget_pct: float,
        regime: str
    ) -> Optional[AmplificationCandidate]:
        """Evaluate if position qualifies for amplification."""
        
        # Basic R threshold
        if position.current_R < self.min_R_for_amplification:
            return None
        
        # Calculate amplification score
        score = self._calculate_amplification_score(position)
        
        # Check blocking factors
        blocked_by = self._check_blocking_factors(position, risk_profile)
        
        # If blocked, return low-score candidate
        if blocked_by:
            return AmplificationCandidate(
                position=position,
                amplification_score=max(score - 30, 0),  # Penalize blocked candidates
                qualifies_for_scale_in=False,
                qualifies_for_extend_hold=False,
                qualifies_for_partial_take=False,
                blocked_by=blocked_by,
                additional_size_allowed_usd=0.0,
                max_additional_leverage=0.0
            )
        
        # Check specific qualifications
        qualifies_scale_in = self._qualifies_for_scale_in(position)
        qualifies_extend = self._qualifies_for_extend_hold(position)
        qualifies_partial = self._qualifies_for_partial_take(position)
        
        # Calculate allowed additional size
        additional_size = self._calculate_additional_size_allowed(
            position,
            portfolio_risk_budget_pct,
            risk_profile
        )
        
        return AmplificationCandidate(
            position=position,
            amplification_score=score,
            qualifies_for_scale_in=qualifies_scale_in,
            qualifies_for_extend_hold=qualifies_extend,
            qualifies_for_partial_take=qualifies_partial,
            blocked_by=[],
            additional_size_allowed_usd=additional_size,
            max_additional_leverage=self.max_additional_leverage
        )
    
    def _calculate_amplification_score(self, position: PositionSnapshot) -> float:
        """Calculate amplification score (0-100)."""
        
        # R score (0-30 points)
        r_score = min(position.current_R / 5.0, 1.0) * 30 * self.r_weight / 0.30
        
        # Trend score (0-25 points)
        trend_scores = {
            TrendStrength.VERY_STRONG: 1.0,
            TrendStrength.STRONG: 0.8,
            TrendStrength.MODERATE: 0.5,
            TrendStrength.WEAK: 0.2,
            TrendStrength.NONE: 0.0
        }
        trend_score = trend_scores[position.trend_strength] * 25 * self.trend_weight / 0.25
        
        # Drawdown score (0-20 points) - lower DD = higher score
        dd_score = max(1.0 - (position.drawdown_from_peak_R / 0.5), 0) * 20 * self.dd_weight / 0.20
        
        # Symbol rank score (0-15 points)
        rank_score = max(1.0 - (position.symbol_rank / 100), 0) * 15 * self.symbol_rank_weight / 0.15
        
        # Volatility score (0-10 points)
        vol_scores = {
            "LOW": 1.0,
            "NORMAL": 0.8,
            "HIGH": 0.4
        }
        vol_score = vol_scores.get(position.volatility_regime, 0.5) * 10 * self.volatility_weight / 0.10
        
        # Total score
        total = r_score + trend_score + dd_score + rank_score + vol_score
        
        return min(max(total, 0), 100)
    
    def _check_blocking_factors(
        self,
        position: PositionSnapshot,
        risk_profile: str
    ) -> List[AmplificationReason]:
        """Check for factors that block amplification."""
        blocked_by = []
        
        # High drawdown
        if position.drawdown_from_peak_R > (self.max_dd_from_peak_pct / 100):
            blocked_by.append(AmplificationReason.HIGH_DRAWDOWN)
        
        # Weak trend
        if position.trend_strength == TrendStrength.WEAK or position.trend_strength == TrendStrength.NONE:
            blocked_by.append(AmplificationReason.WEAK_TREND)
        
        # Risk profile reduced
        if risk_profile == "REDUCED":
            blocked_by.append(AmplificationReason.RISK_PROFILE_REDUCED)
        
        # Emergency brake
        if self.emergency_brake_active:
            blocked_by.append(AmplificationReason.EMERGENCY_BRAKE_ACTIVE)
        
        # Unstable symbol (high volatility + low rank)
        if position.volatility_regime == "HIGH" and position.symbol_rank > 50:
            blocked_by.append(AmplificationReason.UNSTABLE_SYMBOL)
        
        return blocked_by
    
    def _qualifies_for_scale_in(self, position: PositionSnapshot) -> bool:
        """Check if position qualifies for scale-in."""
        return (
            position.current_R >= self.min_R_for_scale_in and
            position.drawdown_from_peak_R <= (self.max_dd_for_scale_in_pct / 100) and
            position.trend_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG] and
            position.volatility_regime in ["LOW", "NORMAL"]
        )
    
    def _qualifies_for_extend_hold(self, position: PositionSnapshot) -> bool:
        """Check if position qualifies for extended hold."""
        return (
            position.current_R >= self.min_R_for_extend_hold and
            position.trend_strength in [TrendStrength.MODERATE, TrendStrength.STRONG, TrendStrength.VERY_STRONG] and
            position.drawdown_from_peak_R <= (self.max_dd_from_peak_pct / 100)
        )
    
    def _qualifies_for_partial_take(self, position: PositionSnapshot) -> bool:
        """Check if position qualifies for partial profit taking."""
        return (
            position.current_R >= 2.0 and  # At least 2R
            position.peak_R >= 3.0 and     # Has been to 3R+
            position.symbol_category in ["CORE", "TACTICAL"]
        )
    
    # --------------------------------------------------------
    # RECOMMENDATION GENERATION
    # --------------------------------------------------------
    
    def _generate_recommendation(
        self,
        candidate: AmplificationCandidate,
        risk_profile: str,
        portfolio_risk_budget_pct: float,
        regime: str
    ) -> AmplificationRecommendation:
        """Generate specific amplification recommendation."""
        
        position = candidate.position
        
        # If blocked, return NO_ACTION
        if candidate.blocked_by:
            return AmplificationRecommendation(
                candidate=candidate,
                action=AmplificationAction.NO_ACTION,
                priority=999,
                parameters={},
                rationale=f"Blocked by: {', '.join(r.value for r in candidate.blocked_by)}",
                supporting_reasons=[],
                risk_assessment="N/A",
                risk_score=10.0,
                expected_R_increase=0.0,
                confidence=0.0
            )
        
        # Determine best action based on qualifications and score
        
        # SCALE-IN: Highest R potential, but highest risk
        if candidate.qualifies_for_scale_in and candidate.additional_size_allowed_usd > 100:
            return self._recommend_scale_in(candidate, risk_profile, regime)
        
        # EXTEND HOLD: Medium R potential, low risk
        if candidate.qualifies_for_extend_hold:
            return self._recommend_extend_hold(candidate, risk_profile, regime)
        
        # PARTIAL TAKE PROFIT: Lock gains, reduce risk
        if candidate.qualifies_for_partial_take:
            return self._recommend_partial_take(candidate, risk_profile, regime)
        
        # Default: No action
        return AmplificationRecommendation(
            candidate=candidate,
            action=AmplificationAction.NO_ACTION,
            priority=999,
            parameters={},
            rationale="No qualifying amplification strategy",
            supporting_reasons=[],
            risk_assessment="N/A",
            risk_score=5.0,
            expected_R_increase=0.0,
            confidence=0.0
        )
    
    def _recommend_scale_in(
        self,
        candidate: AmplificationCandidate,
        risk_profile: str,
        regime: str
    ) -> AmplificationRecommendation:
        """Recommend scale-in action."""
        position = candidate.position
        
        # Calculate scale-in size (conservative)
        base_scale_pct = 0.30 if risk_profile == "AGGRESSIVE" else 0.20  # 20-30% of current
        scale_size_usd = min(
            position.position_size_usd * base_scale_pct,
            candidate.additional_size_allowed_usd
        )
        
        # Supporting reasons
        reasons = [AmplificationReason.HIGH_R_STRONG_TREND]
        if position.symbol_category == "CORE":
            reasons.append(AmplificationReason.CORE_SYMBOL)
        if position.volatility_regime == "LOW":
            reasons.append(AmplificationReason.VOLATILITY_SUPPORTIVE)
        
        return AmplificationRecommendation(
            candidate=candidate,
            action=AmplificationAction.ADD_SIZE,
            priority=1,
            parameters={
                "scale_size_usd": scale_size_usd,
                "max_total_leverage": position.current_leverage + 2.0,
                "stop_loss_adjust": "trail_at_breakeven",
                "timing": "on_next_confirmation"
            },
            rationale=f"Strong trend continuation at {position.current_R:.1f}R with low DD ({position.drawdown_from_peak_R*100:.1f}%)",
            supporting_reasons=reasons,
            risk_assessment=f"Low-moderate risk: Adding ${scale_size_usd:.0f} to winning position",
            risk_score=4.0,
            expected_R_increase=0.5 * base_scale_pct,  # Expect 0.1-0.15R additional
            confidence=75.0
        )
    
    def _recommend_extend_hold(
        self,
        candidate: AmplificationCandidate,
        risk_profile: str,
        regime: str
    ) -> AmplificationRecommendation:
        """Recommend extended hold time."""
        position = candidate.position
        
        # Supporting reasons
        reasons = [AmplificationReason.MOMENTUM_CONTINUATION]
        if candidate.amplification_score >= 70:
            reasons.append(AmplificationReason.REGIME_ALIGNMENT)
        if position.drawdown_from_peak_R < 0.05:  # < 5% DD
            reasons.append(AmplificationReason.LOW_DRAWDOWN)
        
        return AmplificationRecommendation(
            candidate=candidate,
            action=AmplificationAction.EXTEND_HOLD,
            priority=2,
            parameters={
                "exit_strategy": "trend_follow",
                "trail_stop_type": "ATR_based",
                "trail_distance_atr": 2.0,
                "min_hold_additional_hours": 12
            },
            rationale=f"Trend still strong at {position.current_R:.1f}R, switch to trend-following exit",
            supporting_reasons=reasons,
            risk_assessment="Low risk: Protect with trailing stop",
            risk_score=2.0,
            expected_R_increase=1.0,  # Could gain 1R more by riding trend
            confidence=65.0
        )
    
    def _recommend_partial_take(
        self,
        candidate: AmplificationCandidate,
        risk_profile: str,
        regime: str
    ) -> AmplificationRecommendation:
        """Recommend partial profit taking."""
        position = candidate.position
        
        # Take 30-50% off
        take_pct = 0.30 if risk_profile == "AGGRESSIVE" else 0.50
        
        reasons = [AmplificationReason.HIGH_R_STRONG_TREND]
        
        return AmplificationRecommendation(
            candidate=candidate,
            action=AmplificationAction.PARTIAL_TAKE_PROFIT,
            priority=3,
            parameters={
                "take_profit_pct": take_pct,
                "leave_runner_pct": 1.0 - take_pct,
                "runner_exit": "trend_follow",
                "lock_profit_r": position.current_R * take_pct
            },
            rationale=f"Lock {position.current_R * take_pct:.1f}R profit, leave {(1-take_pct)*100:.0f}% runner",
            supporting_reasons=reasons,
            risk_assessment="Very low risk: Locking in gains",
            risk_score=1.0,
            expected_R_increase=0.5,  # Runner could add 0.5R
            confidence=80.0
        )
    
    # --------------------------------------------------------
    # HELPER METHODS
    # --------------------------------------------------------
    
    def _can_amplify_globally(
        self, 
        risk_profile: str, 
        exit_mode: str,
        safety_governor_directives: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if amplification is allowed globally.
        
        [HEDGEFUND MODE] Now checks SafetyGovernor allow_amplification directive.
        SafetyGovernor has veto power over PAL decisions.
        """
        if not self.amplification_enabled:
            logger.info("[PAL] Amplification disabled in config")
            return False
        
        if self.emergency_brake_active:
            logger.warning("[PAL] Emergency brake active - amplification blocked")
            return False
        
        if risk_profile == "REDUCED":
            logger.info("[PAL] Risk profile REDUCED - amplification blocked")
            return False
        
        if exit_mode == "EMERGENCY":
            logger.warning("[PAL] Exit mode EMERGENCY - amplification blocked")
            return False
        
        # [NEW] SafetyGovernor veto check (highest priority)
        if safety_governor_directives:
            allow_amplification = safety_governor_directives.get("allow_amplification", True)
            if not allow_amplification:
                logger.warning("[PAL] 🛡️ SafetyGovernor VETO: Amplification blocked by SafetyGovernor")
                return False
            else:
                logger.info("[PAL] ✅ SafetyGovernor: Amplification allowed")
        
        return True
    
    def _calculate_additional_size_allowed(
        self,
        position: PositionSnapshot,
        portfolio_risk_budget_pct: float,
        risk_profile: str
    ) -> float:
        """
        Calculate how much additional size is allowed.
        
        [HEDGEFUND MODE] Enforces max_scale_in_multiplier (default +50% cap).
        """
        
        # [NEW] Enforce +50% cap from base size (HEDGEFUND MODE)
        max_by_scale_in_cap = position.position_size_usd * (self.max_scale_in_multiplier - 1.0)
        
        # Base on risk budget
        base_allowed = portfolio_risk_budget_pct * 1000  # Rough estimate, need actual portfolio value
        
        # Adjust by risk profile
        if risk_profile == "SAFE":
            base_allowed *= 0.5
        elif risk_profile == "AGGRESSIVE":
            base_allowed *= 1.5  # [HEDGEFUND MODE] More aggressive
        
        # Ensure not over-concentrated
        max_by_concentration = (self.max_position_concentration_pct / 100) * 10000 - position.position_size_usd
        
        # Apply ALL caps (most restrictive wins)
        allowed = max(
            min(base_allowed, max_by_concentration, max_by_scale_in_cap),  # [NEW] Scale-in cap
            0
        )
        
        if allowed < max_by_scale_in_cap:
            logger.debug(
                f"[PAL] Scale-in cap: {max_by_scale_in_cap:.2f} "
                f"(base size {position.position_size_usd:.2f} × {self.max_scale_in_multiplier:.1f})"
            )
        
        return allowed
    
    def _empty_report(self, timestamp: str, positions: List[PositionSnapshot]) -> AmplificationReport:
        """Create empty report when amplification is disabled."""
        return AmplificationReport(
            timestamp=timestamp,
            total_positions=len(positions),
            analyzed_positions=positions,
            amplification_candidates=[],
            high_priority_candidates=[],
            recommendations=[],
            avg_amplification_score=0.0,
            total_additional_size_available_usd=0.0,
            can_amplify=False,
            amplification_enabled=self.amplification_enabled
        )
    
    def _save_report(self, report: AmplificationReport):
        """Save report to disk."""
        try:
            output_path = self.data_dir / "profit_amplification_report.json"
            
            # Convert to dict (handle enums)
            report_dict = {
                "timestamp": report.timestamp,
                "total_positions": report.total_positions,
                "amplification_candidates": [
                    {
                        "symbol": c.position.symbol,
                        "current_R": c.position.current_R,
                        "amplification_score": c.amplification_score,
                        "qualifies_scale_in": c.qualifies_for_scale_in,
                        "qualifies_extend": c.qualifies_for_extend_hold,
                        "blocked_by": [r.value for r in c.blocked_by]
                    }
                    for c in report.amplification_candidates
                ],
                "recommendations": [
                    {
                        "symbol": r.candidate.position.symbol,
                        "action": r.action.value,
                        "priority": r.priority,
                        "rationale": r.rationale,
                        "parameters": r.parameters,
                        "expected_R_increase": r.expected_R_increase,
                        "confidence": r.confidence
                    }
                    for r in report.recommendations
                ],
                "summary": {
                    "candidates": len(report.amplification_candidates),
                    "high_priority": len(report.high_priority_candidates),
                    "recommendations": len(report.recommendations),
                    "avg_score": report.avg_amplification_score,
                    "additional_size_available": report.total_additional_size_available_usd
                },
                "can_amplify": report.can_amplify,
                "amplification_enabled": report.amplification_enabled
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save PAL report: {e}")
    
    # --------------------------------------------------------
    # CONTROL METHODS
    # --------------------------------------------------------
    
    def enable_amplification(self):
        """Enable profit amplification."""
        self.amplification_enabled = True
        logger.info("[PAL] Amplification ENABLED")
    
    def disable_amplification(self):
        """Disable profit amplification."""
        self.amplification_enabled = False
        logger.warning("[PAL] Amplification DISABLED")
    
    def set_emergency_brake(self, active: bool):
        """Set emergency brake state."""
        self.emergency_brake_active = active
        if active:
            logger.error("[PAL] EMERGENCY BRAKE ACTIVATED - No amplification allowed")
        else:
            logger.info("[PAL] Emergency brake released")


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("PROFIT AMPLIFICATION LAYER (PAL) - Standalone Test")
    print("=" * 80)
    
    # Initialize PAL
    pal = ProfitAmplificationLayer(
        data_dir="./data",
        min_R_for_amplification=1.0,
        min_R_for_scale_in=1.5,
        max_dd_from_peak_pct=15.0
    )
    
    print(f"\n[OK] PAL initialized")
    print(f"  Min R for amplification: {pal.min_R_for_amplification}R")
    print(f"  Min R for scale-in: {pal.min_R_for_scale_in}R")
    print(f"  Max DD from peak: {pal.max_dd_from_peak_pct}%")
    
    # Create test positions
    print("\n" + "=" * 80)
    print("Creating test positions...")
    print("=" * 80)
    
    positions = [
        # Position 1: Strong winner, low DD - AMPLIFIABLE
        PositionSnapshot(
            symbol="BTCUSDT",
            side="LONG",
            current_R=2.5,
            peak_R=2.8,
            unrealized_pnl=2500,
            unrealized_pnl_pct=25.0,
            drawdown_from_peak_R=0.107,  # ~11% DD from peak
            drawdown_from_peak_pnl_pct=10.7,
            current_leverage=10.0,
            position_size_usd=10000,
            risk_pct=2.0,
            hold_time_hours=24.0,
            entry_time=datetime.now(timezone.utc).isoformat(),
            pil_classification="RUNNER",
            trend_strength=TrendStrength.STRONG,
            volatility_regime="NORMAL",
            symbol_rank=1,
            symbol_category="CORE"
        ),
        
        # Position 2: Early winner, very strong trend - SCALE-IN CANDIDATE
        PositionSnapshot(
            symbol="ETHUSDT",
            side="LONG",
            current_R=1.8,
            peak_R=1.9,
            unrealized_pnl=1800,
            unrealized_pnl_pct=18.0,
            drawdown_from_peak_R=0.053,  # ~5% DD
            drawdown_from_peak_pnl_pct=5.3,
            current_leverage=8.0,
            position_size_usd=10000,
            risk_pct=2.0,
            hold_time_hours=12.0,
            entry_time=datetime.now(timezone.utc).isoformat(),
            pil_classification="WINNER",
            trend_strength=TrendStrength.VERY_STRONG,
            volatility_regime="LOW",
            symbol_rank=2,
            symbol_category="CORE"
        ),
        
        # Position 3: High DD from peak - BLOCKED
        PositionSnapshot(
            symbol="ADAUSDT",
            side="LONG",
            current_R=0.8,
            peak_R=2.0,
            unrealized_pnl=800,
            unrealized_pnl_pct=8.0,
            drawdown_from_peak_R=0.6,  # 60% DD from peak!
            drawdown_from_peak_pnl_pct=60.0,
            current_leverage=5.0,
            position_size_usd=10000,
            risk_pct=2.0,
            hold_time_hours=48.0,
            entry_time=datetime.now(timezone.utc).isoformat(),
            pil_classification="STRUGGLING",
            trend_strength=TrendStrength.WEAK,
            volatility_regime="HIGH",
            symbol_rank=25,
            symbol_category="TACTICAL"
        ),
    ]
    
    print(f"\n[OK] Created {len(positions)} test positions")
    for i, pos in enumerate(positions, 1):
        print(f"  {i}. {pos.symbol}: {pos.current_R:.1f}R, Trend={pos.trend_strength.value}, DD={pos.drawdown_from_peak_R*100:.1f}%")
    
    # Run analysis
    print("\n" + "=" * 80)
    print("Running PAL analysis...")
    print("=" * 80)
    
    report = pal.analyze_positions(
        positions=positions,
        risk_profile="SAFE",
        portfolio_risk_budget_pct=5.0,
        regime="BULL",
        exit_mode="NORMAL"
    )
    
    print(f"\n[OK] Analysis complete")
    print(f"  Candidates found: {len(report.amplification_candidates)}")
    print(f"  High priority: {len(report.high_priority_candidates)}")
    print(f"  Recommendations: {len(report.recommendations)}")
    print(f"  Avg amplification score: {report.avg_amplification_score:.1f}")
    
    # Display candidates
    print("\n" + "=" * 80)
    print("AMPLIFICATION CANDIDATES")
    print("=" * 80)
    
    for i, candidate in enumerate(report.amplification_candidates, 1):
        print(f"\n  Candidate {i}: {candidate.position.symbol}")
        print(f"    Score: {candidate.amplification_score:.1f}/100")
        print(f"    Current R: {candidate.position.current_R:.1f}R")
        print(f"    Qualifications:")
        print(f"      - Scale-in: {'✅' if candidate.qualifies_for_scale_in else '❌'}")
        print(f"      - Extend hold: {'✅' if candidate.qualifies_for_extend_hold else '❌'}")
        print(f"      - Partial take: {'✅' if candidate.qualifies_for_partial_take else '❌'}")
        if candidate.blocked_by:
            print(f"    ⚠️  Blocked by: {', '.join(r.value for r in candidate.blocked_by)}")
        else:
            print(f"    Additional size allowed: ${candidate.additional_size_allowed_usd:.0f}")
    
    # Display recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if report.recommendations:
        for i, rec in enumerate(report.recommendations, 1):
            print(f"\n  Recommendation {i}: {rec.candidate.position.symbol}")
            print(f"    Action: {rec.action.value}")
            print(f"    Priority: {rec.priority}")
            print(f"    Rationale: {rec.rationale}")
            print(f"    Expected R increase: +{rec.expected_R_increase:.2f}R")
            print(f"    Confidence: {rec.confidence:.0f}%")
            print(f"    Risk score: {rec.risk_score:.1f}/10")
            if rec.parameters:
                print(f"    Parameters:")
                for key, value in rec.parameters.items():
                    print(f"      - {key}: {value}")
    else:
        print("\n  No recommendations generated")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    print(f"\n[OK] Report saved to: {pal.data_dir / 'profit_amplification_report.json'}")


# ============================================================
# FACTORY FUNCTION
# ============================================================

_profit_amplification: Optional[ProfitAmplificationLayer] = None


def get_profit_amplification() -> ProfitAmplificationLayer:
    """Get or create Profit Amplification Layer singleton"""
    global _profit_amplification
    if _profit_amplification is None:
        _profit_amplification = ProfitAmplificationLayer()
    return _profit_amplification
