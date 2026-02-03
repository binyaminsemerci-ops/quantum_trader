#!/usr/bin/env python3
"""
SAFETY GOVERNOR - GLOBAL SAFETY LAYER
======================================

The ultimate safety layer that sits ABOVE all AI systems and enforces
strict risk controls to prevent catastrophic losses.

MIGRATION STEP 1: Risk parameters now driven by PolicyStore v2.
All hardcoded safety thresholds replaced with dynamic policy reads.

PRIORITY HIERARCHY (highest to lowest):
1. Self-Healing (system health, safety policies)
2. Advanced Risk Manager (drawdown, emergency brake)
3. AI-HFOS (supreme coordinator)
4. Portfolio Balancer (portfolio constraints)
5. Profit Amplification Layer (opportunity enhancement)

The Safety Governor DOES NOT disable AI-OS subsystems.
It WRAPS them with additional guardrails and overrides when necessary.

Author: Quantum Trader AI Team
Date: November 23, 2025
Version: 1.0 (Architecture v2 Migration)
"""

import json
import logging
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# [NEW] ARCHITECTURE V2: Import PolicyStore and logger
from backend.core.policy_store import PolicyStore, get_policy_store
from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class GovernorDecision(Enum):
    """Safety Governor decision types"""
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    MODIFY = "MODIFY"
    DOWNGRADE = "DOWNGRADE"


class SafetyLevel(Enum):
    """System safety levels"""
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
    EMERGENCY = "EMERGENCY"


class BlockReason(Enum):
    """Reasons for blocking trades"""
    SELF_HEALING_NO_NEW_TRADES = "SELF_HEALING_NO_NEW_TRADES"
    SELF_HEALING_DEFENSIVE_EXIT = "SELF_HEALING_DEFENSIVE_EXIT"
    SELF_HEALING_EMERGENCY_SHUTDOWN = "SELF_HEALING_EMERGENCY_SHUTDOWN"
    
    RISK_MANAGER_EMERGENCY_BRAKE = "RISK_MANAGER_EMERGENCY_BRAKE"
    RISK_MANAGER_DRAWDOWN_LIMIT = "RISK_MANAGER_DRAWDOWN_LIMIT"
    RISK_MANAGER_LOSING_STREAK = "RISK_MANAGER_LOSING_STREAK"
    
    HFOS_DISALLOW_NEW_TRADES = "HFOS_DISALLOW_NEW_TRADES"
    HFOS_REDUCE_GLOBAL_RISK = "HFOS_REDUCE_GLOBAL_RISK"
    
    PBA_PORTFOLIO_CONSTRAINTS = "PBA_PORTFOLIO_CONSTRAINTS"
    PBA_EXPOSURE_LIMIT = "PBA_EXPOSURE_LIMIT"
    
    PAL_AMPLIFICATION_DISABLED = "PAL_AMPLIFICATION_DISABLED"
    
    GOVERNOR_SAFETY_OVERRIDE = "GOVERNOR_SAFETY_OVERRIDE"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SafetyGovernorDirectives:
    """
    Global safety directives enforced by the Governor.
    These override all subsystem recommendations.
    """
    timestamp: str
    safety_level: SafetyLevel
    
    # Core trading controls
    global_allow_new_trades: bool = True
    global_allow_position_expansion: bool = True
    
    # Risk multipliers (0.0 to 2.0)
    max_leverage_multiplier: float = 1.0
    max_position_size_multiplier: float = 1.0
    max_total_exposure_multiplier: float = 1.0
    
    # Subsystem controls
    allow_amplification: bool = True
    allow_expansion_symbols: bool = True
    force_defensive_exits: bool = False
    emergency_exit_all: bool = False
    
    # Reasons (for transparency)
    primary_reason: str = "NORMAL_OPERATION"
    active_constraints: List[str] = field(default_factory=list)
    overridden_subsystems: List[str] = field(default_factory=list)


@dataclass
class SubsystemInput:
    """Input from a subsystem for Governor decision"""
    subsystem_name: str
    priority: int  # Lower = higher priority
    
    # Recommendations
    allow_new_trades: bool = True
    recommended_leverage_multiplier: float = 1.0
    recommended_size_multiplier: float = 1.0
    
    # Context
    reason: str = ""
    urgency: str = "NORMAL"  # NORMAL, HIGH, CRITICAL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernorDecisionRecord:
    """Record of a Governor decision"""
    timestamp: str
    decision: GovernorDecision
    
    # Request context
    requested_action: str  # NEW_TRADE, EXPAND_POSITION, CLOSE_POSITION
    symbol: str
    size: float
    
    # Decision details
    allowed: bool
    modified: bool
    original_size: float
    final_size: float
    original_leverage: float
    final_leverage: float
    
    # Reasoning
    block_reason: Optional[BlockReason] = None
    reason_detail: str = ""
    subsystem_priority_used: str = ""
    
    # Transparency
    subsystem_votes: Dict[str, bool] = field(default_factory=dict)
    applied_multipliers: Dict[str, float] = field(default_factory=dict)


@dataclass
class GovernorStats:
    """Statistics tracking Governor interventions"""
    start_time: str
    
    # Counters
    total_decisions: int = 0
    trades_allowed: int = 0
    trades_blocked: int = 0
    trades_modified: int = 0
    
    # Breakdown by reason
    blocks_by_reason: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Mode changes
    mode_downgrades: int = 0
    mode_upgrades: int = 0
    
    # Subsystem overrides
    subsystem_override_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Risky symbols
    blocked_symbols: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    modified_symbols: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ============================================================================
# SAFETY GOVERNOR CORE
# ============================================================================

class SafetyGovernor:
    """
    Global Safety Governor - The ultimate safety layer.
    
    Responsibilities:
    - Collect inputs from all subsystems (Self-Healing, Risk Manager, AI-HFOS, PBA, PAL)
    - Apply priority hierarchy to resolve conflicts
    - Enforce global safety directives
    - Block/modify trades based on comprehensive risk assessment
    - Provide transparency on all decisions
    - Track intervention patterns
    
    The Governor does NOT disable AI systems - it wraps them with guardrails.
    """
    
    def __init__(
        self,
        data_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        policy_store: Optional[PolicyStore] = None,  # [NEW] PolicyStore for dynamic limits
        enable_active_slots: bool = True,  # [NEW] Enable Active Positions Controller
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # [NEW] Store PolicyStore reference
        self.policy_store = policy_store
        
        # [NEW] Initialize Active Positions Controller
        self.active_slots_controller = None
        if enable_active_slots and policy_store:
            from backend.services.risk.active_positions_controller import (
                ActivePositionsController,
                SlotConfig,
            )
            self.active_slots_controller = ActivePositionsController(
                policy_store=policy_store,
                config=SlotConfig(),
            )
            logger.info(
                "safety_governor_active_slots_enabled",
                message="Active Positions Controller initialized",
            )
        elif enable_active_slots:
            logger.warning(
                "safety_governor_active_slots_disabled",
                message="Active Slots enabled but PolicyStore not available",
            )
        
        # Configuration (will be enhanced with PolicyStore reads)
        self.config = config or self._default_config()
        
        # Current directives
        self.current_directives: Optional[SafetyGovernorDirectives] = None
        
        # Statistics
        self.stats = GovernorStats(
            start_time=datetime.now(timezone.utc).isoformat()
        )
        
        # Decision history (recent decisions only, for pattern detection)
        self.decision_history: List[GovernorDecisionRecord] = []
        self.max_history_size = 1000
        
        # Subsystem state cache
        self.subsystem_states: Dict[str, SubsystemInput] = {}
        
        if self.policy_store:
            logger.info(
                "safety_governor_initialized",
                policy_store_enabled=True,
                message="PolicyStore integration enabled - dynamic risk thresholds active",
            )
        else:
            logger.warning(
                "safety_governor_initialized",
                policy_store_enabled=False,
                message="PolicyStore not available - using fallback hardcoded thresholds",
            )
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration with PolicyStore fallback values.
        
        MIGRATION STEP 1: These are fallback values only.
        Actual values should be read from PolicyStore v2 RiskProfile at runtime.
        """
        return {
            # Drawdown thresholds (FALLBACK - read from PolicyStore in production)
            "max_daily_drawdown_pct": 5.0,  # Will use RiskProfile.max_daily_drawdown_pct
            "emergency_drawdown_pct": 8.0,  # 1.6x max_daily_drawdown_pct
            
            # Losing streak thresholds
            "max_losing_streak": 5,
            "critical_losing_streak": 8,
            
            # Risk multipliers by safety level
            "safety_multipliers": {
                "NORMAL": {
                    "leverage": 1.0,
                    "position_size": 1.0,
                    "exposure": 1.0
                },
                "CAUTIOUS": {
                    "leverage": 0.75,
                    "position_size": 0.75,
                    "exposure": 0.85
                },
                "DEFENSIVE": {
                    "leverage": 0.5,
                    "position_size": 0.5,
                    "exposure": 0.6
                },
                "EMERGENCY": {
                    "leverage": 0.0,
                    "position_size": 0.0,
                    "exposure": 0.0
                }
            },
            
            # Update interval
            "update_interval_seconds": 60
        }
    
    # ========================================================================
    # SUBSYSTEM INPUT COLLECTION
    # ========================================================================
    
    def collect_self_healing_input(self, self_healing_state: Dict[str, Any]) -> SubsystemInput:
        """
        Collect input from Self-Healing system (HIGHEST PRIORITY).
        
        Args:
            self_healing_state: Self-Healing report including safety_policy
            
        Returns:
            SubsystemInput with Self-Healing recommendations
        """
        safety_policy = self_healing_state.get("safety_policy", "ALLOW_ALL")
        overall_status = self_healing_state.get("overall_status", "UNKNOWN")
        
        input_data = SubsystemInput(
            subsystem_name="SELF_HEALING",
            priority=1  # HIGHEST PRIORITY
        )
        
        # Map safety policy to recommendations
        if safety_policy == "EMERGENCY_SHUTDOWN":
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = "Emergency shutdown - critical system failure"
            input_data.urgency = "CRITICAL"
        
        elif safety_policy == "NO_NEW_TRADES":
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = "No new trades - system health degraded"
            input_data.urgency = "HIGH"
        
        elif safety_policy == "DEFENSIVE_EXIT":
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.5
            input_data.recommended_size_multiplier = 0.5
            input_data.reason = "Defensive exit - reducing exposure"
            input_data.urgency = "HIGH"
        
        elif safety_policy == "SAFE_RISK_PROFILE":
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 0.75
            input_data.recommended_size_multiplier = 0.75
            input_data.reason = "Safe risk profile - conservative mode"
            input_data.urgency = "NORMAL"
        
        else:  # ALLOW_ALL
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = "All systems healthy"
            input_data.urgency = "NORMAL"
        
        input_data.metadata = {
            "overall_status": overall_status,
            "safety_policy": safety_policy
        }
        
        return input_data
    
    async def collect_risk_manager_input_async(self, risk_state: Dict[str, Any]) -> SubsystemInput:
        """
        Collect input from Advanced Risk Manager (PRIORITY 2).
        
        MIGRATION STEP 1: Now reads max_daily_drawdown_pct from PolicyStore v2 RiskProfile.
        
        Args:
            risk_state: Risk Manager state including drawdown, emergency_brake
            
        Returns:
            SubsystemInput with Risk Manager recommendations
        """
        emergency_brake = risk_state.get("emergency_brake_active", False)
        daily_dd_pct = risk_state.get("daily_dd_pct", 0.0)
        losing_streak = risk_state.get("losing_streak", 0)
        
        # [ARCHITECTURE V2] READ DYNAMIC RISK PROFILE
        max_dd_pct = self.config["max_daily_drawdown_pct"]  # Fallback
        emergency_dd_pct = self.config["emergency_drawdown_pct"]  # Fallback
        profile_name = "FALLBACK_NORMAL"
        
        if self.policy_store:
            try:
                if hasattr(self.policy_store, 'get_active_risk_profile'):
                    risk_profile = await self.policy_store.get_active_risk_profile()
                    max_dd_pct = risk_profile.max_daily_drawdown_pct
                    emergency_dd_pct = max_dd_pct * 1.6  # Emergency is 160% of max
                    profile_name = risk_profile.name
                    
                    logger.debug(
                        "safety_governor_risk_profile_loaded",
                        profile_name=profile_name,
                        max_daily_drawdown_pct=max_dd_pct,
                        emergency_drawdown_pct=emergency_dd_pct,
                    )
                else:
                    logger.debug("safety_governor_policystore_no_method", method="get_active_risk_profile")
            except Exception as e:
                logger.error(
                    "safety_governor_policystore_error",
                    error=str(e),
                    fallback="using hardcoded max_daily_drawdown_pct",
                )
        
        input_data = SubsystemInput(
            subsystem_name="RISK_MANAGER",
            priority=2  # SECOND PRIORITY
        )
        
        # Emergency brake activated
        if emergency_brake:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = f"Emergency brake active (profile: {profile_name})"
            input_data.urgency = "CRITICAL"
        
        # Critical drawdown
        elif abs(daily_dd_pct) >= emergency_dd_pct:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = f"Critical drawdown: {daily_dd_pct:.1f}% >= {emergency_dd_pct:.1f}% (profile: {profile_name})"
            input_data.urgency = "CRITICAL"
            
            logger.critical(
                "safety_governor_critical_drawdown",
                profile_name=profile_name,
                daily_dd_pct=daily_dd_pct,
                emergency_dd_pct=emergency_dd_pct,
            )
        
        # High drawdown
        elif abs(daily_dd_pct) >= max_dd_pct:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.5
            input_data.recommended_size_multiplier = 0.5
            input_data.reason = f"High drawdown: {daily_dd_pct:.1f}% >= {max_dd_pct:.1f}% (profile: {profile_name})"
            input_data.urgency = "HIGH"
            
            logger.warning(
                "safety_governor_high_drawdown",
                profile_name=profile_name,
                daily_dd_pct=daily_dd_pct,
                max_dd_pct=max_dd_pct,
            )
        
        # Critical losing streak
        elif losing_streak >= self.config["critical_losing_streak"]:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.5
            input_data.recommended_size_multiplier = 0.5
            input_data.reason = f"Critical losing streak: {losing_streak} (profile: {profile_name})"
            input_data.urgency = "HIGH"
        
        # High losing streak
        elif losing_streak >= self.config["max_losing_streak"]:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 0.75
            input_data.recommended_size_multiplier = 0.75
            input_data.reason = f"High losing streak: {losing_streak} (profile: {profile_name})"
            input_data.urgency = "NORMAL"
        
        # Warning level drawdown
        elif abs(daily_dd_pct) >= max_dd_pct * 0.7:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 0.85
            input_data.recommended_size_multiplier = 0.85
            input_data.reason = f"Elevated drawdown: {daily_dd_pct:.1f}% (profile: {profile_name})"
            input_data.urgency = "NORMAL"
        
        else:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = f"Risk metrics within limits (profile: {profile_name})"
            input_data.urgency = "NORMAL"
        
        input_data.metadata = {
            "emergency_brake": emergency_brake,
            "daily_dd_pct": daily_dd_pct,
            "losing_streak": losing_streak,
            "max_dd_pct": max_dd_pct,
            "emergency_dd_pct": emergency_dd_pct,
            "risk_profile": profile_name,
        }
        
        logger.info(
            "safety_governor_risk_manager_input",
            profile_name=profile_name,
            daily_dd_pct=daily_dd_pct,
            max_dd_pct=max_dd_pct,
            emergency_dd_pct=emergency_dd_pct,
            losing_streak=losing_streak,
            allow_new_trades=input_data.allow_new_trades,
            leverage_multiplier=input_data.recommended_leverage_multiplier,
        )
        
        return input_data
    
    def collect_risk_manager_input(self, risk_state: Dict[str, Any]) -> SubsystemInput:
        """
        Synchronous wrapper for collect_risk_manager_input_async.
        
        DEPRECATED: Use collect_risk_manager_input_async for PolicyStore v2 integration.
        This method uses fallback hardcoded values.
        """
        logger.warning(
            "safety_governor_sync_method_called",
            message="Using deprecated sync method - PolicyStore integration disabled",
        )
        logger.warning(
            "safety_governor_sync_method_called",
            message="Using deprecated sync method - PolicyStore integration disabled",
        )
        
        # Original sync logic (fallback only)
        emergency_brake = risk_state.get("emergency_brake_active", False)
        daily_dd_pct = risk_state.get("daily_dd_pct", 0.0)
        max_dd_pct = risk_state.get("max_daily_dd_pct", 5.0)
        losing_streak = risk_state.get("losing_streak", 0)
        
        input_data = SubsystemInput(
            subsystem_name="RISK_MANAGER",
            priority=2  # SECOND PRIORITY
        )
        
        # Emergency brake activated
        if emergency_brake:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = "Emergency brake active"
            input_data.urgency = "CRITICAL"
        
        # Critical drawdown
        elif abs(daily_dd_pct) >= self.config["emergency_drawdown_pct"]:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = f"Critical drawdown: {daily_dd_pct:.1f}%"
            input_data.urgency = "CRITICAL"
        
        # High drawdown
        elif abs(daily_dd_pct) >= self.config["max_daily_drawdown_pct"]:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.5
            input_data.recommended_size_multiplier = 0.5
            input_data.reason = f"High drawdown: {daily_dd_pct:.1f}%"
            input_data.urgency = "HIGH"
        
        # Critical losing streak
        elif losing_streak >= self.config["critical_losing_streak"]:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.5
            input_data.recommended_size_multiplier = 0.5
            input_data.reason = f"Critical losing streak: {losing_streak}"
            input_data.urgency = "HIGH"
        
        # High losing streak
        elif losing_streak >= self.config["max_losing_streak"]:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 0.75
            input_data.recommended_size_multiplier = 0.75
            input_data.reason = f"High losing streak: {losing_streak}"
            input_data.urgency = "NORMAL"
        
        # Warning level drawdown
        elif abs(daily_dd_pct) >= max_dd_pct * 0.7:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 0.85
            input_data.recommended_size_multiplier = 0.85
            input_data.reason = f"Elevated drawdown: {daily_dd_pct:.1f}%"
            input_data.urgency = "NORMAL"
        
        else:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = "Risk metrics within limits"
            input_data.urgency = "NORMAL"
        
        input_data.metadata = {
            "emergency_brake": emergency_brake,
            "daily_dd_pct": daily_dd_pct,
            "losing_streak": losing_streak
        }
        
        return input_data
    
    def collect_hfos_input(self, hfos_output: Any) -> SubsystemInput:
        """
        Collect input from AI-HFOS (PRIORITY 3).
        
        Args:
            hfos_output: AIHFOSOutput object with global_directives
            
        Returns:
            SubsystemInput with AI-HFOS recommendations
        """
        directives = hfos_output.global_directives
        risk_mode = hfos_output.supreme_decision.risk_mode
        
        input_data = SubsystemInput(
            subsystem_name="AI_HFOS",
            priority=3  # THIRD PRIORITY
        )
        
        # Map HFOS directives
        input_data.allow_new_trades = directives.allow_new_trades and directives.allow_new_positions
        input_data.recommended_leverage_multiplier = directives.scale_position_sizes
        input_data.recommended_size_multiplier = directives.scale_position_sizes
        
        # Determine urgency based on risk mode
        if directives.reduce_global_risk:
            input_data.urgency = "HIGH"
            input_data.reason = f"AI-HFOS reducing risk - Mode: {risk_mode.value}"
        elif not directives.allow_new_trades:
            input_data.urgency = "HIGH"
            input_data.reason = f"AI-HFOS blocking new trades - Mode: {risk_mode.value}"
        else:
            input_data.urgency = "NORMAL"
            input_data.reason = f"AI-HFOS active - Mode: {risk_mode.value}"
        
        input_data.metadata = {
            "risk_mode": risk_mode.value,
            "conflicts_detected": hfos_output.supreme_decision.conflicts_detected,
            "confidence": hfos_output.supreme_decision.confidence
        }
        
        return input_data
    
    def collect_pba_input(self, pba_violations: List[str], portfolio_state: Dict[str, Any]) -> SubsystemInput:
        """
        Collect input from Portfolio Balancer (PRIORITY 4).
        
        Args:
            pba_violations: List of portfolio constraint violations
            portfolio_state: Current portfolio state
            
        Returns:
            SubsystemInput with PBA recommendations
        """
        input_data = SubsystemInput(
            subsystem_name="PORTFOLIO_BALANCER",
            priority=4  # FOURTH PRIORITY
        )
        
        # Check for critical violations
        critical_violations = [v for v in pba_violations if "CRITICAL" in v or "EXCEED" in v]
        
        if critical_violations:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.75
            input_data.recommended_size_multiplier = 0.75
            input_data.reason = f"Portfolio constraints violated: {len(critical_violations)} critical"
            input_data.urgency = "HIGH"
        
        elif pba_violations:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 0.9
            input_data.recommended_size_multiplier = 0.9
            input_data.reason = f"Portfolio warnings: {len(pba_violations)}"
            input_data.urgency = "NORMAL"
        
        else:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = "Portfolio balanced"
            input_data.urgency = "NORMAL"
        
        input_data.metadata = {
            "violations": pba_violations,
            "total_positions": portfolio_state.get("total_positions", 0),
            "total_exposure_pct": portfolio_state.get("gross_exposure", 0.0)
        }
        
        return input_data
    
    async def collect_active_slots_input_async(
        self,
        symbol: str,
        action: str,
        candidate_score: float,
        candidate_returns: Optional[List[float]],
        market_features: Dict[str, float],
        portfolio_positions: List[Any],
        total_margin_usage_pct: float,
        active_slots_controller: Optional[Any] = None,
    ) -> SubsystemInput:
        """
        Collect input from Active Positions Controller (PRIORITY 2.5 - between Risk Manager and HFOS).
        
        Enforces:
        - Dynamic slot allocation (3-6 based on regime)
        - Capital rotation (close weakest for better)
        - Correlation caps (prevents >80% corr)
        - Margin caps (prevents >65% usage)
        - Policy-driven (no hardcoded symbols)
        
        Args:
            symbol: Trading symbol
            action: NEW_TRADE or EXPAND_POSITION
            candidate_score: Quality score for candidate [0..100]
            candidate_returns: Log-returns series for correlation check
            market_features: Features for regime detection
            portfolio_positions: List of current positions
            total_margin_usage_pct: Current margin usage %
            active_slots_controller: Instance of ActivePositionsController
            
        Returns:
            SubsystemInput with Active Slots recommendations
        """
        from backend.services.risk.active_positions_controller import (
            ActivePositionsController,
            SlotDecision,
        )
        
        input_data = SubsystemInput(
            subsystem_name="ACTIVE_SLOTS",
            priority=2.5  # Between RISK_MANAGER (2) and AI_HFOS (3)
        )
        
        # If no controller provided, allow (backward compatible)
        if active_slots_controller is None:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = "Active Slots Controller not enabled"
            input_data.urgency = "NORMAL"
            return input_data
        
        # Evaluate position request
        decision, record = await active_slots_controller.evaluate_position_request(
            symbol=symbol,
            action=action,
            candidate_score=candidate_score,
            candidate_returns=candidate_returns,
            market_features=market_features,
            portfolio_positions=portfolio_positions,
            total_margin_usage_pct=total_margin_usage_pct,
        )
        
        # Map decision to subsystem input
        if decision == SlotDecision.ALLOW:
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = f"Slots available ({record.open_positions_count}/{record.desired_slots})"
            input_data.urgency = "NORMAL"
        
        elif decision == SlotDecision.ROTATION_TRIGGERED:
            # Rotation: close weakest, allow new
            input_data.allow_new_trades = True
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = f"Rotation: close {record.weakest_symbol} for {symbol}"
            input_data.urgency = "NORMAL"
            input_data.metadata["rotation_close_symbol"] = record.weakest_symbol
            input_data.metadata["rotation_triggered"] = True
        
        elif decision == SlotDecision.BLOCKED_SLOTS_FULL:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = f"Slots full: {record.open_positions_count}/{record.desired_slots} (regime: {record.regime.value})"
            input_data.urgency = "NORMAL"
        
        elif decision == SlotDecision.BLOCKED_NO_POLICY:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = f"No policy: {record.block_reason}"
            input_data.urgency = "CRITICAL"
        
        elif decision == SlotDecision.BLOCKED_CORRELATION:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 1.0
            input_data.recommended_size_multiplier = 1.0
            input_data.reason = f"Correlation {record.correlation_with_portfolio:.2f} > 0.80"
            input_data.urgency = "NORMAL"
        
        elif decision == SlotDecision.BLOCKED_MARGIN:
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.5
            input_data.recommended_size_multiplier = 0.5
            input_data.reason = f"Margin usage {record.margin_usage_pct:.1f}% > 65%"
            input_data.urgency = "HIGH"
        
        else:
            # Fallback
            input_data.allow_new_trades = False
            input_data.recommended_leverage_multiplier = 0.0
            input_data.recommended_size_multiplier = 0.0
            input_data.reason = f"Unknown decision: {decision}"
            input_data.urgency = "HIGH"
        
        input_data.metadata = {
            "decision": decision.value,
            "desired_slots": record.desired_slots,
            "open_positions_count": record.open_positions_count,
            "regime": record.regime.value,
            "candidate_score": candidate_score,
        }
        
        logger.info(
            "safety_governor_active_slots_input",
            symbol=symbol,
            decision=decision.value,
            desired_slots=record.desired_slots,
            open_count=record.open_positions_count,
            regime=record.regime.value,
            allow_new_trades=input_data.allow_new_trades,
        )
        
        return input_data
    
    # ========================================================================
    # DECISION ENGINE
    # ========================================================================
    
    def compute_directives(
        self,
        subsystem_inputs: List[SubsystemInput]
    ) -> SafetyGovernorDirectives:
        """
        Compute global safety directives based on all subsystem inputs.
        
        Priority hierarchy:
        1. Self-Healing (system health)
        2. Risk Manager (drawdown, emergency brake)
        3. AI-HFOS (supreme coordinator)
        4. Portfolio Balancer (portfolio constraints)
        5. Profit Amplification (opportunity enhancement)
        
        Args:
            subsystem_inputs: List of inputs from all subsystems
            
        Returns:
            SafetyGovernorDirectives with final decisions
        """
        # Sort by priority (lower number = higher priority)
        sorted_inputs = sorted(subsystem_inputs, key=lambda x: x.priority)
        
        # Initialize directives
        directives = SafetyGovernorDirectives(
            timestamp=datetime.now(timezone.utc).isoformat(),
            safety_level=SafetyLevel.NORMAL
        )
        
        # Track which subsystems we're listening to
        active_constraints = []
        overridden_subsystems = []
        
        # Apply inputs in priority order
        for idx, input_data in enumerate(sorted_inputs):
            self.subsystem_states[input_data.subsystem_name] = input_data
            
            # Highest urgency subsystem wins for binary decisions
            if input_data.urgency in ["CRITICAL", "HIGH"]:
                # Update binary controls
                if not input_data.allow_new_trades:
                    directives.global_allow_new_trades = False
                    directives.primary_reason = input_data.reason
                    active_constraints.append(f"{input_data.subsystem_name}: {input_data.reason}")
                
                # Override lower priority subsystems
                if idx < len(sorted_inputs) - 1:
                    overridden_subsystems.extend([s.subsystem_name for s in sorted_inputs[idx+1:]])
            
            # Take minimum multiplier across all subsystems (most conservative)
            directives.max_leverage_multiplier = min(
                directives.max_leverage_multiplier,
                input_data.recommended_leverage_multiplier
            )
            directives.max_position_size_multiplier = min(
                directives.max_position_size_multiplier,
                input_data.recommended_size_multiplier
            )
        
        # Determine safety level based on multipliers
        avg_multiplier = (directives.max_leverage_multiplier + directives.max_position_size_multiplier) / 2
        
        if avg_multiplier == 0.0:
            directives.safety_level = SafetyLevel.EMERGENCY
            directives.global_allow_new_trades = False
            directives.allow_amplification = False
            directives.allow_expansion_symbols = False
        elif avg_multiplier <= 0.5:
            directives.safety_level = SafetyLevel.DEFENSIVE
            directives.allow_amplification = False
            directives.allow_expansion_symbols = False
        elif avg_multiplier <= 0.75:
            directives.safety_level = SafetyLevel.CAUTIOUS
            directives.allow_expansion_symbols = False
        else:
            directives.safety_level = SafetyLevel.NORMAL
        
        # Apply additional restrictions in emergency mode
        if directives.safety_level == SafetyLevel.EMERGENCY:
            directives.force_defensive_exits = True
        
        # Set exposure multiplier based on safety level
        directives.max_total_exposure_multiplier = self.config["safety_multipliers"][directives.safety_level.value]["exposure"]
        
        directives.active_constraints = active_constraints
        directives.overridden_subsystems = list(set(overridden_subsystems))
        
        # Update current directives
        self.current_directives = directives
        
        logger.info(
            f"🛡️ [SAFETY GOVERNOR] Directives computed: "
            f"Level={directives.safety_level.value}, "
            f"AllowTrades={directives.global_allow_new_trades}, "
            f"LevMult={directives.max_leverage_multiplier:.2f}, "
            f"SizeMult={directives.max_position_size_multiplier:.2f}"
        )
        
        return directives
    
    def evaluate_trade_request(
        self,
        symbol: str,
        action: str,
        size: float,
        leverage: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[GovernorDecision, GovernorDecisionRecord]:
        """
        Evaluate a trade request against current safety directives.
        
        Args:
            symbol: Trading symbol
            action: NEW_TRADE, EXPAND_POSITION, CLOSE_POSITION
            size: Requested position size
            leverage: Requested leverage
            confidence: Signal confidence
            metadata: Additional context
            
        Returns:
            Tuple of (decision, detailed_record)
        """
        metadata = metadata or {}
        
        if self.current_directives is None:
            logger.error("🛡️ [SAFETY GOVERNOR] No directives available - blocking trade")
            decision = GovernorDecision.BLOCK
            record = self._create_decision_record(
                decision=decision,
                symbol=symbol,
                action=action,
                size=size,
                leverage=leverage,
                allowed=False,
                reason=BlockReason.GOVERNOR_SAFETY_OVERRIDE,
                reason_detail="No safety directives computed yet"
            )
            return decision, record
        
        directives = self.current_directives
        
        # Check global trade permission
        if not directives.global_allow_new_trades and action in ["NEW_TRADE", "EXPAND_POSITION"]:
            decision = GovernorDecision.BLOCK
            reason = self._determine_block_reason()
            
            record = self._create_decision_record(
                decision=decision,
                symbol=symbol,
                action=action,
                size=size,
                leverage=leverage,
                allowed=False,
                reason=reason,
                reason_detail=directives.primary_reason
            )
            
            self._update_stats(record)
            return decision, record
        
        # Check if amplification is allowed (for EXPAND_POSITION actions)
        if action == "EXPAND_POSITION" and not directives.allow_amplification:
            decision = GovernorDecision.BLOCK
            
            record = self._create_decision_record(
                decision=decision,
                symbol=symbol,
                action=action,
                size=size,
                leverage=leverage,
                allowed=False,
                reason=BlockReason.PAL_AMPLIFICATION_DISABLED,
                reason_detail=f"Amplification disabled in {directives.safety_level.value} mode"
            )
            
            self._update_stats(record)
            return decision, record
        
        # Check if expansion symbols are allowed
        symbol_category = metadata.get("category", "EXPANSION")
        if symbol_category == "EXPANSION" and not directives.allow_expansion_symbols:
            decision = GovernorDecision.BLOCK
            
            record = self._create_decision_record(
                decision=decision,
                symbol=symbol,
                action=action,
                size=size,
                leverage=leverage,
                allowed=False,
                reason=BlockReason.GOVERNOR_SAFETY_OVERRIDE,
                reason_detail=f"Expansion symbols blocked in {directives.safety_level.value} mode"
            )
            
            self._update_stats(record)
            return decision, record
        
        # Apply multipliers
        final_size = size * directives.max_position_size_multiplier
        final_leverage = leverage * directives.max_leverage_multiplier
        
        # Determine if trade was modified
        size_modified = abs(final_size - size) > 0.01
        leverage_modified = abs(final_leverage - leverage) > 0.01
        
        if size_modified or leverage_modified:
            decision = GovernorDecision.MODIFY
            
            record = self._create_decision_record(
                decision=decision,
                symbol=symbol,
                action=action,
                size=size,
                leverage=leverage,
                allowed=True,
                modified=True,
                final_size=final_size,
                final_leverage=final_leverage,
                reason_detail=f"Trade modified: {directives.safety_level.value} mode multipliers applied"
            )
        else:
            decision = GovernorDecision.ALLOW
            
            record = self._create_decision_record(
                decision=decision,
                symbol=symbol,
                action=action,
                size=size,
                leverage=leverage,
                allowed=True,
                modified=False,
                final_size=size,
                final_leverage=leverage,
                reason_detail="Trade allowed - all safety checks passed"
            )
        
        self._update_stats(record)
        return decision, record
    
    def _determine_block_reason(self) -> BlockReason:
        """Determine the primary reason for blocking a trade based on subsystem states"""
        # Check in priority order
        if "SELF_HEALING" in self.subsystem_states:
            sh = self.subsystem_states["SELF_HEALING"]
            if not sh.allow_new_trades:
                policy = sh.metadata.get("safety_policy", "")
                if policy == "EMERGENCY_SHUTDOWN":
                    return BlockReason.SELF_HEALING_EMERGENCY_SHUTDOWN
                elif policy == "DEFENSIVE_EXIT":
                    return BlockReason.SELF_HEALING_DEFENSIVE_EXIT
                else:
                    return BlockReason.SELF_HEALING_NO_NEW_TRADES
        
        if "RISK_MANAGER" in self.subsystem_states:
            rm = self.subsystem_states["RISK_MANAGER"]
            if not rm.allow_new_trades:
                if rm.metadata.get("emergency_brake", False):
                    return BlockReason.RISK_MANAGER_EMERGENCY_BRAKE
                elif "drawdown" in rm.reason.lower():
                    return BlockReason.RISK_MANAGER_DRAWDOWN_LIMIT
                elif "streak" in rm.reason.lower():
                    return BlockReason.RISK_MANAGER_LOSING_STREAK
        
        if "AI_HFOS" in self.subsystem_states:
            hfos = self.subsystem_states["AI_HFOS"]
            if not hfos.allow_new_trades:
                return BlockReason.HFOS_DISALLOW_NEW_TRADES
        
        if "PORTFOLIO_BALANCER" in self.subsystem_states:
            pba = self.subsystem_states["PORTFOLIO_BALANCER"]
            if not pba.allow_new_trades:
                return BlockReason.PBA_PORTFOLIO_CONSTRAINTS
        
        return BlockReason.GOVERNOR_SAFETY_OVERRIDE
    
    def _create_decision_record(
        self,
        decision: GovernorDecision,
        symbol: str,
        action: str,
        size: float,
        leverage: float,
        allowed: bool,
        modified: bool = False,
        final_size: Optional[float] = None,
        final_leverage: Optional[float] = None,
        reason: Optional[BlockReason] = None,
        reason_detail: str = ""
    ) -> GovernorDecisionRecord:
        """Create a decision record"""
        record = GovernorDecisionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            decision=decision,
            requested_action=action,
            symbol=symbol,
            size=size,
            allowed=allowed,
            modified=modified,
            original_size=size,
            final_size=final_size or size,
            original_leverage=leverage,
            final_leverage=final_leverage or leverage,
            block_reason=reason,
            reason_detail=reason_detail
        )
        
        # Add subsystem votes
        for name, state in self.subsystem_states.items():
            record.subsystem_votes[name] = state.allow_new_trades
        
        # Add applied multipliers
        if self.current_directives:
            record.applied_multipliers = {
                "leverage": self.current_directives.max_leverage_multiplier,
                "size": self.current_directives.max_position_size_multiplier,
                "exposure": self.current_directives.max_total_exposure_multiplier
            }
        
        # Add to history
        self.decision_history.append(record)
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size:]
        
        return record
    
    def _update_stats(self, record: GovernorDecisionRecord):
        """Update statistics based on decision"""
        self.stats.total_decisions += 1
        
        if record.allowed:
            if record.modified:
                self.stats.trades_modified += 1
                self.stats.modified_symbols[record.symbol] += 1
            else:
                self.stats.trades_allowed += 1
        else:
            self.stats.trades_blocked += 1
            self.stats.blocked_symbols[record.symbol] += 1
            
            if record.block_reason:
                self.stats.blocks_by_reason[record.block_reason.value] += 1
    
    # ========================================================================
    # LOGGING & TRANSPARENCY
    # ========================================================================
    
    def log_decision(self, record: GovernorDecisionRecord):
        """
        Log a Governor decision with full transparency.
        
        Args:
            record: Decision record to log
        """
        if record.allowed:
            if record.modified:
                logger.warning(
                    f"🛡️ [SAFETY GOVERNOR] MODIFIED: {record.requested_action} {record.symbol} | "
                    f"Size: {record.original_size:.2f} → {record.final_size:.2f} | "
                    f"Leverage: {record.original_leverage:.1f}x → {record.final_leverage:.1f}x | "
                    f"Reason: {record.reason_detail}"
                )
            else:
                logger.info(
                    f"🛡️ [SAFETY GOVERNOR] ALLOWED: {record.requested_action} {record.symbol} | "
                    f"Size: {record.final_size:.2f} | Leverage: {record.final_leverage:.1f}x"
                )
        else:
            logger.error(
                f"🛡️ [SAFETY GOVERNOR] ❌ BLOCKED: {record.requested_action} {record.symbol} | "
                f"Size: {record.size:.2f} | Leverage: {record.leverage:.1f}x | "
                f"Reason: {record.block_reason.value if record.block_reason else 'UNKNOWN'} | "
                f"Detail: {record.reason_detail}"
            )
            
            # Log subsystem votes for transparency
            votes_str = ", ".join([f"{k}={'✓' if v else '✗'}" for k, v in record.subsystem_votes.items()])
            logger.info(f"🛡️ [SAFETY GOVERNOR] Subsystem votes: {votes_str}")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive Safety Governor report.
        
        Returns:
            Report dict with statistics, patterns, recommendations
        """
        now = datetime.now(timezone.utc).isoformat()
        
        # Calculate intervention rate
        intervention_rate = 0.0
        if self.stats.total_decisions > 0:
            intervention_rate = ((self.stats.trades_blocked + self.stats.trades_modified) / 
                               self.stats.total_decisions * 100)
        
        # Find most blocked symbols
        top_blocked = sorted(
            self.stats.blocked_symbols.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Find most modified symbols
        top_modified = sorted(
            self.stats.modified_symbols.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        report = {
            "timestamp": now,
            "current_directives": asdict(self.current_directives) if self.current_directives else None,
            
            "statistics": {
                "total_decisions": self.stats.total_decisions,
                "trades_allowed": self.stats.trades_allowed,
                "trades_blocked": self.stats.trades_blocked,
                "trades_modified": self.stats.trades_modified,
                "intervention_rate_pct": intervention_rate,
                "mode_downgrades": self.stats.mode_downgrades,
                "mode_upgrades": self.stats.mode_upgrades
            },
            
            "blocks_by_reason": dict(self.stats.blocks_by_reason),
            
            "risky_symbols": {
                "most_blocked": [{"symbol": s, "count": c} for s, c in top_blocked],
                "most_modified": [{"symbol": s, "count": c} for s, c in top_modified]
            },
            
            "subsystem_overrides": dict(self.stats.subsystem_override_counts),
            
            "recent_decisions": [
                {
                    "timestamp": r.timestamp,
                    "decision": r.decision.value,
                    "symbol": r.symbol,
                    "action": r.requested_action,
                    "allowed": r.allowed,
                    "reason": r.reason_detail
                }
                for r in self.decision_history[-20:]
            ]
        }
        
        # Save report to file
        report_file = self.data_dir / "safety_governor_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(
            f"🛡️ [SAFETY GOVERNOR] Report generated: "
            f"{self.stats.total_decisions} decisions, "
            f"{intervention_rate:.1f}% intervention rate"
        )
        
        return report
    
    async def monitor_loop(self, interval_seconds: int = 60):
        """
        Background monitoring loop for periodic reporting.
        
        Args:
            interval_seconds: Reporting interval
        """
        logger.info(f"🛡️ [SAFETY GOVERNOR] Starting monitor loop (interval: {interval_seconds}s)")
        
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                self.generate_report()
                
            except asyncio.CancelledError:
                logger.info("🛡️ [SAFETY GOVERNOR] Monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"🛡️ [SAFETY GOVERNOR] Error in monitor loop: {e}", exc_info=True)
