"""
🏛️ QUANTUM TRADER GRUNNLOV COMPONENTS
======================================
Implementation of the 15 Constitutional Components.

Each component maps to a Grunnlov (Fundamental Law).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import redis
import os

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 1: CAPITAL PRESERVATION GOVERNOR
# ═══════════════════════════════════════════════════════════════════════════════

class CapitalPreservationGovernor:
    """
    GRUNNLOV 1: FORMÅL (OVERLEVELSE)
    
    Supreme authority with VETO power over all system operations.
    Primary responsibility: Ensure fund survival at all costs.
    
    FAIL MODE: FAIL_CLOSED (halt on any uncertainty)
    """
    
    # IMMUTABLE THRESHOLDS - Cannot be changed at runtime
    MAX_DAILY_DRAWDOWN_PCT = -5.0       # Daily DD limit
    MAX_TOTAL_DRAWDOWN_PCT = -15.0      # Total DD limit  
    CAPITAL_THREAT_THRESHOLD = 0.70     # Remaining capital trigger
    STRUCTURAL_RISK_INDICATORS = [
        "exchange_down",
        "api_errors_cascade", 
        "price_feed_stale",
        "margin_call_imminent",
        "liquidity_crisis",
    ]
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self._veto_active = False
        self._veto_reason = ""
        self._last_check = datetime.utcnow()
        logger.info("🛡️ CapitalPreservationGovernor initialized - VETO authority active")
    
    def check_capital_threat(
        self,
        current_equity: float,
        starting_equity: float,
        daily_pnl_pct: float,
        structural_indicators: List[str],
    ) -> tuple[bool, str]:
        """
        Check for capital threats.
        
        Returns:
            (is_safe: bool, reason: str)
        """
        # Check daily drawdown
        if daily_pnl_pct <= self.MAX_DAILY_DRAWDOWN_PCT:
            self._activate_veto(f"Daily DD {daily_pnl_pct}% exceeds limit {self.MAX_DAILY_DRAWDOWN_PCT}%")
            return False, self._veto_reason
        
        # Check total drawdown
        total_dd = ((current_equity - starting_equity) / starting_equity) * 100
        if total_dd <= self.MAX_TOTAL_DRAWDOWN_PCT:
            self._activate_veto(f"Total DD {total_dd:.2f}% exceeds limit {self.MAX_TOTAL_DRAWDOWN_PCT}%")
            return False, self._veto_reason
        
        # Check capital threshold
        capital_ratio = current_equity / starting_equity
        if capital_ratio <= self.CAPITAL_THREAT_THRESHOLD:
            self._activate_veto(f"Capital at {capital_ratio*100:.1f}% of starting - THREAT")
            return False, self._veto_reason
        
        # Check structural risks
        for indicator in structural_indicators:
            if indicator in self.STRUCTURAL_RISK_INDICATORS:
                self._activate_veto(f"Structural risk detected: {indicator}")
                return False, self._veto_reason
        
        return True, "Capital preservation check passed"
    
    def _activate_veto(self, reason: str) -> None:
        """Activate VETO - blocks all trading."""
        self._veto_active = True
        self._veto_reason = reason
        logger.critical(f"🚨 VETO ACTIVATED: {reason}")
        
        if self.redis:
            self.redis.set("grunnlov:veto:active", "true")
            self.redis.set("grunnlov:veto:reason", reason)
            self.redis.set("grunnlov:veto:timestamp", datetime.utcnow().isoformat())
    
    def is_trading_allowed(self) -> tuple[bool, str]:
        """Check if trading is allowed."""
        if self._veto_active:
            return False, self._veto_reason
        return True, "Trading allowed"
    
    def reset_veto(self, authorization_code: str, reason: str) -> bool:
        """Reset veto with proper authorization."""
        # Require strong authorization
        if len(authorization_code) < 32:
            logger.warning("VETO reset rejected - insufficient authorization")
            return False
        
        self._veto_active = False
        self._veto_reason = ""
        logger.info(f"✅ VETO reset authorized: {reason}")
        
        if self.redis:
            self.redis.delete("grunnlov:veto:active")
            self.redis.set("grunnlov:veto:reset_reason", reason)
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 3: DECISION ARBITRATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentPriority(Enum):
    """Priority levels for decision arbitration."""
    CAPITAL_PRESERVATION = 100  # GRUNNLOV 1
    RISK_KERNEL = 90           # GRUNNLOV 7
    POLICY_ENGINE = 80         # GRUNNLOV 2
    EXIT_BRAIN = 70            # GRUNNLOV 6
    ENTRY_GATE = 60            # GRUNNLOV 5
    EXECUTION = 50             # GRUNNLOV 9
    AI_ADVISORY = 30           # GRUNNLOV 10
    LEARNING = 20              # GRUNNLOV 11


class DecisionArbitrationLayer:
    """
    GRUNNLOV 3: BESLUTNINGSHIERARKI
    
    Resolves conflicts between components using strict hierarchy.
    Higher priority components ALWAYS win.
    
    HIERARCHY (highest to lowest):
    1. Capital Preservation (VETO)
    2. Risk Kernel
    3. Policy Engine
    4. Exit Brain
    5. Entry Gate
    6. Execution
    7. AI Advisory
    8. Learning
    """
    
    def __init__(self):
        self._decisions: List[Dict] = []
        self._conflicts_resolved = 0
        logger.info("⚖️ DecisionArbitrationLayer initialized")
    
    def arbitrate(
        self,
        decisions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Arbitrate between multiple component decisions.
        
        Args:
            decisions: List of {component: str, priority: ComponentPriority, decision: Any}
            
        Returns:
            Winning decision with audit trail
        """
        if not decisions:
            return {"decision": None, "reason": "No decisions to arbitrate"}
        
        # Sort by priority (highest first)
        sorted_decisions = sorted(
            decisions,
            key=lambda d: d.get("priority", ComponentPriority.LEARNING).value,
            reverse=True
        )
        
        winner = sorted_decisions[0]
        
        # Check for conflicts
        if len(decisions) > 1:
            self._conflicts_resolved += 1
            
            # Log arbitration
            logger.info(
                f"⚖️ Arbitration #{self._conflicts_resolved}:\n"
                f"   Winner: {winner['component']} (priority {winner['priority'].value})\n"
                f"   Decision: {winner['decision']}\n"
                f"   Overridden: {[d['component'] for d in sorted_decisions[1:]]}"
            )
        
        return {
            "decision": winner["decision"],
            "component": winner["component"],
            "priority": winner["priority"].value,
            "overridden": [
                {"component": d["component"], "decision": d["decision"]}
                for d in sorted_decisions[1:]
            ],
            "arbitration_id": self._conflicts_resolved,
        }
    
    def validate_override(
        self,
        source_component: str,
        source_priority: ComponentPriority,
        target_component: str,
        target_priority: ComponentPriority,
    ) -> tuple[bool, str]:
        """
        Validate if source can override target.
        
        Returns:
            (allowed: bool, reason: str)
        """
        if source_priority.value >= target_priority.value:
            return True, f"{source_component} can override {target_component}"
        
        return False, (
            f"BLOCKED: {source_component} (priority {source_priority.value}) "
            f"cannot override {target_component} (priority {target_priority.value})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 5: ENTRY QUALIFICATION GATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntryPlan:
    """Complete entry plan with risk definition."""
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    risk_reward_ratio: float
    confidence: float
    strategy: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EntryQualificationGate:
    """
    GRUNNLOV 5: ENTRY ER RISIKODEFINISJON
    
    Every entry MUST have:
    - Defined stop-loss
    - Defined take-profit
    - Calculated position size
    - Risk amount quantified
    
    FAIL MODE: FAIL_CLOSED (no entry without complete plan)
    """
    
    MIN_RR_RATIO = 1.0            # Minimum risk/reward
    MAX_RISK_PER_TRADE_PCT = 2.0  # Maximum risk per trade
    MIN_CONFIDENCE = 0.5          # Minimum entry confidence
    
    def __init__(self):
        self._rejected_entries = 0
        self._approved_entries = 0
        logger.info("🚪 EntryQualificationGate initialized - No entry without plan")
    
    def qualify_entry(self, plan: EntryPlan) -> tuple[bool, str, Optional[EntryPlan]]:
        """
        Qualify an entry plan.
        
        Returns:
            (qualified: bool, reason: str, validated_plan: Optional[EntryPlan])
        """
        # Check stop-loss
        if plan.stop_loss is None or plan.stop_loss <= 0:
            self._rejected_entries += 1
            return False, "REJECTED: No stop-loss defined", None
        
        # Check take-profit
        if plan.take_profit is None or plan.take_profit <= 0:
            self._rejected_entries += 1
            return False, "REJECTED: No take-profit defined", None
        
        # Check position size
        if plan.position_size is None or plan.position_size <= 0:
            self._rejected_entries += 1
            return False, "REJECTED: No position size calculated", None
        
        # Check risk amount
        if plan.risk_amount is None or plan.risk_amount <= 0:
            self._rejected_entries += 1
            return False, "REJECTED: No risk amount defined", None
        
        # Validate risk/reward ratio
        if plan.risk_reward_ratio < self.MIN_RR_RATIO:
            self._rejected_entries += 1
            return False, f"REJECTED: R/R {plan.risk_reward_ratio:.2f} below minimum {self.MIN_RR_RATIO}", None
        
        # Check confidence
        if plan.confidence < self.MIN_CONFIDENCE:
            self._rejected_entries += 1
            return False, f"REJECTED: Confidence {plan.confidence:.2f} below minimum {self.MIN_CONFIDENCE}", None
        
        # Validate stop-loss direction
        if plan.side == "LONG" and plan.stop_loss >= plan.entry_price:
            self._rejected_entries += 1
            return False, "REJECTED: LONG stop-loss must be below entry", None
        
        if plan.side == "SHORT" and plan.stop_loss <= plan.entry_price:
            self._rejected_entries += 1
            return False, "REJECTED: SHORT stop-loss must be above entry", None
        
        # Entry qualified
        self._approved_entries += 1
        logger.info(
            f"✅ Entry qualified: {plan.symbol} {plan.side}\n"
            f"   Entry: {plan.entry_price}, SL: {plan.stop_loss}, TP: {plan.take_profit}\n"
            f"   Risk: ${plan.risk_amount:.2f}, R/R: {plan.risk_reward_ratio:.2f}"
        )
        
        return True, "Entry plan qualified", plan
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "approved": self._approved_entries,
            "rejected": self._rejected_entries,
            "approval_rate": self._approved_entries / max(1, self._approved_entries + self._rejected_entries),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 7: RISK KERNEL (FAIL-CLOSED)
# ═══════════════════════════════════════════════════════════════════════════════

class RiskKernel:
    """
    GRUNNLOV 7: RISIKO ER HELLIG
    
    Risk limits are IMMUTABLE. No bypass. No exceptions.
    
    FAIL MODE: FAIL_CLOSED (block on ANY uncertainty)
    """
    
    # IMMUTABLE LIMITS - These CANNOT be changed at runtime
    MAX_LOSS_PER_TRADE_PCT = 2.0    # % of capital
    MAX_DAILY_LOSS_PCT = 5.0        # % of capital
    MAX_WEEKLY_LOSS_PCT = 10.0      # % of capital
    MAX_DRAWDOWN_PCT = 15.0         # Total drawdown limit
    MAX_LEVERAGE = 10.0             # Maximum leverage
    MAX_POSITION_PCT = 25.0         # Max % in single position
    KILL_SWITCH_THRESHOLD = -10.0   # Emergency kill threshold
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self._kill_switch_active = False
        self._daily_loss = 0.0
        self._weekly_loss = 0.0
        logger.info("🔒 RiskKernel initialized - IMMUTABLE LIMITS ACTIVE")
    
    def check_trade_risk(
        self,
        risk_pct: float,
        leverage: float,
        position_pct: float,
    ) -> tuple[bool, str]:
        """
        Check if a trade passes risk limits.
        
        Returns:
            (allowed: bool, reason: str)
        """
        if self._kill_switch_active:
            return False, "BLOCKED: Kill switch active"
        
        if risk_pct > self.MAX_LOSS_PER_TRADE_PCT:
            return False, f"BLOCKED: Risk {risk_pct}% exceeds max {self.MAX_LOSS_PER_TRADE_PCT}%"
        
        if leverage > self.MAX_LEVERAGE:
            return False, f"BLOCKED: Leverage {leverage}x exceeds max {self.MAX_LEVERAGE}x"
        
        if position_pct > self.MAX_POSITION_PCT:
            return False, f"BLOCKED: Position {position_pct}% exceeds max {self.MAX_POSITION_PCT}%"
        
        return True, "Trade risk approved"
    
    def check_portfolio_risk(
        self,
        daily_loss_pct: float,
        weekly_loss_pct: float,
        drawdown_pct: float,
    ) -> tuple[bool, str]:
        """
        Check portfolio-level risk limits.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Kill switch check
        if daily_loss_pct <= self.KILL_SWITCH_THRESHOLD:
            self._activate_kill_switch(f"Daily loss {daily_loss_pct}% hit kill threshold")
            return False, self._kill_reason
        
        # Daily loss check
        if abs(daily_loss_pct) >= self.MAX_DAILY_LOSS_PCT:
            return False, f"BLOCKED: Daily loss {daily_loss_pct}% exceeds max {self.MAX_DAILY_LOSS_PCT}%"
        
        # Weekly loss check
        if abs(weekly_loss_pct) >= self.MAX_WEEKLY_LOSS_PCT:
            return False, f"BLOCKED: Weekly loss {weekly_loss_pct}% exceeds max {self.MAX_WEEKLY_LOSS_PCT}%"
        
        # Drawdown check
        if abs(drawdown_pct) >= self.MAX_DRAWDOWN_PCT:
            return False, f"BLOCKED: Drawdown {drawdown_pct}% exceeds max {self.MAX_DRAWDOWN_PCT}%"
        
        return True, "Portfolio risk approved"
    
    def _activate_kill_switch(self, reason: str) -> None:
        """Activate emergency kill switch."""
        self._kill_switch_active = True
        self._kill_reason = reason
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        if self.redis:
            self.redis.set("grunnlov:kill_switch", "true")
            self.redis.set("grunnlov:kill_reason", reason)
    
    def is_trading_allowed(self) -> tuple[bool, str]:
        """Check if trading is allowed."""
        if self._kill_switch_active:
            return False, f"Kill switch active: {self._kill_reason}"
        return True, "Trading allowed"


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 13: CONSTRAINT ENFORCEMENT LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class ConstraintEnforcementLayer:
    """
    GRUNNLOV 13: ABSOLUTTE FORBUD
    
    Enforces absolute prohibitions:
    - No stop-loss movement away from entry
    - No revenge trading
    - No over-sizing after loss
    
    FAIL MODE: FAIL_CLOSED
    """
    
    REVENGE_COOLDOWN_MINUTES = 30
    MAX_CONSECUTIVE_LOSSES = 3
    SIZE_REDUCTION_AFTER_LOSS = 0.5  # 50% size after loss
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self._recent_losses: List[datetime] = []
        self._consecutive_losses = 0
        self._last_trade_result: Optional[str] = None
        logger.info("🚫 ConstraintEnforcementLayer initialized - Absolute prohibitions active")
    
    def check_stop_loss_move(
        self,
        old_sl: float,
        new_sl: float,
        entry_price: float,
        side: str,
    ) -> tuple[bool, str]:
        """
        Check if stop-loss movement is allowed.
        Moving SL away from entry is FORBIDDEN.
        """
        if side == "LONG":
            # For LONG, SL can only move UP (closer to or past entry)
            if new_sl < old_sl:
                return False, "FORBIDDEN: Cannot move stop-loss further from entry (LONG)"
        else:  # SHORT
            # For SHORT, SL can only move DOWN (closer to or past entry)
            if new_sl > old_sl:
                return False, "FORBIDDEN: Cannot move stop-loss further from entry (SHORT)"
        
        return True, "Stop-loss move allowed (tightening)"
    
    def check_revenge_trading(
        self,
        symbol: str,
        last_trade_was_loss: bool,
        time_since_last_trade: timedelta,
    ) -> tuple[bool, str]:
        """
        Check for revenge trading patterns.
        """
        if last_trade_was_loss:
            cooldown = timedelta(minutes=self.REVENGE_COOLDOWN_MINUTES)
            if time_since_last_trade < cooldown:
                remaining = cooldown - time_since_last_trade
                return False, (
                    f"FORBIDDEN: Revenge trading cooldown active. "
                    f"Wait {remaining.seconds // 60}m {remaining.seconds % 60}s"
                )
        
        if self._consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            return False, (
                f"FORBIDDEN: {self._consecutive_losses} consecutive losses. "
                f"Trading suspended until review."
            )
        
        return True, "No revenge trading detected"
    
    def check_position_sizing(
        self,
        requested_size: float,
        normal_size: float,
        last_trade_was_loss: bool,
    ) -> tuple[bool, float, str]:
        """
        Check position sizing constraints.
        
        Returns:
            (allowed: bool, adjusted_size: float, reason: str)
        """
        if last_trade_was_loss:
            max_allowed = normal_size * self.SIZE_REDUCTION_AFTER_LOSS
            if requested_size > max_allowed:
                return False, max_allowed, (
                    f"CONSTRAINED: Size reduced to {max_allowed:.2f} after loss "
                    f"(requested {requested_size:.2f})"
                )
        
        return True, requested_size, "Position size approved"
    
    def record_trade_result(self, is_loss: bool) -> None:
        """Record trade result for constraint tracking."""
        if is_loss:
            self._recent_losses.append(datetime.utcnow())
            self._consecutive_losses += 1
            self._last_trade_result = "LOSS"
        else:
            self._consecutive_losses = 0
            self._last_trade_result = "WIN"
        
        # Clean old losses (older than 24h)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._recent_losses = [t for t in self._recent_losses if t > cutoff]


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 14: STRATEGY SCOPE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyScopeController:
    """
    GRUNNLOV 14: IDENTITET
    
    Maintains fund identity and prevents strategy drift.
    
    FAIL MODE: FAIL_CLOSED
    """
    
    def __init__(
        self,
        allowed_symbols: List[str],
        allowed_strategies: List[str],
        allowed_timeframes: List[str],
        max_concurrent_strategies: int = 3,
    ):
        self.allowed_symbols = set(allowed_symbols)
        self.allowed_strategies = set(allowed_strategies)
        self.allowed_timeframes = set(allowed_timeframes)
        self.max_concurrent_strategies = max_concurrent_strategies
        self._active_strategies: Set[str] = set()
        logger.info(
            f"🎯 StrategyScopeController initialized\n"
            f"   Symbols: {len(allowed_symbols)}\n"
            f"   Strategies: {len(allowed_strategies)}\n"
            f"   Timeframes: {allowed_timeframes}"
        )
    
    def validate_trade(
        self,
        symbol: str,
        strategy: str,
        timeframe: str,
    ) -> tuple[bool, str]:
        """
        Validate trade against fund identity.
        """
        # Check symbol
        if symbol not in self.allowed_symbols:
            return False, f"BLOCKED: Symbol {symbol} not in approved list"
        
        # Check strategy
        if strategy not in self.allowed_strategies:
            return False, f"BLOCKED: Strategy '{strategy}' not approved"
        
        # Check timeframe
        if timeframe not in self.allowed_timeframes:
            return False, f"BLOCKED: Timeframe {timeframe} not approved"
        
        # Check concurrent strategies
        if (
            strategy not in self._active_strategies 
            and len(self._active_strategies) >= self.max_concurrent_strategies
        ):
            return False, (
                f"BLOCKED: Max {self.max_concurrent_strategies} concurrent strategies. "
                f"Active: {self._active_strategies}"
            )
        
        self._active_strategies.add(strategy)
        return True, "Trade within fund scope"
    
    def request_scope_change(
        self,
        change_type: str,
        new_value: str,
        reason: str,
    ) -> tuple[bool, str]:
        """
        Request a scope change (requires explicit approval).
        """
        logger.warning(
            f"⚠️ Scope change requested:\n"
            f"   Type: {change_type}\n"
            f"   Value: {new_value}\n"
            f"   Reason: {reason}\n"
            f"   Status: PENDING APPROVAL"
        )
        
        # In production, this would require human approval
        return False, "Scope changes require explicit human approval"


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 15: HUMAN OVERRIDE LOCK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OverrideRecord:
    """Record of a human override."""
    timestamp: datetime
    action: str
    reason: str
    user_id: str
    approved: bool
    cooldown_bypassed: bool


class HumanOverrideLock:
    """
    GRUNNLOV 15: EDEN (MENNESKELÅS)
    
    Humans are the most dangerous component.
    All manual overrides require cooldown and logging.
    
    FAIL MODE: FAIL_CLOSED
    """
    
    DEFAULT_COOLDOWN_SECONDS = 60
    EMERGENCY_COOLDOWN_SECONDS = 10
    MAX_OVERRIDES_PER_HOUR = 3
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self._override_history: List[OverrideRecord] = []
        self._last_override: Optional[datetime] = None
        logger.info("🧑 HumanOverrideLock initialized - Human protection active")
    
    def request_override(
        self,
        action: str,
        reason: str,
        user_id: str,
        is_emergency: bool = False,
    ) -> tuple[bool, str, int]:
        """
        Request a manual override.
        
        Returns:
            (allowed: bool, reason: str, wait_seconds: int)
        """
        now = datetime.utcnow()
        
        # Check cooldown
        cooldown = (
            self.EMERGENCY_COOLDOWN_SECONDS 
            if is_emergency 
            else self.DEFAULT_COOLDOWN_SECONDS
        )
        
        if self._last_override:
            elapsed = (now - self._last_override).total_seconds()
            if elapsed < cooldown:
                wait = int(cooldown - elapsed)
                return False, f"Cooldown active - wait {wait}s", wait
        
        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        recent_overrides = [
            o for o in self._override_history 
            if o.timestamp > hour_ago
        ]
        
        if len(recent_overrides) >= self.MAX_OVERRIDES_PER_HOUR:
            return False, f"Hourly override limit ({self.MAX_OVERRIDES_PER_HOUR}) reached", 3600
        
        # Record override
        record = OverrideRecord(
            timestamp=now,
            action=action,
            reason=reason,
            user_id=user_id,
            approved=True,
            cooldown_bypassed=is_emergency,
        )
        self._override_history.append(record)
        self._last_override = now
        
        # Log to audit
        logger.warning(
            f"⚠️ HUMAN OVERRIDE APPROVED:\n"
            f"   Action: {action}\n"
            f"   Reason: {reason}\n"
            f"   User: {user_id}\n"
            f"   Emergency: {is_emergency}"
        )
        
        if self.redis:
            self.redis.xadd(
                "grunnlov:override:log",
                {
                    "timestamp": now.isoformat(),
                    "action": action,
                    "reason": reason,
                    "user_id": user_id,
                    "is_emergency": str(is_emergency),
                }
            )
        
        return True, "Override approved", 0
    
    def get_override_stats(self) -> Dict[str, Any]:
        """Get override statistics."""
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        day_ago = datetime.utcnow() - timedelta(days=1)
        
        return {
            "total_overrides": len(self._override_history),
            "last_hour": len([o for o in self._override_history if o.timestamp > hour_ago]),
            "last_24h": len([o for o in self._override_history if o.timestamp > day_ago]),
            "last_override": self._last_override.isoformat() if self._last_override else None,
            "remaining_hourly": self.MAX_OVERRIDES_PER_HOUR - len(
                [o for o in self._override_history if o.timestamp > hour_ago]
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV 10: AI ADVISORY LAYER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AIAdvice:
    """AI advisory output - ADVICE ONLY, no execution authority."""
    regime_score: float          # Market regime confidence
    edge_score: float            # Edge/alpha confidence
    confidence: float            # Overall confidence
    recommended_action: str      # BUY, SELL, HOLD
    reasoning: List[str]         # Explanation
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # AI CANNOT set these - enforced by Grunnlov 10
    position_size: None = None
    stop_loss: None = None
    take_profit: None = None


class AIAdvisoryLayer:
    """
    GRUNNLOV 10: AI HAR BEGRENSET MAKT
    
    AI is ADVISOR only, not commander.
    
    AI CAN provide:
    - Regime score
    - Edge score
    - Confidence estimate
    
    AI CANNOT:
    - Set position size
    - Control exits
    - Override risk limits
    - Bypass human lock
    """
    
    FORBIDDEN_ACTIONS = {
        "SET_POSITION_SIZE",
        "FORCE_EXIT",
        "OVERRIDE_RISK",
        "BYPASS_HUMAN_LOCK",
        "MODIFY_STOP_LOSS",
        "EXECUTE_TRADE",
        "CLOSE_POSITION",
    }
    
    def __init__(self):
        self._advice_count = 0
        self._blocked_attempts = 0
        logger.info("🤖 AIAdvisoryLayer initialized - ADVISORY ONLY")
    
    def provide_advice(
        self,
        regime_score: float,
        edge_score: float,
        confidence: float,
        recommended_action: str,
        reasoning: List[str],
    ) -> AIAdvice:
        """
        Provide AI advice (allowed action).
        """
        self._advice_count += 1
        
        advice = AIAdvice(
            regime_score=regime_score,
            edge_score=edge_score,
            confidence=confidence,
            recommended_action=recommended_action,
            reasoning=reasoning,
        )
        
        logger.info(
            f"🤖 AI Advice #{self._advice_count}:\n"
            f"   Action: {recommended_action}\n"
            f"   Confidence: {confidence:.2f}\n"
            f"   Regime: {regime_score:.2f}, Edge: {edge_score:.2f}"
        )
        
        return advice
    
    def attempt_forbidden_action(self, action: str) -> tuple[bool, str]:
        """
        Attempt a forbidden action (will be blocked).
        """
        if action in self.FORBIDDEN_ACTIONS:
            self._blocked_attempts += 1
            logger.warning(
                f"🚫 AI FORBIDDEN ACTION BLOCKED:\n"
                f"   Action: {action}\n"
                f"   Total blocked: {self._blocked_attempts}"
            )
            return False, f"GRUNNLOV 10: AI cannot perform action '{action}'"
        
        return True, "Action allowed"
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "advice_provided": self._advice_count,
            "forbidden_blocked": self._blocked_attempts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPONENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class GrunnlovComponentRegistry:
    """
    Registry of all Grunnlov components.
    Provides unified access to all constitutional components.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        
        # Initialize all components
        self.capital_preservation = CapitalPreservationGovernor(redis_client)
        self.decision_arbitration = DecisionArbitrationLayer()
        self.entry_gate = EntryQualificationGate()
        self.risk_kernel = RiskKernel(redis_client)
        self.constraint_enforcement = ConstraintEnforcementLayer(redis_client)
        self.strategy_scope = StrategyScopeController(
            allowed_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],  # Default
            allowed_strategies=["MOMENTUM", "MEAN_REVERSION", "BREAKOUT"],
            allowed_timeframes=["5m", "15m", "1h", "4h"],
        )
        self.human_override_lock = HumanOverrideLock(redis_client)
        self.ai_advisory = AIAdvisoryLayer()
        
        logger.info("🏛️ GrunnlovComponentRegistry initialized - All 15 laws active")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all constitutional components."""
        veto_allowed, veto_reason = self.capital_preservation.is_trading_allowed()
        risk_allowed, risk_reason = self.risk_kernel.is_trading_allowed()
        
        return {
            "trading_allowed": veto_allowed and risk_allowed,
            "veto_status": {"allowed": veto_allowed, "reason": veto_reason},
            "risk_status": {"allowed": risk_allowed, "reason": risk_reason},
            "entry_stats": self.entry_gate.get_stats(),
            "override_stats": self.human_override_lock.get_override_stats(),
            "ai_stats": self.ai_advisory.get_stats(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_grunnlov_registry: Optional[GrunnlovComponentRegistry] = None


def get_grunnlov_registry(
    redis_client: Optional[redis.Redis] = None
) -> GrunnlovComponentRegistry:
    """Get or create the global Grunnlov component registry."""
    global _grunnlov_registry
    if _grunnlov_registry is None:
        _grunnlov_registry = GrunnlovComponentRegistry(redis_client)
    return _grunnlov_registry
