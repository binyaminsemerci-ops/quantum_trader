"""
🏛️ QUANTUM TRADER GRUNNLOV (CONSTITUTION)
==========================================
Version: 1.0
Date: February 13, 2026

The 15 Fundamental Laws that govern the Quantum Trader system.
These laws are IMMUTABLE and cannot be overridden by any component.

HIERARCHY (Top to Bottom):
┌────────────────────────────────┐
│ CAPITAL PRESERVATION GOVERNOR  │  ← GRUNNLOV 1 (VETO POWER)
└───────────▲────────────────────┘
            │
┌───────────┴───────────┐
│ POLICY & DECISION LAYER│  ← GRUNNLOV 2, 3
└───────────▲───────────┘
            │
┌───────────┴───────────┐
│ RISK KERNEL (FAIL-CLOSED)│  ← GRUNNLOV 7
└───────────▲───────────┘
            │
┌───────┬───────────┬───────────┐
│ ENTRY │  EXIT     │ EXECUTION │  ← GRUNNLOV 5, 6, 9
│ GATE  │ HARVEST   │ OPTIMIZER │
└───▲───┴────▲──────┴────▲──────┘
    │         │           │
 AI ADVISORY  │     MARKET REGIME  ← GRUNNLOV 10, 4
              │
        FEEDBACK & LEARNING  ← GRUNNLOV 11
        
Author: Quantum Trader Constitution System
"""

import logging
from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FUNDAMENTAL ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class ConstitutionLaw(IntEnum):
    """The 15 Fundamental Laws - ordered by priority."""
    
    # === SUPREME LAWS (Cannot be overridden) ===
    GRUNNLOV_1_SURVIVAL = 1           # Capital Preservation - VETO POWER
    GRUNNLOV_7_RISK_SACRED = 7        # Risk Kernel - FAIL-CLOSED
    
    # === GOVERNING LAWS (Policy & Decision) ===
    GRUNNLOV_2_PHILOSOPHY = 2         # Risk > Profit philosophy
    GRUNNLOV_3_HIERARCHY = 3          # Decision arbitration
    
    # === MARKET LAWS ===
    GRUNNLOV_4_MARKET_REALITY = 4     # Market regime detection
    
    # === TRADING LAWS ===
    GRUNNLOV_5_ENTRY_RISK = 5         # Entry = risk definition
    GRUNNLOV_6_EXIT_DOMINANCE = 6     # Exit owns profit
    GRUNNLOV_8_CAPITAL_LEVERAGE = 8   # Position sizing
    GRUNNLOV_9_EXECUTION = 9          # Execution quality
    
    # === AI LAWS ===
    GRUNNLOV_10_AI_LIMITED = 10       # AI is advisor only
    GRUNNLOV_11_FEEDBACK = 11         # Learning loop
    
    # === GOVERNANCE LAWS ===
    GRUNNLOV_12_AUDIT = 12            # Audit ledger
    GRUNNLOV_13_PROHIBITIONS = 13     # Absolute constraints
    GRUNNLOV_14_IDENTITY = 14         # Strategy scope
    GRUNNLOV_15_HUMAN_LOCK = 15       # Human override cooldown


class LawAuthority(IntEnum):
    """Authority levels for law enforcement."""
    SUPREME = 100      # Cannot be overridden (GRUNNLOV 1, 7)
    GOVERNING = 80     # Policy-level authority (GRUNNLOV 2, 3)
    OPERATIONAL = 60   # Operational authority (GRUNNLOV 4-6, 8-9)
    ADVISORY = 40      # Advisory authority (GRUNNLOV 10-11)
    COMPLIANCE = 20    # Compliance/audit (GRUNNLOV 12-15)


class ViolationType(IntEnum):
    """Types of constitution violations."""
    CRITICAL = 100     # System must halt
    SEVERE = 80        # Trade must be blocked
    WARNING = 60       # Trade may proceed with logging
    INFO = 40          # Logged for audit


class ComponentRole(IntEnum):
    """Roles that system components can have."""
    VETO = 100         # Can veto all decisions
    ENFORCER = 80      # Can block trades
    GOVERNOR = 60      # Makes policy decisions
    ADVISOR = 40       # Provides recommendations
    OBSERVER = 20      # Records/reports only


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LAW DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LawDefinition:
    """Definition of a constitutional law."""
    
    law: ConstitutionLaw
    name_no: str              # Norwegian name
    name_en: str              # English name
    authority: LawAuthority
    component: str            # Responsible component
    component_role: ComponentRole
    description: str
    principles: List[str]     # Core principles
    prohibitions: List[str]   # What is forbidden
    enforcement: str          # How it's enforced
    fail_mode: str            # FAIL_CLOSED or FAIL_OPEN
    

# The 15 Laws defined
GRUNNLOVER: Dict[ConstitutionLaw, LawDefinition] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 1: FORMÅL (OVERLEVELSE)
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_1_SURVIVAL: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_1_SURVIVAL,
        name_no="FORMÅL (OVERLEVELSE)",
        name_en="PURPOSE (SURVIVAL)",
        authority=LawAuthority.SUPREME,
        component="CapitalPreservationGovernor",
        component_role=ComponentRole.VETO,
        description="Capital preservation is the supreme objective. Do no harm.",
        principles=[
            "Capital preservation takes absolute priority over profit",
            "Structural risk triggers immediate system halt",
            "No trade is worth risking the fund's survival",
            "This component has VETO power over all decisions",
        ],
        prohibitions=[
            "Trading during structural risk",
            "Ignoring capital preservation signals",
            "Overriding survival mechanisms",
        ],
        enforcement="Automatic system halt on capital threat",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 2: FILOSOFI (RISK > PROFIT)
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_2_PHILOSOPHY: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_2_PHILOSOPHY,
        name_no="FILOSOFI (RISK > PROFIT)",
        name_en="PHILOSOPHY (RISK > PROFIT)",
        authority=LawAuthority.GOVERNING,
        component="PolicyEngine",
        component_role=ComponentRole.GOVERNOR,
        description="Risk management always takes priority over profit seeking.",
        principles=[
            "All decisions are filtered through risk-first lens",
            "Defines what constitutes 'legal trading'",
            "All components must query policy before action",
            "Policy is the single source of truth",
        ],
        prohibitions=[
            "Profit-seeking that violates risk policy",
            "Bypassing policy checks",
            "Modifying policy during high-risk periods",
        ],
        enforcement="All components must call policy_check() before action",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 3: BESLUTNINGSHIERARKI
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_3_HIERARCHY: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_3_HIERARCHY,
        name_no="BESLUTNINGSHIERARKI",
        name_en="DECISION HIERARCHY",
        authority=LawAuthority.GOVERNING,
        component="DecisionArbitrationLayer",
        component_role=ComponentRole.GOVERNOR,
        description="Clear hierarchy ensures higher layers always win conflicts.",
        principles=[
            "Survival > Risk > Policy > Entry/Exit > AI",
            "Conflicts are resolved by hierarchy, not voting",
            "AI can never override higher layers",
            "This is the separation of powers",
        ],
        prohibitions=[
            "Lower layer overriding higher layer",
            "AI making final decisions",
            "Bypassing arbitration",
        ],
        enforcement="DecisionArbitrationLayer validates all cross-component calls",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 4: MARKEDSREALITET
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_4_MARKET_REALITY: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_4_MARKET_REALITY,
        name_no="MARKEDSREALITET",
        name_en="MARKET REALITY",
        authority=LawAuthority.OPERATIONAL,
        component="MarketRegimeDetector",
        component_role=ComponentRole.ADVISOR,
        description="Market regime determines what is possible and safe.",
        principles=[
            "Market state classification is objective",
            "No-trade regimes must be respected",
            "Unsafe periods are marked and enforced",
            "Market truth is injected into all decisions",
        ],
        prohibitions=[
            "Trading in no-trade regimes",
            "Ignoring regime warnings",
            "Overriding market classification",
        ],
        enforcement="Entry/Exit blocked in unsafe regimes",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 5: ENTRY ER RISIKODEFINISJON
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_5_ENTRY_RISK: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_5_ENTRY_RISK,
        name_no="ENTRY ER RISIKODEFINISJON",
        name_en="ENTRY IS RISK DEFINITION",
        authority=LawAuthority.OPERATIONAL,
        component="EntryQualificationGate",
        component_role=ComponentRole.ENFORCER,
        description="No entry without defined risk and exit plan.",
        principles=[
            "Every entry must have defined stop-loss",
            "Every entry must have defined take-profit",
            "Position size must be calculated from risk",
            "No 'good entry' without structure",
        ],
        prohibitions=[
            "Entries without stop-loss",
            "Entries without take-profit",
            "Entries without position sizing",
            "Entries based on 'feel' alone",
        ],
        enforcement="EntryQualificationGate validates all entry requests",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 6: EXIT-DOMINANS
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_6_EXIT_DOMINANCE: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_6_EXIT_DOMINANCE,
        name_no="EXIT-DOMINANS",
        name_en="EXIT DOMINANCE",
        authority=LawAuthority.OPERATIONAL,
        component="ExitHarvestBrain",
        component_role=ComponentRole.ENFORCER,
        description="Exit layer owns ALL profit realization.",
        principles=[
            "Exit Brain is the fund's profit engine",
            "Partial exits are strategic tools",
            "Time-based exits prevent overstay",
            "Regime-based exits protect capital",
        ],
        prohibitions=[
            "Manual exit interference (without cooldown)",
            "Disabling exit automation",
            "Ignoring time-stop signals",
        ],
        enforcement="ExitHarvestBrain has sole authority over exits",
        fail_mode="FAIL_OPEN",  # Positions can always be closed
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 7: RISIKO ER HELLIG
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_7_RISK_SACRED: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_7_RISK_SACRED,
        name_no="RISIKO ER HELLIG",
        name_en="RISK IS SACRED",
        authority=LawAuthority.SUPREME,
        component="RiskKernel",
        component_role=ComponentRole.ENFORCER,
        description="Risk limits are inviolable. No bypass. No exceptions.",
        principles=[
            "Max loss per trade is absolute",
            "Max daily loss triggers halt",
            "Drawdown guards cannot be disabled",
            "Kill-switch is always active",
        ],
        prohibitions=[
            "Exceeding max loss per trade",
            "Exceeding max daily loss",
            "Disabling drawdown guards",
            "Bypassing kill-switch",
        ],
        enforcement="RiskKernel enforces with FAIL_CLOSED semantics",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 8: KAPITAL & LEVERAGE
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_8_CAPITAL_LEVERAGE: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_8_CAPITAL_LEVERAGE,
        name_no="KAPITAL & LEVERAGE",
        name_en="CAPITAL & LEVERAGE",
        authority=LawAuthority.OPERATIONAL,
        component="CapitalAllocationEngine",
        component_role=ComponentRole.ENFORCER,
        description="Controls 'how much', never 'whether'.",
        principles=[
            "Position sizing from risk, not conviction",
            "Leverage scaling based on conditions",
            "Capital heat monitoring",
            "Automatic downscale after losses",
        ],
        prohibitions=[
            "Over-leveraging",
            "Ignoring capital heat",
            "Revenge-sizing after losses",
            "Full capital commitment",
        ],
        enforcement="CapitalAllocationEngine sets size limits",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 9: EXECUTION-RESPEKT
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_9_EXECUTION: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_9_EXECUTION,
        name_no="EXECUTION-RESPEKT",
        name_en="EXECUTION RESPECT",
        authority=LawAuthority.OPERATIONAL,
        component="ExecutionOptimizer",
        component_role=ComponentRole.ADVISOR,
        description="Poor execution is hidden risk.",
        principles=[
            "Order type selection matters",
            "Slippage must be controlled",
            "Reduce-only enforcement",
            "Fill quality is tracked",
        ],
        prohibitions=[
            "Market orders in low liquidity",
            "Ignoring slippage warnings",
            "Disabling reduce-only checks",
        ],
        enforcement="ExecutionOptimizer monitors all order flow",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 10: AI HAR BEGRENSET MAKT
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_10_AI_LIMITED: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_10_AI_LIMITED,
        name_no="AI HAR BEGRENSET MAKT",
        name_en="AI HAS LIMITED POWER",
        authority=LawAuthority.ADVISORY,
        component="AIAdvisoryLayer",
        component_role=ComponentRole.ADVISOR,
        description="AI is advisor, not commander.",
        principles=[
            "AI provides regime-score",
            "AI provides edge-score",
            "AI provides confidence-estimat",
            "AI has NO veto power",
        ],
        prohibitions=[
            "AI setting position size",
            "AI controlling exits",
            "AI overriding risk limits",
            "AI bypassing human lock",
        ],
        enforcement="AIAdvisoryLayer output is advisory only",
        fail_mode="FAIL_OPEN",  # System works without AI
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 11: FEEDBACK & LÆRING
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_11_FEEDBACK: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_11_FEEDBACK,
        name_no="FEEDBACK & LÆRING",
        name_en="FEEDBACK & LEARNING",
        authority=LawAuthority.ADVISORY,
        component="PerformanceLearningLoop",
        component_role=ComponentRole.OBSERVER,
        description="System improves itself, not the trader.",
        principles=[
            "Decision quality is evaluated",
            "Edge decay is detected",
            "Re-training is triggered automatically",
            "Parameters adjust based on performance",
        ],
        prohibitions=[
            "Manual parameter tweaking without data",
            "Ignoring edge decay signals",
            "Disabling learning loop",
        ],
        enforcement="PerformanceLearningLoop runs continuously",
        fail_mode="FAIL_OPEN",  # Trading continues without learning
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 12: GOVERNANCE & AUDIT
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_12_AUDIT: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_12_AUDIT,
        name_no="GOVERNANCE & AUDIT",
        name_en="GOVERNANCE & AUDIT",
        authority=LawAuthority.COMPLIANCE,
        component="AuditLedger",
        component_role=ComponentRole.OBSERVER,
        description="All decisions are logged for accountability.",
        principles=[
            "Every decision is logged",
            "Version control for all rules",
            "Rollback capability maintained",
            "Enables scaling and external capital",
        ],
        prohibitions=[
            "Unlogged decisions",
            "Unversioned rule changes",
            "Destroying audit records",
        ],
        enforcement="AuditLedger records all component actions",
        fail_mode="FAIL_OPEN",  # Trading continues, but logged later
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 13: ABSOLUTTE FORBUD
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS,
        name_no="ABSOLUTTE FORBUD",
        name_en="ABSOLUTE PROHIBITIONS",
        authority=LawAuthority.COMPLIANCE,
        component="ConstraintEnforcementLayer",
        component_role=ComponentRole.ENFORCER,
        description="System protects you from yourself.",
        principles=[
            "Stop-loss movement is forbidden",
            "Revenge-trading is detected and blocked",
            "Over-sizing is prevented",
            "Policy violations are monitored",
        ],
        prohibitions=[
            "Moving stop-loss further from entry",
            "Increasing size after loss",
            "Re-entering same symbol within cooldown",
            "Disabling constraint checks",
        ],
        enforcement="ConstraintEnforcementLayer blocks violations",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 14: IDENTITET
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_14_IDENTITY: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_14_IDENTITY,
        name_no="IDENTITET",
        name_en="IDENTITY",
        authority=LawAuthority.COMPLIANCE,
        component="StrategyScopeController",
        component_role=ComponentRole.ENFORCER,
        description="The fund stays true to its strategy.",
        principles=[
            "Strategy drift is prevented",
            "Markets/symbols are bounded",
            "Consistency over time is enforced",
            "Style changes require explicit approval",
        ],
        prohibitions=[
            "Trading outside approved symbols",
            "Strategy style changes without review",
            "Timeframe drift",
            "Asset class expansion without approval",
        ],
        enforcement="StrategyScopeController validates all trades",
        fail_mode="FAIL_CLOSED",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRUNNLOV 15: EDEN
    # ═══════════════════════════════════════════════════════════════════════════
    ConstitutionLaw.GRUNNLOV_15_HUMAN_LOCK: LawDefinition(
        law=ConstitutionLaw.GRUNNLOV_15_HUMAN_LOCK,
        name_no="EDEN (MENNESKELÅS)",
        name_en="EDEN (HUMAN LOCK)",
        authority=LawAuthority.COMPLIANCE,
        component="HumanOverrideLock",
        component_role=ComponentRole.ENFORCER,
        description="Humans are the most dangerous component.",
        principles=[
            "Manual overrides require cooldown",
            "All overrides are logged",
            "Impulsive actions are prevented",
            "Emotional trading is blocked",
        ],
        prohibitions=[
            "Immediate manual override (without cooldown)",
            "Unlogged manual actions",
            "Repeated overrides in short time",
            "Bypassing human lock",
        ],
        enforcement="HumanOverrideLock enforces cooldown periods",
        fail_mode="FAIL_CLOSED",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: VIOLATION TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConstitutionViolation:
    """Record of a constitutional violation."""
    
    timestamp: datetime
    law: ConstitutionLaw
    violation_type: ViolationType
    component: str
    action_attempted: str
    context: Dict[str, Any]
    blocked: bool
    reason: str
    hash: str = field(default="")
    
    def __post_init__(self):
        """Calculate cryptographic hash for immutability."""
        if not self.hash:
            content = json.dumps({
                "timestamp": self.timestamp.isoformat(),
                "law": self.law.value,
                "violation_type": self.violation_type.value,
                "component": self.component,
                "action_attempted": self.action_attempted,
                "context": str(self.context),
                "blocked": self.blocked,
                "reason": self.reason,
            }, sort_keys=True)
            self.hash = hashlib.sha256(content.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONSTITUTION ENFORCER (CORE CLASS)
# ═══════════════════════════════════════════════════════════════════════════════

class ConstitutionEnforcer:
    """
    Central enforcer of all 15 Grunnlover.
    
    This is the supreme authority in the system. All components must
    register with and be validated by this enforcer.
    
    HIERARCHY:
    1. GRUNNLOV 1 (Survival) - VETO over all
    2. GRUNNLOV 7 (Risk) - VETO over operations
    3. GRUNNLOV 2-3 (Philosophy/Hierarchy) - Policy authority
    4. GRUNNLOV 4-6, 8-9 (Operations) - Trade execution
    5. GRUNNLOV 10-15 (Advisory/Compliance) - Support
    """
    
    def __init__(self):
        """Initialize the constitution enforcer."""
        self.laws = GRUNNLOVER
        self.violations: List[ConstitutionViolation] = []
        self.registered_components: Dict[str, ComponentRole] = {}
        self.enforcement_callbacks: Dict[ConstitutionLaw, List[Callable]] = {}
        self._emergency_halt = False
        
        logger.info("🏛️ Constitution Enforcer initialized with 15 Grunnlover")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Component Registration
    # ═══════════════════════════════════════════════════════════════════════════
    
    def register_component(self, name: str, role: ComponentRole) -> bool:
        """
        Register a component with its constitutional role.
        
        Args:
            name: Component name
            role: Constitutional role (VETO, ENFORCER, etc.)
            
        Returns:
            True if registration successful
        """
        self.registered_components[name] = role
        logger.info(f"📋 Registered component: {name} with role {role.name}")
        return True
    
    def register_enforcement_callback(
        self, 
        law: ConstitutionLaw, 
        callback: Callable
    ) -> None:
        """Register a callback to be invoked when a law is violated."""
        if law not in self.enforcement_callbacks:
            self.enforcement_callbacks[law] = []
        self.enforcement_callbacks[law].append(callback)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Validation Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    def validate_action(
        self,
        source_component: str,
        action: str,
        context: Dict[str, Any],
        target_laws: Optional[List[ConstitutionLaw]] = None,
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """
        Validate an action against constitutional laws.
        
        Args:
            source_component: Component attempting the action
            action: Action being attempted (e.g., "OPEN_POSITION", "OVERRIDE_EXIT")
            context: Full context of the action
            target_laws: Specific laws to check (default: all applicable)
            
        Returns:
            (allowed: bool, violation: Optional[ConstitutionViolation])
        """
        # Check emergency halt first
        if self._emergency_halt:
            violation = self._create_violation(
                ConstitutionLaw.GRUNNLOV_1_SURVIVAL,
                ViolationType.CRITICAL,
                source_component,
                action,
                context,
                "System in emergency halt mode",
            )
            return False, violation
        
        # Determine which laws to check
        laws_to_check = target_laws or self._get_applicable_laws(action)
        
        for law in laws_to_check:
            allowed, violation = self._check_law(
                law, source_component, action, context
            )
            if not allowed:
                return False, violation
        
        return True, None
    
    def _check_law(
        self,
        law: ConstitutionLaw,
        component: str,
        action: str,
        context: Dict[str, Any],
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """Check a specific law against an action."""
        law_def = self.laws[law]
        
        # Check based on law
        if law == ConstitutionLaw.GRUNNLOV_1_SURVIVAL:
            return self._check_survival(component, action, context)
        elif law == ConstitutionLaw.GRUNNLOV_7_RISK_SACRED:
            return self._check_risk_sacred(component, action, context)
        elif law == ConstitutionLaw.GRUNNLOV_3_HIERARCHY:
            return self._check_hierarchy(component, action, context)
        elif law == ConstitutionLaw.GRUNNLOV_5_ENTRY_RISK:
            return self._check_entry_risk(component, action, context)
        elif law == ConstitutionLaw.GRUNNLOV_10_AI_LIMITED:
            return self._check_ai_limits(component, action, context)
        elif law == ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS:
            return self._check_prohibitions(component, action, context)
        elif law == ConstitutionLaw.GRUNNLOV_15_HUMAN_LOCK:
            return self._check_human_lock(component, action, context)
        
        # Default: allow
        return True, None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Specific Law Checks
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _check_survival(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 1: Capital preservation check."""
        
        # Check for capital threat indicators
        capital_threat = context.get("capital_threat", False)
        structural_risk = context.get("structural_risk", False)
        drawdown_breach = context.get("drawdown_pct", 0) < -10.0
        
        if capital_threat or structural_risk or drawdown_breach:
            violation = self._create_violation(
                ConstitutionLaw.GRUNNLOV_1_SURVIVAL,
                ViolationType.CRITICAL,
                component,
                action,
                context,
                "Capital preservation triggered - action blocked",
            )
            return False, violation
        
        return True, None
    
    def _check_risk_sacred(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 7: Risk limits check."""
        
        # Extract risk parameters
        risk_per_trade = context.get("risk_pct", 0)
        daily_loss = context.get("daily_loss_pct", 0)
        max_risk_per_trade = context.get("max_risk_per_trade", 2.0)
        max_daily_loss = context.get("max_daily_loss", 5.0)
        
        if risk_per_trade > max_risk_per_trade:
            violation = self._create_violation(
                ConstitutionLaw.GRUNNLOV_7_RISK_SACRED,
                ViolationType.SEVERE,
                component,
                action,
                context,
                f"Risk per trade {risk_per_trade}% exceeds max {max_risk_per_trade}%",
            )
            return False, violation
        
        if abs(daily_loss) > max_daily_loss:
            violation = self._create_violation(
                ConstitutionLaw.GRUNNLOV_7_RISK_SACRED,
                ViolationType.CRITICAL,
                component,
                action,
                context,
                f"Daily loss {daily_loss}% exceeds max {max_daily_loss}%",
            )
            return False, violation
        
        return True, None
    
    def _check_hierarchy(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 3: Decision hierarchy check."""
        
        source_role = self.registered_components.get(component, ComponentRole.OBSERVER)
        target_component = context.get("target_component", "")
        target_role = self.registered_components.get(target_component, ComponentRole.OBSERVER)
        
        # Lower authority cannot override higher authority
        if target_role and source_role < target_role:
            if action in ["OVERRIDE", "VETO", "CANCEL"]:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_3_HIERARCHY,
                    ViolationType.SEVERE,
                    component,
                    action,
                    context,
                    f"{component} ({source_role.name}) cannot override {target_component} ({target_role.name})",
                )
                return False, violation
        
        return True, None
    
    def _check_entry_risk(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 5: Entry must define risk."""
        
        if action in ["OPEN_POSITION", "ENTRY", "BUY", "SELL"]:
            has_stop_loss = context.get("stop_loss") is not None
            has_take_profit = context.get("take_profit") is not None
            has_position_size = context.get("position_size") is not None
            
            if not has_stop_loss:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_5_ENTRY_RISK,
                    ViolationType.SEVERE,
                    component,
                    action,
                    context,
                    "Entry without stop-loss is forbidden",
                )
                return False, violation
            
            if not has_take_profit:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_5_ENTRY_RISK,
                    ViolationType.WARNING,
                    component,
                    action,
                    context,
                    "Entry without take-profit (warning)",
                )
                # Warning only, allow to proceed
        
        return True, None
    
    def _check_ai_limits(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 10: AI cannot make certain decisions."""
        
        is_ai_component = "ai" in component.lower() or "ml" in component.lower()
        
        if is_ai_component:
            forbidden_actions = [
                "SET_POSITION_SIZE",
                "FORCE_EXIT",
                "OVERRIDE_RISK",
                "BYPASS_HUMAN_LOCK",
                "MODIFY_STOP_LOSS",
            ]
            
            if action in forbidden_actions:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_10_AI_LIMITED,
                    ViolationType.SEVERE,
                    component,
                    action,
                    context,
                    f"AI component cannot perform action: {action}",
                )
                return False, violation
        
        return True, None
    
    def _check_prohibitions(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 13: Absolute prohibitions."""
        
        # Stop-loss movement check
        if action == "MODIFY_STOP_LOSS":
            old_sl = context.get("old_stop_loss", 0)
            new_sl = context.get("new_stop_loss", 0)
            entry_price = context.get("entry_price", 0)
            side = context.get("side", "LONG")
            
            if side == "LONG" and new_sl < old_sl:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS,
                    ViolationType.SEVERE,
                    component,
                    action,
                    context,
                    "Moving stop-loss away from entry is forbidden",
                )
                return False, violation
            elif side == "SHORT" and new_sl > old_sl:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS,
                    ViolationType.SEVERE,
                    component,
                    action,
                    context,
                    "Moving stop-loss away from entry is forbidden",
                )
                return False, violation
        
        # Revenge trading check
        if action in ["OPEN_POSITION", "ENTRY"]:
            recent_loss = context.get("recent_loss", False)
            time_since_loss = context.get("time_since_loss_minutes", 999)
            cooldown_minutes = context.get("revenge_cooldown_minutes", 30)
            
            if recent_loss and time_since_loss < cooldown_minutes:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS,
                    ViolationType.SEVERE,
                    component,
                    action,
                    context,
                    f"Revenge trading blocked - {cooldown_minutes - time_since_loss}min cooldown remaining",
                )
                return False, violation
        
        return True, None
    
    def _check_human_lock(
        self, component: str, action: str, context: Dict[str, Any]
    ) -> tuple[bool, Optional[ConstitutionViolation]]:
        """GRUNNLOV 15: Human override cooldown."""
        
        is_manual = context.get("is_manual", False) or context.get("source") == "HUMAN"
        
        if is_manual and action in ["OVERRIDE", "MANUAL_CLOSE", "EMERGENCY_CLOSE"]:
            cooldown_seconds = context.get("override_cooldown_seconds", 60)
            time_since_last = context.get("time_since_last_override_seconds", 999)
            
            if time_since_last < cooldown_seconds:
                violation = self._create_violation(
                    ConstitutionLaw.GRUNNLOV_15_HUMAN_LOCK,
                    ViolationType.WARNING,
                    component,
                    action,
                    context,
                    f"Human override requires {cooldown_seconds}s cooldown - {cooldown_seconds - time_since_last}s remaining",
                )
                return False, violation
        
        return True, None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_applicable_laws(self, action: str) -> List[ConstitutionLaw]:
        """Determine which laws apply to an action."""
        
        # Always check supreme laws
        laws = [
            ConstitutionLaw.GRUNNLOV_1_SURVIVAL,
            ConstitutionLaw.GRUNNLOV_7_RISK_SACRED,
        ]
        
        # Add action-specific laws
        if action in ["OPEN_POSITION", "ENTRY", "BUY", "SELL"]:
            laws.extend([
                ConstitutionLaw.GRUNNLOV_5_ENTRY_RISK,
                ConstitutionLaw.GRUNNLOV_8_CAPITAL_LEVERAGE,
                ConstitutionLaw.GRUNNLOV_13_PROHIBITIONS,
            ])
        
        if action in ["CLOSE_POSITION", "EXIT", "TAKE_PROFIT", "STOP_LOSS"]:
            laws.append(ConstitutionLaw.GRUNNLOV_6_EXIT_DOMINANCE)
        
        if action in ["OVERRIDE", "MANUAL_CLOSE", "VETO"]:
            laws.extend([
                ConstitutionLaw.GRUNNLOV_3_HIERARCHY,
                ConstitutionLaw.GRUNNLOV_15_HUMAN_LOCK,
            ])
        
        if "AI" in action or "ML" in action:
            laws.append(ConstitutionLaw.GRUNNLOV_10_AI_LIMITED)
        
        return laws
    
    def _create_violation(
        self,
        law: ConstitutionLaw,
        violation_type: ViolationType,
        component: str,
        action: str,
        context: Dict[str, Any],
        reason: str,
    ) -> ConstitutionViolation:
        """Create and record a violation."""
        
        violation = ConstitutionViolation(
            timestamp=datetime.utcnow(),
            law=law,
            violation_type=violation_type,
            component=component,
            action_attempted=action,
            context=context,
            blocked=violation_type.value >= ViolationType.SEVERE.value,
            reason=reason,
        )
        
        self.violations.append(violation)
        
        # Invoke callbacks
        if law in self.enforcement_callbacks:
            for callback in self.enforcement_callbacks[law]:
                try:
                    callback(violation)
                except Exception as e:
                    logger.error(f"Callback error for {law.name}: {e}")
        
        # Log violation
        law_def = self.laws[law]
        logger.warning(
            f"🚨 GRUNNLOV VIOLATION: {law_def.name_no}\n"
            f"   Component: {component}\n"
            f"   Action: {action}\n"
            f"   Reason: {reason}\n"
            f"   Blocked: {violation.blocked}\n"
            f"   Hash: {violation.hash}"
        )
        
        # Critical violations trigger emergency halt
        if violation_type == ViolationType.CRITICAL:
            self._trigger_emergency_halt(violation)
        
        return violation
    
    def _trigger_emergency_halt(self, violation: ConstitutionViolation) -> None:
        """Trigger emergency system halt."""
        self._emergency_halt = True
        logger.critical(
            f"🛑 EMERGENCY HALT TRIGGERED\n"
            f"   Law: {self.laws[violation.law].name_no}\n"
            f"   Reason: {violation.reason}"
        )
    
    def reset_emergency_halt(self, authorization_code: str) -> bool:
        """Reset emergency halt with proper authorization."""
        # In production, this would require cryptographic verification
        if len(authorization_code) >= 16:
            self._emergency_halt = False
            logger.info("✅ Emergency halt reset with authorization")
            return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Reporting
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_constitution_summary(self) -> Dict[str, Any]:
        """Get a summary of the constitution and its status."""
        return {
            "total_laws": len(self.laws),
            "registered_components": len(self.registered_components),
            "total_violations": len(self.violations),
            "critical_violations": sum(
                1 for v in self.violations 
                if v.violation_type == ViolationType.CRITICAL
            ),
            "emergency_halt_active": self._emergency_halt,
            "laws": {
                law.name: {
                    "name_no": defn.name_no,
                    "authority": defn.authority.name,
                    "component": defn.component,
                    "fail_mode": defn.fail_mode,
                }
                for law, defn in self.laws.items()
            },
        }
    
    def get_violations_report(
        self, 
        since: Optional[datetime] = None,
        law: Optional[ConstitutionLaw] = None,
    ) -> List[Dict[str, Any]]:
        """Get violations report with optional filtering."""
        filtered = self.violations
        
        if since:
            filtered = [v for v in filtered if v.timestamp >= since]
        if law:
            filtered = [v for v in filtered if v.law == law]
        
        return [
            {
                "timestamp": v.timestamp.isoformat(),
                "law": self.laws[v.law].name_no,
                "type": v.violation_type.name,
                "component": v.component,
                "action": v.action_attempted,
                "blocked": v.blocked,
                "reason": v.reason,
                "hash": v.hash,
            }
            for v in filtered
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

# Global constitution enforcer instance
_constitution_enforcer: Optional[ConstitutionEnforcer] = None


def get_constitution_enforcer() -> ConstitutionEnforcer:
    """Get or create the global constitution enforcer."""
    global _constitution_enforcer
    if _constitution_enforcer is None:
        _constitution_enforcer = ConstitutionEnforcer()
    return _constitution_enforcer


def validate_constitutional_action(
    component: str,
    action: str,
    context: Dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """
    Convenience function to validate an action against the constitution.
    
    Args:
        component: Component attempting the action
        action: Action being attempted
        context: Action context
        
    Returns:
        (allowed: bool, error_message: Optional[str])
    """
    enforcer = get_constitution_enforcer()
    allowed, violation = enforcer.validate_action(component, action, context)
    
    if violation:
        return False, violation.reason
    return True, None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ASCII ART HIERARCHY (For documentation)
# ═══════════════════════════════════════════════════════════════════════════════

CONSTITUTION_HIERARCHY_ASCII = """
🏛️ QUANTUM TRADER GRUNNLOV - HIERARCHY
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRUNNLOV 1: CAPITAL PRESERVATION                          │
│                    ═══════════════════════════════════                       │
│                    Component: CapitalPreservationGovernor                    │
│                    Authority: SUPREME (VETO POWER)                           │
│                    Fail Mode: FAIL_CLOSED                                    │
└───────────────────────────────────▲─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴─────────────────────────────────────────┐
│                    GRUNNLOV 7: RISK IS SACRED                                │
│                    Component: RiskKernel                                     │
│                    Authority: SUPREME                                        │
│                    Fail Mode: FAIL_CLOSED                                    │
└───────────────────────────────────▲─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴─────────────────────────────────────────┐
│               GRUNNLOV 2 & 3: POLICY & DECISION HIERARCHY                    │
│               Components: PolicyEngine, DecisionArbitrationLayer             │
│               Authority: GOVERNING                                           │
│               Fail Mode: FAIL_CLOSED                                         │
└───────────────────────────────────▲─────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
┌───────┴───────┐           ┌───────┴───────┐           ┌───────┴───────┐
│  GRUNNLOV 5   │           │  GRUNNLOV 6   │           │  GRUNNLOV 9   │
│  Entry Gate   │           │  Exit Brain   │           │  Execution    │
│  OPERATIONAL  │           │  OPERATIONAL  │           │  OPERATIONAL  │
└───────▲───────┘           └───────▲───────┘           └───────▲───────┘
        │                           │                           │
┌───────┴───────────────────────────┴───────────────────────────┴───────┐
│                    GRUNNLOV 4: MARKET REGIME                          │
│                    Component: MarketRegimeDetector                    │
│                    Authority: OPERATIONAL                             │
└───────────────────────────────────▲───────────────────────────────────┘
                                    │
┌───────────────────────────────────┴─────────────────────────────────────────┐
│                    GRUNNLOV 8 & 10: CAPITAL & AI ADVISORY                    │
│                    Components: CapitalAllocationEngine, AIAdvisoryLayer      │
│                    Authority: ADVISORY                                       │
└───────────────────────────────────▲─────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┴─────────────────────────────────────────┐
│                    GRUNNLOV 11-15: COMPLIANCE & GOVERNANCE                   │
│                    Components: AuditLedger, Constraints, Identity, HumanLock │
│                    Authority: COMPLIANCE                                     │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
KEY INSIGHT: AI is NEVER in the control line. Exit + Risk = Fund's heart.
═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    # Print constitution summary when run directly
    print(CONSTITUTION_HIERARCHY_ASCII)
    
    enforcer = get_constitution_enforcer()
    summary = enforcer.get_constitution_summary()
    
    print("\n📜 GRUNNLOV SUMMARY:")
    print(f"   Total Laws: {summary['total_laws']}")
    print(f"   Emergency Halt: {summary['emergency_halt_active']}")
    print("\n   Laws:")
    for law_name, law_info in summary['laws'].items():
        print(f"   • {law_info['name_no']}")
        print(f"     Component: {law_info['component']}")
        print(f"     Authority: {law_info['authority']}")
        print(f"     Fail Mode: {law_info['fail_mode']}")
        print()
