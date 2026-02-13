"""
🏛️ Quantum Trader Governance Module
====================================

This module contains the constitutional framework (15 Grunnlover)
and failure scenarios that govern all trading decisions.

Key Components:
- constitution.py: Core definitions and validation logic
- grunnlov_components.py: Implementation of constitutional components
- grunnlov_integration.py: Integration layer with trading system
- failure_scenarios.py: 14 predefined failure scenarios (5 classes)

Usage:
    from backend.domains.governance import (
        validate_constitutional_action,
        get_grunnlov_status,
        GrunnlovPreTradeValidator,
        get_failure_monitor,
        can_trade,
        SystemState,
    )
"""

from backend.domains.governance.constitution import (
    ConstitutionLaw,
    LawAuthority,
    ViolationType,
    ComponentRole,
    LawDefinition,
    GRUNNLOVER,
    ConstitutionViolation,
    ConstitutionEnforcer,
    get_constitution_enforcer,
    validate_constitutional_action,
    CONSTITUTION_HIERARCHY_ASCII,
)

from backend.domains.governance.grunnlov_components import (
    CapitalPreservationGovernor,
    DecisionArbitrationLayer,
    EntryQualificationGate,
    RiskKernel,
    ConstraintEnforcementLayer,
    StrategyScopeController,
    HumanOverrideLock,
    AIAdvisoryLayer,
    GrunnlovComponentRegistry,
    get_grunnlov_registry,
    EntryPlan,
    AIAdvice,
    ComponentPriority,
)

from backend.domains.governance.grunnlov_integration import (
    GrunnlovPreTradeValidator,
    ConstitutionalViolationError,
    requires_constitutional_approval,
    integrate_grunnlov_with_governor,
    get_grunnlov_status,
    print_grunnlov_hierarchy,
)

from backend.domains.governance.failure_scenarios import (
    FailureClass,
    SystemState,
    FailureScenario,
    TriggerCondition,
    FailureResponse,
    ScenarioDefinition,
    FailureEvent,
    FAILURE_SCENARIOS,
    FailureScenarioMonitor,
    get_failure_monitor,
    can_trade,
    can_exit,
    get_system_state,
    trigger_kill_switch,
    STOPP_KART,
)

from backend.domains.governance.kill_switch_manifest import (
    DangerRank,
    DANGER_RANKING,
    FailureHandler,
    FAILURE_HANDLERS,
    RestartPhase,
    PhaseRequirements,
    RESTART_PHASES,
    RestartState,
    RestartProtocolManager,
    get_restart_manager,
    KILL_SWITCH_MANIFEST,
    print_kill_switch_manifest,
    print_danger_ranking,
    print_restart_phases,
)

from backend.domains.governance.mvp_architecture import (
    MVPComponent,
    FailureHandlerAction,
    ComponentHandler,
    ComponentStatus,
    MVPArchitecture,
    FAILURE_TO_COMPONENT_MAP,
    KOBLINGSTABELL,
    get_mvp_architecture,
)

from backend.domains.governance.black_swan_playbook import (
    BlackSwanScenario,
    BlackSwanResponse,
    BlackSwanEvent,
    CrisisPhase,
    CrisisPhaseAction,
    BlackSwanPlaybook,
    BLACK_SWAN_RESPONSES,
    CRISIS_TIMELINE,
    GOLDEN_RULES,
    get_black_swan_playbook,
)

from backend.domains.governance.test_failure_scenarios import (
    TestSeverity,
    FailureTestCase,
    TestResult,
    FailureTestRunner,
    TEST_CASES,
)

from backend.domains.governance.preflight_checklist import (
    ChecklistCategory,
    CheckStatus,
    CheckItem,
    CategoryResult,
    PreFlightResult,
    PreFlightChecklist,
    CHECKLIST_ITEMS,
    NO_GO_RULES,
    GO_CRITERION,
    get_preflight_checklist,
    reset_preflight_checklist,
)

from backend.domains.governance.crisis_simulation import (
    CrisisSimPhase,
    PhaseConfig,
    CrisisState,
    DiagnosisReport,
    ShadowResult,
    CrisisSimulator,
    CRISIS_PHASES,
    GOLDEN_RULES_CRISIS,
    get_crisis_simulator,
    reset_crisis_simulator,
)

from backend.domains.governance.microservice_architecture import (
    MicroserviceID,
    ServicePower,
    ServicePermission,
    MicroserviceSpec,
    MicroserviceArchitecture,
    MICROSERVICES,
    FAILURE_TO_SERVICE_MAP,
    COMMAND_HIERARCHY,
    COMMAND_HIERARCHY_LEVELS,
    KEY_INSIGHTS,
    ARCHITECTURE_DIAGRAM,
    get_microservice_architecture,
)

from backend.domains.governance.event_flow_architecture import (
    EventChannel,
    EventType,
    EventPhase,
    EventFlowStep,
    MVPPriority,
    MVPComponent,
    TechComponent,
    TechSpec,
    EventFlowArchitecture,
    EVENT_FLOW,
    MASTER_EVENT_FLOW_DIAGRAM,
    MVP_BUILD_ORDER,
    NOT_IN_MVP,
    MVP_TRUTH,
    TECH_STACK,
    EXECUTION_PERMISSIONS,
    EVENT_CHANNELS_DIAGRAM,
    PROFESSIONAL_SUMMARY,
    get_event_flow_architecture,
    reset_event_flow_architecture,
)

from backend.domains.governance.mvp_build_plan import (
    BuildWeek,
    DeliverableStatus,
    Deliverable,
    WeekPlan,
    ExitType,
    ExitFormula,
    SimulationEventType,
    SimulationDecision,
    SimulationEvent,
    MVPBuildPlanManager,
    MVP_BUILD_PLAN,
    EXIT_FORMULAS,
    EXIT_PRIORITY_ORDER,
    LIVE_DAY_SIMULATION,
    LIVE_DAY_SUMMARY,
    SLUTTSANNHET,
    get_mvp_build_manager,
    reset_mvp_build_manager,
)

from backend.domains.governance.stress_test_scaling import (
    StressTestResponse,
    StressTestTrade,
    StressTestResult,
    ScalingLevel,
    ScalingRequirements,
    ScalingActions,
    ScalingLevelSpec,
    DayPhase,
    DayPhaseSpec,
    Day0SuccessCriteria,
    StressTestScalingManager,
    STRESS_TEST_SEQUENCE,
    STRESS_TEST_PASS_CRITERIA,
    PSYCHOLOGICAL_REALITY,
    SCALING_LEVELS,
    SCALING_RULES,
    ANTI_SCALING_RULES,
    AUTO_DOWNSCALE_TRIGGERS,
    DAY_0_PHASES,
    DAY_0_ABSOLUTE_PROHIBITIONS,
    DAY_0_SUCCESS,
    DAY_0_THREE_QUESTIONS,
    FINAL_CONCLUSION,
    get_stress_test_manager,
    reset_stress_test_manager,
)

from backend.domains.governance.founders_operating_manual import (
    SimulationParameters,
    TradeResult,
    LossSeriesStats,
    SimulationResult,
    NoTradeSeverity,
    NoTradeCondition,
    RiskType,
    ManualPart,
    ManualSection,
    FoundersOperatingManual,
    TRADE_100_DISTRIBUTION,
    LOSS_SERIES_REALITY,
    KEY_INSIGHTS_100_TRADES,
    NO_TRADE_CONDITIONS,
    NO_TRADE_GOLDEN_RULE,
    RISK_DISCLOSURE_SHORT,
    FUND_DOES_NOT_PROMISE,
    FUND_EXPLICITLY_PRIORITIZES,
    FULL_RISK_DISCLOSURE,
    RISK_TYPES,
    FOUNDERS_MANUAL,
    SLUTTORD,
    get_founders_manual,
    reset_founders_manual,
)

__all__ = [
    # Core Enums (Constitution)
    "ConstitutionLaw",
    "LawAuthority",
    "ViolationType",
    "ComponentRole",
    "ComponentPriority",
    
    # Core Enums (Failure)
    "FailureClass",
    "SystemState",
    "FailureScenario",
    
    # Core Enums (Kill-Switch)
    "DangerRank",
    "RestartPhase",
    
    # Dataclasses (Constitution)
    "LawDefinition",
    "ConstitutionViolation",
    "EntryPlan",
    "AIAdvice",
    
    # Dataclasses (Failure)
    "TriggerCondition",
    "FailureResponse",
    "ScenarioDefinition",
    "FailureEvent",
    
    # Dataclasses (Kill-Switch)
    "FailureHandler",
    "PhaseRequirements",
    "RestartState",
    
    # Components (Constitution)
    "CapitalPreservationGovernor",
    "DecisionArbitrationLayer",
    "EntryQualificationGate",
    "RiskKernel",
    "ConstraintEnforcementLayer",
    "StrategyScopeController",
    "HumanOverrideLock",
    "AIAdvisoryLayer",
    "GrunnlovComponentRegistry",
    
    # Enforcement
    "ConstitutionEnforcer",
    "GrunnlovPreTradeValidator",
    "ConstitutionalViolationError",
    "FailureScenarioMonitor",
    "RestartProtocolManager",
    
    # Functions (Constitution)
    "get_constitution_enforcer",
    "get_grunnlov_registry",
    "validate_constitutional_action",
    "requires_constitutional_approval",
    "integrate_grunnlov_with_governor",
    "get_grunnlov_status",
    "print_grunnlov_hierarchy",
    
    # Functions (Failure)
    "get_failure_monitor",
    "can_trade",
    "can_exit",
    "get_system_state",
    "trigger_kill_switch",
    
    # Functions (Kill-Switch)
    "get_restart_manager",
    "print_kill_switch_manifest",
    "print_danger_ranking",
    "print_restart_phases",
    
    # MVP Architecture
    "MVPComponent",
    "FailureHandlerAction",
    "ComponentHandler",
    "ComponentStatus",
    "MVPArchitecture",
    "FAILURE_TO_COMPONENT_MAP",
    "KOBLINGSTABELL",
    "get_mvp_architecture",
    
    # Black Swan Playbook
    "BlackSwanScenario",
    "BlackSwanResponse",
    "BlackSwanEvent",
    "CrisisPhase",
    "CrisisPhaseAction",
    "BlackSwanPlaybook",
    "BLACK_SWAN_RESPONSES",
    "CRISIS_TIMELINE",
    "GOLDEN_RULES",
    "get_black_swan_playbook",
    
    # Test Framework
    "TestSeverity",
    "FailureTestCase",
    "TestResult",
    "FailureTestRunner",
    "TEST_CASES",
    
    # Pre-Flight Checklist
    "ChecklistCategory",
    "CheckStatus",
    "CheckItem",
    "CategoryResult",
    "PreFlightResult",
    "PreFlightChecklist",
    "CHECKLIST_ITEMS",
    "NO_GO_RULES",
    "GO_CRITERION",
    "get_preflight_checklist",
    "reset_preflight_checklist",
    
    # Crisis Simulation
    "CrisisSimPhase",
    "PhaseConfig",
    "CrisisState",
    "DiagnosisReport",
    "ShadowResult",
    "CrisisSimulator",
    "CRISIS_PHASES",
    "GOLDEN_RULES_CRISIS",
    "get_crisis_simulator",
    "reset_crisis_simulator",
    
    # Microservice Architecture
    "MicroserviceID",
    "ServicePower",
    "ServicePermission",
    "MicroserviceSpec",
    "MicroserviceArchitecture",
    "MICROSERVICES",
    "FAILURE_TO_SERVICE_MAP",
    "COMMAND_HIERARCHY",
    "COMMAND_HIERARCHY_LEVELS",
    "KEY_INSIGHTS",
    "ARCHITECTURE_DIAGRAM",
    "get_microservice_architecture",
    
    # Event Flow Architecture
    "EventChannel",
    "EventType",
    "EventPhase",
    "EventFlowStep",
    "MVPPriority",
    "MVPComponent",
    "TechComponent",
    "TechSpec",
    "EventFlowArchitecture",
    "EVENT_FLOW",
    "MASTER_EVENT_FLOW_DIAGRAM",
    "MVP_BUILD_ORDER",
    "NOT_IN_MVP",
    "MVP_TRUTH",
    "TECH_STACK",
    "EXECUTION_PERMISSIONS",
    "EVENT_CHANNELS_DIAGRAM",
    "PROFESSIONAL_SUMMARY",
    "get_event_flow_architecture",
    "reset_event_flow_architecture",
    
    # MVP Build Plan (3 Weeks)
    "BuildWeek",
    "DeliverableStatus",
    "Deliverable",
    "WeekPlan",
    "ExitType",
    "ExitFormula",
    "SimulationEventType",
    "SimulationDecision",
    "SimulationEvent",
    "MVPBuildPlanManager",
    "MVP_BUILD_PLAN",
    "EXIT_FORMULAS",
    "EXIT_PRIORITY_ORDER",
    "LIVE_DAY_SIMULATION",
    "LIVE_DAY_SUMMARY",
    "SLUTTSANNHET",
    "get_mvp_build_manager",
    "reset_mvp_build_manager",
    
    # Stress Test & Scaling
    "StressTestResponse",
    "StressTestTrade",
    "StressTestResult",
    "ScalingLevel",
    "ScalingRequirements",
    "ScalingActions",
    "ScalingLevelSpec",
    "DayPhase",
    "DayPhaseSpec",
    "Day0SuccessCriteria",
    "StressTestScalingManager",
    "STRESS_TEST_SEQUENCE",
    "STRESS_TEST_PASS_CRITERIA",
    "PSYCHOLOGICAL_REALITY",
    "SCALING_LEVELS",
    "SCALING_RULES",
    "ANTI_SCALING_RULES",
    "AUTO_DOWNSCALE_TRIGGERS",
    "DAY_0_PHASES",
    "DAY_0_ABSOLUTE_PROHIBITIONS",
    "DAY_0_SUCCESS",
    "DAY_0_THREE_QUESTIONS",
    "FINAL_CONCLUSION",
    "get_stress_test_manager",
    "reset_stress_test_manager",
    
    # Founders Operating Manual
    "SimulationParameters",
    "TradeResult",
    "LossSeriesStats",
    "SimulationResult",
    "NoTradeSeverity",
    "NoTradeCondition",
    "RiskType",
    "ManualPart",
    "ManualSection",
    "FoundersOperatingManual",
    "TRADE_100_DISTRIBUTION",
    "LOSS_SERIES_REALITY",
    "KEY_INSIGHTS_100_TRADES",
    "NO_TRADE_CONDITIONS",
    "NO_TRADE_GOLDEN_RULE",
    "RISK_DISCLOSURE_SHORT",
    "FUND_DOES_NOT_PROMISE",
    "FUND_EXPLICITLY_PRIORITIZES",
    "FULL_RISK_DISCLOSURE",
    "RISK_TYPES",
    "FOUNDERS_MANUAL",
    "SLUTTORD",
    "get_founders_manual",
    "reset_founders_manual",
    
    # Constants
    "GRUNNLOVER",
    "CONSTITUTION_HIERARCHY_ASCII",
    "FAILURE_SCENARIOS",
    "STOPP_KART",
    "DANGER_RANKING",
    "FAILURE_HANDLERS",
    "RESTART_PHASES",
    "KILL_SWITCH_MANIFEST",
]
