"""
🛑 HEDGE FUND FAILURE SCENARIOS
================================
Forhåndsdefinerte scenarier for når systemet SKAL STOPPE SEG SELV.

Ikke feilsøking. Ikke "hva gikk galt?".
Men: "Hva er så farlig at vi ikke får lov å fortsette?"

METAREGEL: Systemet stopper alltid for tidlig – aldri for sent.

Classes:
    A - ABSOLUTT SYSTEMSTOPP (KILL-SWITCH) - Øyeblikkelig, ingen unntak
    B - STRATEGISK PAUSE - Systemet lever, men nekter å trade
    C - EXIT-TVANG - Kun exits tillatt, ingen nye bets
    D - MENNESKELIG RISIKO - Beskytter systemet mot DEG
    E - INFRASTRUKTUR & EKSTERN RISIKO
"""

import logging
from enum import Enum, IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Callable, Tuple
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS: Failure Classes and System States
# ═══════════════════════════════════════════════════════════════════════════════

class FailureClass(str, Enum):
    """
    Failure severity classes - each has different response protocols.
    """
    A = "KILL_SWITCH"      # Absolutt systemstopp - øyeblikkelig
    B = "STRATEGIC_PAUSE"  # Systemet lever, nekter å trade
    C = "EXIT_ONLY"        # Kun exits tillatt
    D = "HUMAN_RISK"       # Beskytter mot mennesker
    E = "INFRASTRUCTURE"   # Ekstern/teknisk risiko


class SystemState(str, Enum):
    """
    System operating states based on failure responses.
    """
    NORMAL = "NORMAL"              # Full operasjon
    SAFE_MODE = "SAFE_MODE"        # Kun exits, ingen entries
    PAUSED = "PAUSED"              # Observerer, ingen trading
    FROZEN = "FROZEN"              # Alt stoppet, krever manuell restart
    REDUCE_ONLY = "REDUCE_ONLY"    # Kun posisjon-reduksjon
    OBSERVER_ONLY = "OBSERVER"     # Kun overvåking, ingen ordrer


class FailureScenario(str, Enum):
    """
    All predefined failure scenarios.
    
    Naming: {CLASS}_{NUMBER}_{SHORT_NAME}
    """
    # ═══ CLASS A: ABSOLUTT SYSTEMSTOPP ═══
    A1_CAPITAL_BREACH = "A1_CAPITAL_BREACH"
    A2_RISK_RULE_VIOLATION = "A2_RISK_RULE_VIOLATION"
    A3_DATA_COLLAPSE = "A3_DATA_COLLAPSE"
    A4_MARKET_PANIC = "A4_MARKET_PANIC"
    
    # ═══ CLASS B: STRATEGISK PAUSE ═══
    B1_REGIME_UNCERTAINTY = "B1_REGIME_UNCERTAINTY"
    B2_EDGE_DECAY = "B2_EDGE_DECAY"
    B3_OVERTRADING = "B3_OVERTRADING"
    
    # ═══ CLASS C: EXIT-TVANG ═══
    C1_REGIME_SHIFT_IN_TRADE = "C1_REGIME_SHIFT_IN_TRADE"
    C2_TIME_BASED_STRESS = "C2_TIME_BASED_STRESS"
    
    # ═══ CLASS D: MENNESKELIG RISIKO ═══
    D1_MANUAL_OVERRIDE_ATTEMPT = "D1_MANUAL_OVERRIDE_ATTEMPT"
    D2_EGO_ESCALATION = "D2_EGO_ESCALATION"
    
    # ═══ CLASS E: INFRASTRUKTUR ═══
    E1_EXCHANGE_RISK = "E1_EXCHANGE_RISK"
    E2_INFRASTRUCTURE_FAILURE = "E2_INFRASTRUCTURE_FAILURE"


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES: Scenario Definitions and Events
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriggerCondition:
    """A single trigger condition for a failure scenario."""
    name: str
    description: str
    check_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    threshold: Optional[float] = None
    comparison: str = ">"  # >, <, >=, <=, ==, !=


@dataclass
class FailureResponse:
    """Response actions when a failure scenario is triggered."""
    primary_action: str
    secondary_actions: List[str]
    system_state: SystemState
    cooldown_seconds: int = 0
    requires_manual_review: bool = False
    reduce_only: bool = False
    audit_required: bool = True


@dataclass
class ScenarioDefinition:
    """Complete definition of a failure scenario."""
    scenario: FailureScenario
    failure_class: FailureClass
    name_no: str  # Norwegian name
    name_en: str  # English name
    description: str
    triggers: List[TriggerCondition]
    response: FailureResponse
    principle: str  # The guiding principle (📌)


@dataclass
class FailureEvent:
    """Record of a triggered failure scenario."""
    scenario: FailureScenario
    failure_class: FailureClass
    timestamp: datetime
    trigger_data: Dict[str, Any]
    response_taken: str
    system_state_before: SystemState
    system_state_after: SystemState
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_note: str = ""
    
    @property
    def event_hash(self) -> str:
        """Unique hash for this event."""
        data = f"{self.scenario.value}:{self.timestamp.isoformat()}:{str(self.trigger_data)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS: All 14 Failure Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

FAILURE_SCENARIOS: Dict[FailureScenario, ScenarioDefinition] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔴 CLASS A: ABSOLUTT SYSTEMSTOPP (KILL-SWITCH)
    # Systemet stopper ØYEBLIKKELIG. Ingen unntak.
    # ═══════════════════════════════════════════════════════════════════════════
    
    FailureScenario.A1_CAPITAL_BREACH: ScenarioDefinition(
        scenario=FailureScenario.A1_CAPITAL_BREACH,
        failure_class=FailureClass.A,
        name_no="Kapitalbrudd",
        name_en="Capital Breach",
        description="Kritiske kapitalbegrensninger er brutt",
        triggers=[
            TriggerCondition("max_daily_loss", "Maks daglig tap nådd", threshold=-5.0, comparison="<="),
            TriggerCondition("max_total_drawdown", "Maks total drawdown nådd", threshold=-15.0, comparison="<="),
            TriggerCondition("pnl_anomaly", "Uforklarlig PnL-avvik", threshold=10.0, comparison=">="),
        ],
        response=FailureResponse(
            primary_action="ALL_TRADING_STOPPED",
            secondary_actions=["REDUCE_ONLY_MODE", "COOLDOWN_BEFORE_RESTART"],
            system_state=SystemState.REDUCE_ONLY,
            cooldown_seconds=3600,  # 1 hour
            requires_manual_review=True,
            reduce_only=True,
        ),
        principle="📌 Overlevelse > forklaring.",
    ),
    
    FailureScenario.A2_RISK_RULE_VIOLATION: ScenarioDefinition(
        scenario=FailureScenario.A2_RISK_RULE_VIOLATION,
        failure_class=FailureClass.A,
        name_no="Risiko-regelbrudd",
        name_en="Risk Rule Violation",
        description="Fundamentale risikogrenser er brutt",
        triggers=[
            TriggerCondition("risk_per_trade_exceeded", "Risiko per trade overskredet", threshold=2.0, comparison=">"),
            TriggerCondition("leverage_over_policy", "Leverage over policy", threshold=10.0, comparison=">"),
            TriggerCondition("stop_loss_manipulated", "Stop/exit manipulert"),
        ],
        response=FailureResponse(
            primary_action="FULL_STOP",
            secondary_actions=["FLAG_POLICY_BREACH", "REQUIRE_MANUAL_REVIEW"],
            system_state=SystemState.FROZEN,
            cooldown_seconds=7200,  # 2 hours
            requires_manual_review=True,
            audit_required=True,
        ),
        principle="📌 Regler brytes aldri 'litt'.",
    ),
    
    FailureScenario.A3_DATA_COLLAPSE: ScenarioDefinition(
        scenario=FailureScenario.A3_DATA_COLLAPSE,
        failure_class=FailureClass.A,
        name_no="Systemisk datakollaps",
        name_en="Systemic Data Collapse",
        description="Kritiske datafeil som gjør trading umulig",
        triggers=[
            TriggerCondition("missing_price_feed", "Manglende prisfeed"),
            TriggerCondition("async_data", "Asynkron data"),
            TriggerCondition("illogical_price_movement", "Ulogiske prisbevegelser", threshold=20.0, comparison=">"),
            TriggerCondition("latency_exceeded", "Latency > toleranse", threshold=5000, comparison=">"),  # 5s
        ],
        response=FailureResponse(
            primary_action="BLOCK_ALL_ENTRY",
            secondary_actions=["ALLOW_EXITS_ONLY", "ENTER_SAFE_MODE"],
            system_state=SystemState.SAFE_MODE,
            cooldown_seconds=300,  # 5 minutes auto-retry
        ),
        principle="📌 Feil data = farligere enn ingen data.",
    ),
    
    FailureScenario.A4_MARKET_PANIC: ScenarioDefinition(
        scenario=FailureScenario.A4_MARKET_PANIC,
        failure_class=FailureClass.A,
        name_no="Marked i panikk / strukturell krise",
        name_en="Market Panic / Structural Crisis",
        description="Ekstreme markedsforhold som krever flat posisjon",
        triggers=[
            TriggerCondition("volatility_explosion", "Volatilitetseksplosjon", threshold=3.0, comparison=">="),  # 3x normal
            TriggerCondition("liquidity_drought", "Likviditet tørker", threshold=0.3, comparison="<="),  # 30% of normal
            TriggerCondition("exchange_anomaly", "Exchange-anomali"),
            TriggerCondition("funding_shock", "Funding-shock", threshold=0.1, comparison=">="),  # 10%+
        ],
        response=FailureResponse(
            primary_action="FLAT_MANDATE",
            secondary_actions=["NO_NEW_POSITIONS", "OBSERVER_ONLY_MODE"],
            system_state=SystemState.OBSERVER_ONLY,
            cooldown_seconds=1800,  # 30 minutes
        ),
        principle="📌 Cash er en posisjon.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟠 CLASS B: STRATEGISK PAUSE (SYSTEMET "TREKKER SEG")
    # Systemet lever, men nekter å trade.
    # ═══════════════════════════════════════════════════════════════════════════
    
    FailureScenario.B1_REGIME_UNCERTAINTY: ScenarioDefinition(
        scenario=FailureScenario.B1_REGIME_UNCERTAINTY,
        failure_class=FailureClass.B,
        name_no="Regime-usikkerhet",
        name_en="Regime Uncertainty",
        description="Systemet er usikkert på markedsregime",
        triggers=[
            TriggerCondition("conflicting_regime_signals", "Motstridende regimesignaler"),
            TriggerCondition("edge_collapse", "Edge-kollaps", threshold=0.0, comparison="<="),
            TriggerCondition("low_confidence", "For lav confidence", threshold=0.65, comparison="<"),
        ],
        response=FailureResponse(
            primary_action="BLOCK_ENTRY",
            secondary_actions=["ALLOW_EXITS_ONLY", "CONTINUE_OBSERVATION"],
            system_state=SystemState.SAFE_MODE,
            cooldown_seconds=900,  # 15 minutes
        ),
        principle="📌 Når systemet er usikker, er svaret alltid NEI.",
    ),
    
    FailureScenario.B2_EDGE_DECAY: ScenarioDefinition(
        scenario=FailureScenario.B2_EDGE_DECAY,
        failure_class=FailureClass.B,
        name_no="Edge-forfall",
        name_en="Edge Decay",
        description="Strategi-edge er svekket eller borte",
        triggers=[
            TriggerCondition("strategy_losing_unexplained", "Strategi taper uten forklaring"),
            TriggerCondition("historical_edge_not_reproducing", "Historisk edge ikke reproduseres"),
            TriggerCondition("performance_drift", "Performance-drift", threshold=-10.0, comparison="<="),
        ],
        response=FailureResponse(
            primary_action="FREEZE_STRATEGY",
            secondary_actions=["REQUIRE_REVALIDATION", "SHADOW_TESTING"],
            system_state=SystemState.PAUSED,
            cooldown_seconds=86400,  # 24 hours
            requires_manual_review=True,
        ),
        principle="📌 Ingen edge er evig.",
    ),
    
    FailureScenario.B3_OVERTRADING: ScenarioDefinition(
        scenario=FailureScenario.B3_OVERTRADING,
        failure_class=FailureClass.B,
        name_no="Overtrading-indikator",
        name_en="Overtrading Indicator",
        description="Systemet trader for mye og for raskt",
        triggers=[
            TriggerCondition("too_many_trades", "For mange trades", threshold=20, comparison=">"),  # per day
            TriggerCondition("too_short_holding", "For kort holdetid", threshold=300, comparison="<"),  # 5 min avg
            TriggerCondition("fees_vs_pnl_ratio", "Økende fees vs PnL", threshold=0.3, comparison=">="),  # 30%+
        ],
        response=FailureResponse(
            primary_action="REDUCE_TRADE_RATE",
            secondary_actions=["COOLDOWN", "SCALE_DOWN_CAPITAL"],
            system_state=SystemState.SAFE_MODE,
            cooldown_seconds=3600,  # 1 hour
        ),
        principle="📌 Markedet belønner ikke hyperaktivitet.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟡 CLASS C: EXIT-TVANG (SYSTEMET REDDER SEG)
    # Systemet tillater kun exits, ingen nye bets.
    # ═══════════════════════════════════════════════════════════════════════════
    
    FailureScenario.C1_REGIME_SHIFT_IN_TRADE: ScenarioDefinition(
        scenario=FailureScenario.C1_REGIME_SHIFT_IN_TRADE,
        failure_class=FailureClass.C,
        name_no="Regime-skift i åpen trade",
        name_en="Regime Shift During Trade",
        description="Markedsregime endrer seg mens vi har åpen posisjon",
        triggers=[
            TriggerCondition("trend_to_chop", "Trend → chop"),
            TriggerCondition("volatility_flip", "Volatility flip"),
            TriggerCondition("macro_change", "Makro-endring"),
        ],
        response=FailureResponse(
            primary_action="PARTIAL_EXITS",
            secondary_actions=["TRAILING_TIGHTENING", "FULL_EXIT_ON_CONFIRMATION"],
            system_state=SystemState.REDUCE_ONLY,
            reduce_only=True,
        ),
        principle="📌 Markedet skylder deg ingenting.",
    ),
    
    FailureScenario.C2_TIME_BASED_STRESS: ScenarioDefinition(
        scenario=FailureScenario.C2_TIME_BASED_STRESS,
        failure_class=FailureClass.C,
        name_no="Tid-basert stress",
        name_en="Time-Based Stress",
        description="Trade har vart for lenge uten fremgang",
        triggers=[
            TriggerCondition("trade_too_long", "Trade varer for lenge", threshold=86400, comparison=">"),  # 24h
            TriggerCondition("pnl_stagnation", "PnL stagnerer", threshold=0.1, comparison="<"),  # <0.1% movement
            TriggerCondition("funding_eating_edge", "Funding spiser edge", threshold=0.05, comparison=">="),  # 5%+
        ],
        response=FailureResponse(
            primary_action="EXIT_REGARDLESS_OF_PNL",
            secondary_actions=["NO_WAIT_MORE"],
            system_state=SystemState.REDUCE_ONLY,
            reduce_only=True,
        ),
        principle="📌 Tid er risiko.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔵 CLASS D: MENNESKELIG RISIKO (DEN FARLIGSTE)
    # Systemet beskytter seg mot DEG.
    # ═══════════════════════════════════════════════════════════════════════════
    
    FailureScenario.D1_MANUAL_OVERRIDE_ATTEMPT: ScenarioDefinition(
        scenario=FailureScenario.D1_MANUAL_OVERRIDE_ATTEMPT,
        failure_class=FailureClass.D,
        name_no="Manuell override-forsøk",
        name_en="Manual Override Attempt",
        description="Menneske forsøker å overstyre systemet",
        triggers=[
            TriggerCondition("live_change_detected", "Endring live"),
            TriggerCondition("parameter_tweaking", "Parameter-tweaking"),
            TriggerCondition("override_after_loss", "Overstyring etter tap"),
        ],
        response=FailureResponse(
            primary_action="FREEZE_SYSTEM",
            secondary_actions=["COOLDOWN", "FULL_AUDIT_LOG"],
            system_state=SystemState.FROZEN,
            cooldown_seconds=300,  # 5 minutes
            requires_manual_review=True,
            audit_required=True,
        ),
        principle="📌 Emosjoner er ikke input.",
    ),
    
    FailureScenario.D2_EGO_ESCALATION: ScenarioDefinition(
        scenario=FailureScenario.D2_EGO_ESCALATION,
        failure_class=FailureClass.D,
        name_no="Ego-eskalering",
        name_en="Ego Escalation",
        description="Farlige mønstre i menneskelig atferd",
        triggers=[
            TriggerCondition("increasing_size_after_loss", "Økende size etter tap"),
            TriggerCondition("just_one_more_trade", "'Bare én trade til'"),
            TriggerCondition("strategy_hopping", "Strategi-hopping"),
        ],
        response=FailureResponse(
            primary_action="LOCK_CAPITAL",
            secondary_actions=["REMOVE_TRADE_PERMISSION_TEMPORARY"],
            system_state=SystemState.FROZEN,
            cooldown_seconds=7200,  # 2 hours
            requires_manual_review=True,
            audit_required=True,
        ),
        principle="📌 De fleste kontoer dør her.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🧱 CLASS E: INFRASTRUKTUR & EKSTERN RISIKO
    # ═══════════════════════════════════════════════════════════════════════════
    
    FailureScenario.E1_EXCHANGE_RISK: ScenarioDefinition(
        scenario=FailureScenario.E1_EXCHANGE_RISK,
        failure_class=FailureClass.E,
        name_no="Exchange-risiko",
        name_en="Exchange Risk",
        description="Problemer med børsen eller API",
        triggers=[
            TriggerCondition("api_error", "API-feil"),
            TriggerCondition("order_inconsistency", "Ordre-inkonsistens"),
            TriggerCondition("balance_discrepancy", "Balance-avvik", threshold=1.0, comparison=">="),  # $1+
        ],
        response=FailureResponse(
            primary_action="STOP_ALL_TRADING",
            secondary_actions=["SECURE_FUNDS", "ISOLATE_EXCHANGE"],
            system_state=SystemState.FROZEN,
            cooldown_seconds=600,  # 10 minutes
            requires_manual_review=True,
        ),
        principle="📌 Trust no exchange completely.",
    ),
    
    FailureScenario.E2_INFRASTRUCTURE_FAILURE: ScenarioDefinition(
        scenario=FailureScenario.E2_INFRASTRUCTURE_FAILURE,
        failure_class=FailureClass.E,
        name_no="Teknisk infrastrukturfeil",
        name_en="Infrastructure Failure",
        description="Systemisk teknisk feil",
        triggers=[
            TriggerCondition("server_unstable", "Server ustabil"),
            TriggerCondition("clock_drift", "Klokke-drift", threshold=1000, comparison=">"),  # 1s
            TriggerCondition("resource_overuse", "Ressurs-overforbruk", threshold=90.0, comparison=">="),  # 90%+ CPU/mem
        ],
        response=FailureResponse(
            primary_action="FAIL_CLOSED",
            secondary_actions=["NO_RESTART_WITHOUT_VERIFICATION"],
            system_state=SystemState.FROZEN,
            requires_manual_review=True,
        ),
        principle="📌 Fail-closed is the only safe default.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE SCENARIO MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class FailureScenarioMonitor:
    """
    Central monitor for all failure scenarios.
    
    🧠 METAREGEL: Systemet stopper alltid for tidlig – aldri for sent.
    
    Hvis du er i tvil om et scenario:
    → det er allerede for sent
    → systemet burde stoppet
    """
    
    def __init__(self):
        self._current_state = SystemState.NORMAL
        self._active_failures: Dict[FailureScenario, FailureEvent] = {}
        self._failure_history: List[FailureEvent] = []
        self._cooldowns: Dict[FailureScenario, datetime] = {}
        self._state_change_callbacks: List[Callable[[SystemState, SystemState, FailureScenario], None]] = []
        
        logger.info("🛑 FailureScenarioMonitor initialized - 14 scenarios active")
    
    @property
    def current_state(self) -> SystemState:
        """Current system operating state."""
        return self._current_state
    
    @property
    def is_normal(self) -> bool:
        """Is system in normal operating mode?"""
        return self._current_state == SystemState.NORMAL
    
    @property
    def can_open_positions(self) -> bool:
        """Can system open new positions?"""
        return self._current_state in [SystemState.NORMAL]
    
    @property
    def can_close_positions(self) -> bool:
        """Can system close positions?"""
        return self._current_state in [
            SystemState.NORMAL, 
            SystemState.SAFE_MODE, 
            SystemState.REDUCE_ONLY
        ]
    
    @property
    def active_failures(self) -> List[FailureScenario]:
        """List of currently active failure scenarios."""
        return list(self._active_failures.keys())
    
    def register_state_callback(
        self, 
        callback: Callable[[SystemState, SystemState, FailureScenario], None]
    ):
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    def trigger_failure(
        self,
        scenario: FailureScenario,
        trigger_data: Dict[str, Any],
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Trigger a failure scenario.
        
        Args:
            scenario: The failure scenario to trigger
            trigger_data: Data about what triggered the scenario
            force: Force trigger even if in cooldown
            
        Returns:
            (success, message)
        """
        defn = FAILURE_SCENARIOS.get(scenario)
        if not defn:
            return False, f"Unknown scenario: {scenario}"
        
        # Check cooldown
        if not force and scenario in self._cooldowns:
            if datetime.utcnow() < self._cooldowns[scenario]:
                remaining = (self._cooldowns[scenario] - datetime.utcnow()).seconds
                return False, f"Scenario {scenario} in cooldown for {remaining}s"
        
        # Create failure event
        old_state = self._current_state
        new_state = defn.response.system_state
        
        event = FailureEvent(
            scenario=scenario,
            failure_class=defn.failure_class,
            timestamp=datetime.utcnow(),
            trigger_data=trigger_data,
            response_taken=defn.response.primary_action,
            system_state_before=old_state,
            system_state_after=new_state,
        )
        
        # Store event
        self._active_failures[scenario] = event
        self._failure_history.append(event)
        
        # Update system state (take most restrictive)
        self._update_system_state()
        
        # Set cooldown
        if defn.response.cooldown_seconds > 0:
            self._cooldowns[scenario] = datetime.utcnow() + timedelta(
                seconds=defn.response.cooldown_seconds
            )
        
        # Log the failure
        logger.critical(
            f"🛑 FAILURE TRIGGERED: {scenario.value}\n"
            f"   Class: {defn.failure_class.value} ({defn.failure_class.name})\n"
            f"   State: {old_state.value} → {new_state.value}\n"
            f"   Action: {defn.response.primary_action}\n"
            f"   Principle: {defn.principle}\n"
            f"   Trigger: {trigger_data}"
        )
        
        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, self._current_state, scenario)
            except Exception as e:
                logger.error(f"State callback error: {e}")
        
        return True, f"Failure {scenario.value} triggered - {defn.response.primary_action}"
    
    def resolve_failure(
        self,
        scenario: FailureScenario,
        resolution_note: str = "",
    ) -> Tuple[bool, str]:
        """
        Resolve/clear a failure scenario.
        
        Args:
            scenario: The scenario to resolve
            resolution_note: Note about how it was resolved
            
        Returns:
            (success, message)
        """
        if scenario not in self._active_failures:
            return False, f"Scenario {scenario} is not active"
        
        defn = FAILURE_SCENARIOS.get(scenario)
        if defn and defn.response.requires_manual_review:
            logger.warning(f"⚠️ Scenario {scenario} requires manual review before resolution")
        
        # Mark as resolved
        event = self._active_failures[scenario]
        event.resolved = True
        event.resolved_at = datetime.utcnow()
        event.resolution_note = resolution_note
        
        # Remove from active
        del self._active_failures[scenario]
        
        # Update system state
        self._update_system_state()
        
        logger.info(f"✅ Failure resolved: {scenario.value} - {resolution_note}")
        
        return True, f"Failure {scenario.value} resolved"
    
    def _update_system_state(self):
        """Update system state based on active failures."""
        if not self._active_failures:
            self._current_state = SystemState.NORMAL
            return
        
        # State priority (most restrictive wins)
        state_priority = {
            SystemState.FROZEN: 100,
            SystemState.OBSERVER_ONLY: 80,
            SystemState.PAUSED: 60,
            SystemState.REDUCE_ONLY: 40,
            SystemState.SAFE_MODE: 20,
            SystemState.NORMAL: 0,
        }
        
        most_restrictive = SystemState.NORMAL
        for event in self._active_failures.values():
            if state_priority.get(event.system_state_after, 0) > state_priority.get(most_restrictive, 0):
                most_restrictive = event.system_state_after
        
        self._current_state = most_restrictive
    
    def check_scenario(
        self,
        scenario: FailureScenario,
        context: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a specific scenario's triggers are met.
        
        Args:
            scenario: Scenario to check
            context: Current system context data
            
        Returns:
            (triggered, trigger_reason)
        """
        defn = FAILURE_SCENARIOS.get(scenario)
        if not defn:
            return False, None
        
        for trigger in defn.triggers:
            if trigger.name in context:
                value = context[trigger.name]
                
                # Check threshold conditions
                if trigger.threshold is not None:
                    triggered = False
                    if trigger.comparison == ">" and value > trigger.threshold:
                        triggered = True
                    elif trigger.comparison == "<" and value < trigger.threshold:
                        triggered = True
                    elif trigger.comparison == ">=" and value >= trigger.threshold:
                        triggered = True
                    elif trigger.comparison == "<=" and value <= trigger.threshold:
                        triggered = True
                    elif trigger.comparison == "==" and value == trigger.threshold:
                        triggered = True
                    elif trigger.comparison == "!=" and value != trigger.threshold:
                        triggered = True
                    
                    if triggered:
                        return True, f"{trigger.name}: {value} {trigger.comparison} {trigger.threshold}"
                
                # Boolean trigger (just presence of True)
                elif value is True:
                    return True, trigger.name
        
        return False, None
    
    def check_all_scenarios(self, context: Dict[str, Any]) -> List[FailureScenario]:
        """
        Check all scenarios against current context.
        
        Args:
            context: Current system context data
            
        Returns:
            List of triggered scenarios
        """
        triggered = []
        
        for scenario in FailureScenario:
            is_triggered, reason = self.check_scenario(scenario, context)
            if is_triggered:
                success, _ = self.trigger_failure(scenario, {"reason": reason, **context})
                if success:
                    triggered.append(scenario)
        
        return triggered
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete monitor status."""
        return {
            "current_state": self._current_state.value,
            "can_open_positions": self.can_open_positions,
            "can_close_positions": self.can_close_positions,
            "active_failures": [
                {
                    "scenario": s.value,
                    "class": FAILURE_SCENARIOS[s].failure_class.value,
                    "triggered_at": e.timestamp.isoformat(),
                    "state": e.system_state_after.value,
                }
                for s, e in self._active_failures.items()
            ],
            "cooldowns": {
                s.value: cd.isoformat()
                for s, cd in self._cooldowns.items()
                if cd > datetime.utcnow()
            },
            "total_failures_today": len([
                e for e in self._failure_history
                if e.timestamp.date() == datetime.utcnow().date()
            ]),
        }
    
    def get_scenario_info(self, scenario: FailureScenario) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific scenario."""
        defn = FAILURE_SCENARIOS.get(scenario)
        if not defn:
            return None
        
        return {
            "scenario": scenario.value,
            "class": defn.failure_class.value,
            "name_no": defn.name_no,
            "name_en": defn.name_en,
            "description": defn.description,
            "triggers": [t.name for t in defn.triggers],
            "response": {
                "primary": defn.response.primary_action,
                "secondary": defn.response.secondary_actions,
                "state": defn.response.system_state.value,
                "cooldown_seconds": defn.response.cooldown_seconds,
                "requires_manual_review": defn.response.requires_manual_review,
            },
            "principle": defn.principle,
            "is_active": scenario in self._active_failures,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_failure_monitor: Optional[FailureScenarioMonitor] = None

def get_failure_monitor() -> FailureScenarioMonitor:
    """Get singleton failure monitor instance."""
    global _failure_monitor
    if _failure_monitor is None:
        _failure_monitor = FailureScenarioMonitor()
    return _failure_monitor


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def can_trade() -> bool:
    """Quick check if trading is allowed."""
    return get_failure_monitor().can_open_positions


def can_exit() -> bool:
    """Quick check if exits are allowed."""
    return get_failure_monitor().can_close_positions


def get_system_state() -> SystemState:
    """Get current system state."""
    return get_failure_monitor().current_state


def trigger_kill_switch(reason: str) -> None:
    """Immediate system stop (Class A)."""
    monitor = get_failure_monitor()
    monitor.trigger_failure(
        FailureScenario.A2_RISK_RULE_VIOLATION,
        {"reason": reason, "source": "KILL_SWITCH"},
        force=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STOPP-KART (SUMMARY)
# ═══════════════════════════════════════════════════════════════════════════════

STOPP_KART = """
🗺️ OPPSUMMERT STOPP-KART

RISIKO BRUDD ───────────► FULL STOPP (Class A)
DATA FEIL ──────────────► SAFE MODE (Class A)
REGIME USIKKER ─────────► PAUSE (Class B)
EDGE FORFALL ───────────► STRATEGI FRYS (Class B)
EXIT STRESS ────────────► TVUNGEN EXIT (Class C)
MENNESKE INNBLANDING ───► LOCKDOWN (Class D)
INFRA FEIL ─────────────► FROZEN (Class E)

🧠 METAREGEL: Systemet stopper alltid for tidlig – aldri for sent.
"""


if __name__ == "__main__":
    # Demo
    print("🛑 HEDGE FUND FAILURE SCENARIOS")
    print("=" * 60)
    
    monitor = get_failure_monitor()
    
    print(f"\nCurrent state: {monitor.current_state.value}")
    print(f"Can open positions: {monitor.can_open_positions}")
    print(f"Can close positions: {monitor.can_close_positions}")
    
    # Simulate a capital breach
    print("\n--- Simulating A1_CAPITAL_BREACH ---")
    success, msg = monitor.trigger_failure(
        FailureScenario.A1_CAPITAL_BREACH,
        {"daily_pnl_pct": -6.5, "reason": "Daily loss limit exceeded"}
    )
    print(f"Result: {msg}")
    print(f"New state: {monitor.current_state.value}")
    print(f"Can open: {monitor.can_open_positions}, Can close: {monitor.can_close_positions}")
    
    # Show status
    print("\n--- Status ---")
    status = monitor.get_status()
    print(f"Active failures: {len(status['active_failures'])}")
    for f in status['active_failures']:
        print(f"  - {f['scenario']} ({f['class']})")
    
    print(STOPP_KART)
