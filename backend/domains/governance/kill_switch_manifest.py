"""
🛑 KILL-SWITCH MANIFEST
========================
Hva som stopper systemet – uten diskusjon.

Formål:
    Kill-switch eksisterer for å beskytte kapital, systemintegritet og beslutningskvalitet.
    Systemet skal alltid stoppe for tidlig, aldri for sent.

Absolutte prinsipper:
    1. Fail-closed er standard
    2. Ingen override uten cooldown + audit
    3. Exit > forklaring
    4. Cash er en posisjon

🧠 META-REGEL: Hvis du føler trang til å starte raskt igjen – er du ikke klar.
              Markedet forsvinner ikke. Kapital kan.
"""

import logging
from enum import Enum, IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DANGER RANKING: Farligste failures for DEG
# Rangert etter sannsynlighet × skadepotensial i crypto futures
# ═══════════════════════════════════════════════════════════════════════════════

class DangerRank(IntEnum):
    """
    Rangering av failures etter farlighet.
    Lavere nummer = farligere.
    """
    HUMAN_OVERRIDE = 1      # 🥇 Ødelegger alle andre sikkerhetslag
    RISK_BREACH = 2         # 🥈 Direkte vei til irreversibel drawdown
    DATA_INCONSISTENCY = 3  # 🥉 Systemet handler på løgn
    REGIME_MISREAD = 4      # Drepende i crypto (trend → chop)
    EDGE_DECAY = 5          # Langsom konto-død
    EXECUTION_ERROR = 6     # Skjult kostnad + stress
    EXCHANGE_ANOMALY = 7    # Sjeldnere, men brutal


DANGER_RANKING = {
    DangerRank.HUMAN_OVERRIDE: {
        "name": "Menneskelig override / ego",
        "why_dangerous": "Ødelegger alle andre sikkerhetslag",
        "component": "HumanOverrideLock",
        "emoji": "🥇",
    },
    DangerRank.RISK_BREACH: {
        "name": "Risiko-brudd (size / leverage)",
        "why_dangerous": "Direkte vei til irreversibel drawdown",
        "component": "RiskKernel",
        "emoji": "🥈",
    },
    DangerRank.DATA_INCONSISTENCY: {
        "name": "Data-inkonsistens / feed-feil",
        "why_dangerous": "Systemet handler på løgn",
        "component": "DataIntegritySentinel",
        "emoji": "🥉",
    },
    DangerRank.REGIME_MISREAD: {
        "name": "Regime-misread",
        "why_dangerous": "Drepende i crypto (trend → chop)",
        "component": "MarketRegimeDetector",
        "emoji": "4️⃣",
    },
    DangerRank.EDGE_DECAY: {
        "name": "Edge-forfall",
        "why_dangerous": "Langsom konto-død",
        "component": "PerformanceDriftMonitor",
        "emoji": "5️⃣",
    },
    DangerRank.EXECUTION_ERROR: {
        "name": "Execution-feil",
        "why_dangerous": "Skjult kostnad + stress",
        "component": "ExecutionOptimizer",
        "emoji": "6️⃣",
    },
    DangerRank.EXCHANGE_ANOMALY: {
        "name": "Exchange-anomali",
        "why_dangerous": "Sjeldnere, men brutal",
        "component": "ExchangeHealthMonitor",
        "emoji": "7️⃣",
    },
}

# Konklusjon: Du må beskyttes mest mot DEG SELV – dernest risiko og data.


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE → COMPONENT MAPPING
# Dette er hvem som håndhever stoppen
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FailureHandler:
    """Definition of a failure type and its handler component."""
    name: str
    component: str
    severity: str  # 🔴 🟠 🟡 🔵
    triggers: List[str]
    actions: List[str]
    danger_rank: DangerRank


FAILURE_HANDLERS: Dict[str, FailureHandler] = {
    
    # 🔴 CRITICAL FAILURES
    
    "HUMAN_OVERRIDE": FailureHandler(
        name="Menneskelig override / emosjonell eskalering",
        component="HumanOverrideLock",
        severity="🔴",
        triggers=[
            "Manuell parameterendring",
            "Size-økning etter tap",
            "Forsøk på å flytte stop",
        ],
        actions=[
            "Full system-freeze",
            "24–72t cooldown",
            "Full audit-logg",
        ],
        danger_rank=DangerRank.HUMAN_OVERRIDE,
    ),
    
    "RISK_BREACH": FailureHandler(
        name="Risiko-brudd",
        component="RiskKernel",
        severity="🔴",
        triggers=[
            "Risiko per trade > policy",
            "Leverage > tillatt",
            "Daglig tap nådd",
        ],
        actions=[
            "Kill-switch",
            "Kun exits",
            "Manuell review",
        ],
        danger_rank=DangerRank.RISK_BREACH,
    ),
    
    "DATA_INCONSISTENCY": FailureHandler(
        name="Data-inkonsistens",
        component="DataIntegritySentinel",
        severity="🔴",
        triggers=[
            "Manglende ticks",
            "Pris-sprang uten volum",
            "Clock-drift / latency",
        ],
        actions=[
            "SAFE MODE",
            "Entry blokkert",
            "Data re-synk",
        ],
        danger_rank=DangerRank.DATA_INCONSISTENCY,
    ),
    
    # 🟠 SERIOUS FAILURES
    
    "REGIME_UNCERTAINTY": FailureHandler(
        name="Regime-usikkerhet",
        component="MarketRegimeDetector",
        severity="🟠",
        triggers=[
            "Motstridende regimesignaler",
            "Confidence < terskel",
        ],
        actions=[
            "Pause trading",
            "Observer-only",
            "Ingen nye entries",
        ],
        danger_rank=DangerRank.REGIME_MISREAD,
    ),
    
    "EDGE_DECAY": FailureHandler(
        name="Edge-forfall",
        component="PerformanceDriftMonitor",
        severity="🟠",
        triggers=[
            "Negativ expectancy over N trades",
            "Avvik fra historisk edge",
        ],
        actions=[
            "Strategi fryses",
            "Shadow-testing",
            "Re-validering",
        ],
        danger_rank=DangerRank.EDGE_DECAY,
    ),
    
    # 🟡 MODERATE FAILURES
    
    "EXECUTION_ERROR": FailureHandler(
        name="Execution-feil",
        component="ExecutionOptimizer",
        severity="🟡",
        triggers=[
            "Slippage > toleranse",
            "Fill-avvik",
            "Rejected orders",
        ],
        actions=[
            "Trade-rate ned",
            "Order-type bytte",
            "Midlertidig pause",
        ],
        danger_rank=DangerRank.EXECUTION_ERROR,
    ),
    
    # 🔵 EXTERNAL FAILURES
    
    "EXCHANGE_ANOMALY": FailureHandler(
        name="Exchange-anomali",
        component="ExchangeHealthMonitor",
        severity="🔵",
        triggers=[
            "API-feil",
            "Balance mismatch",
            "Funding-sjokk",
        ],
        actions=[
            "Isoler exchange",
            "Flat-mandat",
            "Midler sikres",
        ],
        danger_rank=DangerRank.EXCHANGE_ANOMALY,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# RESTART PROTOCOL
# Hvordan komme trygt tilbake – uten å gjøre det verre
# Dette er obligatorisk rekkefølge. Ingen hopp.
# ═══════════════════════════════════════════════════════════════════════════════

class RestartPhase(IntEnum):
    """Restart protocol phases - must be followed IN ORDER."""
    COOLDOWN = 1        # 🧭 Stillstand
    DIAGNOSIS = 2       # 🧪 Diagnose
    AUDIT_FIX = 3       # 🧾 Audit & Fix
    SHADOW_MODE = 4     # 👻 Shadow Mode
    GRADUAL_LIVE = 5    # 🟢 Gradvis Live
    FULL_LIVE = 6       # 🟦 Full Live


@dataclass
class PhaseRequirements:
    """Requirements and criteria for each restart phase."""
    phase: RestartPhase
    name: str
    emoji: str
    duration_hours: Tuple[int, int]  # (min, max)
    description: str
    requirements: List[str]
    exit_criteria: List[str]
    can_skip: bool = False


RESTART_PHASES: Dict[RestartPhase, PhaseRequirements] = {
    
    RestartPhase.COOLDOWN: PhaseRequirements(
        phase=RestartPhase.COOLDOWN,
        name="STILLSTAND (Cooling-Off)",
        emoji="🧭",
        duration_hours=(12, 48),
        description="Emosjonell og systemisk stabilisering",
        requirements=[
            "Ingen beslutninger",
            "Kun observasjon",
            "Ingen kodeendringer",
        ],
        exit_criteria=[
            "Minimum tid oppfylt",
            "Emosjonelt stabil",
            "Klar for objektiv analyse",
        ],
    ),
    
    RestartPhase.DIAGNOSIS: PhaseRequirements(
        phase=RestartPhase.DIAGNOSIS,
        name="DIAGNOSE",
        emoji="🧪",
        duration_hours=(2, 8),
        description="Svar på kun disse tre spørsmål - ingen optimalisering",
        requirements=[
            "Hva trigget stoppen?",
            "Var det forventet i policy?",
            "Kunne høyere lag stoppet tidligere?",
        ],
        exit_criteria=[
            "Alle tre spørsmål besvart",
            "Root cause identifisert",
            "Ingen 'kanskje' eller 'muligens'",
        ],
    ),
    
    RestartPhase.AUDIT_FIX: PhaseRequirements(
        phase=RestartPhase.AUDIT_FIX,
        name="AUDIT & FIX",
        emoji="🧾",
        duration_hours=(4, 24),
        description="Verifiser og fiks - ingen 'små justeringer'",
        requirements=[
            "Verifiser logger",
            "Bekreft data-integritet",
            "Bekreft risikogrenser",
            "Ingen 'små justeringer'",
        ],
        exit_criteria=[
            "Logger verifisert",
            "Data integritet OK",
            "Risikogrenser bekreftet",
            "Fix implementert (hvis nødvendig)",
            "Hvis noe er uklart → bli i pause",
        ],
    ),
    
    RestartPhase.SHADOW_MODE: PhaseRequirements(
        phase=RestartPhase.SHADOW_MODE,
        name="SHADOW MODE",
        emoji="👻",
        duration_hours=(24, 168),  # 1-7 days
        description="Samme signaler, ingen ekte kapital",
        requirements=[
            "Samme signaler som live",
            "Ingen ekte kapital",
            "Sammenlign forventet vs faktisk",
        ],
        exit_criteria=[
            "Stabil oppførsel",
            "Ingen policy-brudd",
            "Edge reproduseres",
            "Minimum 24 timer uten avvik",
        ],
    ),
    
    RestartPhase.GRADUAL_LIVE: PhaseRequirements(
        phase=RestartPhase.GRADUAL_LIVE,
        name="GRADVIS LIVE",
        emoji="🟢",
        duration_hours=(48, 168),  # 2-7 days
        description="Systemet må bevise disiplin før tempo",
        requirements=[
            "Redusert size (25-50%)",
            "Redusert leverage",
            "Kun TOP-N edges",
            "Exit-fokus",
        ],
        exit_criteria=[
            "Ingen brudd i perioden",
            "Positiv eller flat PnL",
            "Alle exits fungerer",
            "System oppfører seg som forventet",
        ],
    ),
    
    RestartPhase.FULL_LIVE: PhaseRequirements(
        phase=RestartPhase.FULL_LIVE,
        name="FULL LIVE",
        emoji="🟦",
        duration_hours=(0, 0),  # Indefinite
        description="Kun hvis alle kriterier er oppfylt",
        requirements=[
            "Ingen brudd i shadow",
            "Ingen manuell override",
            "Risk-kernel aldri trigget",
        ],
        exit_criteria=[
            "N/A - dette er målet",
        ],
        can_skip=False,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# RESTART PROTOCOL MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RestartState:
    """Current state of the restart protocol."""
    current_phase: RestartPhase
    phase_started: datetime
    kill_switch_triggered: datetime
    trigger_reason: str
    trigger_failure: str
    diagnosis_answers: Dict[str, str] = field(default_factory=dict)
    audit_checklist: Dict[str, bool] = field(default_factory=dict)
    shadow_results: Dict[str, Any] = field(default_factory=dict)
    gradual_metrics: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class RestartProtocolManager:
    """
    Manages the 6-phase restart protocol after a kill-switch event.
    
    🧠 META-REGEL: Hvis du føler trang til å starte raskt igjen – er du ikke klar.
    """
    
    def __init__(self):
        self._active_restart: Optional[RestartState] = None
        self._restart_history: List[RestartState] = []
        
        logger.info("🔄 RestartProtocolManager initialized")
    
    @property
    def is_in_restart(self) -> bool:
        """Is there an active restart protocol?"""
        return self._active_restart is not None
    
    @property
    def current_phase(self) -> Optional[RestartPhase]:
        """Current restart phase, if any."""
        return self._active_restart.current_phase if self._active_restart else None
    
    @property
    def can_trade(self) -> bool:
        """Can the system trade (only in GRADUAL_LIVE or FULL_LIVE)."""
        if not self._active_restart:
            return True  # No restart in progress
        return self._active_restart.current_phase >= RestartPhase.GRADUAL_LIVE
    
    def initiate_restart(
        self,
        trigger_reason: str,
        trigger_failure: str,
    ) -> Tuple[bool, str]:
        """
        Initiate restart protocol after kill-switch.
        
        Args:
            trigger_reason: Why the kill-switch was triggered
            trigger_failure: Which failure type caused it
            
        Returns:
            (success, message)
        """
        if self._active_restart:
            return False, "Restart protocol already in progress"
        
        now = datetime.utcnow()
        
        self._active_restart = RestartState(
            current_phase=RestartPhase.COOLDOWN,
            phase_started=now,
            kill_switch_triggered=now,
            trigger_reason=trigger_reason,
            trigger_failure=trigger_failure,
        )
        
        phase_info = RESTART_PHASES[RestartPhase.COOLDOWN]
        
        logger.critical(
            f"🔄 RESTART PROTOCOL INITIATED\n"
            f"   Trigger: {trigger_failure} - {trigger_reason}\n"
            f"   Phase 1: {phase_info.emoji} {phase_info.name}\n"
            f"   Min Duration: {phase_info.duration_hours[0]} hours\n"
            f"   Requirements: {phase_info.requirements}"
        )
        
        return True, f"Restart protocol initiated - Phase 1: {phase_info.name}"
    
    def advance_phase(
        self,
        checklist_passed: bool = False,
        notes: str = "",
    ) -> Tuple[bool, str]:
        """
        Attempt to advance to the next restart phase.
        
        Args:
            checklist_passed: Have all exit criteria been verified?
            notes: Any notes about the phase completion
            
        Returns:
            (success, message)
        """
        if not self._active_restart:
            return False, "No active restart protocol"
        
        current = self._active_restart.current_phase
        current_info = RESTART_PHASES[current]
        
        # Check minimum duration
        elapsed = datetime.utcnow() - self._active_restart.phase_started
        min_hours = current_info.duration_hours[0]
        
        if elapsed < timedelta(hours=min_hours):
            remaining = timedelta(hours=min_hours) - elapsed
            return False, f"Minimum duration not met. {remaining} remaining."
        
        # Check if checklist passed
        if not checklist_passed:
            return False, f"Exit criteria not verified for {current_info.name}"
        
        # Advance to next phase
        if current == RestartPhase.FULL_LIVE:
            return False, "Already at final phase"
        
        next_phase = RestartPhase(current.value + 1)
        next_info = RESTART_PHASES[next_phase]
        
        # Record notes
        if notes:
            self._active_restart.notes.append(
                f"[{current_info.name}] {notes}"
            )
        
        # Update state
        self._active_restart.current_phase = next_phase
        self._active_restart.phase_started = datetime.utcnow()
        
        logger.info(
            f"🔄 RESTART PHASE ADVANCED\n"
            f"   From: {current_info.emoji} {current_info.name}\n"
            f"   To: {next_info.emoji} {next_info.name}\n"
            f"   Requirements: {next_info.requirements}"
        )
        
        # If reaching full live, complete the restart
        if next_phase == RestartPhase.FULL_LIVE:
            logger.info("🟦 RESTART COMPLETE - System ready for full live trading")
        
        return True, f"Advanced to Phase {next_phase.value}: {next_info.name}"
    
    def record_diagnosis(
        self,
        what_triggered: str,
        was_expected: str,
        could_prevent: str,
    ) -> None:
        """Record diagnosis answers (Phase 2)."""
        if not self._active_restart:
            return
        
        self._active_restart.diagnosis_answers = {
            "what_triggered": what_triggered,
            "was_expected_in_policy": was_expected,
            "could_higher_layer_prevent": could_prevent,
        }
        
        logger.info(
            f"🧪 DIAGNOSIS RECORDED\n"
            f"   1. What triggered: {what_triggered}\n"
            f"   2. Expected in policy: {was_expected}\n"
            f"   3. Could prevent earlier: {could_prevent}"
        )
    
    def record_audit_item(self, item: str, passed: bool) -> None:
        """Record audit checklist item (Phase 3)."""
        if not self._active_restart:
            return
        
        self._active_restart.audit_checklist[item] = passed
        status = "✅" if passed else "❌"
        logger.info(f"🧾 AUDIT: {status} {item}")
    
    def record_shadow_result(
        self,
        metric: str,
        expected: Any,
        actual: Any,
        passed: bool,
    ) -> None:
        """Record shadow mode result (Phase 4)."""
        if not self._active_restart:
            return
        
        self._active_restart.shadow_results[metric] = {
            "expected": expected,
            "actual": actual,
            "passed": passed,
        }
        
        status = "✅" if passed else "❌"
        logger.info(f"👻 SHADOW: {status} {metric}: expected={expected}, actual={actual}")
    
    def abort_restart(self, reason: str) -> None:
        """Abort restart and return to kill-switch state."""
        if not self._active_restart:
            return
        
        logger.warning(
            f"🛑 RESTART ABORTED\n"
            f"   Phase: {self._active_restart.current_phase.name}\n"
            f"   Reason: {reason}\n"
            f"   → Returning to kill-switch state"
        )
        
        # Archive the failed restart
        self._active_restart.notes.append(f"ABORTED: {reason}")
        self._restart_history.append(self._active_restart)
        self._active_restart = None
    
    def complete_restart(self) -> Tuple[bool, str]:
        """Complete the restart protocol (if at final phase)."""
        if not self._active_restart:
            return False, "No active restart"
        
        if self._active_restart.current_phase != RestartPhase.FULL_LIVE:
            return False, f"Not at final phase (current: {self._active_restart.current_phase.name})"
        
        # Archive successful restart
        self._restart_history.append(self._active_restart)
        self._active_restart = None
        
        logger.info("🎉 RESTART PROTOCOL COMPLETED - System fully operational")
        
        return True, "Restart complete - full live trading enabled"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current restart protocol status."""
        if not self._active_restart:
            return {
                "in_restart": False,
                "can_trade": True,
            }
        
        rs = self._active_restart
        phase_info = RESTART_PHASES[rs.current_phase]
        elapsed = datetime.utcnow() - rs.phase_started
        min_remaining = max(
            timedelta(0),
            timedelta(hours=phase_info.duration_hours[0]) - elapsed
        )
        
        return {
            "in_restart": True,
            "can_trade": self.can_trade,
            "current_phase": rs.current_phase.value,
            "phase_name": phase_info.name,
            "phase_emoji": phase_info.emoji,
            "phase_started": rs.phase_started.isoformat(),
            "elapsed_hours": elapsed.total_seconds() / 3600,
            "min_remaining_hours": min_remaining.total_seconds() / 3600,
            "trigger_reason": rs.trigger_reason,
            "trigger_failure": rs.trigger_failure,
            "diagnosis_complete": bool(rs.diagnosis_answers),
            "audit_items": len(rs.audit_checklist),
            "shadow_results": len(rs.shadow_results),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_restart_manager: Optional[RestartProtocolManager] = None

def get_restart_manager() -> RestartProtocolManager:
    """Get singleton restart protocol manager."""
    global _restart_manager
    if _restart_manager is None:
        _restart_manager = RestartProtocolManager()
    return _restart_manager


# ═══════════════════════════════════════════════════════════════════════════════
# KILL-SWITCH MANIFEST (SUMMARY)
# ═══════════════════════════════════════════════════════════════════════════════

KILL_SWITCH_MANIFEST = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🛑 KILL-SWITCH MANIFEST                               ║
║                   Hva som stopper systemet – uten diskusjon                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FORMÅL                                                                      ║
║  Kill-switch eksisterer for å beskytte kapital, systemintegritet            ║
║  og beslutningskvalitet.                                                     ║
║                                                                              ║
║  ABSOLUTTE PRINSIPPER                                                        ║
║  1. Fail-closed er standard                                                  ║
║  2. Ingen override uten cooldown + audit                                     ║
║  3. Exit > forklaring                                                        ║
║  4. Cash er en posisjon                                                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  HVA SKJER NÅR KILL-SWITCH TRIGGES                                          ║
║  • All ny trading stoppes umiddelbart                                        ║
║  • Kun reduce-only / exits tillatt                                           ║
║  • Hendelsen logges som Critical Incident                                    ║
║  • Restart krever eksplisitt protokoll (6 faser)                            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🏆 DANGER RANKING (sannsynlighet × skadepotensial)                         ║
║                                                                              ║
║  🥇 Menneskelig override / ego    → Ødelegger alle sikkerhetslag            ║
║  🥈 Risiko-brudd (size/leverage)  → Irreversibel drawdown                   ║
║  🥉 Data-inkonsistens             → Systemet handler på løgn                ║
║  4️⃣  Regime-misread               → Drepende i crypto                       ║
║  5️⃣  Edge-forfall                 → Langsom konto-død                       ║
║  6️⃣  Execution-feil               → Skjult kostnad                          ║
║  7️⃣  Exchange-anomali             → Sjelden, men brutal                     ║
║                                                                              ║
║  👉 Du må beskyttes mest mot DEG SELV – dernest risiko og data.             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔄 RESTART PROTOKOLL (obligatorisk rekkefølge)                             ║
║                                                                              ║
║  🧭 FASE 1: STILLSTAND (12-48t)   – Ingen beslutninger, kun observasjon     ║
║  🧪 FASE 2: DIAGNOSE              – Tre spørsmål, ingen optimalisering      ║
║  🧾 FASE 3: AUDIT & FIX           – Verifiser, ingen 'små justeringer'      ║
║  👻 FASE 4: SHADOW MODE           – Samme signaler, ingen ekte kapital      ║
║  🟢 FASE 5: GRADVIS LIVE          – Redusert size/leverage, exit-fokus      ║
║  🟦 FASE 6: FULL LIVE             – Kun hvis alle kriterier oppfylt         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🧠 META-REGEL (VIKTIGST)                                                   ║
║                                                                              ║
║     Hvis du føler trang til å starte raskt igjen – er du ikke klar.         ║
║                                                                              ║
║     Markedet forsvinner ikke.                                                ║
║     Kapital kan.                                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_kill_switch_manifest():
    """Print the kill-switch manifest."""
    print(KILL_SWITCH_MANIFEST)


def print_danger_ranking():
    """Print the danger ranking."""
    print("\n🏆 DANGER RANKING - Farligste failures for DEG\n")
    print("Rangert etter sannsynlighet × skadepotensial i crypto futures\n")
    print("-" * 60)
    
    for rank, info in DANGER_RANKING.items():
        print(f"{info['emoji']} {info['name']}")
        print(f"   → {info['why_dangerous']}")
        print(f"   → Komponent: {info['component']}")
        print()
    
    print("-" * 60)
    print("👉 Konklusjon: Du må beskyttes mest mot DEG SELV")


def print_restart_phases():
    """Print all restart phases."""
    print("\n🔄 RESTART PROTOKOLL\n")
    print("Hvordan komme trygt tilbake – uten å gjøre det verre")
    print("Dette er obligatorisk rekkefølge. Ingen hopp.\n")
    print("-" * 60)
    
    for phase, info in RESTART_PHASES.items():
        duration = f"{info.duration_hours[0]}-{info.duration_hours[1]}t" if info.duration_hours[1] > 0 else "∞"
        print(f"\n{info.emoji} FASE {phase.value}: {info.name} ({duration})")
        print(f"   {info.description}")
        print("   Krav:")
        for req in info.requirements:
            print(f"     • {req}")
    
    print("\n" + "-" * 60)
    print("🧠 META-REGEL: Hvis du føler trang til å starte raskt igjen – er du ikke klar.")


if __name__ == "__main__":
    print_kill_switch_manifest()
    print_danger_ranking()
    print_restart_phases()
