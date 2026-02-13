"""
🌪️ KRISE SIMULERING — START → STOPP → RESTART
===============================================
Full simulering av én konkret krise – fra trigger → stopp → restart

Scenario: Flash Crash + Data-inkonsistens (vanlig i crypto)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import logging
import asyncio
import time

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CRISIS PHASES (TIDSLINJE)
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisSimPhase(str, Enum):
    """Crisis simulation phases."""
    T0_NORMAL = "T0_NORMAL"           # Normal drift
    T30S_TRIGGER = "T30S_TRIGGER"     # Trigger detected
    T35S_KILLSWITCH = "T35S_KILLSWITCH"  # Kill-switch activated
    T2M_SECURE = "T2M_SECURE"         # Securing positions
    T15M_COOLING = "T15M_COOLING"     # Cooling-off period
    T60M_DIAGNOSE = "T60M_DIAGNOSE"   # Diagnosis phase
    T90M_SHADOW = "T90M_SHADOW"       # Shadow mode
    T3H_GRADUAL = "T3H_GRADUAL"       # Gradual restart
    FULL_LIVE = "FULL_LIVE"           # Full live (if conditions met)


@dataclass
class PhaseConfig:
    """Configuration for a crisis phase."""
    phase: CrisisSimPhase
    name: str
    duration_description: str
    min_duration_seconds: int
    actions: List[str]
    forbidden: List[str]
    success_criteria: List[str]
    can_trade: bool
    can_exit: bool


CRISIS_PHASES: Dict[CrisisSimPhase, PhaseConfig] = {
    
    CrisisSimPhase.T0_NORMAL: PhaseConfig(
        phase=CrisisSimPhase.T0_NORMAL,
        name="🕒 T0 — NORMAL DRIFT",
        duration_description="Normal operations",
        min_duration_seconds=0,
        actions=[
            "System live",
            "Lav/moderat volatilitet",
            "Én åpen posisjon med partials aktiv",
        ],
        forbidden=[],
        success_criteria=["All systems operational"],
        can_trade=True,
        can_exit=True,
    ),
    
    CrisisSimPhase.T30S_TRIGGER: PhaseConfig(
        phase=CrisisSimPhase.T30S_TRIGGER,
        name="🕒 T+30 sek — TRIGGER",
        duration_description="~30 seconds after anomaly",
        min_duration_seconds=5,
        actions=[
            "Spread utvider seg brått",
            "Volum faller",
            "Pris hopper uten støtte",
            "Data Integrity Sentinel flagger inkonsistens",
            "Market Regime Detector mister konsensus",
        ],
        forbidden=[
            "Ignorere signalene",
            "Åpne nye posisjoner",
        ],
        success_criteria=["Anomaly detected and flagged"],
        can_trade=False,
        can_exit=True,
    ),
    
    CrisisSimPhase.T35S_KILLSWITCH: PhaseConfig(
        phase=CrisisSimPhase.T35S_KILLSWITCH,
        name="🛑 T+35 sek — KILL-SWITCH",
        duration_description="5 seconds after trigger",
        min_duration_seconds=5,
        actions=[
            "All ny entry stoppes",
            "Kun reduce-only tillatt",
            "Åpen posisjon: partial exit + tightened trail",
            "Hendelse merket Critical Incident",
            "Ingen diskusjon. Ingen manuell input.",
        ],
        forbidden=[
            "Ny entry",
            "Manuell override",
            "Diskusjon om å fortsette",
        ],
        success_criteria=[
            "Kill-switch activated",
            "Entry blocked",
            "Incident logged",
        ],
        can_trade=False,
        can_exit=True,
    ),
    
    CrisisSimPhase.T2M_SECURE: PhaseConfig(
        phase=CrisisSimPhase.T2M_SECURE,
        name="🧯 T+2 min — SIKRING",
        duration_description="~2 minutes after trigger",
        min_duration_seconds=60,
        actions=[
            "Posisjon fullstendig lukket",
            "System i SAFE MODE",
            "Exchange isolert midlertidig",
        ],
        forbidden=[
            "Prøve å 'vinne tilbake'",
            "Åpne hedge-posisjoner",
            "Manuell trading",
        ],
        success_criteria=[
            "All positions closed",
            "System in SAFE_MODE",
            "Exchange isolated",
        ],
        can_trade=False,
        can_exit=False,  # Already flat
    ),
    
    CrisisSimPhase.T15M_COOLING: PhaseConfig(
        phase=CrisisSimPhase.T15M_COOLING,
        name="❄️ T+15–60 min — COOLING-OFF",
        duration_description="15-60 minutes",
        min_duration_seconds=900,  # 15 min
        actions=[
            "Ingen trading",
            "Kun observasjon",
            "Volatilitet normaliseres",
        ],
        forbidden=[
            "Åpne posisjoner",
            "Gjøre endringer",
            "Ta beslutninger",
        ],
        success_criteria=[
            "Market stabilizing",
            "No trading activity",
            "Operator calm",
        ],
        can_trade=False,
        can_exit=False,
    ),
    
    CrisisSimPhase.T60M_DIAGNOSE: PhaseConfig(
        phase=CrisisSimPhase.T60M_DIAGNOSE,
        name="🔍 T+60–90 min — DIAGNOSE",
        duration_description="60-90 minutes after trigger",
        min_duration_seconds=1800,  # 30 min
        actions=[
            "Svar på tre spørsmål:",
            "  1. Hva trigget? → Data + likviditet",
            "  2. Var det policy-forventet? → Ja/Nei",
            "  3. Kunne høyere lag stoppet før? → Ja/Nei",
            "Ingen endringer gjøres.",
        ],
        forbidden=[
            "Gjøre kodeendringer",
            "Endre parametre",
            "Blame andre",
        ],
        success_criteria=[
            "Root cause identified",
            "Policy review complete",
            "Prevention assessment done",
        ],
        can_trade=False,
        can_exit=False,
    ),
    
    CrisisSimPhase.T90M_SHADOW: PhaseConfig(
        phase=CrisisSimPhase.T90M_SHADOW,
        name="👻 T+90–180 min — SHADOW MODE",
        duration_description="90-180 minutes",
        min_duration_seconds=5400,  # 90 min
        actions=[
            "Samme signaler",
            "Null kapital",
            "Sjekk:",
            "  - Ingen policy-brudd",
            "  - Regime stabil",
            "  - Exit-logikk oppfører seg",
        ],
        forbidden=[
            "Live trades",
            "Real kapital",
            "Hastige vurderinger",
        ],
        success_criteria=[
            "No policy violations in shadow",
            "Regime stable",
            "Exit logic working",
        ],
        can_trade=False,
        can_exit=False,
    ),
    
    CrisisSimPhase.T3H_GRADUAL: PhaseConfig(
        phase=CrisisSimPhase.T3H_GRADUAL,
        name="🟢 T+3–24 t — GRADVIS RESTART",
        duration_description="3-24 hours",
        min_duration_seconds=10800,  # 3 hours
        actions=[
            "25–50% normal size",
            "Kun TOP-N edges",
            "Ingen leverage-eskalering",
            "Exit-dominans",
            "Hvis noe føles 'presset' → tilbake til pause",
        ],
        forbidden=[
            "100% size umiddelbart",
            "Leverage-økning",
            "Ignorere warning signs",
        ],
        success_criteria=[
            "Profitable at reduced size",
            "No new incidents",
            "Risk metrics stable",
        ],
        can_trade=True,  # Limited
        can_exit=True,
    ),
    
    CrisisSimPhase.FULL_LIVE: PhaseConfig(
        phase=CrisisSimPhase.FULL_LIVE,
        name="✅ FULL LIVE",
        duration_description="Only if all conditions met",
        min_duration_seconds=0,
        actions=[
            "Full operations restored",
            "Normal position sizing",
            "All strategies active",
        ],
        forbidden=[
            "Complacency",
            "Forgetting the lesson",
        ],
        success_criteria=[
            "Shadow OK",
            "Ingen overrides",
            "Risk kernel urørt",
            "Data stabil over tid",
        ],
        can_trade=True,
        can_exit=True,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CRISIS STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrisisState:
    """Current state of crisis simulation."""
    phase: CrisisSimPhase
    phase_started_at: datetime
    trigger_reason: str = ""
    positions_at_trigger: int = 0
    pnl_at_trigger: float = 0.0
    positions_closed: int = 0
    incident_logged: bool = False
    diagnosis_complete: bool = False
    shadow_results: Dict[str, Any] = field(default_factory=dict)
    restart_attempts: int = 0


@dataclass
class DiagnosisReport:
    """Diagnosis report (T+60-90 min)."""
    what_triggered: str
    was_policy_expected: bool
    could_prevent_earlier: bool
    root_cause: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ShadowResult:
    """Shadow mode results (T+90-180 min)."""
    duration_minutes: int
    signals_processed: int
    would_have_traded: int
    policy_violations: int
    regime_stable: bool
    exit_logic_ok: bool
    ready_for_restart: bool


# ═══════════════════════════════════════════════════════════════════════════════
# CRISIS SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisSimulator:
    """
    Full crisis simulation from trigger → stop → restart.
    
    Usage:
        sim = CrisisSimulator()
        
        # Start from normal operations
        sim.start_normal()
        
        # Trigger crisis (flash crash + data inconsistency)
        sim.trigger_crisis(
            reason="Flash crash detected",
            spread_ratio=15.0,
            volume_drop=0.9,
        )
        
        # System automatically advances through phases
        await sim.run_simulation()
        
        # Or manually advance phases
        sim.advance_phase()
    """
    
    def __init__(self):
        self._state: Optional[CrisisState] = None
        self._history: List[Dict[str, Any]] = []
        self._diagnosis: Optional[DiagnosisReport] = None
        self._shadow_result: Optional[ShadowResult] = None
        self._callbacks: Dict[CrisisSimPhase, List[Callable]] = {}
    
    @property
    def current_phase(self) -> Optional[CrisisSimPhase]:
        """Get current phase."""
        return self._state.phase if self._state else None
    
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed in current phase."""
        if not self._state:
            return True
        return CRISIS_PHASES[self._state.phase].can_trade
    
    @property
    def can_exit(self) -> bool:
        """Check if exits are allowed in current phase."""
        if not self._state:
            return True
        return CRISIS_PHASES[self._state.phase].can_exit
    
    def start_normal(self):
        """Start simulation in normal operations."""
        self._state = CrisisState(
            phase=CrisisSimPhase.T0_NORMAL,
            phase_started_at=datetime.utcnow(),
        )
        self._log_event("SIMULATION_START", "Normal operations started")
        logger.info("🕒 T0 — NORMAL DRIFT: Simulation started")
    
    def trigger_crisis(
        self,
        reason: str,
        spread_ratio: float = 10.0,
        volume_drop: float = 0.8,
        positions: int = 1,
        pnl: float = 0.0,
    ):
        """
        Trigger a crisis event.
        
        Args:
            reason: Description of trigger
            spread_ratio: Spread expansion ratio vs normal
            volume_drop: Volume drop ratio (0.9 = 90% drop)
            positions: Number of open positions
            pnl: Current PnL at trigger
        """
        if not self._state:
            self.start_normal()
        
        self._state.trigger_reason = reason
        self._state.positions_at_trigger = positions
        self._state.pnl_at_trigger = pnl
        
        # Move to trigger phase
        self._transition_to(CrisisSimPhase.T30S_TRIGGER)
        
        logger.warning(
            f"🕒 T+30 sek — TRIGGER DETECTED:\n"
            f"   Reason: {reason}\n"
            f"   Spread: {spread_ratio}x normal\n"
            f"   Volume: -{volume_drop*100:.0f}%\n"
            f"   Positions: {positions}"
        )
        
        self._log_event("CRISIS_TRIGGER", {
            "reason": reason,
            "spread_ratio": spread_ratio,
            "volume_drop": volume_drop,
            "positions": positions,
            "pnl": pnl,
        })
    
    def activate_kill_switch(self):
        """Activate kill-switch (T+35 sek)."""
        if self._state.phase != CrisisSimPhase.T30S_TRIGGER:
            logger.warning("Kill-switch can only be activated from TRIGGER phase")
            return
        
        self._transition_to(CrisisSimPhase.T35S_KILLSWITCH)
        self._state.incident_logged = True
        
        logger.critical(
            "🛑 T+35 sek — KILL-SWITCH ACTIVATED:\n"
            "   • All ny entry STOPPED\n"
            "   • Kun reduce-only tillatt\n"
            "   • Hendelse merket CRITICAL INCIDENT\n"
            "   • Ingen diskusjon. Ingen manuell input."
        )
        
        self._log_event("KILL_SWITCH", {"incident_logged": True})
    
    def secure_positions(self, positions_closed: int = 0):
        """Secure/close all positions (T+2 min)."""
        if self._state.phase != CrisisSimPhase.T35S_KILLSWITCH:
            logger.warning("Securing can only happen after KILL_SWITCH")
            return
        
        self._transition_to(CrisisSimPhase.T2M_SECURE)
        self._state.positions_closed = positions_closed or self._state.positions_at_trigger
        
        logger.info(
            f"🧯 T+2 min — SIKRING:\n"
            f"   • Posisjoner lukket: {self._state.positions_closed}\n"
            f"   • System: SAFE MODE\n"
            f"   • Exchange: ISOLERT\n"
            f"   • Mål: Ikke tape mer. Ikke 'vinne'."
        )
        
        self._log_event("SECURED", {"positions_closed": self._state.positions_closed})
    
    def start_cooling_off(self):
        """Start cooling-off period (T+15-60 min)."""
        if self._state.phase != CrisisSimPhase.T2M_SECURE:
            logger.warning("Cooling-off requires SECURED phase first")
            return
        
        self._transition_to(CrisisSimPhase.T15M_COOLING)
        
        logger.info(
            "❄️ T+15–60 min — COOLING-OFF:\n"
            "   • Ingen trading\n"
            "   • Kun observasjon\n"
            "   • Vent på volatilitet normaliseres"
        )
        
        self._log_event("COOLING_OFF_START", {})
    
    def start_diagnosis(self):
        """Start diagnosis phase (T+60-90 min)."""
        if self._state.phase != CrisisSimPhase.T15M_COOLING:
            logger.warning("Diagnosis requires COOLING_OFF phase first")
            return
        
        self._transition_to(CrisisSimPhase.T60M_DIAGNOSE)
        
        logger.info(
            "🔍 T+60–90 min — DIAGNOSE:\n"
            "   Svar på tre spørsmål:\n"
            "   1. Hva trigget?\n"
            "   2. Var det policy-forventet?\n"
            "   3. Kunne høyere lag stoppet før?\n"
            "   \n"
            "   ⚠️ Ingen endringer gjøres!"
        )
        
        self._log_event("DIAGNOSIS_START", {})
    
    def complete_diagnosis(
        self,
        what_triggered: str,
        was_policy_expected: bool,
        could_prevent_earlier: bool,
        root_cause: str,
        recommendations: List[str],
    ):
        """Complete diagnosis with answers to the three questions."""
        if self._state.phase != CrisisSimPhase.T60M_DIAGNOSE:
            logger.warning("Diagnosis can only be completed in DIAGNOSE phase")
            return
        
        self._diagnosis = DiagnosisReport(
            what_triggered=what_triggered,
            was_policy_expected=was_policy_expected,
            could_prevent_earlier=could_prevent_earlier,
            root_cause=root_cause,
            recommendations=recommendations,
        )
        
        self._state.diagnosis_complete = True
        
        logger.info(
            f"🔍 DIAGNOSE COMPLETE:\n"
            f"   1. Hva trigget? → {what_triggered}\n"
            f"   2. Policy-forventet? → {'Ja' if was_policy_expected else 'Nei'}\n"
            f"   3. Kunne forebygget? → {'Ja' if could_prevent_earlier else 'Nei'}\n"
            f"   Root cause: {root_cause}"
        )
        
        self._log_event("DIAGNOSIS_COMPLETE", self._diagnosis.__dict__)
    
    def start_shadow_mode(self):
        """Start shadow mode (T+90-180 min)."""
        if not self._state.diagnosis_complete:
            logger.warning("Shadow mode requires completed diagnosis")
            return
        
        self._transition_to(CrisisSimPhase.T90M_SHADOW)
        
        logger.info(
            "👻 T+90–180 min — SHADOW MODE:\n"
            "   • Samme signaler\n"
            "   • Null kapital\n"
            "   • Sjekk policy, regime, exit-logikk"
        )
        
        self._log_event("SHADOW_MODE_START", {})
    
    def complete_shadow_mode(
        self,
        duration_minutes: int,
        signals_processed: int,
        would_have_traded: int,
        policy_violations: int,
        regime_stable: bool,
        exit_logic_ok: bool,
    ):
        """Complete shadow mode with results."""
        if self._state.phase != CrisisSimPhase.T90M_SHADOW:
            logger.warning("Shadow results can only be recorded in SHADOW phase")
            return
        
        ready = (
            policy_violations == 0 and
            regime_stable and
            exit_logic_ok
        )
        
        self._shadow_result = ShadowResult(
            duration_minutes=duration_minutes,
            signals_processed=signals_processed,
            would_have_traded=would_have_traded,
            policy_violations=policy_violations,
            regime_stable=regime_stable,
            exit_logic_ok=exit_logic_ok,
            ready_for_restart=ready,
        )
        
        self._state.shadow_results = self._shadow_result.__dict__
        
        status = "✅ READY" if ready else "❌ NOT READY"
        logger.info(
            f"👻 SHADOW MODE COMPLETE:\n"
            f"   Duration: {duration_minutes} min\n"
            f"   Signals: {signals_processed}\n"
            f"   Would have traded: {would_have_traded}\n"
            f"   Policy violations: {policy_violations}\n"
            f"   Regime stable: {regime_stable}\n"
            f"   Exit logic OK: {exit_logic_ok}\n"
            f"   \n"
            f"   {status} for restart"
        )
        
        self._log_event("SHADOW_MODE_COMPLETE", self._shadow_result.__dict__)
    
    def start_gradual_restart(self):
        """Start gradual restart (T+3-24 t)."""
        if not self._shadow_result or not self._shadow_result.ready_for_restart:
            logger.warning("Gradual restart requires successful shadow mode")
            return
        
        self._transition_to(CrisisSimPhase.T3H_GRADUAL)
        self._state.restart_attempts += 1
        
        logger.info(
            "🟢 T+3–24 t — GRADVIS RESTART:\n"
            "   • 25–50% normal size\n"
            "   • Kun TOP-N edges\n"
            "   • Ingen leverage-eskalering\n"
            "   • Exit-dominans\n"
            "   • Hvis presset → tilbake til pause"
        )
        
        self._log_event("GRADUAL_RESTART", {"attempt": self._state.restart_attempts})
    
    def go_full_live(self):
        """Return to full live trading."""
        if self._state.phase != CrisisSimPhase.T3H_GRADUAL:
            logger.warning("Full live requires successful gradual restart")
            return
        
        # Check all conditions
        conditions = {
            "Shadow OK": self._shadow_result and self._shadow_result.ready_for_restart,
            "Ingen overrides": True,  # Would check override lock
            "Risk kernel urørt": True,  # Would check risk kernel
            "Data stabil": True,  # Would check data stability
        }
        
        all_met = all(conditions.values())
        
        if not all_met:
            failed = [k for k, v in conditions.items() if not v]
            logger.warning(f"❌ Cannot go FULL LIVE - conditions not met: {failed}")
            return
        
        self._transition_to(CrisisSimPhase.FULL_LIVE)
        
        logger.info(
            "✅ FULL LIVE — All conditions met:\n"
            "   • Shadow OK ✓\n"
            "   • Ingen overrides ✓\n"
            "   • Risk kernel urørt ✓\n"
            "   • Data stabil ✓"
        )
        
        self._log_event("FULL_LIVE_RESTORED", conditions)
    
    def _transition_to(self, phase: CrisisSimPhase):
        """Transition to a new phase."""
        old_phase = self._state.phase if self._state else None
        self._state.phase = phase
        self._state.phase_started_at = datetime.utcnow()
        
        # Execute callbacks
        if phase in self._callbacks:
            for callback in self._callbacks[phase]:
                try:
                    callback(self._state)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def _log_event(self, event_type: str, data: Any):
        """Log an event to history."""
        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "phase": self._state.phase.value if self._state else None,
            "event": event_type,
            "data": data,
        })
    
    def on_phase(self, phase: CrisisSimPhase, callback: Callable):
        """Register a callback for a phase transition."""
        if phase not in self._callbacks:
            self._callbacks[phase] = []
        self._callbacks[phase].append(callback)
    
    async def run_simulation(self, fast_mode: bool = True):
        """
        Run full crisis simulation automatically.
        
        Args:
            fast_mode: If True, skip time delays (for testing)
        """
        # T0: Normal
        self.start_normal()
        if not fast_mode:
            await asyncio.sleep(2)
        
        # T+30s: Trigger
        self.trigger_crisis(
            reason="Flash crash + data-inkonsistens",
            spread_ratio=15.0,
            volume_drop=0.9,
            positions=1,
        )
        if not fast_mode:
            await asyncio.sleep(5)
        
        # T+35s: Kill-switch
        self.activate_kill_switch()
        if not fast_mode:
            await asyncio.sleep(5)
        
        # T+2m: Secure
        self.secure_positions(1)
        if not fast_mode:
            await asyncio.sleep(10)
        
        # T+15m: Cooling-off
        self.start_cooling_off()
        if not fast_mode:
            await asyncio.sleep(10)
        
        # T+60m: Diagnosis
        self.start_diagnosis()
        self.complete_diagnosis(
            what_triggered="Data + likviditet",
            was_policy_expected=True,
            could_prevent_earlier=False,
            root_cause="Likviditets-vakuum under høy volatilitet",
            recommendations=[
                "Tighter spread monitoring",
                "Earlier volume triggers",
            ],
        )
        if not fast_mode:
            await asyncio.sleep(10)
        
        # T+90m: Shadow mode
        self.start_shadow_mode()
        self.complete_shadow_mode(
            duration_minutes=90,
            signals_processed=50,
            would_have_traded=3,
            policy_violations=0,
            regime_stable=True,
            exit_logic_ok=True,
        )
        if not fast_mode:
            await asyncio.sleep(5)
        
        # T+3h: Gradual restart
        self.start_gradual_restart()
        if not fast_mode:
            await asyncio.sleep(5)
        
        # Full live
        self.go_full_live()
        
        logger.info("\n✅ CRISIS SIMULATION COMPLETE")
    
    def print_status(self):
        """Print current simulation status."""
        print("\n" + "=" * 70)
        print("🌪️ KRISE SIMULERING STATUS")
        print("=" * 70)
        
        if not self._state:
            print("\n⬜ Ingen aktiv simulering")
            return
        
        phase_config = CRISIS_PHASES[self._state.phase]
        
        print(f"\n{phase_config.name}")
        print(f"Started: {self._state.phase_started_at.isoformat()}")
        print(f"Can trade: {'✅' if phase_config.can_trade else '❌'}")
        print(f"Can exit: {'✅' if phase_config.can_exit else '❌'}")
        
        if self._state.trigger_reason:
            print(f"\nTrigger: {self._state.trigger_reason}")
            print(f"Positions at trigger: {self._state.positions_at_trigger}")
            print(f"Positions closed: {self._state.positions_closed}")
        
        print(f"\n📋 Current phase actions:")
        for action in phase_config.actions:
            print(f"   → {action}")
        
        print(f"\n🚫 Forbidden:")
        for forbidden in phase_config.forbidden:
            print(f"   ✗ {forbidden}")
        
        if self._diagnosis:
            print(f"\n🔍 Diagnosis:")
            print(f"   What triggered: {self._diagnosis.what_triggered}")
            print(f"   Policy expected: {self._diagnosis.was_policy_expected}")
            print(f"   Could prevent: {self._diagnosis.could_prevent_earlier}")
        
        if self._shadow_result:
            print(f"\n👻 Shadow results:")
            print(f"   Ready for restart: {'✅' if self._shadow_result.ready_for_restart else '❌'}")
        
        print("\n" + "=" * 70)
    
    def print_timeline(self):
        """Print full crisis timeline."""
        print("\n" + "=" * 70)
        print("🌪️ KRISE TIDSLINJE")
        print("=" * 70)
        
        for event in self._history:
            print(f"\n[{event['timestamp'][:19]}] {event['event']}")
            if event['phase']:
                print(f"   Phase: {event['phase']}")
        
        print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RULES REMINDER
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_RULES_CRISIS = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    🧠 GYLDNE REGLER (HUSK)                                 ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║    • Flat er suksess i krise                                               ║
║    • Restart sakte – markedet venter                                       ║
║    • Hvis du vil starte raskt, er svaret NEI                               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_simulator: Optional[CrisisSimulator] = None


def get_crisis_simulator() -> CrisisSimulator:
    """Get singleton crisis simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = CrisisSimulator()
    return _simulator


def reset_crisis_simulator() -> CrisisSimulator:
    """Reset and return fresh simulator."""
    global _simulator
    _simulator = CrisisSimulator()
    return _simulator


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run full crisis simulation demo."""
    sim = reset_crisis_simulator()
    
    print(GOLDEN_RULES_CRISIS)
    
    await sim.run_simulation(fast_mode=True)
    
    sim.print_status()
    sim.print_timeline()


if __name__ == "__main__":
    asyncio.run(main())
