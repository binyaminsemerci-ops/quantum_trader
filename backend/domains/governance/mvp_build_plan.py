"""
🏗️ MVP BUILD-PLAN (UKE 1–2–3)
==============================
Bygg det som kan STOPPE før du bygger det som kan TJENE.

Tre hoveddeler:
1️⃣ MVP Build Plan (3 uker)
2️⃣ Exit-formler (5 typer)
3️⃣ Live-dag simulering (time for time)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: MVP BUILD PLAN
# ═══════════════════════════════════════════════════════════════════════════════

class BuildWeek(int, Enum):
    """Build weeks in MVP plan."""
    WEEK_1 = 1  # Sikkerhet & Overlevelse
    WEEK_2 = 2  # Exit Først
    WEEK_3 = 3  # Entry med Tillatelse


class DeliverableStatus(str, Enum):
    """Status of a deliverable."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Deliverable:
    """A deliverable in the build plan."""
    name: str
    description: str
    components: List[str]
    status: DeliverableStatus = DeliverableStatus.NOT_STARTED
    completed_at: Optional[datetime] = None


@dataclass
class WeekPlan:
    """A week's build plan."""
    week: BuildWeek
    name: str
    emoji: str
    goal: str
    deliverables: List[Deliverable]
    result: str
    insight: str


# ═══════════════════════════════════════════════════════════════════════════════
# THE 3-WEEK MVP PLAN
# ═══════════════════════════════════════════════════════════════════════════════

MVP_BUILD_PLAN: Dict[BuildWeek, WeekPlan] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UKE 1 — SIKKERHET & OVERLEVELSE
    # ═══════════════════════════════════════════════════════════════════════════
    
    BuildWeek.WEEK_1: WeekPlan(
        week=BuildWeek.WEEK_1,
        name="SIKKERHET & OVERLEVELSE",
        emoji="🗓️",
        goal="Systemet skal kunne si NEI og stoppe seg selv",
        deliverables=[
            Deliverable(
                name="Policy / Constitution Service",
                description="Alle grunnlover lastet, forbud aktivert",
                components=[
                    "15 grunnlover lastet",
                    "Forbud aktivert",
                    "Validation logic",
                    "Constitutional enforcer",
                ],
            ),
            Deliverable(
                name="Risk Kernel (Fail-Closed)",
                description="Hard limits og global kill-switch",
                components=[
                    "Max risk per trade (2%)",
                    "Max daglig tap (-5%)",
                    "Global kill-switch",
                    "Fail-closed default",
                ],
            ),
            Deliverable(
                name="Data Integrity Sentinel",
                description="Feed-sjekk og latency monitoring",
                components=[
                    "Feed-konsistens sjekk",
                    "Latency monitoring",
                    "Clock drift detection",
                    "Safe mode trigger",
                ],
            ),
            Deliverable(
                name="Audit / Ledger",
                description="Append-only logg for alle hendelser",
                components=[
                    "Append-only logg",
                    "Hendelser logging",
                    "Beslutninger logging",
                    "Immutable history",
                ],
            ),
        ],
        result="Systemet kan ikke tjene penger – men det kan ikke drepe seg selv heller.",
        insight="📌 Bygg det som kan STOPPE før du bygger det som kan TJENE",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UKE 2 — EXIT FØRST (IKKE ENTRY)
    # ═══════════════════════════════════════════════════════════════════════════
    
    BuildWeek.WEEK_2: WeekPlan(
        week=BuildWeek.WEEK_2,
        name="EXIT FØRST (IKKE ENTRY)",
        emoji="🗓️",
        goal="Systemet kan forlate markedet korrekt",
        deliverables=[
            Deliverable(
                name="Exit / Harvest Brain (MVP)",
                description="Grunnleggende exit-logikk",
                components=[
                    "Time-based exit",
                    "Volatility-exit",
                    "Regime-exit",
                    "Panic exit",
                ],
            ),
            Deliverable(
                name="Execution (reduce-only)",
                description="Kun exits, ingen nye entries",
                components=[
                    "Reduce-only mode",
                    "Slippage-kontroll",
                    "Order management",
                    "Fill tracking",
                ],
            ),
            Deliverable(
                name="Position State (minimal)",
                description="Grunnleggende posisjonssporing",
                components=[
                    "Åpen/lukket status",
                    "Tid i trade",
                    "Unrealized PnL",
                    "Entry price tracking",
                ],
            ),
        ],
        result="Systemet kan gå INN feil – men kommer UT riktig.",
        insight="📌 Exit er kunst, entry er lett",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UKE 3 — ENTRY MED TILLATELSE
    # ═══════════════════════════════════════════════════════════════════════════
    
    BuildWeek.WEEK_3: WeekPlan(
        week=BuildWeek.WEEK_3,
        name="ENTRY MED TILLATELSE",
        emoji="🗓️",
        goal="Kun trades som overlever alle porter",
        deliverables=[
            Deliverable(
                name="Market Regime Detector",
                description="Identifisere markedsregime",
                components=[
                    "Trend detection",
                    "Chop detection",
                    "Panic detection",
                    "Regime confidence",
                ],
            ),
            Deliverable(
                name="AI / Signal Advisory (read-only)",
                description="Signalgenerering uten makt",
                components=[
                    "Edge scoring",
                    "Confidence calculation",
                    "Signal generation",
                    "Read-only advisory",
                ],
            ),
            Deliverable(
                name="Capital Allocation (konservativ)",
                description="Konservativ sizing",
                components=[
                    "Position sizing",
                    "Conservative mode",
                    "Correlation check",
                    "Heat calculation",
                ],
            ),
            Deliverable(
                name="Entry Qualification Gate",
                description="Siste filter før markedet",
                components=[
                    "Exit-plan verification",
                    "Risk definition check",
                    "Structure validation",
                    "Policy compliance",
                ],
            ),
            Deliverable(
                name="Pre-flight checklist (automatisk)",
                description="Automatisk GO/NO-GO sjekk",
                components=[
                    "Automated checks",
                    "Category validation",
                    "NO-GO detection",
                    "Report generation",
                ],
            ),
        ],
        result="Systemet trader kun når det er trygt å tape.",
        insight="📌 Kun trades som overlever alle porter",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: EXIT FORMULAS
# ═══════════════════════════════════════════════════════════════════════════════

class ExitType(str, Enum):
    """Types of exit formulas."""
    TIME_BASED = "time_based"          # 1️⃣
    VOLATILITY = "volatility"          # 2️⃣
    REGIME = "regime"                  # 3️⃣ (Viktigste)
    PROFIT_TAKING = "profit_taking"    # 4️⃣
    CAPITAL_STRESS = "capital_stress"  # 5️⃣


@dataclass
class ExitFormula:
    """An exit formula specification."""
    exit_type: ExitType
    number: str
    name: str
    principle: str
    formula: str
    saves_from: List[str]
    priority: int  # Lower = higher priority
    insight: str


EXIT_FORMULAS: Dict[ExitType, ExitFormula] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1️⃣ TIME-BASERT EXIT (UNDERVURDERT, KRITISK)
    # ═══════════════════════════════════════════════════════════════════════════
    
    ExitType.TIME_BASED: ExitFormula(
        exit_type=ExitType.TIME_BASED,
        number="1️⃣",
        name="TIME-BASERT EXIT",
        principle="Tid i trade = risiko. Hvis edge ikke materialiseres raskt → den finnes ikke",
        formula="Exit hvis: Time_in_trade > T_max(regime, volatility)",
        saves_from=[
            "Chop",
            "Funding-erosjon",
            "Mentale feller",
            "Opportunity cost",
        ],
        priority=2,
        insight="📌 UNDERVURDERT, KRITISK - de fleste ignorerer dette",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2️⃣ VOLATILITETS-EXIT
    # ═══════════════════════════════════════════════════════════════════════════
    
    ExitType.VOLATILITY: ExitFormula(
        exit_type=ExitType.VOLATILITY,
        number="2️⃣",
        name="VOLATILITETS-EXIT",
        principle="Volatilitet som endrer karakter = exit",
        formula="Exit hvis: Current_Vol > k × Entry_Vol",
        saves_from=[
            "Squeezes",
            "Liquidation-kaskader",
            "Falske breakouts",
            "Stop hunts",
        ],
        priority=3,
        insight="📌 Marked endrer karakter = exit uansett PnL",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3️⃣ REGIME-EXIT (DEN VIKTIGSTE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    ExitType.REGIME: ExitFormula(
        exit_type=ExitType.REGIME,
        number="3️⃣",
        name="REGIME-EXIT",
        principle="Når markedet slutter å være det du trodde",
        formula="Exit hvis: Regime_now ≠ Regime_entry",
        saves_from=[
            "Trend-til-chop overgang",
            "Flash crashes",
            "Macro regime shifts",
            "All entry-logikk failures",
        ],
        priority=1,  # HIGHEST PRIORITY
        insight="📌 Dette slår ALL entry-logikk. DEN VIKTIGSTE.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4️⃣ PROFIT-TAKING (PARTIALS)
    # ═══════════════════════════════════════════════════════════════════════════
    
    ExitType.PROFIT_TAKING: ExitFormula(
        exit_type=ExitType.PROFIT_TAKING,
        number="4️⃣",
        name="PROFIT-TAKING (PARTIALS)",
        principle="Du trenger ikke vinne hele bevegelsen",
        formula="TP1: lav terskel → risk off\nTP2: trail → let winners run",
        saves_from=[
            "Giving back profits",
            "Greed",
            "Perfect exit syndrome",
            "Reversal risk",
        ],
        priority=4,
        insight="📌 Hedgefond tar profit tidlig, lar resten jobbe gratis",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5️⃣ KAPITAL-STRESS EXIT
    # ═══════════════════════════════════════════════════════════════════════════
    
    ExitType.CAPITAL_STRESS: ExitFormula(
        exit_type=ExitType.CAPITAL_STRESS,
        number="5️⃣",
        name="KAPITAL-STRESS EXIT",
        principle="Systemets helse > trade",
        formula="Exit hvis: Portfolio_Stress > terskel",
        saves_from=[
            "Kaskadetap",
            "Korrelert død",
            "Margin call",
            "System breakdown",
        ],
        priority=1,  # HIGHEST PRIORITY (same as regime)
        insight="📌 Systemets helse > individuell trade",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT FORMULA PRIORITY ORDER
# ═══════════════════════════════════════════════════════════════════════════════

EXIT_PRIORITY_ORDER = """
🧲 EXIT-FORMLER PRIORITET
=========================

1. REGIME-EXIT + KAPITAL-STRESS (høyest)
   → Exit UMIDDELBART, ingen spørsmål

2. TIME-BASED EXIT
   → Hvis edge ikke materialiseres raskt

3. VOLATILITY-EXIT
   → Marked endrer karakter

4. PROFIT-TAKING
   → Ta det du kan, la resten trail

📌 REGEL: Høyere prioritet overstyrer lavere
📌 UNNTAK: Ingen - disse reglene er absolutte
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: LIVE DAY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class SimulationEventType(str, Enum):
    """Types of events in live day simulation."""
    PRE_FLIGHT = "pre_flight"
    OPPORTUNITY = "opportunity"
    IN_POSITION = "in_position"
    REGIME_SHIFT = "regime_shift"
    FULL_EXIT = "full_exit"
    CHAOTIC_MARKET = "chaotic_market"
    NO_REENTRY = "no_reentry"
    DAY_END = "day_end"


class SimulationDecision(str, Enum):
    """Decisions made during simulation."""
    FLAT = "flat"
    ENTRY_ALLOWED = "entry_allowed"
    HOLD = "hold"
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    SAFE_MODE = "safe_mode"
    NO_TRADE = "no_trade"


@dataclass
class SimulationEvent:
    """An event in the live day simulation."""
    time: str
    emoji: str
    name: str
    event_type: SimulationEventType
    conditions: List[str]
    decision: SimulationDecision
    action: str
    insight: str
    pnl_change: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST LIVE DAY SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

LIVE_DAY_SIMULATION: List[SimulationEvent] = [
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕘 08:00 – PRE-FLIGHT
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="08:00",
        emoji="🕘",
        name="PRE-FLIGHT",
        event_type=SimulationEventType.PRE_FLIGHT,
        conditions=[
            "Checklist kjørt",
            "Risk kernel OK",
            "Data OK",
            "Regime: usikker → FLAT",
        ],
        decision=SimulationDecision.FLAT,
        action="Ingen trading",
        insight="✅ Riktig handling - usikker regime = FLAT",
        pnl_change=0.0,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕙 10:00 – FØRSTE MULIGHET
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="10:00",
        emoji="🕙",
        name="FØRSTE MULIGHET",
        event_type=SimulationEventType.OPPORTUNITY,
        conditions=[
            "Regime stabiliseres",
            "AI gir edge-score",
            "Policy: tillatt",
            "Risk: godkjent",
            "Capital: lav size",
        ],
        decision=SimulationDecision.ENTRY_ALLOWED,
        action="Entry tillatt",
        insight="➡️ Alle porter passert - entry med lav size",
        pnl_change=0.0,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕥 10:30 – I POSISJON
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="10:30",
        emoji="🕥",
        name="I POSISJON",
        event_type=SimulationEventType.IN_POSITION,
        conditions=[
            "PnL: +0.3%",
            "Volatilitet stabil",
            "Tid < T_max",
            "Regime konsistent",
        ],
        decision=SimulationDecision.HOLD,
        action="Hold posisjon",
        insight="➡️ Alt OK - la posisjonen jobbe",
        pnl_change=0.3,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕚 11:15 – REGIME VAKLER
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="11:15",
        emoji="🕚",
        name="REGIME VAKLER",
        event_type=SimulationEventType.REGIME_SHIFT,
        conditions=[
            "Regime-detektor mister konsensus",
            "Exit-brain aktiveres",
        ],
        decision=SimulationDecision.PARTIAL_EXIT,
        action="Partial exit + trail strammes",
        insight="➡️ Regime-exit forberedes - ta profit",
        pnl_change=0.35,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕛 12:00 – FULL EXIT
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="12:00",
        emoji="🕛",
        name="FULL EXIT",
        event_type=SimulationEventType.FULL_EXIT,
        conditions=[
            "Regime bekreftet skift",
            "Trade avsluttes",
        ],
        decision=SimulationDecision.FULL_EXIT,
        action="Full exit +0.4%",
        insight="📌 Ikke maks profit, men ✅ Korrekt profit",
        pnl_change=0.4,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕑 14:00 – KAOTISK MARKED
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="14:00",
        emoji="🕑",
        name="KAOTISK MARKED",
        event_type=SimulationEventType.CHAOTIC_MARKET,
        conditions=[
            "Spread utvider",
            "Data sentinel flagger ustabilitet",
        ],
        decision=SimulationDecision.SAFE_MODE,
        action="SAFE MODE → FLAT",
        insight="➡️ Data-problem = ingen trading",
        pnl_change=0.0,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕔 17:00 – INGEN RE-ENTRY
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="17:00",
        emoji="🕔",
        name="INGEN RE-ENTRY",
        event_type=SimulationEventType.NO_REENTRY,
        conditions=[
            "Edge finnes",
            "Men systemstress > terskel",
        ],
        decision=SimulationDecision.NO_TRADE,
        action="NEI - ingen entry",
        insight="📌 Profesjonell disiplin - systemhelse > trade",
        pnl_change=0.0,
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🕗 20:00 – DAG SLUTT
    # ═══════════════════════════════════════════════════════════════════════════
    
    SimulationEvent(
        time="20:00",
        emoji="🕗",
        name="DAG SLUTT",
        event_type=SimulationEventType.DAY_END,
        conditions=[
            "1 trade",
            "+0.4%",
            "0 policy-brudd",
            "0 stress",
        ],
        decision=SimulationDecision.FLAT,
        action="Dag avsluttes FLAT",
        insight="✅ Beste første dag et fond kan ha",
        pnl_change=0.4,  # Final PnL
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE DAY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

LIVE_DAY_SUMMARY = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🕗 DAG SLUTT - OPPSUMMERING                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    📊 STATISTIKK                                                             ║
║    • Trades: 1                                                               ║
║    • PnL: +0.4%                                                              ║
║    • Policy-brudd: 0                                                         ║
║    • Stress-hendelser: 0                                                     ║
║                                                                              ║
║    ✅ RIKTIGE BESLUTNINGER                                                   ║
║    • 08:00: Ingen trading (usikker regime)                                   ║
║    • 10:00: Lav size entry (alle porter passert)                             ║
║    • 11:15: Partial exit (regime vakler)                                     ║
║    • 12:00: Full exit (regime bekreftet)                                     ║
║    • 14:00: Safe mode (data ustabil)                                         ║
║    • 17:00: Ingen re-entry (systemstress)                                    ║
║                                                                              ║
║    📌 Beste første dag et fond kan ha.                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

SLUTTSANNHET = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 SLUTTSANNHET                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    Entry er lett                                                             ║
║    Exit er kunst                                                             ║
║    Overlevelse er strategi                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MVP BUILD PLAN MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class MVPBuildPlanManager:
    """
    Manager for the 3-week MVP build plan.
    
    Tracks progress through:
    - Week 1: Sikkerhet & Overlevelse
    - Week 2: Exit Først
    - Week 3: Entry med Tillatelse
    """
    
    def __init__(self):
        self._plan = MVP_BUILD_PLAN
        self._exit_formulas = EXIT_FORMULAS
        self._simulation = LIVE_DAY_SIMULATION
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD PLAN
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_week(self, week: BuildWeek) -> WeekPlan:
        """Get a specific week's plan."""
        return self._plan[week]
    
    def get_all_weeks(self) -> List[WeekPlan]:
        """Get all weeks in order."""
        return [self._plan[w] for w in BuildWeek]
    
    def get_deliverables(self, week: BuildWeek) -> List[Deliverable]:
        """Get deliverables for a week."""
        return self._plan[week].deliverables
    
    def mark_deliverable_complete(
        self,
        week: BuildWeek,
        deliverable_name: str,
    ) -> bool:
        """Mark a deliverable as complete."""
        for d in self._plan[week].deliverables:
            if d.name == deliverable_name:
                d.status = DeliverableStatus.COMPLETED
                d.completed_at = datetime.now()
                return True
        return False
    
    def get_progress(self) -> Dict[BuildWeek, float]:
        """Get completion percentage per week."""
        progress = {}
        for week in BuildWeek:
            deliverables = self._plan[week].deliverables
            completed = sum(1 for d in deliverables 
                         if d.status == DeliverableStatus.COMPLETED)
            progress[week] = completed / len(deliverables) if deliverables else 0.0
        return progress
    
    def is_week_complete(self, week: BuildWeek) -> bool:
        """Check if a week is fully complete."""
        return all(
            d.status == DeliverableStatus.COMPLETED 
            for d in self._plan[week].deliverables
        )
    
    def can_start_week(self, week: BuildWeek) -> bool:
        """Check if a week can be started (previous weeks complete)."""
        if week == BuildWeek.WEEK_1:
            return True
        
        previous_week = BuildWeek(week.value - 1)
        return self.is_week_complete(previous_week)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXIT FORMULAS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_exit_formula(self, exit_type: ExitType) -> ExitFormula:
        """Get an exit formula."""
        return self._exit_formulas[exit_type]
    
    def get_exit_formulas_by_priority(self) -> List[ExitFormula]:
        """Get exit formulas sorted by priority (highest first)."""
        return sorted(
            self._exit_formulas.values(),
            key=lambda f: f.priority
        )
    
    def should_exit(
        self,
        time_in_trade: float,
        t_max: float,
        current_vol: float,
        entry_vol: float,
        vol_multiplier: float,
        regime_now: str,
        regime_entry: str,
        portfolio_stress: float,
        stress_threshold: float,
    ) -> Tuple[bool, Optional[ExitType], str]:
        """
        Evaluate all exit formulas and return exit decision.
        
        Returns:
            (should_exit, exit_type, reason)
        """
        # 1. Regime Exit (highest priority)
        if regime_now != regime_entry:
            return True, ExitType.REGIME, f"Regime shifted: {regime_entry} → {regime_now}"
        
        # 2. Capital Stress Exit (highest priority)
        if portfolio_stress > stress_threshold:
            return True, ExitType.CAPITAL_STRESS, f"Portfolio stress {portfolio_stress:.1%} > {stress_threshold:.1%}"
        
        # 3. Time-based Exit
        if time_in_trade > t_max:
            return True, ExitType.TIME_BASED, f"Time {time_in_trade:.1f}h > T_max {t_max:.1f}h"
        
        # 4. Volatility Exit
        if current_vol > vol_multiplier * entry_vol:
            return True, ExitType.VOLATILITY, f"Vol {current_vol:.1%} > {vol_multiplier}x entry ({entry_vol:.1%})"
        
        return False, None, "No exit conditions met"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_simulation_events(self) -> List[SimulationEvent]:
        """Get the live day simulation events."""
        return self._simulation
    
    def run_simulation_step(self, event_index: int) -> Optional[SimulationEvent]:
        """Run a single simulation step."""
        if 0 <= event_index < len(self._simulation):
            return self._simulation[event_index]
        return None
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get simulation summary statistics."""
        final_event = self._simulation[-1]
        
        trades = sum(1 for e in self._simulation 
                    if e.decision == SimulationDecision.ENTRY_ALLOWED)
        
        decisions = [e.decision for e in self._simulation]
        
        return {
            "total_events": len(self._simulation),
            "trades": trades,
            "final_pnl": final_event.pnl_change,
            "policy_violations": 0,
            "stress_events": sum(1 for e in self._simulation 
                               if e.decision == SimulationDecision.SAFE_MODE),
            "correct_decisions": len(self._simulation),  # All decisions were correct
            "decisions": decisions,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRINT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_build_plan(self):
        """Print the complete build plan."""
        print("\n" + "=" * 80)
        print("🏗️ MVP BUILD-PLAN (UKE 1–2–3)")
        print("Bygg det som kan STOPPE før du bygger det som kan TJENE")
        print("=" * 80)
        
        for week in BuildWeek:
            plan = self._plan[week]
            print(f"\n{plan.emoji} UKE {week.value} — {plan.name}")
            print("-" * 60)
            print(f"Mål: {plan.goal}")
            print("\nLeveranser:")
            
            for d in plan.deliverables:
                status = "✅" if d.status == DeliverableStatus.COMPLETED else "⬜"
                print(f"  {status} {d.name}")
                print(f"     {d.description}")
            
            print(f"\n📌 Resultat: {plan.result}")
        
        print("\n" + "=" * 80)
    
    def print_exit_formulas(self):
        """Print all exit formulas."""
        print("\n" + "=" * 80)
        print("🧲 EXIT-FORMLER (FØR ENTRY!)")
        print("Dette er hedgefondets hemmelige saus")
        print("=" * 80)
        
        for formula in self.get_exit_formulas_by_priority():
            print(f"\n{formula.number} {formula.name}")
            print("-" * 40)
            print(f"Prinsipp: {formula.principle}")
            print(f"Formel: {formula.formula}")
            print(f"Redder fra: {', '.join(formula.saves_from)}")
            print(f"{formula.insight}")
        
        print(EXIT_PRIORITY_ORDER)
    
    def print_simulation(self):
        """Print the live day simulation."""
        print("\n" + "=" * 80)
        print("⏱️ SIMULERING: FØRSTE LIVE-DAG (TIME FOR TIME)")
        print("=" * 80)
        
        for event in self._simulation:
            print(f"\n{event.emoji} {event.time} – {event.name}")
            print("-" * 40)
            for condition in event.conditions:
                print(f"  • {condition}")
            print(f"\n  ➡️ Beslutning: {event.decision.value}")
            print(f"  📌 {event.insight}")
        
        print(LIVE_DAY_SUMMARY)
    
    def print_full(self):
        """Print everything."""
        self.print_build_plan()
        self.print_exit_formulas()
        self.print_simulation()
        print(SLUTTSANNHET)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_mvp_build_manager: Optional[MVPBuildPlanManager] = None


def get_mvp_build_manager() -> MVPBuildPlanManager:
    """Get singleton MVP build plan manager instance."""
    global _mvp_build_manager
    if _mvp_build_manager is None:
        _mvp_build_manager = MVPBuildPlanManager()
    return _mvp_build_manager


def reset_mvp_build_manager() -> MVPBuildPlanManager:
    """Reset and get fresh manager instance."""
    global _mvp_build_manager
    _mvp_build_manager = MVPBuildPlanManager()
    return _mvp_build_manager


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    manager = get_mvp_build_manager()
    manager.print_full()
    
    # Example: Check should exit
    should_exit, exit_type, reason = manager.should_exit(
        time_in_trade=2.5,
        t_max=4.0,
        current_vol=0.03,
        entry_vol=0.015,
        vol_multiplier=2.0,
        regime_now="chop",
        regime_entry="trend",
        portfolio_stress=0.08,
        stress_threshold=0.10,
    )
    print(f"\nShould exit: {should_exit}")
    if exit_type:
        print(f"Exit type: {exit_type.value}")
    print(f"Reason: {reason}")
