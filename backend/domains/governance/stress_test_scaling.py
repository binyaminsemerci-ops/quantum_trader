"""
🧨 STRESS-TEST, KAPITAL-ESKALERING & DAY-0 HANDBOOK
=====================================================
Tre kritiske institusjonelle komponenter:

1️⃣ Stress-Test: 10 Tap på Rad
2️⃣ Kapital-Eskalering etter Suksess
3️⃣ Day-0 Handbook

Dette er det mest modne steget i hele prosessen.
Tester psykologi + system + kapital samtidig.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: STRESS-TEST (10 TAP PÅ RAD)
# ═══════════════════════════════════════════════════════════════════════════════

class StressTestResponse(str, Enum):
    """System responses during stress test."""
    NORMAL = "normal"                    # Trade 1-2
    CAPITAL_SCALE_DOWN_1 = "scale_down_1"  # Trade 3
    SMALLER_SIZE = "smaller_size"        # Trade 4
    TRADE_RATE_DOWN = "rate_down"        # Trade 5
    PAUSE_WINDOW = "pause_window"        # Trade 6
    STRATEGY_FREEZE = "strategy_freeze"  # Trade 7
    OBSERVE_ONLY = "observe_only"        # Trade 8
    FULL_STOP = "full_stop"              # Trade 9
    SHADOW_ONLY = "shadow_only"          # Trade 10


@dataclass
class StressTestTrade:
    """A trade in the stress test sequence."""
    trade_number: int
    result: str  # "Tap"
    system_response: StressTestResponse
    description: str
    capital_remaining_pct: float
    size_multiplier: float
    can_trade: bool
    warning_triggered: bool


@dataclass 
class StressTestResult:
    """Result of the stress test."""
    passed: bool
    capital_remaining_pct: float
    rules_changed: int
    override_attempts: int
    system_mental_state: str
    trades: List[StressTestTrade]
    failure_reasons: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TEST SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

STRESS_TEST_SEQUENCE: List[StressTestTrade] = [
    StressTestTrade(
        trade_number=1,
        result="Tap",
        system_response=StressTestResponse.NORMAL,
        description="Normal tap, systemet fortsetter uendret",
        capital_remaining_pct=98.0,  # -2% per trade
        size_multiplier=1.0,
        can_trade=True,
        warning_triggered=False,
    ),
    StressTestTrade(
        trade_number=2,
        result="Tap",
        system_response=StressTestResponse.NORMAL,
        description="Fortsatt normal, ingen endring",
        capital_remaining_pct=96.0,
        size_multiplier=1.0,
        can_trade=True,
        warning_triggered=False,
    ),
    StressTestTrade(
        trade_number=3,
        result="Tap",
        system_response=StressTestResponse.CAPITAL_SCALE_DOWN_1,
        description="⚠️ Capital scale-down aktivert",
        capital_remaining_pct=94.0,
        size_multiplier=0.75,
        can_trade=True,
        warning_triggered=True,
    ),
    StressTestTrade(
        trade_number=4,
        result="Tap",
        system_response=StressTestResponse.SMALLER_SIZE,
        description="Mindre size, samme regler",
        capital_remaining_pct=92.5,  # Less loss due to smaller size
        size_multiplier=0.5,
        can_trade=True,
        warning_triggered=False,
    ),
    StressTestTrade(
        trade_number=5,
        result="Tap",
        system_response=StressTestResponse.TRADE_RATE_DOWN,
        description="⚠️ Trade-rate redusert",
        capital_remaining_pct=91.5,
        size_multiplier=0.5,
        can_trade=True,
        warning_triggered=True,
    ),
    StressTestTrade(
        trade_number=6,
        result="Tap",
        system_response=StressTestResponse.PAUSE_WINDOW,
        description="Pause-vindu aktivert (cooldown)",
        capital_remaining_pct=90.5,
        size_multiplier=0.25,
        can_trade=True,
        warning_triggered=False,
    ),
    StressTestTrade(
        trade_number=7,
        result="Tap",
        system_response=StressTestResponse.STRATEGY_FREEZE,
        description="⚠️ Strategi frosset, ingen nye entries",
        capital_remaining_pct=90.0,
        size_multiplier=0.0,
        can_trade=False,
        warning_triggered=True,
    ),
    StressTestTrade(
        trade_number=8,
        result="Tap",
        system_response=StressTestResponse.OBSERVE_ONLY,
        description="Kun observasjon, ingen trading",
        capital_remaining_pct=90.0,  # No trade, no loss
        size_multiplier=0.0,
        can_trade=False,
        warning_triggered=False,
    ),
    StressTestTrade(
        trade_number=9,
        result="Tap",
        system_response=StressTestResponse.FULL_STOP,
        description="⚠️ Full stopp aktivert",
        capital_remaining_pct=90.0,
        size_multiplier=0.0,
        can_trade=False,
        warning_triggered=True,
    ),
    StressTestTrade(
        trade_number=10,
        result="Tap",
        system_response=StressTestResponse.SHADOW_ONLY,
        description="Shadow-only mode, simulering uten kapital",
        capital_remaining_pct=90.0,
        size_multiplier=0.0,
        can_trade=False,
        warning_triggered=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TEST PASS CRITERIA
# ═══════════════════════════════════════════════════════════════════════════════

STRESS_TEST_PASS_CRITERIA = """
✅ TESTEN ER BESTÅTT HVIS:

1. Kapital fortsatt > 90% av start
   → Systemet beskyttet seg selv

2. Ingen regel endret
   → Disiplin opprettholdt

3. Ingen override-forsøk
   → Mennesket holdt seg unna

4. Systemet er mentalt "kaldt"
   → Ingen emosjonell respons
"""

PSYCHOLOGICAL_REALITY = """
🧠 PSYKOLOGISK REALITET

Etter 3 tap → mennesket vil "justere"
Etter 5 tap → ego vil "ta igjen"
Etter 7 tap → de fleste ødelegger kontoer

👉 Derfor stopper systemet FØR deg.

📌 Viktig:
Systemet dør ikke på trade #10.
Det beskytter seg selv før det blir farlig.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: KAPITAL-ESKALERING
# ═══════════════════════════════════════════════════════════════════════════════

class ScalingLevel(int, Enum):
    """Scaling levels for capital."""
    PROOF_MODE = 0      # 🔹 Nivå 0
    CONFIRMED_EDGE = 1  # 🔹 Nivå 1
    STABLE_REGIME = 2   # 🔹 Nivå 2
    SCALABLE_MODE = 3   # 🔹 Nivå 3


@dataclass
class ScalingRequirements:
    """Requirements to reach a scaling level."""
    min_trades: int
    positive_expectancy: bool
    no_policy_violations: bool
    multiple_regimes_survived: bool
    drawdown_controlled: bool
    exit_brain_dominates: bool
    no_human_intervention: bool
    robust_to_loss_series: bool
    kill_switch_never_misused: bool


@dataclass
class ScalingActions:
    """Actions taken at a scaling level."""
    size_increase_pct: float
    leverage_change: str
    additional_actions: List[str]


@dataclass
class ScalingLevelSpec:
    """Specification for a scaling level."""
    level: ScalingLevel
    name: str
    emoji: str
    focus: str
    requirements: ScalingRequirements
    actions: ScalingActions
    description: str


# ═══════════════════════════════════════════════════════════════════════════════
# SCALING LEVELS DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

SCALING_LEVELS: Dict[ScalingLevel, ScalingLevelSpec] = {
    
    ScalingLevel.PROOF_MODE: ScalingLevelSpec(
        level=ScalingLevel.PROOF_MODE,
        name="Proof Mode",
        emoji="🔹",
        focus="Overlevelse",
        requirements=ScalingRequirements(
            min_trades=0,
            positive_expectancy=False,
            no_policy_violations=True,
            multiple_regimes_survived=False,
            drawdown_controlled=True,
            exit_brain_dominates=False,
            no_human_intervention=True,
            robust_to_loss_series=False,
            kill_switch_never_misused=True,
        ),
        actions=ScalingActions(
            size_increase_pct=0.0,
            leverage_change="Minimal",
            additional_actions=["Lav size", "Maks disiplin", "Fokus: overlevelse"],
        ),
        description="Startmodus - bevis at systemet fungerer uten å tape",
    ),
    
    ScalingLevel.CONFIRMED_EDGE: ScalingLevelSpec(
        level=ScalingLevel.CONFIRMED_EDGE,
        name="Confirmed Edge",
        emoji="🔹",
        focus="Edge-bekreftelse",
        requirements=ScalingRequirements(
            min_trades=30,  # 30-50 trades
            positive_expectancy=True,
            no_policy_violations=True,
            multiple_regimes_survived=False,
            drawdown_controlled=True,
            exit_brain_dominates=False,
            no_human_intervention=True,
            robust_to_loss_series=False,
            kill_switch_never_misused=True,
        ),
        actions=ScalingActions(
            size_increase_pct=15.0,  # +10-20%
            leverage_change="Samme",
            additional_actions=["+10-20% size", "Samme leverage"],
        ),
        description="Edge er bekreftet over minimum antall trades",
    ),
    
    ScalingLevel.STABLE_REGIME: ScalingLevelSpec(
        level=ScalingLevel.STABLE_REGIME,
        name="Stable Regime",
        emoji="🔹",
        focus="Regime-robusthet",
        requirements=ScalingRequirements(
            min_trades=100,
            positive_expectancy=True,
            no_policy_violations=True,
            multiple_regimes_survived=True,
            drawdown_controlled=True,
            exit_brain_dominates=True,
            no_human_intervention=True,
            robust_to_loss_series=True,
            kill_switch_never_misused=True,
        ),
        actions=ScalingActions(
            size_increase_pct=10.0,
            leverage_change="Eventuelt marginal økning",
            additional_actions=["+10% size", "Eventuelt marginal leverage"],
        ),
        description="Systemet har overlevd flere markedsfaser",
    ),
    
    ScalingLevel.SCALABLE_MODE: ScalingLevelSpec(
        level=ScalingLevel.SCALABLE_MODE,
        name="Scalable Mode",
        emoji="🔹",
        focus="Skalering",
        requirements=ScalingRequirements(
            min_trades=200,
            positive_expectancy=True,
            no_policy_violations=True,
            multiple_regimes_survived=True,
            drawdown_controlled=True,
            exit_brain_dominates=True,
            no_human_intervention=True,
            robust_to_loss_series=True,
            kill_switch_never_misused=True,
        ),
        actions=ScalingActions(
            size_increase_pct=0.0,  # No size increase per se
            leverage_change="Samme",
            additional_actions=[
                "Gradvis kapitalpåfyll",
                "Flere symbols",
                "IKKE høyere risiko per trade",
            ],
        ),
        description="Full skalering - flere symbols, ikke høyere risiko",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCALING RULES
# ═══════════════════════════════════════════════════════════════════════════════

SCALING_RULES = """
🚀 KAPITAL-ESKALERING REGLER
============================

❌ Hva fond IKKE gjør:
• Dobler size etter god uke
• Øker leverage pga selvtillit
• Antar at edge er "bevist" raskt

✅ Fondets eskalerings-prinsipp:
Skaler KUN etter stabilitet, aldri etter eufori

🛑 AUTOMATISK NEDSKALERING
Ved:
• Økende drawdown
• Regime-usikkerhet
• Edge-forfall

➡️ Systemet går ett nivå NED automatisk.

📌 Opp tar tid. Ned går fort.
"""

ANTI_SCALING_RULES = [
    "Dobler size etter god uke",
    "Øker leverage pga selvtillit",
    "Antar at edge er 'bevist' raskt",
    "Skalerer etter eufori",
    "Ignorerer drawdown",
]

AUTO_DOWNSCALE_TRIGGERS = [
    "Økende drawdown",
    "Regime-usikkerhet",
    "Edge-forfall",
    "Multiple consecutive losses",
    "Policy violations",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: DAY-0 HANDBOOK
# ═══════════════════════════════════════════════════════════════════════════════

class DayPhase(str, Enum):
    """Phases of Day-0."""
    BEFORE_MARKET = "before_market"
    DURING_MARKET = "during_market"
    AFTER_MARKET = "after_market"


@dataclass
class DayPhaseSpec:
    """Specification for a day phase."""
    phase: DayPhase
    time: str
    emoji: str
    title: str
    rules: List[str]
    forbidden: List[str]
    key_insight: str


# ═══════════════════════════════════════════════════════════════════════════════
# DAY-0 PHASES
# ═══════════════════════════════════════════════════════════════════════════════

DAY_0_PHASES: Dict[DayPhase, DayPhaseSpec] = {
    
    DayPhase.BEFORE_MARKET: DayPhaseSpec(
        phase=DayPhase.BEFORE_MARKET,
        time="🕘 FØR MARKED",
        emoji="🕘",
        title="DAY-0 START",
        rules=[
            "Les kun én ting: Pre-flight checklist",
            "Ingen strategiendringer",
            "Ingen forventninger",
        ],
        forbidden=[
            "Parameterendringer",
            "Strategi-tweaking",
            "Sette PnL-mål",
        ],
        key_insight="Målet er ikke å trade. Målet er å oppføre seg riktig.",
    ),
    
    DayPhase.DURING_MARKET: DayPhaseSpec(
        phase=DayPhase.DURING_MARKET,
        time="🕙 UNDER MARKED",
        emoji="🕙",
        title="HANDEL",
        rules=[
            "Systemet handler eller ikke",
            "Du observerer",
            "Ingen inngripen",
            "Noter kun brudd eller friksjon",
        ],
        forbidden=[
            "Manuell entry",
            "Manuell exit",
            "Override av noe slag",
            "Justere stops",
        ],
        key_insight="Du er tilskuer, ikke aktør.",
    ),
    
    DayPhase.AFTER_MARKET: DayPhaseSpec(
        phase=DayPhase.AFTER_MARKET,
        time="🕔 ETTER MARKED",
        emoji="🕔",
        title="EVALUERING",
        rules=[
            "Svar på tre spørsmål:",
            "1. Brøt systemet noen lover?",
            "2. Ville jeg overstyrt hvis jeg kunne?",
            "3. Stoppet systemet når det burde?",
        ],
        forbidden=[
            "Retrospektiv justering",
            "Blame-game",
            "Endre parametre basert på én dag",
        ],
        key_insight="📌 PnL er sekundært dag 0.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DAY-0 ABSOLUTE PROHIBITIONS
# ═══════════════════════════════════════════════════════════════════════════════

DAY_0_ABSOLUTE_PROHIBITIONS = [
    "Ingen justeringer",
    "Ingen 'forbedringer'",
    "Ingen ekstra trades",
    "Ingen forklaringer til deg selv",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DAY-0 SUCCESS CRITERIA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Day0SuccessCriteria:
    """Success criteria for Day-0."""
    rule_violations: int = 0
    overrides: int = 0
    stress_level: int = 0
    pnl_relevant: bool = False


DAY_0_SUCCESS = """
🧭 DAY-0 SUKSESS =

• 0 regelbrudd
• 0 overrides
• 0 stress
• +/– PnL irrelevant

Hvis systemet overlever dag 0, kan det overleve år.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DAY-0 THREE QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

DAY_0_THREE_QUESTIONS = [
    "Brøt systemet noen lover?",
    "Ville jeg overstyrt hvis jeg kunne?",
    "Stoppet systemet når det burde?",
]


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED MANAGER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class StressTestScalingManager:
    """
    Manager for stress testing, capital scaling, and Day-0 operations.
    
    Combines:
    1. 10-loss stress test
    2. Capital scaling levels
    3. Day-0 handbook
    """
    
    def __init__(self):
        self._stress_test = STRESS_TEST_SEQUENCE
        self._scaling_levels = SCALING_LEVELS
        self._day_0_phases = DAY_0_PHASES
        
        # State
        self._current_scaling_level = ScalingLevel.PROOF_MODE
        self._consecutive_losses = 0
        self._total_trades = 0
        self._policy_violations = 0
        self._override_attempts = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRESS TEST
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run_stress_test(
        self,
        initial_capital: float = 100000.0,
        max_loss_per_trade_pct: float = 2.0,
    ) -> StressTestResult:
        """
        Run the 10-loss stress test.
        
        Returns detailed result of the test.
        """
        trades = []
        capital = initial_capital
        rules_changed = 0
        override_attempts = self._override_attempts
        
        for template in self._stress_test:
            # Calculate actual loss based on size multiplier
            if template.can_trade and template.size_multiplier > 0:
                loss_pct = max_loss_per_trade_pct * template.size_multiplier
                capital = capital * (1 - loss_pct / 100)
            
            trade = StressTestTrade(
                trade_number=template.trade_number,
                result=template.result,
                system_response=template.system_response,
                description=template.description,
                capital_remaining_pct=(capital / initial_capital) * 100,
                size_multiplier=template.size_multiplier,
                can_trade=template.can_trade,
                warning_triggered=template.warning_triggered,
            )
            trades.append(trade)
        
        # Evaluate result
        final_capital_pct = (capital / initial_capital) * 100
        
        failure_reasons = []
        if final_capital_pct < 90.0:
            failure_reasons.append(f"Capital below 90%: {final_capital_pct:.1f}%")
        if rules_changed > 0:
            failure_reasons.append(f"Rules changed: {rules_changed}")
        if override_attempts > 0:
            failure_reasons.append(f"Override attempts: {override_attempts}")
        
        passed = len(failure_reasons) == 0
        
        return StressTestResult(
            passed=passed,
            capital_remaining_pct=final_capital_pct,
            rules_changed=rules_changed,
            override_attempts=override_attempts,
            system_mental_state="Kaldt" if passed else "Kompromittert",
            trades=trades,
            failure_reasons=failure_reasons,
        )
    
    def get_stress_test_response(self, consecutive_losses: int) -> StressTestResponse:
        """Get the appropriate system response for consecutive losses."""
        if consecutive_losses < 1:
            return StressTestResponse.NORMAL
        if consecutive_losses > len(self._stress_test):
            return StressTestResponse.SHADOW_ONLY
        return self._stress_test[consecutive_losses - 1].system_response
    
    def record_loss(self) -> Tuple[StressTestResponse, str]:
        """Record a loss and get the system response."""
        self._consecutive_losses += 1
        self._total_trades += 1
        
        response = self.get_stress_test_response(self._consecutive_losses)
        
        if self._consecutive_losses < len(self._stress_test):
            description = self._stress_test[self._consecutive_losses - 1].description
        else:
            description = "Shadow-only mode"
        
        return response, description
    
    def record_win(self):
        """Record a win (resets consecutive losses)."""
        self._consecutive_losses = 0
        self._total_trades += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CAPITAL SCALING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_current_scaling_level(self) -> ScalingLevelSpec:
        """Get current scaling level specification."""
        return self._scaling_levels[self._current_scaling_level]
    
    def can_scale_up(
        self,
        total_trades: int,
        expectancy: float,
        policy_violations: int,
        regimes_survived: int,
        max_drawdown_pct: float,
        human_interventions: int,
        loss_series_survived: bool,
        kill_switch_misused: bool,
    ) -> Tuple[bool, ScalingLevel, List[str]]:
        """
        Check if system can scale up to next level.
        
        Returns:
            (can_scale, target_level, missing_requirements)
        """
        current = self._current_scaling_level
        if current == ScalingLevel.SCALABLE_MODE:
            return False, current, ["Already at maximum scaling level"]
        
        next_level = ScalingLevel(current.value + 1)
        next_spec = self._scaling_levels[next_level]
        reqs = next_spec.requirements
        
        missing = []
        
        if total_trades < reqs.min_trades:
            missing.append(f"Need {reqs.min_trades} trades, have {total_trades}")
        
        if reqs.positive_expectancy and expectancy <= 0:
            missing.append(f"Need positive expectancy, have {expectancy:.2f}")
        
        if reqs.no_policy_violations and policy_violations > 0:
            missing.append(f"Policy violations: {policy_violations}")
        
        if reqs.multiple_regimes_survived and regimes_survived < 3:
            missing.append(f"Need 3+ regimes survived, have {regimes_survived}")
        
        if reqs.drawdown_controlled and max_drawdown_pct > 15:
            missing.append(f"Drawdown {max_drawdown_pct:.1f}% > 15%")
        
        if reqs.no_human_intervention and human_interventions > 0:
            missing.append(f"Human interventions: {human_interventions}")
        
        if reqs.robust_to_loss_series and not loss_series_survived:
            missing.append("Not robust to loss series")
        
        if reqs.kill_switch_never_misused and kill_switch_misused:
            missing.append("Kill-switch was misused")
        
        can_scale = len(missing) == 0
        return can_scale, next_level, missing
    
    def scale_up(self) -> bool:
        """Scale up to next level if possible."""
        if self._current_scaling_level == ScalingLevel.SCALABLE_MODE:
            return False
        self._current_scaling_level = ScalingLevel(self._current_scaling_level.value + 1)
        return True
    
    def should_scale_down(
        self,
        drawdown_increasing: bool,
        regime_uncertain: bool,
        edge_decaying: bool,
    ) -> Tuple[bool, List[str]]:
        """
        Check if system should automatically scale down.
        
        Returns:
            (should_scale_down, reasons)
        """
        reasons = []
        
        if drawdown_increasing:
            reasons.append("Økende drawdown")
        if regime_uncertain:
            reasons.append("Regime-usikkerhet")
        if edge_decaying:
            reasons.append("Edge-forfall")
        
        return len(reasons) > 0, reasons
    
    def scale_down(self) -> bool:
        """Scale down to previous level."""
        if self._current_scaling_level == ScalingLevel.PROOF_MODE:
            return False
        self._current_scaling_level = ScalingLevel(self._current_scaling_level.value - 1)
        return True
    
    def get_size_multiplier(self) -> float:
        """Get current size multiplier based on scaling level."""
        level = self._current_scaling_level
        
        # Base multipliers
        multipliers = {
            ScalingLevel.PROOF_MODE: 0.25,      # 25% of target size
            ScalingLevel.CONFIRMED_EDGE: 0.50,   # 50%
            ScalingLevel.STABLE_REGIME: 0.75,    # 75%
            ScalingLevel.SCALABLE_MODE: 1.0,     # 100%
        }
        
        return multipliers.get(level, 0.25)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DAY-0 HANDBOOK
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_day_0_phase(self, phase: DayPhase) -> DayPhaseSpec:
        """Get Day-0 phase specification."""
        return self._day_0_phases[phase]
    
    def get_day_0_checklist(self) -> Dict[DayPhase, List[str]]:
        """Get Day-0 checklist by phase."""
        return {
            phase: spec.rules
            for phase, spec in self._day_0_phases.items()
        }
    
    def evaluate_day_0(
        self,
        rule_violations: int,
        overrides: int,
        stress_level: int,  # 0-10
    ) -> Tuple[bool, str]:
        """
        Evaluate if Day-0 was successful.
        
        Returns:
            (success, assessment)
        """
        success = (
            rule_violations == 0 and
            overrides == 0 and
            stress_level == 0
        )
        
        if success:
            assessment = "✅ DAY-0 SUKSESS: Systemet kan overleve år."
        else:
            issues = []
            if rule_violations > 0:
                issues.append(f"{rule_violations} regelbrudd")
            if overrides > 0:
                issues.append(f"{overrides} overrides")
            if stress_level > 0:
                issues.append(f"Stress level {stress_level}/10")
            assessment = f"❌ DAY-0 FEIL: {', '.join(issues)}"
        
        return success, assessment
    
    def get_day_0_three_questions(self) -> List[str]:
        """Get the three Day-0 evaluation questions."""
        return DAY_0_THREE_QUESTIONS.copy()
    
    def record_override_attempt(self):
        """Record an override attempt."""
        self._override_attempts += 1
        self._policy_violations += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRINT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_stress_test(self):
        """Print the stress test sequence."""
        print("\n" + "=" * 80)
        print("🧨 STRESS-TEST: 10 TAP PÅ RAD")
        print("Overlever systemet uten å endre personlighet?")
        print("=" * 80)
        
        print("\n🎯 Mål: Forbli disiplinert, liten og rasjonell gjennom tapene.")
        
        print("\n📉 SEKVENS:")
        print("-" * 60)
        print(f"{'Trade':<8} {'Resultat':<10} {'Respons':<20} {'Beskrivelse'}")
        print("-" * 60)
        
        for trade in self._stress_test:
            warning = "⚠️" if trade.warning_triggered else "  "
            print(f"{warning} #{trade.trade_number:<5} {trade.result:<10} {trade.system_response.value:<20} {trade.description}")
        
        print(PSYCHOLOGICAL_REALITY)
        print(STRESS_TEST_PASS_CRITERIA)
    
    def print_scaling_levels(self):
        """Print the scaling levels."""
        print("\n" + "=" * 80)
        print("🚀 KAPITAL-ESKALERING ETTER SUKSESS")
        print("Hvordan vokse uten å endre DNA")
        print("=" * 80)
        
        print(SCALING_RULES)
        
        print("\n🪜 ESKALERINGSSTIGEN:")
        print("-" * 60)
        
        for level, spec in self._scaling_levels.items():
            print(f"\n{spec.emoji} Nivå {level.value} — {spec.name}")
            print(f"   Fokus: {spec.focus}")
            print(f"   Krav: {spec.requirements.min_trades}+ trades, ", end="")
            if spec.requirements.positive_expectancy:
                print("positiv expectancy, ", end="")
            if spec.requirements.multiple_regimes_survived:
                print("flere regimer, ", end="")
            print()
            print(f"   Tiltak: {', '.join(spec.actions.additional_actions)}")
    
    def print_day_0_handbook(self):
        """Print the Day-0 handbook."""
        print("\n" + "=" * 80)
        print("📘 DAY-0 HANDBOOK")
        print("Hvordan fondet faktisk brukes – første dag")
        print("=" * 80)
        
        for phase, spec in self._day_0_phases.items():
            print(f"\n{spec.time}")
            print("-" * 40)
            print(f"Regler:")
            for rule in spec.rules:
                print(f"   • {rule}")
            print(f"\n📌 {spec.key_insight}")
        
        print("\n🚫 ABSOLUTTE FORBUD (DAY-0):")
        for prohibition in DAY_0_ABSOLUTE_PROHIBITIONS:
            print(f"   • {prohibition}")
        
        print(DAY_0_SUCCESS)
    
    def print_full(self):
        """Print everything."""
        self.print_stress_test()
        self.print_scaling_levels()
        self.print_day_0_handbook()
        print(FINAL_CONCLUSION)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════

FINAL_CONCLUSION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 SLUTT-KONKLUSJON                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    Du har nå:                                                                ║
║                                                                              ║
║    ✔️ En realistisk tap-stress-test                                          ║
║    ✔️ En moden kapital-eskaleringsmodell                                     ║
║    ✔️ Et operativt Day-0-regelverk                                           ║
║                                                                              ║
║    Dette er det som skiller et system som varer                              ║
║    fra et som brenner sterkt og dør.                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_stress_test_manager: Optional[StressTestScalingManager] = None


def get_stress_test_manager() -> StressTestScalingManager:
    """Get singleton stress test manager instance."""
    global _stress_test_manager
    if _stress_test_manager is None:
        _stress_test_manager = StressTestScalingManager()
    return _stress_test_manager


def reset_stress_test_manager() -> StressTestScalingManager:
    """Reset and get fresh manager instance."""
    global _stress_test_manager
    _stress_test_manager = StressTestScalingManager()
    return _stress_test_manager


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    manager = get_stress_test_manager()
    manager.print_full()
    
    # Run stress test
    print("\n" + "=" * 40)
    print("Running Stress Test...")
    print("=" * 40)
    
    result = manager.run_stress_test(
        initial_capital=100000.0,
        max_loss_per_trade_pct=2.0,
    )
    
    print(f"\nTest passed: {result.passed}")
    print(f"Capital remaining: {result.capital_remaining_pct:.1f}%")
    print(f"System state: {result.system_mental_state}")
    
    if not result.passed:
        print(f"Failure reasons: {result.failure_reasons}")
