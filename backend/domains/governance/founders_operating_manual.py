"""
📊 FOUNDERS OPERATING MANUAL & FINAL FRAMEWORK
===============================================
Fire avsluttende institusjonelle komponenter:

1️⃣ 100-Trade Simulering (Statistisk fordeling)
2️⃣ No-Trade Days Kalender
3️⃣ Investor-Style Risk Disclosure
4️⃣ Founders Operating Manual

Dette er ferdig designet hedge-fond-arkitektur – ikke en bot.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
from datetime import datetime, date
import logging
import random
import math

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: 100-TRADE SIMULERING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationParameters:
    """Parameters for 100-trade simulation."""
    win_rate: float = 0.45           # 45%
    reward_risk_ratio: float = 1.6    # 1.6:1
    risk_per_trade: float = 0.02      # 2% konstant
    exit_dominance: bool = True
    policy_violations: int = 0


@dataclass
class TradeResult:
    """Result of a single trade."""
    trade_number: int
    outcome: str  # "WIN" or "LOSS"
    pnl_r: float  # In R-multiples
    cumulative_r: float


@dataclass
class LossSeriesStats:
    """Statistics about loss series."""
    three_in_row: int      # Normalt
    five_in_row: int       # Forventet
    seven_in_row: int      # Mulig
    ten_in_row: int        # Sjeldent
    max_consecutive: int


@dataclass
class SimulationResult:
    """Result of 100-trade simulation."""
    total_trades: int
    winners: int
    losers: int
    total_r: float
    expectancy_per_trade: float
    loss_series: LossSeriesStats
    trades: List[TradeResult]
    profit_clusters: int
    flat_periods: int


# ═══════════════════════════════════════════════════════════════════════════════
# 100-TRADE FORDELING (TYPISK)
# ═══════════════════════════════════════════════════════════════════════════════

TRADE_100_DISTRIBUTION = """
📈 FORDELING OVER 100 TRADES (TYPISK)
=====================================

Forutsetninger (konstitusjonelt realistiske):
• Win-rate: 45%
• Avg win : avg loss = 1.6 : 1
• Risiko per trade: konstant
• Exit-dominans aktiv
• Ingen regelbrudd

Fordeling:
• 45 vinnere
• 55 tapere
• Netto expectancy: POSITIV (selv med flere tap enn gevinster)

Expectancy = (0.45 × 1.6) + (0.55 × -1.0)
           = 0.72 - 0.55
           = 0.17R per trade

Over 100 trades = +17R forventet
"""

LOSS_SERIES_REALITY = """
📉 TAP-SERIER (UUNNVÆRLIG SANNHET)
==================================

• 3 tap på rad:  NORMALT
• 5 tap på rad:  FORVENTET
• 7 tap på rad:  MULIG
• 10 tap på rad: SJELDENT, men må overleves

👉 Systemet er designet for dette, ikke overrasket av det.
"""

KEY_INSIGHTS_100_TRADES = """
🧠 VIKTIGSTE INNSIKT
====================

• Profit kommer i KLYNGER
• Lange flate perioder er NORMALT
• Exit-kvalitet avgjør om systemet overlever tap-serier

📌 Et fond bygges for FORDELINGEN – ikke for enkelthandelen.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: NO-TRADE DAYS KALENDER
# ═══════════════════════════════════════════════════════════════════════════════

class NoTradeSeverity(str, Enum):
    """Severity levels for no-trade days."""
    ABSOLUTE = "absolute"          # 🔴 Absolutt - FLAT. Punktum.
    CONDITIONAL = "conditional"    # 🟠 Kondisjonell - Observer-only
    HUMAN = "human"                # 🔵 Menneskelig - Beskytt mot deg selv


@dataclass
class NoTradeCondition:
    """A condition that triggers no-trade."""
    id: str
    name: str
    severity: NoTradeSeverity
    description: str
    action: str
    emoji: str


# ═══════════════════════════════════════════════════════════════════════════════
# NO-TRADE CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════

NO_TRADE_CONDITIONS: List[NoTradeCondition] = [
    # 🔴 ABSOLUTE NO-TRADE DAYS
    NoTradeCondition(
        id="max_daily_loss",
        name="Maks Daglig Tap",
        severity=NoTradeSeverity.ABSOLUTE,
        description="Etter maks daglig tap nådd",
        action="Systemet er FLAT. Punktum.",
        emoji="🔴",
    ),
    NoTradeCondition(
        id="data_inconsistency",
        name="Datainkonsistens",
        severity=NoTradeSeverity.ABSOLUTE,
        description="Under datainkonsistens mellom kilder",
        action="Systemet er FLAT. Punktum.",
        emoji="🔴",
    ),
    NoTradeCondition(
        id="exchange_instability",
        name="Exchange-Instabilitet",
        severity=NoTradeSeverity.ABSOLUTE,
        description="Under exchange-instabilitet eller vedlikehold",
        action="Systemet er FLAT. Punktum.",
        emoji="🔴",
    ),
    NoTradeCondition(
        id="systemic_crisis",
        name="Systemisk Krise",
        severity=NoTradeSeverity.ABSOLUTE,
        description="Ved systemisk krise / flash events",
        action="Systemet er FLAT. Punktum.",
        emoji="🔴",
    ),
    
    # 🟠 CONDITIONAL NO-TRADE DAYS
    NoTradeCondition(
        id="extreme_funding",
        name="Ekstrem Funding",
        severity=NoTradeSeverity.CONDITIONAL,
        description="Ekstrem funding rate (P99)",
        action="Observer-only modus",
        emoji="🟠",
    ),
    NoTradeCondition(
        id="regime_transition",
        name="Regime-Overgang",
        severity=NoTradeSeverity.CONDITIONAL,
        description="Regime-overgang uten konsensus",
        action="Observer-only modus",
        emoji="🟠",
    ),
    NoTradeCondition(
        id="low_liquidity",
        name="Lav Likviditet",
        severity=NoTradeSeverity.CONDITIONAL,
        description="Uvanlig lav likviditet",
        action="Observer-only modus",
        emoji="🟠",
    ),
    NoTradeCondition(
        id="consecutive_losses",
        name="Flere Tap",
        severity=NoTradeSeverity.CONDITIONAL,
        description="Etter flere tap i kort tid",
        action="Observer-only modus",
        emoji="🟠",
    ),
    
    # 🔵 HUMAN NO-TRADE DAYS
    NoTradeCondition(
        id="override_attempt",
        name="Override-Forsøk",
        severity=NoTradeSeverity.HUMAN,
        description="Etter manuell override-forsøk",
        action="Systemet beskytter deg mot deg selv",
        emoji="🔵",
    ),
    NoTradeCondition(
        id="emotional_stress",
        name="Emosjonelt Stress",
        severity=NoTradeSeverity.HUMAN,
        description="Etter emosjonelt stress identifisert",
        action="Systemet beskytter deg mot deg selv",
        emoji="🔵",
    ),
    NoTradeCondition(
        id="unexpected_loss",
        name="Uventet Tap",
        severity=NoTradeSeverity.HUMAN,
        description="Etter større tap enn forventet",
        action="Systemet beskytter deg mot deg selv",
        emoji="🔵",
    ),
    NoTradeCondition(
        id="must_trade_urge",
        name="'Må Trade' Følelse",
        severity=NoTradeSeverity.HUMAN,
        description="Når du føler du 'må' trade",
        action="Systemet beskytter deg mot deg selv",
        emoji="🔵",
    ),
]

NO_TRADE_GOLDEN_RULE = """
🧠 GULLREGEL
============

No-trade days er KAPITALBESKYTTELSE, ikke tapte muligheter.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: INVESTOR-STYLE RISK DISCLOSURE
# ═══════════════════════════════════════════════════════════════════════════════

RISK_DISCLOSURE_SHORT = """
📄 RISIKOFORKLARING (KORTVERSJON)
=================================

Dette systemet handler i volatile derivatmarkeder (crypto futures),
som innebærer betydelig risiko, inkludert – men ikke begrenset til:

• Markedsrisiko
• Likviditetsrisiko
• Motpartsrisiko
• Teknologisk risiko
• Systemisk risiko

Tidligere resultater, simuleringer eller hypotetiske analyser
er IKKE en garanti for fremtidig avkastning.

Systemet er designet for kapitalbevaring og risikokontroll,
men kan oppleve perioder med tap, drawdowns og inaktivitet.
"""

FUND_DOES_NOT_PROMISE = """
❌ HVA FONDET IKKE LOVER
========================

• Kontinuerlig avkastning
• Lav volatilitet
• Ingen tap
• Perfekt timing
"""

FUND_EXPLICITLY_PRIORITIZES = """
✅ HVA FONDET EKSPLISITT PRIORITERER
====================================

• Overlevelse
• Disiplin
• Transparens
• Kontrollerbar risiko

📌 Dette er et RISIKOSTYRT SYSTEM – ikke en profittmaskin.
"""

FULL_RISK_DISCLOSURE = f"""
{'='*80}
                    RISK DISCLOSURE STATEMENT
{'='*80}

{RISK_DISCLOSURE_SHORT}

{FUND_DOES_NOT_PROMISE}

{FUND_EXPLICITLY_PRIORITIZES}

{'='*80}
"""


@dataclass
class RiskType:
    """A type of risk the system faces."""
    name: str
    description: str
    mitigation: str


RISK_TYPES: List[RiskType] = [
    RiskType(
        name="Markedsrisiko",
        description="Uforutsigbare prisbevegelser",
        mitigation="Position sizing, stop-loss, drawdown limits",
    ),
    RiskType(
        name="Likviditetsrisiko",
        description="Manglende evne til å entre/exit posisjoner",
        mitigation="Likviditetskrav, size limits, spread monitoring",
    ),
    RiskType(
        name="Motpartsrisiko",
        description="Exchange failures eller insolvens",
        mitigation="Multi-exchange, kapitalsplitting",
    ),
    RiskType(
        name="Teknologisk risiko",
        description="System failures, connectivity issues",
        mitigation="Fail-closed design, redundancy, monitoring",
    ),
    RiskType(
        name="Systemisk risiko",
        description="Markedsomfattende krise",
        mitigation="Black swan playbook, kill-switch, flat mode",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: FOUNDERS OPERATING MANUAL
# ═══════════════════════════════════════════════════════════════════════════════

class ManualPart(str, Enum):
    """Parts of the Founders Operating Manual."""
    IDENTITY = "I"
    GRUNNLOVER = "II"
    ARCHITECTURE = "III"
    OPERATION = "IV"
    STRESS_SCALING = "V"
    DAY_LIFECYCLE = "VI"


@dataclass
class ManualSection:
    """A section in the Founders Operating Manual."""
    part: ManualPart
    title: str
    components: List[str]
    key_principle: str


# ═══════════════════════════════════════════════════════════════════════════════
# FOUNDERS OPERATING MANUAL STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

FOUNDERS_MANUAL: Dict[ManualPart, ManualSection] = {
    
    ManualPart.IDENTITY: ManualSection(
        part=ManualPart.IDENTITY,
        title="IDENTITET",
        components=[
            "Formål: Overleve først, vokse deretter",
            "Filosofi: Exit > Entry, Risk > Profit",
            "Prinsipp: System > Menneske",
        ],
        key_principle="Overlevelse er suksess",
    ),
    
    ManualPart.GRUNNLOVER: ManualSection(
        part=ManualPart.GRUNNLOVER,
        title="GRUNNLOVER",
        components=[
            "15 grunnlover (policy)",
            "Absolutte forbud",
            "Beslutningshierarki",
        ],
        key_principle="Lover kan ikke brytes, bare tolkes",
    ),
    
    ManualPart.ARCHITECTURE: ManualSection(
        part=ManualPart.ARCHITECTURE,
        title="SYSTEMARKITEKTUR",
        components=[
            "Policy Engine",
            "Risk Kernel (Fail-Closed)",
            "Exit / Harvest Brain",
            "Data Integrity Sentinel",
            "Human Override Lock",
            "Audit & Ledger",
        ],
        key_principle="Arkitekturen beskytter seg selv",
    ),
    
    ManualPart.OPERATION: ManualSection(
        part=ManualPart.OPERATION,
        title="OPERASJON",
        components=[
            "Pre-flight checklist",
            "No-trade kalender",
            "Kill-switch manifest",
            "Restart-protokoll",
        ],
        key_principle="Operasjon følger protokoll, aldri intuisjon",
    ),
    
    ManualPart.STRESS_SCALING: ManualSection(
        part=ManualPart.STRESS_SCALING,
        title="STRESS & SKALERING",
        components=[
            "10-tap-stress-test",
            "100-trade fordeling",
            "Kapital-eskalering i nivåer",
            "Automatisk nedskalering",
        ],
        key_principle="Opp tar tid. Ned går fort.",
    ),
    
    ManualPart.DAY_LIFECYCLE: ManualSection(
        part=ManualPart.DAY_LIFECYCLE,
        title="DAY-0 TIL DAY-∞",
        components=[
            "Day-0 handbook",
            "Live-dag rutiner",
            "Endringskontroll",
            "Shadow-testing før alt nytt",
        ],
        key_principle="Ingen endring uten shadow-test",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SLUTTORD
# ═══════════════════════════════════════════════════════════════════════════════

SLUTTORD = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 SLUTTORD (SANNHETEN)                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║    Du har nå:                                                                ║
║                                                                              ║
║    • Et system som FORVENTER tap                                             ║
║    • Et rammeverk som OVERLEVER dem                                          ║
║    • En struktur som kan SKALERES uten å endre DNA                           ║
║                                                                              ║
║    Dette er ferdig designet hedge-fond-arkitektur – ikke en bot.             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE MANAGER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class FoundersOperatingManual:
    """
    Complete Founders Operating Manual and Final Framework.
    
    Combines:
    1. 100-Trade Simulation
    2. No-Trade Days Calendar
    3. Risk Disclosure
    4. Founders Manual
    """
    
    def __init__(self):
        self._params = SimulationParameters()
        self._no_trade_conditions = NO_TRADE_CONDITIONS
        self._risk_types = RISK_TYPES
        self._manual = FOUNDERS_MANUAL
        
        # State
        self._active_no_trade_conditions: List[str] = []
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 1: 100-TRADE SIMULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def calculate_expectancy(
        self,
        win_rate: float = 0.45,
        reward_risk_ratio: float = 1.6,
    ) -> float:
        """
        Calculate expectancy per trade in R-multiples.
        
        Formula: E = (win_rate × RR) + ((1-win_rate) × -1)
        """
        return (win_rate * reward_risk_ratio) + ((1 - win_rate) * -1)
    
    def simulate_100_trades(
        self,
        win_rate: float = 0.45,
        reward_risk_ratio: float = 1.6,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate 100 trades with realistic distribution.
        
        Returns detailed statistics including loss series.
        """
        if seed is not None:
            random.seed(seed)
        
        trades = []
        cumulative_r = 0.0
        
        # Track loss series
        current_loss_streak = 0
        max_loss_streak = 0
        loss_streaks: List[int] = []
        
        # Track profit clusters and flat periods
        profit_clusters = 0
        flat_periods = 0
        in_profit_cluster = False
        consecutive_flat = 0
        
        for i in range(100):
            # Determine outcome based on win rate
            is_win = random.random() < win_rate
            
            if is_win:
                pnl_r = reward_risk_ratio
                outcome = "WIN"
                
                # End loss streak
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
                
                # Track profit cluster
                if not in_profit_cluster:
                    profit_clusters += 1
                    in_profit_cluster = True
                
                consecutive_flat = 0
            else:
                pnl_r = -1.0
                outcome = "LOSS"
                
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                
                in_profit_cluster = False
                consecutive_flat += 1
                
                if consecutive_flat >= 5:
                    flat_periods += 1
                    consecutive_flat = 0
            
            cumulative_r += pnl_r
            
            trades.append(TradeResult(
                trade_number=i + 1,
                outcome=outcome,
                pnl_r=pnl_r,
                cumulative_r=cumulative_r,
            ))
        
        # Final loss streak
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        # Count loss series
        loss_series = LossSeriesStats(
            three_in_row=sum(1 for s in loss_streaks if s >= 3),
            five_in_row=sum(1 for s in loss_streaks if s >= 5),
            seven_in_row=sum(1 for s in loss_streaks if s >= 7),
            ten_in_row=sum(1 for s in loss_streaks if s >= 10),
            max_consecutive=max_loss_streak,
        )
        
        winners = sum(1 for t in trades if t.outcome == "WIN")
        losers = 100 - winners
        
        return SimulationResult(
            total_trades=100,
            winners=winners,
            losers=losers,
            total_r=cumulative_r,
            expectancy_per_trade=cumulative_r / 100,
            loss_series=loss_series,
            trades=trades,
            profit_clusters=profit_clusters,
            flat_periods=flat_periods,
        )
    
    def get_expected_distribution(self) -> Dict[str, Any]:
        """Get expected distribution for 100 trades."""
        exp = self.calculate_expectancy()
        return {
            "win_rate": 0.45,
            "reward_risk_ratio": 1.6,
            "expected_winners": 45,
            "expected_losers": 55,
            "expectancy_per_trade": exp,
            "expected_total_r": exp * 100,
            "loss_series_3_normal": True,
            "loss_series_5_expected": True,
            "loss_series_7_possible": True,
            "loss_series_10_rare_but_survivable": True,
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 2: NO-TRADE DAYS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_no_trade_conditions(
        self,
        severity: Optional[NoTradeSeverity] = None,
    ) -> List[NoTradeCondition]:
        """Get no-trade conditions, optionally filtered by severity."""
        if severity is None:
            return self._no_trade_conditions
        return [c for c in self._no_trade_conditions if c.severity == severity]
    
    def check_no_trade(
        self,
        daily_loss_hit: bool = False,
        data_inconsistent: bool = False,
        exchange_unstable: bool = False,
        systemic_crisis: bool = False,
        extreme_funding: bool = False,
        regime_uncertain: bool = False,
        low_liquidity: bool = False,
        consecutive_losses: int = 0,
        override_attempted: bool = False,
        emotional_stress: bool = False,
        unexpected_loss: bool = False,
        must_trade_urge: bool = False,
    ) -> Tuple[bool, NoTradeSeverity, List[str]]:
        """
        Check if today should be a no-trade day.
        
        Returns:
            (is_no_trade, highest_severity, triggered_conditions)
        """
        triggered = []
        highest_severity = None
        
        # 🔴 ABSOLUTE
        if daily_loss_hit:
            triggered.append("max_daily_loss")
            highest_severity = NoTradeSeverity.ABSOLUTE
        if data_inconsistent:
            triggered.append("data_inconsistency")
            highest_severity = NoTradeSeverity.ABSOLUTE
        if exchange_unstable:
            triggered.append("exchange_instability")
            highest_severity = NoTradeSeverity.ABSOLUTE
        if systemic_crisis:
            triggered.append("systemic_crisis")
            highest_severity = NoTradeSeverity.ABSOLUTE
        
        # 🟠 CONDITIONAL (only if no absolute)
        if highest_severity != NoTradeSeverity.ABSOLUTE:
            if extreme_funding:
                triggered.append("extreme_funding")
                highest_severity = NoTradeSeverity.CONDITIONAL
            if regime_uncertain:
                triggered.append("regime_transition")
                highest_severity = NoTradeSeverity.CONDITIONAL
            if low_liquidity:
                triggered.append("low_liquidity")
                highest_severity = NoTradeSeverity.CONDITIONAL
            if consecutive_losses >= 3:
                triggered.append("consecutive_losses")
                highest_severity = NoTradeSeverity.CONDITIONAL
        
        # 🔵 HUMAN (only if no absolute or conditional)
        if highest_severity is None:
            if override_attempted:
                triggered.append("override_attempt")
                highest_severity = NoTradeSeverity.HUMAN
            if emotional_stress:
                triggered.append("emotional_stress")
                highest_severity = NoTradeSeverity.HUMAN
            if unexpected_loss:
                triggered.append("unexpected_loss")
                highest_severity = NoTradeSeverity.HUMAN
            if must_trade_urge:
                triggered.append("must_trade_urge")
                highest_severity = NoTradeSeverity.HUMAN
        
        is_no_trade = len(triggered) > 0
        
        # Store active conditions
        self._active_no_trade_conditions = triggered
        
        return is_no_trade, highest_severity, triggered
    
    def get_no_trade_action(self, severity: NoTradeSeverity) -> str:
        """Get the action for a no-trade severity level."""
        actions = {
            NoTradeSeverity.ABSOLUTE: "Systemet er FLAT. Punktum.",
            NoTradeSeverity.CONDITIONAL: "Observer-only modus",
            NoTradeSeverity.HUMAN: "Systemet beskytter deg mot deg selv",
        }
        return actions.get(severity, "Unknown")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 3: RISK DISCLOSURE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_risk_disclosure(self) -> str:
        """Get the full risk disclosure statement."""
        return FULL_RISK_DISCLOSURE
    
    def get_risk_types(self) -> List[RiskType]:
        """Get all risk types with mitigations."""
        return self._risk_types
    
    def get_fund_promises(self) -> Dict[str, List[str]]:
        """Get what the fund does and doesn't promise."""
        return {
            "does_not_promise": [
                "Kontinuerlig avkastning",
                "Lav volatilitet",
                "Ingen tap",
                "Perfekt timing",
            ],
            "explicitly_prioritizes": [
                "Overlevelse",
                "Disiplin",
                "Transparens",
                "Kontrollerbar risiko",
            ],
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PART 4: FOUNDERS MANUAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_manual_part(self, part: ManualPart) -> ManualSection:
        """Get a specific part of the Founders Manual."""
        return self._manual[part]
    
    def get_full_manual(self) -> Dict[ManualPart, ManualSection]:
        """Get the complete Founders Manual."""
        return self._manual
    
    def get_manual_table_of_contents(self) -> List[Tuple[str, str]]:
        """Get table of contents for the manual."""
        return [
            (f"DEL {part.value}", section.title)
            for part, section in self._manual.items()
        ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRINT METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_100_trade_simulation(self):
        """Print 100-trade simulation overview."""
        print("\n" + "=" * 80)
        print("📊 SIMULERING: 100 TRADES (STATISTISK)")
        print("Hvordan systemet oppfører seg over tid")
        print("=" * 80)
        
        print(TRADE_100_DISTRIBUTION)
        print(LOSS_SERIES_REALITY)
        print(KEY_INSIGHTS_100_TRADES)
    
    def print_no_trade_calendar(self):
        """Print no-trade days calendar."""
        print("\n" + "=" * 80)
        print("📅 NO-TRADE DAYS KALENDER")
        print("Profesjonell disiplin = vite når man IKKE skal trade")
        print("=" * 80)
        
        print("\n🔴 ABSOLUTTE NO-TRADE DAGER")
        print("-" * 40)
        for c in self.get_no_trade_conditions(NoTradeSeverity.ABSOLUTE):
            print(f"  • {c.name}: {c.description}")
        print(f"  ➡️ {self.get_no_trade_action(NoTradeSeverity.ABSOLUTE)}")
        
        print("\n🟠 KONDISJONELLE NO-TRADE DAGER")
        print("-" * 40)
        for c in self.get_no_trade_conditions(NoTradeSeverity.CONDITIONAL):
            print(f"  • {c.name}: {c.description}")
        print(f"  ➡️ {self.get_no_trade_action(NoTradeSeverity.CONDITIONAL)}")
        
        print("\n🔵 MENNESKELIGE NO-TRADE DAGER")
        print("-" * 40)
        for c in self.get_no_trade_conditions(NoTradeSeverity.HUMAN):
            print(f"  • {c.name}: {c.description}")
        print(f"  ➡️ {self.get_no_trade_action(NoTradeSeverity.HUMAN)}")
        
        print(NO_TRADE_GOLDEN_RULE)
    
    def print_risk_disclosure(self):
        """Print risk disclosure."""
        print("\n" + "=" * 80)
        print("📄 INVESTOR-STYLE RISK DISCLOSURE")
        print("Ærlig, profesjonell, institusjonell")
        print("=" * 80)
        
        print(FULL_RISK_DISCLOSURE)
    
    def print_founders_manual(self):
        """Print the Founders Operating Manual."""
        print("\n" + "=" * 80)
        print("📘 FOUNDERS OPERATING MANUAL")
        print("Alt samlet – hvordan fondet faktisk drives")
        print("=" * 80)
        
        for part, section in self._manual.items():
            print(f"\nDEL {part.value} — {section.title}")
            print("-" * 40)
            for component in section.components:
                print(f"  • {component}")
            print(f"\n  📌 Prinsipp: {section.key_principle}")
    
    def print_full(self):
        """Print everything."""
        self.print_100_trade_simulation()
        self.print_no_trade_calendar()
        self.print_risk_disclosure()
        self.print_founders_manual()
        print(SLUTTORD)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_founders_manual: Optional[FoundersOperatingManual] = None


def get_founders_manual() -> FoundersOperatingManual:
    """Get singleton Founders Operating Manual instance."""
    global _founders_manual
    if _founders_manual is None:
        _founders_manual = FoundersOperatingManual()
    return _founders_manual


def reset_founders_manual() -> FoundersOperatingManual:
    """Reset and get fresh manual instance."""
    global _founders_manual
    _founders_manual = FoundersOperatingManual()
    return _founders_manual


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    manual = get_founders_manual()
    manual.print_full()
    
    # Run simulation
    print("\n" + "=" * 40)
    print("Running 100-Trade Simulation...")
    print("=" * 40)
    
    result = manual.simulate_100_trades(seed=42)
    
    print(f"\nResults:")
    print(f"  Winners: {result.winners}")
    print(f"  Losers: {result.losers}")
    print(f"  Total R: {result.total_r:.1f}")
    print(f"  Expectancy: {result.expectancy_per_trade:.3f}R")
    print(f"\nLoss Series:")
    print(f"  3+ in row: {result.loss_series.three_in_row}")
    print(f"  5+ in row: {result.loss_series.five_in_row}")
    print(f"  7+ in row: {result.loss_series.seven_in_row}")
    print(f"  Max consecutive: {result.loss_series.max_consecutive}")
    
    # Check no-trade
    print("\n" + "=" * 40)
    print("No-Trade Check Example...")
    print("=" * 40)
    
    is_no_trade, severity, conditions = manual.check_no_trade(
        daily_loss_hit=True,
    )
    print(f"No-trade: {is_no_trade}")
    print(f"Severity: {severity.value if severity else 'None'}")
    print(f"Action: {manual.get_no_trade_action(severity) if severity else 'Trade allowed'}")
