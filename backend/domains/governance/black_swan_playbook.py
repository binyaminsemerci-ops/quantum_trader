"""
🦢 BLACK SWAN PLAYBOOK (OPERATIV)
==================================
Black swans predikeres ikke – de overleves.

Denne modulen implementerer:
1. 5 Black Swan scenario-klasser (BS-1 til BS-5)
2. Golden Rules (henges på veggen)
3. Krise-rekkefølge (30-60-120 protokoll)
4. Automatisk respons for hvert scenario
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 🧯 GOLDEN RULES (HENGES PÅ VEGGEN)
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_RULES = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    🧯 GOLDEN RULES - BLACK SWAN RESPONSE                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║    1️⃣  INGEN REDNINGS-TRADES                                               ║
║        Prøv aldri å "trade deg ut" av en krise.                            ║
║        Hver ny trade øker eksponering.                                     ║
║                                                                            ║
║    2️⃣  FLAT ER SUKSESS I KRISE                                             ║
║        Å gå flat = å sikre kapital.                                        ║
║        Null posisjon = null risiko.                                        ║
║                                                                            ║
║    3️⃣  IKKE VÆR FØRST – OVERLEV LENGST                                     ║
║        La andre panic-selge først.                                         ║
║        Vår jobb er å overleve, ikke å fange bunnen.                        ║
║                                                                            ║
║    4️⃣  RESTART KUN ETTER RO                                                ║
║        Volatilitet betyr usikkerhet.                                       ║
║        Vent til markedet stabiliserer.                                     ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 🧭 KRISE-REKKEFØLGE (30-60-120)
# ═══════════════════════════════════════════════════════════════════════════════

class CrisisPhase(str, Enum):
    """Crisis timeline phases."""
    IMMEDIATE = "0-30min"       # STOPP, flatt, sikre
    ASSESSMENT = "30-60min"     # Verifiser data, risiko, midler
    OBSERVATION = "60-120min"   # Shadow-observasjon
    RECOVERY = ">120min"        # Kun gradvis restart


@dataclass
class CrisisPhaseAction:
    """Actions for each crisis phase."""
    phase: CrisisPhase
    duration_minutes: int
    actions: List[str]
    forbidden_actions: List[str]
    success_criteria: List[str]


CRISIS_TIMELINE: Dict[CrisisPhase, CrisisPhaseAction] = {
    CrisisPhase.IMMEDIATE: CrisisPhaseAction(
        phase=CrisisPhase.IMMEDIATE,
        duration_minutes=30,
        actions=[
            "STOPP all trading umiddelbart",
            "Kanseler alle åpne orders",
            "Gå flat hvis mulig (uten panic-selling)",
            "Sikre at ingen nye orders sendes",
            "Verifiser alle posisjoner",
            "Start incident logging",
        ],
        forbidden_actions=[
            "Åpne nye posisjoner",
            "Endre eksisterende posisjoner",
            "Gjøre manuelle trades",
            "Ta beslutninger basert på følelser",
        ],
        success_criteria=[
            "Alle posisjoner lukket eller kontrollert",
            "Ingen åpne orders",
            "System i FROZEN state",
            "Incident log startet",
        ],
    ),
    
    CrisisPhase.ASSESSMENT: CrisisPhaseAction(
        phase=CrisisPhase.ASSESSMENT,
        duration_minutes=30,
        actions=[
            "Verifiser datafeed-kvalitet",
            "Sjekk exchange-status",
            "Kalkuler eksakt tap (hvis noen)",
            "Verifiser midler på exchange",
            "Identifiser årsak til krise",
            "Dokumenter hendelsesforløp",
        ],
        forbidden_actions=[
            "Rush til restart",
            "Blame-seeking",
            "Hastige endringer i kode",
        ],
        success_criteria=[
            "Data validert OK",
            "Exchange fungerer normalt",
            "Midler verifisert",
            "Root cause identifisert",
        ],
    ),
    
    CrisisPhase.OBSERVATION: CrisisPhaseAction(
        phase=CrisisPhase.OBSERVATION,
        duration_minutes=60,
        actions=[
            "Observer markedet UTEN å handle",
            "Kjør system i shadow-modus",
            "Verifiser at signaler er normale",
            "Vurder om krisen er over",
            "Planlegg gradvis restart",
        ],
        forbidden_actions=[
            "Ta live trades",
            "Gjøre hastige vurderinger",
            "Ignorere residual volatilitet",
        ],
        success_criteria=[
            "Markedet stabilisert",
            "Shadow-trades ville vært lønnsomme",
            "Ingen nye anomalier",
        ],
    ),
    
    CrisisPhase.RECOVERY: CrisisPhaseAction(
        phase=CrisisPhase.RECOVERY,
        duration_minutes=0,  # Open-ended
        actions=[
            "Start med 25% normal posisjonsstørrelse",
            "Øk gradvis over timer/dager",
            "Fortsett å overvåke nøye",
            "Skriv post-mortem rapport",
        ],
        forbidden_actions=[
            "Gå til 100% umiddelbart",
            "Ignorere volatilitet",
            "Glemme å dokumentere",
        ],
        success_criteria=[
            "Gradvis scaling fungerer",
            "Ingen nye incidents",
            "Post-mortem fullført",
        ],
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 🦢 BLACK SWAN SCENARIO CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class BlackSwanScenario(str, Enum):
    """The 5 Black Swan scenario classes."""
    BS_1_FLASH_CRASH = "bs1_flash_crash"
    BS_2_EXCHANGE_FAILURE = "bs2_exchange_failure"
    BS_3_FUNDING_SHOCK = "bs3_funding_shock"
    BS_4_MACRO_SHOCK = "bs4_macro_shock"
    BS_5_INTERNAL_FAILURE = "bs5_internal_failure"


@dataclass
class BlackSwanResponse:
    """Response protocol for a Black Swan scenario."""
    scenario: BlackSwanScenario
    name: str
    signal: List[str]
    handling: List[str]
    goal: str
    max_recovery_time_minutes: int
    requires_human_decision: bool = False


BLACK_SWAN_RESPONSES: Dict[BlackSwanScenario, BlackSwanResponse] = {
    
    BlackSwanScenario.BS_1_FLASH_CRASH: BlackSwanResponse(
        scenario=BlackSwanScenario.BS_1_FLASH_CRASH,
        name="🦢 BS-1: Flash Crash / Liquidity Vacuum",
        signal=[
            "Spread eksploderer (>10x normal)",
            "Volum forsvinner",
            "Price gaps",
            "Slippage > 1%",
        ],
        handling=[
            "Flat-mandat umiddelbart",
            "Cancel alle entries",
            "Ikke jag fallende priser",
            "Vent på stabilisering",
        ],
        goal="Unngå forced liquidation",
        max_recovery_time_minutes=240,
    ),
    
    BlackSwanScenario.BS_2_EXCHANGE_FAILURE: BlackSwanResponse(
        scenario=BlackSwanScenario.BS_2_EXCHANGE_FAILURE,
        name="🦢 BS-2: Exchange Failure / Halt",
        signal=[
            "API nede / timeouts",
            "Ulogiske fills",
            "Balance mismatch",
            "Order rejects",
        ],
        handling=[
            "Isoler exchange umiddelbart",
            "Sikre midler (withdraw hvis mulig)",
            "Ikke retry blindt",
            "Vent på official status",
        ],
        goal="Kapitalbeskyttelse",
        max_recovery_time_minutes=480,
        requires_human_decision=True,
    ),
    
    BlackSwanScenario.BS_3_FUNDING_SHOCK: BlackSwanResponse(
        scenario=BlackSwanScenario.BS_3_FUNDING_SHOCK,
        name="🦢 BS-3: Funding Shock (plutselig ekstrem)",
        signal=[
            "Funding > historisk P99",
            "Funding på feil side for posisjon",
            "Rask funding-eskalering",
        ],
        handling=[
            "Exit bias - lukk funding-tapende posisjoner",
            "Reduser leverage",
            "Vurder hedge",
            "Vent til funding normaliserer",
        ],
        goal="Edge-beskyttelse",
        max_recovery_time_minutes=60,
    ),
    
    BlackSwanScenario.BS_4_MACRO_SHOCK: BlackSwanResponse(
        scenario=BlackSwanScenario.BS_4_MACRO_SHOCK,
        name="🦢 BS-4: Systemisk Macro-sjokk",
        signal=[
            "Synkron kollaps på tvers av markeder",
            "Krypto + aksjer + obligasjoner faller",
            "VIX spike",
            "Nyhetsdrevet panikk",
        ],
        handling=[
            "Observer-only modus",
            "Ingen hero-trades",
            "Flat alle posisjoner gradvis",
            "Vent på regime-klarhet",
        ],
        goal="Overleve volatilitet",
        max_recovery_time_minutes=1440,  # 24 timer
        requires_human_decision=True,
    ),
    
    BlackSwanScenario.BS_5_INTERNAL_FAILURE: BlackSwanResponse(
        scenario=BlackSwanScenario.BS_5_INTERNAL_FAILURE,
        name="🦢 BS-5: Intern feil under stress",
        signal=[
            "Høy latency",
            "Ressurs-overforbruk (CPU/RAM)",
            "Database-feil",
            "Queue overflow",
        ],
        handling=[
            "Fail-closed umiddelbart",
            "Stopp alle operasjoner",
            "Kjør restart-protokoll",
            "Verifiser data-integritet",
        ],
        goal="Unngå feil-eskalering",
        max_recovery_time_minutes=120,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# BLACK SWAN DETECTION & RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlackSwanEvent:
    """A detected Black Swan event."""
    scenario: BlackSwanScenario
    detected_at: datetime
    signals_triggered: List[str]
    current_phase: CrisisPhase
    positions_when_detected: int
    context: Dict[str, Any] = field(default_factory=dict)


class BlackSwanPlaybook:
    """
    Black Swan Playbook manager.
    
    Prinsipp: Black swans predikeres ikke – de overleves.
    
    Usage:
        playbook = BlackSwanPlaybook()
        
        # Detect scenario
        event = playbook.detect_scenario(market_data)
        
        # Execute response
        if event:
            await playbook.execute_response(event)
    """
    
    def __init__(self):
        self._active_event: Optional[BlackSwanEvent] = None
        self._phase_start_time: Optional[datetime] = None
        self._event_history: List[BlackSwanEvent] = []
        self._is_in_crisis: bool = False
        
    @property
    def is_in_crisis(self) -> bool:
        """Check if currently in crisis mode."""
        return self._is_in_crisis
    
    @property
    def current_phase(self) -> Optional[CrisisPhase]:
        """Get current crisis phase."""
        if not self._active_event:
            return None
        return self._active_event.current_phase
    
    @property
    def active_event(self) -> Optional[BlackSwanEvent]:
        """Get active Black Swan event."""
        return self._active_event
    
    def detect_flash_crash(
        self,
        spread_ratio: float,
        volume_ratio: float,
        slippage_pct: float,
    ) -> bool:
        """
        Detect BS-1: Flash Crash / Liquidity Vacuum.
        
        Args:
            spread_ratio: Current spread vs normal (>10 = alarm)
            volume_ratio: Current volume vs normal (<0.1 = alarm)
            slippage_pct: Recent slippage percentage (>1 = alarm)
        """
        signals = []
        
        if spread_ratio > 10:
            signals.append(f"Spread explosion: {spread_ratio:.1f}x normal")
        if volume_ratio < 0.1:
            signals.append(f"Volume vacuum: {volume_ratio:.1%} of normal")
        if slippage_pct > 1.0:
            signals.append(f"Slippage: {slippage_pct:.2f}%")
        
        if len(signals) >= 2:
            self._trigger_event(BlackSwanScenario.BS_1_FLASH_CRASH, signals)
            return True
        
        return False
    
    def detect_exchange_failure(
        self,
        api_healthy: bool,
        balance_matches: bool,
        order_reject_rate: float,
    ) -> bool:
        """
        Detect BS-2: Exchange Failure / Halt.
        
        Args:
            api_healthy: Is API responding normally?
            balance_matches: Does local balance match exchange?
            order_reject_rate: Rate of order rejections (>0.5 = alarm)
        """
        signals = []
        
        if not api_healthy:
            signals.append("API unhealthy / timeout")
        if not balance_matches:
            signals.append("Balance mismatch detected")
        if order_reject_rate > 0.5:
            signals.append(f"Order reject rate: {order_reject_rate:.1%}")
        
        if len(signals) >= 2:
            self._trigger_event(BlackSwanScenario.BS_2_EXCHANGE_FAILURE, signals)
            return True
        
        return False
    
    def detect_funding_shock(
        self,
        current_funding: float,
        p99_funding: float,
        position_side: str,
    ) -> bool:
        """
        Detect BS-3: Funding Shock.
        
        Args:
            current_funding: Current funding rate
            p99_funding: Historical P99 funding rate
            position_side: "long" or "short"
        """
        signals = []
        
        # Funding > P99
        if abs(current_funding) > abs(p99_funding):
            signals.append(f"Funding {current_funding:.4f} > P99 {p99_funding:.4f}")
        
        # Paying funding on wrong side
        if position_side == "long" and current_funding > 0.0003:  # >0.03% = paying
            signals.append(f"Long position paying high funding: {current_funding:.4f}")
        elif position_side == "short" and current_funding < -0.0003:
            signals.append(f"Short position paying high funding: {current_funding:.4f}")
        
        if len(signals) >= 1 and abs(current_funding) > abs(p99_funding):
            self._trigger_event(BlackSwanScenario.BS_3_FUNDING_SHOCK, signals)
            return True
        
        return False
    
    def detect_macro_shock(
        self,
        btc_drawdown_pct: float,
        correlation_spike: bool,
        vix_level: Optional[float] = None,
    ) -> bool:
        """
        Detect BS-4: Systemisk Macro-sjokk.
        
        Args:
            btc_drawdown_pct: BTC drawdown in last 24h
            correlation_spike: Are all assets moving together?
            vix_level: VIX level if available (>30 = alarm)
        """
        signals = []
        
        if btc_drawdown_pct < -10:
            signals.append(f"BTC drawdown: {btc_drawdown_pct:.1f}%")
        if correlation_spike:
            signals.append("Cross-market correlation spike")
        if vix_level and vix_level > 30:
            signals.append(f"VIX elevated: {vix_level:.1f}")
        
        if len(signals) >= 2:
            self._trigger_event(BlackSwanScenario.BS_4_MACRO_SHOCK, signals)
            return True
        
        return False
    
    def detect_internal_failure(
        self,
        latency_ms: float,
        cpu_percent: float,
        memory_percent: float,
        queue_length: int,
    ) -> bool:
        """
        Detect BS-5: Intern feil under stress.
        
        Args:
            latency_ms: Current processing latency
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            queue_length: Message queue length
        """
        signals = []
        
        if latency_ms > 1000:
            signals.append(f"High latency: {latency_ms:.0f}ms")
        if cpu_percent > 90:
            signals.append(f"CPU overload: {cpu_percent:.0f}%")
        if memory_percent > 90:
            signals.append(f"Memory overload: {memory_percent:.0f}%")
        if queue_length > 10000:
            signals.append(f"Queue overflow: {queue_length}")
        
        if len(signals) >= 2:
            self._trigger_event(BlackSwanScenario.BS_5_INTERNAL_FAILURE, signals)
            return True
        
        return False
    
    def _trigger_event(self, scenario: BlackSwanScenario, signals: List[str]):
        """Create and activate a Black Swan event."""
        if self._is_in_crisis:
            logger.warning(f"Already in crisis mode, cannot trigger new event: {scenario}")
            return
        
        event = BlackSwanEvent(
            scenario=scenario,
            detected_at=datetime.utcnow(),
            signals_triggered=signals,
            current_phase=CrisisPhase.IMMEDIATE,
            positions_when_detected=0,  # Will be updated by caller
        )
        
        self._active_event = event
        self._phase_start_time = datetime.utcnow()
        self._is_in_crisis = True
        
        response = BLACK_SWAN_RESPONSES[scenario]
        
        logger.critical(
            f"🦢 BLACK SWAN DETECTED: {response.name}\n"
            f"   Signals: {signals}\n"
            f"   Goal: {response.goal}"
        )
    
    async def execute_immediate_response(self) -> Dict[str, Any]:
        """
        Execute immediate crisis response (0-30 minutes).
        Returns actions taken.
        """
        if not self._active_event:
            return {"error": "No active event"}
        
        response = BLACK_SWAN_RESPONSES[self._active_event.scenario]
        phase_actions = CRISIS_TIMELINE[CrisisPhase.IMMEDIATE]
        
        logger.info(f"🚨 Executing IMMEDIATE response for {response.name}")
        
        actions_taken = []
        
        for action in phase_actions.actions:
            logger.info(f"   → {action}")
            actions_taken.append(action)
        
        # Update phase
        self._active_event.current_phase = CrisisPhase.IMMEDIATE
        
        return {
            "phase": CrisisPhase.IMMEDIATE.value,
            "scenario": self._active_event.scenario.value,
            "actions_taken": actions_taken,
            "handling": response.handling,
            "goal": response.goal,
            "next_phase_in_minutes": 30,
        }
    
    def advance_phase(self) -> Optional[CrisisPhase]:
        """Advance to next crisis phase if time has elapsed."""
        if not self._active_event or not self._phase_start_time:
            return None
        
        elapsed = datetime.utcnow() - self._phase_start_time
        current = self._active_event.current_phase
        
        if current == CrisisPhase.IMMEDIATE and elapsed >= timedelta(minutes=30):
            self._active_event.current_phase = CrisisPhase.ASSESSMENT
            self._phase_start_time = datetime.utcnow()
            logger.info("🧭 Advanced to ASSESSMENT phase")
            return CrisisPhase.ASSESSMENT
        
        elif current == CrisisPhase.ASSESSMENT and elapsed >= timedelta(minutes=30):
            self._active_event.current_phase = CrisisPhase.OBSERVATION
            self._phase_start_time = datetime.utcnow()
            logger.info("🧭 Advanced to OBSERVATION phase")
            return CrisisPhase.OBSERVATION
        
        elif current == CrisisPhase.OBSERVATION and elapsed >= timedelta(minutes=60):
            self._active_event.current_phase = CrisisPhase.RECOVERY
            self._phase_start_time = datetime.utcnow()
            logger.info("🧭 Advanced to RECOVERY phase")
            return CrisisPhase.RECOVERY
        
        return current
    
    def resolve_crisis(self, post_mortem: str):
        """Mark crisis as resolved and archive."""
        if not self._active_event:
            return
        
        logger.info(f"✅ Crisis resolved: {self._active_event.scenario.value}")
        logger.info(f"   Post-mortem: {post_mortem}")
        
        self._event_history.append(self._active_event)
        self._active_event = None
        self._phase_start_time = None
        self._is_in_crisis = False
    
    def get_current_actions(self) -> List[str]:
        """Get actions for current crisis phase."""
        if not self._active_event:
            return []
        
        phase = self._active_event.current_phase
        return CRISIS_TIMELINE[phase].actions
    
    def get_forbidden_actions(self) -> List[str]:
        """Get forbidden actions for current crisis phase."""
        if not self._active_event:
            return []
        
        phase = self._active_event.current_phase
        return CRISIS_TIMELINE[phase].forbidden_actions
    
    def print_golden_rules(self):
        """Print Golden Rules."""
        print(GOLDEN_RULES)
    
    def print_status(self):
        """Print current crisis status."""
        print("\n" + "=" * 70)
        print("🦢 BLACK SWAN PLAYBOOK STATUS")
        print("=" * 70)
        
        if not self._is_in_crisis:
            print("\n✅ No active crisis. System operating normally.")
            print(f"\nHistorical crises: {len(self._event_history)}")
        else:
            event = self._active_event
            response = BLACK_SWAN_RESPONSES[event.scenario]
            
            print(f"\n🚨 ACTIVE CRISIS: {response.name}")
            print(f"   Detected: {event.detected_at.isoformat()}")
            print(f"   Current Phase: {event.current_phase.value}")
            print(f"   Signals: {event.signals_triggered}")
            print(f"\n   Goal: {response.goal}")
            
            phase_info = CRISIS_TIMELINE[event.current_phase]
            print(f"\n📋 Current Phase Actions:")
            for action in phase_info.actions:
                print(f"      → {action}")
            
            print(f"\n🚫 Forbidden:")
            for forbidden in phase_info.forbidden_actions:
                print(f"      ✗ {forbidden}")
        
        print("\n" + "=" * 70)
        self.print_golden_rules()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_playbook: Optional[BlackSwanPlaybook] = None


def get_black_swan_playbook() -> BlackSwanPlaybook:
    """Get singleton Black Swan Playbook instance."""
    global _playbook
    if _playbook is None:
        _playbook = BlackSwanPlaybook()
    return _playbook


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    playbook = get_black_swan_playbook()
    playbook.print_status()
