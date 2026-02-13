"""
🧠 FRA GRUNNLOV → MICROSERVICES-ARKITEKTUR
==========================================
Institusjonell oversettelse (theory → system design)

🧱 PRINSIPP FØRST (VIKTIG):
- Microservices skal IKKE representere strategier
- De representerer ANSVAR og LOV-HÅNDHEVELSE
- Én lov = én eier
- Én failure = én service som stopper systemet
- Ingen service får total makt

Dette er minste hedge-fond-kjerne (11 services).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum, auto
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MICROSERVICE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MicroserviceID(str, Enum):
    """The 11 core microservices for institutional trading."""
    
    # 1️⃣ Grunnlovens vokter
    POLICY_CONSTITUTION = "policy_constitution_service"
    
    # 2️⃣ Markedets sannhet
    MARKET_REALITY = "market_reality_service"
    
    # 3️⃣ Løgnedetektor
    DATA_INTEGRITY = "data_integrity_sentinel"
    
    # 4️⃣ Absolutt makt (Fail-Closed)
    RISK_KERNEL = "risk_kernel_service"
    
    # 5️⃣ Hvor mye – aldri om
    CAPITAL_ALLOCATION = "capital_allocation_service"
    
    # 6️⃣ Rådgiver, ikke sjef
    SIGNAL_AI_ADVISORY = "signal_ai_advisory_service"
    
    # 7️⃣ Siste filter før markedet
    ENTRY_QUALIFICATION = "entry_qualification_gate"
    
    # 8️⃣ Markedets oversetter
    EXECUTION = "execution_service"
    
    # 9️⃣ Der fondet tjener penger
    EXIT_HARVEST = "exit_harvest_brain"
    
    # 🔟 Beskyttelse mot DEG
    HUMAN_OVERRIDE_LOCK = "human_override_lock"
    
    # 1️⃣1️⃣ Minne og sannhet
    AUDIT_LEDGER = "audit_ledger_service"


class ServicePower(str, Enum):
    """Power levels for services."""
    VETO = "veto"           # Can stop everything
    CONTROL = "control"     # Can control flow
    ADVISORY = "advisory"   # Can only advise
    EXECUTE = "execute"     # Can only execute orders
    OBSERVE = "observe"     # Can only observe and log


@dataclass
class ServicePermission:
    """What a service CAN and CANNOT do."""
    can_do: List[str]
    cannot_do: List[str]
    stops_system_when: List[str]


@dataclass
class MicroserviceSpec:
    """Specification for a microservice."""
    id: MicroserviceID
    name: str
    emoji: str
    role: str  # Short description
    
    # Ownership
    owns: List[str]
    
    # Responsibilities
    responsibilities: List[str]
    
    # Permissions
    permissions: ServicePermission
    
    # Dependencies
    depends_on: List[MicroserviceID]
    
    # Power level
    power: ServicePower
    
    # Key insight
    insight: str
    
    # In command hierarchy
    hierarchy_level: int  # 1 = top, 7 = bottom


# ═══════════════════════════════════════════════════════════════════════════════
# THE 11 MICROSERVICES SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════

MICROSERVICES: Dict[MicroserviceID, MicroserviceSpec] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1️⃣ POLICY & CONSTITUTION SERVICE (Grunnlovens vokter)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.POLICY_CONSTITUTION: MicroserviceSpec(
        id=MicroserviceID.POLICY_CONSTITUTION,
        name="Policy & Constitution Service",
        emoji="1️⃣",
        role="Grunnlovens vokter",
        owns=[
            "Alle grunnlover (15 lover)",
            "Forbud",
            "Tillatte handlinger",
            "Policy-parametre",
        ],
        responsibilities=[
            "Hva er lov å gjøre akkurat nå?",
            "Kan dette signalet eksistere?",
            "Er denne handlingen tillatt?",
            "Validere alle requests mot grunnlover",
        ],
        permissions=ServicePermission(
            can_do=[
                "Definere hva som er lov",
                "Avvise ulovlige handlinger",
                "Oppdatere policy (med audit)",
            ],
            cannot_do=[
                "Initiere trades",
                "Overstyre Risk Kernel",
            ],
            stops_system_when=[
                "Grunnlov er uklar",
                "Policy-konflikt oppdaget",
            ],
        ),
        depends_on=[],  # Top of hierarchy
        power=ServicePower.CONTROL,
        hierarchy_level=1,
        insight="📌 Ingen annen service tar beslutninger uten å spørre denne.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 2️⃣ MARKET REALITY / REGIME SERVICE (Markedets sannhet)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.MARKET_REALITY: MicroserviceSpec(
        id=MicroserviceID.MARKET_REALITY,
        name="Market Reality / Regime Service",
        emoji="2️⃣",
        role="Markedets sannhet",
        owns=[
            "Regime (trend / chop / panic)",
            "No-trade soner",
            "Volatilitetstilstand",
            "Markedsstruktur",
        ],
        responsibilities=[
            "Definere nåværende markedsregime",
            "Identifisere no-trade soner",
            "Måle volatilitet og likviditet",
            "Bestemme OM systemet får være aktivt",
        ],
        permissions=ServicePermission(
            can_do=[
                "Pause all trading",
                "Definere regime",
                "Blokkere entries",
            ],
            cannot_do=[
                "Overstyre Risk Kernel",
                "Ignorere data-problemer",
            ],
            stops_system_when=[
                "Regime er udefinert",
                "Markedet er kaotisk",
                "Likviditet forsvinner",
            ],
        ),
        depends_on=[MicroserviceID.DATA_INTEGRITY],
        power=ServicePower.CONTROL,
        hierarchy_level=3,
        insight="📌 Denne service bestemmer OM systemet får lov å være aktivt.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3️⃣ DATA INTEGRITY SENTINEL (Løgnedetektor)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.DATA_INTEGRITY: MicroserviceSpec(
        id=MicroserviceID.DATA_INTEGRITY,
        name="Data Integrity Sentinel",
        emoji="3️⃣",
        role="Løgnedetektor",
        owns=[
            "Prisfeed-konsistens",
            "Klokkesynk",
            "Latency-målinger",
            "Data-kvalitetsscore",
        ],
        responsibilities=[
            "Verifisere data-integritet",
            "Oppdage anomalier",
            "Sammenligne feeds",
            "Måle forsinkelser",
        ],
        permissions=ServicePermission(
            can_do=[
                "Trigge SAFE MODE",
                "Blokkere entries",
                "Flagge inkonsistens",
            ],
            cannot_do=[
                "Overstyre exits",
                "Endre posisjoner",
            ],
            stops_system_when=[
                "Data-inkonsistens oppdaget",
                "Latency over grense",
                "Feed-mismatch",
            ],
        ),
        depends_on=[],  # Independent sentinel
        power=ServicePower.CONTROL,
        hierarchy_level=2,  # Very high - bad data stops everything
        insight="📌 Feil data er farligere enn ingen data.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4️⃣ RISK KERNEL (Absolutt makt - Fail-Closed)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.RISK_KERNEL: MicroserviceSpec(
        id=MicroserviceID.RISK_KERNEL,
        name="Risk Kernel (Fail-Closed)",
        emoji="4️⃣",
        role="Absolutt makt",
        owns=[
            "Risiko per trade (2%)",
            "Daglig tap (-5%)",
            "Total drawdown (-15%)",
            "Leverage (max 10x)",
            "Kill-switch",
        ],
        responsibilities=[
            "Håndheve alle risiko-grenser",
            "Aktivere kill-switch ved brudd",
            "Tvinge reduce-only ved behov",
            "VETO alle handlinger",
        ],
        permissions=ServicePermission(
            can_do=[
                "Stoppe ALLE andre services",
                "Tvinge reduce-only",
                "Fryse systemet",
                "Overstyre enhver beslutning",
            ],
            cannot_do=[
                "Initiere trades",
                "Endre grunnlover",
            ],
            stops_system_when=[
                "Risiko-grense brutt",
                "Daglig tap nådd",
                "Leverage overskredet",
                "ANY uncertainty",
            ],
        ),
        depends_on=[MicroserviceID.POLICY_CONSTITUTION],
        power=ServicePower.VETO,  # Highest power
        hierarchy_level=2,
        insight="📌 Dette er hedgefondets hjerte.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5️⃣ CAPITAL ALLOCATION SERVICE (Hvor mye – aldri om)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.CAPITAL_ALLOCATION: MicroserviceSpec(
        id=MicroserviceID.CAPITAL_ALLOCATION,
        name="Capital Allocation Service",
        emoji="5️⃣",
        role="Hvor mye – aldri om",
        owns=[
            "Position sizing",
            "Leverage-beregning",
            "Capital heat",
            "Korrelasjon-justering",
        ],
        responsibilities=[
            "Beregne optimal posisjonsstørrelse",
            "Justere for korrelasjon",
            "Respektere total eksponering",
            "Aldri bestemme OM - bare HVOR MYE",
        ],
        permissions=ServicePermission(
            can_do=[
                "Beregne size",
                "Justere leverage",
                "Anbefale reduksjon",
            ],
            cannot_do=[
                "Starte en trade selv",
                "Ignorere Risk Kernel",
                "Overstyre regime",
            ],
            stops_system_when=[],  # Never stops - only advises
        ),
        depends_on=[
            MicroserviceID.POLICY_CONSTITUTION,
            MicroserviceID.RISK_KERNEL,
            MicroserviceID.MARKET_REALITY,
        ],
        power=ServicePower.ADVISORY,
        hierarchy_level=4,
        insight="📌 Denne kan ALDRI starte en trade selv.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 6️⃣ SIGNAL / AI ADVISORY SERVICE (Rådgiver, ikke sjef)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.SIGNAL_AI_ADVISORY: MicroserviceSpec(
        id=MicroserviceID.SIGNAL_AI_ADVISORY,
        name="Signal / AI Advisory Service",
        emoji="6️⃣",
        role="Rådgiver, ikke sjef",
        owns=[
            "Edge-score",
            "Confidence",
            "Signal-rangering",
            "AI-modeller",
        ],
        responsibilities=[
            "Generere handelssignaler",
            "Beregne edge-sannsynlighet",
            "Rangere muligheter",
            "Kun ANBEFALE - aldri bestemme",
        ],
        permissions=ServicePermission(
            can_do=[
                "Generere signaler",
                "Beregne confidence",
                "Rangere trades",
            ],
            cannot_do=[
                "Bestemme size",
                "Ignorere risk",
                "Styre exit",
                "Overstyre noen",
            ],
            stops_system_when=[],  # Never stops - only advises
        ),
        depends_on=[
            MicroserviceID.DATA_INTEGRITY,
            MicroserviceID.MARKET_REALITY,
        ],
        power=ServicePower.ADVISORY,
        hierarchy_level=0,  # Outside command chain!
        insight="📌 AI snakker – systemet bestemmer.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 7️⃣ ENTRY QUALIFICATION GATE (Siste filter før markedet)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.ENTRY_QUALIFICATION: MicroserviceSpec(
        id=MicroserviceID.ENTRY_QUALIFICATION,
        name="Entry Qualification Gate",
        emoji="7️⃣",
        role="Siste filter før markedet",
        owns=[
            "Entry-validering",
            "Exit-krav",
            "Strukturkrav",
            "Lovlighetssjekk",
        ],
        responsibilities=[
            "Finnes exit?",
            "Er risk definert?",
            "Er dette lov nå?",
            "Har signalet struktur?",
        ],
        permissions=ServicePermission(
            can_do=[
                "Avvise entries",
                "Kreve exit-plan",
                "Verifisere struktur",
            ],
            cannot_do=[
                "Initiere trades",
                "Endre size",
                "Overstyre risk",
            ],
            stops_system_when=[
                "Entry mangler exit",
                "Risk ikke definert",
                "Strukturkrav ikke møtt",
            ],
        ),
        depends_on=[
            MicroserviceID.POLICY_CONSTITUTION,
            MicroserviceID.RISK_KERNEL,
            MicroserviceID.MARKET_REALITY,
            MicroserviceID.CAPITAL_ALLOCATION,
        ],
        power=ServicePower.CONTROL,
        hierarchy_level=5,
        insight="📌 Her dør 90% av dårlige trades – med vilje.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 8️⃣ EXECUTION SERVICE (Markedets oversetter)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.EXECUTION: MicroserviceSpec(
        id=MicroserviceID.EXECUTION,
        name="Execution Service",
        emoji="8️⃣",
        role="Markedets oversetter",
        owns=[
            "Order-typer",
            "Reduce-only logikk",
            "Slippage-håndtering",
            "Fill-kvalitet",
        ],
        responsibilities=[
            "Sende orders til exchange",
            "Håndtere reduce-only",
            "Måle slippage",
            "Rapportere fills",
        ],
        permissions=ServicePermission(
            can_do=[
                "Sende orders",
                "Håndtere slippage",
                "Rapportere fills",
            ],
            cannot_do=[
                "Initiere en trade",
                "Overstyre risk",
                "Bestemme size",
            ],
            stops_system_when=[
                "Exchange-feil",
                "Slippage for høy",
                "API-problemer",
            ],
        ),
        depends_on=[
            MicroserviceID.ENTRY_QUALIFICATION,
            MicroserviceID.RISK_KERNEL,
        ],
        power=ServicePower.EXECUTE,
        hierarchy_level=6,
        insight="📌 Execution er mekanikk – ikke intelligens.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 9️⃣ EXIT / HARVEST BRAIN (Der fondet tjener penger)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.EXIT_HARVEST: MicroserviceSpec(
        id=MicroserviceID.EXIT_HARVEST,
        name="Exit / Harvest Brain",
        emoji="9️⃣",
        role="Der fondet tjener penger",
        owns=[
            "Partial exits",
            "Trailing logic",
            "Time-based exits",
            "Regime exits",
        ],
        responsibilities=[
            "Bestemme når og hvordan exit",
            "Håndtere partials",
            "Håndheve time-stops",
            "Reagere på regime-endring",
        ],
        permissions=ServicePermission(
            can_do=[
                "Overstyre entry-beslutninger",
                "Tvinge exit",
                "Justere stops",
            ],
            cannot_do=[
                "Åpne nye posisjoner",
                "Ignorere risk kernel",
            ],
            stops_system_when=[],  # Doesn't stop - it CLOSES
        ),
        depends_on=[
            MicroserviceID.RISK_KERNEL,
            MicroserviceID.MARKET_REALITY,
        ],
        power=ServicePower.CONTROL,
        hierarchy_level=7,
        insight="📌 Exit slår alltid entry.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔟 HUMAN OVERRIDE LOCK (Beskyttelse mot DEG)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.HUMAN_OVERRIDE_LOCK: MicroserviceSpec(
        id=MicroserviceID.HUMAN_OVERRIDE_LOCK,
        name="Human Override Lock",
        emoji="🔟",
        role="Beskyttelse mot DEG",
        owns=[
            "Manuelle inngrep",
            "Cooldown-regler",
            "Override-audit",
            "Freeze-protokoll",
        ],
        responsibilities=[
            "Blokkere emosjonelle inngrep",
            "Håndheve cooldown",
            "Logge alle overrides",
            "Beskytte mot revenge trading",
        ],
        permissions=ServicePermission(
            can_do=[
                "Fryse systemet",
                "Blokkere manuelle endringer",
                "Kreve cooldown",
            ],
            cannot_do=[
                "Initiere trades",
                "Endre strategi",
            ],
            stops_system_when=[
                "Manual override forsøkt",
                "Cooldown brutt",
                "Revenge trading oppdaget",
            ],
        ),
        depends_on=[MicroserviceID.AUDIT_LEDGER],
        power=ServicePower.VETO,
        hierarchy_level=1,  # Very high - protects from human
        insight="📌 Dette er den viktigste servicen for privat kapital.",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1️⃣1️⃣ AUDIT & LEDGER SERVICE (Minne og sannhet)
    # ═══════════════════════════════════════════════════════════════════════════
    
    MicroserviceID.AUDIT_LEDGER: MicroserviceSpec(
        id=MicroserviceID.AUDIT_LEDGER,
        name="Audit & Ledger Service",
        emoji="1️⃣1️⃣",
        role="Minne og sannhet",
        owns=[
            "Alle beslutninger",
            "Alle stops",
            "Alle overrides",
            "Komplett historikk",
        ],
        responsibilities=[
            "Logge alt",
            "Bevare sannhet",
            "Muliggjøre analyse",
            "Støtte compliance",
        ],
        permissions=ServicePermission(
            can_do=[
                "Logge alt",
                "Lese all aktivitet",
                "Generere rapporter",
            ],
            cannot_do=[
                "Endre beslutninger",
                "Stoppe trading",
                "Slette historikk",
            ],
            stops_system_when=[],  # Never stops - only observes
        ),
        depends_on=[],  # Independent observer
        power=ServicePower.OBSERVE,
        hierarchy_level=0,  # Outside hierarchy - observes all
        insight="📌 Uten denne kan systemet ikke vokse.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE → SERVICE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

FAILURE_TO_SERVICE_MAP: Dict[str, MicroserviceID] = {
    "Risiko-brudd": MicroserviceID.RISK_KERNEL,
    "Datafeil": MicroserviceID.DATA_INTEGRITY,
    "Regime-kaos": MicroserviceID.MARKET_REALITY,
    "Menneskelig ego": MicroserviceID.HUMAN_OVERRIDE_LOCK,
    "Edge-forfall": MicroserviceID.SIGNAL_AI_ADVISORY,  # + Performance Monitor
    "Execution-feil": MicroserviceID.EXECUTION,
    "Exchange-feil": MicroserviceID.EXECUTION,  # + Exchange Health Monitor
    "Policy-brudd": MicroserviceID.POLICY_CONSTITUTION,
    "Entry-mangel": MicroserviceID.ENTRY_QUALIFICATION,
    "Exit-feil": MicroserviceID.EXIT_HARVEST,
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════════

COMMAND_HIERARCHY = """
🗺️ KOMMANDOHIERARKI (VIKTIG)

    Policy / Constitution
            ↓
    Risk Kernel (VETO)
            ↓
    Market Reality
            ↓
    Capital Allocation
            ↓
    Entry Gate
            ↓
    Execution
            ↓
    Exit Brain

    ⚠️ AI ligger UTENFOR kommandolinjen.
    ⚠️ Audit observerer ALT men påvirker ingenting.
"""

COMMAND_HIERARCHY_LEVELS: Dict[int, List[MicroserviceID]] = {
    1: [MicroserviceID.POLICY_CONSTITUTION, MicroserviceID.HUMAN_OVERRIDE_LOCK],
    2: [MicroserviceID.RISK_KERNEL, MicroserviceID.DATA_INTEGRITY],
    3: [MicroserviceID.MARKET_REALITY],
    4: [MicroserviceID.CAPITAL_ALLOCATION],
    5: [MicroserviceID.ENTRY_QUALIFICATION],
    6: [MicroserviceID.EXECUTION],
    7: [MicroserviceID.EXIT_HARVEST],
    # Level 0 = Outside hierarchy
    0: [MicroserviceID.SIGNAL_AI_ADVISORY, MicroserviceID.AUDIT_LEDGER],
}


# ═══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

KEY_INSIGHTS = """
🧠 VIKTIGSTE PROFESJONELLE INNSIKT

1. Microservices handler om MAKTFORDELING
   → Ingen service får total makt
   → Én lov = én eier

2. Failures er FEATURES, ikke bugs
   → Ett system som stopper seg selv → overlever
   → Ett system som ikke stopper → dør

3. AI uten grenser er FARLIG
   → AI snakker, systemet bestemmer
   → AI ligger utenfor kommandolinjen

4. Exit slår ALLTID entry
   → Der fondet faktisk tjener penger
   → Entry er hypotese, exit er sannhet

5. Human Override Lock er VIKTIGST for privat kapital
   → Beskytter deg mot deg selv
   → Cooldown er obligatorisk
"""


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MicroserviceArchitecture:
    """
    Microservice architecture manager.
    
    Translates constitutional laws to service responsibilities.
    """
    
    def __init__(self):
        self._services = MICROSERVICES
        self._failure_map = FAILURE_TO_SERVICE_MAP
    
    def get_service(self, service_id: MicroserviceID) -> MicroserviceSpec:
        """Get a service specification."""
        return self._services[service_id]
    
    def get_responsible_service(self, failure_type: str) -> Optional[MicroserviceID]:
        """Get the service responsible for a failure type."""
        return self._failure_map.get(failure_type)
    
    def get_services_by_power(self, power: ServicePower) -> List[MicroserviceSpec]:
        """Get all services with a specific power level."""
        return [s for s in self._services.values() if s.power == power]
    
    def get_services_by_level(self, level: int) -> List[MicroserviceSpec]:
        """Get all services at a hierarchy level."""
        if level not in COMMAND_HIERARCHY_LEVELS:
            return []
        return [self._services[sid] for sid in COMMAND_HIERARCHY_LEVELS[level]]
    
    def get_dependency_chain(self, service_id: MicroserviceID) -> List[MicroserviceID]:
        """Get the full dependency chain for a service."""
        service = self._services[service_id]
        chain = list(service.depends_on)
        
        for dep in service.depends_on:
            chain.extend(self.get_dependency_chain(dep))
        
        return list(dict.fromkeys(chain))  # Remove duplicates, preserve order
    
    def can_service_request(
        self,
        requester: MicroserviceID,
        target: MicroserviceID,
    ) -> bool:
        """
        Check if a service can make requests to another service.
        Based on hierarchy level.
        """
        req_spec = self._services[requester]
        tgt_spec = self._services[target]
        
        # Level 0 (AI, Audit) can request from anyone
        if req_spec.hierarchy_level == 0:
            return True
        
        # Target level 0 services can be requested by anyone
        if tgt_spec.hierarchy_level == 0:
            return True
        
        # Otherwise, lower level can request from higher level
        return req_spec.hierarchy_level >= tgt_spec.hierarchy_level
    
    def validate_architecture(self) -> Dict[str, Any]:
        """Validate the architecture is complete and consistent."""
        issues = []
        
        # Check all failures have a responsible service
        for failure, service in self._failure_map.items():
            if service not in self._services:
                issues.append(f"Failure '{failure}' maps to unknown service")
        
        # Check all dependencies exist
        for service_id, spec in self._services.items():
            for dep in spec.depends_on:
                if dep not in self._services:
                    issues.append(f"Service '{service_id}' depends on unknown '{dep}'")
        
        # Check hierarchy completeness
        all_in_hierarchy = set()
        for level, services in COMMAND_HIERARCHY_LEVELS.items():
            all_in_hierarchy.update(services)
        
        missing = set(self._services.keys()) - all_in_hierarchy
        if missing:
            issues.append(f"Services not in hierarchy: {missing}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "service_count": len(self._services),
            "failure_mappings": len(self._failure_map),
        }
    
    def print_architecture(self):
        """Print full architecture overview."""
        print("\n" + "=" * 80)
        print("🧠 MICROSERVICE ARCHITECTURE (11 SERVICES)")
        print("=" * 80)
        print("\n🧱 PRINSIPP:")
        print("   • Microservices representerer ANSVAR og LOV-HÅNDHEVELSE")
        print("   • Én lov = én eier")
        print("   • Én failure = én service som stopper systemet")
        print("   • Ingen service får total makt")
        
        print(COMMAND_HIERARCHY)
        
        print("-" * 80)
        print("SERVICES OVERVIEW:")
        print("-" * 80)
        
        for service_id, spec in self._services.items():
            print(f"\n{spec.emoji} {spec.name}")
            print(f"   Role: {spec.role}")
            print(f"   Power: {spec.power.value}")
            print(f"   Level: {spec.hierarchy_level}")
            print(f"   {spec.insight}")
        
        print("\n" + "-" * 80)
        print("FAILURE → SERVICE MAPPING:")
        print("-" * 80)
        
        for failure, service in self._failure_map.items():
            service_name = self._services[service].name
            print(f"   {failure:20} → {service_name}")
        
        print("\n" + KEY_INSIGHTS)
        print("=" * 80 + "\n")
    
    def print_service_detail(self, service_id: MicroserviceID):
        """Print detailed info for a specific service."""
        spec = self._services[service_id]
        
        print(f"\n{'=' * 60}")
        print(f"{spec.emoji} {spec.name}")
        print(f"{'=' * 60}")
        print(f"\nRole: {spec.role}")
        print(f"Power: {spec.power.value}")
        print(f"Hierarchy Level: {spec.hierarchy_level}")
        
        print(f"\n📦 OWNS:")
        for item in spec.owns:
            print(f"   • {item}")
        
        print(f"\n📋 RESPONSIBILITIES:")
        for item in spec.responsibilities:
            print(f"   • {item}")
        
        print(f"\n✅ CAN DO:")
        for item in spec.permissions.can_do:
            print(f"   • {item}")
        
        print(f"\n❌ CANNOT DO:")
        for item in spec.permissions.cannot_do:
            print(f"   • {item}")
        
        if spec.permissions.stops_system_when:
            print(f"\n🛑 STOPS SYSTEM WHEN:")
            for item in spec.permissions.stops_system_when:
                print(f"   • {item}")
        
        if spec.depends_on:
            print(f"\n🔗 DEPENDS ON:")
            for dep in spec.depends_on:
                print(f"   • {self._services[dep].name}")
        
        print(f"\n💡 {spec.insight}")
        print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ASCII DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

ARCHITECTURE_DIAGRAM = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 MICROSERVICE ARCHITECTURE                              ║
║                    (Grunnlov → System Design)                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                    LEVEL 1: GOVERNANCE                               │    ║
║  │  ┌────────────────────────┐    ┌────────────────────────┐           │    ║
║  │  │ 1️⃣ Policy/Constitution │    │ 🔟 Human Override Lock  │           │    ║
║  │  │    (Grunnlovens vokter)│    │ (Beskyttelse mot DEG)  │           │    ║
║  │  └──────────┬─────────────┘    └───────────┬────────────┘           │    ║
║  └─────────────┼──────────────────────────────┼────────────────────────┘    ║
║                │                              │                              ║
║  ┌─────────────┼──────────────────────────────┼────────────────────────┐    ║
║  │             │  LEVEL 2: SAFETY             │                         │    ║
║  │  ┌──────────▼─────────────┐    ┌───────────▼────────────┐           │    ║
║  │  │ 4️⃣ Risk Kernel (VETO)  │    │ 3️⃣ Data Integrity      │           │    ║
║  │  │    (Absolutt makt)     │    │ (Løgnedetektor)        │           │    ║
║  │  └──────────┬─────────────┘    └───────────┬────────────┘           │    ║
║  └─────────────┼──────────────────────────────┼────────────────────────┘    ║
║                │                              │                              ║
║  ┌─────────────▼──────────────────────────────▼────────────────────────┐    ║
║  │                    LEVEL 3: MARKET                                   │    ║
║  │            ┌────────────────────────────────┐                        │    ║
║  │            │ 2️⃣ Market Reality / Regime     │                        │    ║
║  │            │    (Markedets sannhet)         │                        │    ║
║  │            └──────────────┬─────────────────┘                        │    ║
║  └───────────────────────────┼──────────────────────────────────────────┘    ║
║                              │                                               ║
║  ┌───────────────────────────▼──────────────────────────────────────────┐    ║
║  │                    LEVEL 4-5: ALLOCATION                             │    ║
║  │  ┌────────────────────────┐    ┌────────────────────────┐           │    ║
║  │  │ 5️⃣ Capital Allocation  │───▶│ 7️⃣ Entry Qualification │           │    ║
║  │  │ (Hvor mye - aldri om)  │    │ (Siste filter)         │           │    ║
║  │  └────────────────────────┘    └──────────┬─────────────┘           │    ║
║  └───────────────────────────────────────────┼──────────────────────────┘    ║
║                                              │                              ║
║  ┌───────────────────────────────────────────▼──────────────────────────┐    ║
║  │                    LEVEL 6-7: EXECUTION                              │    ║
║  │  ┌────────────────────────┐    ┌────────────────────────┐           │    ║
║  │  │ 8️⃣ Execution Service   │───▶│ 9️⃣ Exit / Harvest Brain│           │    ║
║  │  │ (Markedets oversetter) │    │ (Der fondet tjener)    │           │    ║
║  │  └────────────────────────┘    └────────────────────────┘           │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                    OUTSIDE HIERARCHY                                 │    ║
║  │  ┌────────────────────────┐    ┌────────────────────────┐           │    ║
║  │  │ 6️⃣ Signal/AI Advisory  │    │ 1️⃣1️⃣ Audit & Ledger     │           │    ║
║  │  │ (Rådgiver, ikke sjef)  │    │ (Minne og sannhet)     │           │    ║
║  │  └────────────────────────┘    └────────────────────────┘           │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_architecture: Optional[MicroserviceArchitecture] = None


def get_microservice_architecture() -> MicroserviceArchitecture:
    """Get singleton microservice architecture instance."""
    global _architecture
    if _architecture is None:
        _architecture = MicroserviceArchitecture()
    return _architecture


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    arch = get_microservice_architecture()
    print(ARCHITECTURE_DIAGRAM)
    arch.print_architecture()
    
    # Validate
    validation = arch.validate_architecture()
    if validation["valid"]:
        print("✅ Architecture validated successfully")
    else:
        print(f"❌ Issues: {validation['issues']}")
