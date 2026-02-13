"""
🎯 MVP-ARKITEKTUR KOBLING (FAILURE → KOMPONENT)
===============================================
Direkte mapping fra failure scenarios til MVP-komponenter.

REGEL: Hvis en failure ikke kan mappes til én komponent → MVP er ufullstendig.

MVP-KJERNE (minimum, men komplett):
1. Data Integrity Sentinel
2. Market Regime Detector
3. Policy Engine
4. Risk Kernel (Fail-Closed)
5. Exit / Harvest Brain
6. Execution Optimizer
7. Audit Ledger
8. Human Override Lock
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MVP COMPONENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MVPComponent(str, Enum):
    """
    De 8 MVP-komponentene som utgjør minimum viable trading system.
    Hver har én klar ansvarsområde.
    """
    DATA_INTEGRITY_SENTINEL = "data_integrity_sentinel"
    MARKET_REGIME_DETECTOR = "market_regime_detector"
    POLICY_ENGINE = "policy_engine"
    RISK_KERNEL = "risk_kernel"
    EXIT_HARVEST_BRAIN = "exit_harvest_brain"
    EXECUTION_OPTIMIZER = "execution_optimizer"
    AUDIT_LEDGER = "audit_ledger"
    HUMAN_OVERRIDE_LOCK = "human_override_lock"
    
    # Extended (not MVP-critical but useful)
    PERFORMANCE_DRIFT_MONITOR = "performance_drift_monitor"
    EXCHANGE_HEALTH_MONITOR = "exchange_health_monitor"


class FailureHandlerAction(str, Enum):
    """Actions a component's handler can take."""
    FREEZE = "freeze"           # Full freeze, no trading
    COOLDOWN = "cooldown"       # Temporary pause with timer
    KILL_SWITCH = "kill_switch" # Full stop, needs restart protocol
    SAFE_MODE = "safe_mode"     # Limited operations
    PAUSE = "pause"             # Strategy pause
    STRATEGY_FREEZE = "strategy_freeze"  # Freeze specific strategy
    RATE_DOWN = "rate_down"     # Reduce execution rate
    ISOLATE = "isolate"         # Isolate failing component


@dataclass
class ComponentHandler:
    """Defines how a component handles a failure."""
    component: MVPComponent
    action: FailureHandlerAction
    description: str
    requires_human_review: bool = False
    max_auto_recovery_attempts: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# FAILURE → COMPONENT MAPPING (KOBLINGSTABELL)
# ═══════════════════════════════════════════════════════════════════════════════

FAILURE_TO_COMPONENT_MAP: Dict[str, ComponentHandler] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔴 CLASS A - Kill-Switch Level
    # ═══════════════════════════════════════════════════════════════════════════
    
    "A1_CAPITAL_BREACH": ComponentHandler(
        component=MVPComponent.RISK_KERNEL,
        action=FailureHandlerAction.KILL_SWITCH,
        description="Risk Kernel håndhever kapitalbeskyttelse",
        requires_human_review=True,
    ),
    
    "A2_LEVERAGE_EXCEEDED": ComponentHandler(
        component=MVPComponent.RISK_KERNEL,
        action=FailureHandlerAction.KILL_SWITCH,
        description="Risk Kernel håndhever leverage-grenser",
        requires_human_review=True,
    ),
    
    "A3_DATA_COLLAPSE": ComponentHandler(
        component=MVPComponent.DATA_INTEGRITY_SENTINEL,
        action=FailureHandlerAction.SAFE_MODE,
        description="Data Sentinel oppdager datakvalitetsproblemer",
    ),
    
    "A4_EXECUTION_FAILURE": ComponentHandler(
        component=MVPComponent.EXECUTION_OPTIMIZER,
        action=FailureHandlerAction.RATE_DOWN,
        description="Execution Optimizer håndterer execution errors",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟠 CLASS B - Strategic Pause
    # ═══════════════════════════════════════════════════════════════════════════
    
    "B1_REGIME_UNCERTAINTY": ComponentHandler(
        component=MVPComponent.MARKET_REGIME_DETECTOR,
        action=FailureHandlerAction.PAUSE,
        description="Regime Detector oppdager usikre markedsforhold",
    ),
    
    "B2_EDGE_DECAY": ComponentHandler(
        component=MVPComponent.PERFORMANCE_DRIFT_MONITOR,
        action=FailureHandlerAction.STRATEGY_FREEZE,
        description="Performance Monitor oppdager edge-forfall",
        requires_human_review=True,
    ),
    
    "B3_DRAWDOWN_VELOCITY": ComponentHandler(
        component=MVPComponent.RISK_KERNEL,
        action=FailureHandlerAction.PAUSE,
        description="Risk Kernel oppdager rask drawdown",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟡 CLASS C - Exit-Only Mode
    # ═══════════════════════════════════════════════════════════════════════════
    
    "C1_CONCENTRATION_LIMIT": ComponentHandler(
        component=MVPComponent.POLICY_ENGINE,
        action=FailureHandlerAction.SAFE_MODE,
        description="Policy Engine håndhever concentration limits",
    ),
    
    "C2_TIME_BASED_STRESS": ComponentHandler(
        component=MVPComponent.EXIT_HARVEST_BRAIN,
        action=FailureHandlerAction.SAFE_MODE,
        description="Exit Brain håndterer tidsbasert stress",
    ),
    
    "C3_FUNDING_BLEED": ComponentHandler(
        component=MVPComponent.EXIT_HARVEST_BRAIN,
        action=FailureHandlerAction.SAFE_MODE,
        description="Exit Brain håndterer funding-kostnad",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟣 CLASS D - Human Risk
    # ═══════════════════════════════════════════════════════════════════════════
    
    "D1_MANUAL_OVERRIDE": ComponentHandler(
        component=MVPComponent.HUMAN_OVERRIDE_LOCK,
        action=FailureHandlerAction.FREEZE,
        description="Override Lock blokkerer manuelle endringer",
        requires_human_review=True,
    ),
    
    "D2_AI_MISBEHAVIOR": ComponentHandler(
        component=MVPComponent.POLICY_ENGINE,
        action=FailureHandlerAction.COOLDOWN,
        description="Policy Engine regulerer AI-handlinger",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔵 CLASS E - Infrastructure
    # ═══════════════════════════════════════════════════════════════════════════
    
    "E1_EXCHANGE_RISK": ComponentHandler(
        component=MVPComponent.EXCHANGE_HEALTH_MONITOR,
        action=FailureHandlerAction.ISOLATE,
        description="Exchange Monitor isolerer ustabil exchange",
    ),
    
    "E2_INFRASTRUCTURE_FAIL": ComponentHandler(
        component=MVPComponent.DATA_INTEGRITY_SENTINEL,
        action=FailureHandlerAction.SAFE_MODE,
        description="Data Sentinel oppdager infrastrukturproblemer",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MVP COMPONENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentStatus:
    """Runtime status of an MVP component."""
    component: MVPComponent
    healthy: bool = True
    last_check: Optional[datetime] = None
    active_failures: List[str] = field(default_factory=list)
    total_failures_handled: int = 0


class MVPArchitecture:
    """
    MVP Architecture manager - ensures all failure scenarios
    map to exactly one component.
    
    REGEL: Hvis en failure ikke kan mappes til én komponent → MVP er ufullstendig.
    """
    
    def __init__(self):
        self._components: Dict[MVPComponent, ComponentStatus] = {
            comp: ComponentStatus(component=comp)
            for comp in MVPComponent
        }
        self._failure_log: List[Dict[str, Any]] = []
    
    def get_handler_for_failure(self, failure_type: str) -> Optional[ComponentHandler]:
        """
        Get the MVP component handler for a failure type.
        
        Args:
            failure_type: The failure scenario ID (e.g., "A1_CAPITAL_BREACH")
        
        Returns:
            ComponentHandler if mapped, None otherwise
        """
        handler = FAILURE_TO_COMPONENT_MAP.get(failure_type)
        
        if not handler:
            logger.error(f"❌ UNMAPPED FAILURE: {failure_type} - MVP is INCOMPLETE!")
            return None
        
        return handler
    
    def route_failure(self, failure_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a failure to its designated MVP component.
        
        Args:
            failure_type: The failure scenario ID
            context: Additional context about the failure
        
        Returns:
            Dict with routing result
        """
        handler = self.get_handler_for_failure(failure_type)
        
        if not handler:
            return {
                "success": False,
                "error": "UNMAPPED_FAILURE",
                "failure_type": failure_type,
                "mvp_complete": False,
            }
        
        # Update component status
        status = self._components[handler.component]
        status.active_failures.append(failure_type)
        status.total_failures_handled += 1
        
        # Log the routing
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "failure_type": failure_type,
            "component": handler.component.value,
            "action": handler.action.value,
            "requires_human_review": handler.requires_human_review,
            "context": context,
        }
        self._failure_log.append(log_entry)
        
        logger.info(
            f"🎯 Routed {failure_type} → {handler.component.value} "
            f"(Action: {handler.action.value})"
        )
        
        return {
            "success": True,
            "component": handler.component.value,
            "action": handler.action.value,
            "description": handler.description,
            "requires_human_review": handler.requires_human_review,
        }
    
    def resolve_failure(self, failure_type: str, component: MVPComponent):
        """Mark a failure as resolved in a component."""
        status = self._components[component]
        if failure_type in status.active_failures:
            status.active_failures.remove(failure_type)
    
    def validate_mvp_completeness(self) -> Dict[str, Any]:
        """
        Validate that all known failure scenarios are mapped.
        
        Returns:
            Dict with validation result
        """
        from .failure_scenarios import FailureScenario
        
        unmapped = []
        mapped = []
        
        for failure in FailureScenario:
            failure_key = failure.name
            if failure_key in FAILURE_TO_COMPONENT_MAP:
                mapped.append(failure_key)
            else:
                unmapped.append(failure_key)
        
        is_complete = len(unmapped) == 0
        
        return {
            "mvp_complete": is_complete,
            "total_failures": len(FailureScenario),
            "mapped_count": len(mapped),
            "unmapped_count": len(unmapped),
            "unmapped_failures": unmapped,
            "message": "MVP is COMPLETE" if is_complete else f"MVP INCOMPLETE: {len(unmapped)} unmapped failures",
        }
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MVP components."""
        return {
            comp.value: {
                "healthy": status.healthy,
                "active_failures": status.active_failures,
                "total_handled": status.total_failures_handled,
            }
            for comp, status in self._components.items()
        }
    
    def print_architecture_report(self):
        """Print MVP architecture report."""
        print("\n" + "=" * 70)
        print("🎯 MVP ARCHITECTURE STATUS")
        print("=" * 70)
        
        # Validation
        validation = self.validate_mvp_completeness()
        status = "✅ COMPLETE" if validation["mvp_complete"] else "❌ INCOMPLETE"
        print(f"\nMVP Status: {status}")
        print(f"Mapped: {validation['mapped_count']}/{validation['total_failures']}")
        
        if validation["unmapped_failures"]:
            print(f"\n⚠️ Unmapped failures:")
            for f in validation["unmapped_failures"]:
                print(f"   - {f}")
        
        # Component overview
        print("\n" + "-" * 70)
        print("MVP COMPONENTS:")
        print("-" * 70)
        
        for comp in MVPComponent:
            status = self._components[comp]
            health = "✅" if status.healthy else "❌"
            active = len(status.active_failures)
            print(f"  {health} {comp.value}: {status.total_failures_handled} handled, {active} active")
        
        # Mapping table
        print("\n" + "-" * 70)
        print("FAILURE → COMPONENT MAPPING:")
        print("-" * 70)
        
        for failure, handler in FAILURE_TO_COMPONENT_MAP.items():
            human = "👤" if handler.requires_human_review else "  "
            print(f"  {failure:25} → {handler.component.value:25} {human} [{handler.action.value}]")
        
        print("\n" + "=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KOBLINGSTABELL (QUICK REFERENCE)
# ═══════════════════════════════════════════════════════════════════════════════

KOBLINGSTABELL = """
╔══════════════════════════╦════════════════════════════╦═══════════════════╗
║ FAILURE                  ║ MVP-KOMPONENT              ║ HÅNDHEVER         ║
╠══════════════════════════╬════════════════════════════╬═══════════════════╣
║ Menneskelig override     ║ Human Override Lock        ║ Freeze + cooldown ║
║ Risiko-brudd            ║ Risk Kernel                ║ Kill-switch       ║
║ Datafeil                ║ Data Integrity Sentinel    ║ SAFE MODE         ║
║ Regime-usikkerhet       ║ Market Regime Detector     ║ Pause             ║
║ Edge-forfall            ║ Performance/Drift Monitor  ║ Strategi-frys     ║
║ Execution-feil          ║ Execution Optimizer        ║ Rate-down / pause ║
║ Exchange-anomali        ║ Exchange Health Monitor    ║ Isolasjon         ║
╚══════════════════════════╩════════════════════════════╩═══════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_mvp_architecture: Optional[MVPArchitecture] = None


def get_mvp_architecture() -> MVPArchitecture:
    """Get singleton MVP architecture instance."""
    global _mvp_architecture
    if _mvp_architecture is None:
        _mvp_architecture = MVPArchitecture()
    return _mvp_architecture


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mvp = get_mvp_architecture()
    mvp.print_architecture_report()
    print(KOBLINGSTABELL)
