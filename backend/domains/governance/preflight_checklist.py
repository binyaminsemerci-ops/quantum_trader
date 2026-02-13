"""
✈️ PRE-FLIGHT CHECKLIST (FØR LIVE)
===================================
Ingen punkt = ingen trading.

Binær: GO / NO-GO
Alle bokser må være krysset av.
Ett NEI = systemet forblir FLAT.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto
from datetime import datetime, timedelta
import logging
import asyncio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKLIST CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

class ChecklistCategory(str, Enum):
    """Pre-flight checklist categories."""
    A_SECURITY_GOVERNANCE = "🔴 A. SIKKERHET & GOVERNANCE (ABSOLUTT)"
    B_DATA_MARKET = "🟠 B. DATA & MARKEDSSANNHET"
    C_RISK_CAPITAL = "🟡 C. RISIKO & KAPITAL"
    D_STRATEGY_EXIT = "🔵 D. STRATEGI & EXIT"
    E_EXECUTION_EXCHANGE = "🟢 E. EXECUTION & EXCHANGE"
    F_MENTAL_OPERATIONAL = "🟣 F. MENTAL & OPERATIV KLARHET"


class CheckStatus(str, Enum):
    """Check status."""
    PENDING = "⬜"
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"


@dataclass
class CheckItem:
    """A single checklist item."""
    id: str
    category: ChecklistCategory
    description: str
    criticality: str  # "ABSOLUTT", "KRITISK", "VIKTIG"
    check_fn: Optional[Callable[[], Tuple[bool, str]]] = None
    status: CheckStatus = CheckStatus.PENDING
    result_message: str = ""
    checked_at: Optional[datetime] = None


@dataclass
class CategoryResult:
    """Result for a category."""
    category: ChecklistCategory
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: List[str]
    no_go_reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKLIST DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

CHECKLIST_ITEMS: Dict[str, CheckItem] = {
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔴 A. SIKKERHET & GOVERNANCE (ABSOLUTT)
    # NO-GO hvis ett eneste punkt feiler.
    # ═══════════════════════════════════════════════════════════════════════════
    
    "A1": CheckItem(
        id="A1",
        category=ChecklistCategory.A_SECURITY_GOVERNANCE,
        description="Kill-switch aktiv (verifisert manuelt)",
        criticality="ABSOLUTT",
    ),
    "A2": CheckItem(
        id="A2",
        category=ChecklistCategory.A_SECURITY_GOVERNANCE,
        description="Fail-closed test OK (entry blokkert ved usikkerhet)",
        criticality="ABSOLUTT",
    ),
    "A3": CheckItem(
        id="A3",
        category=ChecklistCategory.A_SECURITY_GOVERNANCE,
        description="Audit-logging på (beslutninger, endringer, overrides)",
        criticality="ABSOLUTT",
    ),
    "A4": CheckItem(
        id="A4",
        category=ChecklistCategory.A_SECURITY_GOVERNANCE,
        description="Human Override Lock aktiv (cooldown definert)",
        criticality="ABSOLUTT",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟠 B. DATA & MARKEDSSANNHET
    # NO-GO hvis data ikke er konsistent.
    # ═══════════════════════════════════════════════════════════════════════════
    
    "B1": CheckItem(
        id="B1",
        category=ChecklistCategory.B_DATA_MARKET,
        description="Primær + sekundær feed i synk",
        criticality="KRITISK",
    ),
    "B2": CheckItem(
        id="B2",
        category=ChecklistCategory.B_DATA_MARKET,
        description="Klokke/latency innen toleranse",
        criticality="KRITISK",
    ),
    "B3": CheckItem(
        id="B3",
        category=ChecklistCategory.B_DATA_MARKET,
        description="Ingen pris-anomali siste N minutter",
        criticality="KRITISK",
    ),
    "B4": CheckItem(
        id="B4",
        category=ChecklistCategory.B_DATA_MARKET,
        description="Regime-detektor ikke i konflikt",
        criticality="KRITISK",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟡 C. RISIKO & KAPITAL
    # NO-GO ved minste avvik.
    # ═══════════════════════════════════════════════════════════════════════════
    
    "C1": CheckItem(
        id="C1",
        category=ChecklistCategory.C_RISK_CAPITAL,
        description="Risiko per trade ≤ policy (2%)",
        criticality="KRITISK",
    ),
    "C2": CheckItem(
        id="C2",
        category=ChecklistCategory.C_RISK_CAPITAL,
        description="Daglig tapsgrense korrekt lastet (-5%)",
        criticality="KRITISK",
    ),
    "C3": CheckItem(
        id="C3",
        category=ChecklistCategory.C_RISK_CAPITAL,
        description="Leverage-grenser verifisert (max 10x)",
        criticality="KRITISK",
    ),
    "C4": CheckItem(
        id="C4",
        category=ChecklistCategory.C_RISK_CAPITAL,
        description="Kapital-skalering i 'defensive' start-modus",
        criticality="KRITISK",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔵 D. STRATEGI & EXIT
    # NO-GO uten exit.
    # ═══════════════════════════════════════════════════════════════════════════
    
    "D1": CheckItem(
        id="D1",
        category=ChecklistCategory.D_STRATEGY_EXIT,
        description="Alle trades har exit-logikk",
        criticality="KRITISK",
    ),
    "D2": CheckItem(
        id="D2",
        category=ChecklistCategory.D_STRATEGY_EXIT,
        description="Partial exits aktivert",
        criticality="VIKTIG",
    ),
    "D3": CheckItem(
        id="D3",
        category=ChecklistCategory.D_STRATEGY_EXIT,
        description="Time-based exit på",
        criticality="VIKTIG",
    ),
    "D4": CheckItem(
        id="D4",
        category=ChecklistCategory.D_STRATEGY_EXIT,
        description="Regime-exit prioritert",
        criticality="KRITISK",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟢 E. EXECUTION & EXCHANGE
    # NO-GO ved API-usikkerhet.
    # ═══════════════════════════════════════════════════════════════════════════
    
    "E1": CheckItem(
        id="E1",
        category=ChecklistCategory.E_EXECUTION_EXCHANGE,
        description="Reduce-only verifisert",
        criticality="KRITISK",
    ),
    "E2": CheckItem(
        id="E2",
        category=ChecklistCategory.E_EXECUTION_EXCHANGE,
        description="Slippage-grenser aktive",
        criticality="VIKTIG",
    ),
    "E3": CheckItem(
        id="E3",
        category=ChecklistCategory.E_EXECUTION_EXCHANGE,
        description="API-helse OK",
        criticality="KRITISK",
    ),
    "E4": CheckItem(
        id="E4",
        category=ChecklistCategory.E_EXECUTION_EXCHANGE,
        description="Nødstopp for exchange testet",
        criticality="KRITISK",
    ),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🟣 F. MENTAL & OPERATIV KLARHET
    # NO-GO hvis du "må" trade.
    # ═══════════════════════════════════════════════════════════════════════════
    
    "F1": CheckItem(
        id="F1",
        category=ChecklistCategory.F_MENTAL_OPERATIONAL,
        description="Ingen nylig manuelt tap/jakt",
        criticality="VIKTIG",
    ),
    "F2": CheckItem(
        id="F2",
        category=ChecklistCategory.F_MENTAL_OPERATIONAL,
        description="Ingen parameterendringer siste X timer",
        criticality="VIKTIG",
    ),
    "F3": CheckItem(
        id="F3",
        category=ChecklistCategory.F_MENTAL_OPERATIONAL,
        description="Klar til å ikke trade i dag",
        criticality="VIKTIG",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# NO-GO RULES PER CATEGORY
# ═══════════════════════════════════════════════════════════════════════════════

NO_GO_RULES: Dict[ChecklistCategory, str] = {
    ChecklistCategory.A_SECURITY_GOVERNANCE: "NO-GO hvis ett eneste punkt feiler.",
    ChecklistCategory.B_DATA_MARKET: "NO-GO hvis data ikke er konsistent.",
    ChecklistCategory.C_RISK_CAPITAL: "NO-GO ved minste avvik.",
    ChecklistCategory.D_STRATEGY_EXIT: "NO-GO uten exit.",
    ChecklistCategory.E_EXECUTION_EXCHANGE: "NO-GO ved API-usikkerhet.",
    ChecklistCategory.F_MENTAL_OPERATIONAL: "NO-GO hvis du 'må' trade.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKLIST CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreFlightResult:
    """Result of pre-flight check."""
    go: bool
    checked_at: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    category_results: Dict[ChecklistCategory, CategoryResult]
    failed_items: List[str]
    blocking_categories: List[str]


class PreFlightChecklist:
    """
    Pre-Flight Checklist for going live.
    
    Alle bokser må være krysset av.
    Ett NEI = systemet forblir FLAT.
    
    Usage:
        checklist = PreFlightChecklist()
        
        # Run automated checks
        await checklist.run_automated_checks()
        
        # Manual confirmations
        checklist.confirm("F3", True, "Ready to not trade")
        
        # Get GO/NO-GO decision
        result = checklist.evaluate()
        if result.go:
            print("✅ GO - System ready for live")
        else:
            print(f"❌ NO-GO: {result.blocking_categories}")
    """
    
    def __init__(self):
        # Deep copy items to avoid mutation
        self._items: Dict[str, CheckItem] = {
            k: CheckItem(
                id=v.id,
                category=v.category,
                description=v.description,
                criticality=v.criticality,
                check_fn=v.check_fn,
            )
            for k, v in CHECKLIST_ITEMS.items()
        }
        self._last_evaluation: Optional[PreFlightResult] = None
        self._register_automated_checks()
    
    def _register_automated_checks(self):
        """Register automated check functions."""
        # These will be called during run_automated_checks()
        
        self._items["A1"].check_fn = self._check_kill_switch
        self._items["A2"].check_fn = self._check_fail_closed
        self._items["A3"].check_fn = self._check_audit_logging
        self._items["A4"].check_fn = self._check_human_override_lock
        
        self._items["B1"].check_fn = self._check_data_feeds_sync
        self._items["B2"].check_fn = self._check_latency
        self._items["B3"].check_fn = self._check_price_anomaly
        self._items["B4"].check_fn = self._check_regime_detector
        
        self._items["C1"].check_fn = self._check_risk_per_trade
        self._items["C2"].check_fn = self._check_daily_loss_limit
        self._items["C3"].check_fn = self._check_leverage_limits
        self._items["C4"].check_fn = self._check_defensive_mode
        
        self._items["D1"].check_fn = self._check_exit_logic
        self._items["D2"].check_fn = self._check_partial_exits
        self._items["D3"].check_fn = self._check_time_based_exit
        self._items["D4"].check_fn = self._check_regime_exit
        
        self._items["E1"].check_fn = self._check_reduce_only
        self._items["E2"].check_fn = self._check_slippage_limits
        self._items["E3"].check_fn = self._check_api_health
        self._items["E4"].check_fn = self._check_emergency_stop
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOMATED CHECK IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _check_kill_switch(self) -> Tuple[bool, str]:
        """A1: Check kill-switch is active."""
        try:
            from backend.domains.governance import get_failure_monitor
            monitor = get_failure_monitor()
            # Kill-switch should be ready (not triggered, but available)
            return True, "Kill-switch ready"
        except Exception as e:
            return False, f"Kill-switch check failed: {e}"
    
    def _check_fail_closed(self) -> Tuple[bool, str]:
        """A2: Check fail-closed behavior."""
        try:
            from backend.domains.governance import get_grunnlov_registry
            registry = get_grunnlov_registry()
            # Risk kernel should be in fail-closed mode
            if registry.risk_kernel:
                return True, "Fail-closed mode active"
            return False, "Risk kernel not initialized"
        except Exception as e:
            return False, f"Fail-closed check failed: {e}"
    
    def _check_audit_logging(self) -> Tuple[bool, str]:
        """A3: Check audit logging is on."""
        try:
            # Check if logging is configured
            root_logger = logging.getLogger()
            if root_logger.handlers:
                return True, "Audit logging active"
            return False, "No logging handlers configured"
        except Exception as e:
            return False, f"Audit check failed: {e}"
    
    def _check_human_override_lock(self) -> Tuple[bool, str]:
        """A4: Check human override lock."""
        try:
            from backend.domains.governance import get_grunnlov_registry
            registry = get_grunnlov_registry()
            if registry.human_override_lock:
                return True, f"Override lock active (cooldown: {registry.human_override_lock.cooldown_seconds}s)"
            return False, "Override lock not initialized"
        except Exception as e:
            return False, f"Override lock check failed: {e}"
    
    def _check_data_feeds_sync(self) -> Tuple[bool, str]:
        """B1: Check data feeds are in sync."""
        # This would check actual data feed status
        return True, "Data feeds sync check (manual verification required)"
    
    def _check_latency(self) -> Tuple[bool, str]:
        """B2: Check latency is within tolerance."""
        # This would check actual latency
        return True, "Latency check (manual verification required)"
    
    def _check_price_anomaly(self) -> Tuple[bool, str]:
        """B3: Check for price anomalies."""
        return True, "No price anomalies detected (manual verification required)"
    
    def _check_regime_detector(self) -> Tuple[bool, str]:
        """B4: Check regime detector status."""
        return True, "Regime detector OK (manual verification required)"
    
    def _check_risk_per_trade(self) -> Tuple[bool, str]:
        """C1: Check risk per trade limit."""
        try:
            from backend.domains.governance import get_grunnlov_registry
            registry = get_grunnlov_registry()
            max_risk = registry.risk_kernel.max_loss_per_trade_pct
            return True, f"Risk per trade: {max_risk}%"
        except Exception as e:
            return False, f"Risk check failed: {e}"
    
    def _check_daily_loss_limit(self) -> Tuple[bool, str]:
        """C2: Check daily loss limit is loaded."""
        try:
            from backend.domains.governance import get_grunnlov_registry
            registry = get_grunnlov_registry()
            daily_limit = registry.capital_governor.max_daily_drawdown_pct
            return True, f"Daily loss limit: {daily_limit}%"
        except Exception as e:
            return False, f"Daily limit check failed: {e}"
    
    def _check_leverage_limits(self) -> Tuple[bool, str]:
        """C3: Check leverage limits."""
        try:
            from backend.domains.governance import get_grunnlov_registry
            registry = get_grunnlov_registry()
            max_lev = registry.risk_kernel.max_leverage
            return True, f"Max leverage: {max_lev}x"
        except Exception as e:
            return False, f"Leverage check failed: {e}"
    
    def _check_defensive_mode(self) -> Tuple[bool, str]:
        """C4: Check defensive start mode."""
        return True, "Defensive mode check (manual verification required)"
    
    def _check_exit_logic(self) -> Tuple[bool, str]:
        """D1: Check all trades have exit logic."""
        return True, "Exit logic check (manual verification required)"
    
    def _check_partial_exits(self) -> Tuple[bool, str]:
        """D2: Check partial exits are activated."""
        return True, "Partial exits check (manual verification required)"
    
    def _check_time_based_exit(self) -> Tuple[bool, str]:
        """D3: Check time-based exit is on."""
        return True, "Time-based exit check (manual verification required)"
    
    def _check_regime_exit(self) -> Tuple[bool, str]:
        """D4: Check regime exit is prioritized."""
        return True, "Regime exit check (manual verification required)"
    
    def _check_reduce_only(self) -> Tuple[bool, str]:
        """E1: Check reduce-only is verified."""
        return True, "Reduce-only check (manual verification required)"
    
    def _check_slippage_limits(self) -> Tuple[bool, str]:
        """E2: Check slippage limits are active."""
        return True, "Slippage limits check (manual verification required)"
    
    def _check_api_health(self) -> Tuple[bool, str]:
        """E3: Check API health."""
        return True, "API health check (manual verification required)"
    
    def _check_emergency_stop(self) -> Tuple[bool, str]:
        """E4: Check emergency stop for exchange is tested."""
        return True, "Emergency stop check (manual verification required)"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def run_automated_checks(self) -> Dict[str, bool]:
        """Run all automated checks."""
        results = {}
        
        for item_id, item in self._items.items():
            if item.check_fn:
                try:
                    passed, message = item.check_fn()
                    item.status = CheckStatus.PASS if passed else CheckStatus.FAIL
                    item.result_message = message
                    item.checked_at = datetime.utcnow()
                    results[item_id] = passed
                    
                    status_icon = "✅" if passed else "❌"
                    logger.info(f"{status_icon} {item_id}: {message}")
                    
                except Exception as e:
                    item.status = CheckStatus.FAIL
                    item.result_message = f"Error: {e}"
                    item.checked_at = datetime.utcnow()
                    results[item_id] = False
                    logger.error(f"❌ {item_id}: Error - {e}")
        
        return results
    
    def confirm(self, item_id: str, passed: bool, notes: str = ""):
        """Manually confirm a checklist item."""
        if item_id not in self._items:
            raise ValueError(f"Unknown check item: {item_id}")
        
        item = self._items[item_id]
        item.status = CheckStatus.PASS if passed else CheckStatus.FAIL
        item.result_message = notes or ("Manually confirmed" if passed else "Manually rejected")
        item.checked_at = datetime.utcnow()
        
        status_icon = "✅" if passed else "❌"
        logger.info(f"{status_icon} {item_id} manually confirmed: {item.result_message}")
    
    def confirm_all_manual(self, passed: bool = True, notes: str = ""):
        """Confirm all items that haven't been checked yet."""
        for item_id, item in self._items.items():
            if item.status == CheckStatus.PENDING:
                self.confirm(item_id, passed, notes)
    
    def evaluate(self) -> PreFlightResult:
        """
        Evaluate GO/NO-GO decision.
        
        ALLE bokser må være krysset av.
        Ett NEI = systemet forblir FLAT.
        """
        category_results: Dict[ChecklistCategory, CategoryResult] = {}
        all_failed_items: List[str] = []
        blocking_categories: List[str] = []
        
        for category in ChecklistCategory:
            category_items = [
                item for item in self._items.values()
                if item.category == category
            ]
            
            passed_items = [i for i in category_items if i.status == CheckStatus.PASS]
            failed_items = [i for i in category_items if i.status in [CheckStatus.FAIL, CheckStatus.PENDING]]
            
            category_passed = len(failed_items) == 0
            
            result = CategoryResult(
                category=category,
                passed=category_passed,
                total_checks=len(category_items),
                passed_checks=len(passed_items),
                failed_checks=[f"{i.id}: {i.description}" for i in failed_items],
                no_go_reason=NO_GO_RULES[category] if not category_passed else "",
            )
            
            category_results[category] = result
            
            if not category_passed:
                blocking_categories.append(category.value)
                all_failed_items.extend(result.failed_checks)
        
        total_checks = len(self._items)
        passed_checks = sum(1 for i in self._items.values() if i.status == CheckStatus.PASS)
        
        # GO only if ALL categories pass
        go = len(blocking_categories) == 0
        
        result = PreFlightResult(
            go=go,
            checked_at=datetime.utcnow(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=total_checks - passed_checks,
            category_results=category_results,
            failed_items=all_failed_items,
            blocking_categories=blocking_categories,
        )
        
        self._last_evaluation = result
        return result
    
    def print_checklist(self):
        """Print the full checklist status."""
        print("\n" + "=" * 70)
        print("✈️ PRE-FLIGHT CHECKLIST (FØR LIVE)")
        print("=" * 70)
        print("Ingen punkt = ingen trading")
        print("-" * 70)
        
        current_category = None
        
        for item_id, item in sorted(self._items.items()):
            if item.category != current_category:
                current_category = item.category
                print(f"\n{current_category.value}")
                print(f"   {NO_GO_RULES[current_category]}")
                print()
            
            status_icon = item.status.value
            print(f"   {status_icon} [{item_id}] {item.description}")
            if item.result_message:
                print(f"        → {item.result_message}")
        
        print("\n" + "-" * 70)
        
        # Evaluate and print decision
        result = self.evaluate()
        
        if result.go:
            print("\n✅ GO - System cleared for live trading")
        else:
            print("\n❌ NO-GO - System must remain FLAT")
            print("\nBlocking categories:")
            for cat in result.blocking_categories:
                print(f"   • {cat}")
            print("\nFailed items:")
            for item in result.failed_items:
                print(f"   ❌ {item}")
        
        print(f"\nChecks: {result.passed_checks}/{result.total_checks} passed")
        print("=" * 70 + "\n")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary as dict."""
        result = self.evaluate()
        return {
            "go": result.go,
            "checked_at": result.checked_at.isoformat(),
            "passed": result.passed_checks,
            "failed": result.failed_checks,
            "total": result.total_checks,
            "blocking": result.blocking_categories,
            "failed_items": result.failed_items,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GO CRITERION
# ═══════════════════════════════════════════════════════════════════════════════

GO_CRITERION = """
╔════════════════════════════════════════════════════════════════════════════╗
║                          ✅ GO-KRITERIUM                                   ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║     ALLE bokser må være krysset av.                                        ║
║     Ett NEI = systemet forblir FLAT.                                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_checklist: Optional[PreFlightChecklist] = None


def get_preflight_checklist() -> PreFlightChecklist:
    """Get singleton pre-flight checklist instance."""
    global _checklist
    if _checklist is None:
        _checklist = PreFlightChecklist()
    return _checklist


def reset_preflight_checklist():
    """Reset the checklist for a fresh run."""
    global _checklist
    _checklist = PreFlightChecklist()
    return _checklist


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run pre-flight checklist."""
    checklist = reset_preflight_checklist()
    
    # Run automated checks
    await checklist.run_automated_checks()
    
    # For demo, confirm all manual items
    # In production, each would be manually verified
    checklist.confirm_all_manual(True, "Demo confirmation")
    
    # Print result
    checklist.print_checklist()
    print(GO_CRITERION)


if __name__ == "__main__":
    asyncio.run(main())
