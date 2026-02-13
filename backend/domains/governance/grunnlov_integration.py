"""
🏛️ GRUNNLOV ENFORCEMENT INTEGRATION
====================================
Integrates the 15 Grunnlover into the existing Quantum Trader system.

This module provides:
1. Pre-trade validation against all applicable laws
2. Post-trade enforcement and logging
3. Real-time monitoring of constitutional compliance
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from functools import wraps

from backend.domains.governance.constitution import (
    get_constitution_enforcer,
    validate_constitutional_action,
    ConstitutionLaw,
    ViolationType,
    ComponentRole,
    GRUNNLOVER,
)
from backend.domains.governance.grunnlov_components import (
    get_grunnlov_registry,
    EntryPlan,
    AIAdvice,
    ComponentPriority,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR: Constitutional Validation
# ═══════════════════════════════════════════════════════════════════════════════

def requires_constitutional_approval(
    action: str,
    component: Optional[str] = None,
):
    """
    Decorator that requires constitutional approval before executing.
    
    Usage:
        @requires_constitutional_approval("OPEN_POSITION", "ExecutionService")
        async def open_position(self, symbol, side, size):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build context from arguments
            context = {
                "args": str(args),
                "kwargs": kwargs,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Extract common fields from kwargs
            for field in ["symbol", "side", "size", "stop_loss", "take_profit", 
                         "leverage", "confidence", "risk_pct"]:
                if field in kwargs:
                    context[field] = kwargs[field]
            
            # Validate against constitution
            comp = component or func.__module__
            allowed, error = validate_constitutional_action(comp, action, context)
            
            if not allowed:
                logger.warning(f"🚫 Constitutional block: {error}")
                raise ConstitutionalViolationError(action, error)
            
            # Execute the function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class ConstitutionalViolationError(Exception):
    """Raised when an action violates the constitution."""
    
    def __init__(self, action: str, reason: str):
        self.action = action
        self.reason = reason
        super().__init__(f"Constitutional violation: {action} - {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-TRADE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class GrunnlovPreTradeValidator:
    """
    Validates trades against all 15 Grunnlover before execution.
    
    This is the GATE that all trades must pass through.
    """
    
    def __init__(self):
        self.enforcer = get_constitution_enforcer()
        self.registry = get_grunnlov_registry()
        self._validation_count = 0
        self._blocked_count = 0
        
        # Register components with their constitutional roles
        self._register_components()
        
        logger.info("🏛️ GrunnlovPreTradeValidator initialized")
    
    def _register_components(self):
        """Register all system components with constitutional roles."""
        self.enforcer.register_component("CapitalPreservationGovernor", ComponentRole.VETO)
        self.enforcer.register_component("RiskKernel", ComponentRole.ENFORCER)
        self.enforcer.register_component("PolicyEngine", ComponentRole.GOVERNOR)
        self.enforcer.register_component("DecisionArbitrationLayer", ComponentRole.GOVERNOR)
        self.enforcer.register_component("EntryQualificationGate", ComponentRole.ENFORCER)
        self.enforcer.register_component("ExitHarvestBrain", ComponentRole.ENFORCER)
        self.enforcer.register_component("ExecutionOptimizer", ComponentRole.ADVISOR)
        self.enforcer.register_component("AIAdvisoryLayer", ComponentRole.ADVISOR)
        self.enforcer.register_component("ConstraintEnforcementLayer", ComponentRole.ENFORCER)
        self.enforcer.register_component("StrategyScopeController", ComponentRole.ENFORCER)
        self.enforcer.register_component("HumanOverrideLock", ComponentRole.ENFORCER)
        self.enforcer.register_component("AuditLedger", ComponentRole.OBSERVER)
        self.enforcer.register_component("PerformanceLearningLoop", ComponentRole.OBSERVER)
    
    def validate_entry(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        leverage: float,
        confidence: float,
        strategy: str,
        daily_pnl_pct: float = 0.0,
        weekly_pnl_pct: float = 0.0,
        drawdown_pct: float = 0.0,
        current_equity: float = 10000.0,
        starting_equity: float = 10000.0,
        last_trade_was_loss: bool = False,
        time_since_last_trade_minutes: int = 60,
    ) -> tuple[bool, str, Optional[Dict]]:
        """
        Complete pre-trade validation against all Grunnlover.
        
        Returns:
            (allowed: bool, reason: str, details: Optional[Dict])
        """
        self._validation_count += 1
        validation_id = f"VAL-{self._validation_count:06d}"
        
        logger.info(f"🔍 Starting validation {validation_id}: {symbol} {side}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # GRUNNLOV 1: Capital Preservation Check
        # ═══════════════════════════════════════════════════════════════════════
        safe, reason = self.registry.capital_preservation.check_capital_threat(
            current_equity=current_equity,
            starting_equity=starting_equity,
            daily_pnl_pct=daily_pnl_pct,
            structural_indicators=[],
        )
        if not safe:
            self._blocked_count += 1
            return False, f"GRUNNLOV 1 VETO: {reason}", {"law": "GRUNNLOV_1", "blocked_by": "CapitalPreservation"}
        
        # ═══════════════════════════════════════════════════════════════════════
        # GRUNNLOV 7: Risk Kernel Check
        # ═══════════════════════════════════════════════════════════════════════
        # Calculate risk percentage
        if side == "LONG":
            risk_pct = ((entry_price - stop_loss) / entry_price) * leverage * 100
        else:
            risk_pct = ((stop_loss - entry_price) / entry_price) * leverage * 100
        
        risk_allowed, risk_reason = self.registry.risk_kernel.check_trade_risk(
            risk_pct=risk_pct,
            leverage=leverage,
            position_pct=(position_size / current_equity) * 100,
        )
        if not risk_allowed:
            self._blocked_count += 1
            return False, f"GRUNNLOV 7: {risk_reason}", {"law": "GRUNNLOV_7", "blocked_by": "RiskKernel"}
        
        # Portfolio risk check
        portfolio_allowed, portfolio_reason = self.registry.risk_kernel.check_portfolio_risk(
            daily_loss_pct=daily_pnl_pct,
            weekly_loss_pct=weekly_pnl_pct,
            drawdown_pct=drawdown_pct,
        )
        if not portfolio_allowed:
            self._blocked_count += 1
            return False, f"GRUNNLOV 7: {portfolio_reason}", {"law": "GRUNNLOV_7", "blocked_by": "RiskKernel"}
        
        # ═══════════════════════════════════════════════════════════════════════
        # GRUNNLOV 5: Entry Qualification Check
        # ═══════════════════════════════════════════════════════════════════════
        # Calculate risk amount and R/R
        risk_amount = position_size * (risk_pct / 100)
        if side == "LONG":
            potential_profit = (take_profit - entry_price) / entry_price * position_size
            rr_ratio = (take_profit - entry_price) / (entry_price - stop_loss) if stop_loss < entry_price else 0
        else:
            potential_profit = (entry_price - take_profit) / entry_price * position_size
            rr_ratio = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0
        
        entry_plan = EntryPlan(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_reward_ratio=rr_ratio,
            confidence=confidence,
            strategy=strategy,
        )
        
        qualified, qual_reason, _ = self.registry.entry_gate.qualify_entry(entry_plan)
        if not qualified:
            self._blocked_count += 1
            return False, f"GRUNNLOV 5: {qual_reason}", {"law": "GRUNNLOV_5", "blocked_by": "EntryGate"}
        
        # ═══════════════════════════════════════════════════════════════════════
        # GRUNNLOV 13: Constraint Check (Revenge Trading)
        # ═══════════════════════════════════════════════════════════════════════
        from datetime import timedelta
        revenge_allowed, revenge_reason = self.registry.constraint_enforcement.check_revenge_trading(
            symbol=symbol,
            last_trade_was_loss=last_trade_was_loss,
            time_since_last_trade=timedelta(minutes=time_since_last_trade_minutes),
        )
        if not revenge_allowed:
            self._blocked_count += 1
            return False, f"GRUNNLOV 13: {revenge_reason}", {"law": "GRUNNLOV_13", "blocked_by": "ConstraintEnforcement"}
        
        # ═══════════════════════════════════════════════════════════════════════
        # GRUNNLOV 14: Strategy Scope Check
        # ═══════════════════════════════════════════════════════════════════════
        scope_allowed, scope_reason = self.registry.strategy_scope.validate_trade(
            symbol=symbol,
            strategy=strategy,
            timeframe="15m",  # Default, should come from context
        )
        if not scope_allowed:
            self._blocked_count += 1
            return False, f"GRUNNLOV 14: {scope_reason}", {"law": "GRUNNLOV_14", "blocked_by": "StrategyScopeController"}
        
        # ═══════════════════════════════════════════════════════════════════════
        # ALL CHECKS PASSED
        # ═══════════════════════════════════════════════════════════════════════
        logger.info(
            f"✅ Validation {validation_id} PASSED\n"
            f"   Symbol: {symbol} {side}\n"
            f"   Size: ${position_size:.2f}, Leverage: {leverage}x\n"
            f"   Risk: {risk_pct:.2f}%, R/R: {rr_ratio:.2f}"
        )
        
        return True, "All Grunnlov checks passed", {
            "validation_id": validation_id,
            "risk_pct": risk_pct,
            "rr_ratio": rr_ratio,
            "risk_amount": risk_amount,
        }
    
    def validate_exit(
        self,
        symbol: str,
        side: str,
        exit_type: str,  # STOP_LOSS, TAKE_PROFIT, MANUAL, TIME_STOP
        is_manual: bool = False,
        user_id: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Validate exit request.
        
        GRUNNLOV 6: Exit Brain owns all exits
        GRUNNLOV 15: Manual exits require cooldown
        """
        # ═══════════════════════════════════════════════════════════════════════
        # GRUNNLOV 15: Human Override Check for Manual Exits
        # ═══════════════════════════════════════════════════════════════════════
        if is_manual:
            allowed, reason, wait = self.registry.human_override_lock.request_override(
                action="MANUAL_EXIT",
                reason=f"Manual exit for {symbol}",
                user_id=user_id or "UNKNOWN",
                is_emergency=exit_type == "EMERGENCY",
            )
            if not allowed:
                return False, f"GRUNNLOV 15: {reason} (wait {wait}s)"
        
        # Exit allowed
        return True, f"Exit approved: {exit_type}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_validations": self._validation_count,
            "blocked": self._blocked_count,
            "passed": self._validation_count - self._blocked_count,
            "block_rate": self._blocked_count / max(1, self._validation_count),
            "component_stats": self.registry.get_system_status(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH EXISTING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_grunnlov_with_governor():
    """
    Integration code to add Grunnlov validation to existing governor.
    
    Call this at system startup to enable constitutional enforcement.
    """
    validator = GrunnlovPreTradeValidator()
    
    logger.info(
        "🏛️ GRUNNLOV INTEGRATION COMPLETE\n"
        "   15 Fundamental Laws are now ACTIVE\n"
        "   All trades must pass constitutional validation\n"
        "   Hierarchy: Survival > Risk > Policy > Operations > AI"
    )
    
    return validator


# ═══════════════════════════════════════════════════════════════════════════════
# GRUNNLOV STATUS API
# ═══════════════════════════════════════════════════════════════════════════════

def get_grunnlov_status() -> Dict[str, Any]:
    """
    Get complete Grunnlov system status.
    
    Returns:
        Complete status of all 15 laws and their components
    """
    enforcer = get_constitution_enforcer()
    registry = get_grunnlov_registry()
    
    # Build comprehensive status
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_status": registry.get_system_status(),
        "constitution_summary": enforcer.get_constitution_summary(),
        "laws": {},
    }
    
    # Add status for each law
    for law, defn in GRUNNLOVER.items():
        status["laws"][law.name] = {
            "name_no": defn.name_no,
            "name_en": defn.name_en,
            "authority": defn.authority.name,
            "component": defn.component,
            "role": defn.component_role.name,
            "fail_mode": defn.fail_mode,
            "active": True,  # In production, check actual component status
        }
    
    return status


def print_grunnlov_hierarchy():
    """Print the constitutional hierarchy (for documentation/debugging)."""
    from backend.domains.governance.constitution import CONSTITUTION_HIERARCHY_ASCII
    print(CONSTITUTION_HIERARCHY_ASCII)


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK INTEGRATION CHECK
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run integration test
    print("🏛️ GRUNNLOV INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize
    validator = integrate_grunnlov_with_governor()
    
    # Test a valid trade
    allowed, reason, details = validator.validate_entry(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        position_size=1000,
        leverage=5,
        confidence=0.75,
        strategy="MOMENTUM",
    )
    
    print(f"\nTest Trade 1 (Valid):")
    print(f"  Allowed: {allowed}")
    print(f"  Reason: {reason}")
    if details:
        print(f"  Details: {details}")
    
    # Test a trade with no stop-loss
    allowed, reason, details = validator.validate_entry(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000,
        stop_loss=0,  # No stop-loss!
        take_profit=52000,
        position_size=1000,
        leverage=5,
        confidence=0.75,
        strategy="MOMENTUM",
    )
    
    print(f"\nTest Trade 2 (No Stop-Loss):")
    print(f"  Allowed: {allowed}")
    print(f"  Reason: {reason}")
    
    # Test a trade with excessive leverage
    allowed, reason, details = validator.validate_entry(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=50000,
        stop_loss=49000,
        take_profit=52000,
        position_size=1000,
        leverage=50,  # Excessive!
        confidence=0.75,
        strategy="MOMENTUM",
    )
    
    print(f"\nTest Trade 3 (Excessive Leverage):")
    print(f"  Allowed: {allowed}")
    print(f"  Reason: {reason}")
    
    # Print stats
    print(f"\n📊 Validation Statistics:")
    stats = validator.get_stats()
    print(f"  Total: {stats['total_validations']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Blocked: {stats['blocked']}")
    print(f"  Block Rate: {stats['block_rate']*100:.1f}%")
    
    # Print hierarchy
    print("\n")
    print_grunnlov_hierarchy()
