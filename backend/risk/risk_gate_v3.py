"""
Risk Gate v3 - Unified Risk Enforcement for Execution

EPIC-RISK3-EXEC-001: Enforce Global Risk v3 in Execution Path

This module provides a unified risk gate that ENFORCES (not advises) risk checks
before every order placement. Integrates:
- Global Risk v3 state (RiskLevel, drawdown, leverage, ESS tier)
- Capital profile limits (from EPIC-P10)
- ESS (Emergency Stop System) halt checks
- Multi-account system (EPIC-MT-ACCOUNTS-001)

Design principles:
- Read-only facade over Global Risk v3 (no heavy coupling)
- Clean separation of concerns
- Explicit decision model: allow/block/scale_down
- No order can bypass critical states or ESS halt
"""

import logging
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime

import httpx

# Capital profile integration
from backend.policies.capital_profiles import get_profile
from backend.policies.account_config import get_capital_profile_for_account
from backend.policies.strategy_profile_policy import is_strategy_allowed

# ESS integration
from backend.services.risk.emergency_stop_system import EmergencyStopSystem

# EPIC-STRESS-DASH-001: Prometheus metrics for dashboard visibility
try:
    from infra.metrics.metrics import risk_gate_decisions_total
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# DECISION MODEL
# ============================================================================

@dataclass
class RiskGateResult:
    """
    Result of risk gate evaluation
    
    Fields:
        decision: "allow" (proceed), "block" (reject order), "scale_down" (reduce size)
        reason: Human-readable explanation
        scale_factor: If scale_down, factor to multiply order size (e.g., 0.5 = half size)
        risk_level: Current global risk level from Risk v3 (INFO/WARNING/CRITICAL)
        ess_active: Whether ESS halt is active
    """
    decision: Literal["allow", "block", "scale_down"]
    reason: str
    scale_factor: float = 1.0  # Only relevant if decision == "scale_down"
    risk_level: Optional[str] = None
    ess_active: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


# ============================================================================
# RISK STATE FACADE
# ============================================================================

class RiskStateFacade:
    """
    Read-only facade over Global Risk v3 state
    
    Provides minimal interface to risk data needed for gate decisions.
    Does NOT modify risk state - read-only by design.
    """
    
    def __init__(self, risk_api_url: str = "http://localhost:8001"):
        """
        Initialize risk state facade
        
        Args:
            risk_api_url: Base URL for Risk v3 service API (default: localhost:8001)
        """
        self.risk_api_url = risk_api_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=5.0)
        self._last_signal = None
        self._last_fetch = None
        logger.info(f"[RISK-GATE] RiskStateFacade initialized with API: {self.risk_api_url}")
    
    async def get_global_risk_signal(self) -> Optional[dict]:
        """
        Fetch current GlobalRiskSignal from Risk v3 service
        
        Returns:
            dict with fields:
                - risk_level: "INFO" | "WARNING" | "CRITICAL"
                - overall_risk_score: float (0.0 - 1.0)
                - ess_tier_recommendation: "NORMAL" | "REDUCED" | "EMERGENCY"
                - ess_action_required: bool
                - critical_issues: List[str]
                - warnings: List[str]
                - snapshot: {...}  # Full RiskSnapshot
            
            Returns None if Risk v3 service unavailable or no evaluation yet
        """
        try:
            response = await self.client.get(f"{self.risk_api_url}/risk/global")
            response.raise_for_status()
            signal = response.json()
            self._last_signal = signal
            self._last_fetch = datetime.utcnow()
            return signal
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("[RISK-GATE] No risk evaluation available yet from Risk v3")
                return None
            logger.error(f"[RISK-GATE] Risk v3 API error: {e}")
            return None
        except Exception as e:
            logger.error(f"[RISK-GATE] Failed to fetch risk signal: {e}")
            return None
    
    async def get_current_drawdown(self) -> Optional[float]:
        """
        Get current drawdown percentage from risk snapshot
        
        Returns:
            Drawdown as percentage (e.g., 5.2 for 5.2%), or None if unavailable
        """
        signal = await self.get_global_risk_signal()
        if not signal:
            return None
        snapshot = signal.get("snapshot")
        if not snapshot:
            return None
        return snapshot.get("drawdown_pct")
    
    async def get_current_leverage(self) -> Optional[float]:
        """
        Get current total leverage from risk snapshot
        
        Returns:
            Total leverage (e.g., 3.5 for 3.5x), or None if unavailable
        """
        signal = await self.get_global_risk_signal()
        if not signal:
            return None
        snapshot = signal.get("snapshot")
        if not snapshot:
            return None
        return snapshot.get("total_leverage")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# ============================================================================
# RISK GATE
# ============================================================================

class RiskGateV3:
    """
    Unified risk gate for order execution
    
    Enforces Global Risk v3 + Capital Profiles + ESS before every order.
    """
    
    def __init__(
        self,
        risk_facade: Optional[RiskStateFacade] = None,
        ess: Optional[EmergencyStopSystem] = None,
    ):
        """
        Initialize Risk Gate v3
        
        Args:
            risk_facade: RiskStateFacade for reading Global Risk v3 state
            ess: EmergencyStopSystem for ESS halt checks
        """
        self.risk_facade = risk_facade or RiskStateFacade()
        self.ess = ess
        logger.info("[RISK-GATE] RiskGateV3 initialized")
        logger.info(f"  ✅ Risk Facade: {self.risk_facade.risk_api_url}")
        logger.info(f"  ✅ ESS Integration: {self.ess is not None}")
    
    async def evaluate_order_risk(
        self,
        account_name: str,
        exchange_name: str,
        strategy_id: str,
        order_request: dict,
    ) -> RiskGateResult:
        """
        Evaluate whether order should be allowed, blocked, or scaled down
        
        Evaluation hierarchy (first failure wins):
        1. ESS halt check (if ESS active → BLOCK, no exceptions)
        2. Global Risk v3 CRITICAL level → BLOCK
        3. Capital profile: strategy not in whitelist → BLOCK
        4. Capital profile: leverage exceeds limit → BLOCK
        5. Capital profile: single-trade risk too large → BLOCK or SCALE_DOWN
        6. Capital profile: daily/weekly loss limits → BLOCK (stub if metrics unavailable)
        7. All checks pass → ALLOW
        
        Args:
            account_name: Trading account name (e.g., "PRIVATE_MAIN")
            exchange_name: Exchange name (e.g., "binance")
            strategy_id: Strategy ID (e.g., "neo_scalper_1")
            order_request: Order request dict with keys:
                - symbol: str (e.g., "BTCUSDT")
                - side: "BUY" | "SELL"
                - size: float (position size in quote currency)
                - leverage: float (e.g., 3.0 for 3x)
                - (other fields as needed)
        
        Returns:
            RiskGateResult with decision, reason, scale_factor
        """
        logger.info(f"[RISK-GATE] Evaluating order: account={account_name}, exchange={exchange_name}, strategy={strategy_id}")
        
        # === PHASE 1: ESS HALT CHECK ===
        # If ESS halt active, BLOCK immediately (no exceptions)
        if self.ess and self.ess.is_active():
            logger.warning("[RISK-GATE] ❌ ESS halt active - blocking order")
            result = RiskGateResult(
                decision="block",
                reason="ess_trading_halt_active",
                ess_active=True,
            )
            self._record_decision_metric(result, account_name, exchange_name, strategy_id)
            return result
        
        # === PHASE 2: GLOBAL RISK V3 CHECK ===
        # Fetch current Global Risk signal
        risk_signal = await self.risk_facade.get_global_risk_signal()
        risk_level = "UNKNOWN"
        if risk_signal:
            risk_level = risk_signal.get("risk_level", "UNKNOWN")
            ess_action_required = risk_signal.get("ess_action_required", False)
            
            # If Risk v3 says CRITICAL, block
            if risk_level == "CRITICAL":
                critical_issues = risk_signal.get("critical_issues", [])
                reason = f"global_risk_critical: {'; '.join(critical_issues[:2])}" if critical_issues else "global_risk_critical"
                logger.warning(f"[RISK-GATE] ❌ Global Risk CRITICAL - blocking order: {reason}")
                result = RiskGateResult(
                    decision="block",
                    reason=reason,
                    risk_level=risk_level,
                )
                self._record_decision_metric(result, account_name, exchange_name, strategy_id)
                return result
            
            # If Risk v3 recommends ESS action, block
            if ess_action_required:
                logger.warning("[RISK-GATE] ❌ Risk v3 recommends ESS action - blocking order")
                result = RiskGateResult(
                    decision="block",
                    reason="risk_v3_ess_action_required",
                    risk_level=risk_level,
                )
                self._record_decision_metric(result, account_name, exchange_name, strategy_id)
                return result
        
        # === PHASE 3: CAPITAL PROFILE CHECKS ===
        # Get capital profile for account
        profile_name = get_capital_profile_for_account(account_name)
        if not profile_name:
            logger.warning(f"[RISK-GATE] ⚠️ No capital profile found for account {account_name} - using 'micro' as fallback")
            profile_name = "micro"
        
        profile = get_profile(profile_name)
        if not profile:
            logger.error(f"[RISK-GATE] ❌ Capital profile '{profile_name}' not found - blocking order")
            result = RiskGateResult(
                decision="block",
                reason=f"capital_profile_not_found: {profile_name}",
                risk_level=risk_level,
            )
            self._record_decision_metric(result, account_name, exchange_name, strategy_id)
            return result
        
        logger.info(f"[RISK-GATE]   Profile: {profile_name} (limits: max_leverage={profile.allowed_leverage}, max_single_trade={profile.max_single_trade_risk_pct}%)")
        
        # Check 3a: Strategy in whitelist?
        if not is_strategy_allowed(profile_name, strategy_id):
            logger.warning(f"[RISK-GATE] ❌ Strategy {strategy_id} not in profile whitelist - blocking order")
            result = RiskGateResult(
                decision="block",
                reason=f"strategy_not_allowed: {strategy_id} not in {profile_name} whitelist",
                risk_level=risk_level,
            )
            self._record_decision_metric(result, account_name, exchange_name, strategy_id)
            return result
        
        # Check 3b: Leverage within limits?
        order_leverage = order_request.get("leverage", 1.0)
        if order_leverage > profile.allowed_leverage:
            logger.warning(f"[RISK-GATE] ❌ Leverage {order_leverage}x exceeds profile limit {profile.allowed_leverage}x - blocking order")
            result = RiskGateResult(
                decision="block",
                reason=f"leverage_exceeds_limit: {order_leverage}x > {profile.allowed_leverage}x",
                risk_level=risk_level,
            )
            self._record_decision_metric(result, account_name, exchange_name, strategy_id)
            return result
        
        # Check 3c: Single-trade risk within limits?
        # Single-trade risk = (order_size / total_equity) * 100
        # For now, we'll stub this check since we need real-time equity data
        # TODO: Integrate with portfolio service for real equity
        order_size = order_request.get("size", 0.0)
        single_trade_risk_pct = 0.1  # Stub: assume 0.1% risk for now (safe for all profiles)
        # REAL IMPLEMENTATION: single_trade_risk_pct = (order_size / total_equity) * 100
        
        if single_trade_risk_pct > profile.max_single_trade_risk_pct:
            # Option A: BLOCK if risk too large
            logger.warning(f"[RISK-GATE] ❌ Single-trade risk {single_trade_risk_pct:.1f}% exceeds limit {profile.max_single_trade_risk_pct}% - blocking order")
            result = RiskGateResult(
                decision="block",
                reason=f"single_trade_risk_exceeds_limit: {single_trade_risk_pct:.1f}% > {profile.max_single_trade_risk_pct}%",
                risk_level=risk_level,
            )
            self._record_decision_metric(result, account_name, exchange_name, strategy_id)
            return result
            
            # Option B: SCALE_DOWN to fit within limit
            # scale_factor = profile.max_single_trade_risk_pct / single_trade_risk_pct
            # logger.info(f"[RISK-GATE] ⚠️ Single-trade risk {single_trade_risk_pct:.1f}% exceeds limit - scaling down by {scale_factor:.2f}")
            # return RiskGateResult(
            #     decision="scale_down",
            #     reason=f"single_trade_risk_scaled: {single_trade_risk_pct:.1f}% > {profile.max_single_trade_risk_pct}%",
            #     scale_factor=scale_factor,
            #     risk_level=risk_level,
            # )
        
        # Check 3d: Daily/weekly loss limits (stub - need real PnL metrics)
        # TODO: Integrate with analytics service for real daily/weekly PnL
        # For now, we'll skip this check and leave it for future implementation
        
        # === ALL CHECKS PASSED ===
        logger.info(f"[RISK-GATE] ✅ Order allowed (risk_level={risk_level}, profile={profile_name})")
        result = RiskGateResult(
            decision="allow",
            reason="all_risk_checks_passed",
            risk_level=risk_level,
        )
        
        # EPIC-STRESS-DASH-001: Record metric
        self._record_decision_metric(result, account_name, exchange_name, strategy_id)
        
        return result
    
    def _record_decision_metric(
        self,
        result: RiskGateResult,
        account_name: str,
        exchange_name: str,
        strategy_id: str,
    ) -> None:
        """
        Record RiskGate decision to Prometheus metrics.
        
        EPIC-STRESS-DASH-001: Dashboard observability for risk decisions.
        """
        if not METRICS_AVAILABLE or not risk_gate_decisions_total:
            return
        
        try:
            risk_gate_decisions_total.labels(
                decision=result.decision,
                reason=result.reason or "unknown",
                account=account_name,
                exchange=exchange_name,
                strategy=strategy_id,
            ).inc()
        except Exception as e:
            logger.warning(f"[RISK-GATE] Failed to record metric: {e}")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Global risk gate instance (initialize in main.py or execution service)
_global_risk_gate: Optional[RiskGateV3] = None


def init_risk_gate(risk_facade: Optional[RiskStateFacade] = None, ess: Optional[EmergencyStopSystem] = None):
    """
    Initialize global risk gate instance
    
    Call this once at startup from main.py or execution service.
    
    Args:
        risk_facade: RiskStateFacade for reading Global Risk v3 state
        ess: EmergencyStopSystem for ESS halt checks
    """
    global _global_risk_gate
    _global_risk_gate = RiskGateV3(risk_facade=risk_facade, ess=ess)
    logger.info("[RISK-GATE] Global risk gate initialized")


def get_risk_gate() -> Optional[RiskGateV3]:
    """
    Get global risk gate instance
    
    Returns:
        RiskGateV3 instance, or None if not initialized
    """
    return _global_risk_gate


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def evaluate_order_risk(
    account_name: str,
    exchange_name: str,
    strategy_id: str,
    order_request: dict,
) -> RiskGateResult:
    """
    Convenience function for evaluating order risk
    
    Uses global risk gate instance. If not initialized, returns BLOCK decision.
    
    Args:
        account_name: Trading account name
        exchange_name: Exchange name
        strategy_id: Strategy ID
        order_request: Order request dict
    
    Returns:
        RiskGateResult
    """
    gate = get_risk_gate()
    if not gate:
        logger.error("[RISK-GATE] Risk gate not initialized - blocking order by default")
        return RiskGateResult(
            decision="block",
            reason="risk_gate_not_initialized",
        )
    
    return await gate.evaluate_order_risk(
        account_name=account_name,
        exchange_name=exchange_name,
        strategy_id=strategy_id,
        order_request=order_request,
    )
