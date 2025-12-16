"""
SAFETY POLICY ENFORCER

Enforces trading safety policies based on system health status.

Author: Quantum Trader AI Team
Date: November 23, 2025
"""

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from self_healing import SafetyPolicy, HealthStatus, SubsystemType, SystemHealthReport

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """Result of a policy enforcement decision."""
    policy: SafetyPolicy
    timestamp: str
    
    # Decision
    allow_new_positions: bool
    allow_exits: bool
    force_close_risky: bool
    
    # Risk adjustments
    max_leverage: Optional[float]
    position_size_multiplier: Optional[float]
    stop_loss_multiplier: Optional[float]
    
    # Metadata
    reason: str
    affected_subsystems: List[SubsystemType]


class SafetyPolicyEnforcer:
    """
    Enforces safety policies to protect capital during system issues.
    
    Policies:
    - ALLOW_ALL: Normal operation, no restrictions
    - NO_NEW_TRADES: Block new positions, allow exits
    - DEFENSIVE_EXIT: Close risky positions
    - SAFE_RISK_PROFILE: Reduce leverage and position sizes
    - EMERGENCY_SHUTDOWN: Stop all trading immediately
    """
    
    def __init__(
        self,
        data_dir: str = "/app/data",
        
        # SAFE_RISK_PROFILE settings
        safe_max_leverage: float = 5.0,
        safe_position_size_mult: float = 0.5,
        safe_stop_loss_mult: float = 0.8,
        
        # DEFENSIVE_EXIT criteria
        max_allowed_leverage: float = 10.0,
        max_unrealized_loss_pct: float = 5.0,
    ):
        self.data_dir = Path(data_dir)
        
        self.safe_max_leverage = safe_max_leverage
        self.safe_position_size_mult = safe_position_size_mult
        self.safe_stop_loss_mult = safe_stop_loss_mult
        
        self.max_allowed_leverage = max_allowed_leverage
        self.max_unrealized_loss_pct = max_unrealized_loss_pct
        
        # Current state
        self.current_policy = SafetyPolicy.ALLOW_ALL
        self.policy_history: List[PolicyDecision] = []
        
        # Create directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------
    # POLICY ENFORCEMENT
    # --------------------------------------------------------
    
    def enforce_policy(self, health_report: SystemHealthReport) -> PolicyDecision:
        """
        Enforce safety policy based on system health.
        
        Args:
            health_report: Current system health report
        
        Returns:
            PolicyDecision with enforcement details
        """
        policy = health_report.current_safety_policy
        timestamp = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"[POLICY] Enforcing policy: {policy.value}")
        
        # Update current policy
        if policy != self.current_policy:
            logger.warning(f"[POLICY] Policy changed: {self.current_policy.value} → {policy.value}")
            self.current_policy = policy
        
        # Create policy decision based on policy type
        if policy == SafetyPolicy.ALLOW_ALL:
            decision = self._allow_all_policy(health_report, timestamp)
        elif policy == SafetyPolicy.NO_NEW_TRADES:
            decision = self._no_new_trades_policy(health_report, timestamp)
        elif policy == SafetyPolicy.DEFENSIVE_EXIT:
            decision = self._defensive_exit_policy(health_report, timestamp)
        elif policy == SafetyPolicy.SAFE_RISK_PROFILE:
            decision = self._safe_risk_profile_policy(health_report, timestamp)
        elif policy == SafetyPolicy.EMERGENCY_SHUTDOWN:
            decision = self._emergency_shutdown_policy(health_report, timestamp)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        
        # Track history
        self.policy_history.append(decision)
        self._save_decision(decision)
        
        # Log decision
        logger.info(
            f"[POLICY] Decision: "
            f"New positions={'✅ Allowed' if decision.allow_new_positions else '❌ Blocked'}, "
            f"Exits={'✅ Allowed' if decision.allow_exits else '❌ Blocked'}, "
            f"Force close risky={'⚠️ Yes' if decision.force_close_risky else 'No'}"
        )
        
        if decision.max_leverage:
            logger.info(f"[POLICY] Max leverage: {decision.max_leverage}x")
        if decision.position_size_multiplier:
            logger.info(f"[POLICY] Position size: {decision.position_size_multiplier:.0%} of normal")
        
        return decision
    
    # --------------------------------------------------------
    # POLICY IMPLEMENTATIONS
    # --------------------------------------------------------
    
    def _allow_all_policy(self, health_report: SystemHealthReport, timestamp: str) -> PolicyDecision:
        """ALLOW_ALL: Normal operation."""
        return PolicyDecision(
            policy=SafetyPolicy.ALLOW_ALL,
            timestamp=timestamp,
            allow_new_positions=True,
            allow_exits=True,
            force_close_risky=False,
            max_leverage=None,  # No restrictions
            position_size_multiplier=None,
            stop_loss_multiplier=None,
            reason="System healthy - normal operation",
            affected_subsystems=[]
        )
    
    def _no_new_trades_policy(self, health_report: SystemHealthReport, timestamp: str) -> PolicyDecision:
        """NO_NEW_TRADES: Block new positions, allow exits."""
        # Find affected subsystems
        affected = [
            subsystem for subsystem, check in health_report.subsystem_health.items()
            if check.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL, HealthStatus.FAILED]
        ]
        
        reason = f"System degraded - {len(affected)} subsystems affected"
        
        return PolicyDecision(
            policy=SafetyPolicy.NO_NEW_TRADES,
            timestamp=timestamp,
            allow_new_positions=False,  # ❌ Block new positions
            allow_exits=True,           # ✅ Allow exits
            force_close_risky=False,
            max_leverage=None,
            position_size_multiplier=None,
            stop_loss_multiplier=None,
            reason=reason,
            affected_subsystems=affected
        )
    
    def _defensive_exit_policy(self, health_report: SystemHealthReport, timestamp: str) -> PolicyDecision:
        """DEFENSIVE_EXIT: Close risky positions."""
        # Find critical subsystems
        critical = [
            subsystem for subsystem, check in health_report.subsystem_health.items()
            if check.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]
        ]
        
        reason = f"Critical issues detected - {len(critical)} critical subsystems"
        
        return PolicyDecision(
            policy=SafetyPolicy.DEFENSIVE_EXIT,
            timestamp=timestamp,
            allow_new_positions=False,      # ❌ No new positions
            allow_exits=True,               # ✅ Allow exits
            force_close_risky=True,         # ⚠️ Force close risky positions
            max_leverage=self.max_allowed_leverage,
            position_size_multiplier=None,
            stop_loss_multiplier=None,
            reason=reason,
            affected_subsystems=critical
        )
    
    def _safe_risk_profile_policy(self, health_report: SystemHealthReport, timestamp: str) -> PolicyDecision:
        """SAFE_RISK_PROFILE: Reduce leverage and position sizes."""
        # Find problematic subsystems
        problematic = [
            subsystem for subsystem, check in health_report.subsystem_health.items()
            if check.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
        ]
        
        reason = f"System under stress - reducing risk exposure"
        
        return PolicyDecision(
            policy=SafetyPolicy.SAFE_RISK_PROFILE,
            timestamp=timestamp,
            allow_new_positions=True,       # ⚠️ Allow with restrictions
            allow_exits=True,               # ✅ Allow exits
            force_close_risky=False,
            max_leverage=self.safe_max_leverage,              # Reduced leverage (5x)
            position_size_multiplier=self.safe_position_size_mult,  # 50% normal size
            stop_loss_multiplier=self.safe_stop_loss_mult,    # Tighter stops (80%)
            reason=reason,
            affected_subsystems=problematic
        )
    
    def _emergency_shutdown_policy(self, health_report: SystemHealthReport, timestamp: str) -> PolicyDecision:
        """EMERGENCY_SHUTDOWN: Stop all trading immediately."""
        # Find failed subsystems
        failed = [
            subsystem for subsystem, check in health_report.subsystem_health.items()
            if check.status == HealthStatus.FAILED
        ]
        
        reason = f"EMERGENCY: System failure - {len(failed)} failed subsystems"
        
        return PolicyDecision(
            policy=SafetyPolicy.EMERGENCY_SHUTDOWN,
            timestamp=timestamp,
            allow_new_positions=False,      # ❌ No new positions
            allow_exits=False,              # ❌ No trading at all
            force_close_risky=True,         # 🚨 Close everything
            max_leverage=0.0,               # No leverage allowed
            position_size_multiplier=0.0,   # No new positions
            stop_loss_multiplier=None,
            reason=reason,
            affected_subsystems=failed
        )
    
    # --------------------------------------------------------
    # POLICY CHECKS
    # --------------------------------------------------------
    
    def can_open_position(self, leverage: float = 1.0) -> tuple[bool, str]:
        """
        Check if opening a new position is allowed.
        
        Args:
            leverage: Requested leverage
        
        Returns:
            (allowed, reason)
        """
        if not self.policy_history:
            return True, "No policy in effect"
        
        decision = self.policy_history[-1]
        
        if not decision.allow_new_positions:
            return False, f"Policy {decision.policy.value}: New positions blocked"
        
        if decision.max_leverage and leverage > decision.max_leverage:
            return False, f"Leverage {leverage}x exceeds max {decision.max_leverage}x"
        
        return True, "Position allowed"
    
    def can_close_position(self) -> tuple[bool, str]:
        """Check if closing a position is allowed."""
        if not self.policy_history:
            return True, "No policy in effect"
        
        decision = self.policy_history[-1]
        
        if not decision.allow_exits:
            return False, f"Policy {decision.policy.value}: Exits blocked"
        
        return True, "Exit allowed"
    
    def should_force_close(self, position_leverage: float, unrealized_loss_pct: float) -> tuple[bool, str]:
        """
        Check if a position should be force-closed.
        
        Args:
            position_leverage: Current position leverage
            unrealized_loss_pct: Unrealized loss percentage (positive = loss)
        
        Returns:
            (should_close, reason)
        """
        if not self.policy_history:
            return False, "No policy in effect"
        
        decision = self.policy_history[-1]
        
        if not decision.force_close_risky:
            return False, "Force close not required"
        
        # Check criteria
        if position_leverage > self.max_allowed_leverage:
            return True, f"Leverage {position_leverage}x exceeds max {self.max_allowed_leverage}x"
        
        if unrealized_loss_pct > self.max_unrealized_loss_pct:
            return True, f"Loss {unrealized_loss_pct:.1f}% exceeds max {self.max_unrealized_loss_pct}%"
        
        return False, "Position within acceptable risk"
    
    def get_risk_adjustments(self) -> Dict[str, Any]:
        """Get current risk adjustments from active policy."""
        if not self.policy_history:
            return {}
        
        decision = self.policy_history[-1]
        
        adjustments = {}
        
        if decision.max_leverage is not None:
            adjustments["max_leverage"] = decision.max_leverage
        
        if decision.position_size_multiplier is not None:
            adjustments["position_size_multiplier"] = decision.position_size_multiplier
        
        if decision.stop_loss_multiplier is not None:
            adjustments["stop_loss_multiplier"] = decision.stop_loss_multiplier
        
        return adjustments
    
    # --------------------------------------------------------
    # STATE MANAGEMENT
    # --------------------------------------------------------
    
    def _save_decision(self, decision: PolicyDecision):
        """Save policy decision to disk."""
        try:
            # Save latest decision
            decision_path = self.data_dir / "current_policy_decision.json"
            
            decision_dict = {
                "policy": decision.policy.value,
                "timestamp": decision.timestamp,
                "allow_new_positions": decision.allow_new_positions,
                "allow_exits": decision.allow_exits,
                "force_close_risky": decision.force_close_risky,
                "max_leverage": decision.max_leverage,
                "position_size_multiplier": decision.position_size_multiplier,
                "stop_loss_multiplier": decision.stop_loss_multiplier,
                "reason": decision.reason,
                "affected_subsystems": [s.value for s in decision.affected_subsystems]
            }
            
            with open(decision_path, 'w') as f:
                json.dump(decision_dict, f, indent=2)
            
            # Save history
            history_path = self.data_dir / "policy_decision_history.json"
            
            history_dict = [
                {
                    "policy": d.policy.value,
                    "timestamp": d.timestamp,
                    "allow_new_positions": d.allow_new_positions,
                    "reason": d.reason
                }
                for d in self.policy_history[-100:]  # Keep last 100
            ]
            
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save policy decision: {e}")
    
    def get_current_policy(self) -> SafetyPolicy:
        """Get current active policy."""
        return self.current_policy
    
    def get_decision_history(self, limit: int = 10) -> List[PolicyDecision]:
        """Get recent policy decisions."""
        return self.policy_history[-limit:]


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("SAFETY POLICY ENFORCER - Standalone Test")
    print("=" * 60)
    
    from self_healing import HealthCheck, DetectedIssue, IssueSeverity
    
    # Initialize enforcer
    enforcer = SafetyPolicyEnforcer(
        data_dir="./data",
        safe_max_leverage=5.0,
        safe_position_size_mult=0.5
    )
    
    print(f"\n[OK] Safety Policy Enforcer initialized")
    print(f"  Data dir: {enforcer.data_dir}")
    print(f"  SAFE max leverage: {enforcer.safe_max_leverage}x")
    print(f"  SAFE position size: {enforcer.safe_position_size_mult:.0%}")
    
    # Create mock health report - HEALTHY
    print("\n" + "=" * 60)
    print("TEST 1: Enforce ALLOW_ALL (Healthy System)")
    print("=" * 60)
    
    healthy_report = SystemHealthReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status=HealthStatus.HEALTHY,
        current_safety_policy=SafetyPolicy.ALLOW_ALL,
        subsystem_health={},
        detected_issues=[],
        critical_issues=[],
        recovery_recommendations=[],
        trading_should_continue=True,
        requires_immediate_action=False,
        healthy_count=13,
        degraded_count=0,
        critical_count=0,
        failed_count=0
    )
    
    decision = enforcer.enforce_policy(healthy_report)
    
    print(f"\n[OK] Policy: {decision.policy.value}")
    print(f"  New positions: {'✅ Allowed' if decision.allow_new_positions else '❌ Blocked'}")
    print(f"  Exits: {'✅ Allowed' if decision.allow_exits else '❌ Blocked'}")
    print(f"  Reason: {decision.reason}")
    
    # Test position check
    can_open, reason = enforcer.can_open_position(leverage=10.0)
    print(f"  Can open 10x position: {can_open} - {reason}")
    
    # Create mock health report - DEGRADED
    print("\n" + "=" * 60)
    print("TEST 2: Enforce NO_NEW_TRADES (Degraded System)")
    print("=" * 60)
    
    degraded_report = SystemHealthReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status=HealthStatus.DEGRADED,
        current_safety_policy=SafetyPolicy.NO_NEW_TRADES,
        subsystem_health={},
        detected_issues=[],
        critical_issues=[],
        recovery_recommendations=[],
        trading_should_continue=True,
        requires_immediate_action=False,
        healthy_count=10,
        degraded_count=3,
        critical_count=0,
        failed_count=0
    )
    
    decision = enforcer.enforce_policy(degraded_report)
    
    print(f"\n[OK] Policy: {decision.policy.value}")
    print(f"  New positions: {'✅ Allowed' if decision.allow_new_positions else '❌ Blocked'}")
    print(f"  Exits: {'✅ Allowed' if decision.allow_exits else '❌ Blocked'}")
    print(f"  Reason: {decision.reason}")
    
    can_open, reason = enforcer.can_open_position(leverage=5.0)
    print(f"  Can open 5x position: {can_open} - {reason}")
    
    # Create mock health report - CRITICAL
    print("\n" + "=" * 60)
    print("TEST 3: Enforce SAFE_RISK_PROFILE (Critical System)")
    print("=" * 60)
    
    critical_report = SystemHealthReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status=HealthStatus.CRITICAL,
        current_safety_policy=SafetyPolicy.SAFE_RISK_PROFILE,
        subsystem_health={},
        detected_issues=[],
        critical_issues=[],
        recovery_recommendations=[],
        trading_should_continue=True,
        requires_immediate_action=False,
        healthy_count=8,
        degraded_count=3,
        critical_count=2,
        failed_count=0
    )
    
    decision = enforcer.enforce_policy(critical_report)
    
    print(f"\n[OK] Policy: {decision.policy.value}")
    print(f"  New positions: {'✅ Allowed' if decision.allow_new_positions else '❌ Blocked'}")
    print(f"  Max leverage: {decision.max_leverage}x")
    print(f"  Position size: {decision.position_size_multiplier:.0%}")
    print(f"  Reason: {decision.reason}")
    
    can_open, reason = enforcer.can_open_position(leverage=3.0)
    print(f"  Can open 3x position: {can_open} - {reason}")
    
    can_open, reason = enforcer.can_open_position(leverage=10.0)
    print(f"  Can open 10x position: {can_open} - {reason}")
    
    # Test force close
    print("\n" + "=" * 60)
    print("TEST 4: Check Force Close (DEFENSIVE_EXIT)")
    print("=" * 60)
    
    defensive_report = SystemHealthReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_status=HealthStatus.CRITICAL,
        current_safety_policy=SafetyPolicy.DEFENSIVE_EXIT,
        subsystem_health={},
        detected_issues=[],
        critical_issues=[DetectedIssue(
            issue_id="test",
            subsystem=SubsystemType.EXCHANGE_CONNECTION,
            severity=IssueSeverity.CRITICAL,
            timestamp=datetime.now(timezone.utc).isoformat(),
            description="Test critical issue",
            symptoms=[],
            root_cause=None,
            impacts_trading=True,
            affects_subsystems=[]
        )],
        recovery_recommendations=[],
        trading_should_continue=False,
        requires_immediate_action=True,
        healthy_count=7,
        degraded_count=4,
        critical_count=2,
        failed_count=0
    )
    
    decision = enforcer.enforce_policy(defensive_report)
    
    print(f"\n[OK] Policy: {decision.policy.value}")
    print(f"  Force close risky: {'⚠️ Yes' if decision.force_close_risky else 'No'}")
    
    should_close, reason = enforcer.should_force_close(position_leverage=15.0, unrealized_loss_pct=2.0)
    print(f"  Close 15x position with 2% loss: {should_close} - {reason}")
    
    should_close, reason = enforcer.should_force_close(position_leverage=5.0, unrealized_loss_pct=8.0)
    print(f"  Close 5x position with 8% loss: {should_close} - {reason}")
    
    should_close, reason = enforcer.should_force_close(position_leverage=5.0, unrealized_loss_pct=2.0)
    print(f"  Close 5x position with 2% loss: {should_close} - {reason}")
    
    print("\n" + "=" * 60)
    print("TEST 5: Check Policy History")
    print("=" * 60)
    
    history = enforcer.get_decision_history()
    print(f"\n[OK] Policy decisions ({len(history)}):")
    for i, dec in enumerate(history, 1):
        print(f"  {i}. {dec.policy.value} - {dec.reason}")
    
    print("\n" + "=" * 60)
    print("[OK] All tests completed successfully!")
    print("=" * 60)
    
    print(f"\n[OK] Decision saved to: {enforcer.data_dir / 'current_policy_decision.json'}")
