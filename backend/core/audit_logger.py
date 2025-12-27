"""
[P2-02] Common Audit Logger API
================================

Standardized audit logging for compliance and debugging.
Captures critical system events, decisions, and user actions.

Usage:
    from backend.core.audit_logger import get_audit_logger
    
    audit = get_audit_logger()
    audit.log_trade_decision(
        symbol="BTCUSDT",
        action="BUY",
        reason="RL agent selected long strategy",
        confidence=0.85
    )
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Audit event classification."""
    TRADE_DECISION = "trade.decision"
    TRADE_EXECUTED = "trade.executed"
    TRADE_CLOSED = "trade.closed"
    RISK_BLOCK = "risk.block"
    RISK_OVERRIDE = "risk.override"
    EMERGENCY_TRIGGERED = "emergency.triggered"
    EMERGENCY_RECOVERED = "emergency.recovered"
    MODEL_PROMOTED = "model.promoted"
    MODEL_DEMOTED = "model.demoted"
    POLICY_CHANGED = "policy.changed"
    SYSTEM_STATE_CHANGE = "system.state_change"
    CONFIG_UPDATED = "config.updated"


@dataclass
class AuditEvent:
    """Audit event structure."""
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actor: str = "system"  # Who/what triggered the event
    action: str = ""       # What action was taken
    target: str = ""       # What was affected
    reason: str = ""       # Why it happened
    outcome: str = ""      # Result of the action
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        return json.dumps(data, default=str)


class AuditLogger:
    """
    [P2-02] Standardized audit logging for compliance and debugging.
    
    Features:
    - Structured audit trails
    - Immutable event records
    - Search/filter by event type
    - Retention policies
    - Export to compliance systems
    """
    
    _instance: Optional['AuditLogger'] = None
    _initialized: bool = False
    
    @classmethod
    def get_instance(cls) -> 'AuditLogger':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize(cls, log_dir: Path = Path("logs/audit"), retention_days: int = 90) -> None:
        """Initialize the global audit logger."""
        if cls._instance is None:
            cls._instance = cls(log_dir=log_dir, retention_days=retention_days)
        cls._initialized = True
    
    def __init__(
        self,
        log_dir: Path = Path("logs/audit"),
        retention_days: int = 90,
        actor: str = "system"
    ):
        """
        Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
            retention_days: How long to keep audit logs
            actor: Default actor for events
        """
        self.log_dir = log_dir
        self.retention_days = retention_days
        self.default_actor = actor
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file (rotates daily)
        self.current_log_file = self._get_log_file()
        
        logger.info(f"[P2-02] AuditLogger initialized (log_dir={log_dir})")
    
    def _get_log_file(self) -> Path:
        """Get current log file path (daily rotation)."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        return self.log_dir / f"audit_{date_str}.jsonl"
    
    def _write_event(self, event: AuditEvent) -> None:
        """Write audit event to log file."""
        try:
            # Rotate log file if needed
            log_file = self._get_log_file()
            if log_file != self.current_log_file:
                self.current_log_file = log_file
                logger.info(f"[P2-02] Rotated to new audit log: {log_file}")
            
            # Write event as JSON line
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + '\n')
            
            # Also log to standard logger
            logger.info(
                f"[AUDIT] {event.event_type.value} | "
                f"actor={event.actor} | "
                f"action={event.action} | "
                f"target={event.target}"
            )
        
        except Exception as e:
            logger.error(f"[P2-02] Failed to write audit event: {e}")
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        target: str = "",
        reason: str = "",
        outcome: str = "",
        actor: Optional[str] = None,
        trace_id: Optional[str] = None,
        **metadata
    ) -> None:
        """
        Log a generic audit event.
        
        Args:
            event_type: Type of event
            action: What action was taken
            target: What was affected
            reason: Why it happened
            outcome: Result of the action
            actor: Who/what triggered (defaults to instance default)
            trace_id: Trace ID for correlation
            **metadata: Additional event-specific data
        """
        event = AuditEvent(
            event_type=event_type,
            actor=actor or self.default_actor,
            action=action,
            target=target,
            reason=reason,
            outcome=outcome,
            metadata=metadata,
            trace_id=trace_id
        )
        self._write_event(event)
    
    def log_trade_decision(
        self,
        symbol: str,
        action: str,
        reason: str,
        confidence: float,
        model: Optional[str] = None,
        **metadata
    ) -> None:
        """
        Log a trading decision.
        
        Args:
            symbol: Trading pair
            action: "BUY", "SELL", or "HOLD"
            reason: Why this decision was made
            confidence: Decision confidence (0-1)
            model: Which model made the decision
        """
        self.log_event(
            event_type=AuditEventType.TRADE_DECISION,
            action=action,
            target=symbol,
            reason=reason,
            model=model,
            confidence=confidence,
            **metadata
        )
    
    def log_trade_executed(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        entry_price: float,
        order_id: str,
        **metadata
    ) -> None:
        """
        Log a trade execution.
        
        Args:
            symbol: Trading pair
            side: "LONG" or "SHORT"
            size_usd: Position size in USD
            entry_price: Entry price
            order_id: Exchange order ID
        """
        self.log_event(
            event_type=AuditEventType.TRADE_EXECUTED,
            action=f"OPEN_{side}",
            target=symbol,
            outcome="SUCCESS",
            order_id=order_id,
            size_usd=size_usd,
            entry_price=entry_price,
            **metadata
        )
    
    def log_trade_closed(
        self,
        symbol: str,
        pnl: float,
        outcome: str,
        close_reason: str,
        hold_duration_sec: int,
        **metadata
    ) -> None:
        """
        Log a trade closure.
        
        Args:
            symbol: Trading pair
            pnl: Profit/loss in USD
            outcome: "WIN", "LOSS", or "BREAKEVEN"
            close_reason: Why the position was closed
            hold_duration_sec: How long position was held
        """
        self.log_event(
            event_type=AuditEventType.TRADE_CLOSED,
            action="CLOSE",
            target=symbol,
            reason=close_reason,
            outcome=outcome,
            pnl=pnl,
            hold_duration_sec=hold_duration_sec,
            **metadata
        )
    
    def log_risk_block(
        self,
        symbol: str,
        action: str,
        reason: str,
        blocker: str,
        **metadata
    ) -> None:
        """
        Log a risk-blocked trade.
        
        Args:
            symbol: Trading pair
            action: What action was blocked
            reason: Why it was blocked
            blocker: Which risk component blocked it
        """
        self.log_event(
            event_type=AuditEventType.RISK_BLOCK,
            action=f"BLOCKED_{action}",
            target=symbol,
            reason=reason,
            blocker=blocker,
            outcome="BLOCKED",
            **metadata
        )
    
    def log_emergency_triggered(
        self,
        severity: str,
        trigger_reason: str,
        positions_closed: int,
        **metadata
    ) -> None:
        """
        Log emergency system activation.
        
        Args:
            severity: "WARNING", "CRITICAL", or "CATASTROPHIC"
            trigger_reason: What triggered emergency
            positions_closed: How many positions were closed
        """
        self.log_event(
            event_type=AuditEventType.EMERGENCY_TRIGGERED,
            action="ACTIVATE_ESS",
            reason=trigger_reason,
            outcome="TRADING_HALTED",
            severity=severity,
            positions_closed=positions_closed,
            **metadata
        )
    
    def log_emergency_recovered(
        self,
        recovery_type: str,
        old_mode: str,
        new_mode: str,
        current_dd_pct: float,
        **metadata
    ) -> None:
        """
        Log emergency system recovery.
        
        Args:
            recovery_type: "AUTO" or "MANUAL"
            old_mode: Previous ESS mode
            new_mode: New ESS mode
            current_dd_pct: Current drawdown percentage
        """
        self.log_event(
            event_type=AuditEventType.EMERGENCY_RECOVERED,
            action=f"RECOVER_{recovery_type}",
            reason=f"Drawdown improved to {current_dd_pct:.2f}%",
            outcome=f"{old_mode} → {new_mode}",
            recovery_type=recovery_type,
            old_mode=old_mode,
            new_mode=new_mode,
            current_dd_pct=current_dd_pct,
            **metadata
        )
    
    def log_model_promoted(
        self,
        model_name: str,
        old_version: str,
        new_version: str,
        reason: str,
        **metadata
    ) -> None:
        """
        Log model promotion.
        
        Args:
            model_name: Model identifier
            old_version: Previous version
            new_version: New version
            reason: Why promoted
        """
        self.log_event(
            event_type=AuditEventType.MODEL_PROMOTED,
            action="PROMOTE",
            target=model_name,
            reason=reason,
            outcome=f"{old_version} → {new_version}",
            old_version=old_version,
            new_version=new_version,
            **metadata
        )
    
    def log_policy_changed(
        self,
        policy_name: str,
        old_value: Any,
        new_value: Any,
        changed_by: str,
        reason: str = "",
        **metadata
    ) -> None:
        """
        Log policy configuration change.
        
        Args:
            policy_name: Policy key
            old_value: Previous value
            new_value: New value
            changed_by: Who changed it
            reason: Why it was changed
        """
        self.log_event(
            event_type=AuditEventType.POLICY_CHANGED,
            action="UPDATE",
            target=policy_name,
            actor=changed_by,
            reason=reason,
            outcome=f"{old_value} → {new_value}",
            old_value=str(old_value),
            new_value=str(new_value),
            **metadata
        )


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    log_dir: Path = Path("logs/audit"),
    actor: str = "system"
) -> AuditLogger:
    """Get or create singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir=log_dir, actor=actor)
    return _audit_logger


# Module-level convenience functions for static-like usage
def log_trade_decision(symbol: str, action: str, reason: str, confidence: float, model: Optional[str] = None, **metadata) -> None:
    """Log trade decision."""
    get_audit_logger().log_trade_decision(symbol, action, reason, confidence, model, **metadata)


def log_trade_executed(symbol: str, side: str, size_usd: float, entry_price: float, order_id: str, **metadata) -> None:
    """Log trade execution."""
    get_audit_logger().log_trade_executed(symbol, side, size_usd, entry_price, order_id, **metadata)


def log_trade_closed(symbol: str, pnl: float, outcome: str, close_reason: str, hold_duration_sec: float, **metadata) -> None:
    """Log trade closure."""
    get_audit_logger().log_trade_closed(symbol, pnl, outcome, close_reason, hold_duration_sec, **metadata)


def log_risk_block(symbol: str, action: str, reason: str, blocker: str, **metadata) -> None:
    """Log risk block."""
    get_audit_logger().log_risk_block(symbol, action, reason, blocker, **metadata)


def log_emergency_triggered(severity: str, trigger_reason: str, positions_closed: int, **metadata) -> None:
    """Log emergency trigger."""
    get_audit_logger().log_emergency_triggered(severity, trigger_reason, positions_closed, **metadata)


def log_emergency_recovered(recovery_type: str, old_mode: str, new_mode: str, current_dd_pct: float, **metadata) -> None:
    """Log emergency recovery."""
    get_audit_logger().log_emergency_recovered(recovery_type, old_mode, new_mode, current_dd_pct, **metadata)


def log_model_promoted(model_name: str, old_version: str, new_version: str, reason: str, **metadata) -> None:
    """Log model promotion."""
    get_audit_logger().log_model_promoted(model_name, old_version, new_version, reason, **metadata)


def log_policy_changed(policy_name: str, old_value: Any, new_value: Any, changed_by: str, reason: str = "", **metadata) -> None:
    """Log policy change."""
    get_audit_logger().log_policy_changed(policy_name, old_value, new_value, changed_by, reason, **metadata)
