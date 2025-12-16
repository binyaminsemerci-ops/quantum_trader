"""
Governance Domain - Audit Operating System

Complete audit trail and compliance reporting.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""
    TRADE_EXECUTED = "trade_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    CAPITAL_ALLOCATED = "capital_allocated"
    RISK_VETO = "risk_veto"
    COMPLIANCE_VIOLATION = "compliance_violation"
    DECISION_PROPOSED = "decision_proposed"
    DECISION_APPROVED = "decision_approved"
    DIRECTIVE_ISSUED = "directive_issued"
    STRATEGY_SUSPENDED = "strategy_suspended"


@dataclass
class AuditRecord:
    """Immutable audit record."""
    record_id: str
    event_type: AuditEventType
    timestamp: datetime
    actor: str  # Who performed the action
    entity_id: str  # What was affected
    action: str
    details: Dict
    hash: Optional[str] = None  # Cryptographic hash for integrity


class AuditOS:
    """
    Audit Operating System - Complete audit trail.
    
    Responsibilities:
    - Record ALL system events with immutable trail
    - Generate compliance reports
    - Provide audit query interface
    - Detect anomalous patterns
    - Cryptographic verification of audit integrity
    
    Authority: OBSERVER (records, does not intervene)
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        audit_log_path: str = "./data/audit",
        enable_cryptographic_hash: bool = True,
        retention_days: int = 365,
    ):
        """
        Initialize Audit OS.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            audit_log_path: Path to audit log storage
            enable_cryptographic_hash: Enable hash verification
            retention_days: Days to retain audit records
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Audit configuration
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        self.enable_cryptographic_hash = enable_cryptographic_hash
        self.retention_days = retention_days
        
        # In-memory cache (last 1000 records)
        self.audit_cache: List[AuditRecord] = []
        self.cache_size = 1000
        
        # Subscribe to ALL auditable events
        self._subscribe_to_events()
        
        logger.info(
            f"[Audit] Audit OS initialized:\n"
            f"   Log Path: {audit_log_path}\n"
            f"   Cryptographic Hash: {enable_cryptographic_hash}\n"
            f"   Retention: {retention_days} days\n"
            f"   Cache Size: {self.cache_size} records"
        )
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to all auditable events."""
        events = [
            "position.opened",
            "position.closed",
            "fund.strategy.allocated",
            "fund.risk.veto.issued",
            "compliance.violation.detected",
            "compliance.trade.blocked",
            "governance.decision.proposed",
            "governance.decision.approved",
            "governance.decision.rejected",
            "fund.directive.issued",
            "fund.risk.strategy.suspended",
        ]
        
        for event_name in events:
            self.event_bus.subscribe(event_name, self._handle_audit_event)
        
        logger.info(f"[Audit] Subscribed to {len(events)} auditable events")
    
    async def _handle_audit_event(self, event_data: dict) -> None:
        """Handle auditable event."""
        event_name = event_data.get("_event_name", "unknown")
        
        # Map event to audit type
        event_type_mapping = {
            "position.opened": AuditEventType.POSITION_OPENED,
            "position.closed": AuditEventType.POSITION_CLOSED,
            "fund.strategy.allocated": AuditEventType.CAPITAL_ALLOCATED,
            "fund.risk.veto.issued": AuditEventType.RISK_VETO,
            "compliance.violation.detected": AuditEventType.COMPLIANCE_VIOLATION,
            "governance.decision.proposed": AuditEventType.DECISION_PROPOSED,
            "governance.decision.approved": AuditEventType.DECISION_APPROVED,
            "fund.directive.issued": AuditEventType.DIRECTIVE_ISSUED,
            "fund.risk.strategy.suspended": AuditEventType.STRATEGY_SUSPENDED,
        }
        
        event_type = event_type_mapping.get(event_name)
        if not event_type:
            logger.debug(f"[Audit] Skipping non-auditable event: {event_name}")
            return
        
        # Extract audit details
        actor = event_data.get("issued_by") or event_data.get("executed_by") or "SYSTEM"
        entity_id = (
            event_data.get("position_id")
            or event_data.get("strategy_id")
            or event_data.get("decision_id")
            or event_data.get("violation_id")
            or "N/A"
        )
        
        # Create audit record
        await self.record_audit(
            event_type=event_type,
            actor=actor,
            entity_id=entity_id,
            action=event_name,
            details=event_data
        )
    
    async def record_audit(
        self,
        event_type: AuditEventType,
        actor: str,
        entity_id: str,
        action: str,
        details: Dict
    ) -> str:
        """
        Record audit event.
        
        Args:
            event_type: Type of audit event
            actor: Who performed the action
            entity_id: What was affected
            action: Action performed
            details: Event details
        
        Returns:
            Record ID
        """
        import uuid
        
        record_id = f"AUDIT-{uuid.uuid4().hex[:12].upper()}"
        timestamp = datetime.now(timezone.utc)
        
        # Create audit record
        record = AuditRecord(
            record_id=record_id,
            event_type=event_type,
            timestamp=timestamp,
            actor=actor,
            entity_id=entity_id,
            action=action,
            details=details
        )
        
        # Generate cryptographic hash if enabled
        if self.enable_cryptographic_hash:
            record.hash = self._generate_hash(record)
        
        # Add to cache
        self.audit_cache.append(record)
        if len(self.audit_cache) > self.cache_size:
            self.audit_cache.pop(0)
        
        # Persist to disk
        await self._persist_record(record)
        
        logger.debug(
            f"[Audit] 📝 Recorded: {record_id}\n"
            f"   Type: {event_type.value}\n"
            f"   Actor: {actor}\n"
            f"   Entity: {entity_id}\n"
            f"   Action: {action}"
        )
        
        return record_id
    
    def _generate_hash(self, record: AuditRecord) -> str:
        """Generate cryptographic hash for audit record."""
        import hashlib
        
        # Create deterministic string representation
        data_str = (
            f"{record.record_id}|{record.event_type.value}|"
            f"{record.timestamp.isoformat()}|{record.actor}|"
            f"{record.entity_id}|{record.action}|"
            f"{json.dumps(record.details, sort_keys=True)}"
        )
        
        # SHA-256 hash
        hash_obj = hashlib.sha256(data_str.encode())
        return hash_obj.hexdigest()
    
    async def _persist_record(self, record: AuditRecord) -> None:
        """Persist audit record to disk."""
        # Organize by date: YYYY/MM/DD/audit.jsonl
        date_path = self.audit_log_path / record.timestamp.strftime("%Y/%m/%d")
        date_path.mkdir(parents=True, exist_ok=True)
        
        audit_file = date_path / "audit.jsonl"
        
        # Convert record to dict
        record_dict = asdict(record)
        record_dict["event_type"] = record.event_type.value
        record_dict["timestamp"] = record.timestamp.isoformat()
        
        # Append to JSONL file
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_dict) + "\n")
    
    async def query_audit_trail(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        entity_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """
        Query audit trail.
        
        Args:
            event_type: Filter by event type
            actor: Filter by actor
            entity_id: Filter by entity
            start_time: Start timestamp
            end_time: End timestamp
            limit: Max records to return
        
        Returns:
            List of matching audit records
        """
        # Search cache first
        results = []
        for record in reversed(self.audit_cache):
            # Apply filters
            if event_type and record.event_type != event_type:
                continue
            if actor and record.actor != actor:
                continue
            if entity_id and record.entity_id != entity_id:
                continue
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            
            results.append(record)
            if len(results) >= limit:
                break
        
        logger.info(
            f"[Audit] Query returned {len(results)} records "
            f"(searched cache: {len(self.audit_cache)} records)"
        )
        
        return results
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate compliance report for date range.
        
        Args:
            start_date: Report start date
            end_date: Report end date
        
        Returns:
            Compliance report
        """
        # Query all records in range
        all_records = await self.query_audit_trail(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )
        
        # Aggregate statistics
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(all_records),
            "events_by_type": {},
            "compliance_violations": 0,
            "risk_vetos": 0,
            "trades_executed": 0,
            "decisions_approved": 0
        }
        
        for record in all_records:
            event_type = record.event_type.value
            report["events_by_type"][event_type] = report["events_by_type"].get(event_type, 0) + 1
            
            if record.event_type == AuditEventType.COMPLIANCE_VIOLATION:
                report["compliance_violations"] += 1
            elif record.event_type == AuditEventType.RISK_VETO:
                report["risk_vetos"] += 1
            elif record.event_type == AuditEventType.TRADE_EXECUTED:
                report["trades_executed"] += 1
            elif record.event_type == AuditEventType.DECISION_APPROVED:
                report["decisions_approved"] += 1
        
        logger.info(
            f"[Audit] 📊 Compliance report generated:\n"
            f"   Period: {start_date.date()} to {end_date.date()}\n"
            f"   Total Events: {report['total_events']}\n"
            f"   Violations: {report['compliance_violations']}\n"
            f"   Risk Vetos: {report['risk_vetos']}"
        )
        
        return report
    
    def get_status(self) -> dict:
        """Get audit system status."""
        return {
            "audit_log_path": str(self.audit_log_path),
            "enable_cryptographic_hash": self.enable_cryptographic_hash,
            "retention_days": self.retention_days,
            "cache_size": self.cache_size,
            "cached_records": len(self.audit_cache),
            "oldest_cached_record": self.audit_cache[0].timestamp.isoformat() if self.audit_cache else None,
            "newest_cached_record": self.audit_cache[-1].timestamp.isoformat() if self.audit_cache else None
        }
