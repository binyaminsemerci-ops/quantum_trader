"""
Governance Domain - Compliance Operating System

Real-time compliance monitoring and enforcement.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceRuleType(Enum):
    """Types of compliance rules."""
    POSITION_LIMIT = "position_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    TRADING_HOUR = "trading_hour"
    MARKET_MANIPULATION = "market_manipulation"
    WASH_TRADING = "wash_trading"


class ViolationSeverity(Enum):
    """Severity levels for violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    """Compliance violation record."""
    violation_id: str
    rule_type: ComplianceRuleType
    severity: ViolationSeverity
    description: str
    entity_id: str  # position_id, strategy_id, etc.
    detected_at: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None


class ComplianceOS:
    """
    Compliance Operating System - Real-time compliance monitoring.
    
    Responsibilities:
    - Monitor all trading activity for compliance violations
    - Enforce position/leverage/concentration limits
    - Detect suspicious patterns (wash trading, manipulation)
    - Generate compliance reports
    - Block non-compliant trades
    
    Authority: ENFORCER (can block trades, escalate to CRO)
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        max_position_size: float = 0.15,      # 15% max single position
        max_leverage: float = 30.0,           # 30x max leverage
        max_sector_concentration: float = 0.40,  # 40% max in one sector
        enable_wash_trading_detection: bool = True,
    ):
        """
        Initialize Compliance OS.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            max_position_size: Max single position size (% of portfolio)
            max_leverage: Maximum allowed leverage
            max_sector_concentration: Max sector concentration
            enable_wash_trading_detection: Enable wash trading detection
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Compliance limits
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.max_sector_concentration = max_sector_concentration
        self.enable_wash_trading_detection = enable_wash_trading_detection
        
        # Violation tracking
        self.violations: List[ComplianceViolation] = []
        self.blocked_trades: Dict[str, ComplianceViolation] = {}
        
        # Subscribe to events
        self.event_bus.subscribe("position.opened", self._check_position_compliance)
        self.event_bus.subscribe("position.closed", self._check_wash_trading)
        self.event_bus.subscribe("fund.strategy.allocated", self._check_allocation_compliance)
        
        logger.info(
            f"[Compliance] Compliance OS initialized:\n"
            f"   Max Position Size: {max_position_size:.1%}\n"
            f"   Max Leverage: {max_leverage:.0f}x\n"
            f"   Max Sector Concentration: {max_sector_concentration:.1%}\n"
            f"   Wash Trading Detection: {enable_wash_trading_detection}"
        )
    
    async def check_trade_compliance(
        self,
        position_id: str,
        symbol: str,
        size_usd: float,
        leverage: float
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a trade is compliant before execution.
        
        Args:
            position_id: Position identifier
            symbol: Trading symbol
            size_usd: Position size in USD
            leverage: Leverage used
        
        Returns:
            (is_compliant, violation_reason)
        """
        # Check leverage limit
        if leverage > self.max_leverage:
            violation = await self._record_violation(
                ComplianceRuleType.LEVERAGE_LIMIT,
                ViolationSeverity.HIGH,
                f"Leverage {leverage:.0f}x exceeds max {self.max_leverage:.0f}x",
                position_id
            )
            return False, f"Leverage violation: {leverage:.0f}x > {self.max_leverage:.0f}x"
        
        # Check position size limit
        portfolio_value = await self._get_portfolio_value()
        position_pct = size_usd / portfolio_value if portfolio_value > 0 else 0
        
        if position_pct > self.max_position_size:
            violation = await self._record_violation(
                ComplianceRuleType.POSITION_LIMIT,
                ViolationSeverity.HIGH,
                f"Position size {position_pct:.1%} exceeds max {self.max_position_size:.1%}",
                position_id
            )
            return False, f"Position size violation: {position_pct:.1%} > {self.max_position_size:.1%}"
        
        # All checks passed
        return True, None
    
    async def _check_position_compliance(self, event_data: dict) -> None:
        """Check position opening compliance."""
        position_id = event_data.get("position_id")
        symbol = event_data.get("symbol")
        size_usd = event_data.get("size_usd", 0.0)
        leverage = event_data.get("leverage", 1.0)
        
        is_compliant, violation_reason = await self.check_trade_compliance(
            position_id, symbol, size_usd, leverage
        )
        
        if not is_compliant:
            logger.critical(
                f"[Compliance] 🚫 TRADE BLOCKED: {position_id}\n"
                f"   Symbol: {symbol}\n"
                f"   Size: ${size_usd:,.2f}\n"
                f"   Leverage: {leverage:.0f}x\n"
                f"   Reason: {violation_reason}"
            )
            
            # Publish block event
            await self.event_bus.publish(
                "compliance.trade.blocked",
                {
                    "position_id": position_id,
                    "symbol": symbol,
                    "size_usd": size_usd,
                    "leverage": leverage,
                    "violation_reason": violation_reason,
                    "blocked_at": datetime.now(timezone.utc).isoformat(),
                    "blocked_by": "ComplianceOS"
                }
            )
        else:
            logger.debug(
                f"[Compliance] ✅ Trade compliant: {position_id} ({symbol})"
            )
    
    async def _check_wash_trading(self, event_data: dict) -> None:
        """Check for potential wash trading patterns."""
        if not self.enable_wash_trading_detection:
            return
        
        position_id = event_data.get("position_id")
        symbol = event_data.get("symbol")
        side = event_data.get("side")
        closed_at = event_data.get("closed_at")
        
        # TODO: Implement wash trading detection logic
        # Look for:
        # - Same symbol traded in opposite directions within short time
        # - Minimal profit/loss (potential offsetting)
        # - High frequency of reversals
        
        logger.debug(
            f"[Compliance] Wash trading check: {position_id} ({symbol})"
        )
    
    async def _check_allocation_compliance(self, event_data: dict) -> None:
        """Check capital allocation compliance."""
        strategy_id = event_data.get("strategy_id")
        allocation = event_data.get("capital_allocation")
        
        # Check if allocation exceeds limits
        if allocation > self.max_position_size:
            await self._record_violation(
                ComplianceRuleType.CONCENTRATION_LIMIT,
                ViolationSeverity.MEDIUM,
                f"Strategy allocation {allocation:.1%} exceeds max {self.max_position_size:.1%}",
                strategy_id
            )
            
            logger.warning(
                f"[Compliance] ⚠️ Strategy allocation violation: "
                f"{strategy_id} = {allocation:.1%}"
            )
    
    async def _record_violation(
        self,
        rule_type: ComplianceRuleType,
        severity: ViolationSeverity,
        description: str,
        entity_id: str
    ) -> ComplianceViolation:
        """Record compliance violation."""
        import uuid
        
        violation_id = f"COMP-VIO-{uuid.uuid4().hex[:8].upper()}"
        
        violation = ComplianceViolation(
            violation_id=violation_id,
            rule_type=rule_type,
            severity=severity,
            description=description,
            entity_id=entity_id,
            detected_at=datetime.now(timezone.utc),
            resolved=False
        )
        
        self.violations.append(violation)
        
        logger.warning(
            f"[Compliance] ⚠️ Violation recorded: {violation_id}\n"
            f"   Rule: {rule_type.value}\n"
            f"   Severity: {severity.value}\n"
            f"   Entity: {entity_id}\n"
            f"   Description: {description}"
        )
        
        # Publish violation event
        await self.event_bus.publish(
            "compliance.violation.detected",
            {
                "violation_id": violation_id,
                "rule_type": rule_type.value,
                "severity": severity.value,
                "description": description,
                "entity_id": entity_id,
                "detected_at": violation.detected_at.isoformat()
            }
        )
        
        # Escalate critical violations to CRO
        if severity == ViolationSeverity.CRITICAL:
            await self.event_bus.publish(
                "compliance.violation.escalated",
                {
                    "violation_id": violation_id,
                    "escalated_to": "CRO",
                    "severity": severity.value,
                    "description": description
                }
            )
        
        return violation
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        # TODO: Integrate with actual portfolio tracking
        return 10000.0  # Placeholder
    
    def get_status(self) -> dict:
        """Get compliance status."""
        unresolved_violations = [v for v in self.violations if not v.resolved]
        
        violations_by_severity = {
            "critical": len([v for v in unresolved_violations if v.severity == ViolationSeverity.CRITICAL]),
            "high": len([v for v in unresolved_violations if v.severity == ViolationSeverity.HIGH]),
            "medium": len([v for v in unresolved_violations if v.severity == ViolationSeverity.MEDIUM]),
            "low": len([v for v in unresolved_violations if v.severity == ViolationSeverity.LOW])
        }
        
        return {
            "max_position_size": self.max_position_size,
            "max_leverage": self.max_leverage,
            "max_sector_concentration": self.max_sector_concentration,
            "total_violations": len(self.violations),
            "unresolved_violations": len(unresolved_violations),
            "violations_by_severity": violations_by_severity,
            "blocked_trades": len(self.blocked_trades)
        }
