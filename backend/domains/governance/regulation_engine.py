"""
Governance Domain - Regulation Engine

Dynamic regulatory compliance and policy enforcement.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RegulationType(Enum):
    """Types of regulations."""
    POSITION_LIMIT = "position_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    TRADING_HOURS = "trading_hours"
    CONCENTRATION_LIMIT = "concentration_limit"
    MARKET_ACCESS = "market_access"
    REPORTING_REQUIREMENT = "reporting_requirement"


class RegulatoryJurisdiction(Enum):
    """Regulatory jurisdictions."""
    US_SEC = "us_sec"
    US_CFTC = "us_cftc"
    EU_ESMA = "eu_esma"
    UK_FCA = "uk_fca"
    CRYPTO_EXCHANGE = "crypto_exchange"


@dataclass
class RegulatoryRule:
    """Regulatory rule definition."""
    rule_id: str
    regulation_type: RegulationType
    jurisdiction: RegulatoryJurisdiction
    description: str
    parameters: Dict
    effective_from: datetime
    effective_until: Optional[datetime] = None
    is_active: bool = True


class RegulationEngine:
    """
    Regulation Engine - Dynamic regulatory compliance.
    
    Responsibilities:
    - Maintain regulatory rule database
    - Apply jurisdiction-specific rules
    - Validate trades against regulations
    - Adapt to regulatory changes
    - Generate regulatory reports
    
    Authority: ENFORCER (blocks non-compliant actions)
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        active_jurisdictions: Optional[List[RegulatoryJurisdiction]] = None,
    ):
        """
        Initialize Regulation Engine.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            active_jurisdictions: Active regulatory jurisdictions
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Active jurisdictions (default: crypto exchanges only)
        if active_jurisdictions is None:
            active_jurisdictions = [RegulatoryJurisdiction.CRYPTO_EXCHANGE]
        self.active_jurisdictions = active_jurisdictions
        
        # Regulatory rules database
        self.rules: Dict[str, RegulatoryRule] = {}
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Subscribe to events
        self.event_bus.subscribe("position.opened", self._validate_position)
        self.event_bus.subscribe("compliance.trade.blocked", self._handle_blocked_trade)
        
        logger.info(
            f"[Regulation] Regulation Engine initialized:\n"
            f"   Active Jurisdictions: {', '.join([j.value for j in active_jurisdictions])}\n"
            f"   Active Rules: {len([r for r in self.rules.values() if r.is_active])}"
        )
    
    def _initialize_default_rules(self) -> None:
        """Initialize default regulatory rules."""
        # Crypto exchange rules (Binance-like)
        self.add_rule(
            regulation_type=RegulationType.LEVERAGE_LIMIT,
            jurisdiction=RegulatoryJurisdiction.CRYPTO_EXCHANGE,
            description="Maximum leverage 125x for crypto futures",
            parameters={"max_leverage": 125.0}
        )
        
        self.add_rule(
            regulation_type=RegulationType.POSITION_LIMIT,
            jurisdiction=RegulatoryJurisdiction.CRYPTO_EXCHANGE,
            description="Maximum single position 50% of account",
            parameters={"max_position_pct": 0.50}
        )
        
        # US CFTC rules (if trading US exchanges)
        if RegulatoryJurisdiction.US_CFTC in self.active_jurisdictions:
            self.add_rule(
                regulation_type=RegulationType.LEVERAGE_LIMIT,
                jurisdiction=RegulatoryJurisdiction.US_CFTC,
                description="Maximum leverage 2x for retail (CFTC)",
                parameters={"max_leverage": 2.0, "applies_to": "retail"}
            )
        
        logger.info(f"[Regulation] Initialized {len(self.rules)} default rules")
    
    def add_rule(
        self,
        regulation_type: RegulationType,
        jurisdiction: RegulatoryJurisdiction,
        description: str,
        parameters: Dict,
        effective_from: Optional[datetime] = None,
        effective_until: Optional[datetime] = None
    ) -> str:
        """
        Add regulatory rule.
        
        Args:
            regulation_type: Type of regulation
            jurisdiction: Regulatory jurisdiction
            description: Rule description
            parameters: Rule parameters
            effective_from: Effective start date (None = now)
            effective_until: Effective end date (None = indefinite)
        
        Returns:
            Rule ID
        """
        import uuid
        
        rule_id = f"REG-{uuid.uuid4().hex[:8].upper()}"
        
        if effective_from is None:
            effective_from = datetime.now(timezone.utc)
        
        rule = RegulatoryRule(
            rule_id=rule_id,
            regulation_type=regulation_type,
            jurisdiction=jurisdiction,
            description=description,
            parameters=parameters,
            effective_from=effective_from,
            effective_until=effective_until,
            is_active=True
        )
        
        self.rules[rule_id] = rule
        
        logger.info(
            f"[Regulation] ✅ Rule added: {rule_id}\n"
            f"   Type: {regulation_type.value}\n"
            f"   Jurisdiction: {jurisdiction.value}\n"
            f"   Description: {description}\n"
            f"   Parameters: {parameters}"
        )
        
        return rule_id
    
    async def validate_trade(
        self,
        symbol: str,
        size_usd: float,
        leverage: float,
        account_value: float
    ) -> tuple[bool, List[str]]:
        """
        Validate trade against regulatory rules.
        
        Args:
            symbol: Trading symbol
            size_usd: Position size in USD
            leverage: Leverage used
            account_value: Account value
        
        Returns:
            (is_compliant, violations)
        """
        violations = []
        
        # Check all active rules
        for rule in self.rules.values():
            if not rule.is_active:
                continue
            
            # Check if rule applies to this jurisdiction
            if rule.jurisdiction not in self.active_jurisdictions:
                continue
            
            # Check if rule is currently effective
            now = datetime.now(timezone.utc)
            if now < rule.effective_from:
                continue
            if rule.effective_until and now > rule.effective_until:
                continue
            
            # Validate based on regulation type
            if rule.regulation_type == RegulationType.LEVERAGE_LIMIT:
                max_leverage = rule.parameters.get("max_leverage", float("inf"))
                if leverage > max_leverage:
                    violations.append(
                        f"Leverage {leverage:.0f}x exceeds {rule.jurisdiction.value} "
                        f"limit {max_leverage:.0f}x (rule {rule.rule_id})"
                    )
            
            elif rule.regulation_type == RegulationType.POSITION_LIMIT:
                max_position_pct = rule.parameters.get("max_position_pct", 1.0)
                position_pct = size_usd / account_value if account_value > 0 else 0
                if position_pct > max_position_pct:
                    violations.append(
                        f"Position size {position_pct:.1%} exceeds {rule.jurisdiction.value} "
                        f"limit {max_position_pct:.1%} (rule {rule.rule_id})"
                    )
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            logger.warning(
                f"[Regulation] ⚠️ Trade validation FAILED:\n"
                f"   Symbol: {symbol}\n"
                f"   Size: ${size_usd:,.2f}\n"
                f"   Leverage: {leverage:.0f}x\n"
                f"   Violations: {len(violations)}"
            )
            for violation in violations:
                logger.warning(f"   - {violation}")
        
        return is_compliant, violations
    
    async def _validate_position(self, event_data: dict) -> None:
        """Validate position opening against regulations."""
        position_id = event_data.get("position_id")
        symbol = event_data.get("symbol")
        size_usd = event_data.get("size_usd", 0.0)
        leverage = event_data.get("leverage", 1.0)
        
        # Get account value (placeholder)
        account_value = 10000.0  # TODO: Get from portfolio
        
        is_compliant, violations = await self.validate_trade(
            symbol, size_usd, leverage, account_value
        )
        
        if not is_compliant:
            # Publish regulatory violation event
            await self.event_bus.publish(
                "regulation.violation.detected",
                {
                    "position_id": position_id,
                    "symbol": symbol,
                    "size_usd": size_usd,
                    "leverage": leverage,
                    "violations": violations,
                    "detected_at": datetime.now(timezone.utc).isoformat()
                }
            )
    
    async def _handle_blocked_trade(self, event_data: dict) -> None:
        """Handle blocked trades - analyze regulatory implications."""
        position_id = event_data.get("position_id")
        violation_reason = event_data.get("violation_reason")
        
        logger.info(
            f"[Regulation] Trade blocked by Compliance: {position_id}\n"
            f"   Reason: {violation_reason}"
        )
        
        # Log for regulatory reporting
    
    async def generate_regulatory_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Generate regulatory compliance report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
        
        Returns:
            Regulatory report
        """
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "jurisdictions": [j.value for j in self.active_jurisdictions],
            "active_rules": len([r for r in self.rules.values() if r.is_active]),
            "rules_by_jurisdiction": {},
            "violations_detected": 0,  # TODO: Query from audit trail
            "compliance_rate": 1.0  # TODO: Calculate from audit data
        }
        
        for jurisdiction in self.active_jurisdictions:
            jurisdiction_rules = [
                r for r in self.rules.values()
                if r.jurisdiction == jurisdiction and r.is_active
            ]
            report["rules_by_jurisdiction"][jurisdiction.value] = len(jurisdiction_rules)
        
        logger.info(
            f"[Regulation] 📊 Regulatory report generated:\n"
            f"   Period: {start_date.date()} to {end_date.date()}\n"
            f"   Jurisdictions: {len(self.active_jurisdictions)}\n"
            f"   Active Rules: {report['active_rules']}\n"
            f"   Compliance Rate: {report['compliance_rate']:.1%}"
        )
        
        return report
    
    def get_status(self) -> dict:
        """Get regulation engine status."""
        active_rules = [r for r in self.rules.values() if r.is_active]
        
        rules_by_type = {}
        for rule in active_rules:
            rule_type = rule.regulation_type.value
            rules_by_type[rule_type] = rules_by_type.get(rule_type, 0) + 1
        
        return {
            "active_jurisdictions": [j.value for j in self.active_jurisdictions],
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "rules_by_type": rules_by_type,
            "rules_by_jurisdiction": {
                j.value: len([r for r in active_rules if r.jurisdiction == j])
                for j in self.active_jurisdictions
            }
        }
