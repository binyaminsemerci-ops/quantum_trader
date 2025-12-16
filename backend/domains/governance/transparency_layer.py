"""
Governance Domain - Decision Transparency Layer

Explainability and interpretability for all system decisions.

Author: Quantum Trader - Hedge Fund OS v2
Date: December 3, 2025
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionCategory(Enum):
    """Categories of decisions."""
    TRADE_EXECUTION = "trade_execution"
    RISK_MANAGEMENT = "risk_management"
    CAPITAL_ALLOCATION = "capital_allocation"
    STRATEGIC_PLANNING = "strategic_planning"
    COMPLIANCE_ENFORCEMENT = "compliance_enforcement"


@dataclass
class DecisionExplanation:
    """Explanation for a system decision."""
    decision_id: str
    category: DecisionCategory
    timestamp: datetime
    decision_maker: str  # CEO, CRO, CIO, AI Model, etc.
    decision: str
    rationale: str
    input_factors: Dict
    alternatives_considered: List[Dict]
    confidence_score: float
    explainability_score: float


class DecisionTransparencyLayer:
    """
    Decision Transparency Layer - Explainable AI decisions.
    
    Responsibilities:
    - Explain all system decisions in human terms
    - Provide rationale for trade/risk/allocation decisions
    - Track decision quality over time
    - Generate transparency reports for stakeholders
    - Enable decision auditing and review
    
    Authority: OBSERVER (explains, does not decide)
    """
    
    def __init__(
        self,
        policy_store,
        event_bus,
        min_explainability_score: float = 0.70,  # 70% explainability required
    ):
        """
        Initialize Decision Transparency Layer.
        
        Args:
            policy_store: PolicyStore v2 instance
            event_bus: EventBus v2 instance
            min_explainability_score: Minimum explainability score required
        """
        self.policy_store = policy_store
        self.event_bus = event_bus
        
        # Configuration
        self.min_explainability_score = min_explainability_score
        
        # Decision tracking
        self.decisions: List[DecisionExplanation] = []
        self.cache_size = 500  # Keep last 500 decisions
        
        # Subscribe to decision events
        self._subscribe_to_events()
        
        logger.info(
            f"[Transparency] Decision Transparency Layer initialized:\n"
            f"   Min Explainability Score: {min_explainability_score:.0%}\n"
            f"   Cache Size: {self.cache_size} decisions"
        )
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to decision-making events."""
        events = [
            "position.opened",
            "position.closed",
            "fund.risk.veto.issued",
            "fund.strategy.allocated",
            "governance.decision.approved",
            "fund.directive.issued",
        ]
        
        for event_name in events:
            self.event_bus.subscribe(event_name, self._explain_decision)
        
        logger.info(f"[Transparency] Subscribed to {len(events)} decision events")
    
    async def _explain_decision(self, event_data: dict) -> None:
        """Generate explanation for a decision event."""
        event_name = event_data.get("_event_name", "unknown")
        
        # Map event to decision category
        category_mapping = {
            "position.opened": DecisionCategory.TRADE_EXECUTION,
            "position.closed": DecisionCategory.TRADE_EXECUTION,
            "fund.risk.veto.issued": DecisionCategory.RISK_MANAGEMENT,
            "fund.strategy.allocated": DecisionCategory.CAPITAL_ALLOCATION,
            "governance.decision.approved": DecisionCategory.STRATEGIC_PLANNING,
            "fund.directive.issued": DecisionCategory.STRATEGIC_PLANNING,
        }
        
        category = category_mapping.get(event_name)
        if not category:
            return
        
        # Generate explanation based on event type
        if event_name == "position.opened":
            await self._explain_trade_decision(event_data, category)
        elif event_name == "fund.risk.veto.issued":
            await self._explain_risk_decision(event_data, category)
        elif event_name == "fund.strategy.allocated":
            await self._explain_allocation_decision(event_data, category)
        elif event_name == "governance.decision.approved":
            await self._explain_governance_decision(event_data, category)
    
    async def _explain_trade_decision(
        self,
        event_data: dict,
        category: DecisionCategory
    ) -> None:
        """Explain trade execution decision."""
        position_id = event_data.get("position_id")
        symbol = event_data.get("symbol")
        side = event_data.get("side")
        size_usd = event_data.get("size_usd", 0.0)
        confidence = event_data.get("confidence", 0.0)
        
        # Extract decision factors
        input_factors = {
            "symbol": symbol,
            "side": side,
            "size_usd": size_usd,
            "confidence": confidence,
            "strategy": event_data.get("strategy_id", "unknown"),
            "signal_strength": event_data.get("signal_strength", "unknown")
        }
        
        # Generate rationale
        rationale = (
            f"Trade executed for {symbol} ({side}) with ${size_usd:,.2f} position size. "
            f"Decision based on {input_factors['strategy']} strategy signals with "
            f"{confidence:.1%} confidence."
        )
        
        # Alternatives (simplified)
        alternatives = [
            {"action": "no_trade", "reason": "Confidence below threshold"},
            {"action": "smaller_size", "reason": "Risk-adjusted sizing"},
        ]
        
        # Calculate explainability score
        explainability = self._calculate_explainability(input_factors, confidence)
        
        # Create explanation
        explanation = DecisionExplanation(
            decision_id=position_id,
            category=category,
            timestamp=datetime.now(timezone.utc),
            decision_maker=event_data.get("executed_by", "AI_System"),
            decision=f"Open {side} position on {symbol}",
            rationale=rationale,
            input_factors=input_factors,
            alternatives_considered=alternatives,
            confidence_score=confidence,
            explainability_score=explainability
        )
        
        await self._record_explanation(explanation)
    
    async def _explain_risk_decision(
        self,
        event_data: dict,
        category: DecisionCategory
    ) -> None:
        """Explain risk management decision (veto)."""
        veto_id = event_data.get("veto_id")
        decision_type = event_data.get("decision_type")
        reason = event_data.get("reason")
        risk_metrics = event_data.get("risk_metrics", {})
        
        # Generate rationale
        rationale = (
            f"CRO issued veto ({decision_type}) due to risk concerns: {reason}. "
            f"Risk metrics: {risk_metrics}"
        )
        
        # Create explanation
        explanation = DecisionExplanation(
            decision_id=veto_id,
            category=category,
            timestamp=datetime.now(timezone.utc),
            decision_maker="CRO",
            decision=f"Veto: {decision_type}",
            rationale=rationale,
            input_factors=risk_metrics,
            alternatives_considered=[
                {"action": "allow_with_conditions", "reason": "Risk acceptable with constraints"}
            ],
            confidence_score=1.0,  # CRO veto is absolute
            explainability_score=0.95  # Very transparent
        )
        
        await self._record_explanation(explanation)
    
    async def _explain_allocation_decision(
        self,
        event_data: dict,
        category: DecisionCategory
    ) -> None:
        """Explain capital allocation decision."""
        strategy_id = event_data.get("strategy_id")
        allocation = event_data.get("capital_allocation")
        expected_return = event_data.get("expected_return", 0.0)
        max_drawdown = event_data.get("max_drawdown", 0.0)
        reason = event_data.get("allocation_reason", "No reason provided")
        
        # Generate rationale
        rationale = (
            f"CEO approved {allocation:.1%} capital allocation to {strategy_id}. "
            f"Expected return: {expected_return:.1%}, Max drawdown: {max_drawdown:.1%}. "
            f"Rationale: {reason}"
        )
        
        # Create explanation
        explanation = DecisionExplanation(
            decision_id=f"{strategy_id}-ALLOC",
            category=category,
            timestamp=datetime.now(timezone.utc),
            decision_maker="CEO",
            decision=f"Allocate {allocation:.1%} to {strategy_id}",
            rationale=rationale,
            input_factors={
                "strategy_id": strategy_id,
                "allocation": allocation,
                "expected_return": expected_return,
                "max_drawdown": max_drawdown
            },
            alternatives_considered=[
                {"action": "reject", "reason": "Allocation exceeds limits"},
                {"action": "reduce_allocation", "reason": "Risk-adjusted sizing"}
            ],
            confidence_score=0.85,
            explainability_score=0.90
        )
        
        await self._record_explanation(explanation)
    
    async def _explain_governance_decision(
        self,
        event_data: dict,
        category: DecisionCategory
    ) -> None:
        """Explain governance decision."""
        decision_id = event_data.get("decision_id")
        decision_type = event_data.get("decision_type")
        approve_votes = event_data.get("approve_votes", 0)
        total_votes = event_data.get("total_votes", 0)
        
        # Generate rationale
        rationale = (
            f"Governance decision ({decision_type}) approved by consensus. "
            f"Vote: {approve_votes}/{total_votes} approve."
        )
        
        # Create explanation
        explanation = DecisionExplanation(
            decision_id=decision_id,
            category=category,
            timestamp=datetime.now(timezone.utc),
            decision_maker="Federation",
            decision=f"Approve: {decision_type}",
            rationale=rationale,
            input_factors={
                "decision_type": decision_type,
                "approve_votes": approve_votes,
                "total_votes": total_votes,
                "approval_rate": approve_votes / total_votes if total_votes > 0 else 0
            },
            alternatives_considered=[
                {"action": "reject", "reason": "Insufficient votes"}
            ],
            confidence_score=approve_votes / total_votes if total_votes > 0 else 0,
            explainability_score=0.95
        )
        
        await self._record_explanation(explanation)
    
    def _calculate_explainability(
        self,
        input_factors: Dict,
        confidence: float
    ) -> float:
        """
        Calculate explainability score.
        
        Args:
            input_factors: Decision input factors
            confidence: Decision confidence
        
        Returns:
            Explainability score (0-1)
        """
        # Factors contributing to explainability:
        # 1. Number of input factors (more = better)
        # 2. Confidence level (higher = better)
        # 3. Presence of rationale (yes/no)
        
        factor_score = min(len(input_factors) / 10.0, 1.0)  # Max at 10 factors
        confidence_score = confidence
        
        explainability = (factor_score * 0.4) + (confidence_score * 0.6)
        
        return explainability
    
    async def _record_explanation(self, explanation: DecisionExplanation) -> None:
        """Record decision explanation."""
        # Add to cache
        self.decisions.append(explanation)
        if len(self.decisions) > self.cache_size:
            self.decisions.pop(0)
        
        # Check if explainability meets threshold
        if explanation.explainability_score < self.min_explainability_score:
            logger.warning(
                f"[Transparency] ⚠️ Low explainability: {explanation.decision_id}\n"
                f"   Score: {explanation.explainability_score:.1%} < "
                f"{self.min_explainability_score:.1%}\n"
                f"   Decision: {explanation.decision}\n"
                f"   Decision Maker: {explanation.decision_maker}"
            )
            
            # Publish low explainability warning
            await self.event_bus.publish(
                "transparency.low_explainability",
                {
                    "decision_id": explanation.decision_id,
                    "explainability_score": explanation.explainability_score,
                    "min_required": self.min_explainability_score,
                    "decision_maker": explanation.decision_maker,
                    "category": explanation.category.value
                }
            )
        else:
            logger.debug(
                f"[Transparency] ✅ Decision explained: {explanation.decision_id}\n"
                f"   Category: {explanation.category.value}\n"
                f"   Decision Maker: {explanation.decision_maker}\n"
                f"   Explainability: {explanation.explainability_score:.1%}"
            )
    
    async def get_decision_explanation(self, decision_id: str) -> Optional[DecisionExplanation]:
        """
        Get explanation for a specific decision.
        
        Args:
            decision_id: Decision identifier
        
        Returns:
            Decision explanation or None
        """
        for explanation in reversed(self.decisions):
            if explanation.decision_id == decision_id:
                return explanation
        
        return None
    
    async def generate_transparency_report(
        self,
        category: Optional[DecisionCategory] = None
    ) -> Dict:
        """
        Generate transparency report.
        
        Args:
            category: Filter by decision category (None = all)
        
        Returns:
            Transparency report
        """
        # Filter decisions
        filtered = self.decisions
        if category:
            filtered = [d for d in self.decisions if d.category == category]
        
        # Calculate metrics
        if not filtered:
            avg_explainability = 0.0
            avg_confidence = 0.0
            low_explainability_count = 0
        else:
            avg_explainability = sum(d.explainability_score for d in filtered) / len(filtered)
            avg_confidence = sum(d.confidence_score for d in filtered) / len(filtered)
            low_explainability_count = len([
                d for d in filtered if d.explainability_score < self.min_explainability_score
            ])
        
        # Group by decision maker
        by_maker = {}
        for decision in filtered:
            maker = decision.decision_maker
            if maker not in by_maker:
                by_maker[maker] = []
            by_maker[maker].append(decision)
        
        report = {
            "total_decisions": len(filtered),
            "avg_explainability": avg_explainability,
            "avg_confidence": avg_confidence,
            "low_explainability_count": low_explainability_count,
            "low_explainability_rate": low_explainability_count / len(filtered) if filtered else 0,
            "decisions_by_maker": {
                maker: len(decisions) for maker, decisions in by_maker.items()
            },
            "category": category.value if category else "all"
        }
        
        logger.info(
            f"[Transparency] 📊 Transparency report:\n"
            f"   Total Decisions: {report['total_decisions']}\n"
            f"   Avg Explainability: {report['avg_explainability']:.1%}\n"
            f"   Avg Confidence: {report['avg_confidence']:.1%}\n"
            f"   Low Explainability: {low_explainability_count} "
            f"({report['low_explainability_rate']:.1%})"
        )
        
        return report
    
    def get_status(self) -> dict:
        """Get transparency layer status."""
        recent_decisions = self.decisions[-10:] if self.decisions else []
        
        return {
            "min_explainability_score": self.min_explainability_score,
            "cache_size": self.cache_size,
            "cached_decisions": len(self.decisions),
            "recent_decisions": [
                {
                    "decision_id": d.decision_id,
                    "category": d.category.value,
                    "decision_maker": d.decision_maker,
                    "explainability_score": d.explainability_score,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in recent_decisions
            ]
        }
