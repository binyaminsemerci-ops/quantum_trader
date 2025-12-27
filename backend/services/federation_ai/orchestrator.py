"""
Federation AI Orchestrator
===========================

Central coordinator for all Federation roles.

Responsibilities:
- Route events to appropriate roles
- Collect decisions from all roles
- Apply priority system (Supervisor > CRO > CEO > CIO > CFO > Researcher)
- Publish decisions to EventBus
- Handle decision conflicts
- Track decision history

Architecture:
- Event-driven: Subscribes to portfolio.*, system.*, model.* events
- Publishes: ai.federation.decision_made events
- Priority enforcement: Higher priority roles can veto/override
"""

from typing import List, Dict
import asyncio
from datetime import datetime
import structlog

from backend.services.federation_ai.roles.ceo import AICEO
from backend.services.federation_ai.roles.cio import AICIO
from backend.services.federation_ai.roles.cro import AICRO
from backend.services.federation_ai.roles.cfo import AICFO
from backend.services.federation_ai.roles.researcher import AIResearcher
from backend.services.federation_ai.roles.supervisor import AISupervisor
from backend.services.federation_ai.models import (
    FederationDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    ModelPerformance,
    DecisionPriority,
)

logger = structlog.get_logger(__name__)


class FederationOrchestrator:
    """
    Federation AI Orchestrator
    
    Coordinates all 6 executive roles and manages decision flow.
    """
    
    # Role priority hierarchy (lower number = higher priority)
    ROLE_PRIORITY = {
        "supervisor": 1,  # Highest - can override anyone
        "cro": 2,         # Risk manager
        "ceo": 3,         # Governor
        "cio": 4,         # Strategy allocator
        "cfo": 5,         # Capital allocator
        "researcher": 6,  # Lowest - advisory only
    }
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        
        # Initialize all roles
        self.ceo = AICEO()
        self.cio = AICIO()
        self.cro = AICRO()
        self.cfo = AICFO()
        self.researcher = AIResearcher()
        self.supervisor = AISupervisor()
        
        self.all_roles = [
            self.ceo,
            self.cio,
            self.cro,
            self.cfo,
            self.researcher,
            self.supervisor,
        ]
        
        # Decision history
        self.decision_history: List[FederationDecision] = []
        self.pending_decisions: List[FederationDecision] = []
        
        self.logger.info("Federation AI Orchestrator initialized", num_roles=len(self.all_roles))
    
    async def on_portfolio_update(self, snapshot: PortfolioSnapshot):
        """
        Route portfolio update to all roles.
        
        Flow:
        1. Collect decisions from all roles
        2. Apply priority system
        3. Check for conflicts
        4. Publish final decisions
        """
        self.logger.debug(
            "Portfolio update received",
            equity=snapshot.total_equity,
            dd_pct=snapshot.drawdown_pct,
            pnl_today=snapshot.realized_pnl_today,
        )
        
        # Collect decisions from all roles in parallel
        decision_tasks = [
            role.on_portfolio_update(snapshot)
            for role in self.all_roles
            if role.is_enabled()
        ]
        
        decisions_per_role = await asyncio.gather(*decision_tasks, return_exceptions=True)
        
        # Flatten and filter
        all_decisions = []
        for i, decisions in enumerate(decisions_per_role):
            if isinstance(decisions, Exception):
                self.logger.error(
                    "Role error",
                    role=self.all_roles[i].role_name,
                    error=str(decisions),
                )
                continue
            
            all_decisions.extend(decisions)
        
        # Process decisions
        await self._process_decisions(all_decisions)
    
    async def on_health_update(self, health: SystemHealthSnapshot):
        """
        Route system health update to all roles.
        """
        self.logger.debug(
            "Health update received",
            system_status=health.system_status,
            ess_state=health.ess_state,
        )
        
        # Collect decisions
        decision_tasks = [
            role.on_health_update(health)
            for role in self.all_roles
            if role.is_enabled()
        ]
        
        decisions_per_role = await asyncio.gather(*decision_tasks, return_exceptions=True)
        
        # Flatten and filter
        all_decisions = []
        for i, decisions in enumerate(decisions_per_role):
            if isinstance(decisions, Exception):
                self.logger.error(
                    "Role error",
                    role=self.all_roles[i].role_name,
                    error=str(decisions),
                )
                continue
            
            all_decisions.extend(decisions)
        
        # Process decisions
        await self._process_decisions(all_decisions)
    
    async def on_model_update(self, performance: ModelPerformance):
        """
        Route model performance update to relevant roles.
        """
        self.logger.debug(
            "Model update received",
            model=performance.model_name,
            sharpe=performance.sharpe_ratio,
            win_rate=performance.win_rate,
        )
        
        # Only CIO and Researcher care about model performance
        decision_tasks = [
            self.cio.on_model_update(performance),
            self.researcher.on_model_update(performance),
        ]
        
        decisions_per_role = await asyncio.gather(*decision_tasks, return_exceptions=True)
        
        # Flatten
        all_decisions = []
        for decisions in decisions_per_role:
            if isinstance(decisions, Exception):
                self.logger.error("Model update error", error=str(decisions))
                continue
            
            all_decisions.extend(decisions)
        
        # Process decisions
        await self._process_decisions(all_decisions)
    
    async def _process_decisions(self, decisions: List[FederationDecision]):
        """
        Process decisions: apply priority, check conflicts, publish.
        
        Priority Rules:
        1. Supervisor can override any decision
        2. Within same decision type, highest role priority wins
        3. CRITICAL priority decisions execute immediately
        """
        if not decisions:
            return
        
        self.logger.info("Processing decisions", num_decisions=len(decisions))
        
        # Sort by priority (CRITICAL first, then by role hierarchy)
        sorted_decisions = sorted(
            decisions,
            key=lambda d: (
                d.priority.value,  # DecisionPriority: CRITICAL=1, HIGH=2, etc.
                self.ROLE_PRIORITY.get(d.role_source, 10),  # Role priority
            )
        )
        
        # Check for conflicts (same decision type from multiple roles)
        final_decisions = self._resolve_conflicts(sorted_decisions)
        
        # Add to history
        self.decision_history.extend(final_decisions)
        
        # Publish decisions (would integrate with EventBus here)
        for decision in final_decisions:
            await self._publish_decision(decision)
        
        self.logger.info(
            "Decisions processed",
            num_final=len(final_decisions),
            num_conflicts_resolved=len(decisions) - len(final_decisions),
        )
    
    def _resolve_conflicts(self, decisions: List[FederationDecision]) -> List[FederationDecision]:
        """
        Resolve conflicts when multiple roles make decisions on same topic.
        
        Rule: Highest role priority wins for same decision type.
        """
        decision_map: Dict[str, FederationDecision] = {}
        
        for decision in decisions:
            decision_type = decision.decision_type.value
            
            # Check if we already have a decision for this type
            if decision_type in decision_map:
                existing = decision_map[decision_type]
                
                # Compare role priorities
                existing_priority = self.ROLE_PRIORITY.get(existing.role_source, 10)
                new_priority = self.ROLE_PRIORITY.get(decision.role_source, 10)
                
                if new_priority < existing_priority:
                    # New decision has higher priority
                    self.logger.info(
                        "Decision conflict resolved",
                        decision_type=decision_type,
                        winner=decision.role_source,
                        overridden=existing.role_source,
                    )
                    decision_map[decision_type] = decision
                else:
                    self.logger.debug(
                        "Decision ignored (lower priority)",
                        decision_type=decision_type,
                        role=decision.role_source,
                    )
            else:
                # First decision of this type
                decision_map[decision_type] = decision
        
        return list(decision_map.values())
    
    async def _publish_decision(self, decision: FederationDecision):
        """
        Publish decision to EventBus.
        
        Event format:
        {
            "event_type": "ai.federation.decision_made",
            "decision_id": "uuid",
            "decision_type": "MODE_CHANGE",
            "role_source": "ceo",
            "priority": 1,
            "timestamp": "2024-01-01T00:00:00Z",
            "reason": "...",
            "payload": {...}
        }
        """
        # TODO: Integrate with actual EventBus
        # For now, just log
        self.logger.info(
            "Decision published",
            decision_id=decision.decision_id,
            decision_type=decision.decision_type.value,
            role=decision.role_source,
            priority=decision.priority.value,
        )
        
        # In production:
        # await event_bus.publish("ai.federation.decision_made", decision.model_dump())
    
    def get_decision_history(self, limit: int = 100) -> List[FederationDecision]:
        """Get recent decision history"""
        return self.decision_history[-limit:]
    
    def get_role_status(self) -> Dict[str, bool]:
        """Get status of all roles"""
        return {
            role.role_name: role.is_enabled()
            for role in self.all_roles
        }
    
    def enable_role(self, role_name: str):
        """Enable a specific role"""
        for role in self.all_roles:
            if role.role_name == role_name:
                role.enable()
                self.logger.info("Role enabled", role=role_name)
                return
        
        self.logger.warning("Role not found", role=role_name)
    
    def disable_role(self, role_name: str):
        """Disable a specific role"""
        for role in self.all_roles:
            if role.role_name == role_name:
                role.disable()
                self.logger.info("Role disabled", role=role_name)
                return
        
        self.logger.warning("Role not found", role=role_name)
