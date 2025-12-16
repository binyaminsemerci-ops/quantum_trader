"""
Federation AI v3 - Base Role
=============================

Abstract base class for all Federation AI roles.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import structlog

from backend.services.federation_ai.models import (
    FederationDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    ModelPerformance,
)

logger = structlog.get_logger(__name__)


class FederationRole(ABC):
    """
    Abstract base class for Federation AI roles.
    
    Each role:
    - Receives input events (portfolio, health, model performance)
    - Makes decisions based on its domain expertise
    - Returns FederationDecision objects for orchestrator
    """
    
    def __init__(self, role_name: str):
        self.role_name = role_name
        self.logger = logger.bind(role=role_name)
        self.enabled = True
    
    @abstractmethod
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Handle portfolio state update.
        
        Args:
            snapshot: Current portfolio snapshot
            
        Returns:
            List of decisions to execute
        """
        pass
    
    @abstractmethod
    async def on_health_update(
        self,
        health: SystemHealthSnapshot
    ) -> List[FederationDecision]:
        """
        Handle system health update.
        
        Args:
            health: Current system health
            
        Returns:
            List of decisions to execute
        """
        pass
    
    async def on_model_update(
        self,
        performance: ModelPerformance
    ) -> List[FederationDecision]:
        """
        Handle model performance update (optional for roles).
        
        Args:
            performance: Model performance metrics
            
        Returns:
            List of decisions to execute
        """
        return []
    
    def enable(self):
        """Enable this role"""
        self.enabled = True
        self.logger.info(f"{self.role_name} enabled")
    
    def disable(self):
        """Disable this role"""
        self.enabled = False
        self.logger.info(f"{self.role_name} disabled")
    
    def is_enabled(self) -> bool:
        """Check if role is enabled"""
        return self.enabled
