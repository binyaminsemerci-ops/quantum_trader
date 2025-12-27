"""
Federation AI v3 Service
========================

Supreme executive layer for AI orchestration, strategy coordination,
risk governance, and capital allocation.

Roles:
- AI-CEO: Global mode, capital profile, on/off control
- AI-CIO: Strategy allocation, symbol universe management
- AI-CRO: Global risk limits, ESS tuning, exposure control
- AI-CFO: Profit allocation, cash buffer, reinvestment
- AI-Researcher: Research task generation, AutoML proposals
- AI-Supervisor: Override, rollback, freeze trading

Author: Quantum Trader AI Team
Date: December 4, 2025
Version: 3.0 (EPIC-FAI-001)
"""

from .orchestrator import FederationOrchestrator
from .models import (
    FederationDecision,
    CapitalProfileDecision,
    TradingModeDecision,
    RiskAdjustmentDecision,
    ESSPolicyDecision,
    StrategyAllocationDecision,
    SymbolUniverseDecision,
    CashflowDecision,
    ResearchTaskDecision,
    OverrideDecision,
    FreezeDecision,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    ModelPerformance,
    TradingMode,
    CapitalProfile,
    DecisionPriority,
    DecisionType,
)
from .roles.ceo import AICEO
from .roles.cio import AICIO
from .roles.cro import AICRO
from .roles.cfo import AICFO
from .roles.researcher import AIResearcher
from .roles.supervisor import AISupervisor
from .adapters import (
    PolicyStoreAdapter,
    PortfolioAdapter,
    AIEngineAdapter,
    ESSAdapter,
)

__version__ = "3.0.0"

__all__ = [
    # Orchestrator
    "FederationOrchestrator",
    
    # Decision Models
    "FederationDecision",
    "CapitalProfileDecision",
    "TradingModeDecision",
    "RiskAdjustmentDecision",
    "ESSPolicyDecision",
    "StrategyAllocationDecision",
    "SymbolUniverseDecision",
    "CashflowDecision",
    "ResearchTaskDecision",
    "OverrideDecision",
    "FreezeDecision",
    
    # Input Models
    "PortfolioSnapshot",
    "SystemHealthSnapshot",
    "ModelPerformance",
    
    # Enums
    "TradingMode",
    "CapitalProfile",
    "DecisionPriority",
    "DecisionType",
    
    # Roles
    "AICEO",
    "AICIO",
    "AICRO",
    "AICFO",
    "AIResearcher",
    "AISupervisor",
    
    # Adapters
    "PolicyStoreAdapter",
    "PortfolioAdapter",
    "AIEngineAdapter",
    "ESSAdapter",
]
