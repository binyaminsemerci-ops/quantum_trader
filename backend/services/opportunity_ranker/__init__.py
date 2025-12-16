"""
Market Opportunity Ranker for Quantum Trader.

Scores and ranks symbols to identify best trading opportunities.
"""

from .models import SymbolScore, RankingCriteria
from .ranker import MarketOpportunityRanker

# Import from old location for backward compatibility
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    # Try to import OpportunityRanking from old opportunity_ranker.py file
    import backend.services.opportunity_ranker as old_module
    if hasattr(old_module, 'OpportunityRanking'):
        OpportunityRanking = old_module.OpportunityRanking
    else:
        # Fallback stub
        from dataclasses import dataclass
        from datetime import datetime
        from typing import Dict, Any
        
        @dataclass
        class OpportunityRanking:
            symbol: str
            overall_score: float
            rank: int
            metric_scores: Dict[str, float]
            metadata: Dict[str, Any]
            timestamp: datetime
except Exception:
    # Fallback stub
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    
    @dataclass
    class OpportunityRanking:
        symbol: str
        overall_score: float
        rank: int
        metric_scores: Dict[str, float]
        metadata: Dict[str, Any]
        timestamp: datetime

# Alias for backward compatibility
OpportunityRanker = MarketOpportunityRanker

__all__ = [
    "MarketOpportunityRanker",
    "OpportunityRanker",
    "OpportunityRanking",
    "SymbolScore",
    "RankingCriteria",
]
