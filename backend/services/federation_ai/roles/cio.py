"""
AI-CIO (Chief Investment Officer)
==================================

Strategy allocation and symbol universe manager.

Responsibilities:
- Allocate capital across AI models (XGB, LightGBM, NHITS, PatchTST)
- Select active trading symbols from universe
- Balance diversification vs concentration
- Respond to model performance drift
- Coordinate with AI-CEO on capital profile

Decision Logic:
- Model weighting: Recent Sharpe ratio, win rate, profit factor
- Symbol selection: Liquidity, volatility, correlation
- Dynamic rebalancing based on performance
"""

from typing import List
import structlog

from backend.services.federation_ai.roles.base import FederationRole
from backend.services.federation_ai.models import (
    FederationDecision,
    StrategyAllocationDecision,
    SymbolUniverseDecision,
    ModelPerformance,
    PortfolioSnapshot,
    DecisionType,
    DecisionPriority,
)

logger = structlog.get_logger(__name__)


class AICIO(FederationRole):
    """AI Chief Investment Officer - Strategy allocator"""
    
    # Available AI models
    AVAILABLE_MODELS = ["xgboost", "lightgbm", "nhits", "patchtst"]
    
    # Base equal weighting
    BASE_WEIGHTS = {
        "xgboost": 0.25,
        "lightgbm": 0.25,
        "nhits": 0.25,
        "patchtst": 0.25,
    }
    
    # Symbol universe (example - should be dynamic)
    SYMBOL_UNIVERSE = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "ADAUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT",
        "LINKUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
    ]
    
    def __init__(self):
        super().__init__("ai-cio")
        self.current_weights = self.BASE_WEIGHTS.copy()
        self.active_symbols = self.SYMBOL_UNIVERSE[:5]  # Start with top 5
        self.excluded_symbols = []
        self.model_performance_history = {}
    
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Evaluate symbol universe based on portfolio state.
        
        Logic:
        - High profits → expand symbols (more opportunities)
        - High DD → contract symbols (focus on best)
        - Low win rate → rotate symbols
        """
        decisions = []
        
        # Expand/contract symbol universe
        universe_decision = self._evaluate_symbol_universe(snapshot)
        if universe_decision:
            self.logger.info(
                "Symbol universe adjustment",
                num_active=len(universe_decision.payload["active_symbols"]),
                num_excluded=len(universe_decision.payload.get("excluded_symbols", [])),
            )
            decisions.append(universe_decision)
        
        return decisions
    
    async def on_model_update(
        self,
        performance: ModelPerformance
    ) -> List[FederationDecision]:
        """
        Rebalance model weights based on recent performance.
        
        Logic:
        - High Sharpe → increase weight
        - Low win rate → decrease weight
        - Consistent losses → disable model
        """
        decisions = []
        
        # Store performance
        self.model_performance_history[performance.model_name] = performance
        
        # Rebalance if we have data for all models
        if len(self.model_performance_history) >= len(self.AVAILABLE_MODELS):
            allocation_decision = self._rebalance_models()
            if allocation_decision:
                self.logger.info(
                    "Model rebalancing",
                    new_weights=allocation_decision.payload["model_weights"],
                )
                decisions.append(allocation_decision)
        
        return decisions
    
    def _evaluate_symbol_universe(self, snapshot: PortfolioSnapshot) -> FederationDecision:
        """
        Adjust active trading symbols based on portfolio state.
        
        Expansion: DD < 2%, win rate > 60%, capital > baseline
        Contraction: DD > 5%, win rate < 45%
        """
        dd_pct = snapshot.drawdown_pct
        win_rate = snapshot.win_rate_today
        num_positions = snapshot.num_positions
        
        # Contract universe if struggling
        if dd_pct > 0.05 or win_rate < 0.45:
            # Focus on top 3 symbols
            new_active = self.SYMBOL_UNIVERSE[:3]
            
            if set(new_active) != set(self.active_symbols):
                self.active_symbols = new_active
                
                return FederationDecision(
                    decision_type=DecisionType.SYMBOL_UNIVERSE,
                    role_source="cio",
                    priority=DecisionPriority.HIGH,
                    reason=f"High DD ({dd_pct:.1%}) or low WR ({win_rate:.1%}), contracting to top 3 symbols",
                    payload={
                        "active_symbols": new_active,
                        "excluded_symbols": [s for s in self.SYMBOL_UNIVERSE if s not in new_active],
                    },
                )
        
        # Expand universe if performing well
        elif dd_pct < 0.02 and win_rate > 0.60 and num_positions < 8:
            # Expand to 8 symbols
            new_active = self.SYMBOL_UNIVERSE[:8]
            
            if len(new_active) > len(self.active_symbols):
                self.active_symbols = new_active
                
                return FederationDecision(
                    decision_type=DecisionType.SYMBOL_UNIVERSE,
                    role_source="cio",
                    priority=DecisionPriority.NORMAL,
                    reason=f"Strong performance (DD {dd_pct:.1%}, WR {win_rate:.1%}), expanding to 8 symbols",
                    payload={
                        "active_symbols": new_active,
                        "excluded_symbols": [s for s in self.SYMBOL_UNIVERSE if s not in new_active],
                    },
                )
        
        return None
    
    def _rebalance_models(self) -> FederationDecision:
        """
        Rebalance model weights based on recent performance.
        
        Scoring:
        - Sharpe ratio (primary)
        - Win rate (secondary)
        - Profit factor (tie-breaker)
        """
        # Calculate scores for each model
        scores = {}
        for model_name, perf in self.model_performance_history.items():
            # Base score on Sharpe ratio
            sharpe_score = perf.sharpe_ratio if perf.sharpe_ratio else 0.0
            
            # Bonus for high win rate
            win_rate_bonus = (perf.win_rate - 0.5) * 2.0 if perf.win_rate else 0.0
            
            # Total score
            scores[model_name] = max(0.1, sharpe_score + win_rate_bonus)  # Minimum 10%
        
        # Normalize to weights
        total_score = sum(scores.values())
        new_weights = {
            model: score / total_score
            for model, score in scores.items()
        }
        
        # Check if weights changed significantly (>5% for any model)
        weight_changed = any(
            abs(new_weights.get(model, 0) - self.current_weights.get(model, 0)) > 0.05
            for model in self.AVAILABLE_MODELS
        )
        
        if weight_changed:
            self.current_weights = new_weights
            
            # Determine active strategies (weight > 10%)
            active_strategies = [
                model for model, weight in new_weights.items()
                if weight >= 0.10
            ]
            
            return FederationDecision(
                decision_type=DecisionType.STRATEGY_ALLOCATION,
                role_source="cio",
                priority=DecisionPriority.NORMAL,
                reason=f"Model performance drift detected, rebalancing",
                payload={
                    "model_weights": new_weights,
                    "active_strategies": active_strategies,
                },
            )
        
        return None
