"""
AI-Researcher (Research Director)
==================================

System evolution and research task generator.

Responsibilities:
- Identify improvement opportunities
- Generate research tasks (hyperparameter tuning, new features, backtests)
- Prioritize research based on impact potential
- Monitor model staleness and drift
- Propose experiments for testing

Decision Logic:
- Model underperformance → hyperparameter research
- Low diversity → new feature research
- Stable system → backtest research
- Market regime change → adaptation research
"""

from typing import List
import structlog

from backend.services.federation_ai.roles.base import FederationRole
from backend.services.federation_ai.models import (
    FederationDecision,
    ResearchTaskDecision,
    ModelPerformance,
    PortfolioSnapshot,
    SystemHealthSnapshot,
    DecisionType,
    DecisionPriority,
)

logger = structlog.get_logger(__name__)


class AIResearcher(FederationRole):
    """AI Researcher - System evolution manager"""
    
    # Research task types
    TASK_TYPES = [
        "hyperparameter_tuning",
        "feature_engineering",
        "backtest_strategy",
        "model_retraining",
        "risk_analysis",
        "market_regime_analysis",
    ]
    
    def __init__(self):
        super().__init__("ai-researcher")
        self.pending_tasks = []
        self.last_research_timestamp = {}
        self.model_staleness_tracker = {}
    
    async def on_model_update(
        self,
        performance: ModelPerformance
    ) -> List[FederationDecision]:
        """
        Evaluate model performance and generate research tasks.
        
        Logic:
        - Low Sharpe → hyperparameter tuning
        - Low win rate → feature engineering
        - Model drift → retraining task
        """
        decisions = []
        
        model_name = performance.model_name
        
        # Track model staleness (days since last training)
        if performance.days_since_training:
            self.model_staleness_tracker[model_name] = performance.days_since_training
        
        # Generate research task if needed
        research_task = self._evaluate_research_need(performance)
        if research_task:
            self.logger.info(
                "Research task generated",
                model=model_name,
                task_type=research_task.payload["task_type"],
                priority=research_task.priority.value,
            )
            decisions.append(research_task)
        
        return decisions
    
    async def on_portfolio_update(
        self,
        snapshot: PortfolioSnapshot
    ) -> List[FederationDecision]:
        """
        Generate strategic research tasks based on portfolio state.
        
        Logic:
        - High DD → risk analysis research
        - Low win rate → strategy backtest
        """
        decisions = []
        
        # High DD - research risk management improvements
        if snapshot.drawdown_pct > 0.06:
            research_task = FederationDecision(
                decision_type=DecisionType.RESEARCH_TASK,
                role_source="researcher",
                priority=DecisionPriority.HIGH,
                reason=f"High DD ({snapshot.drawdown_pct:.1%}), researching risk improvements",
                payload={
                    "task_type": "risk_analysis",
                    "description": "Analyze current risk parameters and ESS thresholds. Identify opportunities to reduce drawdown without sacrificing returns.",
                    "estimated_duration_hours": 4,
                    "expected_impact": "Reduce max DD by 20-30%",
                },
            )
            decisions.append(research_task)
        
        # Low win rate - backtest alternative strategies
        elif snapshot.win_rate_today < 0.45:
            research_task = FederationDecision(
                decision_type=DecisionType.RESEARCH_TASK,
                role_source="researcher",
                priority=DecisionPriority.NORMAL,
                reason=f"Low win rate ({snapshot.win_rate_today:.1%}), backtesting alternatives",
                payload={
                    "task_type": "backtest_strategy",
                    "description": "Backtest alternative signal generation methods. Test different confidence thresholds and ensemble weights.",
                    "estimated_duration_hours": 6,
                    "expected_impact": "Improve win rate by 5-10%",
                },
            )
            decisions.append(research_task)
        
        return decisions
    
    async def on_health_update(
        self,
        health: SystemHealthSnapshot
    ) -> List[FederationDecision]:
        """
        Generate operational research tasks based on system health.
        """
        decisions = []
        
        # System degraded - research performance bottlenecks
        if health.system_status == "DEGRADED":
            research_task = FederationDecision(
                decision_type=DecisionType.RESEARCH_TASK,
                role_source="researcher",
                priority=DecisionPriority.HIGH,
                reason="System degraded, investigating performance issues",
                payload={
                    "task_type": "performance_analysis",
                    "description": "Profile backend services to identify bottlenecks. Analyze API latency, model inference time, database queries.",
                    "estimated_duration_hours": 2,
                    "expected_impact": "Improve system latency by 20-40%",
                },
            )
            decisions.append(research_task)
        
        return decisions
    
    def _evaluate_research_need(self, performance: ModelPerformance) -> FederationDecision:
        """
        Determine if model needs research attention.
        
        Triggers:
        - Sharpe < 1.0 → hyperparameter tuning
        - Win rate < 50% → feature engineering
        - Staleness > 30 days → retraining
        """
        model_name = performance.model_name
        
        # Model underperforming - tune hyperparameters
        if performance.sharpe_ratio and performance.sharpe_ratio < 1.0:
            return FederationDecision(
                decision_type=DecisionType.RESEARCH_TASK,
                role_source="researcher",
                priority=DecisionPriority.HIGH,
                reason=f"Model {model_name} Sharpe {performance.sharpe_ratio:.2f} below target",
                payload={
                    "task_type": "hyperparameter_tuning",
                    "description": f"Tune {model_name} hyperparameters using Optuna. Focus on learning rate, regularization, tree depth.",
                    "estimated_duration_hours": 8,
                    "expected_impact": "Improve Sharpe by 0.3-0.5",
                    "model_name": model_name,
                },
            )
        
        # Low win rate - engineer features
        elif performance.win_rate and performance.win_rate < 0.50:
            return FederationDecision(
                decision_type=DecisionType.RESEARCH_TASK,
                role_source="researcher",
                priority=DecisionPriority.NORMAL,
                reason=f"Model {model_name} win rate {performance.win_rate:.1%} needs improvement",
                payload={
                    "task_type": "feature_engineering",
                    "description": f"Add new features for {model_name}: market regime indicators, volume profiles, order book depth.",
                    "estimated_duration_hours": 12,
                    "expected_impact": "Improve win rate by 5-10%",
                    "model_name": model_name,
                },
            )
        
        # Model stale - retrain
        elif self.model_staleness_tracker.get(model_name, 0) > 30:
            return FederationDecision(
                decision_type=DecisionType.RESEARCH_TASK,
                role_source="researcher",
                priority=DecisionPriority.NORMAL,
                reason=f"Model {model_name} stale ({self.model_staleness_tracker[model_name]} days)",
                payload={
                    "task_type": "model_retraining",
                    "description": f"Retrain {model_name} on recent market data. Include last 90 days of trades.",
                    "estimated_duration_hours": 4,
                    "expected_impact": "Adapt to current market regime",
                    "model_name": model_name,
                },
            )
        
        return None
