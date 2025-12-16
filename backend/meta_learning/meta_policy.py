"""Meta-Policy - High-Level Decision Rules.

Defines when and how to trigger meta-learning actions:
- When to run SESA evolution
- When to retrain RL models
- When to promote shadow strategies to production
- Explore vs Exploit balance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetaDecisionType(Enum):
    """Types of meta-level decisions."""
    
    # SESA decisions
    TRIGGER_SESA_EVOLUTION = "trigger_sesa_evolution"
    SKIP_SESA_EVOLUTION = "skip_sesa_evolution"
    
    # RL training decisions
    TRIGGER_RL_RETRAINING = "trigger_rl_retraining"
    SKIP_RL_RETRAINING = "skip_rl_retraining"
    
    # Promotion decisions
    PROMOTE_SHADOW_TO_PRODUCTION = "promote_shadow_to_production"
    KEEP_SHADOW_TESTING = "keep_shadow_testing"
    REJECT_SHADOW = "reject_shadow"
    
    # Mode decisions
    SWITCH_TO_EXPLORE = "switch_to_explore"
    SWITCH_TO_EXPLOIT = "switch_to_exploit"
    
    # Adjustment decisions
    INCREASE_RISK_LIMITS = "increase_risk_limits"
    DECREASE_RISK_LIMITS = "decrease_risk_limits"
    MAINTAIN_RISK_LIMITS = "maintain_risk_limits"


@dataclass
class MetaDecision:
    """
    A meta-level decision with reasoning.
    """
    
    decision_type: MetaDecisionType
    confidence: float  # 0-1
    reasoning: str
    
    # Parameters for the decision
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision_type": self.decision_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MetaPolicyRules:
    """
    Configuration for meta-policy decision rules.
    """
    
    # SESA Evolution Triggers
    sesa_trigger_interval_hours: float = 24.0
    sesa_trigger_on_performance_drop: bool = True
    sesa_performance_drop_threshold: float = 0.15  # 15% drop
    sesa_min_mutations_per_run: int = 20
    sesa_max_mutations_per_run: int = 50
    
    # RL Retraining Triggers
    rl_retrain_interval_hours: float = 12.0
    rl_retrain_on_regime_change: bool = True
    rl_retrain_min_new_episodes: int = 100
    
    # Shadow→Production Promotion
    shadow_min_test_days: int = 7
    shadow_min_trades: int = 50
    shadow_required_sharpe: float = 1.0
    shadow_max_drawdown: float = 0.12
    shadow_min_win_rate: float = 0.50
    
    # Explore vs Exploit
    exploit_mode_min_performance: float = 0.70  # 70% of target
    explore_mode_max_performance: float = 0.90  # Below 90% → explore more
    exploration_rate_min: float = 0.05
    exploration_rate_max: float = 0.30
    
    # Risk Adjustments
    risk_increase_on_streak_wins: int = 5
    risk_decrease_on_streak_losses: int = 3
    risk_adjustment_step: float = 0.10  # 10% adjustment


class MetaPolicy:
    """
    Meta-Policy for high-level decision making.
    
    Makes strategic decisions about:
    - When to evolve strategies (SESA)
    - When to retrain models (RL)
    - When to promote strategies (Shadow→Production)
    - Explore vs Exploit balance
    - Risk limit adjustments
    
    Usage:
        policy = MetaPolicy(rules=MetaPolicyRules())
        
        # Check if should run SESA
        decision = policy.should_run_sesa_evolution(
            hours_since_last_run=25,
            recent_performance=0.65,
        )
        
        if decision.decision_type == MetaDecisionType.TRIGGER_SESA_EVOLUTION:
            print(f"Trigger SESA: {decision.reasoning}")
            # Run evolution with decision.parameters
    """
    
    def __init__(
        self,
        rules: Optional[MetaPolicyRules] = None,
    ):
        """
        Initialize Meta-Policy.
        
        Args:
            rules: Policy rules (uses defaults if None)
        """
        self.rules = rules or MetaPolicyRules()
        logger.info("MetaPolicy initialized")
    
    def should_run_sesa_evolution(
        self,
        hours_since_last_run: float,
        recent_performance: float,
        recent_performance_baseline: float = 1.0,
    ) -> MetaDecision:
        """
        Decide if SESA evolution should be triggered.
        
        Args:
            hours_since_last_run: Hours since last SESA run
            recent_performance: Recent performance score (0-1)
            recent_performance_baseline: Baseline performance for comparison
        
        Returns:
            MetaDecision
        """
        # Check interval trigger
        interval_trigger = hours_since_last_run >= self.rules.sesa_trigger_interval_hours
        
        # Check performance drop trigger
        performance_drop = (recent_performance_baseline - recent_performance) / recent_performance_baseline
        performance_trigger = (
            self.rules.sesa_trigger_on_performance_drop
            and performance_drop > self.rules.sesa_performance_drop_threshold
        )
        
        # Decision logic
        if interval_trigger or performance_trigger:
            # Calculate num_mutations based on urgency
            if performance_trigger:
                num_mutations = self.rules.sesa_max_mutations_per_run
                reasoning = (
                    f"Performance dropped {performance_drop:.1%} "
                    f"(threshold: {self.rules.sesa_performance_drop_threshold:.1%}). "
                    f"Triggering aggressive evolution with {num_mutations} mutations."
                )
                confidence = 0.85
            else:
                num_mutations = self.rules.sesa_min_mutations_per_run
                reasoning = (
                    f"Scheduled evolution trigger "
                    f"({hours_since_last_run:.1f}h since last run, "
                    f"interval: {self.rules.sesa_trigger_interval_hours:.1f}h). "
                    f"Running standard evolution with {num_mutations} mutations."
                )
                confidence = 0.70
            
            return MetaDecision(
                decision_type=MetaDecisionType.TRIGGER_SESA_EVOLUTION,
                confidence=confidence,
                reasoning=reasoning,
                parameters={
                    "num_mutations": num_mutations,
                    "mutation_magnitude": 0.3 if performance_trigger else 0.2,
                },
            )
        else:
            reasoning = (
                f"No trigger: {hours_since_last_run:.1f}h since last run "
                f"(need {self.rules.sesa_trigger_interval_hours:.1f}h), "
                f"performance drop {performance_drop:.1%} "
                f"(need {self.rules.sesa_performance_drop_threshold:.1%})"
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.SKIP_SESA_EVOLUTION,
                confidence=0.90,
                reasoning=reasoning,
            )
    
    def should_retrain_rl_models(
        self,
        hours_since_last_training: float,
        new_episodes_count: int,
        regime_change_detected: bool = False,
    ) -> MetaDecision:
        """
        Decide if RL models should be retrained.
        
        Args:
            hours_since_last_training: Hours since last training
            new_episodes_count: Number of new episodes collected
            regime_change_detected: Whether market regime changed
        
        Returns:
            MetaDecision
        """
        # Check interval trigger
        interval_trigger = hours_since_last_training >= self.rules.rl_retrain_interval_hours
        
        # Check regime change trigger
        regime_trigger = (
            self.rules.rl_retrain_on_regime_change
            and regime_change_detected
        )
        
        # Check sufficient new data
        data_available = new_episodes_count >= self.rules.rl_retrain_min_new_episodes
        
        # Decision logic
        if (interval_trigger or regime_trigger) and data_available:
            if regime_trigger:
                reasoning = (
                    f"Regime change detected with {new_episodes_count} new episodes. "
                    f"Triggering immediate retraining to adapt."
                )
                confidence = 0.90
            else:
                reasoning = (
                    f"Scheduled retraining trigger "
                    f"({hours_since_last_training:.1f}h since last training, "
                    f"{new_episodes_count} new episodes available)."
                )
                confidence = 0.75
            
            return MetaDecision(
                decision_type=MetaDecisionType.TRIGGER_RL_RETRAINING,
                confidence=confidence,
                reasoning=reasoning,
                parameters={
                    "episodes_count": new_episodes_count,
                    "priority": "high" if regime_trigger else "normal",
                },
            )
        else:
            if not data_available:
                reason = f"insufficient data ({new_episodes_count}/{self.rules.rl_retrain_min_new_episodes})"
            else:
                reason = f"not enough time ({hours_since_last_training:.1f}h/{self.rules.rl_retrain_interval_hours:.1f}h)"
            
            reasoning = f"Skip retraining: {reason}"
            
            return MetaDecision(
                decision_type=MetaDecisionType.SKIP_RL_RETRAINING,
                confidence=0.85,
                reasoning=reasoning,
            )
    
    def should_promote_shadow_strategy(
        self,
        shadow_test_days: int,
        shadow_trades: int,
        shadow_sharpe: float,
        shadow_drawdown: float,
        shadow_win_rate: float,
        production_sharpe: float,
    ) -> MetaDecision:
        """
        Decide if shadow strategy should be promoted to production.
        
        Args:
            shadow_test_days: Days in shadow testing
            shadow_trades: Total trades executed
            shadow_sharpe: Sharpe ratio
            shadow_drawdown: Max drawdown
            shadow_win_rate: Win rate
            production_sharpe: Current production Sharpe (for comparison)
        
        Returns:
            MetaDecision
        """
        # Check minimum requirements
        sufficient_test_period = shadow_test_days >= self.rules.shadow_min_test_days
        sufficient_trades = shadow_trades >= self.rules.shadow_min_trades
        meets_sharpe = shadow_sharpe >= self.rules.shadow_required_sharpe
        meets_drawdown = shadow_drawdown <= self.rules.shadow_max_drawdown
        meets_win_rate = shadow_win_rate >= self.rules.shadow_min_win_rate
        
        # Check if better than production
        better_than_production = shadow_sharpe > production_sharpe * 1.1  # 10% better
        
        # All requirements met?
        all_requirements_met = (
            sufficient_test_period
            and sufficient_trades
            and meets_sharpe
            and meets_drawdown
            and meets_win_rate
        )
        
        # Decision logic
        if all_requirements_met and better_than_production:
            reasoning = (
                f"Shadow strategy ready for promotion: "
                f"{shadow_test_days}d testing, {shadow_trades} trades, "
                f"Sharpe={shadow_sharpe:.2f} (prod={production_sharpe:.2f}), "
                f"DD={shadow_drawdown:.1%}, WR={shadow_win_rate:.1%}"
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.PROMOTE_SHADOW_TO_PRODUCTION,
                confidence=0.85,
                reasoning=reasoning,
            )
        elif all_requirements_met:
            reasoning = (
                f"Shadow meets requirements but not significantly better than production "
                f"(Sharpe: {shadow_sharpe:.2f} vs {production_sharpe:.2f}). "
                f"Continue shadow testing."
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.KEEP_SHADOW_TESTING,
                confidence=0.75,
                reasoning=reasoning,
            )
        elif sufficient_test_period:
            # Failed after sufficient testing → reject
            failures = []
            if not meets_sharpe:
                failures.append(f"Sharpe {shadow_sharpe:.2f} < {self.rules.shadow_required_sharpe}")
            if not meets_drawdown:
                failures.append(f"DD {shadow_drawdown:.1%} > {self.rules.shadow_max_drawdown:.1%}")
            if not meets_win_rate:
                failures.append(f"WR {shadow_win_rate:.1%} < {self.rules.shadow_min_win_rate:.1%}")
            
            reasoning = f"Reject shadow after {shadow_test_days}d: " + ", ".join(failures)
            
            return MetaDecision(
                decision_type=MetaDecisionType.REJECT_SHADOW,
                confidence=0.80,
                reasoning=reasoning,
            )
        else:
            reasoning = (
                f"Continue shadow testing: {shadow_test_days}/{self.rules.shadow_min_test_days}d, "
                f"{shadow_trades}/{self.rules.shadow_min_trades} trades"
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.KEEP_SHADOW_TESTING,
                confidence=0.90,
                reasoning=reasoning,
            )
    
    def determine_explore_exploit_mode(
        self,
        current_performance: float,
        target_performance: float = 1.0,
    ) -> MetaDecision:
        """
        Determine whether to explore (try new strategies) or exploit (use best known).
        
        Args:
            current_performance: Current performance (0-1)
            target_performance: Target performance (default 1.0 = 100%)
        
        Returns:
            MetaDecision
        """
        performance_ratio = current_performance / target_performance
        
        if performance_ratio < self.rules.exploit_mode_min_performance:
            # Very poor performance → explore aggressively
            exploration_rate = self.rules.exploration_rate_max
            reasoning = (
                f"Performance {current_performance:.1%} well below target "
                f"({self.rules.exploit_mode_min_performance:.1%}). "
                f"Switch to EXPLORE mode (rate={exploration_rate:.1%})."
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.SWITCH_TO_EXPLORE,
                confidence=0.85,
                reasoning=reasoning,
                parameters={"exploration_rate": exploration_rate},
            )
        elif performance_ratio > self.rules.explore_mode_max_performance:
            # Good performance → exploit more
            exploration_rate = self.rules.exploration_rate_min
            reasoning = (
                f"Performance {current_performance:.1%} above "
                f"{self.rules.explore_mode_max_performance:.1%} threshold. "
                f"Switch to EXPLOIT mode (rate={exploration_rate:.1%})."
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.SWITCH_TO_EXPLOIT,
                confidence=0.80,
                reasoning=reasoning,
                parameters={"exploration_rate": exploration_rate},
            )
        else:
            # Middle ground → balanced exploration
            exploration_rate = (self.rules.exploration_rate_min + self.rules.exploration_rate_max) / 2
            reasoning = (
                f"Performance {current_performance:.1%} in normal range. "
                f"Maintain balanced mode (rate={exploration_rate:.1%})."
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.SWITCH_TO_EXPLOIT,
                confidence=0.70,
                reasoning=reasoning,
                parameters={"exploration_rate": exploration_rate},
            )
    
    def adjust_risk_limits(
        self,
        recent_wins: int,
        recent_losses: int,
        current_risk_multiplier: float = 1.0,
    ) -> MetaDecision:
        """
        Decide if risk limits should be adjusted based on recent performance.
        
        Args:
            recent_wins: Number of recent winning trades
            recent_losses: Number of recent losing trades
            current_risk_multiplier: Current risk multiplier (1.0 = baseline)
        
        Returns:
            MetaDecision
        """
        # Check winning streak
        winning_streak = recent_wins >= self.rules.risk_increase_on_streak_wins
        
        # Check losing streak
        losing_streak = recent_losses >= self.rules.risk_decrease_on_streak_losses
        
        if winning_streak and current_risk_multiplier < 1.5:
            new_multiplier = min(
                current_risk_multiplier + self.rules.risk_adjustment_step,
                1.5  # Cap at 1.5x
            )
            
            reasoning = (
                f"Winning streak detected ({recent_wins} wins). "
                f"Increase risk multiplier: {current_risk_multiplier:.2f} → {new_multiplier:.2f}"
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.INCREASE_RISK_LIMITS,
                confidence=0.75,
                reasoning=reasoning,
                parameters={"new_risk_multiplier": new_multiplier},
            )
        elif losing_streak and current_risk_multiplier > 0.5:
            new_multiplier = max(
                current_risk_multiplier - self.rules.risk_adjustment_step,
                0.5  # Floor at 0.5x
            )
            
            reasoning = (
                f"Losing streak detected ({recent_losses} losses). "
                f"Decrease risk multiplier: {current_risk_multiplier:.2f} → {new_multiplier:.2f}"
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.DECREASE_RISK_LIMITS,
                confidence=0.85,
                reasoning=reasoning,
                parameters={"new_risk_multiplier": new_multiplier},
            )
        else:
            reasoning = (
                f"No risk adjustment needed "
                f"(wins: {recent_wins}, losses: {recent_losses}, "
                f"current multiplier: {current_risk_multiplier:.2f})"
            )
            
            return MetaDecision(
                decision_type=MetaDecisionType.MAINTAIN_RISK_LIMITS,
                confidence=0.90,
                reasoning=reasoning,
                parameters={"current_risk_multiplier": current_risk_multiplier},
            )
