"""Selection Engine - Top Performer Selection and Promotion.

Selects best-performing strategies from evaluation results and flags them
for shadow testing or production promotion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np

from backend.sesa.evaluation_engine import EvaluationResult, PerformanceMetrics

logger = logging.getLogger(__name__)


class CandidateStatus(Enum):
    """Status for evaluated strategies."""
    
    REJECTED = "rejected"  # Did not meet minimum requirements
    SHADOW_CANDIDATE = "shadow_candidate"  # Passed, ready for shadow testing
    PRODUCTION_CANDIDATE = "production_candidate"  # Exceptional, ready for production


@dataclass
class SelectionCriteria:
    """
    Criteria for selecting top performers.
    """
    
    # Selection parameters
    top_k: int = 5  # Select top K strategies
    min_composite_score: float = 60.0  # Minimum composite score
    max_risk_score: float = 60.0  # Maximum risk score
    
    # Shadow vs Production thresholds
    shadow_score_threshold: float = 65.0  # Score for shadow candidate
    production_score_threshold: float = 80.0  # Score for production candidate
    
    # Performance gates
    min_sharpe: float = 0.5
    min_win_rate: float = 0.48
    min_profit_factor: float = 1.2
    max_drawdown: float = 0.20
    
    # Diversity settings
    enforce_diversity: bool = True
    min_mutation_distance: float = 0.3  # Minimum difference between selected strategies


@dataclass
class SelectedStrategy:
    """
    A selected strategy with status and reasoning.
    """
    
    strategy_id: str
    mutation_id: Optional[str]
    
    # Selection metadata
    rank: int  # 1 = best
    status: CandidateStatus
    composite_score: float
    risk_score: float
    
    # Performance summary
    metrics: PerformanceMetrics
    
    # Reasoning
    selection_reason: str
    promotion_readiness: float  # 0-1 score for production readiness
    
    # Timestamp
    selected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "mutation_id": self.mutation_id,
            "rank": self.rank,
            "status": self.status.value,
            "composite_score": self.composite_score,
            "risk_score": self.risk_score,
            "metrics": self.metrics.to_dict(),
            "selection_reason": self.selection_reason,
            "promotion_readiness": self.promotion_readiness,
            "selected_at": self.selected_at.isoformat(),
        }


@dataclass
class SelectionResult:
    """
    Complete selection result.
    """
    
    # Selected strategies
    selected: list[SelectedStrategy]
    
    # Rejected strategies (for analysis)
    rejected: list[str] = field(default_factory=list)
    rejected_reasons: dict[str, str] = field(default_factory=dict)
    
    # Selection metadata
    total_evaluated: int = 0
    selection_timestamp: datetime = field(default_factory=datetime.now)
    selection_criteria: Optional[SelectionCriteria] = None
    
    # Statistics
    avg_score_selected: float = 0.0
    avg_risk_selected: float = 0.0
    diversity_score: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "selected": [s.to_dict() for s in self.selected],
            "rejected": self.rejected,
            "rejected_reasons": self.rejected_reasons,
            "total_evaluated": self.total_evaluated,
            "selection_timestamp": self.selection_timestamp.isoformat(),
            "avg_score_selected": self.avg_score_selected,
            "avg_risk_selected": self.avg_risk_selected,
            "diversity_score": self.diversity_score,
        }


class SelectionEngine:
    """
    Selection Engine for identifying top performers.
    
    Features:
    - Rank strategies by composite score
    - Apply multiple filtering criteria
    - Enforce diversity in selected strategies
    - Flag strategies for shadow or production deployment
    - Provide detailed selection reasoning
    
    Usage:
        engine = SelectionEngine(
            criteria=SelectionCriteria(top_k=5, min_composite_score=60)
        )
        
        result = engine.select_top_performers(evaluation_results)
        
        for strategy in result.selected:
            print(f"Rank {strategy.rank}: {strategy.strategy_id}")
            print(f"  Status: {strategy.status.value}")
            print(f"  Score: {strategy.composite_score:.2f}")
            print(f"  Reason: {strategy.selection_reason}")
    """
    
    def __init__(
        self,
        criteria: Optional[SelectionCriteria] = None,
    ):
        """
        Initialize Selection Engine.
        
        Args:
            criteria: Selection criteria (uses defaults if None)
        """
        self.criteria = criteria or SelectionCriteria()
        logger.info(f"SelectionEngine initialized with top_k={self.criteria.top_k}")
    
    def select_top_performers(
        self,
        evaluation_results: list[EvaluationResult],
    ) -> SelectionResult:
        """
        Select top performing strategies from evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
        
        Returns:
            SelectionResult with ranked selections
        """
        logger.info(f"Selecting top performers from {len(evaluation_results)} evaluations")
        
        # Filter: Only strategies that passed minimum requirements
        passed = [
            r for r in evaluation_results
            if r.passed_minimum_requirements
        ]
        
        rejected = [
            r.strategy_id for r in evaluation_results
            if not r.passed_minimum_requirements
        ]
        rejected_reasons = {
            r.strategy_id: r.reason_if_failed or "Unknown"
            for r in evaluation_results
            if not r.passed_minimum_requirements
        }
        
        logger.info(f"Passed minimum requirements: {len(passed)}/{len(evaluation_results)}")
        
        if not passed:
            logger.warning("No strategies passed minimum requirements")
            return SelectionResult(
                selected=[],
                rejected=rejected,
                rejected_reasons=rejected_reasons,
                total_evaluated=len(evaluation_results),
                selection_criteria=self.criteria,
            )
        
        # Apply additional filters
        filtered = self._apply_filters(passed)
        
        logger.info(f"After additional filters: {len(filtered)}/{len(passed)}")
        
        # Rank by composite score
        ranked = sorted(
            filtered,
            key=lambda r: r.composite_score,
            reverse=True,
        )
        
        # Select top K
        top_k = ranked[:self.criteria.top_k]
        
        # Enforce diversity if enabled
        if self.criteria.enforce_diversity and len(top_k) > 1:
            top_k = self._enforce_diversity(top_k)
        
        # Create SelectedStrategy objects
        selected_strategies: list[SelectedStrategy] = []
        
        for rank, result in enumerate(top_k, start=1):
            status = self._determine_status(result)
            promotion_readiness = self._calculate_promotion_readiness(result)
            reason = self._generate_selection_reason(result, rank, status)
            
            selected = SelectedStrategy(
                strategy_id=result.strategy_id,
                mutation_id=result.mutation_id,
                rank=rank,
                status=status,
                composite_score=result.composite_score,
                risk_score=result.risk_score,
                metrics=result.metrics,
                selection_reason=reason,
                promotion_readiness=promotion_readiness,
            )
            
            selected_strategies.append(selected)
        
        # Calculate statistics
        avg_score = float(np.mean([s.composite_score for s in selected_strategies])) if selected_strategies else 0
        avg_risk = float(np.mean([s.risk_score for s in selected_strategies])) if selected_strategies else 0
        diversity = self._calculate_diversity_score(selected_strategies)
        
        result = SelectionResult(
            selected=selected_strategies,
            rejected=rejected,
            rejected_reasons=rejected_reasons,
            total_evaluated=len(evaluation_results),
            selection_criteria=self.criteria,
            avg_score_selected=avg_score,
            avg_risk_selected=avg_risk,
            diversity_score=diversity,
        )
        
        logger.info(
            f"Selection complete: {len(selected_strategies)} selected, "
            f"avg_score={avg_score:.2f}, avg_risk={avg_risk:.2f}"
        )
        
        return result
    
    def _apply_filters(
        self,
        results: list[EvaluationResult],
    ) -> list[EvaluationResult]:
        """
        Apply additional filtering criteria.
        
        Filters:
        - Minimum composite score
        - Maximum risk score
        - Performance gates (Sharpe, win rate, profit factor, drawdown)
        """
        filtered: list[EvaluationResult] = []
        
        for result in results:
            # Composite score filter
            if result.composite_score < self.criteria.min_composite_score:
                continue
            
            # Risk score filter
            if result.risk_score > self.criteria.max_risk_score:
                continue
            
            # Performance gates
            metrics = result.metrics
            
            if metrics.sharpe_ratio < self.criteria.min_sharpe:
                continue
            
            if metrics.win_rate < self.criteria.min_win_rate:
                continue
            
            if metrics.profit_factor < self.criteria.min_profit_factor:
                continue
            
            if metrics.max_drawdown_pct > self.criteria.max_drawdown:
                continue
            
            # Passed all filters
            filtered.append(result)
        
        return filtered
    
    def _enforce_diversity(
        self,
        results: list[EvaluationResult],
    ) -> list[EvaluationResult]:
        """
        Enforce diversity in selected strategies.
        
        Removes strategies that are too similar to higher-ranked ones.
        """
        if len(results) <= 1:
            return results
        
        diverse: list[EvaluationResult] = [results[0]]  # Always keep best
        
        for candidate in results[1:]:
            # Check distance to all already-selected strategies
            is_diverse = True
            
            for selected in diverse:
                distance = self._calculate_strategy_distance(candidate, selected)
                
                if distance < self.criteria.min_mutation_distance:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse.append(candidate)
        
        logger.info(f"Diversity enforcement: {len(results)} -> {len(diverse)}")
        
        return diverse
    
    def _calculate_strategy_distance(
        self,
        result1: EvaluationResult,
        result2: EvaluationResult,
    ) -> float:
        """
        Calculate distance between two strategies.
        
        Uses normalized metric differences.
        """
        m1 = result1.metrics
        m2 = result2.metrics
        
        # Normalized differences
        sharpe_diff = abs(m1.sharpe_ratio - m2.sharpe_ratio) / 5.0
        winrate_diff = abs(m1.win_rate - m2.win_rate)
        pf_diff = abs(m1.profit_factor - m2.profit_factor) / 5.0
        dd_diff = abs(m1.max_drawdown_pct - m2.max_drawdown_pct) / 0.5
        
        # Euclidean distance
        distance = np.sqrt(
            sharpe_diff**2 + winrate_diff**2 + pf_diff**2 + dd_diff**2
        )
        
        return float(distance)
    
    def _determine_status(
        self,
        result: EvaluationResult,
    ) -> CandidateStatus:
        """
        Determine candidate status (shadow vs production).
        """
        score = result.composite_score
        
        if score >= self.criteria.production_score_threshold:
            # Exceptional performance - production candidate
            return CandidateStatus.PRODUCTION_CANDIDATE
        elif score >= self.criteria.shadow_score_threshold:
            # Good performance - shadow candidate
            return CandidateStatus.SHADOW_CANDIDATE
        else:
            # Passed minimum but not exceptional
            return CandidateStatus.SHADOW_CANDIDATE
    
    def _calculate_promotion_readiness(
        self,
        result: EvaluationResult,
    ) -> float:
        """
        Calculate readiness score (0-1) for production promotion.
        
        Factors:
        - Composite score
        - Risk score (inverted)
        - Consistency metrics
        - Data completeness
        """
        # Composite score component (40%)
        score_component = (result.composite_score / 100.0) * 0.4
        
        # Risk component (30%)
        risk_component = (1.0 - result.risk_score / 100.0) * 0.3
        
        # Performance consistency (20%)
        metrics = result.metrics
        consistency = 1.0
        
        # Penalize high volatility
        if metrics.volatility > 0.05:
            consistency *= 0.7
        elif metrics.volatility > 0.03:
            consistency *= 0.85
        
        # Penalize low win rate
        if metrics.win_rate < 0.50:
            consistency *= 0.8
        
        consistency_component = consistency * 0.2
        
        # Data completeness (10%)
        data_component = result.data_completeness * 0.1
        
        readiness = score_component + risk_component + consistency_component + data_component
        
        return float(np.clip(readiness, 0.0, 1.0))
    
    def _generate_selection_reason(
        self,
        result: EvaluationResult,
        rank: int,
        status: CandidateStatus,
    ) -> str:
        """
        Generate human-readable selection reasoning.
        """
        metrics = result.metrics
        
        # Build reason string
        parts = [
            f"Rank #{rank}",
            f"Score: {result.composite_score:.1f}",
        ]
        
        # Highlight strengths
        strengths = []
        
        if metrics.sharpe_ratio > 1.5:
            strengths.append(f"excellent Sharpe ({metrics.sharpe_ratio:.2f})")
        elif metrics.sharpe_ratio > 1.0:
            strengths.append(f"strong Sharpe ({metrics.sharpe_ratio:.2f})")
        
        if metrics.win_rate > 0.55:
            strengths.append(f"high win rate ({metrics.win_rate:.1%})")
        
        if metrics.profit_factor > 2.0:
            strengths.append(f"strong edge (PF={metrics.profit_factor:.2f})")
        
        if metrics.max_drawdown_pct < 0.10:
            strengths.append(f"low drawdown ({metrics.max_drawdown_pct:.1%})")
        
        if strengths:
            parts.append(f"Strengths: {', '.join(strengths)}")
        
        # Status explanation
        if status == CandidateStatus.PRODUCTION_CANDIDATE:
            parts.append("Ready for production promotion")
        elif status == CandidateStatus.SHADOW_CANDIDATE:
            parts.append("Recommended for shadow testing")
        
        return ". ".join(parts) + "."
    
    def _calculate_diversity_score(
        self,
        selected: list[SelectedStrategy],
    ) -> float:
        """
        Calculate diversity score for selected strategies.
        
        Higher score = more diverse.
        """
        if len(selected) <= 1:
            return 1.0
        
        # Calculate pairwise distances
        distances: list[float] = []
        
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                m1 = selected[i].metrics
                m2 = selected[j].metrics
                
                # Metric differences
                sharpe_diff = abs(m1.sharpe_ratio - m2.sharpe_ratio) / 5.0
                winrate_diff = abs(m1.win_rate - m2.win_rate)
                pf_diff = abs(m1.profit_factor - m2.profit_factor) / 5.0
                
                distance = np.sqrt(sharpe_diff**2 + winrate_diff**2 + pf_diff**2)
                distances.append(distance)
        
        # Average distance normalized
        avg_distance = float(np.mean(distances)) if distances else 0
        diversity = np.clip(avg_distance / 2.0, 0.0, 1.0)
        
        return float(diversity)
