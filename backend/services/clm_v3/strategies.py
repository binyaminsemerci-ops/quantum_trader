"""
CLM v3 Strategy Evolution Engine - Strategy candidate generation and testing.

Provides:
- StrategyCandidate model
- StrategyEvolutionEngine (skeleton for genetic/evolutionary algorithms)
- Performance-based strategy mutation
- Integration with CLM v3 for training new strategy variants

This is a SKELETON for future Strategy Evolution work.
Not a full genetic algorithm implementation yet.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from backend.services.clm_v3.models import ModelType, TriggerReason

logger = logging.getLogger(__name__)


# ============================================================================
# Enums & Models
# ============================================================================

class StrategyOrigin(str, Enum):
    """How strategy candidate was created."""
    
    MANUAL = "manual"              # Manually created by user
    MUTATION = "mutation"          # Mutated from existing strategy
    CROSSOVER = "crossover"        # Combined from two parent strategies
    RANDOM = "random"              # Randomly generated
    REGIME_ADAPTATION = "regime_adaptation"  # Adapted for new market regime


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""
    
    PROPOSED = "proposed"          # Generated but not trained
    TRAINING = "training"          # Currently training
    SHADOW = "shadow"              # Shadow testing (0% allocation)
    ACTIVE = "active"              # Active in production
    RETIRED = "retired"            # Decommissioned
    FAILED = "failed"              # Failed evaluation


class StrategyCandidate(BaseModel):
    """
    Strategy candidate for testing.
    
    Represents a strategy variant (parameter set) to be trained and evaluated.
    """
    
    id: UUID = Field(default_factory=uuid4)
    base_strategy: str  # e.g., "trend_following", "mean_reversion", "momentum"
    model_type: ModelType  # Which model type to use
    
    # Parameters (strategy-specific)
    params: Dict[str, Any]  # e.g., {"lookback": 20, "threshold": 0.02, "stop_loss": 0.03}
    
    # Origin
    origin: StrategyOrigin
    parent_ids: List[UUID] = Field(default_factory=list)  # Parent strategy IDs (for mutation/crossover)
    mutation_description: Optional[str] = None
    
    # Status
    status: StrategyStatus = StrategyStatus.PROPOSED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Performance (after evaluation)
    performance_metrics: Optional[Dict[str, float]] = None
    fitness_score: Optional[float] = None  # For genetic algorithm ranking
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# ============================================================================
# Strategy Evolution Engine
# ============================================================================

class StrategyEvolutionEngine:
    """
    Strategy Evolution Engine - Generates and tests strategy variants.
    
    SKELETON IMPLEMENTATION for EPIC-CLM3-001.
    
    Future enhancements (EPIC-CLM3-002+):
    - Genetic algorithm (selection, crossover, mutation)
    - Multi-objective optimization (Sharpe vs MDD vs WR)
    - Regime-aware strategy selection
    - Automatic parameter tuning
    - Ensemble strategy generation
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
    ):
        """
        Initialize Strategy Evolution Engine.
        
        Args:
            config: Evolution engine configuration
        """
        self.config = config or self._default_config()
        
        # State
        self.candidates: Dict[UUID, StrategyCandidate] = {}
        self.performance_history: List[Dict] = []
        
        logger.info("[CLM v3 Evolution] StrategyEvolutionEngine initialized (SKELETON)")
    
    @staticmethod
    def _default_config() -> Dict:
        """Default evolution configuration."""
        return {
            "enabled": False,  # Disabled by default (skeleton)
            
            # When to generate new candidates
            "trigger_on_poor_performance": True,
            "poor_performance_threshold_sharpe": 0.5,
            
            # Mutation parameters
            "mutation_rate": 0.1,  # 10% of params mutated
            "mutation_magnitude": 0.2,  # ±20% change
            
            # Selection parameters
            "max_active_candidates": 5,
            "min_fitness_for_promotion": 0.7,
        }
    
    # ========================================================================
    # Candidate Generation
    # ========================================================================
    
    async def propose_new_candidates(
        self,
        performance_data: Dict,
        base_strategy: str = "trend_following",
        model_type: ModelType = ModelType.XGBOOST,
    ) -> List[StrategyCandidate]:
        """
        Propose new strategy candidates based on current performance.
        
        Logic (SKELETON):
        1. Analyze performance_data
        2. If performance is poor:
           - Identify weak parameters
           - Generate parameter variants (mutations)
        3. Return list of new candidates
        
        Args:
            performance_data: Recent performance metrics
            base_strategy: Base strategy to mutate
            model_type: Model type to use
        
        Returns:
            List of StrategyCandidate to be trained
        """
        if not self.config["enabled"]:
            logger.info("[CLM v3 Evolution] Evolution disabled - no candidates generated")
            return []
        
        logger.info(
            f"[CLM v3 Evolution] Proposing new candidates "
            f"(base={base_strategy}, model={model_type.value})"
        )
        
        # Check if we should generate candidates
        sharpe = performance_data.get("sharpe_ratio", 1.0)
        threshold = self.config["poor_performance_threshold_sharpe"]
        
        if sharpe >= threshold:
            logger.info(
                f"[CLM v3 Evolution] Performance acceptable (Sharpe={sharpe:.3f} >= {threshold}) "
                f"- no new candidates needed"
            )
            return []
        
        logger.warning(
            f"[CLM v3 Evolution] Poor performance detected (Sharpe={sharpe:.3f} < {threshold}) "
            f"- generating strategy variants"
        )
        
        # Generate candidates
        candidates = []
        
        # Candidate 1: Increase lookback period
        candidate1 = StrategyCandidate(
            base_strategy=base_strategy,
            model_type=model_type,
            params={
                "lookback": 30,  # Increased from default 20
                "threshold": 0.02,
                "stop_loss": 0.03,
            },
            origin=StrategyOrigin.MUTATION,
            mutation_description="Increased lookback period to 30 (was 20)",
        )
        candidates.append(candidate1)
        self.candidates[candidate1.id] = candidate1
        
        # Candidate 2: Tighter stop loss
        candidate2 = StrategyCandidate(
            base_strategy=base_strategy,
            model_type=model_type,
            params={
                "lookback": 20,
                "threshold": 0.02,
                "stop_loss": 0.02,  # Tighter (was 0.03)
            },
            origin=StrategyOrigin.MUTATION,
            mutation_description="Tightened stop loss to 2% (was 3%)",
        )
        candidates.append(candidate2)
        self.candidates[candidate2.id] = candidate2
        
        # Candidate 3: Lower threshold
        candidate3 = StrategyCandidate(
            base_strategy=base_strategy,
            model_type=model_type,
            params={
                "lookback": 20,
                "threshold": 0.015,  # Lower (was 0.02)
                "stop_loss": 0.03,
            },
            origin=StrategyOrigin.MUTATION,
            mutation_description="Lowered threshold to 1.5% (was 2%)",
        )
        candidates.append(candidate3)
        self.candidates[candidate3.id] = candidate3
        
        logger.info(
            f"[CLM v3 Evolution] Generated {len(candidates)} strategy candidates "
            f"(poor performance response)"
        )
        
        return candidates
    
    async def mutate_strategy(
        self,
        parent: StrategyCandidate,
        mutation_rate: Optional[float] = None,
    ) -> StrategyCandidate:
        """
        Mutate a strategy (change parameters randomly).
        
        SKELETON: Simple random parameter changes.
        Future: Smarter mutations based on gradient/performance.
        
        Args:
            parent: Parent strategy to mutate
            mutation_rate: Probability of mutating each parameter
        
        Returns:
            New StrategyCandidate (mutated)
        """
        mutation_rate = mutation_rate or self.config["mutation_rate"]
        magnitude = self.config["mutation_magnitude"]
        
        # Copy parent params
        new_params = parent.params.copy()
        
        # Mutate numeric params
        mutations = []
        for key, value in new_params.items():
            if isinstance(value, (int, float)):
                # Random mutation
                import random
                if random.random() < mutation_rate:
                    # ±magnitude change
                    change = random.uniform(-magnitude, magnitude)
                    new_value = value * (1 + change)
                    new_params[key] = new_value
                    mutations.append(f"{key}: {value} → {new_value:.4f}")
        
        # Create mutated candidate
        mutated = StrategyCandidate(
            base_strategy=parent.base_strategy,
            model_type=parent.model_type,
            params=new_params,
            origin=StrategyOrigin.MUTATION,
            parent_ids=[parent.id],
            mutation_description=f"Mutated: {', '.join(mutations)}",
        )
        
        self.candidates[mutated.id] = mutated
        
        logger.info(
            f"[CLM v3 Evolution] Mutated strategy {parent.id} → {mutated.id} "
            f"({len(mutations)} params changed)"
        )
        
        return mutated
    
    async def crossover_strategies(
        self,
        parent1: StrategyCandidate,
        parent2: StrategyCandidate,
    ) -> StrategyCandidate:
        """
        Crossover two strategies (combine parameters).
        
        SKELETON: Simple parameter averaging.
        Future: Smart crossover based on performance contribution.
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
        
        Returns:
            New StrategyCandidate (crossover)
        """
        if parent1.base_strategy != parent2.base_strategy:
            logger.warning(
                f"[CLM v3 Evolution] Cannot crossover different strategies "
                f"({parent1.base_strategy} vs {parent2.base_strategy})"
            )
            return parent1  # Return parent1 unchanged
        
        # Average numeric parameters
        new_params = {}
        for key in parent1.params.keys():
            val1 = parent1.params.get(key)
            val2 = parent2.params.get(key)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Average
                new_params[key] = (val1 + val2) / 2
            else:
                # Non-numeric - pick from parent1
                new_params[key] = val1
        
        # Create crossover candidate
        child = StrategyCandidate(
            base_strategy=parent1.base_strategy,
            model_type=parent1.model_type,
            params=new_params,
            origin=StrategyOrigin.CROSSOVER,
            parent_ids=[parent1.id, parent2.id],
            mutation_description=f"Crossover: {parent1.id} × {parent2.id}",
        )
        
        self.candidates[child.id] = child
        
        logger.info(
            f"[CLM v3 Evolution] Crossover: {parent1.id} × {parent2.id} → {child.id}"
        )
        
        return child
    
    # ========================================================================
    # Evaluation & Selection
    # ========================================================================
    
    def update_candidate_performance(
        self,
        candidate_id: UUID,
        metrics: Dict[str, float],
    ):
        """
        Update candidate performance after evaluation.
        
        Args:
            candidate_id: Candidate UUID
            metrics: Performance metrics (Sharpe, WR, PF, etc.)
        """
        if candidate_id not in self.candidates:
            logger.warning(f"[CLM v3 Evolution] Candidate {candidate_id} not found")
            return
        
        candidate = self.candidates[candidate_id]
        candidate.performance_metrics = metrics
        
        # Calculate fitness score (0-1, higher is better)
        # Simple weighted sum of metrics
        sharpe = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0.5)
        profit_factor = metrics.get("profit_factor", 1.0)
        
        fitness = (
            sharpe / 3.0 * 0.5 +       # Sharpe (target 3.0)
            win_rate * 0.3 +            # Win rate (0-1)
            (profit_factor - 1) / 2 * 0.2  # Profit factor (target 3.0)
        )
        candidate.fitness_score = max(0, min(1, fitness))
        
        logger.info(
            f"[CLM v3 Evolution] Updated candidate {candidate_id} performance "
            f"(fitness={candidate.fitness_score:.3f}, sharpe={sharpe:.3f})"
        )
    
    def select_top_candidates(self, n: int = 5) -> List[StrategyCandidate]:
        """
        Select top N candidates by fitness score.
        
        Args:
            n: Number of candidates to select
        
        Returns:
            List of top N StrategyCandidate
        """
        # Filter candidates with performance metrics
        evaluated = [
            c for c in self.candidates.values()
            if c.performance_metrics is not None and c.fitness_score is not None
        ]
        
        # Sort by fitness (descending)
        evaluated.sort(key=lambda c: c.fitness_score, reverse=True)
        
        top = evaluated[:n]
        
        logger.info(
            f"[CLM v3 Evolution] Selected top {len(top)} candidates "
            f"(fitness: {[c.fitness_score for c in top]})"
        )
        
        return top
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def get_candidate(self, candidate_id: UUID) -> Optional[StrategyCandidate]:
        """Get candidate by ID."""
        return self.candidates.get(candidate_id)
    
    def list_candidates(
        self,
        status: Optional[StrategyStatus] = None,
    ) -> List[StrategyCandidate]:
        """List all candidates with optional status filter."""
        candidates = list(self.candidates.values())
        
        if status:
            candidates = [c for c in candidates if c.status == status]
        
        # Sort by created_at (newest first)
        candidates.sort(key=lambda c: c.created_at, reverse=True)
        
        return candidates
    
    def get_stats(self) -> Dict:
        """Get evolution engine statistics."""
        status_counts = {}
        for candidate in self.candidates.values():
            status = candidate.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_candidates": len(self.candidates),
            "status_counts": status_counts,
            "enabled": self.config["enabled"],
        }
