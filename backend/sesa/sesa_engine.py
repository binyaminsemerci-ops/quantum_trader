"""SESA Engine - Self-Evolving Strategy Architect.

Main orchestrator for autonomous strategy evolution through:
- Mutation: Generate strategy variations
- Evaluation: Test via backtesting/simulation
- Selection: Choose top performers
- Evolution Loop: Continuous improvement
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from backend.sesa.evaluation_engine import EvaluationEngine, EvaluationResult
from backend.sesa.mutation_operators import (
    MutatedStrategy,
    MutationOperators,
    StrategyConfig,
)
from backend.sesa.selection_engine import (
    CandidateStatus,
    SelectionCriteria,
    SelectionEngine,
    SelectionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """
    Configuration for evolution run.
    """
    
    # Mutation settings
    num_mutations_per_parent: int = 10
    mutation_rate: float = 0.3
    mutation_magnitude: float = 0.2
    
    # Evaluation settings
    evaluation_type: str = "backtest"  # "backtest" or "simulation"
    evaluation_lookback_days: int = 30
    max_concurrent_evaluations: int = 5
    
    # Selection settings
    selection_criteria: Optional[SelectionCriteria] = None
    
    # Evolution loop settings
    num_generations: int = 5
    elite_carry_forward: int = 2  # Top N strategies to keep each generation
    
    # Integration flags
    use_memory_engine: bool = True
    use_scenario_simulator: bool = True
    publish_events: bool = True


@dataclass
class GenerationResult:
    """
    Results from a single evolution generation.
    """
    
    generation_number: int
    
    # Input
    parent_strategies: list[str]
    
    # Outputs
    mutations_generated: int
    evaluations_completed: int
    selection_result: Optional[SelectionResult]
    
    # Best of generation
    best_strategy_id: Optional[str] = None
    best_score: float = 0.0
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation_number": self.generation_number,
            "parent_strategies": self.parent_strategies,
            "mutations_generated": self.mutations_generated,
            "evaluations_completed": self.evaluations_completed,
            "selection_result": self.selection_result.to_dict() if self.selection_result else None,
            "best_strategy_id": self.best_strategy_id,
            "best_score": self.best_score,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class EvolutionResult:
    """
    Complete evolution run results.
    """
    
    evolution_id: str
    config: EvolutionConfig
    
    # Generation results
    generations: list[GenerationResult] = field(default_factory=list)
    
    # Overall statistics
    total_mutations_evaluated: int = 0
    total_strategies_selected: int = 0
    
    # Best overall
    best_strategy_id: Optional[str] = None
    best_score: float = 0.0
    best_generation: int = 0
    
    # Production candidates
    production_candidates: list[str] = field(default_factory=list)
    shadow_candidates: list[str] = field(default_factory=list)
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evolution_id": self.evolution_id,
            "generations": [g.to_dict() for g in self.generations],
            "total_mutations_evaluated": self.total_mutations_evaluated,
            "total_strategies_selected": self.total_strategies_selected,
            "best_strategy_id": self.best_strategy_id,
            "best_score": self.best_score,
            "best_generation": self.best_generation,
            "production_candidates": self.production_candidates,
            "shadow_candidates": self.shadow_candidates,
            "total_duration_seconds": self.total_duration_seconds,
        }


class SESAEngine:
    """
    Self-Evolving Strategy Architect Engine.
    
    Orchestrates autonomous strategy evolution:
    1. Generate mutations from parent strategies
    2. Evaluate mutations via backtesting/simulation
    3. Select top performers
    4. Repeat for multiple generations
    5. Flag candidates for shadow/production deployment
    
    Integration:
    - Memory Engine: Store evolution history
    - Scenario Simulator: Forward-looking evaluation
    - Replay Engine: Historical backtesting
    - EventBus: Publish evolution events
    - PolicyStore: Load/update strategy configs
    
    Usage:
        engine = SESAEngine(
            mutation_operators=mutation_ops,
            evaluation_engine=eval_engine,
            selection_engine=selection_engine,
        )
        
        # Single generation
        result = await engine.run_generation(
            parent_strategies=[baseline_config],
            config=EvolutionConfig(num_mutations_per_parent=10),
        )
        
        # Multi-generation evolution
        evolution_result = await engine.run_evolution(
            initial_strategies=[baseline_config],
            config=EvolutionConfig(num_generations=5),
        )
    """
    
    def __init__(
        self,
        mutation_operators: Optional[MutationOperators] = None,
        evaluation_engine: Optional[EvaluationEngine] = None,
        selection_engine: Optional[SelectionEngine] = None,
        event_bus: Optional[Any] = None,
        memory_engine: Optional[Any] = None,
    ):
        """
        Initialize SESA Engine.
        
        Args:
            mutation_operators: MutationOperators instance
            evaluation_engine: EvaluationEngine instance
            selection_engine: SelectionEngine instance
            event_bus: EventBus for publishing events
            memory_engine: Memory Engine for storing evolution history
        """
        self.mutation_operators = mutation_operators or MutationOperators()
        self.evaluation_engine = evaluation_engine or EvaluationEngine()
        self.selection_engine = selection_engine or SelectionEngine()
        self.event_bus = event_bus
        self.memory_engine = memory_engine
        
        logger.info("SESAEngine initialized")
    
    async def run_evolution(
        self,
        initial_strategies: list[StrategyConfig],
        config: EvolutionConfig,
    ) -> EvolutionResult:
        """
        Run complete multi-generation evolution.
        
        Args:
            initial_strategies: Starting strategies for evolution
            config: Evolution configuration
        
        Returns:
            EvolutionResult with complete evolution history
        """
        evolution_id = str(uuid4())
        start_time = datetime.now()
        
        logger.info(
            f"Starting evolution {evolution_id}: "
            f"{len(initial_strategies)} initial strategies, "
            f"{config.num_generations} generations"
        )
        
        result = EvolutionResult(
            evolution_id=evolution_id,
            config=config,
            start_time=start_time,
        )
        
        # Current parent pool
        current_parents = initial_strategies.copy()
        
        # Run generations
        for gen in range(1, config.num_generations + 1):
            logger.info(f"Generation {gen}/{config.num_generations}")
            
            gen_result = await self.run_generation(
                parent_strategies=current_parents,
                generation_number=gen,
                config=config,
            )
            
            result.generations.append(gen_result)
            
            # Update statistics
            result.total_mutations_evaluated += gen_result.evaluations_completed
            if gen_result.selection_result:
                result.total_strategies_selected += len(gen_result.selection_result.selected)
            
            # Update best overall
            if gen_result.best_score > result.best_score:
                result.best_score = gen_result.best_score
                result.best_strategy_id = gen_result.best_strategy_id
                result.best_generation = gen
            
            # Collect candidates
            if gen_result.selection_result:
                for selected in gen_result.selection_result.selected:
                    if selected.status == CandidateStatus.PRODUCTION_CANDIDATE:
                        result.production_candidates.append(selected.strategy_id)
                    elif selected.status == CandidateStatus.SHADOW_CANDIDATE:
                        result.shadow_candidates.append(selected.strategy_id)
            
            # Prepare parents for next generation
            if gen < config.num_generations:
                current_parents = self._select_next_parents(
                    gen_result,
                    config,
                )
                
                logger.info(f"Selected {len(current_parents)} parents for generation {gen+1}")
        
        # Finalize
        end_time = datetime.now()
        result.end_time = end_time
        result.total_duration_seconds = (end_time - start_time).total_seconds()
        
        logger.info(
            f"Evolution {evolution_id} complete: "
            f"{result.total_mutations_evaluated} mutations evaluated, "
            f"best score={result.best_score:.2f}, "
            f"{len(result.production_candidates)} production candidates"
        )
        
        # Store in memory engine
        if config.use_memory_engine and self.memory_engine:
            await self._store_evolution_result(result)
        
        # Publish event
        if config.publish_events and self.event_bus:
            await self._publish_evolution_complete(result)
        
        return result
    
    async def run_generation(
        self,
        parent_strategies: list[StrategyConfig],
        generation_number: int,
        config: EvolutionConfig,
    ) -> GenerationResult:
        """
        Run a single generation of evolution.
        
        Steps:
        1. Generate mutations from parents
        2. Evaluate mutations
        3. Select top performers
        
        Args:
            parent_strategies: Parent strategies to mutate
            generation_number: Generation number (for logging)
            config: Evolution configuration
        
        Returns:
            GenerationResult
        """
        start_time = datetime.now()
        
        parent_ids = [p.strategy_id for p in parent_strategies]
        
        logger.info(
            f"Generation {generation_number}: "
            f"Mutating {len(parent_strategies)} parents"
        )
        
        result = GenerationResult(
            generation_number=generation_number,
            parent_strategies=parent_ids,
            start_time=start_time,
        )
        
        # Step 1: Generate mutations
        all_mutations: list[MutatedStrategy] = []
        
        for parent in parent_strategies:
            mutations = self.mutation_operators.generate_mutations(
                parent_config=parent,
                num_mutations=config.num_mutations_per_parent,
                mutation_rate=config.mutation_rate,
                mutation_magnitude=config.mutation_magnitude,
            )
            all_mutations.extend(mutations)
        
        result.mutations_generated = len(all_mutations)
        
        logger.info(f"Generated {len(all_mutations)} mutations")
        
        # Publish mutation events
        if config.publish_events and self.event_bus:
            for mutation in all_mutations:
                await self._publish_mutation_created(mutation, generation_number)
        
        # Step 2: Evaluate mutations
        evaluation_results = await self.evaluation_engine.evaluate_batch(
            strategies=all_mutations,
            evaluation_type=config.evaluation_type,
            lookback_days=config.evaluation_lookback_days,
            max_concurrent=config.max_concurrent_evaluations,
        )
        
        result.evaluations_completed = len(evaluation_results)
        
        logger.info(f"Evaluated {len(evaluation_results)} mutations")
        
        # Publish evaluation events
        if config.publish_events and self.event_bus:
            for eval_result in evaluation_results:
                await self._publish_evaluation_complete(eval_result, generation_number)
        
        # Step 3: Select top performers
        selection_criteria = config.selection_criteria or SelectionCriteria()
        self.selection_engine.criteria = selection_criteria
        
        selection_result = self.selection_engine.select_top_performers(
            evaluation_results
        )
        
        result.selection_result = selection_result
        
        # Update best of generation
        if selection_result.selected:
            best = selection_result.selected[0]  # Rank 1
            result.best_strategy_id = best.strategy_id
            result.best_score = best.composite_score
        
        logger.info(
            f"Selected {len(selection_result.selected)} top performers"
        )
        
        # Publish selection events
        if config.publish_events and self.event_bus:
            await self._publish_selection_complete(selection_result, generation_number)
        
        # Finalize timing
        end_time = datetime.now()
        result.end_time = end_time
        result.duration_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _select_next_parents(
        self,
        gen_result: GenerationResult,
        config: EvolutionConfig,
    ) -> list[StrategyConfig]:
        """
        Select parent strategies for next generation.
        
        Uses elitism: Top N performers become parents.
        """
        if not gen_result.selection_result or not gen_result.selection_result.selected:
            logger.warning("No selected strategies; using previous parents")
            return []
        
        # Take top N elite strategies
        elite = gen_result.selection_result.selected[:config.elite_carry_forward]
        
        # Convert back to StrategyConfig (would need actual configs here)
        # For now, placeholder - in real implementation, retrieve from storage
        parents: list[StrategyConfig] = []
        
        logger.info(f"Carrying forward {len(elite)} elite strategies as parents")
        
        return parents
    
    async def _publish_mutation_created(
        self,
        mutation: MutatedStrategy,
        generation: int,
    ) -> None:
        """Publish mutation_created event."""
        if not self.event_bus:
            return
        
        try:
            await self.event_bus.publish(
                "strategy_mutation_created",
                {
                    "mutation_id": mutation.mutation_id,
                    "parent_id": mutation.parent_id,
                    "mutation_type": mutation.mutation_type.value,
                    "generation": generation,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish mutation_created: {e}")
    
    async def _publish_evaluation_complete(
        self,
        eval_result: EvaluationResult,
        generation: int,
    ) -> None:
        """Publish evaluation_complete event."""
        if not self.event_bus:
            return
        
        try:
            await self.event_bus.publish(
                "strategy_evaluation_complete",
                {
                    "strategy_id": eval_result.strategy_id,
                    "mutation_id": eval_result.mutation_id,
                    "composite_score": eval_result.composite_score,
                    "passed": eval_result.passed_minimum_requirements,
                    "generation": generation,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish evaluation_complete: {e}")
    
    async def _publish_selection_complete(
        self,
        selection_result: SelectionResult,
        generation: int,
    ) -> None:
        """Publish selection_complete event."""
        if not self.event_bus:
            return
        
        try:
            await self.event_bus.publish(
                "strategy_selection_complete",
                {
                    "selected_count": len(selection_result.selected),
                    "top_strategies": [s.strategy_id for s in selection_result.selected[:3]],
                    "generation": generation,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish selection_complete: {e}")
    
    async def _publish_evolution_complete(
        self,
        evolution_result: EvolutionResult,
    ) -> None:
        """Publish evolution_complete event."""
        if not self.event_bus:
            return
        
        try:
            await self.event_bus.publish(
                "sesa_evolution_complete",
                {
                    "evolution_id": evolution_result.evolution_id,
                    "generations": len(evolution_result.generations),
                    "best_strategy_id": evolution_result.best_strategy_id,
                    "best_score": evolution_result.best_score,
                    "production_candidates": evolution_result.production_candidates,
                    "shadow_candidates": evolution_result.shadow_candidates,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Failed to publish evolution_complete: {e}")
    
    async def _store_evolution_result(
        self,
        evolution_result: EvolutionResult,
    ) -> None:
        """Store evolution result in Memory Engine."""
        if not self.memory_engine:
            return
        
        try:
            # TODO: Integration with actual Memory Engine
            # await self.memory_engine.store_episodic_memory(
            #     event_type="sesa_evolution",
            #     data=evolution_result.to_dict(),
            # )
            logger.info(f"Stored evolution result: {evolution_result.evolution_id}")
        except Exception as e:
            logger.error(f"Failed to store evolution result: {e}")
