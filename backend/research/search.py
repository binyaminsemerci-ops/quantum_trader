"""
Strategy search engine using genetic algorithms.

Generates and evolves strategy candidates through evolutionary optimization.
"""

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from .models import StrategyConfig, StrategyStats, StrategyStatus, RegimeFilter
from .backtest import StrategyBacktester
from .repositories import StrategyRepository

logger = logging.getLogger(__name__)


class StrategySearchEngine:
    """
    Generates and evolves strategy candidates using genetic algorithms.
    
    Workflow:
    1. Generate random population
    2. Backtest each candidate
    3. Rank by fitness
    4. Select top performers as parents
    5. Create offspring via mutation + crossover
    6. Repeat
    """
    
    def __init__(
        self,
        backtester: StrategyBacktester,
        repository: StrategyRepository,
        backtest_symbols: list[str],
        backtest_days: int = 90
    ):
        """
        Initialize search engine.
        
        Args:
            backtester: Backtesting engine
            repository: Strategy storage
            backtest_symbols: Symbols to use for backtesting
            backtest_days: Historical period for evaluation
        """
        self.backtester = backtester
        self.repository = repository
        self.backtest_symbols = backtest_symbols
        self.backtest_days = backtest_days
    
    def run_generation(
        self,
        population_size: int = 20,
        generation: int = 0
    ) -> list[tuple[StrategyConfig, StrategyStats]]:
        """
        Create and evaluate a population of strategies.
        
        Args:
            population_size: Number of candidates to generate
            generation: Generation number for tracking
            
        Returns:
            List of (config, stats) tuples, sorted by fitness (best first)
        """
        logger.info(f"🧬 Running generation {generation} with population={population_size}")
        
        # Generate population
        population = self._generate_population(population_size, generation)
        
        # Evaluate each candidate
        results: list[tuple[StrategyConfig, StrategyStats]] = []
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.backtest_days)
        
        for config in population:
            try:
                stats = self.backtester.backtest(
                    config=config,
                    symbols=self.backtest_symbols,
                    start=start_date,
                    end=end_date
                )
                
                # Save to repository
                self.repository.save_strategy(config)
                self.repository.save_stats(stats)
                
                results.append((config, stats))
                
                logger.info(
                    f"  ✓ {config.name}: Trades={stats.total_trades}, "
                    f"PF={stats.profit_factor:.2f}, Fitness={stats.fitness_score:.1f}"
                )
                
            except Exception as e:
                logger.error(f"  ✗ Failed to evaluate {config.name}: {e}")
                continue
        
        # Sort by fitness (descending)
        results.sort(key=lambda x: x[1].fitness_score, reverse=True)
        
        logger.info(
            f"✅ Generation {generation} complete. "
            f"Best fitness: {results[0][1].fitness_score:.1f}"
        )
        
        return results
    
    def evolve(
        self,
        parents: list[StrategyConfig],
        offspring_count: int,
        generation: int
    ) -> list[StrategyConfig]:
        """
        Generate new strategies via genetic operators.
        
        Args:
            parents: High-performing strategies to breed from
            offspring_count: Number of offspring to create
            generation: Current generation number
            
        Returns:
            List of new strategy configs
        """
        logger.info(f"🧬 Evolving {offspring_count} offspring from {len(parents)} parents")
        
        offspring: list[StrategyConfig] = []
        
        for i in range(offspring_count):
            if len(parents) >= 2 and random.random() < 0.7:
                # Crossover (70% chance)
                child = self._crossover(
                    random.choice(parents),
                    random.choice(parents),
                    generation
                )
            else:
                # Mutation only (30% chance)
                child = self._mutate(random.choice(parents), generation)
            
            offspring.append(child)
        
        logger.info(f"✅ Created {len(offspring)} offspring")
        return offspring
    
    def _generate_population(
        self,
        size: int,
        generation: int
    ) -> list[StrategyConfig]:
        """Generate random strategy population"""
        
        population: list[StrategyConfig] = []
        
        entry_types = ["ENSEMBLE_CONSENSUS", "MOMENTUM", "MEAN_REVERSION"]
        regime_filters = list(RegimeFilter)
        
        for i in range(size):
            config = StrategyConfig(
                strategy_id=f"strat_{uuid.uuid4().hex[:8]}",
                name=f"Gen{generation}_Strategy{i+1}",
                
                # Randomized filters
                regime_filter=random.choice(regime_filters),
                symbols=random.sample(
                    self.backtest_symbols,
                    k=min(3, len(self.backtest_symbols))
                ),
                timeframes=[random.choice(["5m", "15m", "1h"])],
                min_confidence=random.uniform(0.55, 0.75),
                
                # Randomized entry
                entry_type=random.choice(entry_types),
                entry_params={},
                
                # Randomized exits
                tp_percent=random.uniform(0.010, 0.025),  # 1-2.5%
                sl_percent=random.uniform(0.005, 0.015),  # 0.5-1.5%
                use_trailing=random.choice([True, False]),
                trailing_callback=random.uniform(0.010, 0.020),
                
                # Randomized risk
                max_risk_per_trade=random.uniform(0.01, 0.03),  # 1-3%
                max_leverage=random.uniform(5.0, 20.0),
                max_concurrent_positions=random.randint(3, 10),
                
                # Metadata
                generation=generation,
                status=StrategyStatus.CANDIDATE,
            )
            
            population.append(config)
        
        return population
    
    def _mutate(
        self,
        parent: StrategyConfig,
        generation: int
    ) -> StrategyConfig:
        """
        Create mutated copy of parent strategy.
        
        Randomly adjusts one or more parameters.
        """
        # Clone parent
        child = StrategyConfig(
            strategy_id=f"strat_{uuid.uuid4().hex[:8]}",
            name=f"{parent.name}_mutant",
            regime_filter=parent.regime_filter,
            symbols=parent.symbols.copy(),
            timeframes=parent.timeframes.copy(),
            min_confidence=parent.min_confidence,
            entry_type=parent.entry_type,
            entry_params=parent.entry_params.copy(),
            tp_percent=parent.tp_percent,
            sl_percent=parent.sl_percent,
            use_trailing=parent.use_trailing,
            trailing_callback=parent.trailing_callback,
            max_risk_per_trade=parent.max_risk_per_trade,
            max_leverage=parent.max_leverage,
            max_concurrent_positions=parent.max_concurrent_positions,
            generation=generation,
            parent_ids=[parent.strategy_id],
        )
        
        # Apply mutations (mutate 1-3 parameters)
        num_mutations = random.randint(1, 3)
        
        for _ in range(num_mutations):
            mutation = random.choice([
                "min_confidence",
                "tp_percent",
                "sl_percent",
                "max_risk",
                "leverage",
                "regime",
                "trailing"
            ])
            
            if mutation == "min_confidence":
                child.min_confidence = max(0.5, min(0.8,
                    child.min_confidence + random.uniform(-0.05, 0.05)
                ))
            elif mutation == "tp_percent":
                child.tp_percent = max(0.008, min(0.030,
                    child.tp_percent * random.uniform(0.8, 1.2)
                ))
            elif mutation == "sl_percent":
                child.sl_percent = max(0.004, min(0.020,
                    child.sl_percent * random.uniform(0.8, 1.2)
                ))
            elif mutation == "max_risk":
                child.max_risk_per_trade = max(0.01, min(0.05,
                    child.max_risk_per_trade * random.uniform(0.8, 1.2)
                ))
            elif mutation == "leverage":
                child.max_leverage = max(5.0, min(25.0,
                    child.max_leverage * random.uniform(0.8, 1.2)
                ))
            elif mutation == "regime":
                child.regime_filter = random.choice(list(RegimeFilter))
            elif mutation == "trailing":
                child.use_trailing = not child.use_trailing
        
        return child
    
    def _crossover(
        self,
        parent1: StrategyConfig,
        parent2: StrategyConfig,
        generation: int
    ) -> StrategyConfig:
        """
        Create child strategy by combining two parents.
        
        Takes random parameters from each parent.
        """
        child = StrategyConfig(
            strategy_id=f"strat_{uuid.uuid4().hex[:8]}",
            name=f"{parent1.name}x{parent2.name}",
            
            # Mix parameters
            regime_filter=random.choice([parent1.regime_filter, parent2.regime_filter]),
            symbols=random.choice([parent1.symbols, parent2.symbols]),
            timeframes=random.choice([parent1.timeframes, parent2.timeframes]),
            min_confidence=(parent1.min_confidence + parent2.min_confidence) / 2,
            
            entry_type=random.choice([parent1.entry_type, parent2.entry_type]),
            entry_params=random.choice([parent1.entry_params, parent2.entry_params]),
            
            tp_percent=(parent1.tp_percent + parent2.tp_percent) / 2,
            sl_percent=(parent1.sl_percent + parent2.sl_percent) / 2,
            use_trailing=random.choice([parent1.use_trailing, parent2.use_trailing]),
            trailing_callback=(parent1.trailing_callback + parent2.trailing_callback) / 2,
            
            max_risk_per_trade=(parent1.max_risk_per_trade + parent2.max_risk_per_trade) / 2,
            max_leverage=(parent1.max_leverage + parent2.max_leverage) / 2,
            max_concurrent_positions=random.choice([
                parent1.max_concurrent_positions,
                parent2.max_concurrent_positions
            ]),
            
            generation=generation,
            parent_ids=[parent1.strategy_id, parent2.strategy_id],
        )
        
        return child
