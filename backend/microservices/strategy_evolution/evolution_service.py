import os
import json
import redis
import time
import random
import logging
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Configuration
MEMORY_DIR = "/app/memory_bank"
os.makedirs(MEMORY_DIR, exist_ok=True)

MAX_MEMORY = int(os.getenv("MAX_MEMORY", 100))  # Maximum stored strategies
SURVIVORS = int(os.getenv("SURVIVORS", 3))  # Keep top 3 alive
EVOLUTION_INTERVAL = int(os.getenv("EVOLUTION_INTERVAL", 3600 * 24))  # Daily evolution
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE", 0.7))  # 70% crossover probability

def load_memory():
    """
    Load all strategies from memory bank
    """
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".json")]
    data = []
    
    for f in files:
        try:
            filepath = os.path.join(MEMORY_DIR, f)
            with open(filepath, 'r') as file:
                strategy = json.load(file)
                # Add file metadata
                strategy['_filename'] = f
                strategy['_loaded_at'] = datetime.utcnow().isoformat()
                data.append(strategy)
        except Exception as e:
            logging.warning(f"[EVOLUTION] Could not load {f}: {e}")
            continue
    
    logging.info(f"[EVOLUTION] Loaded {len(data)} strategies from memory bank")
    return data

def save_to_memory(strategy):
    """
    Archive strategy to memory bank
    """
    try:
        # Create unique filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        variant_id = strategy.get('variant_id', 'unknown')
        filename = f"{variant_id}_{timestamp}.json"
        filepath = os.path.join(MEMORY_DIR, filename)
        
        # Add metadata
        strategy['_archived_at'] = datetime.utcnow().isoformat()
        strategy['_archive_filename'] = filename
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(strategy, f, indent=2)
        
        logging.info(f"[EVOLUTION] 📁 Archived strategy: {filename}")
        return True
    except Exception as e:
        logging.error(f"[EVOLUTION] Failed to save strategy: {e}")
        return False

def fetch_current_best_from_phase9():
    """
    Fetch the current best strategy from Phase 9 Meta-Cognitive Evaluator
    """
    try:
        # Get best strategy from Phase 9
        best_strategy = r.hgetall("meta_best_strategy")
        if best_strategy:
            # Convert to proper format
            strategy = {
                'variant_id': best_strategy.get('variant_id', 'unknown'),
                'score': float(best_strategy.get('composite_score', 0)),
                'sharpe': float(best_strategy.get('sharpe', 0)),
                'sortino': float(best_strategy.get('sortino', 0)),
                'drawdown': float(best_strategy.get('drawdown', 0)),
                'total_pnl_%': float(best_strategy.get('total_pnl_%', 0)),
                'win_rate_%': float(best_strategy.get('win_rate_%', 0)),
                'generation': int(best_strategy.get('generation', 0)),
                'timestamp': best_strategy.get('timestamp', datetime.utcnow().isoformat()),
                'source': 'phase9_meta_cognitive'
            }
            
            # Get current policy for parameters
            policy_str = r.get("current_policy")
            if policy_str:
                policy = json.loads(policy_str)
                strategy.update({
                    'risk_factor': policy.get('risk_factor', 1.0),
                    'momentum_sensitivity': policy.get('momentum_sensitivity', 1.0),
                    'mean_reversion': policy.get('mean_reversion', 0.5),
                    'position_scaler': policy.get('position_scaler', 1.0)
                })
            
            logging.info(f"[EVOLUTION] Fetched best strategy from Phase 9: {strategy['variant_id']}")
            return strategy
        
        logging.warning("[EVOLUTION] No best strategy found in Phase 9")
        return None
    except Exception as e:
        logging.error(f"[EVOLUTION] Error fetching Phase 9 strategy: {e}")
        return None

def evaluate_fitness(strategy):
    """
    Calculate comprehensive fitness score for strategy
    
    Fitness considers:
    - Performance metrics (PnL, Sharpe, Sortino)
    - Risk metrics (Drawdown)
    - Survival time (how long strategy has existed)
    - Generation (older generations get bonus for stability)
    """
    try:
        # Performance components
        score = strategy.get('score', 0)
        sharpe = strategy.get('sharpe', 0)
        sortino = strategy.get('sortino', 0)
        pnl = strategy.get('total_pnl_%', 0)
        
        # Risk components (penalties)
        drawdown = strategy.get('drawdown', 0)
        
        # Survival bonus
        timestamp_str = strategy.get('timestamp', strategy.get('_archived_at', ''))
        try:
            if timestamp_str:
                strategy_age = datetime.utcnow() - datetime.fromisoformat(timestamp_str.replace('Z', ''))
                age_days = strategy_age.total_seconds() / 86400
                # Bonus for strategies that have survived longer (up to 30 days)
                survival_bonus = min(age_days / 30.0, 1.0) * 0.2
            else:
                survival_bonus = 0
        except:
            survival_bonus = 0
        
        # Generation bonus (older generations that still perform well are valuable)
        generation = strategy.get('generation', 0)
        generation_bonus = min(generation / 50.0, 0.3)  # Up to 30% bonus
        
        # Composite fitness calculation
        fitness = (
            score * 0.35 +           # Composite score from Phase 9
            sharpe * 0.25 +          # Risk-adjusted returns
            sortino * 0.15 +         # Downside risk focus
            (pnl / 100) * 0.15 +     # Absolute profitability
            survival_bonus * 5.0 +   # Survival time bonus
            generation_bonus * 5.0 - # Generation stability bonus
            (drawdown / 100) * 0.5   # Drawdown penalty
        )
        
        return round(fitness, 4)
    
    except Exception as e:
        logging.error(f"[EVOLUTION] Fitness evaluation error: {e}")
        return 0.0

def select_survivors(memory):
    """
    Natural selection: Keep only the top N fittest strategies
    """
    if not memory:
        logging.warning("[EVOLUTION] No strategies in memory to select from")
        return []
    
    # Calculate fitness for all strategies
    for strategy in memory:
        strategy['_fitness'] = evaluate_fitness(strategy)
    
    # Sort by fitness (descending)
    ranked = sorted(memory, key=lambda x: x.get('_fitness', 0), reverse=True)
    
    # Select survivors
    survivors = ranked[:SURVIVORS]
    
    # Log results
    logging.info(f"[EVOLUTION] ╔═══════════════════════════════════════════════╗")
    logging.info(f"[EVOLUTION] ║   NATURAL SELECTION - TOP {SURVIVORS} SURVIVORS   ║")
    logging.info(f"[EVOLUTION] ╚═══════════════════════════════════════════════╝")
    
    for i, survivor in enumerate(survivors, 1):
        logging.info(f"[EVOLUTION]   #{i}: {survivor.get('variant_id', 'unknown')} "
                    f"(Fitness: {survivor.get('_fitness', 0):.4f}, "
                    f"Score: {survivor.get('score', 0):.3f}, "
                    f"Sharpe: {survivor.get('sharpe', 0):.3f})")
    
    # Store survivors in Redis
    survivor_ids = [s.get('variant_id', 'unknown') for s in survivors]
    r.set("meta_survivors", json.dumps(survivor_ids))
    
    return survivors

def crossover(parent1, parent2):
    """
    Genetic crossover: Combine two parent strategies to create offspring
    
    Uses uniform crossover for numeric parameters and
    random selection for non-numeric parameters
    """
    child = {}
    
    # Parameters that can be crossed over
    numeric_params = ['risk_factor', 'momentum_sensitivity', 'mean_reversion', 'position_scaler']
    
    for param in numeric_params:
        if param in parent1 and param in parent2:
            if random.random() < CROSSOVER_RATE:
                # Weighted average with random bias
                weight = random.uniform(0.3, 0.7)
                child[param] = parent1[param] * weight + parent2[param] * (1 - weight)
                # Add slight mutation
                child[param] *= random.uniform(0.95, 1.05)
            else:
                # Random selection
                child[param] = random.choice([parent1[param], parent2[param]])
        elif param in parent1:
            child[param] = parent1[param]
        elif param in parent2:
            child[param] = parent2[param]
        else:
            # Default value
            defaults = {'risk_factor': 1.0, 'momentum_sensitivity': 1.0, 
                       'mean_reversion': 0.5, 'position_scaler': 1.0}
            child[param] = defaults.get(param, 1.0)
    
    # Metadata
    child['variant_id'] = f"evolved_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
    child['generation'] = max(parent1.get('generation', 0), parent2.get('generation', 0)) + 1
    child['parent_ids'] = [parent1.get('variant_id', 'unknown'), parent2.get('variant_id', 'unknown')]
    child['created_at'] = datetime.utcnow().isoformat()
    child['evolution_method'] = 'genetic_crossover'
    child['source'] = 'phase10_evolution'
    
    logging.info(f"[EVOLUTION] 🧬 Crossover: {parent1.get('variant_id', 'P1')} × {parent2.get('variant_id', 'P2')} "
                f"→ {child['variant_id']} (Gen {child['generation']})")
    
    return child

def mutate(strategy):
    """
    Apply random mutations to a strategy
    """
    mutated = strategy.copy()
    
    numeric_params = ['risk_factor', 'momentum_sensitivity', 'mean_reversion', 'position_scaler']
    
    for param in numeric_params:
        if param in mutated and random.random() < 0.3:  # 30% mutation rate per parameter
            mutation_factor = random.uniform(0.85, 1.15)  # ±15% mutation
            mutated[param] *= mutation_factor
            # Ensure bounds
            mutated[param] = max(0.1, min(3.0, mutated[param]))
    
    mutated['variant_id'] = f"mutated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
    mutated['parent_ids'] = [strategy.get('variant_id', 'unknown')]
    mutated['evolution_method'] = 'mutation'
    mutated['created_at'] = datetime.utcnow().isoformat()
    
    logging.info(f"[EVOLUTION] 🦠 Mutation: {strategy.get('variant_id', 'unknown')} → {mutated['variant_id']}")
    
    return mutated

def evolve_population():
    """
    Main evolution cycle:
    1. Load memory
    2. Fetch latest from Phase 9
    3. Select survivors
    4. Create new offspring via crossover and mutation
    5. Update production policy
    6. Prune old strategies
    """
    try:
        logging.info("[EVOLUTION] ═══════════════════════════════════════")
        logging.info("[EVOLUTION] Starting evolution cycle...")
        
        # Step 1: Load existing memory
        memory = load_memory()
        
        # Step 2: Fetch latest best strategy from Phase 9
        phase9_best = fetch_current_best_from_phase9()
        if phase9_best:
            # Archive it if not already in memory
            existing_ids = [s.get('variant_id') for s in memory]
            if phase9_best.get('variant_id') not in existing_ids:
                save_to_memory(phase9_best)
                memory.append(phase9_best)
                logging.info(f"[EVOLUTION] Added Phase 9 best strategy to memory pool")
        
        # Check if we have enough strategies
        if len(memory) < 3:
            logging.info(f"[EVOLUTION] Insufficient strategies in memory ({len(memory)}/3), waiting for more data...")
            return
        
        # Step 3: Natural selection - select survivors
        survivors = select_survivors(memory)
        
        if len(survivors) < 2:
            logging.warning("[EVOLUTION] Not enough survivors for evolution")
            return
        
        # Step 4: Generate new offspring
        offspring = []
        
        # Create 2 children via crossover
        for i in range(2):
            parents = random.sample(survivors, 2)
            child = crossover(parents[0], parents[1])
            offspring.append(child)
            save_to_memory(child)
        
        # Create 1 child via mutation
        parent = random.choice(survivors)
        mutant = mutate(parent)
        offspring.append(mutant)
        save_to_memory(mutant)
        
        # Step 5: Select best offspring to become production policy
        best_offspring = max(offspring, key=lambda x: evaluate_fitness(x))
        
        # Update production policy
        policy = {
            'id': best_offspring['variant_id'],
            'generation': best_offspring.get('generation', 0),
            'risk_factor': best_offspring.get('risk_factor', 1.0),
            'momentum_sensitivity': best_offspring.get('momentum_sensitivity', 1.0),
            'mean_reversion': best_offspring.get('mean_reversion', 0.5),
            'position_scaler': best_offspring.get('position_scaler', 1.0),
            'created_at': best_offspring.get('created_at'),
            'parent_ids': best_offspring.get('parent_ids', []),
            'evolution_method': best_offspring.get('evolution_method')
        }
        
        r.set("current_policy", json.dumps(policy))
        
        # Update meta_best_strategy
        r.hset("evolution_best", mapping={
            'variant_id': best_offspring['variant_id'],
            'fitness': evaluate_fitness(best_offspring),
            'generation': best_offspring.get('generation', 0),
            'evolution_method': best_offspring.get('evolution_method', 'unknown'),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        logging.info(f"[EVOLUTION] ✅ New production policy: {policy['id']} (Gen {policy['generation']})")
        logging.info(f"[EVOLUTION]    Evolution method: {policy['evolution_method']}")
        logging.info(f"[EVOLUTION]    Risk factor: {policy['risk_factor']:.3f}")
        logging.info(f"[EVOLUTION]    Momentum: {policy['momentum_sensitivity']:.3f}")
        logging.info(f"[EVOLUTION]    Mean reversion: {policy['mean_reversion']:.3f}")
        logging.info(f"[EVOLUTION]    Position scaler: {policy['position_scaler']:.3f}")
        
        # Step 6: Prune old strategies if over capacity
        memory = load_memory()  # Reload with new additions
        if len(memory) > MAX_MEMORY:
            # Sort by fitness, keep best MAX_MEMORY
            ranked = sorted(memory, key=lambda x: evaluate_fitness(x), reverse=True)
            to_remove = ranked[MAX_MEMORY:]
            
            removed_count = 0
            for strategy in to_remove:
                try:
                    filename = strategy.get('_filename')
                    if filename:
                        filepath = os.path.join(MEMORY_DIR, filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            removed_count += 1
                except Exception as e:
                    logging.warning(f"[EVOLUTION] Could not remove strategy: {e}")
            
            logging.info(f"[EVOLUTION] 🗑️  Pruned {removed_count} weak strategies (keeping best {MAX_MEMORY})")
        
        # Store evolution statistics
        stats = {
            'total_strategies_in_memory': len(load_memory()),
            'survivors_count': len(survivors),
            'offspring_generated': len(offspring),
            'current_best_id': policy['id'],
            'current_generation': policy['generation'],
            'last_evolution': datetime.utcnow().isoformat()
        }
        r.set("evolution_stats", json.dumps(stats))
        
        logging.info(f"[EVOLUTION] 📊 Evolution complete: {len(memory)} strategies in memory")
        logging.info("[EVOLUTION] ═══════════════════════════════════════")
        
    except Exception as e:
        logging.error(f"[EVOLUTION] Evolution cycle error: {e}")
        import traceback
        traceback.print_exc()

def run_loop():
    """
    Main evolution loop
    """
    logging.info("")
    logging.info("    ╔═══════════════════════════════════════════════════════════════╗")
    logging.info("    ║  PHASE 10: AUTONOMOUS STRATEGY EVOLUTION & LONG-TERM MEMORY   ║")
    logging.info("    ║  Status: ACTIVE                                               ║")
    logging.info("    ╚═══════════════════════════════════════════════════════════════╝")
    logging.info("")
    logging.info(f"[EVOLUTION] Configuration:")
    logging.info(f"[EVOLUTION]   - Evolution Interval: {EVOLUTION_INTERVAL} seconds ({EVOLUTION_INTERVAL/3600:.1f} hours)")
    logging.info(f"[EVOLUTION]   - Survivors per Cycle: {SURVIVORS}")
    logging.info(f"[EVOLUTION]   - Max Memory Capacity: {MAX_MEMORY} strategies")
    logging.info(f"[EVOLUTION]   - Crossover Rate: {CROSSOVER_RATE*100}%")
    logging.info(f"[EVOLUTION]   - Memory Bank: {MEMORY_DIR}")
    logging.info("")
    logging.info("[EVOLUTION] 🧬 Starting evolutionary ecosystem...")
    logging.info("")
    
    # Perform initial evolution
    logging.info("[EVOLUTION] Performing initial evolution cycle...")
    evolve_population()
    
    while True:
        try:
            logging.info(f"[EVOLUTION] Sleeping for {EVOLUTION_INTERVAL/3600:.1f} hours until next evolution...")
            logging.info("")
            time.sleep(EVOLUTION_INTERVAL)
            
            evolve_population()
            
        except Exception as e:
            logging.error(f"[EVOLUTION] Error in evolution loop: {e}")
            import traceback
            traceback.print_exc()
            logging.info("[EVOLUTION] Retrying in 10 minutes...")
            time.sleep(600)

if __name__ == "__main__":
    run_loop()
