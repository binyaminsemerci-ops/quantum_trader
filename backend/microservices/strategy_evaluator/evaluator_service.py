import os
import json
import time
import redis
import random
import logging
from datetime import datetime
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
SANDBOX_DIR = "/app/sandbox_strategies"
os.makedirs(SANDBOX_DIR, exist_ok=True)

EVALUATION_INTERVAL = int(os.getenv("EVALUATION_INTERVAL", 3600 * 12))  # 12 hours
NUM_VARIANTS = int(os.getenv("NUM_VARIANTS", 5))
MUTATION_RANGE = float(os.getenv("MUTATION_RANGE", 0.2))  # ±20%

def generate_strategy_variant(base_policy):
    """
    Generate a mutated variant of the base policy by randomly adjusting parameters
    """
    variant = base_policy.copy()
    variant["id"] = f"variant_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
    variant["parent_id"] = base_policy.get("id", "base")
    variant["generation"] = base_policy.get("generation", 0) + 1
    variant["created_at"] = datetime.utcnow().isoformat()
    
    # Mutate numeric parameters
    for key in variant:
        if key in ["id", "parent_id", "generation", "created_at"]:
            continue
        if isinstance(variant[key], (int, float)):
            mutation_factor = random.uniform(1 - MUTATION_RANGE, 1 + MUTATION_RANGE)
            variant[key] = variant[key] * mutation_factor
            # Ensure values stay in reasonable ranges
            if key in ["risk_factor", "momentum_sensitivity", "mean_reversion", "position_scaler"]:
                variant[key] = max(0.1, min(3.0, variant[key]))
    
    logging.info(f"[META] Generated variant: {variant['id']}")
    return variant

def get_historical_performance():
    """
    Fetch historical trading performance from Trade Journal
    """
    try:
        report = r.get("latest_report")
        if report:
            data = json.loads(report)
            return {
                "pnl": data.get("total_pnl_%", 0.0),
                "sharpe": data.get("sharpe_ratio", 0.0),
                "sortino": data.get("sortino_ratio", 0.0),
                "drawdown": data.get("max_drawdown_%", 0.0),
                "win_rate": data.get("win_rate_%", 0.0)
            }
    except Exception as e:
        logging.warning(f"[META] Could not fetch historical performance: {e}")
    
    return {"pnl": 0.0, "sharpe": 0.0, "sortino": 0.0, "drawdown": 0.0, "win_rate": 50.0}

def simulate_backtest(variant, historical_baseline):
    """
    Simulate backtesting of strategy variant using stochastic model
    Enhanced with historical baseline for more realistic evaluation
    """
    try:
        # Base performance on historical data + variant adjustments
        base_pnl = historical_baseline["pnl"]
        base_sharpe = historical_baseline["sharpe"]
        base_dd = historical_baseline["drawdown"]
        
        # Simulate strategy performance with variant adjustments
        risk_adjusted_pnl = base_pnl * variant.get("risk_factor", 1.0)
        momentum_boost = variant.get("momentum_sensitivity", 1.0) * random.uniform(0.8, 1.2)
        mean_reversion_boost = variant.get("mean_reversion", 0.5) * random.uniform(0.7, 1.3)
        
        # Generate synthetic PnL series
        num_trades = 100
        daily_returns = np.random.normal(
            loc=risk_adjusted_pnl * momentum_boost * 0.01,
            scale=0.02 * variant.get("position_scaler", 1.0),
            size=num_trades
        )
        
        # Add mean reversion component
        daily_returns += np.random.normal(0, 0.005 * mean_reversion_boost, size=num_trades)
        
        # Calculate metrics
        cumulative_returns = np.cumsum(daily_returns)
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        
        # Sortino ratio (downside deviation only)
        negative_returns = daily_returns[daily_returns < 0]
        downside_dev = np.std(negative_returns) if len(negative_returns) > 0 else np.std(daily_returns)
        sortino_ratio = np.mean(daily_returns) / (downside_dev + 1e-6) * np.sqrt(252)
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win rate
        win_rate = (daily_returns > 0).sum() / len(daily_returns) * 100
        
        # Total PnL
        total_pnl = cumulative_returns[-1] * 100  # Convert to percentage
        
        # Calculate composite score
        # Higher Sharpe/Sortino is better, lower drawdown is better
        sharpe_score = sharpe_ratio * 0.4
        sortino_score = sortino_ratio * 0.3
        drawdown_penalty = max_drawdown * 2.0  # Penalty for high drawdown
        pnl_score = total_pnl * 0.3
        
        composite_score = sharpe_score + sortino_score + pnl_score - drawdown_penalty
        
        result = {
            "variant_id": variant["id"],
            "parent_id": variant.get("parent_id", "base"),
            "generation": variant.get("generation", 1),
            "sharpe": round(sharpe_ratio, 3),
            "sortino": round(sortino_ratio, 3),
            "drawdown": round(max_drawdown * 100, 2),  # Convert to percentage
            "total_pnl_%": round(total_pnl, 2),
            "win_rate_%": round(win_rate, 2),
            "composite_score": round(composite_score, 3),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logging.info(f"[META] Backtest complete for {variant['id']}: Score={result['composite_score']}, Sharpe={result['sharpe']}, DD={result['drawdown']}%")
        return result
    
    except Exception as e:
        logging.error(f"[META] Backtest error for {variant['id']}: {e}")
        return {
            "variant_id": variant["id"],
            "composite_score": -999,
            "error": str(e)
        }

def promote_best_strategy(results):
    """
    Select and promote the best performing strategy variant
    """
    # Filter out failed simulations
    valid_results = [r for r in results if r.get("composite_score", -999) > -999]
    
    if not valid_results:
        logging.warning("[META] No valid strategy variants to promote")
        return None
    
    # Sort by composite score
    best = max(valid_results, key=lambda x: x["composite_score"])
    
    # Store in Redis
    r.hset("meta_best_strategy", mapping=best)
    
    # Update history
    history_key = "meta_strategy_history"
    r.lpush(history_key, json.dumps(best))
    r.ltrim(history_key, 0, 99)  # Keep last 100 evaluations
    
    # Log all results for comparison
    logging.info(f"[META] ╔═══════════════════════════════════════╗")
    logging.info(f"[META] ║   STRATEGY EVALUATION COMPLETE        ║")
    logging.info(f"[META] ╚═══════════════════════════════════════╝")
    logging.info(f"[META] Evaluated {len(results)} variants:")
    
    for i, res in enumerate(sorted(valid_results, key=lambda x: x["composite_score"], reverse=True), 1):
        logging.info(f"[META]   #{i}: {res['variant_id']} - Score: {res['composite_score']} | Sharpe: {res['sharpe']} | DD: {res['drawdown']}%")
    
    logging.info(f"[META] 🏆 PROMOTED BEST STRATEGY: {best['variant_id']}")
    logging.info(f"[META]    Score: {best['composite_score']}")
    logging.info(f"[META]    Sharpe: {best['sharpe']}")
    logging.info(f"[META]    Sortino: {best['sortino']}")
    logging.info(f"[META]    Drawdown: {best['drawdown']}%")
    logging.info(f"[META]    PnL: {best['total_pnl_%']}%")
    logging.info(f"[META]    Win Rate: {best['win_rate_%']}%")
    
    return best

def get_current_policy():
    """
    Retrieve current trading policy from Redis
    """
    try:
        policy_str = r.get("current_policy")
        if policy_str:
            policy = json.loads(policy_str)
            if "id" in policy:
                return policy
    except Exception as e:
        logging.warning(f"[META] Could not load current policy: {e}")
    
    # Default initial policy
    default_policy = {
        "id": "base_policy",
        "generation": 0,
        "risk_factor": 1.0,
        "momentum_sensitivity": 1.0,
        "mean_reversion": 0.5,
        "position_scaler": 1.0,
        "created_at": datetime.utcnow().isoformat()
    }
    
    logging.info("[META] Initializing with default policy")
    return default_policy

def save_variant_to_file(variant, result):
    """
    Save strategy variant and results to file for future analysis
    """
    try:
        combined = {
            "policy": variant,
            "backtest_results": result
        }
        filepath = f"{SANDBOX_DIR}/{variant['id']}.json"
        with open(filepath, "w") as f:
            json.dump(combined, f, indent=2)
        logging.info(f"[META] Saved variant to {filepath}")
    except Exception as e:
        logging.error(f"[META] Could not save variant to file: {e}")

def get_evaluation_stats():
    """
    Get statistics about strategy evolution
    """
    try:
        history = r.lrange("meta_strategy_history", 0, -1)
        if not history:
            return None
        
        strategies = [json.loads(h) for h in history]
        
        avg_score = np.mean([s.get("composite_score", 0) for s in strategies])
        avg_sharpe = np.mean([s.get("sharpe", 0) for s in strategies])
        avg_dd = np.mean([s.get("drawdown", 0) for s in strategies])
        
        best_ever = max(strategies, key=lambda x: x.get("composite_score", -999))
        
        return {
            "total_evaluations": len(strategies),
            "avg_score": round(avg_score, 3),
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_drawdown": round(avg_dd, 2),
            "best_ever_score": best_ever.get("composite_score", 0),
            "best_ever_id": best_ever.get("variant_id", "unknown"),
            "latest_generation": strategies[0].get("generation", 0) if strategies else 0
        }
    except Exception as e:
        logging.error(f"[META] Could not get evaluation stats: {e}")
        return None

def run_loop():
    """
    Main evaluation loop
    """
    logging.info("")
    logging.info("    ╔═══════════════════════════════════════════════════════════════╗")
    logging.info("    ║  PHASE 9: META-COGNITIVE STRATEGY EVALUATOR                   ║")
    logging.info("    ║  Status: ACTIVE                                               ║")
    logging.info("    ╚═══════════════════════════════════════════════════════════════╝")
    logging.info("")
    logging.info(f"[META] Configuration:")
    logging.info(f"[META]   - Evaluation Interval: {EVALUATION_INTERVAL} seconds ({EVALUATION_INTERVAL/3600:.1f} hours)")
    logging.info(f"[META]   - Variants per Cycle: {NUM_VARIANTS}")
    logging.info(f"[META]   - Mutation Range: ±{MUTATION_RANGE*100}%")
    logging.info(f"[META]   - Sandbox Directory: {SANDBOX_DIR}")
    logging.info("")
    logging.info("[META] 🧠 Starting autonomous strategy evolution...")
    logging.info("")
    
    # Perform initial evaluation
    logging.info("[META] Performing initial strategy evaluation...")
    
    while True:
        try:
            # Get current best policy
            base_policy = get_current_policy()
            logging.info(f"[META] Base policy: {base_policy.get('id', 'unknown')} (Generation {base_policy.get('generation', 0)})")
            
            # Get historical performance baseline
            historical_baseline = get_historical_performance()
            logging.info(f"[META] Historical baseline: Sharpe={historical_baseline['sharpe']}, DD={historical_baseline['drawdown']}%")
            
            # Generate variants
            logging.info(f"[META] Generating {NUM_VARIANTS} strategy variants...")
            variants = [generate_strategy_variant(base_policy) for _ in range(NUM_VARIANTS)]
            
            # Simulate backtests
            logging.info(f"[META] Running backtest simulations...")
            results = []
            for variant in variants:
                result = simulate_backtest(variant, historical_baseline)
                results.append(result)
                save_variant_to_file(variant, result)
            
            # Promote best strategy
            best = promote_best_strategy(results)
            
            if best:
                # Update current policy if better than base
                current_score = best.get("composite_score", -999)
                base_result = simulate_backtest(base_policy, historical_baseline)
                base_score = base_result.get("composite_score", -999)
                
                if current_score > base_score:
                    # Find the full variant data
                    best_variant = next((v for v in variants if v["id"] == best["variant_id"]), None)
                    if best_variant:
                        r.set("current_policy", json.dumps(best_variant))
                        logging.info(f"[META] ✅ Updated current_policy to {best_variant['id']} (improved by {current_score - base_score:.3f})")
                else:
                    logging.info(f"[META] ℹ️  Base policy still optimal (score: {base_score:.3f} vs best variant: {current_score:.3f})")
            
            # Display evolution statistics
            stats = get_evaluation_stats()
            if stats:
                logging.info(f"[META] 📊 Evolution Statistics:")
                logging.info(f"[META]    Total Evaluations: {stats['total_evaluations']}")
                logging.info(f"[META]    Avg Score: {stats['avg_score']}")
                logging.info(f"[META]    Avg Sharpe: {stats['avg_sharpe']}")
                logging.info(f"[META]    Avg Drawdown: {stats['avg_drawdown']}%")
                logging.info(f"[META]    Best Ever: {stats['best_ever_id']} (score: {stats['best_ever_score']})")
                logging.info(f"[META]    Current Generation: {stats['latest_generation']}")
            
            # Store stats in Redis
            if stats:
                r.set("meta_evolution_stats", json.dumps(stats))
            
            logging.info(f"[META] Sleeping for {EVALUATION_INTERVAL/3600:.1f} hours until next evaluation...")
            logging.info("")
            
            time.sleep(EVALUATION_INTERVAL)
            
        except Exception as e:
            logging.error(f"[META] Error in evaluation loop: {e}")
            import traceback
            traceback.print_exc()
            logging.info("[META] Retrying in 10 minutes...")
            time.sleep(600)

if __name__ == "__main__":
    run_loop()
