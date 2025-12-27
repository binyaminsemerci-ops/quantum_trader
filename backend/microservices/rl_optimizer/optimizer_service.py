"""
PHASE 8: REINFORCEMENT LEARNING OPTIMIZATION LOOP
==================================================
Continuous learning system that optimizes model weights and risk parameters
based on real trading performance (PnL, Sharpe, Drawdown).

Features:
- Reward-based learning (PnL + Sharpe - Drawdown)
- Automatic ensemble weight adjustment
- Exploration vs exploitation balance
- Real-time adaptation to market conditions
- Closed-loop intelligent trading system

Author: Quantum Trader AI System
Date: 2025-12-20
"""

import os
import json
import time
import random
import redis
import logging
from datetime import datetime
from typing import Dict, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

# Redis connection
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# RL Hyperparameters
ALPHA = float(os.getenv("RL_ALPHA", "0.3"))          # Learning rate
GAMMA = float(os.getenv("RL_GAMMA", "0.95"))         # Discount factor
EPSILON = float(os.getenv("RL_EPSILON", "0.1"))      # Exploration rate
UPDATE_INTERVAL = int(os.getenv("RL_UPDATE_INTERVAL", "1800"))  # 30 minutes

# Model keys
MODEL_KEYS = ["xgb", "lgbm", "nhits", "patchtst"]

# Reward weights
REWARD_PNL_WEIGHT = float(os.getenv("REWARD_PNL_WEIGHT", "0.7"))
REWARD_SHARPE_WEIGHT = float(os.getenv("REWARD_SHARPE_WEIGHT", "0.25"))
REWARD_DRAWDOWN_WEIGHT = float(os.getenv("REWARD_DRAWDOWN_WEIGHT", "0.05"))

# Safety limits
MIN_WEIGHT = float(os.getenv("MIN_WEIGHT", "0.05"))  # Minimum 5% per model
MAX_WEIGHT = float(os.getenv("MAX_WEIGHT", "0.60"))  # Maximum 60% per model


def fetch_reward() -> float:
    """
    Calculate reward signal from latest performance report.
    
    Reward formula:
    reward = (PnL * 0.7) + (Sharpe * 0.25) - (Drawdown * 0.05)
    
    This balances profit maximization with risk management.
    
    Returns:
        Float reward value (can be negative)
    """
    try:
        report_json = r.get("latest_report")
        if not report_json:
            logging.warning("[RL] No latest_report found in Redis")
            return 0.0
        
        report = json.loads(report_json)
        
        # Extract metrics
        pnl = report.get("total_pnl_%", 0)
        sharpe = report.get("sharpe_ratio", 0)
        drawdown = report.get("max_drawdown_%", 0)
        
        # Calculate weighted reward
        reward = (
            pnl * REWARD_PNL_WEIGHT +
            sharpe * REWARD_SHARPE_WEIGHT -
            drawdown * REWARD_DRAWDOWN_WEIGHT
        )
        
        logging.info(f"[RL] Reward components: PnL={pnl:.2f}%, Sharpe={sharpe:.2f}, DD={drawdown:.2f}%")
        logging.info(f"[RL] Calculated reward={reward:.3f}")
        
        return reward
    except Exception as e:
        logging.error(f"[RL] Error fetching reward: {e}")
        return 0.0


def load_weights() -> Dict[str, float]:
    """
    Load current model weights from Redis.
    
    Returns:
        Dictionary of model weights
    """
    try:
        weights = r.hgetall("governance_weights")
        
        if not weights:
            logging.warning("[RL] No weights found, initializing to equal distribution")
            weights = {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}
            return weights
        
        # Convert to float
        weights = {k: float(v) for k, v in weights.items()}
        
        # Ensure all models have weights
        for key in MODEL_KEYS:
            if key not in weights:
                weights[key] = 1.0 / len(MODEL_KEYS)
        
        return weights
    except Exception as e:
        logging.error(f"[RL] Error loading weights: {e}")
        return {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0 with safety constraints.
    
    Args:
        weights: Dictionary of model weights
    
    Returns:
        Normalized weights with MIN_WEIGHT and MAX_WEIGHT constraints
    """
    # Apply min/max constraints
    constrained = {}
    for k, v in weights.items():
        constrained[k] = max(MIN_WEIGHT, min(MAX_WEIGHT, v))
    
    # Normalize to sum = 1.0
    total = sum(constrained.values())
    if total == 0:
        return {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}
    
    normalized = {k: v / total for k, v in constrained.items()}
    
    return normalized


def update_weights() -> None:
    """
    RL weight update using epsilon-greedy strategy.
    
    Strategy:
    - With probability EPSILON: Explore (random adjustment)
    - With probability (1-EPSILON): Exploit (reward-based adjustment)
    """
    # Load current weights
    old_weights = load_weights()
    logging.info(f"[RL] Current weights: {old_weights}")
    
    # Calculate reward from performance
    reward = fetch_reward()
    
    # Store reward history for tracking
    try:
        r.lpush("rl_reward_history", json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "reward": reward
        }))
        r.ltrim("rl_reward_history", 0, 99)  # Keep last 100
    except Exception as e:
        logging.error(f"[RL] Error storing reward history: {e}")
    
    # Epsilon-greedy strategy
    if random.random() < EPSILON:
        # EXPLORATION: Random adjustment
        model_to_adjust = random.choice(MODEL_KEYS)
        adjustment_factor = random.uniform(0.9, 1.1)
        old_weights[model_to_adjust] *= adjustment_factor
        
        logging.info(f"[RL] 🔍 EXPLORATION: Adjusted {model_to_adjust} by {adjustment_factor:.3f}x")
    else:
        # EXPLOITATION: Reward-based adjustment
        for model_key in MODEL_KEYS:
            # Calculate adjustment based on reward
            # Positive reward → increase weights
            # Negative reward → decrease weights
            delta = ALPHA * reward * random.uniform(0.95, 1.05)
            
            if reward > 0:
                old_weights[model_key] += abs(delta)
            else:
                old_weights[model_key] -= abs(delta)
        
        logging.info(f"[RL] 🎯 EXPLOITATION: Reward-based adjustment (reward={reward:.3f})")
    
    # Normalize and apply safety constraints
    new_weights = normalize(old_weights)
    
    # Update Redis
    try:
        r.hset("governance_weights", mapping={k: round(v, 4) for k, v in new_weights.items()})
        r.set("rl_last_update", datetime.utcnow().isoformat())
        
        # Store update history
        r.lpush("rl_update_history", json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "old_weights": {k: round(v, 4) for k, v in old_weights.items()},
            "new_weights": {k: round(v, 4) for k, v in new_weights.items()},
            "reward": reward
        }))
        r.ltrim("rl_update_history", 0, 99)  # Keep last 100
        
        logging.info(f"[RL] ✅ Updated weights: {new_weights}")
        
        # Log significant changes
        for key in MODEL_KEYS:
            change = new_weights[key] - old_weights.get(key, 0)
            if abs(change) > 0.05:
                logging.info(f"[RL] 📊 Significant change in {key}: {change:+.4f}")
        
    except Exception as e:
        logging.error(f"[RL] Error updating weights in Redis: {e}")


def get_rl_stats() -> Dict:
    """Get RL optimizer statistics."""
    try:
        weights = load_weights()
        last_update = r.get("rl_last_update") or "Never"
        
        # Get reward history
        reward_history = []
        for item in r.lrange("rl_reward_history", 0, 9):
            try:
                reward_history.append(json.loads(item))
            except:
                continue
        
        avg_reward = np.mean([h["reward"] for h in reward_history]) if reward_history else 0.0
        
        return {
            "current_weights": weights,
            "last_update": last_update,
            "avg_reward_last_10": round(avg_reward, 3),
            "total_updates": r.llen("rl_update_history"),
            "hyperparameters": {
                "alpha": ALPHA,
                "gamma": GAMMA,
                "epsilon": EPSILON,
                "update_interval_seconds": UPDATE_INTERVAL
            }
        }
    except Exception as e:
        logging.error(f"[RL] Error getting stats: {e}")
        return {}


def run_loop():
    """Main RL optimization loop."""
    logging.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  PHASE 8: REINFORCEMENT LEARNING OPTIMIZATION LOOP        ║
    ║  Status: ACTIVE                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    logging.info(f"[RL] Configuration:")
    logging.info(f"  - Learning Rate (α): {ALPHA}")
    logging.info(f"  - Discount Factor (γ): {GAMMA}")
    logging.info(f"  - Exploration Rate (ε): {EPSILON}")
    logging.info(f"  - Update Interval: {UPDATE_INTERVAL} seconds ({UPDATE_INTERVAL // 60} minutes)")
    logging.info(f"  - Model Keys: {MODEL_KEYS}")
    logging.info(f"  - Weight Constraints: {MIN_WEIGHT:.2f} - {MAX_WEIGHT:.2f}")
    logging.info(f"  - Reward Weights: PnL={REWARD_PNL_WEIGHT}, Sharpe={REWARD_SHARPE_WEIGHT}, DD={REWARD_DRAWDOWN_WEIGHT}")
    
    logging.info("[RL] 🚀 Starting continuous optimization loop...")
    
    # Initialize weights if not present
    if not r.exists("governance_weights"):
        initial_weights = {k: 1.0 / len(MODEL_KEYS) for k in MODEL_KEYS}
        r.hset("governance_weights", mapping={k: round(v, 4) for k, v in initial_weights.items()})
        logging.info(f"[RL] Initialized equal weights: {initial_weights}")
    
    # Perform initial update
    try:
        logging.info("[RL] Performing initial weight update...")
        update_weights()
    except Exception as e:
        logging.error(f"[RL] Error in initial update: {e}")
    
    # Main loop
    while True:
        try:
            time.sleep(UPDATE_INTERVAL)
            
            logging.info(f"[RL] ⏰ Scheduled update triggered (interval: {UPDATE_INTERVAL // 60}m)")
            update_weights()
            
            # Log stats every 5 updates
            stats = get_rl_stats()
            if stats.get("total_updates", 0) % 5 == 0:
                logging.info(f"[RL] 📊 Stats: {stats}")
            
        except KeyboardInterrupt:
            logging.info("[RL] 🛑 Shutting down RL Optimizer...")
            break
        except Exception as e:
            logging.error(f"[RL] ❌ Error in main loop: {e}")
            time.sleep(120)  # Wait 2 minutes on error


if __name__ == "__main__":
    run_loop()
