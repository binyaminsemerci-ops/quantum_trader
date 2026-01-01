"""RL Intelligence Dashboard Router"""
from fastapi import APIRouter
import redis
import os
import logging
import json
from typing import List, Dict

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rl-dashboard", tags=["RL Intelligence"])

@router.get("/")
def get_rl_dashboard():
    """Get RL Intelligence dashboard data from Redis
    
    Returns RL agent performance metrics for tracked symbols.
    """
    try:
        redis_host = os.getenv('REDIS_HOST', 'redis')
        r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Get RL metrics from Redis
        # Check for RL reward keys
        rl_keys = r.keys("quantum:rl:reward:*")
        
        symbols_data = []
        total_reward = 0.0
        best_symbol = None
        best_reward = -float('inf')
        
        for key in rl_keys:
            # Extract symbol from key: quantum:rl:reward:BTCUSDT -> BTCUSDT
            symbol = key.split(':')[-1]
            value = r.get(key)
            
            if value:
                # Parse dict string from Binance PnL tracker
                try:
                    data = eval(value)  # Safe since we control the source
                    reward_val = float(data.get('pnl_pct', 0))
                    pnl_usd = float(data.get('pnl', 0))
                except:
                    # Fallback to old format (simple float)
                    reward_val = float(value)
                    pnl_usd = 0.0
                
                total_reward += reward_val
                
                symbols_data.append({
                    "symbol": symbol,
                    "reward": round(reward_val, 4),
                    "pnl_usd": round(pnl_usd, 2),
                    "status": "active" if abs(reward_val) > 0.01 else "idle"
                })
                
                if reward_val > best_reward:
                    best_reward = reward_val
                    best_symbol = symbol
        
        # Calculate average reward
        avg_reward = total_reward / len(symbols_data) if symbols_data else 0.0
        
        logger.info(f"✅ RL Dashboard: {len(symbols_data)} symbols, avg reward: {avg_reward:.4f}")
        
        return {
            "status": "online" if symbols_data else "offline",
            "symbols_tracked": len(symbols_data),
            "symbols": symbols_data,
            "best_performer": best_symbol if best_symbol else "N/A",
            "best_reward": round(best_reward, 4) if best_symbol else 0.0,
            "avg_reward": round(avg_reward, 4),
            "message": "RL agents active" if symbols_data else "No active RL agents"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get RL data: {e}")
        return {
            "status": "offline",
            "symbols_tracked": 0,
            "symbols": [],
            "best_performer": "N/A",
            "best_reward": 0.0,
            "avg_reward": 0.0,
            "message": f"RL data unavailable: {str(e)}"
        }

@router.get("/history/{symbol}")
def get_rl_history(symbol: str, limit: int = 100):
    """Get RL reward history for a specific symbol"""
    try:
        redis_host = os.getenv('REDIS_HOST', 'redis')
        r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Get last N entries from sorted set
        history = r.zrevrange(f"quantum:rl:history:{symbol}", 0, limit-1, withscores=True)
        
        rewards = []
        timestamps = []
        
        for entry, score in history:
            ts, reward = entry.split(':')
            timestamps.append(float(ts))
            rewards.append(float(reward))
        
        return {
            "symbol": symbol,
            "timestamps": timestamps,
            "rewards": rewards,
            "count": len(rewards)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get RL history for {symbol}: {e}")
        return {
            "symbol": symbol,
            "timestamps": [],
            "rewards": [],
            "count": 0
        }
