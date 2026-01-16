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
        
        # NEW: Read from quantum:stream:trade.intent (RL shadow intents)
        # Get last 500 entries to aggregate RL performance per symbol
        try:
            stream_entries = r.xrevrange('quantum:stream:trade.intent', '+', '-', count=500)
        except:
            # Fallback to empty if stream doesn't exist
            stream_entries = []
        
        # Aggregate RL confidence and effects by symbol
        symbol_stats = {}
        
        for entry_id, fields in stream_entries:
            try:
                # Parse JSON payload from stream entry
                payload_str = fields.get('payload')
                if not payload_str:
                    continue
                
                import json
                payload = json.loads(payload_str)
                
                symbol = payload.get('symbol')
                rl_confidence = payload.get('rl_confidence')
                rl_gate_pass = payload.get('rl_gate_pass')
                rl_effect = payload.get('rl_effect')
                
                if symbol and rl_confidence is not None:
                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {
                            'confidences': [],
                            'passes': 0,
                            'total': 0,
                            'would_flip': 0,
                            'reinforce': 0
                        }
                    
                    symbol_stats[symbol]['confidences'].append(float(rl_confidence))
                    symbol_stats[symbol]['total'] += 1
                    
                    if rl_gate_pass:
                        symbol_stats[symbol]['passes'] += 1
                    
                    if rl_effect == 'would_flip':
                        symbol_stats[symbol]['would_flip'] += 1
                    elif rl_effect == 'reinforce':
                        symbol_stats[symbol]['reinforce'] += 1
                        
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                continue
        
        # Build symbols data from aggregated stats
        symbols_data = []
        total_confidence = 0.0
        best_symbol = None
        best_pass_rate = -float('inf')
        
        for symbol, stats in symbol_stats.items():
            # Calculate metrics
            avg_confidence = sum(stats['confidences']) / len(stats['confidences'])
            pass_rate = stats['passes'] / stats['total'] if stats['total'] > 0 else 0
            
            total_confidence += avg_confidence
            
            # Use pass_rate as "reward" proxy
            symbols_data.append({
                "symbol": symbol,
                "reward": round(pass_rate, 4),  # RL gate pass rate
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "total_pnl": 0.0,
                "unrealized_pct": round(avg_confidence, 4),  # Show RL confidence instead
                "realized_pct": 0.0,
                "realized_trades": stats['total'],
                "status": "active" if stats['passes'] > 0 else "idle"
            })
            
            if pass_rate > best_pass_rate:
                best_pass_rate = pass_rate
                best_symbol = symbol
        
        # Calculate average confidence across all symbols
        avg_confidence_all = total_confidence / len(symbols_data) if symbols_data else 0.0
        
        logger.info(f"✅ RL Dashboard: {len(symbols_data)} symbols, avg confidence: {avg_confidence_all:.4f}, {len(stream_entries)} intents")
        
        return {
            "status": "online" if symbols_data else "offline",
            "symbols_tracked": len(symbols_data),
            "symbols": symbols_data,
            "best_performer": best_symbol if best_symbol else "N/A",
            "best_reward": round(best_pass_rate, 4) if best_symbol else 0.0,
            "avg_reward": round(avg_confidence_all, 4),
            "message": f"RL shadow monitoring - {len(stream_entries)} intents analyzed" if symbols_data else "No RL data"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get RL data: {e}", exc_info=True)
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
