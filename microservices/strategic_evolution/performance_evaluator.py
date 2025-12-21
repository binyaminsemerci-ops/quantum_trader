import json
import logging
from datetime import datetime

class PerformanceEvaluator:
    """Evaluerer strategier på tvers av PnL, WinRate, SharpeRatio, Drawdown"""
    
    def __init__(self, redis_client):
        self.redis = redis_client

    def evaluate(self):
        """
        Evaluerer strategier basert på:
        - Sharpe Ratio (40% weight)
        - Win Rate (30% weight)
        - Max Drawdown (20% weight, negativ)
        - Consistency (10% weight)
        """
        try:
            data = self.redis.lrange("quantum:strategy:performance", 0, 200)
            
            if not data:
                logging.warning(json.dumps({
                    "event": "No strategy performance data found",
                    "level": "warning"
                }))
                return []
            
            results = []
            for row in data:
                try:
                    obj = json.loads(row)
                    
                    # Calculate composite score
                    score = (
                        (obj.get("sharpe_ratio", 0) * 0.4) +
                        (obj.get("win_rate", 0) * 0.3) -
                        (obj.get("max_drawdown", 0) * 0.2) +
                        (obj.get("consistency", 0) * 0.1)
                    )
                    
                    results.append({
                        "name": obj.get("strategy", "unknown"),
                        "score": round(score, 4),
                        "sharpe": obj.get("sharpe_ratio", 0),
                        "win_rate": obj.get("win_rate", 0),
                        "drawdown": obj.get("max_drawdown", 0),
                        "consistency": obj.get("consistency", 0),
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                except Exception as e:
                    logging.error(json.dumps({
                        "event": "Failed to parse strategy data",
                        "error": str(e),
                        "level": "error"
                    }))
                    continue
            
            # Rank strategies by score
            ranked = sorted(results, key=lambda x: x["score"], reverse=True)
            
            # Store rankings in Redis
            self.redis.set("quantum:evolution:rankings", json.dumps(ranked))
            
            logging.info(json.dumps({
                "event": "Performance evaluation complete",
                "strategies_evaluated": len(ranked),
                "top_strategy": ranked[0]["name"] if ranked else None,
                "top_score": ranked[0]["score"] if ranked else 0,
                "level": "info"
            }))
            
            return ranked
            
        except Exception as e:
            logging.error(json.dumps({
                "event": "Performance evaluation failed",
                "error": str(e),
                "level": "error"
            }))
            return []
