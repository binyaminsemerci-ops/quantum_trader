"""
Federation Engine - Coordinates collaboration between models and builds consensus.
"""
import json
import time
import os
import redis
from model_broker import ModelBroker
from consensus_calculator import ConsensusCalculator
from trust_memory import TrustMemory


class FederationEngine:
    """Coordinates interaction between models and builds consensus."""
    
    def __init__(self, redis_url=None):
        if redis_url is None:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_url = f"redis://{redis_host}:{redis_port}/0"
        
        self.redis = redis.from_url(redis_url)
        self.broker = ModelBroker(self.redis)
        self.calculator = ConsensusCalculator(self.redis)
        self.trust = TrustMemory(self.redis)
        self.iteration = 0
        
    def run(self):
        """Main federation loop."""
        print(json.dumps({
            "event": "Model Federation Engine started",
            "status": "active",
            "level": "info"
        }))
        
        while True:
            try:
                self.iteration += 1
                
                # 1. Collect signals from all models
                signals = self.broker.collect_signals()
                
                if not signals:
                    time.sleep(5)
                    continue
                
                print(json.dumps({
                    "iteration": self.iteration,
                    "event": "Signals collected",
                    "signal_count": len(signals),
                    "models": [s["model"] for s in signals],
                    "level": "info"
                }))
                
                # 2. Build weighted consensus
                consensus = self.calculator.build_consensus(signals, self.trust)
                
                # 3. Store consensus signal
                self.redis.set("quantum:consensus:signal", json.dumps(consensus))
                
                print(json.dumps({
                    "event": "Consensus built",
                    "action": consensus["action"],
                    "confidence": consensus["confidence"],
                    "models_used": consensus["models_used"],
                    "level": "info"
                }))
                
                # 4. Update trust weights based on agreement
                self.trust.update_trust(signals, consensus)
                
                # 5. Store federation metrics
                metrics = {
                    "iteration": self.iteration,
                    "consensus": consensus,
                    "trust_weights": consensus.get("trust_weights", {}),
                    "timestamp": time.time()
                }
                self.redis.set("quantum:federation:metrics", json.dumps(metrics))
                
                time.sleep(10)
                
            except Exception as e:
                print(json.dumps({
                    "event": "Federation error",
                    "error": str(e),
                    "level": "error"
                }))
                time.sleep(5)


if __name__ == "__main__":
    FederationEngine().run()
