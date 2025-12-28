import os, time, redis, json, random, math
from datetime import datetime
from collections import deque

REDIS_HOST = os.getenv("REDIS_HOST","redis")
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

class LightweightTPSLCalibrator:
    """Lightweight RL calibrator without PyTorch - uses exponential moving average"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Learning rate
        self.ema_reward = 0.0
        self.history = deque(maxlen=500)
        self.adjustment_momentum = 0.0
        self.momentum_decay = 0.9
        
    def compute_reward(self, trade):
        pnl = float(trade.get("realized_pnl", 0))
        conf = float(trade.get("confidence", 0.5))
        vol = float(trade.get("volatility", 0.02))
        lev = float(trade.get("leverage", 5))
        tp = float(trade.get("take_profit", 0))
        sl = float(trade.get("stop_loss", 0))
        
        # Reward: PnL adjusted for volatility and leverage risk
        ratio = abs(tp - sl) / max(sl, 1e-6)
        reward = pnl - (vol * 0.5) - (0.001 * lev)
        normalized_reward = reward / (ratio + 1e-3)
        
        return normalized_reward
    
    def fetch_recent_trades(self, limit=200):
        keys = r.keys("quantum:exit_log:*")
        samples = []
        for k in list(keys)[-limit:]:
            try:
                d = json.loads(r.get(k))
                samples.append(d)
            except Exception:
                continue
        return samples
    
    def update(self, trades):
        if not trades:
            return 0.0
        
        # Compute rewards
        rewards = [self.compute_reward(t) for t in trades]
        avg_reward = sum(rewards) / len(rewards)
        
        # Update EMA
        self.ema_reward = self.alpha * avg_reward + (1 - self.alpha) * self.ema_reward
        
        # Compute gradient estimate (positive = increase TP/SL, negative = decrease)
        recent_avg = sum(rewards[-50:]) / min(len(rewards), 50) if rewards else 0
        gradient = recent_avg - self.ema_reward
        
        # Update momentum
        self.adjustment_momentum = self.momentum_decay * self.adjustment_momentum + (1 - self.momentum_decay) * gradient
        
        # Scale to reasonable adjustment range [-0.002, +0.002] = [-0.2%, +0.2%]
        delta = max(-0.002, min(0.002, self.adjustment_momentum / 10))
        
        return delta

model = LightweightTPSLCalibrator(alpha=0.15)
print("🧠 Lightweight TP/SL Reinforcement Calibration Active (no PyTorch)")
last_update = time.time()
iteration = 0

while True:
    try:
        trades = model.fetch_recent_trades(150)
        if not trades:
            time.sleep(5)
            continue
        
        delta = model.update(trades)
        
        r.hset("quantum:ai_policy_adjustment", mapping={
            "delta": delta,
            "ema_reward": model.ema_reward,
            "momentum": model.adjustment_momentum,
            "timestamp": datetime.utcnow().isoformat(),
            "samples": len(trades),
            "iteration": iteration
        })
        
        if time.time() - last_update > 60:
            print(f"[{datetime.utcnow()}] ΔTP/SL: {delta:+.4f} | EMA reward: {model.ema_reward:.4f} | samples: {len(trades)} | iter: {iteration}")
            last_update = time.time()
        
        iteration += 1
        time.sleep(10)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(5)
