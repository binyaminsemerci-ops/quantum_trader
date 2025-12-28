import redis, json, torch, torch.nn as nn, torch.optim as optim, os, time, random, sys
from datetime import datetime

print("Starting StrategyOps...", flush=True)
sys.stdout.flush()

REDIS_HOST=os.getenv("REDIS_HOST","redis")
try:
    r=redis.Redis(host=REDIS_HOST,port=6379,db=0)
    r.ping()
    print(f"✅ Connected to Redis at {REDIS_HOST}", flush=True)
except Exception as e:
    print(f"❌ Redis connection failed: {e}", flush=True)
    sys.exit(1)

symbols=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
class PolicyNet(nn.Module):
  def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3),nn.Tanh())
  def forward(self,x): return self.net(x)

policy=PolicyNet(); opt=optim.Adam(policy.parameters(),lr=1e-4)
def reward_fn(pnl,conf,vol): return pnl*conf*(1-min(vol,2)/2)
print("🧠 StrategyOps active - starting signal generation", flush=True); 
while True:
  try:
    for s in symbols:
        pnl=random.uniform(-0.02,0.03); conf=random.uniform(0.5,1.0); vol=random.uniform(0.1,1.5)
        x=torch.rand(10); act=policy(x).mean(); rew=reward_fn(pnl,conf,vol)
        loss=-torch.log(torch.abs(act)+1e-6)*rew
        opt.zero_grad(); loss.backward(); opt.step()
        payload=json.dumps({"symbol":s,"signal":"BUY" if act>0 else "SELL","confidence":conf,"reward":rew,"timestamp":datetime.utcnow().isoformat()})
        r.publish("quantum:signal:strategy",payload)
        print(f"📡 Published {s} signal: {payload[:80]}...", flush=True)
    time.sleep(5)
  except Exception as e:
    print(f"❌ Error in main loop: {e}", flush=True)
    time.sleep(5)
