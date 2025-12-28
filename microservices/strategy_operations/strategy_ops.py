import redis, json, torch, torch.nn as nn, torch.optim as optim, os, time, random
from datetime import datetime
REDIS_HOST=os.getenv("REDIS_HOST","redis")
r=redis.Redis(host=REDIS_HOST,port=6379,db=0)
symbols=["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]
class PolicyNet(nn.Module):
  def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3),nn.Tanh())
  def forward(self,x): return self.net(x)
policy=PolicyNet(); opt=optim.Adam(policy.parameters(),lr=1e-4)
def reward_fn(pnl,conf,vol): return pnl*conf*(1-min(vol,2)/2)
print("🧠 StrategyOps active"); 
while True:
  for s in symbols:
    pnl=random.uniform(-0.02,0.03); conf=random.uniform(0.5,1.0); vol=random.uniform(0.1,1.5)
    x=torch.rand(10); act=policy(x).mean(); rew=reward_fn(pnl,conf,vol)
    loss=-torch.log(torch.abs(act)+1e-6)*rew
    opt.zero_grad(); loss.backward(); opt.step()
    r.publish("quantum:signal:strategy",json.dumps({"symbol":s,"signal":"BUY" if act>0 else "SELL","confidence":conf,"reward":rew,"timestamp":datetime.utcnow().isoformat()}))
  time.sleep(5)
