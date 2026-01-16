import redis,json,time,torch,torch.nn as nn,torch.optim as optim,os,random
from datetime import datetime,timezone
from zoneinfo import ZoneInfo

r=redis.Redis(host=os.getenv("REDIS_HOST","redis"),port=6379)
STREAM="quantum:signal:strategy"

class Adjuster(nn.Module):
  def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh())
  def forward(self,x): return self.net(x)

net=Adjuster(); opt=optim.Adam(net.parameters(),lr=1e-5)
print("🚀 RL Feedback Bridge v2 running")
last="0-0"

while True:
  msgs=r.xread({STREAM:last},block=5000,count=1)
  if not msgs: continue
  _,m=msgs[0]; mid,data=m[0]; last=mid
  d={k.decode():v.decode() for k,v in data.items()}
  pnl=float(d.get("reward",0)); conf=float(d.get("confidence",0.8))
  x=torch.tensor([pnl,conf,random.random(),0]).float()
  adj=net(x); loss=-pnl*adj.mean(); opt.zero_grad(); loss.backward(); opt.step()
  r.hset("quantum:ai_policy_adjustment",mapping={"delta":float(adj.detach().mean()),"reward":pnl,"timestamp":datetime.now(ZoneInfo("Europe/Oslo")).isoformat(),"symbol":d.get("symbol","?")})
