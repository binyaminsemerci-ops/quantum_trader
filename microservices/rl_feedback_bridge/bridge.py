import redis, json, time, torch, torch.nn as nn, torch.optim as optim, os

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
STREAM = "quantum:stream:exitbrain.pnl"
r = redis.Redis(host=REDIS_HOST, port=6379, db=0)

class Actor(nn.Module):
    def __init__(self, input_dim=16, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# Load or initialize models
actor = Actor(); critic = Critic()
if os.path.exists("actor.pth"): actor.load_state_dict(torch.load("actor.pth"))
if os.path.exists("critic.pth"): critic.load_state_dict(torch.load("critic.pth"))

opt_a = optim.Adam(actor.parameters(), lr=1e-4)
opt_c = optim.Adam(critic.parameters(), lr=1e-4)

def get_cached_state(symbol):
    # Fetch latest signal state from AI Engine's Redis keys
    data = r.hgetall(f"quantum:ai_state:{symbol}")
    if not data: return torch.zeros(16)
    vals = [float(v.decode()) for v in data.values()]
    return torch.tensor(vals[:16])

def get_cached_action(symbol):
    # Last action (BUY=1, SELL=0)
    val = r.get(f"quantum:ai_action:{symbol}")
    return 1 if (val and val.decode()=="BUY") else 0

print("🚀 RL Feedback Bridge active — listening to PnL stream")
last_id = "0-0"
while True:
    resp = r.xread({STREAM: last_id}, block=5000, count=1)
    if not resp: continue
    _, msgs = resp[0]
    msg_id, data = msgs[0]
    last_id = msg_id

    d = {k.decode(): v.decode() for k,v in data.items()}
    symbol = d.get("symbol", "BTCUSDT")
    pnl = float(d.get("pnl", 0))
    confidence = float(d.get("confidence", 0.8))
    reward = pnl * confidence

    state = get_cached_state(symbol).float().requires_grad_(False)
    action = torch.tensor(get_cached_action(symbol))
    reward_tensor = torch.tensor(reward, dtype=torch.float32, requires_grad=False)
    
    # Critic update first (to get advantage)
    value = critic(state)
    advantage = reward_tensor - value.detach()
    value_pred = critic(state)
    loss_critic = (reward_tensor - value_pred).pow(2)
    opt_c.zero_grad(); loss_critic.backward(); opt_c.step()

    # Actor update
    with torch.no_grad():
        advantage_detached = advantage.detach()
    probs = actor(state)
    logprob = torch.log(probs[action] + 1e-8)  # Add epsilon for numerical stability
    loss_actor = -(logprob * advantage_detached)
    opt_a.zero_grad(); loss_actor.backward(); opt_a.step()

    advantage_val = advantage.item() if hasattr(advantage, 'item') else advantage
    print(f"[{symbol}] Reward={reward:.3f}, Advantage={advantage_val:.3f}, Loss_Actor={loss_actor.item():.3f}, Loss_Critic={loss_critic.item():.3f}")
    
    # Save models every 10 updates to reduce I/O
    if int(time.time()) % 10 == 0:
        torch.save(actor.state_dict(), "actor.pth")
        torch.save(critic.state_dict(), "critic.pth")
