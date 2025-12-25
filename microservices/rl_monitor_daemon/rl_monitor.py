import redis, os, time, csv, datetime, requests, statistics, json
import matplotlib.pyplot as plt

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
OUTFILE = "/var/log/rl_learning_rewards.csv"
PLOTFILE = "/var/log/rl_learning_rewards.png"

r = redis.Redis(host=REDIS_HOST, port=6379, db=0)
last_id = "0-0"
rewards = []

def send_discord(msg):
    if not DISCORD_WEBHOOK: return
    try:
        requests.post(DISCORD_WEBHOOK, json={"content": msg})
    except Exception as e:
        print("Discord send error:", e)

def save_plot():
    if len(rewards) < 5: return
    plt.clf()
    xs = range(len(rewards))
    plt.plot(xs, rewards, marker="o")
    plt.title("RL Reward History")
    plt.xlabel("Iteration")
    plt.ylabel("Reward %")
    plt.grid(True)
    plt.savefig(PLOTFILE)

print("🚀 RL Monitoring Daemon started — reading from quantum:stream:exitbrain.pnl")
send_discord("🧠 RL Monitor started: observing ExitBrain feedback loop...")

with open(OUTFILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "symbol", "pnl", "reward_signal"])
    while True:
        resp = r.xread({"quantum:stream:exitbrain.pnl": last_id}, block=5000, count=1)
        if not resp: continue
        _, msgs = resp[0]
        msg_id, data = msgs[0]
        last_id = msg_id
        d = {k.decode(): v.decode() for k,v in data.items()}
        pnl = float(d.get("pnl", 0))
        reward = pnl * float(d.get("confidence", 0.8))
        rewards.append(reward)
        writer.writerow([datetime.datetime.utcnow().isoformat(), d.get("symbol",""), pnl, reward])
        f.flush()
        print(f"[{datetime.datetime.utcnow()}] {d.get('symbol')} → pnl={pnl:.2f}% → reward={reward:.3f}")
        if len(rewards) % 10 == 0:
            avg = statistics.mean(rewards[-10:])
            save_plot()
            if avg > 1.5:
                send_discord(f"🔥 RL reward average improving! Last10 avg={avg:.2f}%")
