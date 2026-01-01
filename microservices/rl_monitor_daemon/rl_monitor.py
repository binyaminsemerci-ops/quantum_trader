import redis, os, time, csv, datetime, requests, statistics, json
import matplotlib.pyplot as plt

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")
OUTFILE = "/var/log/rl_learning_rewards.csv"
PLOTFILE = "/var/log/rl_learning_rewards.png"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))  # 10 seconds default

r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
rewards = []
last_rewards = {}  # Track last reward per symbol to detect changes

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

print(f"🚀 RL Monitoring Daemon started — polling Redis quantum:rl:reward:* keys every {POLL_INTERVAL}s")
send_discord("🧠 RL Monitor started: observing Binance PnL feedback loop...")

with open(OUTFILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "symbol", "pnl_usd", "pnl_pct", "reward_signal"])
    
    while True:
        try:
            # Find all RL reward keys
            keys = r.keys("quantum:rl:reward:*")
            
            for key in keys:
                value = r.get(key)
                if not value:
                    continue
                
                # Parse reward data (it's stored as string representation of dict)
                try:
                    data = eval(value)  # Safe here since we control the data
                    symbol = data.get("symbol", "UNKNOWN")
                    pnl_usd = float(data.get("pnl", 0))
                    pnl_pct = float(data.get("pnl_pct", 0))
                    
                    # Calculate reward signal (PnL percentage is our reward)
                    reward = pnl_pct
                    
                    # Only log if changed significantly
                    last_reward = last_rewards.get(symbol, 0)
                    if abs(reward - last_reward) > 0.1 or symbol not in last_rewards:
                        rewards.append(reward)
                        last_rewards[symbol] = reward
                        
                        # Store in sorted set for history
                        timestamp = time.time()
                        r.zadd(f"quantum:rl:history:{symbol}", {f"{timestamp}:{reward}": timestamp})
                        r.expire(f"quantum:rl:history:{symbol}", 86400)  # 24 hours
                        
                        writer.writerow([datetime.datetime.utcnow().isoformat(), symbol, pnl_usd, pnl_pct, reward])
                        f.flush()
                        print(f"[{datetime.datetime.utcnow()}] {symbol} → PnL=${pnl_usd:.2f} ({pnl_pct:+.2f}%) → reward={reward:+.3f}")
                    
                except Exception as e:
                    print(f"⚠️  Failed to parse data for {key}: {e}")
                    continue
            
            # Send Discord alerts on good performance
            if len(rewards) >= 10 and len(rewards) % 10 == 0:
                avg = statistics.mean(rewards[-10:])
                save_plot()
                if avg > 1.0:
                    send_discord(f"🔥 RL reward average improving! Last10 avg={avg:+.2f}%")
                elif avg < -1.0:
                    send_discord(f"⚠️  RL rewards declining. Last10 avg={avg:+.2f}%")
            
            time.sleep(POLL_INTERVAL)
            
        except Exception as e:
            print(f"❌ Error in monitor loop: {e}")
            time.sleep(5)

