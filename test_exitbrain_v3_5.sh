#!/bin/bash
set -e
cd ~/quantum_trader
LOG=~/quantum_trader/logs/exit_validation_$(date +%Y%m%d_%H%M%S).log
mkdir -p ~/quantum_trader/logs
echo "🚀 Starting ExitBrain v3.5 Validation Test  —  $(date)" | tee -a "$LOG"

# 1️⃣ Ensure services are up
echo -e "\n=== [Container Check] ===" | tee -a "$LOG"
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'position_monitor|ai_engine|redis' | tee -a "$LOG"

# 2️⃣ Check that ExitBrain module exists
echo -e "\n=== [ExitBrain Module Presence] ===" | tee -a "$LOG"
if [ -d "microservices/exitbrain_v3_5" ]; then
  echo "✅ exitbrain_v3_5 module found." | tee -a "$LOG"
else
  echo "❌ exitbrain_v3_5 module missing! Sync from dev repo first." | tee -a "$LOG"
  exit 1
fi

# 3️⃣ Inject synthetic test positions into Redis
echo -e "\n=== [Injecting Test Positions] ===" | tee -a "$LOG"
docker exec quantum_redis redis-cli DEL quantum:positions:test:BTCUSDT > /dev/null
docker exec quantum_redis redis-cli HSET quantum:positions:test:BTCUSDT side long entry 52000 mark 52300 pnl 5.8 leverage 20 size 0.1 > /dev/null
docker exec quantum_redis redis-cli HSET quantum:positions:test:ETHUSDT side short entry 3200 mark 3150 pnl 1.6 leverage 20 size -0.3 > /dev/null
echo "Inserted 2 test positions (BTCUSDT long + ETHUSDT short)" | tee -a "$LOG"

# 4️⃣ Run ExitBrain validation routine inside position_monitor
echo -e "\n=== [Triggering ExitBrain Checks] ===" | tee -a "$LOG"
docker exec quantum_position_monitor python3 - <<PY
import os, time, redis, json, sys
sys.path.insert(0, '/app/microservices')
from exitbrain_v3_5.exit_brain import ExitBrainV35, SignalContext

r = redis.Redis(host=os.getenv("REDIS_HOST","redis"), port=6379, db=0)
brain = ExitBrainV35()
print("🚀 Running ExitBrain v3.5 test loop...")

for i in range(2):
    positions = [p.decode() for p in r.keys("quantum:positions:test:*")]
    print(f"\\n[Loop {i+1}] Found {len(positions)} test positions")
    
    for pos_key in positions:
        data = r.hgetall(pos_key)
        sym = pos_key.split(":")[-1]
        side = data[b'side'].decode()
        entry = float(data[b'entry'])
        mark = float(data[b'mark'])
        pnl = float(data[b'pnl'])
        leverage = float(data[b'leverage'])
        
        print(f"[Check] {sym} {side}: entry={entry}, mark={mark}, pnl={pnl}%, leverage={leverage}x")
        
        # Create signal context for exit evaluation
        ctx = SignalContext(
            symbol=sym,
            side=side,
            confidence=0.75,
            entry_price=entry,
            atr_value=entry * 0.02,
            timestamp=time.time()
        )
        
        try:
            exit_plan = brain.calculate_exits(ctx)
            print(f"[Decision] {sym}: leverage={exit_plan.leverage:.1f}x, TP={exit_plan.take_profit_pct:.2f}%, SL={exit_plan.stop_loss_pct:.2f}%")
        except Exception as e:
            print(f"[Error] {sym}: {e}")
    
    if i < 1:
        time.sleep(3)
PY

# 5️⃣ Verify Redis updates (exit triggers or log entries)
echo -e "\n=== [Redis Exit Log Check] ===" | tee -a "$LOG"
docker exec quantum_redis redis-cli KEYS "quantum:exit_log:*" | tee -a "$LOG"
docker exec quantum_redis redis-cli LRANGE quantum:exit_log:recent 0 5 | tee -a "$LOG"

# 6️⃣ Final summary
echo -e "\n=== [Summary] ===" | tee -a "$LOG"
echo "✅ If you see [Decision] lines for BTCUSDT and ETHUSDT, ExitBrain v3.5 is active and responding." | tee -a "$LOG"
echo "✅ If quantum:exit_log:* keys exist, ExitBrain is writing decisions to Redis." | tee -a "$LOG"
echo "🧠 Full log → $LOG"
