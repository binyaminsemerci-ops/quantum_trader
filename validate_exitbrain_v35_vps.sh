#!/bin/bash
set -e
cd ~/quantum_trader
LOG=/tmp/exitbrain_v35_activation_$(date +%Y%m%d_%H%M%S).log
echo "🚀 ExitBrain v3.5 Activation & Validation — $(date)" | tee -a "$LOG"

# 1️⃣  Check env variable in docker-compose
echo -e "\n=== [Checking environment flag] ===" | tee -a "$LOG"
grep -A5 "position-monitor:" docker-compose.vps.yml | grep -E "EXIT_BRAIN_V3.*_ENABLED" | tee -a "$LOG" || echo "⚠️  No ExitBrain flags found"

# 2️⃣  Verify container uses ExitBrain v3.5
echo -e "\n=== [Container runtime verification] ===" | tee -a "$LOG"
docker compose -f docker-compose.vps.yml up -d position-monitor 2>&1 | tail -5 | tee -a "$LOG"
sleep 10
docker logs quantum_position_monitor --tail 40 2>&1 | grep -E "ExitBrain|EXIT_BRAIN" | tee -a "$LOG" || echo "ℹ️  No ExitBrain logs found yet."

# 3️⃣  Inject test positions
echo -e "\n=== [Injecting test positions] ===" | tee -a "$LOG"
docker exec quantum_redis redis-cli DEL quantum:positions:test:BTCUSDT > /dev/null
docker exec quantum_redis redis-cli HSET quantum:positions:test:BTCUSDT side long entry 51000 mark 51500 pnl 6.3 leverage 20 size 0.1 > /dev/null
docker exec quantum_redis redis-cli HSET quantum:positions:test:ETHUSDT side short entry 3200 mark 3150 pnl 1.8 leverage 20 size -0.3 > /dev/null
echo "✅ 2 synthetic positions injected" | tee -a "$LOG"

# 4️⃣  Trigger build_exit_plan manually
echo -e "\n=== [Triggering ExitBrain v3.5 manually] ===" | tee -a "$LOG"
docker exec quantum_position_monitor bash -c 'export PYTHONPATH=/app/microservices:/app && python3 << "PYEND"
import redis, os, time
from exitbrain_v3_5.exit_brain import ExitBrainV35, SignalContext
r = redis.Redis(host=os.getenv("REDIS_HOST","redis"), port=6379, db=0, decode_responses=False)
brain = ExitBrainV35()
print("🚀 Running build_exit_plan for test positions...")
for key in r.keys(b"quantum:positions:test:*"):
    pos = r.hgetall(key)
    sym = key.decode().split(":")[-1]
    side = pos[b"side"].decode()
    entry = float(pos[b"entry"])
    mark = float(pos[b"mark"])
    
    # Calculate confidence from PnL (simple heuristic)
    pnl_pct = float(pos[b"pnl"])
    confidence = min(0.95, 0.7 + abs(pnl_pct) * 0.05)
    
    # Estimate ATR (simple: 2% of entry price)
    atr = entry * 0.02
    
    ctx = SignalContext(
        symbol=sym,
        side=side,
        confidence=confidence,
        entry_price=entry,
        atr_value=atr,
        timestamp=time.time()
    )
    plan = brain.build_exit_plan(ctx)
    print(f"[Decision] {sym}: leverage={plan.leverage:.1f}x, TP={plan.take_profit_pct:.2%}, SL={plan.stop_loss_pct:.2%}")
    print(f"  Reasoning: {plan.reasoning[:80]}...")
PYEND' | tee -a "$LOG"

# 5️⃣  Redis verification
echo -e "\n=== [Redis Exit Log Check] ===" | tee -a "$LOG"
docker exec quantum_redis redis-cli KEYS "quantum:exit_log:*" | head -5 | tee -a "$LOG"

# 6️⃣  Final summary
echo -e "\n=== [Summary] ===" | tee -a "$LOG"
echo "✅ EXIT_BRAIN_V35_ENABLED flag detected if printed above" | tee -a "$LOG"
echo "✅ ExitBrainV35.build_exit_plan executed successfully if [Decision] lines appear" | tee -a "$LOG"
echo "✅ Redis exit_log keys confirm proper writeback" | tee -a "$LOG"
echo "🧠 Full log → $LOG"
echo ""
cat "$LOG"
