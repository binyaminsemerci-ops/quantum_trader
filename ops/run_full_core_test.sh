#!/bin/bash
# ================================================================
# Quantum Trader – Full Core Test Suite
# Validates Tier-1 Core Execution Loop end-to-end
# Author: Quantum Systems Automation
# ================================================================

set -e
cd /home/qt/quantum_trader
export PYTHONPATH=/home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

LOGFILE="/var/log/quantum/core_test_$(date +%Y%m%d_%H%M%S).log"
echo "📊 Quantum Trader – Full Core Test" | tee -a "$LOGFILE"
echo "====================================================" | tee -a "$LOGFILE"
START=$(date +%s)

# ---------------------------------------------------------------
# 1️⃣ Validate service health
# ---------------------------------------------------------------
echo "1️⃣ Checking service status..." | tee -a "$LOGFILE"
for svc in quantum-risk-safety.service quantum-execution.service quantum-position-monitor.service; do
    systemctl is-active --quiet $svc && echo "✅ $svc ACTIVE" | tee -a "$LOGFILE" \
        || echo "❌ $svc NOT RUNNING" | tee -a "$LOGFILE"
done

# ---------------------------------------------------------------
# 2️⃣ Core validation script
# ---------------------------------------------------------------
echo -e "\n2️⃣ Running ops/validate_core_loop.py ..." | tee -a "$LOGFILE"
python3 ops/validate_core_loop.py 2>&1 | tee -a "$LOGFILE"

# ---------------------------------------------------------------
# 3️⃣ Edge-case tests
# ---------------------------------------------------------------
if [ -f "tmp/test_tier1_pipeline.py" ]; then
  echo -e "\n3️⃣ Running edge-case tests..." | tee -a "$LOGFILE"
  python3 tmp/test_tier1_pipeline.py 2>&1 | tee -a "$LOGFILE"
else
  echo "⚠️  Edge-case script not found – skipping" | tee -a "$LOGFILE"
fi

# ---------------------------------------------------------------
# 4️⃣ Manual BUY test (EventBus)
# ---------------------------------------------------------------
echo -e "\n4️⃣ Manual BUY publish test..." | tee -a "$LOGFILE"
python3 <<'EOF' 2>&1 | tee -a "$LOGFILE"
import asyncio
from ai_engine.services.eventbus_bridge import EventBusClient, TradeSignal
from datetime import datetime

async def main():
    async with EventBusClient() as bus:
        s = TradeSignal(symbol="BTCUSDT", action="BUY", confidence=0.88,
                        timestamp=datetime.utcnow().isoformat()+"Z",
                        source="auto_core_test", meta_override=False)
        msg_id = await bus.publish_signal(s)
        print(f"✅ Published BUY signal ({msg_id})")
    await asyncio.sleep(3)
    async with EventBusClient() as bus:
        safe = await bus.get_stream_length("trade.signal.safe")
        execs = await bus.get_stream_length("trade.execution.res")
        print(f"trade.signal.safe = {safe}")
        print(f"trade.execution.res = {execs}")

asyncio.run(main())
EOF

# ---------------------------------------------------------------
# 5️⃣ Prometheus metrics check
# ---------------------------------------------------------------
echo -e "\n5️⃣ Checking Prometheus metrics..." | tee -a "$LOGFILE"
curl -s http://localhost:9101/metrics | grep -E "quantum_signals_total|quantum_approved_total|quantum_executed_total" | tee -a "$LOGFILE" || echo "⚠️  Metrics endpoint not reachable" | tee -a "$LOGFILE"

# ---------------------------------------------------------------
# 6️⃣ Summary
# ---------------------------------------------------------------
END=$(date +%s)
RUNTIME=$((END-START))
echo -e "\n✅ Full core test completed in ${RUNTIME}s" | tee -a "$LOGFILE"
echo "Results saved to $LOGFILE"
echo "====================================================" | tee -a "$LOGFILE"
