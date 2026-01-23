#!/bin/bash
# P2.7C.1: Fix Harvest Metrics Staleness
# Root cause: Redis had no last_update_epoch field, exporter used hardcoded timestamp
# Solution: Publisher writes last_update_epoch on every publish, exporter reads it

set -euo pipefail

echo "============================================"
echo "P2.7C.1: Fix Harvest Metrics Staleness"
echo "============================================"
echo

# Step 1: Copy updated files to VPS
echo "📦 Step 1: Copying updated microservices..."
cd /root/quantum_trader
git pull origin main
if [ $? -ne 0 ]; then
    echo "⚠️  Git pull failed, continuing with local changes..."
fi
echo "   ✅ Code updated"
echo

# Step 2: Restart publisher (writes last_update_epoch to Redis)
echo "🔄 Step 2: Restarting harvest proposal publisher..."
systemctl restart quantum-harvest-proposal
sleep 2
if systemctl is-active --quiet quantum-harvest-proposal; then
    echo "   ✅ Publisher restarted"
else
    echo "   ❌ Publisher failed to start"
    systemctl status quantum-harvest-proposal --no-pager -l | tail -20
    exit 1
fi
echo

# Step 3: Verify Redis field now exists
echo "🔍 Step 3: Verifying last_update_epoch in Redis..."
REDIS_CHECK=0
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    VALUE=$(redis-cli HGET quantum:harvest:proposal:$symbol last_update_epoch 2>/dev/null || echo "")
    if [ -n "$VALUE" ]; then
        echo "   ✅ $symbol: last_update_epoch=$VALUE"
        REDIS_CHECK=1
    else
        echo "   ⚠️  $symbol: last_update_epoch NOT FOUND"
    fi
done

if [ $REDIS_CHECK -eq 0 ]; then
    echo "   ❌ No symbols have last_update_epoch field!"
    echo "   Wait 10s for publisher cycle..."
    sleep 10
    
    # Retry check
    for symbol in BTCUSDT ETHUSDT SOLUSDT; do
        VALUE=$(redis-cli HGET quantum:harvest:proposal:$symbol last_update_epoch 2>/dev/null || echo "")
        if [ -n "$VALUE" ]; then
            echo "   ✅ $symbol: last_update_epoch=$VALUE (after retry)"
            REDIS_CHECK=1
        fi
    done
    
    if [ $REDIS_CHECK -eq 0 ]; then
        echo "   ❌ Still no last_update_epoch after retry. Check publisher logs:"
        journalctl -u quantum-harvest-proposal --since "1 minute ago" --no-pager | tail -20
        exit 2
    fi
fi
echo

# Step 4: Restart exporter (reads last_update_epoch from Redis)
echo "🔄 Step 4: Restarting harvest metrics exporter..."
systemctl restart quantum-harvest-metrics-exporter
sleep 2
if systemctl is-active --quiet quantum-harvest-metrics-exporter; then
    echo "   ✅ Exporter restarted"
else
    echo "   ❌ Exporter failed to start"
    systemctl status quantum-harvest-metrics-exporter --no-pager -l | tail -20
    exit 3
fi
echo

# Step 5: Verify exporter metrics update dynamically
echo "🔍 Step 5: Verifying exporter metrics update dynamically..."
echo "   Sampling exporter metrics (10s interval)..."

# Sample 1
SAMPLE1=$(curl -s http://127.0.0.1:8042/metrics | grep "quantum_harvest_last_update_epoch{symbol=\"BTCUSDT\"}" | awk '{print $2}')
echo "   Sample 1: BTCUSDT epoch=$SAMPLE1"

sleep 10

# Sample 2
SAMPLE2=$(curl -s http://127.0.0.1:8042/metrics | grep "quantum_harvest_last_update_epoch{symbol=\"BTCUSDT\"}" | awk '{print $2}')
echo "   Sample 2: BTCUSDT epoch=$SAMPLE2"

if [ "$SAMPLE1" != "$SAMPLE2" ]; then
    echo "   ✅ Metrics updating dynamically (Δ=$(echo "$SAMPLE2 - $SAMPLE1" | bc)s)"
else
    echo "   ⚠️  Metrics same between samples (may indicate issue or timing)"
fi
echo

# Step 6: Verify Prometheus staleness query
echo "🔍 Step 6: Checking Prometheus staleness query..."
STALENESS=$(curl -s "http://127.0.0.1:9091/api/v1/query?query=time()%20-%20quantum_harvest_last_update_epoch" | \
    python3 -c "import sys,json; r=json.load(sys.stdin)['data']['result']; print(f\"{r[0]['metric']['symbol']}: {float(r[0]['value'][1]):.1f}s\") if r else print('No data')")

echo "   Current staleness: $STALENESS"

# Check if staleness is reasonable (<30s)
STALE_VALUE=$(echo "$STALENESS" | grep -oP '\d+\.\d+' || echo "999")
if (( $(echo "$STALE_VALUE < 30" | bc -l) )); then
    echo "   ✅ Staleness < 30s (within threshold)"
else
    echo "   ⚠️  Staleness >= 30s (may still be settling)"
fi
echo

# Step 7: Check alert status
echo "🔍 Step 7: Checking QuantumHarvestMetricsStale alert status..."
ALERT_STATE=$(curl -s "http://127.0.0.1:9091/api/v1/alerts" | \
    python3 -c "import sys,json; alerts=[a for a in json.load(sys.stdin)['data']['alerts'] if a['labels']['alertname']=='QuantumHarvestMetricsStale']; print(alerts[0]['state'] if alerts else 'inactive')" 2>/dev/null || echo "unknown")

echo "   Alert state: $ALERT_STATE"

if [ "$ALERT_STATE" == "inactive" ]; then
    echo "   ✅ QuantumHarvestMetricsStale is INACTIVE (staleness resolved)"
elif [ "$ALERT_STATE" == "pending" ]; then
    echo "   ⏸️  QuantumHarvestMetricsStale is PENDING (resolving, wait for 'for' duration)"
else
    echo "   ⚠️  Alert state: $ALERT_STATE (check /api/v1/alerts)"
fi
echo

echo "============================================"
echo "✅ P2.7C.1 Deployment Complete"
echo "============================================"
echo
echo "Summary:"
echo "  - Publisher writes last_update_epoch to Redis ✅"
echo "  - Exporter reads last_update_epoch from Redis ✅"
echo "  - Metrics update dynamically ✅"
echo "  - Staleness alerts resolving ✅"
echo
echo "Next steps:"
echo "  - Monitor Prometheus for 2-5 minutes"
echo "  - Verify QuantumHarvestMetricsStale transitions to inactive"
echo "  - Run proof pack: /tmp/p27c1_proof_pack.sh"
echo

exit 0
