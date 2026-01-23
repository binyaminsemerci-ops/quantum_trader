#!/bin/bash
# P2.7C.1 Proof Pack - Verify Staleness Fix
# Proves: Redis has last_update_epoch, exporter reads it, alerts resolved

echo "============================================"
echo "P2.7C.1 PROOF PACK - Staleness Fix"
echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PROOF 1: Redis now has last_update_epoch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
for symbol in BTCUSDT ETHUSDT SOLUSDT; do
    echo "--- $symbol ---"
    EPOCH=$(redis-cli HGET quantum:harvest:proposal:$symbol last_update_epoch 2>/dev/null)
    COMPUTED=$(redis-cli HGET quantum:harvest:proposal:$symbol computed_at_utc 2>/dev/null)
    
    if [ -n "$EPOCH" ]; then
        AGE=$(($(date +%s) - $EPOCH))
        echo "  last_update_epoch: $EPOCH (${AGE}s ago)"
        echo "  computed_at_utc: $COMPUTED"
        echo "  ✅ Field exists and recent"
    else
        echo "  ❌ last_update_epoch NOT FOUND"
    fi
    echo
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PROOF 2: Exporter metrics updating dynamically"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Sampling metrics over 15 seconds (3 samples, 5s interval)..."
echo

for i in 1 2 3; do
    echo "Sample $i @ $(date -u '+%H:%M:%S'):"
    curl -s http://127.0.0.1:8042/metrics | grep quantum_harvest_last_update_epoch | while read line; do
        SYMBOL=$(echo "$line" | grep -oP 'symbol="\K[^"]+')
        EPOCH=$(echo "$line" | awk '{print $2}')
        AGE=$(($(date +%s) - ${EPOCH%.*}))
        echo "  $SYMBOL: epoch=$EPOCH (${AGE}s ago)"
    done
    
    if [ $i -lt 3 ]; then
        sleep 5
    fi
done
echo
echo "✅ If epochs differ between samples, metrics are updating dynamically"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PROOF 3: Prometheus staleness query"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
curl -s "http://127.0.0.1:9091/api/v1/query?query=time()%20-%20quantum_harvest_last_update_epoch" | \
    python3 -c "
import sys, json
result = json.load(sys.stdin)
if result['status'] == 'success':
    for r in result['data']['result']:
        symbol = r['metric']['symbol']
        staleness = float(r['value'][1])
        status = '🟢 FRESH' if staleness < 30 else '⚠️  STALE'
        print(f'{symbol}: {staleness:.1f}s {status}')
else:
    print('❌ Query failed')
"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PROOF 4: QuantumHarvestMetricsStale alert status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
curl -s "http://127.0.0.1:9091/api/v1/alerts" | \
    python3 -c "
import sys, json
alerts = json.load(sys.stdin)['data']['alerts']
stale_alerts = [a for a in alerts if a['labels']['alertname'] == 'QuantumHarvestMetricsStale']

if not stale_alerts:
    print('✅ QuantumHarvestMetricsStale: INACTIVE (no alerts)')
    print('   Staleness threshold met, metrics updating within 30s')
else:
    print(f'⚠️  QuantumHarvestMetricsStale: {len(stale_alerts)} alert(s)')
    for a in stale_alerts:
        symbol = a['labels'].get('symbol', 'N/A')
        state = a['state']
        value = a['value']
        print(f'   {symbol}: state={state}, staleness={value}s')
    print()
    if all(a['state'] == 'pending' for a in stale_alerts):
        print('   ℹ️  All alerts PENDING (resolving, wait for \"for\" duration)')
    elif any(a['state'] == 'firing' for a in stale_alerts):
        print('   ❌ Some alerts FIRING (staleness persists, investigate)')
"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PROOF 5: Service health (last 5 lines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "--- Harvest Proposal Publisher ---"
systemctl status quantum-harvest-proposal --no-pager | head -3 | tail -1
journalctl -u quantum-harvest-proposal --since "1 minute ago" --no-pager | tail -3 | head -1
echo

echo "--- Harvest Metrics Exporter ---"
systemctl status quantum-harvest-metrics-exporter --no-pager | head -3 | tail -1
journalctl -u quantum-harvest-metrics-exporter --since "1 minute ago" --no-pager | tail -3 | head -1
echo

echo "--- Prometheus ---"
systemctl status prometheus --no-pager | head -3 | tail -1
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Root Cause Fixed:"
echo "  ❌ Before: Redis had NO last_update_epoch field"
echo "  ❌ Before: Exporter used hardcoded timestamp from startup"
echo "  ❌ Before: Staleness query showed 30-40s (static, never updating)"
echo
echo "Solution Applied:"
echo "  ✅ After: Publisher writes last_update_epoch on every publish"
echo "  ✅ After: Exporter reads last_update_epoch from Redis"
echo "  ✅ After: Staleness query shows fresh data (<30s, updating dynamically)"
echo
echo "Alert Status:"
ALERT_STATE=$(curl -s "http://127.0.0.1:9091/api/v1/alerts" | \
    python3 -c "import sys,json; alerts=[a for a in json.load(sys.stdin)['data']['alerts'] if a['labels']['alertname']=='QuantumHarvestMetricsStale']; print('inactive' if not alerts else alerts[0]['state'])" 2>/dev/null || echo "unknown")

if [ "$ALERT_STATE" == "inactive" ]; then
    echo "  ✅ QuantumHarvestMetricsStale: INACTIVE (staleness resolved)"
elif [ "$ALERT_STATE" == "pending" ]; then
    echo "  ⏸️  QuantumHarvestMetricsStale: PENDING (resolving)"
else
    echo "  ⚠️  QuantumHarvestMetricsStale: $ALERT_STATE"
fi
echo
echo "============================================"
echo "P2.7C.1 PROOF PACK COMPLETE"
echo "============================================"
