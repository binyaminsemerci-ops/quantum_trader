#!/bin/bash
# P1-B: Alert Testing Script
# Simulates various error conditions to test Prometheus alerts

echo "========================================"
echo "P1-B: ALERT FIRING TEST"
echo "========================================"
echo

# Test 1: High Error Rate Alert
echo "=== Test 1: HighErrorRate Alert ==="
echo "Injecting 100 ERROR logs to trigger >10 errors/sec threshold..."

for i in {1..100}; do
    docker exec quantum_auto_executor python3 -c "import logging; logging.basicConfig(level=logging.ERROR); logging.error('TEST_ERROR_$i: Simulated high error rate')" 2>/dev/null &
done
wait

echo "✅ Injected 100 errors in ~2 seconds"
echo "Waiting 3 minutes for Prometheus to evaluate alert..."
sleep 180

echo
echo "=== Checking Alert Status ==="
curl -s "http://localhost:9090/api/v1/alerts" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    alerts = d.get('data', {}).get('alerts', [])
    firing = [a for a in alerts if a['state'] == 'firing' and 'p1b' in a['labels'].get('alertname', '').lower()]
    if firing:
        print(f'🔥 FIRING ALERTS ({len(firing)}):')
        for a in firing:
            print(f\"  - {a['labels']['alertname']}: {a['annotations'].get('summary', 'N/A')}\")
    else:
        print('ℹ️  No P1-B alerts currently firing')
        print('   (Alert may need more time to fire or threshold not reached)')
except Exception as e:
    print(f'Error parsing alerts: {e}')
"

echo
echo "=== Alertmanager Status ==="
curl -s "http://localhost:9093/api/v2/alerts" | python3 -c "
import sys, json
try:
    alerts = json.load(sys.stdin)
    if alerts:
        print(f'📬 Alertmanager has {len(alerts)} active alerts')
        for a in alerts:
            labels = a.get('labels', {})
            print(f\"  - {labels.get('alertname', 'Unknown')}: {a.get('status', {}).get('state', 'unknown')}\")
    else:
        print('ℹ️  No alerts in Alertmanager')
except Exception as e:
    print(f'Error: {e}')
"

echo
echo "========================================"
echo "Test 2: Container Restart Alert"
echo "========================================"
echo "To test ContainerRestartLoop alert:"
echo "  docker restart quantum_auto_executor quantum_auto_executor quantum_auto_executor"
echo "  (Restart 3+ times within 10 minutes)"
echo

echo "========================================"
echo "Test 3: Loki Down Alert"
echo "========================================"
echo "To test LokiDown alert:"
echo "  docker stop quantum_loki"
echo "  (Wait 2+ minutes for alert to fire)"
echo "  docker start quantum_loki  # To clear alert"
echo

echo "========================================"
echo "ALERT TEST COMPLETE"
echo "========================================"
echo "Check Grafana Alerting UI: http://46.224.116.254:3000/alerting/list"
echo "Check Prometheus Alerts: http://46.224.116.254:9090/alerts"
echo "Check Alertmanager: http://46.224.116.254:9093"
