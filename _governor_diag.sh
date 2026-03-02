#!/bin/bash
echo "=== GOVERNOR SERVICE LOGS (last 30 lines) ==="
journalctl -u quantum-governor.service -n 30 --no-pager 2>/dev/null

echo ""
echo "=== GOVERNOR SERVICE ENV ==="
systemctl show quantum-governor.service --property=Environment --no-pager 2>/dev/null
cat /etc/systemd/system/quantum-governor.service 2>/dev/null | head -30

echo ""
echo "=== META-REGIME (DEAD) - why? ==="
journalctl -u quantum-meta-regime.service -n 20 --no-pager 2>/dev/null

echo ""
echo "=== FEATURES: which symbols actually have data ==="
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT; do
  count=$(redis-cli XLEN quantum:stream:features.$sym 2>/dev/null)
  echo "  features.$sym => $count"
done

echo ""
echo "=== auto_permit_p33.py check ==="
cat /opt/quantum/scripts/auto_permit_p33.py 2>/dev/null | head -40

echo ""
echo "=== CRON: permit jobs ==="
crontab -l 2>/dev/null | grep -iE 'permit|p33|governor'
ls /etc/cron.d/ 2>/dev/null | head -10
cat /etc/cron.d/quantum* 2>/dev/null 2>/dev/null | grep -iE 'permit|p33|governor'

echo ""
echo "=== GOVERNOR SOURCE: what sets permit:governor ==="
grep -n 'permit:governor' /opt/quantum/microservices/apply_layer/main.py 2>/dev/null | head -10

echo ""
echo "=== REGIME KEY in Redis ==="
redis-cli GET quantum:meta:regime 2>/dev/null
redis-cli GET quantum:regime:current 2>/dev/null
redis-cli KEYS 'quantum:*regime*' 2>/dev/null

echo ""
echo "=== SIGNAL risk_context values (are they using initialization?) ==="
redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 10 2>/dev/null | grep -A1 'risk_context'

echo ""
echo "=== APPLY LAYER LOGS (last 20) ==="
journalctl -u quantum-apply-layer.service -n 20 --no-pager 2>/dev/null | tail -25
