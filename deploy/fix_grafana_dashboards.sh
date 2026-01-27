#!/bin/bash
set -e

echo "=== Fixing Quantum Dashboards (Hybrid Approach) ==="

# Fix RL Shadow Monitoring
echo ""
echo "1/3 Fixing RL Shadow Monitoring..."
curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/rl-shadow-clean | \
python3 /tmp/fix_dashboard.py > /tmp/fixed_rl.json 2>&1
curl -s -u admin:admin -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" -d @/tmp/fixed_rl.json | \
  python3 -c "import sys,json; r=json.load(sys.stdin); print(f'✅ RL Shadow: {r[\"status\"]} v{r[\"version\"]}')"

# Fix Quantum Safety Telemetry
echo ""
echo "2/3 Fixing Quantum Safety Telemetry..."
curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/7288b2bf-0d94-404d-8dda-934fb3ab88cc | \
python3 /tmp/fix_dashboard.py > /tmp/fixed_safety.json 2>&1
curl -s -u admin:admin -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" -d @/tmp/fixed_safety.json | \
  python3 -c "import sys,json; r=json.load(sys.stdin); print(f'✅ Safety Telemetry: {r[\"status\"]} v{r[\"version\"]}')"

# Fix Redis Metrics
echo ""
echo "3/3 Fixing Redis Metrics..."
curl -s -u admin:admin http://localhost:3000/api/dashboards/uid/redis-clean | \
python3 /tmp/fix_dashboard.py > /tmp/fixed_redis.json 2>&1
curl -s -u admin:admin -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" -d @/tmp/fixed_redis.json | \
  python3 -c "import sys,json; r=json.load(sys.stdin); print(f'✅ Redis Metrics: {r[\"status\"]} v{r[\"version\"]}')"

echo ""
echo "=== Dashboard Fix Complete ==="
echo ""
echo "Fixed dashboards:"
echo "  • RL Shadow Monitoring: https://quantumfond.com/grafana/d/rl-shadow-clean"
echo "  • Quantum Safety Telemetry: https://quantumfond.com/grafana/d/7288b2bf-0d94-404d-8dda-934fb3ab88cc"
echo "  • Redis Metrics: https://quantumfond.com/grafana/d/redis-clean"
echo ""
echo "Dashboards kept as-is (likely working):"
echo "  • System Overview - Systemd"
echo "  • Infrastructure - Systemd"
echo "  • Quantum Services"
echo ""
echo "Test dashboard:"
echo "  • Quantum Metrics Test: https://quantumfond.com/grafana/d/d31466b8-8cd2-44f9-a02b-82a08de7c9d8"
