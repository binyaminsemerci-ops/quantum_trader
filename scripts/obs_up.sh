#!/bin/bash
# observability/obs_up.sh - Deploy Quantum Trader Observability Stack
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Deploying Quantum Trader Observability Stack..."
echo ""

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Navigate to project root
cd "$PROJECT_ROOT"

# Deploy the stack
echo "📦 Starting observability services..."
docker compose -f docker-compose.observability.yml up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to become healthy (max 60s)..."
sleep 10

TIMEOUT=60
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    echo -n "."
    
    # Check Prometheus
    PROM_HEALTH=$(docker compose -f docker-compose.observability.yml exec -T prometheus wget -qO- http://localhost:9090/-/healthy 2>/dev/null || echo "unhealthy")
    
    # Check Grafana
    GRAFANA_HEALTH=$(docker compose -f docker-compose.observability.yml exec -T grafana wget -qO- http://localhost:3000/api/health 2>/dev/null || echo "unhealthy")
    
    # Check Alertmanager
    ALERT_HEALTH=$(docker compose -f docker-compose.observability.yml exec -T alertmanager wget -qO- http://localhost:9093/-/healthy 2>/dev/null || echo "unhealthy")
    
    if [[ "$PROM_HEALTH" == "Prometheus is Healthy." ]] && [[ "$GRAFANA_HEALTH" == *"\"database\": \"ok\""* ]] && [[ "$ALERT_HEALTH" == "Alertmanager is Healthy." ]]; then
        echo ""
        echo "✅ All core services healthy!"
        break
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo "⚠️  Timeout waiting for services. Checking status..."
fi

echo ""
echo "📊 Observability Stack Status:"
echo "================================"
docker compose -f docker-compose.observability.yml ps

echo ""
echo "🔍 Service Health:"
echo "================================"

# Prometheus
if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus: http://localhost:9090 (HEALTHY)"
    PROM_TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length' 2>/dev/null || echo "N/A")
    echo "   → Scraping $PROM_TARGETS targets"
else
    echo "❌ Prometheus: http://localhost:9090 (DOWN)"
fi

# Grafana
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana: http://localhost:3000 (HEALTHY)"
    echo "   → Login: admin / ${GRAFANA_ADMIN_PASSWORD:-quantum2026secure}"
else
    echo "❌ Grafana: http://localhost:3000 (DOWN)"
fi

# Alertmanager
if curl -sf http://localhost:9093/-/healthy > /dev/null 2>&1; then
    echo "✅ Alertmanager: http://localhost:9093 (HEALTHY)"
else
    echo "❌ Alertmanager: http://localhost:9093 (DOWN)"
fi

# Node Exporter
if curl -sf http://localhost:9100/metrics > /dev/null 2>&1; then
    echo "✅ Node Exporter: http://localhost:9100/metrics (UP)"
else
    echo "⚠️  Node Exporter: http://localhost:9100/metrics (DOWN)"
fi

# cAdvisor
if curl -sf http://localhost:8080/metrics > /dev/null 2>&1; then
    echo "✅ cAdvisor: http://localhost:8080/metrics (UP)"
else
    echo "⚠️  cAdvisor: http://localhost:8080/metrics (DOWN)"
fi

# Redis Exporter
if curl -sf http://localhost:9121/metrics > /dev/null 2>&1; then
    echo "✅ Redis Exporter: http://localhost:9121/metrics (UP)"
else
    echo "⚠️  Redis Exporter: http://localhost:9121/metrics (DOWN)"
fi

echo ""
echo "📈 Prometheus Targets:"
echo "================================"
curl -s http://localhost:9090/api/v1/targets 2>/dev/null | jq -r '.data.activeTargets[] | "\(.labels.job): \(.health) (\(.lastScrape))"' 2>/dev/null || echo "Could not fetch targets"

echo ""
echo "🎯 Quick Access URLs:"
echo "================================"
echo "Grafana Dashboard:   http://localhost:3000"
echo "Prometheus:          http://localhost:9090"
echo "Alertmanager:        http://localhost:9093"
echo ""
echo "🔐 Grafana Credentials:"
echo "   Username: admin"
echo "   Password: ${GRAFANA_ADMIN_PASSWORD:-quantum2026secure}"
echo ""
echo "✅ Observability stack deployment complete!"
