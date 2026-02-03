#!/bin/bash
# observability/obs_status.sh - Check Quantum Trader Observability Stack Status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🔍 Quantum Trader Observability Stack Status"
echo "=============================================="
echo ""

# Container Status
echo "📦 Container Status:"
echo "--------------------------------------------"
docker compose -f docker-compose.observability.yml ps
echo ""

# Service Health Checks
echo "🏥 Service Health:"
echo "--------------------------------------------"

# Prometheus
PROM_STATUS="❌ DOWN"
if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
    PROM_STATUS="✅ HEALTHY"
    PROM_VERSION=$(curl -s http://localhost:9090/api/v1/status/buildinfo 2>/dev/null | jq -r '.data.version' 2>/dev/null || echo "unknown")
    PROM_UPTIME=$(curl -s http://localhost:9090/api/v1/query?query=time\(\)-process_start_time_seconds 2>/dev/null | jq -r '.data.result[0].value[1]' 2>/dev/null | awk '{printf "%.0f", $1/3600}' || echo "N/A")
    echo "Prometheus:        $PROM_STATUS (v$PROM_VERSION, uptime: ${PROM_UPTIME}h)"
else
    echo "Prometheus:        $PROM_STATUS"
fi

# Grafana
GRAFANA_STATUS="❌ DOWN"
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    GRAFANA_STATUS="✅ HEALTHY"
    GRAFANA_VERSION=$(curl -s http://localhost:3000/api/health 2>/dev/null | jq -r '.version' 2>/dev/null || echo "unknown")
    echo "Grafana:           $GRAFANA_STATUS (v$GRAFANA_VERSION)"
else
    echo "Grafana:           $GRAFANA_STATUS"
fi

# Alertmanager
ALERT_STATUS="❌ DOWN"
if curl -sf http://localhost:9093/-/healthy > /dev/null 2>&1; then
    ALERT_STATUS="✅ HEALTHY"
    ALERT_VERSION=$(curl -s http://localhost:9093/api/v1/status 2>/dev/null | jq -r '.data.versionInfo.version' 2>/dev/null || echo "unknown")
    echo "Alertmanager:      $ALERT_STATUS (v$ALERT_VERSION)"
else
    echo "Alertmanager:      $ALERT_STATUS"
fi

# Node Exporter
NODE_STATUS="❌ DOWN"
if curl -sf http://localhost:9100/metrics > /dev/null 2>&1; then
    NODE_STATUS="✅ UP"
fi
echo "Node Exporter:     $NODE_STATUS"

# cAdvisor
CADVISOR_STATUS="❌ DOWN"
if curl -sf http://localhost:8080/metrics > /dev/null 2>&1; then
    CADVISOR_STATUS="✅ UP"
fi
echo "cAdvisor:          $CADVISOR_STATUS"

# Redis Exporter
REDIS_EXP_STATUS="❌ DOWN"
if curl -sf http://localhost:9121/metrics > /dev/null 2>&1; then
    REDIS_EXP_STATUS="✅ UP"
    REDIS_UP=$(curl -s http://localhost:9121/metrics 2>/dev/null | grep '^redis_up ' | awk '{print $2}')
    if [[ "$REDIS_UP" == "1" ]]; then
        echo "Redis Exporter:    $REDIS_EXP_STATUS (Redis: ✅ CONNECTED)"
    else
        echo "Redis Exporter:    $REDIS_EXP_STATUS (Redis: ❌ DISCONNECTED)"
    fi
else
    echo "Redis Exporter:    $REDIS_EXP_STATUS"
fi

echo ""

# Prometheus Targets
if curl -sf http://localhost:9090/api/v1/targets > /dev/null 2>&1; then
    echo "🎯 Prometheus Scrape Targets:"
    echo "--------------------------------------------"
    curl -s http://localhost:9090/api/v1/targets 2>/dev/null | jq -r '.data.activeTargets[] | "\(.labels.job | .[0:30]): \(.health | ascii_upcase) (last: \(.lastScrape // "never"))"' 2>/dev/null | column -t
    echo ""
    
    TOTAL_TARGETS=$(curl -s http://localhost:9090/api/v1/targets 2>/dev/null | jq '.data.activeTargets | length')
    UP_TARGETS=$(curl -s http://localhost:9090/api/v1/targets 2>/dev/null | jq '[.data.activeTargets[] | select(.health=="up")] | length')
    echo "Summary: $UP_TARGETS/$TOTAL_TARGETS targets UP"
    echo ""
fi

# Alert Rules
if curl -sf http://localhost:9090/api/v1/rules > /dev/null 2>&1; then
    echo "⚠️  Alert Rules:"
    echo "--------------------------------------------"
    TOTAL_RULES=$(curl -s http://localhost:9090/api/v1/rules 2>/dev/null | jq '[.data.groups[].rules[]] | length')
    FIRING_ALERTS=$(curl -s http://localhost:9090/api/v1/rules 2>/dev/null | jq '[.data.groups[].rules[] | select(.state=="firing")] | length')
    PENDING_ALERTS=$(curl -s http://localhost:9090/api/v1/rules 2>/dev/null | jq '[.data.groups[].rules[] | select(.state=="pending")] | length')
    
    echo "Total Rules: $TOTAL_RULES"
    echo "Firing:      $FIRING_ALERTS"
    echo "Pending:     $PENDING_ALERTS"
    
    if [ "$FIRING_ALERTS" -gt 0 ]; then
        echo ""
        echo "🚨 FIRING ALERTS:"
        curl -s http://localhost:9090/api/v1/rules 2>/dev/null | jq -r '.data.groups[].rules[] | select(.state=="firing") | "  - \(.name) (severity: \(.labels.severity // "unknown"))"'
    fi
    echo ""
fi

# Grafana Dashboards
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "📊 Grafana Dashboards:"
    echo "--------------------------------------------"
    echo "Dashboard are provisioned from:"
    echo "  observability/grafana/dashboards/"
    echo ""
    echo "Expected dashboards:"
    echo "  1. Quantum Trader - Overview"
    echo "  2. Quantum Trader - Execution"
    echo "  3. Quantum Trader - Infrastructure"
    echo "  4. Quantum Trader - Redis & Postgres"
    echo ""
fi

# Access URLs
echo "🔗 Access URLs:"
echo "--------------------------------------------"
echo "Grafana:           http://localhost:3000"
echo "Prometheus:        http://localhost:9090"
echo "Alertmanager:      http://localhost:9093"
echo "Node Exporter:     http://localhost:9100/metrics"
echo "cAdvisor:          http://localhost:8080"
echo "Redis Exporter:    http://localhost:9121/metrics"
echo ""
echo "🔐 Grafana Login:"
echo "   Username: admin"
echo "   Password: ${GRAFANA_ADMIN_PASSWORD:-quantum2026secure}"
echo ""

# Overall Status
echo "📋 Overall Status:"
echo "--------------------------------------------"
if [[ "$PROM_STATUS" == *"HEALTHY"* ]] && [[ "$GRAFANA_STATUS" == *"HEALTHY"* ]] && [[ "$ALERT_STATUS" == *"HEALTHY"* ]]; then
    echo "✅ Observability stack is fully operational!"
else
    echo "⚠️  Some services are not healthy. Check logs with:"
    echo "   docker compose -f docker-compose.observability.yml logs [service_name]"
fi
