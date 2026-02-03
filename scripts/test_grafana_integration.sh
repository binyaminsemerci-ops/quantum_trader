#!/bin/bash
# Quick test script for Grafana integration

echo "============================================"
echo "🧪 Testing Grafana Integration in Dashboard"
echo "============================================"
echo ""

# Test 1: Check Grafana is running
echo "1️⃣  Testing Grafana container..."
docker ps --filter name=grafana --format "{{.Names}}: {{.Status}}"
echo ""

# Test 2: Check Grafana health
echo "2️⃣  Testing Grafana health endpoint..."
curl -s http://localhost:3001/api/health | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"Database: {data['database']}, Version: {data['version']}\")"
echo ""

# Test 3: Check external Grafana access
echo "3️⃣  Testing external Grafana access..."
curl -I https://app.quantumfond.com/grafana/ 2>&1 | grep "HTTP\|200"
echo ""

# Test 4: Check dashboard frontend
echo "4️⃣  Testing dashboard frontend..."
docker ps --filter name=dashboard_frontend --format "{{.Names}}: {{.Status}}"
echo ""

# Test 5: Check main dashboard access
echo "5️⃣  Testing main dashboard access..."
curl -I https://app.quantumfond.com 2>&1 | grep "HTTP\|200"
echo ""

echo "============================================"
echo "✅ All tests completed!"
echo "============================================"
echo ""
echo "🌐 Access URLs:"
echo "   Main Dashboard: https://app.quantumfond.com"
echo "   System Page:    https://app.quantumfond.com/system"
echo "   Grafana:        https://app.quantumfond.com/grafana"
echo ""
echo "📊 Dashboard Tabs (System page):"
echo "   Tab 1: 📊 Overview - System health gauges"
echo "   Tab 2: 📈 Performance Metrics - P1-C Grafana (NEW!)"
echo "   Tab 3: 📝 Log Analysis - P1-B Grafana (NEW!)"
echo ""
echo "🔑 Grafana Admin Access:"
echo "   URL: https://app.quantumfond.com/grafana/login"
echo "   User: admin"
echo "   Pass: quantum2026secure"
echo ""
