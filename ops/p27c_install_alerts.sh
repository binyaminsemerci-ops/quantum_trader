#!/bin/bash
# P2.7C: Install Harvest Alert Rules and Generate Proof Pack
# Purpose: Deploy Prometheus alert rules for harvest observability
# Author: Quantum Trader Operations
# Date: 2026-01-23

set -euo pipefail

echo "============================================"
echo "P2.7C: Harvest Alert Rules Installation"
echo "============================================"
echo ""

# Step 1: Create rules directory if not exists
echo "📁 Step 1: Ensuring rules directory exists..."
mkdir -p /etc/prometheus/rules
echo "   ✅ Directory ready: /etc/prometheus/rules"
echo ""

# Step 2: Copy rules file
echo "📋 Step 2: Installing alert rules file..."
RULES_SOURCE="/tmp/quantum_harvest_alerts.yml"
RULES_TARGET="/etc/prometheus/rules/quantum_harvest_alerts.yml"

if [ ! -f "$RULES_SOURCE" ]; then
    echo "   ❌ ERROR: Source file not found: $RULES_SOURCE"
    echo "   Please upload quantum_harvest_alerts.yml to /tmp/ first"
    exit 1
fi

cp "$RULES_SOURCE" "$RULES_TARGET"
chmod 644 "$RULES_TARGET"
chown prometheus:prometheus "$RULES_TARGET" 2>/dev/null || true

echo "   ✅ Rules file installed: $RULES_TARGET"
echo ""

# Step 3: Validate rules
echo "🔍 Step 3: Validating alert rules..."

# Try promtool if available
if command -v promtool >/dev/null 2>&1; then
    echo "   Using promtool for validation..."
    if promtool check rules "$RULES_TARGET"; then
        echo "   ✅ Rules syntax valid (promtool)"
    else
        echo "   ❌ ERROR: Rules syntax invalid!"
        exit 2
    fi
else
    echo "   ⚠️  promtool not available, using Python YAML check..."
    python3 << 'PYEOF'
import yaml
import sys

try:
    with open('/etc/prometheus/rules/quantum_harvest_alerts.yml') as f:
        rules = yaml.safe_load(f)
    
    # Basic structure validation
    if 'groups' not in rules:
        print("   ❌ ERROR: Missing 'groups' key in rules file")
        sys.exit(2)
    
    groups = rules['groups']
    if not isinstance(groups, list) or len(groups) == 0:
        print("   ❌ ERROR: 'groups' must be a non-empty list")
        sys.exit(2)
    
    # Check first group
    group = groups[0]
    if 'name' not in group or 'rules' not in group:
        print("   ❌ ERROR: Group missing 'name' or 'rules'")
        sys.exit(2)
    
    print(f"   ✅ Rules syntax valid (Python YAML check)")
    print(f"      Group: {group['name']}")
    print(f"      Rules count: {len(group['rules'])}")
    
except Exception as e:
    print(f"   ❌ ERROR: YAML validation failed: {e}")
    sys.exit(2)
PYEOF
fi
echo ""

# Step 4: Reload Prometheus
echo "🔄 Step 4: Reloading Prometheus..."

# Try reload first (graceful)
if systemctl reload prometheus 2>/dev/null; then
    echo "   ✅ Prometheus reloaded (graceful)"
else
    echo "   ⚠️  Reload not supported, restarting..."
    systemctl restart prometheus
    echo "   ✅ Prometheus restarted"
fi

sleep 3

# Check Prometheus is active
if systemctl is-active prometheus >/dev/null 2>&1; then
    echo "   ✅ Prometheus is active"
else
    echo "   ❌ ERROR: Prometheus is not active!"
    systemctl status prometheus --no-pager
    exit 3
fi
echo ""

# Step 5: Verify rules loaded
echo "✅ Step 5: Verifying rules loaded in Prometheus..."
PROM_PORT=9091  # Adjust if needed

RULES_CHECK=$(curl -s "http://127.0.0.1:${PROM_PORT}/api/v1/rules" | \
  python3 -c "import sys,json; d=json.load(sys.stdin); groups=[g for g in d.get('data',{}).get('groups',[]) if g.get('name')=='quantum_harvest_alerts']; print(len(groups))")

if [ "$RULES_CHECK" -gt 0 ]; then
    echo "   ✅ Rules group 'quantum_harvest_alerts' found in Prometheus"
    
    # Count alert rules
    ALERT_COUNT=$(curl -s "http://127.0.0.1:${PROM_PORT}/api/v1/rules" | \
      python3 -c "import sys,json; d=json.load(sys.stdin); groups=[g for g in d.get('data',{}).get('groups',[]) if g.get('name')=='quantum_harvest_alerts']; print(len(groups[0]['rules']) if groups else 0)")
    
    echo "   📊 Alert rules loaded: $ALERT_COUNT"
else
    echo "   ❌ ERROR: Rules group not found in Prometheus!"
    echo "   Check Prometheus logs: journalctl -u prometheus -n 50"
    exit 4
fi
echo ""

echo "============================================"
echo "✅ P2.7C Installation Complete"
echo "============================================"
echo ""
echo "Alert Rules Deployed:"
echo "  1. QuantumHarvestKillScoreHigh (kill_score >= 0.6)"
echo "  2. QuantumHarvestKillScoreCritical (kill_score >= 0.8)"
echo "  3. QuantumHarvestMetricsStale (no updates >30s)"
echo "  4. QuantumHarvestExporterDown (scrape failing)"
echo ""
echo "Next: Run proof pack generation"
echo "  ./p27c_proof_pack.sh > /tmp/p27c_proof_output.txt"
