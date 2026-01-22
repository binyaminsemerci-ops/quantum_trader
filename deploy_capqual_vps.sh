#!/bin/bash
# P0.CAP+QUAL VPS Deployment - Jan 19 2026
set -e

echo "=== PHASE 0: RECONNAISSANCE ==="
date -u
hostname
echo ""

echo "Service status:"
systemctl is-active quantum-ai-strategy-router.service || true
echo ""

echo "Finding router path..."
ROUTER_PY=$(systemctl show -p ExecStart --value quantum-ai-strategy-router.service | grep -oP '/[^ ]*ai_strategy_router\.py' | head -1)
if [ -z "$ROUTER_PY" ]; then
    echo "ERROR: Could not extract router path from ExecStart"
    systemctl cat quantum-ai-strategy-router.service | grep ExecStart
    exit 1
fi
echo "ROUTER_PY=$ROUTER_PY"
ls -lh "$ROUTER_PY"
echo ""

echo "=== PHASE 1: PRE-PROOF ==="
echo "Setting capacity key..."
redis-cli SET quantum:state:open_orders 0
redis-cli GET quantum:state:open_orders
echo ""

echo "BEFORE: Publish count (last 2min)..."
BEFORE_COUNT=$(journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -ci "published" || echo "0")
echo "Publishes: $BEFORE_COUNT"
echo ""

echo "BEFORE: Governor today..."
TODAY=$(date -u +%Y%m%d)
echo "Date: $TODAY"
BEFORE_GOV=$(redis-cli GET quantum:governor:daily_trades:$TODAY || echo "NULL")
echo "Governor: $BEFORE_GOV"
echo ""

echo "=== PHASE 2: BACKUP ==="
DIR=/tmp/p0_capqual_deploy_$(date -u +%Y%m%d_%H%M%S)
mkdir -p $DIR
echo "Backup dir: $DIR"

cp -a "$ROUTER_PY" "$DIR/router.py.backup"
echo "Backed up: $ROUTER_PY"
ls -lh "$DIR/router.py.backup"
echo ""

if [ -f /tmp/ai_strategy_router.py ]; then
    cp -a /tmp/ai_strategy_router.py "$DIR/router.py.candidate"
    echo "Candidate preserved: /tmp/ai_strategy_router.py"
    ls -lh "$DIR/router.py.candidate"
else
    echo "WARNING: /tmp/ai_strategy_router.py not found"
    exit 1
fi
echo ""

echo "=== PHASE 3: DEPLOY + VALIDATE ==="
if [ ! -f /tmp/ai_strategy_router.py ]; then
    echo "ERROR: Candidate /tmp/ai_strategy_router.py missing!"
    exit 1
fi

echo "Installing candidate to $ROUTER_PY..."
cp -a /tmp/ai_strategy_router.py "$ROUTER_PY"
ls -lh "$ROUTER_PY"
echo ""

echo "Validating syntax..."
if python3 -m py_compile "$ROUTER_PY"; then
    echo "✅ Syntax valid"
else
    echo "❌ SYNTAX ERROR - Rolling back..."
    cp -a "$DIR/router.py.backup" "$ROUTER_PY"
    systemctl restart quantum-ai-strategy-router.service
    echo "Rollback complete. Last 80 lines:"
    journalctl -u quantum-ai-strategy-router.service -n 80 --no-pager
    exit 1
fi
echo ""

echo "=== PHASE 4: RESTART + HEALTH ==="
echo "Restarting service..."
systemctl restart quantum-ai-strategy-router.service
sleep 3

echo "Service status:"
systemctl --no-pager -l status quantum-ai-strategy-router.service | head -40
echo ""

ACTIVE=$(systemctl is-active quantum-ai-strategy-router.service)
if [ "$ACTIVE" != "active" ]; then
    echo "❌ SERVICE FAILED - Rolling back..."
    cp -a "$DIR/router.py.backup" "$ROUTER_PY"
    systemctl restart quantum-ai-strategy-router.service
    echo "Rollback complete. Error logs:"
    journalctl -u quantum-ai-strategy-router.service -n 80 --no-pager
    exit 1
fi
echo "✅ Service active"
echo ""

echo "Evidence from last 2 minutes:"
journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -E "P0.CAPQUAL|CAPACITY_|TARGET_|BEST_OF|SIZE_BOOST|published|ERROR|Traceback" | tail -160 || true
echo ""

echo "=== PHASE 5: WAIT + AFTER-PROOF ==="
echo "Waiting 30s for metrics..."
sleep 30
echo ""

echo "AFTER: Publish count (last 2min)..."
AFTER_COUNT=$(journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -ci "published" || echo "0")
echo "Publishes: $AFTER_COUNT"
echo "CHANGE: $BEFORE_COUNT → $AFTER_COUNT"

if [ "$AFTER_COUNT" -lt "$BEFORE_COUNT" ]; then
    REDUCTION=$(echo "scale=1; 100 - ($AFTER_COUNT * 100.0 / $BEFORE_COUNT)" | bc 2>/dev/null || echo "N/A")
    echo "REDUCTION: ${REDUCTION}%"
fi
echo ""

echo "AFTER: Governor today..."
AFTER_GOV=$(redis-cli GET quantum:governor:daily_trades:$TODAY || echo "NULL")
echo "Governor: $AFTER_GOV"
echo "CHANGE: $BEFORE_GOV → $AFTER_GOV"
echo ""

echo "Dedup keys:"
DEDUP_COUNT=$(redis-cli --scan --pattern "quantum:dedup:trade_intent:*" | wc -l)
echo "Count: $DEDUP_COUNT"
echo ""

echo "=== DEPLOYMENT SUCCESS ==="
echo "Router: $ROUTER_PY"
echo "Backup: $DIR/router.py.backup"
echo "Status: ✅ ACTIVE"
echo ""

echo "Key CAP+QUAL logs (last 60 lines):"
journalctl -u quantum-ai-strategy-router.service --since "3 minutes ago" --no-pager | grep -E "CAPQUAL|CAPACITY|TARGET|BEST_OF|SIZE_BOOST" | tail -60 || echo "(No CAP+QUAL logs yet - may need more time)"
echo ""

echo "Monitor: journalctl -u quantum-ai-strategy-router.service -f"
