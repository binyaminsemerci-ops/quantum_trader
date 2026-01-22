#!/bin/bash
# P0.CAP+QUAL Deployment Script
# Date: 2026-01-19 12:15 UTC

set -e

echo "=== PHASE 0: RECONNAISSANCE ==="
date -u
uname -a
echo ""

echo "--- Router Service Status ---"
systemctl is-active quantum-ai-strategy-router.service || true
echo ""

echo "--- Router Service Config ---"
systemctl cat quantum-ai-strategy-router.service | grep -E "ExecStart|WorkingDirectory"
echo ""

echo "--- BEFORE: Publish Rate (last 2min) ---"
BEFORE_COUNT=$(journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -c "Trade Intent published" || echo "0")
echo "Trade Intent publishes: $BEFORE_COUNT"
echo ""

echo "--- BEFORE: Governor Daily Trades ---"
TODAY=$(date -u +%Y%m%d)
echo "Date: $TODAY"
BEFORE_GOVERNOR=$(redis-cli GET quantum:governor:daily_trades:$TODAY || echo "NULL")
echo "Governor count: $BEFORE_GOVERNOR"
redis-cli TTL quantum:governor:daily_trades:$TODAY || true
echo ""

echo "--- BEFORE: Dedup Keys ---"
BEFORE_DEDUP=$(redis-cli --scan --pattern "quantum:dedup:trade_intent:*" | wc -l)
echo "Dedup keys: $BEFORE_DEDUP"
echo ""

echo "--- BEFORE: Capacity Key ---"
BEFORE_CAPACITY=$(redis-cli GET quantum:state:open_orders || echo "NULL")
echo "Open orders: $BEFORE_CAPACITY"
echo ""

echo "=== PHASE 1: BACKUP ==="
BACKUP_DIR=/tmp/p0_capqual_$(date -u +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR/{before,after,backup,logs}
echo "Backup dir: $BACKUP_DIR"

echo "Backing up ai_strategy_router.py..."
cp -a /home/qt/quantum_trader/ai_strategy_router.py $BACKUP_DIR/backup/ai_strategy_router.py.$(date -u +%H%M%S)
echo "✅ Backup created"
echo ""

echo "=== PHASE 2: VERIFY UPLOADED FILE ==="
echo "Checking if new file exists..."
if [ ! -f /home/qt/quantum_trader/ai_strategy_router.py.new ]; then
    echo "❌ ERROR: ai_strategy_router.py.new not found!"
    echo "Please upload modified file first"
    exit 1
fi

echo "Verifying Python syntax..."
python3 -m py_compile /home/qt/quantum_trader/ai_strategy_router.py.new
if [ $? -eq 0 ]; then
    echo "✅ Syntax check passed"
else
    echo "❌ Syntax check FAILED"
    exit 1
fi
echo ""

echo "Creating diff..."
diff -u /home/qt/quantum_trader/ai_strategy_router.py /home/qt/quantum_trader/ai_strategy_router.py.new > $BACKUP_DIR/after/router.diff || true
echo "Diff saved to: $BACKUP_DIR/after/router.diff"
echo ""

echo "=== PHASE 3: DEPLOY ==="
echo "Moving new file into place..."
mv /home/qt/quantum_trader/ai_strategy_router.py.new /home/qt/quantum_trader/ai_strategy_router.py
echo "✅ File deployed"
echo ""

echo "=== PHASE 4: SET CAPACITY KEY ==="
echo "Setting quantum:state:open_orders to 0 (safe default)..."
redis-cli SET quantum:state:open_orders 0
CAPACITY_SET=$(redis-cli GET quantum:state:open_orders)
echo "Capacity key set to: $CAPACITY_SET"
echo ""

echo "=== PHASE 5: RESTART SERVICE ==="
echo "Restarting quantum-ai-strategy-router.service..."
systemctl restart quantum-ai-strategy-router.service
sleep 3

echo "Service status:"
systemctl --no-pager -l status quantum-ai-strategy-router.service | head -40
echo ""

ACTIVE=$(systemctl is-active quantum-ai-strategy-router.service)
if [ "$ACTIVE" != "active" ]; then
    echo "❌ SERVICE FAILED TO START!"
    echo "Rolling back..."
    cp $BACKUP_DIR/backup/ai_strategy_router.py.* /home/qt/quantum_trader/ai_strategy_router.py
    systemctl restart quantum-ai-strategy-router.service
    echo "❌ ROLLBACK COMPLETE - Check logs"
    exit 1
fi
echo "✅ Service is active"
echo ""

echo "=== PHASE 6: WAIT FOR PROOF (30s) ==="
echo "Waiting 30 seconds for service to process..."
sleep 30
echo ""

echo "=== PHASE 7: PROOF AFTER ==="
echo "--- AFTER: Publish Rate (last 2min) ---"
AFTER_COUNT=$(journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -c "Trade Intent published" || echo "0")
echo "Trade Intent publishes: $AFTER_COUNT"
echo "CHANGE: $BEFORE_COUNT → $AFTER_COUNT"
echo ""

echo "--- AFTER: P0.CAPQUAL Logs ---"
journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -E "P0.CAP|CAPACITY|TARGET|BEST_OF|SIZE_BOOST" | tail -40
echo ""

echo "--- AFTER: Governor Daily Trades ---"
AFTER_GOVERNOR=$(redis-cli GET quantum:governor:daily_trades:$TODAY || echo "NULL")
echo "Governor count: $AFTER_GOVERNOR"
echo "CHANGE: $BEFORE_GOVERNOR → $AFTER_GOVERNOR"
echo ""

echo "--- AFTER: Dedup Keys ---"
AFTER_DEDUP=$(redis-cli --scan --pattern "quantum:dedup:trade_intent:*" | wc -l)
echo "Dedup keys: $AFTER_DEDUP"
echo "CHANGE: $BEFORE_DEDUP → $AFTER_DEDUP"
echo ""

echo "--- AFTER: Capacity Key ---"
AFTER_CAPACITY=$(redis-cli GET quantum:state:open_orders || echo "NULL")
echo "Open orders: $AFTER_CAPACITY"
echo ""

echo "=== DEPLOYMENT COMPLETE ==="
echo "Backup dir: $BACKUP_DIR"
echo "Diff: $BACKUP_DIR/after/router.diff"
echo ""

echo "SUCCESS CRITERIA CHECK:"
if [ "$AFTER_COUNT" -lt "$BEFORE_COUNT" ]; then
    echo "✅ Publish rate decreased ($BEFORE_COUNT → $AFTER_COUNT)"
else
    echo "⚠️  Publish rate not decreased ($BEFORE_COUNT → $AFTER_COUNT)"
fi

if systemctl is-active quantum-ai-strategy-router.service > /dev/null; then
    echo "✅ Service is active"
else
    echo "❌ Service is NOT active"
fi

echo ""
echo "Monitor logs: journalctl -u quantum-ai-strategy-router.service -f"
