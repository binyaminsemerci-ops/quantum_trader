#!/bin/bash
# P0.EXIT_GUARD - Complete deployment script
set -e

echo "=== P0.EXIT_GUARD DEPLOYMENT ==="
echo "Time: $(date -u)"
echo ""

SOURCE="/home/qt/quantum_trader/services/exit_monitor_service.py"
BACKUP="/tmp/p0_exitbrain_20260119_133345/exitbrain.py.backup"
WORKING="/tmp/exit_monitor_patched.py"

# Verify backup
if [ ! -f "$BACKUP" ]; then
    echo "❌ Backup not found"
    exit 1
fi
echo "✅ Backup verified: $BACKUP"

# Copy to working file
cp "$SOURCE" "$WORKING"
echo "✅ Working copy created"

# Apply patch using Python
python3 - << 'PYTHON_EOF'
source = "/tmp/exit_monitor_patched.py"

# Read file
with open(source, "r") as f:
    lines = f.readlines()

# Find insertion point (before send_close_order)
insert_idx = None
for i, line in enumerate(lines):
    if "async def send_close_order" in line:
        insert_idx = i
        break

if not insert_idx:
    print("❌ Cannot find send_close_order function")
    exit(1)

# Insert guard functions
guard_code = """
# === EXIT GUARD FUNCTIONS (P0.EXIT_GUARD) ===
_exit_processed = set()
_exit_cooldown = {}

def check_exit_dedup(position_symbol, order_id):
    key = f"{position_symbol}_{order_id}"
    if key in _exit_processed:
        logger.info(f"🔴 EXIT_DEDUP skip key={key}")
        return True
    _exit_processed.add(key)
    return False

def check_exit_cooldown(symbol, side):
    from datetime import datetime, timedelta
    key = f"{symbol}_{side}"
    now = datetime.utcnow()
    if key in _exit_cooldown:
        if now - _exit_cooldown[key] < timedelta(seconds=30):
            logger.info(f"⏸️ EXIT_COOLDOWN skip symbol={symbol} side={side}")
            return True
    _exit_cooldown[key] = now
    return False

"""

lines.insert(insert_idx, guard_code)

# Find send_close_order body start (after docstring)
for i in range(insert_idx, len(lines)):
    if '"""Send close order to execution service"""' in lines[i]:
        # Insert guards after docstring
        guard_call = """    # === EXIT GUARDS ===
    if check_exit_dedup(position.symbol, position.order_id):
        return
    if check_exit_cooldown(position.symbol, position.side):
        return
    
"""
        lines.insert(i + 1, guard_call)
        break

# Change log message
for i, line in enumerate(lines):
    if 'f"🎯 EXIT TRIGGERED:' in line:
        lines[i] = line.replace('f"🎯 EXIT TRIGGERED:', 'f"📤 EXIT_PUBLISH:')
        break

# Write modified file
with open(source, "w") as f:
    f.writelines(lines)

print("✅ Patch applied successfully")
PYTHON_EOF

echo "✅ Patch code inserted"

# Validate syntax
echo "Validating syntax..."
python3 -m py_compile "$WORKING"
if [ $? -eq 0 ]; then
    echo "✅ Syntax validation PASSED"
else
    echo "❌ Syntax validation FAILED"
    exit 1
fi

# Show diff summary
echo ""
echo "=== CHANGES SUMMARY ==="
CHANGES=$(diff -u "$SOURCE" "$WORKING" | wc -l)
echo "Total diff lines: $CHANGES"
echo ""
echo "First 30 lines of diff:"
diff -u "$SOURCE" "$WORKING" | head -30

# Deploy
echo ""
echo "=== DEPLOYING ==="
cp "$WORKING" "$SOURCE"
echo "✅ Deployed to $SOURCE"

# Restart service
echo ""
echo "=== RESTARTING SERVICE ==="
systemctl restart quantum-exit-monitor.service
sleep 3

# Check status
systemctl --no-pager -l status quantum-exit-monitor.service | head -40

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Check logs: tail -f /var/log/quantum/exit-monitor.log"
