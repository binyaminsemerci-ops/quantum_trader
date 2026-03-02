#!/bin/bash
set -e

echo "=== Step 1: Apply permit key TTL patches ==="
python3 /tmp/fix_remaining_ttl.py

echo ""
echo "=== Step 2: Restart harvest-brain (picks up home dir changes) ==="
systemctl restart quantum-harvest-brain
sleep 3
systemctl is-active quantum-harvest-brain && echo "harvest-brain: OK"

echo ""
echo "=== Step 3: Flush old permit TTL=-1 keys ==="
FLUSHED=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && redis-cli DEL "$KEY" >/dev/null && FLUSHED=$((FLUSHED+1))
done
echo "Flushed $FLUSHED permanent permit keys"

echo ""
echo "=== Step 4: Clean quantum:metrics bloat ==="
echo "Current count: $(redis-cli KEYS 'quantum:metrics:*' 2>/dev/null | wc -l)"

# Delete ALL quantum:metrics:exit:* keys (they're historical exit logs, 1.2M of them)
# Using KEYS + xargs for batch delete (safe since these are analytics-only)
echo "Deleting quantum:metrics:exit:* keys in batches..."
TOTAL_DEL=0
while true; do
    BATCH=$(redis-cli KEYS "quantum:metrics:exit:*" 2>/dev/null | head -1000)
    COUNT=$(echo "$BATCH" | grep -c "quantum:metrics" 2>/dev/null || echo 0)
    [ "$COUNT" = "0" ] && break
    echo "$BATCH" | xargs redis-cli DEL >/dev/null 2>&1
    TOTAL_DEL=$((TOTAL_DEL + COUNT))
    echo "  Deleted $TOTAL_DEL so far..."
    sleep 0.1
done
echo "Total deleted: $TOTAL_DEL quantum:metrics:exit keys"

echo ""
echo "=== Step 5: Final state ==="
echo "Permit keys remaining: $(redis-cli KEYS 'quantum:permit:*' 2>/dev/null | wc -l)"
PERM=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -30); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && PERM=$((PERM+1))
done
echo "Permit keys TTL=-1: $PERM"
echo ""
echo "Redis total keys: $(redis-cli dbsize 2>/dev/null)"
echo "Redis memory: $(redis-cli info memory 2>/dev/null | grep used_memory_human | head -1)"

echo ""
echo "=== Step 6: Git commit ==="
cd /home/qt/quantum_trader
git add microservices/harvest_brain/harvest_brain.py scripts/auto_permit_p33.py
git diff --cached --name-only
git commit -m "fix(H1): patch harvest_brain home dir + auto_permit_p33 with 24h TTL" 2>&1 || echo "nothing new"
git push origin main 2>&1

echo "DONE"
