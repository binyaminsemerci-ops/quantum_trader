#!/bin/bash
# Position cleanup - delete all ghost positions from Redis

echo "=== POSITION CLEANUP $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""

echo "--- Phase 1: Backup current state ---"
redis-cli KEYS "quantum:position:*" > /tmp/position_keys_backup_$(date +%s).txt
echo "Backed up $(wc -l < /tmp/position_keys_backup_$(date +%s).txt 2>/dev/null || echo '?') keys"

echo ""
echo "--- Phase 2: Identify ghost positions ---"
ghost_count=0
real_count=0

for key in $(redis-cli KEYS "quantum:position:*"); do
    # Skip snapshot and ledger keys - handle separately
    if [[ "$key" == *"snapshot:"* ]] || [[ "$key" == *"ledger:"* ]]; then
        continue
    fi
    
    qty=$(redis-cli HGET "$key" quantity 2>/dev/null)
    side=$(redis-cli HGET "$key" side 2>/dev/null)
    
    # Check if ghost (qty=0 or missing qty, or side=NONE/FLAT)
    if [ -z "$qty" ] || [ "$qty" = "0" ] || [ "$qty" = "0.0" ] || [ "$side" = "NONE" ] || [ "$side" = "FLAT" ] || [ "$side" = "" ]; then
        echo "GHOST: $key (qty=$qty, side=$side)"
        redis-cli DEL "$key"
        ((ghost_count++))
    else
        # Verify qty is actually non-zero (handle negative values)
        qty_abs=$(echo "$qty" | tr -d '-')
        if [ "$(echo "$qty_abs > 0.001" | bc 2>/dev/null || echo 0)" -eq 1 ]; then
            echo "KEEP: $key (qty=$qty, side=$side)"
            ((real_count++))
        else
            echo "GHOST: $key (qty=$qty near zero, side=$side)"
            redis-cli DEL "$key"
            ((ghost_count++))
        fi
    fi
done

echo ""
echo "--- Phase 3: Clean up snapshot keys ---"
snapshot_count=0
for key in $(redis-cli KEYS "quantum:position:snapshot:*"); do
    echo "DELETE snapshot: $key"
    redis-cli DEL "$key"
    ((snapshot_count++))
done

echo ""
echo "--- Phase 4: Clean up ledger keys (keep only active) ---"
ledger_count=0
for key in $(redis-cli KEYS "quantum:position:ledger:*"); do
    qty=$(redis-cli HGET "$key" position_amt 2>/dev/null)
    if [ -z "$qty" ] || [ "$qty" = "0" ] || [ "$qty" = "0.0" ]; then
        echo "DELETE ledger: $key (qty=$qty)"
        redis-cli DEL "$key"
        ((ledger_count++))
    else
        echo "KEEP ledger: $key (qty=$qty)"
    fi
done

echo ""
echo "=== CLEANUP SUMMARY ==="
echo "Ghost positions deleted: $ghost_count"
echo "Snapshot keys deleted: $snapshot_count"
echo "Ledger keys deleted: $ledger_count"
echo "Real positions kept: $real_count"
echo ""
echo "Total positions remaining:"
redis-cli KEYS "quantum:position:*" | wc -l

echo ""
echo "=== VERIFICATION ==="
echo "Remaining position keys:"
for key in $(redis-cli KEYS "quantum:position:*" | head -20); do
    symbol=${key#quantum:position:}
    qty=$(redis-cli HGET "$key" quantity 2>/dev/null || redis-cli HGET "$key" position_amt 2>/dev/null)
    side=$(redis-cli HGET "$key" side 2>/dev/null)
    echo "  $symbol: side=$side qty=$qty"
done

echo ""
echo "=== CLEANUP COMPLETE ===" 
