#!/bin/bash
# Flush remaining TTL=-1 permit keys (residual from before patch)
echo "=== Flushing remaining permanent permit keys ==="
FLUSHED=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null); do
    T=$(redis-cli TTL "$KEY")
    if [ "$T" = "-1" ]; then
        redis-cli DEL "$KEY" >/dev/null
        FLUSHED=$((FLUSHED+1))
    fi
done
echo "Flushed $FLUSHED permanent keys"

echo ""
echo "=== Remaining permit keys ==="
TOTAL=$(redis-cli KEYS "quantum:permit:*" 2>/dev/null | wc -l)
echo "Total: $TOTAL"

echo ""
echo "=== Sample TTLs on remaining keys ==="
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -5); do
    echo "  $KEY: TTL=$(redis-cli TTL $KEY)"
done

echo ""
echo "=== Git push (home repo) ==="
cd /home/qt/quantum_trader
git push origin main 2>&1 || echo "Push failed (check remote)"

echo ""
echo "=== Git push (/opt/quantum) ==="
cd /opt/quantum
git push origin main 2>&1 || echo "Push failed or no remote configured"

echo "DONE"
