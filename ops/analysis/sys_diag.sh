#!/bin/bash
echo "=== SYSTEM MODE ==="
redis-cli GET quantum:system:mode

echo ""
echo "=== LOCKDOWN KEYS ==="
redis-cli KEYS 'quantum:lockdown:*'

echo ""
echo "=== ANTI-CHURN ALL KEYS ==="
redis-cli KEYS 'quantum:anti_churn:*'

echo ""
echo "=== COOLDOWN last_exec_ts values ==="
for k in $(redis-cli KEYS 'quantum:cooldown:last_exec_ts:*' | head -10); do
    val=$(redis-cli GET "$k")
    ttl=$(redis-cli TTL "$k")
    echo "  $k = $val  TTL=${ttl}s"
done

echo ""
echo "=== GOVERNOR STATE ==="
redis-cli GET quantum:governor:state
redis-cli GET quantum:governor:mode
redis-cli KEYS 'quantum:governor:*' | head -10

echo ""
echo "=== INTENT BRIDGE LOG (last 30) ==="
journalctl -u quantum-intent-bridge -n 30 --no-pager 2>&1

echo ""
echo "=== AUTONOMOUS TRADER LOG (last 20) ==="
journalctl -u quantum-autonomous-trader -n 20 --no-pager 2>&1

echo ""
echo "=== HARVEST BRAIN - which file is running? ==="
systemctl show quantum-harvest-brain --property=ExecStart 2>/dev/null
cat /etc/systemd/system/quantum-harvest-brain.service | grep -E 'ExecStart|WorkingDirectory|Environment' 2>/dev/null

echo ""
echo "=== HARVEST BRAIN LOG ANYWHERE ==="
journalctl -u quantum-harvest-brain -n 5 --no-pager 2>&1
# Try finding the process directly
ps aux | grep -i harvest | grep -v grep

echo ""
echo "=== ACTIVE POSITIONS NOW ==="
redis-cli KEYS 'quantum:positions:*' | head -10
