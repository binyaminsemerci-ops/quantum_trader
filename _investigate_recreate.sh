#!/bin/bash
echo "=== INVESTIGATE: Who creates snapshot keys? ==="

echo ""
echo "--- Services running p34 or position bootstrap ---"
systemctl list-units --type=service --state=active 2>/dev/null | grep -iE "p34|bootstrap|position|sync" | head -20

echo ""
echo "--- Find timer units that could trigger snapshot creation ---"
systemctl list-units --type=timer --state=active 2>/dev/null | head -20

echo ""
echo "--- Search for snapshot key creation in codebase ---"
grep -r "position:snapshot" /opt/quantum/ /home/qt/quantum_trader/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__" | head -30

echo ""
echo "--- Snapshot keys current TTL ---"
for k in $(redis-cli KEYS "quantum:position:snapshot:*" | head -5); do
    TTL=$(redis-cli TTL "$k")
    echo "  $k TTL=$TTL"
done

echo ""
echo "=== INVESTIGATE: Who creates ADAUSDT cooldown? ==="

echo ""
echo "--- Search for 'cooldown:last_exec_ts' in codebase ---"
grep -r "last_exec_ts" /opt/quantum/ /home/qt/quantum_trader/ 2>/dev/null \
  | grep -v ".pyc" | grep -v "__pycache__" | grep -v "Binary" | head -20

echo ""
echo "--- intent-executor logs around cooldown set for ADAUSDT ---"
journalctl -u quantum-intent-executor.service --since "30 minutes ago" --no-pager 2>/dev/null \
  | grep -iE "ADA|cooldown|exec_ts" | tail -20

echo ""
echo "=== VERIFY P6: No new WRONGTYPE in last 5 min ==="
journalctl -u quantum-intent-executor.service --since "5 minutes ago" --no-pager 2>/dev/null \
  | grep -iE "WRONGTYPE|ResponseError" | tail -10
echo "  (blank = no new WRONGTYPE errors = P6 FIXED)"

echo ""
echo "=== Current HV2 tick stats ==="
journalctl -u quantum-harvest-v2.service --since "2 minutes ago" --no-pager 2>/dev/null \
  | grep "HV2_TICK" | tail -5
