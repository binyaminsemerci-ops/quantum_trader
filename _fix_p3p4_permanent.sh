#!/bin/bash
set -e
echo "=============================="
echo " FIX P3: position_provider.py"
echo "=============================="

PP="/home/qt/quantum_trader/microservices/harvest_v2/feeds/position_provider.py"

echo "--- Backup ---"
cp "$PP" "${PP}.bak.$(date +%s)"
echo "  Backup OK"

echo "--- Applying patch: filter snapshot+ledger keys ---"
python3 - <<'PYEOF'
import re

fpath = "/home/qt/quantum_trader/microservices/harvest_v2/feeds/position_provider.py"
with open(fpath, "r") as f:
    content = f.read()

# Old fetch_positions start
old = '''    def fetch_positions(self) -> FetchResult:
        result = FetchResult()
        keys = self._redis.keys(f"{POSITION_KEY_PREFIX}*")
        if not keys:
            return result

        result.total_keys = len(keys)'''

new = '''    def fetch_positions(self) -> FetchResult:
        result = FetchResult()
        all_keys = self._redis.keys(f"{POSITION_KEY_PREFIX}*")
        # Filter out sub-namespace keys — only live position keys
        # snapshot:* = reconcile-engine snapshots for ALL symbols (qty=0 when no pos)
        # ledger:*   = execution ledger entries managed by intent_executor
        _skip = (POSITION_KEY_PREFIX + "snapshot:", POSITION_KEY_PREFIX + "ledger:")
        keys = [k for k in all_keys if not any(k.startswith(p) for p in _skip)]
        if not keys:
            return result

        result.total_keys = len(keys)'''

if old not in content:
    print("ERROR: old pattern not found in file")
    import sys; sys.exit(1)

content = content.replace(old, new, 1)
with open(fpath, "w") as f:
    f.write(content)
print("  PATCHED position_provider.py")
PYEOF

echo "--- Verify patch applied ---"
grep -n "snapshot.*ledger\|_skip\|all_keys" "$PP" | head -10

echo ""
echo "=============================="
echo " FIX P4: intent_executor cooldown TTL"
echo "=============================="

IE="/opt/quantum/microservices/intent_executor/main.py"

echo "--- Backup ---"
cp "$IE" "${IE}.bak.$(date +%s)"
echo "  Backup OK"

echo "--- Applying patch: add ex=3600 to cooldown SET ---"
python3 - <<'PYEOF'
fpath = "/opt/quantum/microservices/intent_executor/main.py"
with open(fpath, "r") as f:
    content = f.read()

old = '                self.redis.set(cooldown_key, str(int(time.time() * 1000)))'
new = '                self.redis.set(cooldown_key, str(int(time.time() * 1000)), ex=3600)  # 1h TTL — prevents permanent cooldown'

if old not in content:
    print("ERROR: old pattern not found in intent_executor")
    import sys; sys.exit(1)

content = content.replace(old, new, 1)
with open(fpath, "w") as f:
    f.write(content)
print("  PATCHED intent_executor/main.py")
PYEOF

echo "--- Verify patch applied ---"
grep -n "cooldown_key\|ex=3600" "$IE" | head -5

echo ""
echo "=============================="
echo " CLEANUP: Delete current permanent cooldowns"
echo "=============================="
echo "--- Delete all TTL=-1 cooldown keys ---"
for k in $(redis-cli KEYS "quantum:cooldown:last_exec_ts:*" 2>/dev/null); do
    TTL=$(redis-cli TTL "$k")
    if [ "$TTL" = "-1" ]; then
        redis-cli DEL "$k" > /dev/null
        echo "  DELETED permanent cooldown: $k"
    fi
done

echo ""
echo "=============================="
echo " RESTART services"
echo "=============================="

echo "--- Restart quantum-harvest-v2.service ---"
systemctl restart quantum-harvest-v2.service
sleep 3
echo "  harvest-v2: $(systemctl is-active quantum-harvest-v2.service)"

echo "--- Restart quantum-intent-executor.service ---"
systemctl restart quantum-intent-executor.service
sleep 3
echo "  intent-executor: $(systemctl is-active quantum-intent-executor.service)"

echo ""
echo "=============================="
echo " VERIFY: 30s wait then check"
echo "=============================="
sleep 30

echo "--- HV2 skip count after fix ---"
journalctl -u quantum-harvest-v2.service --since "1 minute ago" --no-pager 2>/dev/null \
  | grep "HV2_TICK" | tail -3

echo "--- Snapshot keys remaining ---"
COUNT=$(redis-cli KEYS "quantum:position:snapshot:*" 2>/dev/null | grep -c "snapshot" || echo 0)
echo "  Snapshot keys: $COUNT (still exist in Redis, but HV2 now ignores them)"

echo "--- ADAUSDT cooldown TTL after fix ---"
redis-cli TTL "quantum:cooldown:last_exec_ts:ADAUSDT"

echo "[DONE] P3+P4 permanent fixes applied"
