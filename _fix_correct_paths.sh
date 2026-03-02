#!/bin/bash
set -e

echo "=== CHECK: Both files exist? ==="
ls -la /opt/quantum/microservices/harvest_v2/feeds/position_provider.py
ls -la /home/qt/quantum_trader/microservices/harvest_v2/feeds/position_provider.py
ls -la /opt/quantum/microservices/intent_executor/main.py
ls -la /home/qt/quantum_trader/microservices/intent_executor/main.py

echo ""
echo "=== CHECK: Already patched? ==="
echo "  opt/hv2 position_provider:"
grep -c "_skip\|all_keys" /opt/quantum/microservices/harvest_v2/feeds/position_provider.py || echo "  NOT PATCHED"

echo "  home/hv2 position_provider:"
grep -c "_skip\|all_keys" /home/qt/quantum_trader/microservices/harvest_v2/feeds/position_provider.py || echo "  NOT PATCHED"

echo "  opt/ie main.py cooldown TTL:"
grep -c "ex=3600" /opt/quantum/microservices/intent_executor/main.py || echo "  NOT PATCHED"

echo "  home/ie main.py cooldown TTL:"
grep -c "ex=3600" /home/qt/quantum_trader/microservices/intent_executor/main.py || echo "  NOT PATCHED"

echo ""
echo "=============================="
echo " FIX: Patch /opt/quantum harvest_v2 position_provider (CORRECT PATH)"
echo "=============================="

PP="/opt/quantum/microservices/harvest_v2/feeds/position_provider.py"

# Check if already patched
if grep -q "_skip\|all_keys" "$PP" 2>/dev/null; then
    echo "  Already patched: $PP"
else
    cp "$PP" "${PP}.bak.$(date +%s)"
    python3 - <<'PYEOF'
fpath = "/opt/quantum/microservices/harvest_v2/feeds/position_provider.py"
with open(fpath, "r") as f:
    content = f.read()

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
        # snapshot:* = reconcile-engine snapshots for all symbols (qty=0 if no pos)
        # ledger:*   = execution ledger (managed by intent_executor)
        _skip = (POSITION_KEY_PREFIX + "snapshot:", POSITION_KEY_PREFIX + "ledger:")
        keys = [k for k in all_keys if not any(k.startswith(p) for p in _skip)]
        if not keys:
            return result

        result.total_keys = len(keys)'''

if old not in content:
    print("ERROR: pattern not found in /opt version")
    import sys; sys.exit(1)

content = content.replace(old, new, 1)
with open(fpath, "w") as f:
    f.write(content)
print("  PATCHED /opt/quantum/...position_provider.py")
PYEOF
fi

echo ""
echo "=============================="
echo " FIX: Patch /home/qt intent_executor (CORRECT PATH)"
echo "=============================="

IE="/home/qt/quantum_trader/microservices/intent_executor/main.py"

echo "--- Verify cooldown line in home/qt version ---"
grep -n "cooldown_key\|last_exec_ts" "$IE" | head -5

if grep -q "ex=3600" "$IE" 2>/dev/null; then
    echo "  Already patched: $IE"
else
    cp "$IE" "${IE}.bak.$(date +%s)"
    python3 - <<'PYEOF'
fpath = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(fpath, "r") as f:
    content = f.read()

old = '                self.redis.set(cooldown_key, str(int(time.time() * 1000)))'
new = '                self.redis.set(cooldown_key, str(int(time.time() * 1000)), ex=3600)  # 1h TTL — prevents permanent cooldown'

if old not in content:
    # Try opt version pattern
    print("Pattern not found in home/qt version")
    print("Searching for alternatives...")
    idx = content.find("cooldown_key")
    if idx > 0:
        print("Context:", content[idx-50:idx+200])
    import sys; sys.exit(1)

content = content.replace(old, new, 1)
with open(fpath, "w") as f:
    f.write(content)
print("  PATCHED /home/qt/.../intent_executor/main.py")
PYEOF
fi

echo ""
echo "=============================="
echo " RESTART services (correct files now patched)"
echo "=============================="
systemctl restart quantum-harvest-v2.service
sleep 2
echo "  harvest-v2: $(systemctl is-active quantum-harvest-v2.service)"

systemctl restart quantum-intent-executor.service
sleep 2
echo "  intent-executor: $(systemctl is-active quantum-intent-executor.service)"

echo ""
echo "=============================="
echo " Verify patches applied"
echo "=============================="
echo "  opt/hv2 position_provider patch:"
grep -n "_skip\|all_keys" /opt/quantum/microservices/harvest_v2/feeds/position_provider.py

echo "  home/ie cooldown TTL patch:"
grep -n "ex=3600" /home/qt/quantum_trader/microservices/intent_executor/main.py || echo "  NOT FOUND - investigate"

echo ""
echo "=============================="
echo " ADAUSDT position investigation"
echo "=============================="
echo "--- ADA position key ---"
redis-cli HGETALL "quantum:position:ADAUSDT"
echo ""
echo "--- ADA ledger ---"
redis-cli HGETALL "quantum:position:ledger:ADAUSDT"
echo ""
echo "--- ADA proposals ---"
redis-cli HGETALL "quantum:harvest:proposal:ADAUSDT"
echo ""
echo "--- Current all proposals ---"
for k in $(redis-cli KEYS "quantum:harvest:proposal:*"); do
    SYM=$(echo "$k" | sed 's/quantum:harvest:proposal://')
    ACT=$(redis-cli HGET "$k" action)
    echo "  $SYM: $ACT"
done
