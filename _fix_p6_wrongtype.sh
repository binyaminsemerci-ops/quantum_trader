#!/bin/bash
set -e
echo "=============================="
echo " FIX P6: intent_executor WRONGTYPE"
echo " Root cause: P3.3 uses setex(json) = STRING"
echo " intent_executor uses hgetall = expects HASH"
echo " Fix: add _read_permit_safe() to handle both"
echo "=============================="

IE="/home/qt/quantum_trader/microservices/intent_executor/main.py"

echo "--- Backup ---"
cp "$IE" "${IE}.bak.p6fix.$(date +%s)"
echo "  Backup OK"

python3 - <<'PYEOF'
import re

fpath = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(fpath, "r") as f:
    content = f.read()

# === PATCH 1: Add _read_permit_safe method before _wait_for_permit ===
old_wait_def = '''    def _wait_for_permit(self, plan_id: str) -> Optional[Dict]:
        """Wait for P3.3 permit with timeout"""
        key = f"quantum:permit:p33:{plan_id}"
        start_time = time.time()

        while time.time() - start_time < PERMIT_TIMEOUT_SEC:
            # Permit is stored as HASH not STRING
            permit_data = self.redis.hgetall(key)
            if permit_data:
                try:
                    # Convert bytes keys/values to strings
                    permit = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in permit_data.items()
                    }
                    return permit
                except Exception as e:
                    logger.warning(f"Failed to parse permit: {e}")
                    return None

            time.sleep(PERMIT_POLL_INTERVAL)
        
        return None'''

new_wait_def = '''    def _read_permit_safe(self, key: str) -> Optional[Dict]:
        """Read P3.3 permit key — handles both STRING (json) and HASH types.
        P3.3 position_state_brain uses: setex(key, ttl, json.dumps(data)) -> STRING.
        auto_permit_p33.py uses: hset() -> HASH.
        """
        try:
            data = self.redis.hgetall(key)
            if data:
                return {
                    k.decode() if isinstance(k, bytes) else k:
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in data.items()
                }
        except Exception:
            # WRONGTYPE: key is STRING type (P3.3 setex json) — fall through
            pass
        try:
            raw = self.redis.get(key)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode()
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"Error reading permit {key}: {e}")
        return None

    def _wait_for_permit(self, plan_id: str) -> Optional[Dict]:
        """Wait for P3.3 permit with timeout. Handles both STRING and HASH permit types."""
        key = f"quantum:permit:p33:{plan_id}"
        start_time = time.time()

        while time.time() - start_time < PERMIT_TIMEOUT_SEC:
            permit = self._read_permit_safe(key)
            if permit:
                return permit
            time.sleep(PERMIT_POLL_INTERVAL)

        return None'''

if old_wait_def not in content:
    print("ERROR: _wait_for_permit old pattern not found")
    import sys; sys.exit(1)

content = content.replace(old_wait_def, new_wait_def, 1)
print("  PATCH 1 applied: added _read_permit_safe(), simplified _wait_for_permit()")

# === PATCH 2: Fix inline permit check (hgetall -> _read_permit_safe) ===
old_inline = '''            permit_data = self.redis.hgetall(permit_key)  # FIX: HASH not STRING
            if permit_data:
                # Permit already exists (cached)
                try:
                    # Convert bytes keys/values to strings
                    permit = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in permit_data.items()
                    }
                    if source == '':
                        logger.info(f"✅ Permit found immediately (P3.3 bypass): {plan_id[:8]}")
                    else:
                        logger.info(f"✅ Permit cached: {plan_id[:8]}")
                except Exception as e:
                    logger.warning(f"Failed to parse cached permit: {e}")
                    permit = None
            else:
                # Permit not in cache - wait for it
                # This handles race condition where P3.3 permit not created yet
                if source == '':
                    logger.info(f"⏳ P3.3 permit not cached, waiting for creation: {plan_id[:8]}")
                else:
                    logger.info(f"⏳ Waiting for P3.3 permit: {plan_id[:8]}")
                permit = self._wait_for_permit(plan_id)'''

new_inline = '''            # _read_permit_safe handles both STRING (P3.3 setex json) and HASH types
            permit = self._read_permit_safe(permit_key)
            if permit:
                if source == '':
                    logger.info(f"✅ Permit found (P3.3 bypass): {plan_id[:8]}")
                else:
                    logger.info(f"✅ Permit found: {plan_id[:8]}")
            else:
                # Permit not in cache yet - wait for it
                if source == '':
                    logger.info(f"⏳ P3.3 permit not created yet, waiting: {plan_id[:8]}")
                else:
                    logger.info(f"⏳ Waiting for P3.3 permit: {plan_id[:8]}")
                permit = self._wait_for_permit(plan_id)'''

if old_inline not in content:
    print("ERROR: inline permit check old pattern not found")
    # Show context for debugging
    idx = content.find("permit_data = self.redis.hgetall(permit_key)")
    if idx > 0:
        print("Context found at:", idx)
        print(repr(content[idx-20:idx+200]))
    import sys; sys.exit(1)

content = content.replace(old_inline, new_inline, 1)
print("  PATCH 2 applied: inline permit check uses _read_permit_safe()")

with open(fpath, "w") as f:
    f.write(content)
print("  SAVED")
PYEOF

echo ""
echo "--- Verify patches ---"
echo "  _read_permit_safe method exists:"
grep -n "_read_permit_safe" "$IE" | head -5

echo ""
echo "--- No more bare hgetall on permit_key ---"
grep -n "hgetall(permit_key)\|hgetall(key)" "$IE" | head -5 || echo "  None found (GOOD)"

echo ""
echo "=============================="
echo " RESTART intent-executor"
echo "=============================="
systemctl restart quantum-intent-executor.service
sleep 3
echo "  intent-executor: $(systemctl is-active quantum-intent-executor.service)"

echo ""
echo "=============================="
echo " VERIFY: Wait 60s and check for WRONGTYPE errors"
echo "=============================="
echo "  Waiting 60s..."
sleep 60

NEW_ERRORS=$(journalctl -u quantum-intent-executor.service --since "30 seconds ago" --no-pager 2>/dev/null | grep -c "WRONGTYPE" || echo 0)
echo "  WRONGTYPE errors in last 30s: $NEW_ERRORS (expect 0)"

echo ""
echo "--- Recent intent-executor log (last 30s) ---"
journalctl -u quantum-intent-executor.service --since "30 seconds ago" --no-pager 2>/dev/null | tail -15
