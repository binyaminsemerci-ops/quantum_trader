#!/usr/bin/env python3
"""P0.EXIT_GUARD.3 Patcher - Adds position-open tracking, rate limiting, stale reject"""
import re
import sys
from datetime import datetime

FILE = "/home/qt/quantum_trader/services/exit_monitor_service.py"

def main():
    with open(FILE, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Idempotent check
    if "P0_EXIT_GUARD_V3" in content:
        print("✅ P0_EXIT_GUARD_V3 already present (idempotent)")
        return 0
    
    # Find REDIS_EXIT_GUARD_V1 block
    match = re.search(r"# REDIS_EXIT_GUARD_V1", content)
    if not match:
        print("❌ ERROR: REDIS_EXIT_GUARD_V1 not found")
        return 1
    
    v1_pos = match.start()
    
    # Find end of existing guard block (before first async def after guards)
    after_guards = re.search(r"\n\nasync def ", content[v1_pos:])
    if not after_guards:
        print("❌ ERROR: Could not find insertion point")
        return 1
    
    insert_pos = v1_pos + after_guards.start()
    
    # V3 guard additions
    v3_code = """

# P0_EXIT_GUARD_V3 - Position-open tracking, rate limiting, stale reject
from datetime import datetime, timezone
EXIT_RATE_WINDOW_SEC = 10
EXIT_RATE_MAX_PER_WINDOW = 1
POSITION_OPEN_TTL_SEC = 21600  # 6 hours

def _key_open(symbol, order_id):
    return f"quantum:pos:open:{symbol}:{order_id}"

def _key_rate(symbol):
    import time
    window = int(time.time() // EXIT_RATE_WINDOW_SEC)
    return f"quantum:rate:exit:{symbol}:{window}"

def mark_position_open(symbol, order_id):
    r = _r()
    if r is None:
        return
    key = _key_open(symbol, order_id)
    try:
        r.set(key, "1", ex=POSITION_OPEN_TTL_SEC)
    except Exception as e:
        logger.error(f"❌ mark_position_open failed: {e}")

def clear_position_open(symbol, order_id):
    r = _r()
    if r is None:
        return
    key = _key_open(symbol, order_id)
    try:
        r.delete(key)
    except Exception as e:
        logger.error(f"❌ clear_position_open failed: {e}")

def is_position_open(symbol, order_id):
    r = _r()
    if r is None:
        return True  # fail-open: allow exit attempt
    key = _key_open(symbol, order_id)
    try:
        return r.exists(key) > 0
    except Exception as e:
        logger.error(f"❌ is_position_open failed: {e}")
        return True  # fail-open

def exit_rate_limited(symbol):
    r = _r()
    if r is None:
        return False
    key = _key_rate(symbol)
    try:
        count = r.incr(key)
        if count == 1:
            r.expire(key, EXIT_RATE_WINDOW_SEC)
        if count > EXIT_RATE_MAX_PER_WINDOW:
            logger.info(f"🚫 EXIT_RATE_LIMIT symbol={symbol} n={count}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ exit_rate_limited failed: {e}")
        return False  # fail-open
"""
    
    # Insert V3 code
    content = content[:insert_pos] + v3_code + content[insert_pos:]
    
    # Modify send_close_order - add guards at top
    send_close_pattern = r"(async def send_close_order\(position: TrackedPosition, reason: str\):\s*\n\s*\"\"\"[^\"]*\"\"\"\s*\n)"
    send_close_match = re.search(send_close_pattern, content)
    if not send_close_match:
        print("❌ ERROR: Could not find send_close_order function")
        return 1
    
    # Insert after docstring
    docstring_end = send_close_match.end()
    
    guard_checks = """    # P0_EXIT_GUARD_V3: position-open + rate-limit checks
    if not is_position_open(position.symbol, position.order_id):
        logger.info(f"🔴 EXIT_ALREADY_CLOSED symbol={position.symbol} oid={position.order_id}")
        return
    if exit_rate_limited(position.symbol):
        return
    
"""
    
    content = content[:docstring_end] + guard_checks + content[docstring_end:]
    
    # Find the "await redis_client.xadd" line in send_close_order and add clear after it
    xadd_pattern = r"(await redis_client\.xadd\([^)]+\)\s*\n)"
    xadd_matches = list(re.finditer(xadd_pattern, content))
    if xadd_matches:
        # Add clear after last xadd (should be the exit publish)
        last_xadd = xadd_matches[-1]
        clear_call = "        clear_position_open(position.symbol, position.order_id)\n"
        content = content[:last_xadd.end()] + clear_call + content[last_xadd.end():]
    
    # Modify position_listener - add mark_position_open and stale reject
    # Find where TrackedPosition is created and added
    listener_pattern = r"(tracked_positions\[result\.symbol\] = position\s*\n)"
    listener_match = re.search(listener_pattern, content)
    if listener_match:
        mark_call = """                # P0_EXIT_GUARD_V3: mark position as open
                mark_position_open(result.symbol, result.order_id)
"""
        content = content[:listener_match.end()] + mark_call + content[listener_match.end():]
    
    # Add stale reject check in position_listener after getting result
    stale_check = """
                # P0_EXIT_GUARD_V3: TRACK_STALE_REJECT
                try:
                    if hasattr(result, 'timestamp') and result.timestamp:
                        ts_str = result.timestamp.replace("Z", "+00:00") if isinstance(result.timestamp, str) else str(result.timestamp)
                        result_time = datetime.fromisoformat(ts_str) if "+" in ts_str else datetime.fromisoformat(ts_str + "+00:00")
                        age_sec = (datetime.now(timezone.utc) - result_time).total_seconds()
                        if age_sec > 60:
                            logger.info(f"⏰ TRACK_STALE_REJECT symbol={result.symbol} age={age_sec:.1f}s")
                            continue
                except Exception as e:
                    logger.debug(f"Stale check skip: {e}")
"""
    
    # Insert stale check before "if result.status == "FILLED""
    filled_check = re.search(r"(\s+if result\.status == \"FILLED\":)", content)
    if filled_check:
        content = content[:filled_check.start()] + stale_check + content[filled_check.start():]
    
    # Write patched file
    with open(FILE, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ P0_EXIT_GUARD_V3 patched successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
