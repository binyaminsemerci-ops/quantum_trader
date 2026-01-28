#!/usr/bin/env python3
"""Add P0.6 commit-after-publish logic to router."""

import re

router_file = "/home/qt/quantum_trader/ai_strategy_router.py"

with open(router_file, "r") as f:
    content = f.read()

# 1. Update datetime import
content = content.replace(
    "from datetime import datetime",
    "from datetime import datetime, timedelta, timezone"
)
print("✅ Step 1: Updated datetime imports")

# 2. Add helper methods after __init__
init_section = '''        self.safety = create_safety_kernel(self.redis)

    async def setup(self):'''

helper_methods = '''        self.safety = create_safety_kernel(self.redis)

    def _utc_day_key(self) -> str:
        """Return YYYYMMDD key for today (UTC)."""
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    
    def _ttl_to_next_utc_midnight(self) -> int:
        """Return seconds until next UTC midnight (minimum 60s)."""
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).date()
        expire_at = datetime.combine(tomorrow, datetime.min.time(), timezone.utc)
        ttl_seconds = int((expire_at - now).total_seconds())
        return max(60, ttl_seconds)
    
    def _commit_daily_trade(self) -> int:
        """P0.6 Commit daily trade counter after successful publish."""
        key = f"quantum:governor:daily_trades:{self._utc_day_key()}"
        pipe = self.redis.pipeline()
        pipe.incr(key)
        ttl = self.redis.ttl(key)
        if ttl is None or ttl < 0:
            pipe.expire(key, self._ttl_to_next_utc_midnight())
        res = pipe.execute()
        count = int(res[0]) if res else 0
        logger.info(f"[P0.6] COMMIT_DAILY_TRADE key={key} count={count}")
        return count

    async def setup(self):'''

content = content.replace(init_section, helper_methods)
print("✅ Step 2: Added helper methods (_utc_day_key, _ttl_to_next_utc_midnight, _commit_daily_trade)")

# 3. Add commit call after successful trade.intent publish
# Find the exact location right after redis.xadd for TRADE_INTENT_STREAM
publish_section = '''            logger.info(f"🚀 Trade Intent published: {symbol} {side}")'''

publish_with_commit = '''            logger.info(f"🚀 Trade Intent published: {symbol} {side}")
            
            # P0.6: Commit daily trade counter (after successful publish)
            self._commit_daily_trade()'''

content = content.replace(publish_section, publish_with_commit)
print("✅ Step 3: Added commit call after successful publish")

# Write back
with open(router_file, "w") as f:
    f.write(content)

print("\n✅ P0.6 commit-after-publish integrated into router")
