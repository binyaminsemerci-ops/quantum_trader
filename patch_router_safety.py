#!/usr/bin/env python3
"""Patch ai_strategy_router.py to integrate safety kernel."""

import re

router_file = "/home/qt/quantum_trader/ai_strategy_router.py"

with open(router_file, "r") as f:
    content = f.read()

# 1. Add import after "from typing import..."
if "from safety_kernel import" not in content:
    content = content.replace(
        "from typing import Optional, Tuple\n",
        "from typing import Optional, Tuple\nfrom safety_kernel import create_safety_kernel\n"
    )
    print("✅ Added safety_kernel import")
else:
    print("⏭️  Import already exists")

# 2. Initialize safety kernel in __init__
if "self.safety = create_safety_kernel" not in content:
    content = content.replace(
        "        self._last_invalid_warn_ts = 0.0\n",
        "        self._last_invalid_warn_ts = 0.0\n        self.safety = create_safety_kernel(self.redis)\n"
    )
    print("✅ Added safety kernel initialization")
else:
    print("⏭️  Initialization already exists")

# 3. Inject safety gate before redis.xadd
safety_gate = '''            # === CORE SAFETY KERNEL: Last line of defense ===
            allowed, reason, meta = self.safety.should_publish_intent(
                symbol=symbol,
                side=side,
                correlation_id=corr_id_clean,
                trace_id=trace_id_clean
            )
            
            if not allowed:
                logger.warning(
                    f"[SAFETY] 🛑 BLOCKED | reason={reason} {symbol} {side} | {meta}"
                )
                return
            
            # === END SAFETY KERNEL ===
            
            '''

if "CORE SAFETY KERNEL" not in content:
    # Find the exact location: right before "# Wrap in EventBus format"
    content = content.replace(
        "            # Wrap in EventBus format (execution service expects \"data\" field)\n",
        safety_gate + "            # Wrap in EventBus format (execution service expects \"data\" field)\n"
    )
    print("✅ Injected safety gate at publish boundary")
else:
    print("⏭️  Safety gate already injected")

# Write back
with open(router_file, "w") as f:
    f.write(content)

print("\n✅ Router patched successfully")
