#!/usr/bin/env python3
"""
Fix: Enable Universe OS support in autonomous_trader.py
This allows trading with 50+ symbols instead of hardcoded 12
"""

import re

file_path = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(file_path, "r") as f:
    content = f.read()

changes = 0

# =============================================
# FIX 1: Add json import at top if not present
# =============================================
if "import json" not in content:
    # Add after "import os"
    content = content.replace(
        "import os\n",
        "import os\nimport json\n"
    )
    changes += 1
    print("✅ Fix 1: Added json import")

# =============================================
# FIX 2: Add Universe OS config in __init__
# =============================================
old_config = '''        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.65"))

        # Components'''

new_config = '''        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.65"))

        # Universe OS configuration
        self.use_universe_os = os.getenv("USE_UNIVERSE_OS", "false").lower() == "true"
        self.universe_max_symbols = int(os.getenv("UNIVERSE_MAX_SYMBOLS", "50"))

        # Components'''

if old_config in content:
    content = content.replace(old_config, new_config)
    changes += 1
    print("✅ Fix 2: Added Universe OS config")

# =============================================
# FIX 3: Update logging to show Universe OS status
# =============================================
old_log = '''        logger.info(f"  Candidate symbols: {len(self.candidate_symbols)} (will be filtered by funding rates on startup)")'''

new_log = '''        logger.info(f"  USE_UNIVERSE_OS: {self.use_universe_os}")
        if self.use_universe_os:
            logger.info(f"  Universe max symbols: {self.universe_max_symbols}")
        else:
            logger.info(f"  Candidate symbols: {len(self.candidate_symbols)} (hardcoded from ENV)")'''

if old_log in content:
    content = content.replace(old_log, new_log)
    changes += 1
    print("✅ Fix 3: Updated logging")

# =============================================
# FIX 4: Add _get_universe_symbols method
# =============================================
# Find where to insert - after __init__ method ends, before start()
method_to_add = '''
    async def _get_universe_symbols(self):
        """
        Fetch symbols from Universe Service (Redis)
        Returns dynamic symbol list from quantum:cfg:universe:active
        Falls back to ENV symbols if Universe Service unavailable
        """
        try:
            universe_data = self.redis.get("quantum:cfg:universe:active")
            if not universe_data:
                logger.warning("[Universe] No universe data in Redis, using ENV fallback")
                return self.candidate_symbols

            data = json.loads(universe_data)
            all_symbols = data.get("symbols", [])

            if not all_symbols:
                logger.warning("[Universe] Empty symbol list, using ENV fallback")
                return self.candidate_symbols

            # Prioritize major symbols (always include these)
            priority_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "ADAUSDT"]
            priority_set = set(priority_symbols)

            # Start with priority symbols that exist in universe
            symbols = [s for s in priority_symbols if s in all_symbols]

            # Add remaining symbols up to max
            remaining = [s for s in all_symbols if s not in priority_set]
            slots_left = self.universe_max_symbols - len(symbols)
            symbols.extend(remaining[:slots_left])
            
            logger.info(f"[Universe] Loaded {len(symbols)} symbols from Universe Service (total available: {len(all_symbols)})")
            return symbols

        except Exception as e:
            logger.error(f"[Universe] Failed to fetch universe: {e}, using ENV fallback")
            return self.candidate_symbols

'''

# Insert before "async def start(self):"
if "async def _get_universe_symbols" not in content:
    content = content.replace(
        "    async def start(self):",
        method_to_add + "    async def start(self):"
    )
    changes += 1
    print("✅ Fix 4: Added _get_universe_symbols method")

# =============================================
# FIX 5: Modify start() to use Universe OS
# =============================================
old_start = '''        self._running = True

        # Filter symbols by funding rate BEFORE starting
        logger.info(f"[AutonomousTrader] Filtering {len(self.candidate_symbols)} symbols by funding rate...")
        safe_symbols = await get_filtered_symbols(self.candidate_symbols)'''

new_start = '''        self._running = True

        # Get symbols - from Universe OS OR ENV
        if self.use_universe_os:
            logger.info("[AutonomousTrader] 🌐 UNIVERSE OS ENABLED - Fetching dynamic symbols...")
            candidate_symbols = await self._get_universe_symbols()
        else:
            logger.info("[AutonomousTrader] 📋 Using hardcoded ENV symbols")
            candidate_symbols = self.candidate_symbols

        # Filter symbols by funding rate BEFORE starting
        logger.info(f"[AutonomousTrader] Filtering {len(candidate_symbols)} symbols by funding rate...")
        safe_symbols = await get_filtered_symbols(candidate_symbols)'''

if old_start in content:
    content = content.replace(old_start, new_start)
    changes += 1
    print("✅ Fix 5: Modified start() to use Universe OS")

# =============================================
# FIX 6: Update the comparison after filtering
# =============================================
old_compare = '''        if len(safe_symbols) < len(self.candidate_symbols):
            removed = len(self.candidate_symbols) - len(safe_symbols)'''

new_compare = '''        if len(safe_symbols) < len(candidate_symbols):
            removed = len(candidate_symbols) - len(safe_symbols)'''

if old_compare in content:
    content = content.replace(old_compare, new_compare)
    changes += 1
    print("✅ Fix 6: Updated symbol comparison")

# Write changes
if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} fixes to autonomous_trader.py")
    print("🌐 Universe OS is now enabled!")
else:
    print("\n⚠️ No changes applied - patterns may not match")
    print("Checking what's in the file...")
