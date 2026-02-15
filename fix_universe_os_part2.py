#!/usr/bin/env python3
"""
Fix 2: Add missing Universe OS variable definitions and start() logic
"""

file_path = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(file_path, "r") as f:
    content = f.read()

changes = 0

# FIX: Add Universe OS variable definitions after min_confidence
# Check if already defined properly
if 'self.use_universe_os = os.getenv' not in content:
    # Find the line with min_confidence and add after it
    old_pattern = '''        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.65"))

        # Components'''
    new_pattern = '''        self.min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.65"))

        # Universe OS configuration
        self.use_universe_os = os.getenv("USE_UNIVERSE_OS", "false").lower() == "true"
        self.universe_max_symbols = int(os.getenv("UNIVERSE_MAX_SYMBOLS", "50"))

        # Components'''
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        changes += 1
        print("✅ Added Universe OS variable definitions")
else:
    print("✓ Universe OS variables already defined")

# FIX: Add Universe OS logic in start() method
# Check if the logic is already there
parts = content.split("async def start")
if len(parts) > 1 and "if self.use_universe_os:" not in parts[1][:800]:
    old_start = '''        self._running = True

        # Filter symbols by funding rate BEFORE starting'''
    new_start = '''        self._running = True

        # Get symbols - from Universe OS OR ENV
        if self.use_universe_os:
            logger.info("[AutonomousTrader] 🌐 UNIVERSE OS ENABLED - Fetching dynamic symbols...")
            candidate_symbols = await self._get_universe_symbols()
        else:
            logger.info("[AutonomousTrader] 📋 Using hardcoded ENV symbols")
            candidate_symbols = self.candidate_symbols

        # Filter symbols by funding rate BEFORE starting'''
    
    if old_start in content:
        content = content.replace(old_start, new_start)
        changes += 1
        print("✅ Added Universe OS logic in start()")
else:
    print("✓ Universe OS start() logic already present")

# FIX: Update filtering to use candidate_symbols variable instead of self.candidate_symbols
old_filter = 'len(self.candidate_symbols)} symbols by funding rate'
new_filter = 'len(candidate_symbols)} symbols by funding rate'

if old_filter in content:
    content = content.replace(old_filter, new_filter)
    changes += 1
    print("✅ Updated filtering log to use candidate_symbols")

old_filter2 = "safe_symbols = await get_filtered_symbols(self.candidate_symbols)"
new_filter2 = "safe_symbols = await get_filtered_symbols(candidate_symbols)"

if old_filter2 in content:
    content = content.replace(old_filter2, new_filter2)
    changes += 1
    print("✅ Updated get_filtered_symbols call")

if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} additional fixes")
else:
    print("\n✓ All fixes already applied or patterns don't match")
