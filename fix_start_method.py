#!/usr/bin/env python3
"""
Fix start() method to use Universe OS
"""

file_path = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(file_path, "r") as f:
    content = f.read()

# Fix start() to use Universe OS
old_start = '''        self._running = True
        
        # Filter symbols by funding rate BEFORE starting
        logger.info(f"[AutonomousTrader] Filtering {len(self.candidate_symbols)} symbols by funding rate...")
        safe_symbols = await get_filtered_symbols(self.candidate_symbols)
        
        if len(safe_symbols) < len(candidate_symbols):
            removed = len(candidate_symbols) - len(safe_symbols)'''

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
        safe_symbols = await get_filtered_symbols(candidate_symbols)
        
        if len(safe_symbols) < len(candidate_symbols):
            removed = len(candidate_symbols) - len(safe_symbols)'''

if old_start in content:
    content = content.replace(old_start, new_start)
    with open(file_path, "w") as f:
        f.write(content)
    print("✅ Added Universe OS logic to start() method")
else:
    print("❌ Pattern not found - checking content...")
    # Try to find the problematic section
    if "self._running = True" in content and "Filter symbols by funding rate" in content:
        print("Found markers but pattern doesn't match exactly")
        # Show context
        idx = content.find("self._running = True")
        print(f"Context around self._running = True:\n{content[idx:idx+500]}")
