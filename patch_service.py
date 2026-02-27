#!/usr/bin/env python3
"""Patch service.py with Fix 2 (TradeOutcome) and Fix 5 (equity from Redis)"""

import re
import sys

FILE = "/opt/quantum/microservices/ai_engine/service.py"

with open(FILE, "r") as f:
    content = f.read()

# ============================================================
# FIX 2: Add 5 missing args to TradeOutcome constructor call
# ============================================================
OLD_OUTCOME = """                    outcome = TradeOutcome(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        action=event_data.get("action", "unknown"),
                        confidence=event_data.get("confidence", 0.5),
                        pnl=pnl_percent / 100.0,  # Convert to decimal
                        position_size=event_data.get("position_size", 0.0),
                        entry_price=event_data.get("entry_price", 0.0)
                    )"""

NEW_OUTCOME = """                    outcome = TradeOutcome(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        action=event_data.get("action", "unknown"),
                        confidence=event_data.get("confidence", 0.5),
                        pnl=pnl_percent / 100.0,  # Convert to decimal
                        position_size=event_data.get("position_size", 0.0),
                        entry_price=event_data.get("entry_price", 0.0),
                        exit_price=event_data.get("exit_price", event_data.get("entry_price", 0.0)),
                        duration_seconds=float(event_data.get("duration_seconds", 0.0)),
                        regime=event_data.get("regime", "unknown"),
                        model_votes=event_data.get("model_breakdown", {}),
                        setup_hash=event_data.get("setup_hash", symbol),
                    )"""

if OLD_OUTCOME in content:
    content = content.replace(OLD_OUTCOME, NEW_OUTCOME, 1)
    print("FIX2: TradeOutcome patched - 5 missing args added")
else:
    print("FIX2: WARN - OLD pattern not found exactly, trying flexible match")
    # Try with flexible whitespace
    pattern = r'outcome = TradeOutcome\(\s+timestamp=datetime\.now\(timezone\.utc\),\s+symbol=symbol,\s+action=event_data\.get\("action", "unknown"\),\s+confidence=event_data\.get\("confidence", 0\.5\),\s+pnl=pnl_percent / 100\.0,.*?entry_price=event_data\.get\("entry_price", 0\.0\)\s+\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        print(f"FIX2: Found at char {match.start()}")
        print("FIX2: Manual patch needed - review service.py")
    else:
        print("FIX2: Pattern not found - already patched or structure changed")

# ============================================================
# FIX 5a: equity_usd hardcode at the RL sizer block
# ============================================================
OLD_EQUITY_A = """                # TODO: Get real account equity from execution-service
                equity_usd = 10000.0  # Default $10K account"""

NEW_EQUITY_A = """                # Read real account equity from Redis (quantum:equity:current)
                try:
                    _eq_raw = await self.redis_client.hget("quantum:equity:current", "equity")
                    equity_usd = float(_eq_raw) if _eq_raw else 10000.0
                except Exception:
                    equity_usd = 10000.0  # Fallback if Redis unavailable"""

if OLD_EQUITY_A in content:
    content = content.replace(OLD_EQUITY_A, NEW_EQUITY_A, 1)
    print("FIX5a: equity_usd (RL sizer block) patched to use Redis")
else:
    print("FIX5a: WARN - equity_usd pattern A not found exactly")
    # Search for approximate match
    idx = content.find("equity_usd = 10000.0  # Default $10K account")
    if idx >= 0:
        print(f"FIX5a: Found at char {idx}, context:")
        print(repr(content[idx-150:idx+60]))
    else:
        print("FIX5a: Not found - already patched or line changed")

# ============================================================
# FIX 5b: account_equity hardcode at the AI sizer bridge patch
# ============================================================
OLD_EQUITY_B = """                # TODO: Get real account equity from Binance account info
                account_equity = 10000.0  # Placeholder"""

NEW_EQUITY_B = """                # Read real account equity from Redis (quantum:equity:current)
                try:
                    _eq_raw_b = await self.redis_client.hget("quantum:equity:current", "equity")
                    account_equity = float(_eq_raw_b) if _eq_raw_b else 10000.0
                except Exception:
                    account_equity = 10000.0  # Fallback if Redis unavailable"""

if OLD_EQUITY_B in content:
    content = content.replace(OLD_EQUITY_B, NEW_EQUITY_B, 1)
    print("FIX5b: account_equity (AI sizer bridge) patched to use Redis")
else:
    print("FIX5b: WARN - account_equity pattern B not found exactly")
    idx = content.find("account_equity = 10000.0  # Placeholder")
    if idx >= 0:
        print(f"FIX5b: Found at char {idx}, context:")
        print(repr(content[idx-150:idx+60]))
    else:
        print("FIX5b: Not found - already patched or line changed")

# Write back
with open(FILE, "w") as f:
    f.write(content)

print("DONE: service.py written")
