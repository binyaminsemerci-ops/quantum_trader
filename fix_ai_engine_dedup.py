#!/usr/bin/env python3
"""
AI Engine Deduplication Fix
============================

Problem: AI Engine publiserer flere identiske trade.intent signaler per sekund
for samme symbol, noe som fører til mange små ordre i stedet for én konsolidert.

Root cause: generate_signal() kalles for hver market.tick event (mange per sekund),
og hver gang publiseres et nytt signal uten deduplication.

Solution: Legg til symbol + side basert deduplication med kort TTL (5 sek)
før publisering til trade.intent stream.
"""

import sys

file_path = "/home/qt/quantum_trader/service.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''            print(f"[DEBUG] About to publish trade.intent: {trade_intent_payload}")
            await self.event_bus.publish("trade.intent", trade_intent_payload)
            print(f"[DEBUG] trade.intent published to Redis!")'''

new_code = '''            # DEDUPLICATION: Prevent multiple identical signals within 5 seconds
            dedup_key = f"quantum:dedup:ai_signal:{symbol}:{action}"
            was_set = self.redis.set(
                dedup_key,
                "1",
                nx=True,  # Only set if not exists
                ex=5  # 5 second TTL (short window for market ticks)
            )
            
            if not was_set:
                logger.debug(f"[AI-ENGINE] 🔁 DUPLICATE_SKIP: {symbol} {action} (already published in last 5s)")
                return decision
            
            print(f"[DEBUG] About to publish trade.intent: {trade_intent_payload}")
            await self.event_bus.publish("trade.intent", trade_intent_payload)
            print(f"[DEBUG] trade.intent published to Redis!")'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Added deduplication to AI Engine service.py")
    sys.exit(0)
else:
    print("❌ Could not find code to replace")
    print("Searching for alternative pattern...")
    sys.exit(1)
