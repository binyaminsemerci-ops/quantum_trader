#!/usr/bin/env python3
"""Fix the one-shot filter patch in trade_intent_subscriber.py"""

# Read file
with open("/home/qt/quantum_trader/backend/events/subscribers/trade_intent_subscriber.py", "r") as f:
    content = f.read()

# Remove any mangled insertions
bad_patterns = [
    r'# ONE-SHOT MODE:.*?(?=\n            if should_skip)',
]

import re
for pattern in bad_patterns:
    content = re.sub(pattern, '', content, flags=re.DOTALL)

# Find insertion point
lines = content.split("\n")
insert_idx = None
for i, line in enumerate(lines):
    if "should_skip_execution = self.safe_drain_mode or is_stale" in line:
        # Check if next line already has ONE-SHOT comment
        if i + 1 < len(lines) and "ONE-SHOT MODE" in lines[i + 1]:
            print("Filter already exists, skipping")
            exit(0)
        insert_idx = i + 1
        break

if not insert_idx:
    print("ERROR: Could not find insertion point")
    exit(1)

# Insert ONE-SHOT filter
one_shot_filter = """
            # 🎯 ONE-SHOT MODE: Filter by source tag
            import os as _os
            _one_shot_source = _os.getenv("TRADE_INTENT_ONE_SHOT_SOURCE", "")
            if _one_shot_source:
                _actual_source = payload.get("source", "")
                if _actual_source != _one_shot_source:
                    self.logger.info(f"[trade_intent] 🎯 ONE-SHOT: Skipping source={_actual_source} != {_one_shot_source}")
                    return
                self.logger.info(f"[trade_intent] 🎯 ONE-SHOT: Matched source={_one_shot_source}")
"""

lines.insert(insert_idx, one_shot_filter)
content = "\n".join(lines)

# Write back
with open("/home/qt/quantum_trader/backend/events/subscribers/trade_intent_subscriber.py", "w") as f:
    f.write(content)

print(f"✅ SUCCESS: One-shot filter inserted at line {insert_idx}")
print("Lines 80-95:")
for i in range(80, min(95, len(lines))):
    print(f"  {i}: {lines[i][:60]}")
