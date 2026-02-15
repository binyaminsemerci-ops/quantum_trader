#!/usr/bin/env python3
"""
Fix RL Position Sizing Agent - Correct method call
The method is get_position_size_multiplier(), not get_action()
"""

file_path = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(file_path, "r") as f:
    content = f.read()

changes = 0

# Fix 1: Replace get_action with get_position_size_multiplier
if "self.rl_agent.get_action(state)" in content:
    content = content.replace(
        "self.rl_agent.get_action(state)",
        "self.rl_agent.get_position_size_multiplier(state)"
    )
    changes += 1
    print("✅ Fix 1: Replaced get_action with get_position_size_multiplier")

# Fix 2: The old code expects a dict from get_action, but multiplier returns float
# Need to change the logic that parses the action
old_parsing = """            if action:
                # Parse RL output (position%, leverage, tp%, sl%)
                position_usd = min(self.max_position_usd, action.get("position_size_usd", 300.0))
                leverage = max(1.0, min(5.0, action.get("leverage", 2.0)))
                tp_pct = action.get("tp_pct", 2.0)
                sl_pct = action.get("sl_pct", 1.0)"""

new_parsing = """            # action is now a float multiplier [0.5-1.5], not a dict
            if action is not None:
                base_size = 300.0
                position_usd = min(self.max_position_usd, base_size * action)
                # Leverage based on confidence
                leverage = 3.0 if opportunity.confidence >= 0.80 else 2.0"""

if old_parsing in content:
    content = content.replace(old_parsing, new_parsing)
    changes += 1
    print("✅ Fix 2: Adapted parsing for float multiplier")

# Write changes
if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} fixes to autonomous_trader.py")
else:
    print("\n❌ No changes applied - patterns not found")
