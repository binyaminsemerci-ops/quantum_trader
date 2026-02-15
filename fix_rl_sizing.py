import re

# Read the file
with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'r') as f:
    content = f.read()

# Find and replace the buggy code block
old_block = '''            # Get RL agent decision
            action = self.rl_agent.get_action(state)

            # Default sizing if RL fails
            position_usd = min(self.max_position_usd, 300.0)
            leverage = 2.0
            tp_pct = 2.0
            sl_pct = 1.0

            if action:
                # Parse RL output (position%, leverage, tp%, sl%)
                position_usd = min(self.max_position_usd, action.get("position_size_usd", 300.0))
                leverage = max(1.0, min(5.0, action.get("leverage", 2.0)))
                tp_pct = action.get("tp_pct", 2.0)
                sl_pct = action.get("sl_pct", 1.0)'''

new_block = '''            # Get RL position size multiplier (returns float 0.5-1.5)
            multiplier = self.rl_agent.get_position_size_multiplier(state)
            logger.info(f"[RL-Sizing] {opportunity.symbol}: multiplier={multiplier:.2f}")

            # Apply multiplier to base sizing
            base_position = 200.0  # Base position USD
            position_usd = min(self.max_position_usd, base_position * multiplier)
            leverage = 2.0  # Fixed leverage for now
            tp_pct = 2.0
            sl_pct = 1.0'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'w') as f:
        f.write(content)
    print('FIXED: Replaced get_action with get_position_size_multiplier')
else:
    print('Block not found exactly - checking for get_action...')
    if 'get_action' in content:
        # Simple replacement
        content = content.replace('self.rl_agent.get_action(state)', 'self.rl_agent.get_position_size_multiplier(state)')
        with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'w') as f:
            f.write(content)
        print('PARTIAL FIX: Replaced get_action call')
    else:
        print('No get_action found')
