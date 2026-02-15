# Fix the RL sizing logic - action is now a float multiplier, not a dict
with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'r') as f:
    content = f.read()

old_block = '''            # Get RL agent decision
            action = self.rl_agent.get_position_size_multiplier(state)

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
            leverage = 2.0  # Fixed leverage
            tp_pct = 2.0
            sl_pct = 1.0'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'w') as f:
        f.write(content)
    print('SUCCESS: Full block replaced')
else:
    print('Block not found - trying line by line...')
    # Replace line by line approach
    lines = content.split('\n')
    new_lines = []
    skip_until_return = False
    for line in lines:
        if 'action = self.rl_agent.get_position_size_multiplier(state)' in line:
            # Replace this line and flag to skip the if action block
            new_lines.append('            # Get RL position size multiplier (returns float 0.5-1.5)')
            new_lines.append('            multiplier = self.rl_agent.get_position_size_multiplier(state)')
            new_lines.append('            logger.info(f"[RL-Sizing] {opportunity.symbol}: multiplier={multiplier:.2f}")')
            new_lines.append('')
            new_lines.append('            # Apply multiplier to base sizing')
            new_lines.append('            base_position = 200.0  # Base position USD')
            new_lines.append('            position_usd = min(self.max_position_usd, base_position * multiplier)')
            new_lines.append('            leverage = 2.0  # Fixed leverage')
            new_lines.append('            tp_pct = 2.0')
            new_lines.append('            sl_pct = 1.0')
            skip_until_return = True
        elif skip_until_return:
            if 'return {' in line:
                skip_until_return = False
                new_lines.append(line)
            # Skip old lines
        else:
            new_lines.append(line)
    
    with open('/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py', 'w') as f:
        f.write('\n'.join(new_lines))
    print('SUCCESS: Line by line replacement done')
