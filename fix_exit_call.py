#!/usr/bin/env python3
"""Fix _execute_exit call in autonomous_trader.py"""

filepath = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(filepath, "r") as f:
    lines = f.readlines()

# Find and fix the line with wrong _execute_exit call
fixed = False
for i, line in enumerate(lines):
    if "_execute_exit(position=position, action=action" in line:
        # Replace the single problematic line with two corrected lines
        lines[i] = '            decision = ExitDecision(action=action, percentage=percentage, reason=f"ai_event:{reason}", exit_score=exit_score, hold_score=hold_score)\n'
        lines.insert(i+1, '            await self._execute_exit(position, decision)\n')
        print(f"Fixed line {i+1}")
        fixed = True
        break

if fixed:
    with open(filepath, "w") as f:
        f.writelines(lines)
    print("File updated successfully")
else:
    print("Target line not found - maybe already fixed?")
