#!/usr/bin/env python3
"""Fix ExitDecision constructor with all required args"""

filepath = "/home/qt/quantum_trader/microservices/autonomous_trader/autonomous_trader.py"

with open(filepath, "r") as f:
    content = f.read()

old_line = 'decision = ExitDecision(action=action, percentage=percentage, reason=f"ai_event:{reason}", exit_score=exit_score, hold_score=hold_score)'
new_line = 'decision = ExitDecision(symbol=symbol, action=action, percentage=percentage, reason=f"ai_event:{reason}", hold_score=hold_score, exit_score=exit_score, factors={"ai_stream": True})'

content = content.replace(old_line, new_line)

with open(filepath, "w") as f:
    f.write(content)

print("Fixed ExitDecision constructor!")
