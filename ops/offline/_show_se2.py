path = "/opt/quantum/microservices/exit_management_agent/scoring_engine.py"
with open(path) as f:
    t = f.read()
idx = t.find("action, urgency, reason = _apply_decision_map")
print("=== _apply_decision_map call + return ===")
print(repr(t[idx:idx+1200]))
