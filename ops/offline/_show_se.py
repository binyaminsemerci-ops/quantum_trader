path = "/opt/quantum/microservices/exit_management_agent/scoring_engine.py"
with open(path) as f:
    t = f.read()
idx = t.find("def _apply_decision_map")
print(repr(t[idx:idx+2000]))
