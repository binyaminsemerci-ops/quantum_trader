path = "/opt/quantum/microservices/exit_management_agent/decision_engine.py"
with open(path) as f:
    txt = f.read()
idx = txt.find("Adaptive harvest")
print(repr(txt[idx-4:idx+150]))
