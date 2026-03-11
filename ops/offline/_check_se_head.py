p = "/opt/quantum/microservices/exit_management_agent/scoring_engine.py"
src = open(p).read()
lines = src.splitlines()
for i, l in enumerate(lines[:30], start=1):
    print(i, l)
