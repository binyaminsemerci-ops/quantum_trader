import sys
p = "/opt/quantum/microservices/exit_management_agent/scoring_engine.py"
src = open(p).read()
lines = src.splitlines()
for i, l in enumerate(lines[100:130], start=101):
    print(i, l)
