p = "/opt/quantum/microservices/exit_management_agent/main.py"
lines = open(p).readlines()
for i, l in enumerate(lines[75:145], start=76):
    print(i, l.rstrip())
