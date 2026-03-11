base = "/opt/quantum/microservices/exit_management_agent"
p = f"{base}/main.py"
lines = open(p).readlines()
print("main.py _tick AI path (lines 290-370):")
for i, l in enumerate(lines[288:370], start=289):
    print(i, l.rstrip())
