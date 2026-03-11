path = "/opt/quantum/microservices/exit_management_agent/perception.py"
with open(path) as f:
    txt = f.read()

# Show decision_engine Rule 4 area
idx = txt.find("return PerceptionResult(")
block = txt[idx:idx+450]
print("=== Full return block ===")
print(repr(block))
