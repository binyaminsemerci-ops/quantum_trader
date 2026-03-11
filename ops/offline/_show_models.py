path = "/opt/quantum/microservices/exit_management_agent/models.py"
with open(path) as f:
    txt = f.read()
idx = txt.find("market_regime")
print("count:", txt.count("market_regime"))
print(repr(txt[idx-4:idx+120]))
