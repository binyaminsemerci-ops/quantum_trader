path = "/opt/quantum/microservices/exit_management_agent/models.py"
with open(path) as f:
    txt = f.read()
idx = txt.find("market_regime")
# show from first occurrence to end of second
print(repr(txt[idx:idx+250]))
