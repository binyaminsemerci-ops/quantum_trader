import pathlib
f = pathlib.Path("/opt/quantum/microservices/exit_management_agent/perception.py")
src = f.read_text()
idx = src.find("if _RISK_SETTINGS_AVAILABLE:\n        targets = compute_harvest_r_targets")
print(repr(src[idx:idx+300]))
