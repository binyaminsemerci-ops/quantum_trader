import pathlib
files = [
    "/opt/quantum/microservices/execution/exit_brain_v3/dynamic_tp_calculator.py",
    "/opt/quantum/microservices/execution/exit_brain_v3/tp_profiles_v3.py",
    "/opt/quantum/microservices/harvest_brain/harvest_brain.py",
    "/opt/quantum/microservices/exit_management_agent/scoring_engine.py",
    "/opt/quantum/microservices/exit_management_agent/decision_engine.py",
]
SEP = "=" * 60
for path in files:
    p = pathlib.Path(path)
    if p.exists():
        print(SEP)
        print("FILE: " + path)
        print(SEP)
        print(p.read_text())
    else:
        print("MISSING: " + path)
