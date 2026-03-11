import pathlib
SEP = "=" * 60
for path in [
    "/opt/quantum/microservices/execution/exit_brain_v3/dynamic_tp_calculator.py",
    "/opt/quantum/microservices/execution/exit_brain_v3/tp_profiles_v3.py",
]:
    p = pathlib.Path(path)
    if p.exists():
        print(SEP)
        print("FILE: " + path)
        print(SEP)
        print(p.read_text())
    else:
        print("MISSING: " + path)
