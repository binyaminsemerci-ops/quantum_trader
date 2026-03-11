import pathlib
SEP = "=" * 60
for path in [
    "/opt/quantum/microservices/exit_management_agent/perception.py",
    "/opt/quantum/microservices/exit_management_agent/models.py",
    "/opt/quantum/microservices/exit_management_agent/intent_writer.py",
]:
    p = pathlib.Path(path)
    if p.exists():
        print(SEP)
        print("FILE: " + path)
        print(SEP)
        print(p.read_text())
    else:
        print("MISSING: " + path)
