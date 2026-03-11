import pathlib
f = pathlib.Path("/opt/quantum/microservices/exit_management_agent/perception.py")
src = f.read_text()
idx = src.find("_get_r_targets(snapshot.leverage)")
print(repr(src[idx:idx+350]))
