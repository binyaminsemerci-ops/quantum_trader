p = "/opt/quantum/microservices/exit_management_agent/replay_writer.py"
src = open(p).read()
lines = src.splitlines()
for i, l in enumerate(lines, start=1):
    if "evaluator" in l.lower() or "deepseek" in l.lower() or "evaluate_replay" in l.lower():
        print(i, l)
