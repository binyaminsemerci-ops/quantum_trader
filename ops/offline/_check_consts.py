base = "/opt/quantum/microservices/exit_management_agent"

# Check scoring_engine.py lines around constants definition (lines 55-105)
p = f"{base}/scoring_engine.py"
lines = open(p).readlines()
print("scoring_engine.py lines 55-105:")
for i, l in enumerate(lines[54:104], start=55):
    print(i, l.rstrip())

# Check if qwen3_layer.Qwen3Layer is used anywhere
print("\n\nAll qwen3_layer usages:")
import os
for f in sorted(os.listdir(base)):
    if f.endswith(".py"):
        src = open(os.path.join(base, f)).read()
        for i, l in enumerate(src.splitlines(), 1):
            if "Qwen3Layer" in l and f not in ("qwen3_layer.py", "ai_brain.py"):
                print(f"  {f}:{i}: {l.strip()}")
