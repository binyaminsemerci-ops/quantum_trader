import os
base = "/opt/quantum/microservices/exit_management_agent"
for f in sorted(os.listdir(base)):
    if f.endswith(".py"):
        src = open(os.path.join(base, f)).read()
        if "qwen3_layer" in src or "ai_brain" in src:
            # Find the lines
            for i, l in enumerate(src.splitlines(), 1):
                if "qwen3_layer" in l or "ai_brain" in l:
                    print(f"  {f}:{i}: {l.strip()}")
