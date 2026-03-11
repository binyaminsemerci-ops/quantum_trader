base = "/opt/quantum/microservices/exit_management_agent"
import os
for f in sorted(os.listdir(base)):
    if f.endswith(".py"):
        src = open(os.path.join(base, f)).read()
        if "rate_throttled" in src or "qwen3_rate_throttled" in src:
            for i, l in enumerate(src.splitlines(), 1):
                if "rate_throttled" in l or "qwen3_rate" in l:
                    print(f"{f}:{i}: {l.strip()}")
