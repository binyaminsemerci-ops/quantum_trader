p = "/opt/quantum/microservices/exit_management_agent/main.py"
lines = open(p).readlines()
for i, l in enumerate(lines, 1):
    if any(k in l for k in ["qwen3_layer", "AIBrain", "AIJudge", "GroqModel", "DeepSeek", "ai_brain", "ai_judge", "groq_client"]):
        print(i, l.rstrip())
