import os, subprocess

# Check service file
result = subprocess.run(
    ["cat", "/etc/systemd/system/quantum-exit-management-agent.service"],
    capture_output=True, text=True
)
print("=== SERVICE FILE ===")
for l in result.stdout.splitlines():
    if any(k in l for k in ["Exec", "Work", "User", "Env", "python", "Group"]):
        print(l)

# Check working dir code
svc_dir = "/home/qt/quantum_trader/microservices/exit_management_agent"
print("\n=== WORKING DIR AGENT FILES ===")
if os.path.isdir(svc_dir):
    files = sorted(os.listdir(svc_dir))
    print(f"Files ({len(files)}):", files[:20])
    
    # Check main.py imports in working dir
    main_path = os.path.join(svc_dir, "main.py")
    if os.path.exists(main_path):
        for i, l in enumerate(open(main_path).readlines(), 1):
            if any(k in l for k in ["qwen3_layer", "AIBrain", "AIJudge", "GroqModel", "ai_judge", "groq_client", "ai_brain"]):
                print(f"  main.py:{i}: {l.rstrip()}")
else:
    print("Directory does not exist!")
