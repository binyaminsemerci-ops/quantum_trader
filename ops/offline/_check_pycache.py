import os, time

base = "/opt/quantum/microservices/exit_management_agent"
pycache = os.path.join(base, "__pycache__")

print("=== pycache files and freshness ===")
if os.path.isdir(pycache):
    for f in sorted(os.listdir(pycache)):
        pyc_path = os.path.join(pycache, f)
        pyc_mtime = os.path.getmtime(pyc_path)
        # find matching .py file
        py_name = f.split(".")[0] + ".py"
        py_path = os.path.join(base, py_name)
        if os.path.exists(py_path):
            py_mtime = os.path.getmtime(py_path)
            diff = pyc_mtime - py_mtime
            status = "FRESH" if diff >= -0.1 else "STALE"
            print(f"  {f}: diff={diff:.2f}s [{status}]")
        else:
            print(f"  {f}: no .py source")
else:
    print("  No __pycache__ directory")
    
# Also test: what class is self._qwen3?
# We'll import and check at runtime
import sys
sys.path.insert(0, "/opt/quantum")
try:
    from microservices.exit_management_agent.ai_judge import AIJudge
    from microservices.exit_management_agent.groq_client import GroqModelClient
    print(f"\nAIJudge importable: {AIJudge}")
    print(f"GroqModelClient importable: {GroqModelClient}")
except Exception as exc:
    print(f"Import failed: {exc}")
