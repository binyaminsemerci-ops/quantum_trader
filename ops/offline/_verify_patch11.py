"""
Verify PATCH-11 integrity and imports.
Run on VPS: python3 /tmp/verify_patch11.py
"""
import sys
import os
import importlib.util

base = "/opt/quantum/microservices/exit_management_agent"
sys.path.insert(0, "/opt/quantum")
os.chdir("/opt/quantum")

print("=== PATCH-11 IMPORT VERIFICATION ===\n")

# 1. Check pycache for main.py — report modification timestamps
pycache = os.path.join(base, "__pycache__")
main_py = os.path.join(base, "main.py")
main_pyc_candidates = []
if os.path.isdir(pycache):
    for f in os.listdir(pycache):
        if f.startswith("main."):
            main_pyc_candidates.append(f)
            pyc_mtime = os.path.getmtime(os.path.join(pycache, f))
            py_mtime = os.path.getmtime(main_py)
            status = "STALE" if pyc_mtime < py_mtime else "FRESH"
            print(f"  pycache/main.* = {f} [{status}]")

# 2. Try to import the module
print()
print("Attempting module imports:")
try:
    spec = importlib.util.spec_from_file_location(
        "microservices.exit_management_agent.groq_client",
        os.path.join(base, "groq_client.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  [OK] groq_client.py — GroqModelClient:", hasattr(mod, "GroqModelClient"))
except Exception as exc:
    print(f"  [FAIL] groq_client.py: {exc}")

try:
    spec = importlib.util.spec_from_file_location(
        "microservices.exit_management_agent.judge_validator",
        os.path.join(base, "judge_validator.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  [OK] judge_validator.py — validate:", hasattr(mod, "validate"))
except Exception as exc:
    print(f"  [FAIL] judge_validator.py: {exc}")

# 3. Check main.py imports section
print()
print("main.py import lines:")
for i, l in enumerate(open(main_py).readlines(), 1):
    if l.startswith("from .") or l.startswith("import "):
        print(f"  {i}: {l.rstrip()}")

# 4. Check config fallback_model field
cfg_path = os.path.join(base, "config.py")
cfg_src = open(cfg_path).read()
print()
print("config.py PATCH-11 fields detected:")
for field in ["fallback_model", "evaluator_model", "judge_confidence_threshold",
              "mistral_model", "deepseek_model"]:
    present = field + ":" in cfg_src or field + " =" in cfg_src
    print(f"  {field}: {'FOUND' if present else 'MISSING'}")
