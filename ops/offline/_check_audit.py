"""
Check Redis audit stream for PATCH-11 AIJudge evidence.
"""
import subprocess
import json

result = subprocess.run(
    ["redis-cli", "XREVRANGE", "quantum:stream:exit.audit", "+", "-", "COUNT", "3"],
    capture_output=True, text=True
)
print("=== Recent audit stream entries ===")
print(result.stdout[:4000])
print()

# Also check if judge_validator can actually be imported as part of the package
import sys
sys.path.insert(0, "/opt/quantum")
import importlib
print("=== Package import test ===")
try:
    import microservices.exit_management_agent.judge_validator as jv
    print("judge_validator OK, validate:", hasattr(jv, "validate"))
except Exception as exc:
    print(f"judge_validator FAIL: {exc}")

try:
    import microservices.exit_management_agent.ai_judge as aj
    print("ai_judge OK, AIJudge:", hasattr(aj, "AIJudge"))
except Exception as exc:
    print(f"ai_judge FAIL: {exc}")
