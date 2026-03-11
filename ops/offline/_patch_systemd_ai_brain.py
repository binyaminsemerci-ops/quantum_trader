#!/usr/bin/env python3
"""PATCH-11: Update systemd service file env vars for 3-tier AI brain."""
import sys

SERVICE = "/etc/systemd/system/quantum-exit-management-agent.service"
content = open(SERVICE).read()

# 1. Change primary model: llama -> qwen3-32b
OLD1 = "Environment=EXIT_AGENT_QWEN3_MODEL=llama-3.3-70b-versatile"
NEW1 = "Environment=EXIT_AGENT_QWEN3_MODEL=qwen/qwen3-32b"
if OLD1 not in content:
    print(f"FAIL: '{OLD1}' not found in service file")
    sys.exit(1)
content = content.replace(OLD1, NEW1)
print("OK   PRIMARY model -> qwen/qwen3-32b")

# 2. Add Mistral + DeepSeek env vars after the primary model line
INSERT_AFTER = "Environment=EXIT_AGENT_QWEN3_MODEL=qwen/qwen3-32b"
MISTRAL_DEEPSEEK = (
    "\n"
    "# ── PATCH-11: Tier-2 fallback (Mistral Small 3.1) + Tier-3 evaluator (DeepSeek-R1) ──\n"
    "Environment=EXIT_AGENT_MISTRAL_MODEL=mistral-small-3.1-24b-instruct\n"
    "Environment=EXIT_AGENT_MISTRAL_TIMEOUT_MS=2000\n"
    "Environment=EXIT_AGENT_DEEPSEEK_MODEL=deepseek-r1-distill-llama-70b\n"
    "Environment=EXIT_AGENT_DEEPSEEK_TIMEOUT_MS=8000\n"
    "Environment=EXIT_AGENT_DEEPSEEK_ENABLED=true"
)
content = content.replace(INSERT_AFTER, INSERT_AFTER + MISTRAL_DEEPSEEK)
print("OK   Added Mistral + DeepSeek env vars")

open(SERVICE, "w").write(content)
print("\nService file updated")
