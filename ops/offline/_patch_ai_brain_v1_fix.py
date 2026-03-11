#!/usr/bin/env python3
"""Fix the one failed patch: replay_writer.py evaluator call before xadd."""
import sys

path = "/opt/quantum/microservices/exit_management_agent/replay_writer.py"
content = open(path).read()

OLD = "        try:\n            await self._redis.xadd(self._replay_stream, record)"
NEW = (
    "        # PATCH-11: DeepSeek-R1 offline evaluation (post-trade verdict).\n"
    "        if self._evaluator is not None:\n"
    "            try:\n"
    "                eval_fields = await self._evaluator.evaluate_replay(record)\n"
    "                record.update(eval_fields)\n"
    "            except Exception as _eval_exc:\n"
    "                _log.warning(\"PATCH-11: DeepSeek eval error for %s: %s\", symbol, _eval_exc)\n"
    "        try:\n"
    "            await self._redis.xadd(self._replay_stream, record)"
)

count = content.count(OLD)
if count == 0:
    print("FAIL: oldString not found")
    sys.exit(1)
if count > 1:
    print(f"FAIL: {count} matches")
    sys.exit(1)

open(path, "w").write(content.replace(OLD, NEW))
print("OK   replay_writer.py: call DeepSeek evaluator before xadd")
print("\nPATCH-11 fix complete")
