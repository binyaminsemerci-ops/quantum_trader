"""
Check audit stream for NEW entries (post-restart) by looking at recent loop IDs.
"""
import subprocess
import re

# Get journal loop IDs from new process (PID 3895945) - the last few ticks
# Recent tick loop IDs from journal: e54b57a04a4d, c169e11dc18a, etc.
# Check the LATEST 20 entries in audit stream

result = subprocess.run(
    ["redis-cli", "XREVRANGE", "quantum:stream:exit.audit", "+", "-", "COUNT", "30"],
    capture_output=True, text=True
)

entries_raw = result.stdout.strip()
# Parse loop IDs from result
loop_ids = re.findall(r"loop_id\n(\w+)", entries_raw)
patch_vals = re.findall(r"patch\n(\S+)", entries_raw)
qwen3_reasons = re.findall(r"qwen3_reason\n(\S+)", entries_raw)
ts_vals = re.findall(r"\n(\d{13,16}-\d+)\n", entries_raw)

print("Stream IDs (first few):", ts_vals[:5])
print("Loop IDs:", loop_ids[:10])
print("Patch values:", list(set(patch_vals))[:5])
print("qwen3_reasons:", list(set(qwen3_reasons)))

# Check if any entries have AIJudge reason format
aijudge_entries = [r for r in qwen3_reasons if r.startswith("t0:") or r.startswith("t1:") or r.startswith("t2:")]
old_entries = [r for r in qwen3_reasons if r == "qwen3_rate_throttled"]
print(f"\nAIJudge format reasons: {aijudge_entries}")
print(f"Old qwen3_rate_throttled: {len(old_entries)} entries")
print(f"Total entries checked: {len(loop_ids)}")
