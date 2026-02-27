"""
Proof script: pull live logs from VPS and compute per-agent prediction stats.
"""
import re
import subprocess
import numpy as np

result = subprocess.run(
    ["wsl", "bash", "-c",
     "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ensemble-predictor -n 5000 --no-pager' 2>&1"],
    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40
)
lines = result.stdout.splitlines()
print(f"Lines fetched: {len(lines)}")

stats = {}

for line in lines:
    # LGBM
    m = re.search(r"LGBM \w+: (BUY|SELL|HOLD) \(conf=([\d.]+)", line)
    if m:
        agent, action, conf = "LGBM", m.group(1), float(m.group(2))
        stats.setdefault(agent, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[agent][action] += 1
        stats[agent]["confs"].append(conf)

    # XGB
    m = re.search(r"XGB \w+: (BUY|SELL|HOLD) ([\d.]+)%", line)
    if m:
        agent, action, conf = "XGB", m.group(1), float(m.group(2)) / 100
        stats.setdefault(agent, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[agent][action] += 1
        stats[agent]["confs"].append(conf)

    # NHiTS
    m = re.search(r"N-HiTS \w+: (BUY|SELL|HOLD) \(conf=([\d.]+)", line)
    if m:
        agent, action, conf = "NHiTS", m.group(1), float(m.group(2))
        stats.setdefault(agent, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[agent][action] += 1
        stats[agent]["confs"].append(conf)

    # PatchTST
    m = re.search(r"PatchTST[^\|]+: (BUY|SELL|HOLD) \(conf=([\d.]+)", line)
    if m:
        agent, action, conf = "PatchTST", m.group(1), float(m.group(2))
        stats.setdefault(agent, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[agent][action] += 1
        stats[agent]["confs"].append(conf)

    # TFT: "[TFT-Agent] [INFO] ... | BTCUSDT → BUY (TFT, conf=0.418)"
    # The → (U+2192) may be mangled on Windows; use .{1,5} to tolerate encoding variants
    m = re.search(r"TFT-Agent.*\| \w+.{1,5}(BUY|SELL|HOLD) \(TFT, conf=([\d.]+)\)", line)
    if m:
        agent, action, conf = "TFT", m.group(1), float(m.group(2))
        stats.setdefault(agent, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[agent][action] += 1
        stats[agent]["confs"].append(conf)

print()
print("=" * 68)
print(f"{'AGENT':<12} {'SELL':>7} {'HOLD':>7} {'BUY':>7}  {'TOTAL':>6}  {'AVG CONF':>9}  VERDICT")
print("-" * 68)

order = ["LGBM", "XGB", "NHiTS", "PatchTST", "TFT"]
all_ok = True
for agent in order:
    if agent not in stats:
        print(f"{agent:<12}  ⚠️  NOT SEEN IN LOGS")
        all_ok = False
        continue
    s = stats[agent]
    total = s["BUY"] + s["SELL"] + s["HOLD"]
    avg_conf = float(np.mean(s["confs"])) if s["confs"] else 0
    sell_pct = s["SELL"] / total * 100 if total else 0
    hold_pct = s["HOLD"] / total * 100 if total else 0
    buy_pct  = s["BUY"]  / total * 100 if total else 0
    max_pct  = max(sell_pct, hold_pct, buy_pct)
    # OK if all 3 classes appear and no single class > 85%
    ok = s["BUY"] > 0 and s["SELL"] > 0 and max_pct < 85
    verdict = "✅ OK" if ok else "❌ BIASED"
    if not ok:
        all_ok = False
    print(f"{agent:<12} {sell_pct:6.1f}% {hold_pct:6.1f}% {buy_pct:6.1f}%  {total:6}   {avg_conf:.3f}      {verdict}")

print("=" * 68)
print()
if all_ok:
    print("✅ ALL 5 AGENTS PREDICTING CORRECTLY — no single-class bias detected.")
else:
    print("⚠️  Some agents flagged above.")
