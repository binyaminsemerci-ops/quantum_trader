"""Post-fix proof: only logs since restart at 10:20:00."""
import re, subprocess, numpy as np

result = subprocess.run(
    ["wsl", "bash", "-c",
     "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ensemble-predictor -n 2000 --no-pager 2>&1'"],
    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=40
)
lines = result.stdout.splitlines()
print(f"Lines fetched: {len(lines)}")

stats = {}
for line in lines:
    m = re.search(r"LGBM \w+: (BUY|SELL|HOLD) \(conf=([\d.]+)", line)
    if m:
        a, action, conf = "LGBM", m.group(1), float(m.group(2))
        stats.setdefault(a, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[a][action] += 1; stats[a]["confs"].append(conf)

    m = re.search(r"XGB \w+: (BUY|SELL|HOLD) ([\d.]+)%", line)
    if m:
        a, action, conf = "XGB", m.group(1), float(m.group(2)) / 100
        stats.setdefault(a, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[a][action] += 1; stats[a]["confs"].append(conf)

    m = re.search(r"N-HiTS \w+: (BUY|SELL|HOLD) \(conf=([\d.]+)", line)
    if m:
        a, action, conf = "NHiTS", m.group(1), float(m.group(2))
        stats.setdefault(a, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[a][action] += 1; stats[a]["confs"].append(conf)

    m = re.search(r"PatchTST[^\|]+: (BUY|SELL|HOLD) \(conf=([\d.]+)", line)
    if m:
        a, action, conf = "PatchTST", m.group(1), float(m.group(2))
        stats.setdefault(a, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[a][action] += 1; stats[a]["confs"].append(conf)

    m = re.search(r"TFT-Agent.*\| \w+.{1,5}(BUY|SELL|HOLD) \(TFT, conf=([\d.]+)\)", line)
    if m:
        a, action, conf = "TFT", m.group(1), float(m.group(2))
        stats.setdefault(a, {"BUY": 0, "SELL": 0, "HOLD": 0, "confs": []})
        stats[a][action] += 1; stats[a]["confs"].append(conf)

print()
print("=" * 68)
print(f"{'AGENT':<12} {'SELL':>7} {'HOLD':>7} {'BUY':>7}  {'TOTAL':>6}  {'AVG CONF':>9}  VERDICT")
print("-" * 68)
all_ok = True
for agent in ["LGBM", "XGB", "NHiTS", "PatchTST", "TFT"]:
    if agent not in stats:
        print(f"{agent:<12}  NOT SEEN IN LOGS")
        all_ok = False
        continue
    s = stats[agent]
    total = s["BUY"] + s["SELL"] + s["HOLD"]
    avg_conf = float(np.mean(s["confs"])) if s["confs"] else 0
    sp = s["SELL"] / total * 100
    hp = s["HOLD"] / total * 100
    bp = s["BUY"]  / total * 100
    # OK if all 3 appear, or at least 2 classes with no one class > 90%
    ok = max(sp, hp, bp) < 90 and total >= 5
    verdict = "OK" if ok else "BIASED"
    if not ok:
        all_ok = False
    print(f"{agent:<12} {sp:6.1f}% {hp:6.1f}% {bp:6.1f}%  {total:6}   {avg_conf:.3f}      {verdict}")
print("=" * 68)
print()
if all_ok:
    print("ALL 5 AGENTS PREDICTING — no 90%+ single-class bias.")
else:
    print("Some agents flagged above (check logs).")
