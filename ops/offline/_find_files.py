import subprocess, os
r = subprocess.run(["find", "/opt/quantum", "-name", "*.py", "-newer", "/tmp"], capture_output=True, text=True)
lines = [l for l in r.stdout.splitlines() if any(k in l for k in ["math","harvest","brain","rl_","dynamic","formula","exit_calc","calc_exit"])]
print("\n".join(lines[:40]))
print("---ALL PY FILES---")
r2 = subprocess.run(["find", "/opt/quantum", "-name", "*.py"], capture_output=True, text=True)
for l in sorted(r2.stdout.splitlines()):
    print(l)
