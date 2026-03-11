import subprocess
r = subprocess.run(["find", "/opt/quantum/microservices", "-name", "*.py"], capture_output=True, text=True)
for l in sorted(r.stdout.splitlines()):
    print(l)
print("---COMMON---")
r2 = subprocess.run(["find", "/opt/quantum/common", "-name", "*.py"], capture_output=True, text=True)
for l in sorted(r2.stdout.splitlines()):
    print(l)
