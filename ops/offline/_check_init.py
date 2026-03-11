import os
base = "/opt/quantum/microservices/exit_management_agent"
# check __init__.py
init = os.path.join(base, "__init__.py")
if os.path.exists(init):
    print("__init__.py exists:")
    print(open(init).read()[:500])
else:
    print("no __init__.py")

# also check if anything imports __all__ or star imports
for f in sorted(os.listdir(base)):
    if f.endswith(".py") and f != "__init__.py":
        src = open(os.path.join(base, f)).read()
        if "import *" in src:
            print(f"STAR IMPORT in {f}")
