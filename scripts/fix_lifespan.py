#!/usr/bin/env python3
"""Fix FastAPI lifespan for exit_monitor_service.py"""

import re

filepath = "/home/qt/quantum_trader/services/exit_monitor_service.py"

with open(filepath, "r") as f:
    content = f.read()

# Step 1: Add contextlib import after asyncio/logging
if "from contextlib import asynccontextmanager" not in content:
    content = content.replace(
        "import asyncio\nimport logging",
        "import asyncio\nimport logging\nfrom contextlib import asynccontextmanager"
    )
    print("✅ Added asynccontextmanager import")

# Step 2: Replace app creation with lifespan version
old_app_line = 'app = FastAPI(title="Exit Monitor Service", version="1.0.0")'

if old_app_line in content:
    new_app_section = '''@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    await shutdown()

app = FastAPI(title="Exit Monitor Service", version="1.0.0", lifespan=lifespan)'''
    
    content = content.replace(old_app_line, new_app_section)
    print("✅ Replaced app creation with lifespan")

# Step 3: Remove @app.on_event decorators
content = content.replace('@app.on_event("startup")\n', '')
content = content.replace('@app.on_event("shutdown")\n', '')
print("✅ Removed deprecated @app.on_event decorators")

# Write back
with open(filepath, "w") as f:
    f.write(content)

print("✅ Phase 1 Complete: Lifespan fixed - startup() will now be called properly")
