"""Route handlers package.

Having this file present guarantees `backend.routes` imports resolve cleanly
under both plain execution and test discovery.
"""

# This file intentionally left minimal.
"""backend.routes package.

Import the submodules at package import time so that router-side-effects are
registered predictably. This avoids import-order issues where tests may import
``backend.main:app`` before submodules are loaded which can cause routes like
``/candles`` to be missing in some CI/test harnesses.
"""

# Import submodules to ensure their top-level router objects and handlers are
# created when the package is imported. Keep the list explicit so linters
# and tooling can see what's exported.
# Switch to lazy import pattern to avoid cascading failures during early
# test discovery when optional dependencies or configuration may not be
# available. Individual modules should import the specific route modules
# they need.

__all__ = []  # Explicitly empty; routers imported by backend.main on startup
