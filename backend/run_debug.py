"""Debug runner that starts the FastAPI app with uvicorn.run and prints exceptions.

This is used to capture any exception that occurs during import/startup in a
more debuggable way than invoking `python -m uvicorn` from the shell.
"""
import sys
import traceback

try:
    import os
    import uvicorn
    import sys

    # Ensure project root is on sys.path so `import backend` works when run from
    # the repository root.
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    from backend.main import app

    print("Starting uvicorn via run_debug...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
except Exception:
    print("Exception starting uvicorn:")
    traceback.print_exc()
    sys.exit(1)
