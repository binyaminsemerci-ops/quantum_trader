#!/usr/bin/env python3
"""Start Quantum Trader Backend with AI Auto Trading.

This script starts the backend server with AI auto trading capabilities.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    # Check if we're in the right directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        return False

    # Check if simple_main.py exists
    main_file = backend_dir / "simple_main.py"
    if not main_file.exists():
        return False

    # Check if ai_auto_trading_service.py exists
    ai_service_file = Path("ai_auto_trading_service.py")
    return ai_service_file.exists()


def start_backend():
    """Start the backend server."""
    try:
        # Change to backend directory and start server
        os.chdir("backend")

        # Start the server
        process = subprocess.run([sys.executable, "simple_main.py"], check=False)

        return process.returncode

    except KeyboardInterrupt:
        return 0
    except Exception:
        return 1


def main():

    if not check_dependencies():
        return 1

    input("Press Enter to start the backend server...")

    return start_backend()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
