#!/usr/bin/env python3
"""
Start Quantum Trader Backend with AI Auto Trading

This script starts the backend server with AI auto trading capabilities.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")

    # Check if we're in the right directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print(
            "❌ Backend directory not found. Please run from quantum_trader root directory."
        )
        return False

    # Check if simple_main.py exists
    main_file = backend_dir / "simple_main.py"
    if not main_file.exists():
        print("❌ simple_main.py not found in backend directory.")
        return False

    # Check if ai_auto_trading_service.py exists
    ai_service_file = Path("ai_auto_trading_service.py")
    if not ai_service_file.exists():
        print("❌ ai_auto_trading_service.py not found in root directory.")
        return False

    print("✅ All required files found")
    return True


def start_backend():
    """Start the backend server"""
    print("🚀 Starting Quantum Trader Backend with AI Auto Trading...")
    print("📍 Backend will be available at: http://127.0.0.1:8001")
    print("🧠 AI Auto Trading endpoints will be available at: /api/v1/ai-trading/*")
    print("📊 WebSocket for AI updates: ws://127.0.0.1:8001/ws/ai-trading")
    print("")

    try:
        # Change to backend directory and start server
        os.chdir("backend")

        # Start the server
        print("⏳ Initializing backend server...")
        process = subprocess.run([sys.executable, "simple_main.py"], check=False)

        return process.returncode

    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return 1


def main():
    print("🤖 Quantum Trader AI Auto Trading Backend Launcher")
    print("=" * 55)

    if not check_dependencies():
        return 1

    print("\n💡 Tips:")
    print("   - Use Ctrl+C to stop the backend server")
    print("   - Access the API documentation at http://127.0.0.1:8001/docs")
    print("   - Test AI trading with: python test_ai_trading_integration.py")
    print("   - Start frontend with: cd frontend && npm run dev")
    print("")

    input("Press Enter to start the backend server...")

    return start_backend()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
