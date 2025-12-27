import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Quantum Trader Dashboard Backend...")
    print(f"📁 Backend directory: {backend_dir}")
    print(f"🐍 Python path: {sys.path[:3]}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for stability during testing
        log_level="info"
    )
