#!/usr/bin/env python3
"""
E2E Test Runner - Full System Verification
════════════════════════════════════════════

Runs end-to-end tests from prediction to profit taking with detailed reporting.

Usage:
  # Run full e2e test
  python test_e2e_prediction_to_profit.py
  
  # Or with environment variables set
  BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python test_e2e_prediction_to_profit.py

Requirements:
  - Backend running on http://localhost:8000
  - AI Engine running on http://localhost:8001 (optional)
  - Binance API credentials in environment variables
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime

def run_e2e_test():
    """Run the e2e test script"""
    script_path = Path(__file__).parent / "test_e2e_prediction_to_profit.py"
    
    if not script_path.exists():
        print(f"Error: Test script not found at {script_path}")
        return False
    
    print("═" * 80)
    print("E2E TEST RUNNER - Prediction to Profit Taking")
    print("═" * 80)
    print(f"Test Start: {datetime.now().isoformat()}")
    print(f"Script: {script_path}")
    print("")
    
    # Check environment
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    ai_engine_url = os.getenv("AI_ENGINE_URL", "http://localhost:8001")
    
    print("Environment Configuration:")
    print(f"  Backend URL: {backend_url}")
    print(f"  AI Engine URL: {ai_engine_url}")
    print(f"  API Key configured: {'Yes' if api_key else 'No'}")
    print("")
    
    # Run the test
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent,
            capture_output=False
        )
        
        # Check report file
        report_file = Path(__file__).parent / "e2e_test_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            print("")
            print("═" * 80)
            print("FINAL RESULTS")
            print("═" * 80)
            print(json.dumps(report["summary"], indent=2))
            print("═" * 80)
            
            return result.returncode == 0
        else:
            print("Warning: Report file not generated")
            return result.returncode == 0
            
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
