#!/usr/bin/env python3
"""
Main entry point for Robust Exit Engine

Runs continuous exit monitoring loop with reduceOnly plan emission
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from risk_guard.robust_exit_engine import RobustExitEngine, ExitEngineConfig

if __name__ == '__main__':
    config = ExitEngineConfig()
    engine = RobustExitEngine(config)
    engine.run_loop()
