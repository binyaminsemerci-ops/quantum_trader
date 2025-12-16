"""
Exit Brain Status CLI Tool
===========================

Command-line tool to check Exit Brain v3 activation status.

Usage:
    python backend/tools/print_exit_status.py
    
Or from workspace root:
    python -m backend.tools.print_exit_status
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from backend.diagnostics.exit_brain_status import print_exit_brain_status


if __name__ == "__main__":
    print_exit_brain_status()
