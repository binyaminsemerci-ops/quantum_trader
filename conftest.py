"""Pytest configuration and shared fixtures.

Centralizes path adjustments so individual tests don't need to mutate sys.path
at the top (reduces E402 lint warnings) and provides a lightweight DB fixture
if needed later.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Example placeholder fixture (can be expanded)
# import pytest
# @pytest.fixture(scope="session")
# def db_url():
#     return os.environ.get("QUANTUM_TRADER_DATABASE_URL", "sqlite:///:memory:")
