"""Small helper to ensure repository root is on sys.path and import modules
that require the project root to be present. Use this from scripts that are
executed directly (e.g., from the scripts/ folder) to avoid repeated
sys.path manipulation and to keep imports at the top of the module without
ruff E402 noqa comments.

Usage:
    from scripts.import_helper import ensure_repo_root, import_module
    ensure_repo_root()
    MyClass = import_module('backend.database', 'MyClass')
    # or import the module object:
    mod = import_module('backend.database')
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Optional


def ensure_repo_root() -> None:
    """Ensure the repository root (one level up from scripts/) is on sys.path."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def import_module(module_name: str, attr: Optional[str] = None) -> Any:
    """Import a module (or attribute) after ensuring the repo root is on sys.path.

    If attr is provided, return getattr(module, attr). Otherwise return the
    module object.
    """
    ensure_repo_root()
    module = importlib.import_module(module_name)
    if attr:
        return getattr(module, attr)
    return module


# Ensure repo root is present when this helper is imported so other modules can
# safely `import import_helper` at the top of their file and rely on the repo
# root being on sys.path.
ensure_repo_root()
