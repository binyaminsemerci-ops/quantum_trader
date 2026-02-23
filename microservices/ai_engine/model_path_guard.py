"""
MODEL PATH GUARD
================
Controlled Refactor — Phase: Freeze Live Model Mutation
Deployed: 2026-02-21

Enforces immutable load/write boundaries for the live AI engine:

  LOAD  → /opt/quantum/model_registry/approved/   (read-only for live services)
  WRITE → /opt/quantum/model_registry/staging/    (write-only for retrain workers)

Any attempt to load a model from outside APPROVED_DIR raises a RuntimeError,
intentionally crashing the service startup so the violation is immediately visible.

Usage
-----
from model_path_guard import assert_approved_load_path, assert_staging_write_path

# In any loader:
path = assert_approved_load_path(settings.XGB_MODEL_PATH, label="xgb_model")
model = pickle.load(open(path, "rb"))

# In any trainer/saver:
path = assert_staging_write_path(self.save_path, label="rl_policy")
torch.save(checkpoint, path)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical boundary definitions
# ---------------------------------------------------------------------------

APPROVED_DIR: Path = Path(
    os.environ.get("QT_APPROVED_MODEL_DIR", "/opt/quantum/model_registry/approved")
)

STAGING_DIR: Path = Path(
    os.environ.get("QT_STAGING_MODEL_DIR", "/opt/quantum/model_registry/staging")
)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def assert_approved_load_path(path: "str | Path", label: str = "") -> Path:
    """
    Verify *path* is under APPROVED_DIR before loading a model.

    Raises RuntimeError (fatal) if the path resolves outside the approved
    registry directory.  This is intentional — a mis-pathed load is a
    production safety violation that must be caught at startup, not silently
    swallowed.

    Args:
        path:  Model file path to validate.
        label: Human-readable model name used in the error message.

    Returns:
        The resolved absolute Path if valid.

    Raises:
        RuntimeError: Path is not under APPROVED_DIR.
    """
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(APPROVED_DIR.resolve())
    except ValueError:
        raise RuntimeError(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════╗\n"
            f"║  [MODEL-GUARD] FATAL LOAD VIOLATION                         ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Model  : {label:<52} ║\n"
            f"║  Path   : {str(resolved):<52} ║\n"
            f"║  Allowed: {str(APPROVED_DIR):<52} ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  The live AI engine may ONLY load models from the approved   ║\n"
            f"║  registry.  To deploy a new model:                           ║\n"
            f"║    1. Retrain  → writes to staging/                          ║\n"
            f"║    2. Evaluate → manual or CLM evaluation pass               ║\n"
            f"║    3. Promote  → cp staging/<file> approved/<file>           ║\n"
            f"╚══════════════════════════════════════════════════════════════╝"
        )
    return resolved


def assert_staging_write_path(path: "str | Path", label: str = "") -> Path:
    """
    Verify *path* is under STAGING_DIR before writing a model.

    Raises RuntimeError (fatal) if a retrain worker attempts to write
    directly into the approved directory, bypassing the promotion gate.

    Args:
        path:  Destination file path for the serialized model.
        label: Human-readable model name used in the error message.

    Returns:
        The resolved absolute Path if valid.

    Raises:
        RuntimeError: Path is not under STAGING_DIR.
    """
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(STAGING_DIR.resolve())
    except ValueError:
        raise RuntimeError(
            f"\n"
            f"╔══════════════════════════════════════════════════════════════╗\n"
            f"║  [MODEL-GUARD] FATAL WRITE VIOLATION                        ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Model  : {label:<52} ║\n"
            f"║  Path   : {str(resolved):<52} ║\n"
            f"║  Allowed: {str(STAGING_DIR):<52} ║\n"
            f"╠══════════════════════════════════════════════════════════════╣\n"
            f"║  Retrain workers may ONLY write to the staging directory.    ║\n"
            f"║  Writing directly to approved/ bypasses the promotion gate.  ║\n"
            f"╚══════════════════════════════════════════════════════════════╝"
        )
    return resolved


def approved_path(filename: str) -> str:
    """Convenience: return absolute path string under APPROVED_DIR."""
    return str(APPROVED_DIR / filename)


def staging_path(filename: str) -> str:
    """Convenience: return absolute path string under STAGING_DIR."""
    return str(STAGING_DIR / filename)
