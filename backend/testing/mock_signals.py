"""Compatibility shim for moved mock_signals.

The real test/demo generator was moved to `backend/tests/utils/mock_signals.py`.
This shim raises ImportError to prevent accidental runtime imports from
production code and to make the move explicit.
"""

raise ImportError(
    "backend.testing.mock_signals has been moved to backend.tests.utils.mock_signals; import from there in tests."
)
