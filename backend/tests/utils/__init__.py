"""
Package marker for test utilities.

This file makes `backend.tests.utils` a proper package so tests can import
`backend.tests.utils.mock_signals` via importlib in `backend.routes.signals`.
"""

__all__ = ["mock_signals"]
