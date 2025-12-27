"""
Legacy Policy Store wrapper.

This module re-exports PolicyDefaults from the governance submodule
to maintain backward compatibility with imports.
"""

from backend.services.governance.legacy_policy_store import PolicyDefaults

__all__ = ["PolicyDefaults"]
