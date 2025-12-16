"""
Signals Domain - AI signal tracking
EPIC: DASHBOARD-V3-TRADING-PANELS

Provides signal storage and retrieval for dashboard.
"""

from backend.domains.signals.models import SignalRecord
from backend.domains.signals.service import SignalService

__all__ = ["SignalRecord", "SignalService"]
