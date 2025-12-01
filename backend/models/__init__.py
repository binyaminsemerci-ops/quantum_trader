# models/__init__.py
# Clear SQLAlchemy metadata to prevent caching issues
import sys
if 'database' in sys.modules:
    from database import Base
    Base.metadata.clear()

from .trade import Trade
from .trade_log import TradeLog

__all__ = ["Trade", "TradeLog"]
