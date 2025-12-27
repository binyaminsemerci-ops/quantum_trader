from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from datetime import datetime
from ..connection import Base

class TradeJournal(Base):
    """
    Trade Journal model for storing executed trades with full context.
    Enables replay intelligence and explainability analysis.
    """
    __tablename__ = "trade_journal"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    direction = Column(String, nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)  # Filled on close
    pnl = Column(Float, nullable=True)  # Profit/Loss
    tp = Column(Float, nullable=False)  # Take Profit target
    sl = Column(Float, nullable=False)  # Stop Loss threshold
    trailing_stop = Column(Float, nullable=False)  # Trailing stop distance
    confidence = Column(Float, nullable=False)  # AI confidence 0-1
    model = Column(String, nullable=False)  # AI model name
    features = Column(JSON, nullable=True)  # AI features snapshot
    policy_state = Column(JSON, nullable=True)  # Risk policy state
    exit_reason = Column(String, default="open")  # open, tp_hit, sl_hit, manual, timeout
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<TradeJournal(id={self.id}, symbol={self.symbol}, direction={self.direction}, pnl={self.pnl})>"
