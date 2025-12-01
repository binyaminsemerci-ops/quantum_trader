# models/trade.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from database import Base

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    pnl = Column(Float, default=0.0)
    fee = Column(Float, default=0.0)
    status = Column(String(20), default="pending")
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side})>"
