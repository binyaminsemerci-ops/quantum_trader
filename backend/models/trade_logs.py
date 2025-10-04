from sqlalchemy import Column, Integer, String, Float, DateTime
from backend.database import Base


class TradeLog(Base):
    __tablename__ = "trade_logs"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False)
