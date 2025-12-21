from sqlalchemy import Column, Integer, String, Float, DateTime
from backend.database import Base


class TradeLog(Base):
    __tablename__ = "trade_logs"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    timestamp = Column(DateTime, nullable=False)
    
    # Analytics fields
    realized_pnl = Column(Float, nullable=True)
    realized_pnl_pct = Column(Float, nullable=True)
    equity_after = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    strategy_id = Column(String, nullable=True)
