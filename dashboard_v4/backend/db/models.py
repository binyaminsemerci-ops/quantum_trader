from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
from .connection import Base

class Example(Base):
    """Example model for demonstration"""
    __tablename__ = "examples"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    
    def __repr__(self):
        return f"<Example(id={self.id}, name={self.name})>"


class Trade(Base):
    """Trade model for storing trading history"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False)  # 'BUY' or 'SELL'
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String, default="PENDING")  # PENDING, FILLED, CANCELLED
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, side={self.side}, qty={self.quantity})>"


class Position(Base):
    """Position model for tracking active positions"""
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    unrealized_pnl = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True, index=True)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Position(symbol={self.symbol}, qty={self.quantity}, entry={self.entry_price})>"


class ControlLog(Base):
    """Control action audit log for admin/analyst actions"""
    __tablename__ = "control_log"
    
    id = Column(Integer, primary_key=True, index=True)
    action = Column(String, index=True, nullable=False)  # retrain, heal, mode_switch
    user = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # admin, analyst
    status = Column(String, default="success")  # success, failed
    details = Column(String, nullable=True)  # Additional context (e.g., mode value)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<ControlLog(id={self.id}, action={self.action}, user={self.user}, timestamp={self.timestamp})>"


class SystemEvent(Base):
    """System health events and metrics log"""
    __tablename__ = "system_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event = Column(String, nullable=False, index=True)  # health_check, container_restart, high_cpu, etc.
    cpu = Column(Float)
    ram = Column(Float)
    disk = Column(Float, nullable=True)
    details = Column(String, nullable=True)  # Additional context
    severity = Column(String, default="info")  # info, warning, critical
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<SystemEvent(id={self.id}, event={self.event}, severity={self.severity}, timestamp={self.timestamp})>"
