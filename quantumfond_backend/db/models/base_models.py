"""
Base Database Models
Core models for system operations and audit logging
"""
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Boolean, Text
from datetime import datetime
from ..connection import Base

class ControlLog(Base):
    """Audit log for all control actions"""
    __tablename__ = "control_log"
    
    id = Column(Integer, primary_key=True, index=True)
    user = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False)
    action = Column(String(200), nullable=False)
    status = Column(String(50), nullable=False)
    metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class SystemEvent(Base):
    """System health and performance events"""
    __tablename__ = "system_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event = Column(String(200), nullable=False)
    cpu = Column(Float, nullable=True)
    ram = Column(Float, nullable=True)
    disk = Column(Float, nullable=True)
    severity = Column(String(20), default="info")
    metadata = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class Trade(Base):
    """Trade records"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    strategy = Column(String(100), nullable=True)
    status = Column(String(20), default="open")
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, nullable=True)

class RiskMetric(Base):
    """Real-time risk metrics"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String(50), nullable=False)
    value = Column(Float, nullable=False)
    threshold = Column(Float, nullable=True)
    status = Column(String(20), default="normal")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class AIModel(Base):
    """AI model tracking"""
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    accuracy = Column(Float, nullable=True)
    status = Column(String(20), default="active")
    deployed_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)

class Incident(Base):
    """Incident tracking"""
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(String(20), nullable=False)
    status = Column(String(20), default="open")
    assigned_to = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, nullable=True)

class User(Base):
    """User accounts"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
