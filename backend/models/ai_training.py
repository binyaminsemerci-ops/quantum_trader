"""
Database models for AI training and continuous learning.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base


class AITrainingSample(Base):
    """
    Stores features, predictions, and actual outcomes for continuous learning.
    
    Each sample represents one trading decision with:
    - Input features (market indicators at decision time)
    - AI prediction (BUY/SELL/HOLD + confidence)
    - Actual outcome (what happened: P&L, duration, etc.)
    """
    __tablename__ = "ai_training_samples"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Trading context
    symbol = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    run_id = Column(Integer, ForeignKey("liquidity_runs.id"), nullable=True)
    
    # AI prediction at decision time
    predicted_action = Column(String, nullable=False)  # BUY/SELL/HOLD
    prediction_score = Column(Float, nullable=False)
    prediction_confidence = Column(Float, nullable=False)
    model_version = Column(String, nullable=True)
    
    # Market features (JSON serialized)
    features = Column(Text, nullable=False)  # JSON array of feature values
    feature_names = Column(Text, nullable=True)  # JSON array of feature names
    
    # Execution details
    executed = Column(Boolean, default=False)
    execution_side = Column(String, nullable=True)  # BUY/SELL if executed
    entry_price = Column(Float, nullable=True)
    entry_quantity = Column(Float, nullable=True)
    entry_time = Column(DateTime(timezone=True), nullable=True)
    
    # Outcome (filled when position closes)
    outcome_known = Column(Boolean, default=False)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    realized_pnl = Column(Float, nullable=True)
    hold_duration_seconds = Column(Integer, nullable=True)
    
    # Label for training (derived from outcome)
    target_label = Column(Float, nullable=True)  # Regression: % return or 0
    target_class = Column(String, nullable=True)  # Classification: WIN/LOSS/NEUTRAL
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return (
            f"<AITrainingSample(id={self.id}, symbol={self.symbol}, "
            f"predicted={self.predicted_action}, executed={self.executed}, "
            f"pnl={self.realized_pnl})>"
        )


class AIModelVersion(Base):
    """
    Tracks different versions of trained models with their performance metrics.
    """
    __tablename__ = "ai_model_versions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Model identification
    version_id = Column(String, nullable=False, unique=True, index=True)
    model_type = Column(String, nullable=False)  # xgboost, ensemble, etc.
    file_path = Column(String, nullable=False)
    
    # Training metadata
    trained_at = Column(DateTime(timezone=True), nullable=False)
    training_samples = Column(Integer, nullable=False)
    training_duration_seconds = Column(Float, nullable=True)
    
    # Performance metrics (on training/validation set)
    train_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    train_mae = Column(Float, nullable=True)  # Mean Absolute Error
    validation_mae = Column(Float, nullable=True)
    
    # Live performance (updated as model is used)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    live_accuracy = Column(Float, nullable=True)
    total_pnl = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=False)
    replaced_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return (
            f"<AIModelVersion(id={self.id}, version={self.version_id}, "
            f"type={self.model_type}, active={self.is_active})>"
        )
