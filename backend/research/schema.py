"""
Strategy Generator AI - Database Schema

SQLAlchemy models for storing strategies and performance statistics.
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON, Index
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Strategy(Base):
    """Strategy configuration storage"""
    __tablename__ = "sg_strategies"
    
    # Primary key
    strategy_id = Column(String(50), primary_key=True)
    
    # Metadata
    name = Column(String(200), nullable=False)
    status = Column(String(20), nullable=False, index=True)  # CANDIDATE/SHADOW/LIVE/DISABLED
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Strategy configuration
    regime_filter = Column(String(20), nullable=False)  # TRENDING/RANGING/CHOPPY/ANY
    entry_type = Column(String(30), nullable=False)  # ENSEMBLE_CONSENSUS/MOMENTUM/MEAN_REVERSION
    min_confidence = Column(Float, nullable=False)
    
    # Risk parameters
    tp_percent = Column(Float, nullable=False)
    sl_percent = Column(Float, nullable=False)
    use_trailing = Column(Boolean, nullable=False, default=False)
    trailing_callback = Column(Float, nullable=True)
    max_risk_per_trade = Column(Float, nullable=False)
    max_leverage = Column(Float, nullable=False)
    max_concurrent_positions = Column(Integer, nullable=False, default=1)
    
    # Trading parameters (stored as JSON)
    symbols = Column(JSON, nullable=False)  # List of symbols
    timeframes = Column(JSON, nullable=False)  # List of timeframes
    entry_params = Column(JSON, nullable=True)  # Additional entry parameters
    
    # Evolution tracking
    generation = Column(Integer, nullable=False, default=0)
    parent_ids = Column(JSON, nullable=True)  # List of parent strategy IDs
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_status_generation', 'status', 'generation'),
        Index('idx_created_at', 'created_at'),
    )


class StrategyStatistics(Base):
    """Performance statistics for strategies"""
    __tablename__ = "sg_strategy_stats"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key
    strategy_id = Column(String(50), nullable=False, index=True)
    
    # Source of statistics
    source = Column(String(20), nullable=False, index=True)  # BACKTEST/SHADOW/LIVE
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Period
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Trade statistics
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    
    # Performance metrics
    total_pnl = Column(Float, nullable=False, default=0.0)
    gross_profit = Column(Float, nullable=False, default=0.0)
    gross_loss = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=False, default=0.0)
    win_rate = Column(Float, nullable=False, default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Float, nullable=False, default=0.0)
    max_drawdown_pct = Column(Float, nullable=False, default=0.0)
    sharpe_ratio = Column(Float, nullable=False, default=0.0)
    
    # Average metrics
    avg_win = Column(Float, nullable=False, default=0.0)
    avg_loss = Column(Float, nullable=False, default=0.0)
    avg_rr_ratio = Column(Float, nullable=False, default=0.0)
    avg_bars_in_trade = Column(Float, nullable=False, default=0.0)
    
    # Fitness score
    fitness_score = Column(Float, nullable=False, default=0.0, index=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_strategy_source', 'strategy_id', 'source'),
        Index('idx_strategy_timestamp', 'strategy_id', 'timestamp'),
        Index('idx_source_fitness', 'source', 'fitness_score'),
    )
