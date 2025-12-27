"""
PostgreSQL implementation of StrategyRepository.

Stores strategies and statistics in the existing Quantum Trader database.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from .repositories import StrategyRepository
from .models import StrategyConfig, StrategyStats, StrategyStatus
from .schema import Strategy, StrategyStatistics

logger = logging.getLogger(__name__)


class PostgresStrategyRepository(StrategyRepository):
    """
    PostgreSQL-backed strategy repository.
    
    Uses SQLAlchemy with the existing Quantum Trader database connection.
    """
    
    def __init__(self, session_factory):
        """
        Initialize repository.
        
        Args:
            session_factory: SQLAlchemy session factory (e.g., sessionmaker)
        """
        self.session_factory = session_factory
    
    def save_strategy(self, config: StrategyConfig) -> None:
        """Save or update strategy configuration"""
        with self.session_factory() as session:
            try:
                # Check if exists
                existing = session.query(Strategy).filter_by(
                    strategy_id=config.strategy_id
                ).first()
                
                if existing:
                    # Update existing
                    existing.name = config.name
                    existing.status = config.status.value
                    existing.updated_at = datetime.utcnow()
                    existing.regime_filter = config.regime_filter.value
                    existing.entry_type = config.entry_type
                    existing.min_confidence = config.min_confidence
                    existing.tp_percent = config.tp_percent
                    existing.sl_percent = config.sl_percent
                    existing.use_trailing = config.use_trailing
                    existing.trailing_callback = config.trailing_callback
                    existing.max_risk_per_trade = config.max_risk_per_trade
                    existing.max_leverage = config.max_leverage
                    existing.max_concurrent_positions = config.max_concurrent_positions
                    existing.symbols = config.symbols
                    existing.timeframes = config.timeframes
                    existing.entry_params = config.entry_params
                    existing.generation = config.generation
                    existing.parent_ids = config.parent_ids
                    
                    logger.info(f"Updated strategy: {config.name} ({config.strategy_id})")
                else:
                    # Create new
                    strategy = Strategy(
                        strategy_id=config.strategy_id,
                        name=config.name,
                        status=config.status.value,
                        created_at=config.created_at,
                        updated_at=datetime.utcnow(),
                        regime_filter=config.regime_filter.value,
                        entry_type=config.entry_type,
                        min_confidence=config.min_confidence,
                        tp_percent=config.tp_percent,
                        sl_percent=config.sl_percent,
                        use_trailing=config.use_trailing,
                        trailing_callback=config.trailing_callback,
                        max_risk_per_trade=config.max_risk_per_trade,
                        max_leverage=config.max_leverage,
                        max_concurrent_positions=config.max_concurrent_positions,
                        symbols=config.symbols,
                        timeframes=config.timeframes,
                        entry_params=config.entry_params,
                        generation=config.generation,
                        parent_ids=config.parent_ids
                    )
                    session.add(strategy)
                    logger.info(f"Created strategy: {config.name} ({config.strategy_id})")
                
                session.commit()
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save strategy {config.strategy_id}: {e}")
                raise
    
    def get_strategies_by_status(self, status: StrategyStatus) -> list[StrategyConfig]:
        """Get all strategies with given status"""
        with self.session_factory() as session:
            strategies = session.query(Strategy).filter_by(status=status.value).all()
            
            results = []
            for s in strategies:
                from .models import RegimeFilter
                
                config = StrategyConfig(
                    strategy_id=s.strategy_id,
                    name=s.name,
                    regime_filter=RegimeFilter(s.regime_filter),
                    symbols=s.symbols,
                    timeframes=s.timeframes,
                    min_confidence=s.min_confidence,
                    entry_type=s.entry_type,
                    entry_params=s.entry_params or {},
                    tp_percent=s.tp_percent,
                    sl_percent=s.sl_percent,
                    use_trailing=s.use_trailing,
                    trailing_callback=s.trailing_callback,
                    max_risk_per_trade=s.max_risk_per_trade,
                    max_leverage=s.max_leverage,
                    max_concurrent_positions=s.max_concurrent_positions,
                    created_at=s.created_at,
                    status=StrategyStatus(s.status),
                    generation=s.generation,
                    parent_ids=s.parent_ids or []
                )
                results.append(config)
            
            logger.info(f"Retrieved {len(results)} strategies with status {status.value}")
            return results
    
    def update_status(self, strategy_id: str, status: StrategyStatus) -> None:
        """Update strategy status"""
        with self.session_factory() as session:
            try:
                strategy = session.query(Strategy).filter_by(
                    strategy_id=strategy_id
                ).first()
                
                if strategy:
                    old_status = strategy.status
                    strategy.status = status.value
                    strategy.updated_at = datetime.utcnow()
                    session.commit()
                    
                    logger.info(
                        f"Updated strategy {strategy_id} status: "
                        f"{old_status} → {status.value}"
                    )
                else:
                    logger.warning(f"Strategy not found: {strategy_id}")
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update strategy status: {e}")
                raise
    
    def save_stats(self, stats: StrategyStats) -> None:
        """Save performance statistics"""
        with self.session_factory() as session:
            try:
                stat_record = StrategyStatistics(
                    strategy_id=stats.strategy_id,
                    source=stats.source,
                    timestamp=stats.timestamp,
                    start_date=stats.start_date,
                    end_date=stats.end_date,
                    total_trades=stats.total_trades,
                    winning_trades=stats.winning_trades,
                    losing_trades=stats.losing_trades,
                    total_pnl=stats.total_pnl,
                    gross_profit=stats.gross_profit,
                    gross_loss=stats.gross_loss,
                    profit_factor=stats.profit_factor,
                    win_rate=stats.win_rate,
                    max_drawdown=stats.max_drawdown,
                    max_drawdown_pct=stats.max_drawdown_pct,
                    sharpe_ratio=stats.sharpe_ratio,
                    avg_win=stats.avg_win,
                    avg_loss=stats.avg_loss,
                    avg_rr_ratio=stats.avg_rr_ratio,
                    avg_bars_in_trade=stats.avg_bars_in_trade,
                    fitness_score=stats.fitness_score
                )
                
                session.add(stat_record)
                session.commit()
                
                logger.debug(
                    f"Saved stats for {stats.strategy_id} ({stats.source}): "
                    f"PF={stats.profit_factor:.2f}, Fitness={stats.fitness_score:.1f}"
                )
                
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save stats: {e}")
                raise
    
    def get_stats(
        self,
        strategy_id: str,
        source: Optional[str] = None,
        days: Optional[int] = None
    ) -> list[StrategyStats]:
        """Get performance statistics"""
        with self.session_factory() as session:
            query = session.query(StrategyStatistics).filter_by(
                strategy_id=strategy_id
            )
            
            if source:
                query = query.filter_by(source=source)
            
            if days:
                cutoff = datetime.utcnow() - timedelta(days=days)
                query = query.filter(StrategyStatistics.timestamp >= cutoff)
            
            query = query.order_by(StrategyStatistics.timestamp.desc())
            
            records = query.all()
            
            results = []
            for r in records:
                stat = StrategyStats(
                    strategy_id=r.strategy_id,
                    source=r.source,
                    start_date=r.start_date,
                    end_date=r.end_date,
                    timestamp=r.timestamp,
                    total_trades=r.total_trades,
                    winning_trades=r.winning_trades,
                    losing_trades=r.losing_trades,
                    total_pnl=r.total_pnl,
                    gross_profit=r.gross_profit,
                    gross_loss=r.gross_loss,
                    profit_factor=r.profit_factor,
                    win_rate=r.win_rate,
                    max_drawdown=r.max_drawdown,
                    max_drawdown_pct=r.max_drawdown_pct,
                    sharpe_ratio=r.sharpe_ratio,
                    avg_win=r.avg_win,
                    avg_loss=r.avg_loss,
                    avg_rr_ratio=r.avg_rr_ratio,
                    avg_bars_in_trade=r.avg_bars_in_trade,
                    fitness_score=r.fitness_score
                )
                results.append(stat)
            
            return results
