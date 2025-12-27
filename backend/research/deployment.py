"""
Strategy deployment manager.

Promotes/demotes strategies based on performance thresholds.
"""

import logging
from datetime import datetime, timedelta

from .models import StrategyConfig, StrategyStats, StrategyStatus
from .repositories import StrategyRepository

logger = logging.getLogger(__name__)


class StrategyDeploymentManager:
    """
    Promotes/demotes strategies based on performance.
    
    Workflow:
    - CANDIDATE → SHADOW: Strong backtest stats
    - SHADOW → LIVE: Proven forward performance
    - LIVE → DISABLED: Underperformance detected
    """
    
    def __init__(
        self,
        repository: StrategyRepository,
        # Promotion thresholds
        candidate_min_pf: float = 1.5,
        candidate_min_trades: int = 50,
        candidate_max_dd: float = 0.20,
        shadow_min_pf: float = 1.3,
        shadow_min_trades: int = 20,
        shadow_min_days: int = 14,
        # Demotion thresholds
        live_min_pf: float = 1.1,
        live_max_dd: float = 0.25,
        live_check_days: int = 30
    ):
        """
        Initialize deployment manager.
        
        Args:
            repository: Strategy storage
            candidate_min_pf: Min profit factor for CANDIDATE → SHADOW
            candidate_min_trades: Min trades for CANDIDATE → SHADOW
            candidate_max_dd: Max drawdown for CANDIDATE → SHADOW
            shadow_min_pf: Min profit factor for SHADOW → LIVE
            shadow_min_trades: Min trades for SHADOW → LIVE
            shadow_min_days: Min forward test period for SHADOW → LIVE
            live_min_pf: Min profit factor to stay LIVE
            live_max_dd: Max drawdown to stay LIVE
            live_check_days: Period to check LIVE performance
        """
        self.repository = repository
        
        # Promotion thresholds
        self.candidate_min_pf = candidate_min_pf
        self.candidate_min_trades = candidate_min_trades
        self.candidate_max_dd = candidate_max_dd
        
        self.shadow_min_pf = shadow_min_pf
        self.shadow_min_trades = shadow_min_trades
        self.shadow_min_days = shadow_min_days
        
        # Demotion thresholds
        self.live_min_pf = live_min_pf
        self.live_max_dd = live_max_dd
        self.live_check_days = live_check_days
    
    def review_and_promote(self) -> list[str]:
        """
        Promote qualifying strategies.
        
        Returns:
            List of promoted strategy IDs
        """
        logger.info("📈 Reviewing strategies for promotion")
        
        promoted: list[str] = []
        
        # Promote CANDIDATE → SHADOW
        candidates = self.repository.get_strategies_by_status(StrategyStatus.CANDIDATE)
        
        for config in candidates:
            if self._should_promote_to_shadow(config):
                logger.info(f"  ✅ Promoting to SHADOW: {config.name}")
                self.repository.update_status(config.strategy_id, StrategyStatus.SHADOW)
                promoted.append(config.strategy_id)
        
        # Promote SHADOW → LIVE
        shadow_strategies = self.repository.get_strategies_by_status(StrategyStatus.SHADOW)
        
        for config in shadow_strategies:
            if self._should_promote_to_live(config):
                logger.info(f"  ✅ Promoting to LIVE: {config.name}")
                self.repository.update_status(config.strategy_id, StrategyStatus.LIVE)
                promoted.append(config.strategy_id)
        
        logger.info(f"Promoted {len(promoted)} strategies")
        return promoted
    
    def review_and_disable(self) -> list[str]:
        """
        Disable underperforming strategies.
        
        Returns:
            List of disabled strategy IDs
        """
        logger.info("📉 Reviewing strategies for demotion")
        
        disabled: list[str] = []
        
        # Check LIVE strategies
        live_strategies = self.repository.get_strategies_by_status(StrategyStatus.LIVE)
        
        for config in live_strategies:
            if self._should_disable(config):
                logger.warning(f"  ⚠️  Disabling underperformer: {config.name}")
                self.repository.update_status(config.strategy_id, StrategyStatus.DISABLED)
                disabled.append(config.strategy_id)
        
        logger.info(f"Disabled {len(disabled)} strategies")
        return disabled
    
    def _should_promote_to_shadow(self, config: StrategyConfig) -> bool:
        """Check if CANDIDATE should be promoted to SHADOW"""
        
        # Get backtest stats
        stats_list = self.repository.get_stats(
            config.strategy_id,
            source="BACKTEST",
            days=None
        )
        
        if not stats_list:
            return False
        
        # Use most recent backtest
        stats = stats_list[0]
        
        # Check thresholds
        if stats.total_trades < self.candidate_min_trades:
            logger.debug(f"    {config.name}: Insufficient trades ({stats.total_trades})")
            return False
        
        if stats.profit_factor < self.candidate_min_pf:
            logger.debug(f"    {config.name}: Low PF ({stats.profit_factor:.2f})")
            return False
        
        if stats.max_drawdown_pct > self.candidate_max_dd:
            logger.debug(f"    {config.name}: High DD ({stats.max_drawdown_pct:.1%})")
            return False
        
        logger.info(
            f"    {config.name} qualifies for SHADOW: "
            f"PF={stats.profit_factor:.2f}, Trades={stats.total_trades}, "
            f"DD={stats.max_drawdown_pct:.1%}"
        )
        return True
    
    def _should_promote_to_live(self, config: StrategyConfig) -> bool:
        """Check if SHADOW should be promoted to LIVE"""
        
        # Get shadow stats (last N days)
        stats_list = self.repository.get_stats(
            config.strategy_id,
            source="SHADOW",
            days=self.shadow_min_days
        )
        
        if not stats_list:
            logger.debug(f"    {config.name}: No shadow stats yet")
            return False
        
        # Aggregate shadow stats
        total_trades = sum(s.total_trades for s in stats_list)
        total_pnl = sum(s.total_pnl for s in stats_list)
        gross_profit = sum(s.gross_profit for s in stats_list)
        gross_loss = sum(s.gross_loss for s in stats_list)
        
        if total_trades < self.shadow_min_trades:
            logger.debug(f"    {config.name}: Insufficient shadow trades ({total_trades})")
            return False
        
        if gross_loss == 0:
            pf = float('inf') if gross_profit > 0 else 0.0
        else:
            pf = gross_profit / abs(gross_loss)
        
        if pf < self.shadow_min_pf:
            logger.debug(f"    {config.name}: Low shadow PF ({pf:.2f})")
            return False
        
        # Check age (must be in shadow for minimum period)
        age_days = (datetime.utcnow() - config.created_at).days
        if age_days < self.shadow_min_days:
            logger.debug(
                f"    {config.name}: Too young ({age_days} days < {self.shadow_min_days})"
            )
            return False
        
        logger.info(
            f"    {config.name} qualifies for LIVE: "
            f"PF={pf:.2f}, Trades={total_trades}, Age={age_days}d"
        )
        return True
    
    def _should_disable(self, config: StrategyConfig) -> bool:
        """Check if LIVE strategy should be disabled"""
        
        # Get recent live stats
        stats_list = self.repository.get_stats(
            config.strategy_id,
            source="LIVE",
            days=self.live_check_days
        )
        
        if not stats_list:
            # No live stats yet, give it time
            return False
        
        # Aggregate recent performance
        total_trades = sum(s.total_trades for s in stats_list)
        
        if total_trades < 10:
            # Not enough data yet
            return False
        
        gross_profit = sum(s.gross_profit for s in stats_list)
        gross_loss = sum(s.gross_loss for s in stats_list)
        max_dd = max(s.max_drawdown_pct for s in stats_list)
        
        if gross_loss == 0:
            pf = float('inf') if gross_profit > 0 else 0.0
        else:
            pf = gross_profit / abs(gross_loss)
        
        # Check demotion conditions
        if pf < self.live_min_pf:
            logger.warning(
                f"    {config.name} underperforming: PF={pf:.2f} < {self.live_min_pf}"
            )
            return True
        
        if max_dd > self.live_max_dd:
            logger.warning(
                f"    {config.name} excessive DD: {max_dd:.1%} > {self.live_max_dd:.1%}"
            )
            return True
        
        return False
