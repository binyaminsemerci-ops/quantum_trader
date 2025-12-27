"""
Meta Strategy Controller (MSC AI)

The supreme decision-making brain of Quantum Trader that evaluates system-wide
performance and sets global trading policies.

MSC AI determines:
- Risk mode (AGGRESSIVE / NORMAL / DEFENSIVE)
- Which strategies are allowed to trade
- Global parameters (confidence, risk per trade, max positions)

All other components read these policies to constrain their behavior.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol, Optional
from statistics import mean, stdev

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class RiskMode(str, Enum):
    """System-wide risk posture"""
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"


class RegimeType(str, Enum):
    """Market regime types"""
    BULL_TRENDING = "BULL_TRENDING"
    BEAR_TRENDING = "BEAR_TRENDING"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    VOLATILE = "VOLATILE"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class StrategyStats:
    """Performance statistics for a strategy"""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    max_drawdown_pct: float
    profit_factor: float
    winrate: float
    avg_win: float
    avg_loss: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    @property
    def is_statistically_significant(self) -> bool:
        """Check if strategy has enough trades for reliable statistics"""
        return self.total_trades >= 30


@dataclass
class StrategyConfig:
    """Strategy configuration from SG AI"""
    strategy_id: str
    name: str
    status: str
    regime_tags: list[str]  # Regimes this strategy is designed for
    min_confidence: float
    max_risk_per_trade: float
    created_at: datetime
    updated_at: datetime


@dataclass
class GlobalPolicy:
    """Global trading policy set by MSC AI"""
    risk_mode: RiskMode
    allowed_strategies: list[str]
    max_risk_per_trade: float
    global_min_confidence: float
    max_positions: int
    max_daily_trades: Optional[int] = None
    max_slippage_pct: Optional[float] = None
    allowed_symbols: Optional[list[str]] = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['risk_mode'] = self.risk_mode.value
        data['updated_at'] = self.updated_at.isoformat()
        return data


@dataclass
class StrategyScore:
    """Scored strategy with selection metadata"""
    strategy_id: str
    score: float
    profit_factor: float
    winrate: float
    drawdown_penalty: float
    trade_count_factor: float
    regime_bonus: float
    total_trades: int
    reasons: list[str]


@dataclass
class SystemHealth:
    """Aggregated system health metrics"""
    current_drawdown_pct: float
    global_winrate: float
    equity_curve_slope: float  # Daily return % trend
    global_regime: str
    volatility_level: str
    recent_trade_count: int
    consecutive_losses: int
    days_since_profit: int


# ============================================================================
# Repository Interfaces
# ============================================================================

class MetricsRepository(Protocol):
    """Interface for accessing system-wide metrics"""
    
    def get_current_drawdown_pct(self) -> float:
        """Get current drawdown as percentage"""
        ...
    
    def get_global_winrate(self, last_trades: int = 200) -> float:
        """Get global winrate from recent trades"""
        ...
    
    def get_equity_curve(self, days: int = 30) -> list[tuple[datetime, float]]:
        """Get equity curve as (timestamp, balance) tuples"""
        ...
    
    def get_global_regime(self) -> str:
        """Get current market regime"""
        ...
    
    def get_volatility_level(self) -> str:
        """Get current volatility regime (LOW/NORMAL/HIGH/EXTREME)"""
        ...
    
    def get_consecutive_losses(self) -> int:
        """Get number of consecutive losing trades"""
        ...
    
    def get_days_since_last_profit(self) -> int:
        """Get days since last profitable trade"""
        ...


class StrategyRepository(Protocol):
    """Interface for accessing strategy data"""
    
    def get_strategies_by_status(self, status: str) -> list[StrategyConfig]:
        """Get all strategies with given status"""
        ...
    
    def get_recent_stats(
        self, 
        strategy_id: str, 
        source: str, 
        days: int
    ) -> StrategyStats:
        """Get performance stats for strategy from specific source"""
        ...
    
    def get_strategy_config(self, strategy_id: str) -> StrategyConfig:
        """Get strategy configuration"""
        ...


class PolicyStore(Protocol):
    """Interface for storing/retrieving global policy"""
    
    def get(self) -> dict:
        """Get current policy"""
        ...
    
    def update(self, policy: dict) -> None:
        """Update policy"""
        ...


# ============================================================================
# Strategy Scoring Logic
# ============================================================================

class StrategyScorer:
    """
    Scores strategies based on recent performance and context.
    
    Scoring factors:
    - Profit factor (40%)
    - Winrate (30%)
    - Drawdown control (20%)
    - Trade volume (10%)
    - Regime compatibility (bonus)
    """
    
    def __init__(
        self,
        *,
        min_trades_for_full_score: int = 50,
        regime_bonus: float = 0.15,
        drawdown_penalty_factor: float = 2.0
    ):
        self.min_trades_for_full_score = min_trades_for_full_score
        self.regime_bonus = regime_bonus
        self.drawdown_penalty_factor = drawdown_penalty_factor
    
    def score_strategy(
        self,
        stats: StrategyStats,
        config: StrategyConfig,
        current_regime: str
    ) -> StrategyScore:
        """
        Score a single strategy.
        
        Returns StrategyScore with breakdown of scoring factors.
        """
        reasons = []
        
        # 1. Profit Factor Component (40%)
        # Normalize profit factor: 2.0+ = excellent, 1.0 = break-even, <1.0 = losing
        pf_score = min(stats.profit_factor / 2.0, 1.0) * 0.40
        reasons.append(f"PF={stats.profit_factor:.2f} → {pf_score:.3f}")
        
        # 2. Winrate Component (30%)
        # 60%+ winrate = excellent, 50% = neutral, <40% = poor
        wr_normalized = (stats.winrate - 0.40) / 0.30  # Scale 40-70% to 0-1
        wr_score = max(0, min(wr_normalized, 1.0)) * 0.30
        reasons.append(f"WR={stats.winrate:.1%} → {wr_score:.3f}")
        
        # 3. Drawdown Control Component (20%)
        # Lower drawdown = better
        # 2% DD = 1.0, 5% DD = 0.5, 10%+ DD = 0.0
        dd_penalty = 1.0 - min(stats.max_drawdown_pct / 10.0, 1.0)
        dd_score = dd_penalty * 0.20
        reasons.append(f"DD={stats.max_drawdown_pct:.1f}% → {dd_score:.3f}")
        
        # 4. Trade Volume Component (10%)
        # More trades = more confidence in statistics
        trade_factor = min(stats.total_trades / self.min_trades_for_full_score, 1.0)
        tv_score = trade_factor * 0.10
        reasons.append(f"Trades={stats.total_trades} → {tv_score:.3f}")
        
        # 5. Regime Compatibility Bonus
        regime_match = current_regime in config.regime_tags if config.regime_tags else False
        regime_bonus_applied = self.regime_bonus if regime_match else 0.0
        if regime_match:
            reasons.append(f"Regime match +{regime_bonus_applied:.3f}")
        
        # Total score
        base_score = pf_score + wr_score + dd_score + tv_score
        total_score = base_score + regime_bonus_applied
        
        return StrategyScore(
            strategy_id=stats.strategy_id,
            score=total_score,
            profit_factor=pf_score,
            winrate=wr_score,
            drawdown_penalty=dd_score,
            trade_count_factor=tv_score,
            regime_bonus=regime_bonus_applied,
            total_trades=stats.total_trades,
            reasons=reasons
        )
    
    def score_all_strategies(
        self,
        strategies: list[tuple[StrategyConfig, StrategyStats]],
        current_regime: str
    ) -> list[StrategyScore]:
        """Score all strategies and return sorted by score (descending)"""
        scores = []
        
        for config, stats in strategies:
            score = self.score_strategy(stats, config, current_regime)
            scores.append(score)
        
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        
        return scores


# ============================================================================
# Risk Mode Selection Logic
# ============================================================================

class RiskModeSelector:
    """
    Determines system-wide risk mode based on health metrics.
    
    Logic:
    - DEFENSIVE: System under stress (DD, losses, volatility)
    - NORMAL: Healthy baseline operation
    - AGGRESSIVE: Strong performance, favorable conditions
    """
    
    def __init__(
        self,
        *,
        defensive_dd_threshold: float = 5.0,
        aggressive_dd_threshold: float = 2.0,
        defensive_winrate_threshold: float = 0.45,
        aggressive_winrate_threshold: float = 0.60,
        consecutive_loss_limit: int = 5,
        days_no_profit_limit: int = 3
    ):
        self.defensive_dd_threshold = defensive_dd_threshold
        self.aggressive_dd_threshold = aggressive_dd_threshold
        self.defensive_winrate_threshold = defensive_winrate_threshold
        self.aggressive_winrate_threshold = aggressive_winrate_threshold
        self.consecutive_loss_limit = consecutive_loss_limit
        self.days_no_profit_limit = days_no_profit_limit
    
    def select_risk_mode(self, health: SystemHealth) -> tuple[RiskMode, list[str]]:
        """
        Select risk mode based on system health.
        
        Returns:
            (RiskMode, reasons for decision)
        """
        reasons = []
        defensive_signals = 0
        aggressive_signals = 0
        
        # 1. Drawdown check
        if health.current_drawdown_pct >= self.defensive_dd_threshold:
            defensive_signals += 2
            reasons.append(f"High DD: {health.current_drawdown_pct:.1f}% (DEFENSIVE)")
        elif health.current_drawdown_pct <= self.aggressive_dd_threshold:
            aggressive_signals += 1
            reasons.append(f"Low DD: {health.current_drawdown_pct:.1f}% (AGGRESSIVE)")
        
        # 2. Winrate check
        if health.global_winrate < self.defensive_winrate_threshold:
            defensive_signals += 2
            reasons.append(f"Low WR: {health.global_winrate:.1%} (DEFENSIVE)")
        elif health.global_winrate >= self.aggressive_winrate_threshold:
            aggressive_signals += 1
            reasons.append(f"High WR: {health.global_winrate:.1%} (AGGRESSIVE)")
        
        # 3. Equity curve trend
        if health.equity_curve_slope < -0.5:  # Losing 0.5%+ per day
            defensive_signals += 2
            reasons.append(f"Negative trend: {health.equity_curve_slope:.2f}%/day (DEFENSIVE)")
        elif health.equity_curve_slope > 1.0:  # Gaining 1%+ per day
            aggressive_signals += 1
            reasons.append(f"Positive trend: {health.equity_curve_slope:.2f}%/day (AGGRESSIVE)")
        
        # 4. Consecutive losses
        if health.consecutive_losses >= self.consecutive_loss_limit:
            defensive_signals += 1
            reasons.append(f"Losing streak: {health.consecutive_losses} (DEFENSIVE)")
        
        # 5. Days without profit
        if health.days_since_profit >= self.days_no_profit_limit:
            defensive_signals += 1
            reasons.append(f"No profit for {health.days_since_profit} days (DEFENSIVE)")
        
        # 6. Volatility regime
        if health.volatility_level in ["HIGH", "EXTREME"]:
            defensive_signals += 1
            reasons.append(f"High volatility: {health.volatility_level} (DEFENSIVE)")
        elif health.volatility_level == "LOW":
            # Low vol can be good for trend strategies
            pass
        
        # 7. Market regime
        if health.global_regime in ["CHOPPY", "LOW_LIQUIDITY"]:
            defensive_signals += 1
            reasons.append(f"Unfavorable regime: {health.global_regime} (DEFENSIVE)")
        elif health.global_regime in ["BULL_TRENDING", "BEAR_TRENDING"]:
            aggressive_signals += 1
            reasons.append(f"Trending regime: {health.global_regime} (AGGRESSIVE)")
        
        # Decision logic
        if defensive_signals >= 3:
            return RiskMode.DEFENSIVE, reasons
        elif aggressive_signals >= 2 and defensive_signals == 0:
            return RiskMode.AGGRESSIVE, reasons
        else:
            return RiskMode.NORMAL, reasons


# ============================================================================
# Policy Builder
# ============================================================================

class PolicyBuilder:
    """
    Constructs global policy parameters based on risk mode and context.
    """
    
    # Parameter tables by risk mode
    RISK_PARAMS = {
        RiskMode.DEFENSIVE: {
            'max_risk_per_trade': 0.003,  # 0.3%
            'global_min_confidence': 0.70,
            'max_positions': 8,
            'max_daily_trades': 10,
            'max_slippage_pct': 0.001,  # 0.1%
        },
        RiskMode.NORMAL: {
            'max_risk_per_trade': 0.0075,  # 0.75%
            'global_min_confidence': 0.60,
            'max_positions': 10,
            'max_daily_trades': 30,
            'max_slippage_pct': 0.002,  # 0.2%
        },
        RiskMode.AGGRESSIVE: {
            'max_risk_per_trade': 0.015,  # 1.5%
            'global_min_confidence': 0.50,
            'max_positions': 20,
            'max_daily_trades': 50,
            'max_slippage_pct': 0.003,  # 0.3%
        }
    }
    
    def build_policy(
        self,
        risk_mode: RiskMode,
        allowed_strategies: list[str],
        *,
        allowed_symbols: Optional[list[str]] = None
    ) -> GlobalPolicy:
        """
        Build complete policy from risk mode and selected strategies.
        """
        params = self.RISK_PARAMS[risk_mode]
        
        return GlobalPolicy(
            risk_mode=risk_mode,
            allowed_strategies=allowed_strategies,
            max_risk_per_trade=params['max_risk_per_trade'],
            global_min_confidence=params['global_min_confidence'],
            max_positions=params['max_positions'],
            max_daily_trades=params['max_daily_trades'],
            max_slippage_pct=params['max_slippage_pct'],
            allowed_symbols=allowed_symbols
        )


# ============================================================================
# Meta Strategy Controller (Main Orchestrator)
# ============================================================================

class MetaStrategyController:
    """
    Supreme decision-making brain of Quantum Trader.
    
    Evaluates system-wide performance and sets global trading policies that
    constrain all other components.
    
    Usage:
        controller = MetaStrategyController(
            metrics_repo=metrics_repo,
            strategy_repo=strategy_repo,
            policy_store=policy_store
        )
        
        policy = controller.evaluate_and_update_policy()
    """
    
    def __init__(
        self,
        metrics_repo: MetricsRepository,
        strategy_repo: StrategyRepository,
        policy_store: PolicyStore,
        *,
        min_stats_days: int = 7,
        evaluation_period_days: int = 30,
        min_strategies: int = 2,
        max_strategies: int = 10,
        min_trades_per_strategy: int = 10,
        opportunity_ranker=None  # NEW: OpportunityRanker from app state
    ):
        """
        Initialize Meta Strategy Controller.
        
        Args:
            metrics_repo: Repository for system-wide metrics
            strategy_repo: Repository for strategy data
            policy_store: Store for global policy
            min_stats_days: Minimum days of data for strategy evaluation
            evaluation_period_days: Lookback period for strategy scoring
            min_strategies: Minimum number of strategies to keep active
            max_strategies: Maximum number of strategies to keep active
            min_trades_per_strategy: Minimum trades required for consideration
            opportunity_ranker: OpportunityRanker instance (optional)
        """
        self.metrics_repo = metrics_repo
        self.strategy_repo = strategy_repo
        self.policy_store = policy_store
        self.opportunity_ranker = opportunity_ranker  # NEW
        
        self.min_stats_days = min_stats_days
        self.evaluation_period_days = evaluation_period_days
        self.min_strategies = min_strategies
        self.max_strategies = max_strategies
        self.min_trades_per_strategy = min_trades_per_strategy
        
        # Initialize helper components
        self.scorer = StrategyScorer()
        self.mode_selector = RiskModeSelector()
        self.policy_builder = PolicyBuilder()
        
        logger.info(
            f"MetaStrategyController initialized: "
            f"eval_period={evaluation_period_days}d, "
            f"strategies={min_strategies}-{max_strategies}"
        )
        
        if self.opportunity_ranker:
            logger.info("[MSC AI] OpportunityRanker integration: ENABLED")
    
    def evaluate_and_update_policy(self) -> dict:
        """
        Main entry point: Evaluate system and update global policy.
        
        Pipeline:
        1. Read system metrics
        2. Determine risk mode
        3. Score LIVE strategies
        4. Select allowed strategies
        5. Build policy parameters
        6. Write to PolicyStore
        
        Returns:
            Updated policy as dict
        """
        logger.info("="*80)
        logger.info("MSC AI: Starting policy evaluation cycle")
        logger.info("="*80)
        
        # Step 1: Gather system health metrics
        health = self._gather_system_health()
        self._log_system_health(health)
        
        # Step 2: Determine risk mode
        risk_mode, mode_reasons = self.mode_selector.select_risk_mode(health)
        logger.info(f"\n[RISK MODE] Selected: {risk_mode.value}")
        for reason in mode_reasons:
            logger.info(f"  - {reason}")
        
        # NEW: Adjust risk mode based on OpportunityRanker scores
        if self.opportunity_ranker:
            adjusted_risk_mode, opp_reason = self._adjust_risk_for_opportunities(risk_mode)
            if adjusted_risk_mode != risk_mode:
                logger.info(f"[OPPORTUNITY BOOST] {risk_mode.value} → {adjusted_risk_mode.value}")
                logger.info(f"  Reason: {opp_reason}")
                risk_mode = adjusted_risk_mode
        
        # Step 3: Get and score LIVE strategies
        live_strategies = self._get_live_strategies_with_stats()
        
        if not live_strategies:
            logger.warning("No LIVE strategies found - using empty policy")
            policy = self._create_fallback_policy(risk_mode)
            self.policy_store.update(policy)
            return policy
        
        strategy_scores = self.scorer.score_all_strategies(
            live_strategies,
            health.global_regime
        )
        
        self._log_strategy_scores(strategy_scores)
        
        # Step 4: Select allowed strategies
        allowed_strategies = self._select_allowed_strategies(
            strategy_scores,
            risk_mode
        )
        
        logger.info(f"\n[SELECTION] Selected {len(allowed_strategies)} strategies:")
        for strat_id in allowed_strategies:
            logger.info(f"  ✓ {strat_id}")
        
        # Step 5: Build final policy
        policy_obj = self.policy_builder.build_policy(
            risk_mode=risk_mode,
            allowed_strategies=allowed_strategies,
            allowed_symbols=self._get_opportunity_symbols()  # NEW: Use top-ranked symbols
        )
        
        policy_dict = policy_obj.to_dict()
        
        # Step 6: Write to policy store
        self.policy_store.update(policy_dict)
        
        logger.info("\n[POLICY] Updated successfully")
        logger.info(f"  Risk Mode: {policy_dict['risk_mode']}")
        logger.info(f"  Max Risk/Trade: {policy_dict['max_risk_per_trade']:.2%}")
        logger.info(f"  Min Confidence: {policy_dict['global_min_confidence']:.1%}")
        logger.info(f"  Max Positions: {policy_dict['max_positions']}")
        logger.info(f"  Active Strategies: {len(policy_dict['allowed_strategies'])}")
        logger.info("="*80)
        
        return policy_dict
    
    def _gather_system_health(self) -> SystemHealth:
        """Collect all system-wide health metrics"""
        return SystemHealth(
            current_drawdown_pct=self.metrics_repo.get_current_drawdown_pct(),
            global_winrate=self.metrics_repo.get_global_winrate(last_trades=200),
            equity_curve_slope=self._calculate_equity_slope(),
            global_regime=self.metrics_repo.get_global_regime(),
            volatility_level=self.metrics_repo.get_volatility_level(),
            recent_trade_count=self._get_recent_trade_count(),
            consecutive_losses=self.metrics_repo.get_consecutive_losses(),
            days_since_profit=self.metrics_repo.get_days_since_last_profit()
        )
    
    def _calculate_equity_slope(self) -> float:
        """
        Calculate daily return trend from equity curve.
        
        Returns average daily return % over evaluation period.
        """
        equity_curve = self.metrics_repo.get_equity_curve(
            days=self.evaluation_period_days
        )
        
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_balance = equity_curve[i-1][1]
            curr_balance = equity_curve[i][1]
            
            if prev_balance > 0:
                daily_return_pct = ((curr_balance - prev_balance) / prev_balance) * 100
                daily_returns.append(daily_return_pct)
        
        return mean(daily_returns) if daily_returns else 0.0
    
    def _get_recent_trade_count(self) -> int:
        """Get number of recent trades for activity check"""
        # This would ideally come from metrics_repo
        # For now, we'll use equity curve length as proxy
        equity_curve = self.metrics_repo.get_equity_curve(days=7)
        return len(equity_curve)
    
    def _get_live_strategies_with_stats(
        self
    ) -> list[tuple[StrategyConfig, StrategyStats]]:
        """
        Get all LIVE strategies with their recent performance stats.
        
        Filters out strategies without minimum data requirements.
        """
        live_strategies = self.strategy_repo.get_strategies_by_status("LIVE")
        
        strategies_with_stats = []
        
        for config in live_strategies:
            try:
                stats = self.strategy_repo.get_recent_stats(
                    strategy_id=config.strategy_id,
                    source="LIVE",
                    days=self.evaluation_period_days
                )
                
                # Filter: Must have minimum trades
                if stats.total_trades >= self.min_trades_per_strategy:
                    strategies_with_stats.append((config, stats))
                else:
                    logger.debug(
                        f"Strategy {config.strategy_id} excluded: "
                        f"only {stats.total_trades} trades "
                        f"(min {self.min_trades_per_strategy})"
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Failed to get stats for {config.strategy_id}: {e}"
                )
                continue
        
        return strategies_with_stats
    
    def _select_allowed_strategies(
        self,
        strategy_scores: list[StrategyScore],
        risk_mode: RiskMode
    ) -> list[str]:
        """
        Select subset of strategies to allow based on scores and risk mode.
        
        Selection rules:
        - DEFENSIVE: Top 2-4 strategies (most stable)
        - NORMAL: Top 4-8 strategies
        - AGGRESSIVE: Top 6-10 strategies
        """
        # Define target count by risk mode
        target_counts = {
            RiskMode.DEFENSIVE: (2, 4),
            RiskMode.NORMAL: (4, 8),
            RiskMode.AGGRESSIVE: (6, 10)
        }
        
        min_count, max_count = target_counts[risk_mode]
        
        # Ensure we respect global limits
        min_count = max(min_count, self.min_strategies)
        max_count = min(max_count, self.max_strategies)
        
        # Filter: Only strategies with positive scores
        positive_scores = [s for s in strategy_scores if s.score > 0.3]
        
        if not positive_scores:
            logger.warning("No strategies with positive scores - using top scores anyway")
            positive_scores = strategy_scores
        
        # Select top N strategies
        selected_count = min(len(positive_scores), max_count)
        selected_count = max(selected_count, min(min_count, len(positive_scores)))
        
        selected = positive_scores[:selected_count]
        
        return [s.strategy_id for s in selected]
    
    def _create_fallback_policy(self, risk_mode: RiskMode) -> dict:
        """Create minimal policy when no strategies are available"""
        policy = self.policy_builder.build_policy(
            risk_mode=risk_mode,
            allowed_strategies=[]
        )
        return policy.to_dict()
    
    def _log_system_health(self, health: SystemHealth) -> None:
        """Log system health metrics"""
        logger.info("\n[SYSTEM HEALTH]")
        logger.info(f"  Drawdown: {health.current_drawdown_pct:.2f}%")
        logger.info(f"  Global Winrate: {health.global_winrate:.1%}")
        logger.info(f"  Equity Slope: {health.equity_curve_slope:+.2f}%/day")
        logger.info(f"  Regime: {health.global_regime}")
        logger.info(f"  Volatility: {health.volatility_level}")
        logger.info(f"  Consecutive Losses: {health.consecutive_losses}")
        logger.info(f"  Days Since Profit: {health.days_since_profit}")
    
    def _adjust_risk_for_opportunities(self, base_risk_mode: RiskMode) -> tuple[RiskMode, str]:
        """
        Adjust risk mode based on OpportunityRanker scores.
        
        High opportunity scores → upgrade risk mode (DEFENSIVE→NORMAL, NORMAL→AGGRESSIVE)
        Low opportunity scores → downgrade risk mode (AGGRESSIVE→NORMAL, NORMAL→DEFENSIVE)
        
        Returns:
            (adjusted_risk_mode, reason_string)
        """
        if not self.opportunity_ranker:
            return base_risk_mode, "OpportunityRanker unavailable"
        
        try:
            rankings = self.opportunity_ranker.get_rankings()
            
            if not rankings:
                return base_risk_mode, "No opportunity rankings available"
            
            # Calculate average opportunity score (top 10 symbols)
            top_scores = [r.composite_score for r in rankings[:10]]
            avg_opportunity_score = mean(top_scores) if top_scores else 0.5
            
            # Count high-opportunity symbols (score >= 0.70)
            high_opp_count = sum(1 for r in rankings if r.composite_score >= 0.70)
            
            logger.info(f"[OPPORTUNITY] Avg score (top 10): {avg_opportunity_score:.2f}")
            logger.info(f"[OPPORTUNITY] High-opp symbols (>=0.70): {high_opp_count}")
            
            # Decision logic
            if avg_opportunity_score >= 0.75 and high_opp_count >= 5:
                # Exceptional opportunities → upgrade
                if base_risk_mode == RiskMode.DEFENSIVE:
                    return RiskMode.NORMAL, f"High opportunity detected ({avg_opportunity_score:.2f} avg, {high_opp_count} symbols)"
                elif base_risk_mode == RiskMode.NORMAL:
                    return RiskMode.AGGRESSIVE, f"Exceptional opportunities ({avg_opportunity_score:.2f} avg, {high_opp_count} symbols)"
                
            elif avg_opportunity_score <= 0.50 or high_opp_count <= 2:
                # Poor opportunities → downgrade
                if base_risk_mode == RiskMode.AGGRESSIVE:
                    return RiskMode.NORMAL, f"Limited opportunities ({avg_opportunity_score:.2f} avg, {high_opp_count} symbols)"
                elif base_risk_mode == RiskMode.NORMAL:
                    return RiskMode.DEFENSIVE, f"Poor opportunities ({avg_opportunity_score:.2f} avg, {high_opp_count} symbols)"
            
            return base_risk_mode, f"Opportunities moderate ({avg_opportunity_score:.2f} avg)"
            
        except Exception as e:
            logger.error(f"[MSC AI] Error adjusting for opportunities: {e}")
            return base_risk_mode, f"Error: {e}"
    
    def _get_opportunity_symbols(self) -> Optional[list[str]]:
        """
        Get list of top-ranked symbols from OpportunityRanker.
        
        Returns symbols with score >= 0.65, max 20 symbols.
        Returns None if OpportunityRanker unavailable (allows all symbols).
        """
        if not self.opportunity_ranker:
            return None  # No filtering
        
        try:
            rankings = self.opportunity_ranker.get_rankings()
            
            if not rankings:
                return None
            
            # Filter by minimum score and take top 20
            top_symbols = [
                r.symbol for r in rankings 
                if r.composite_score >= 0.65
            ][:20]
            
            if top_symbols:
                logger.info(f"[MSC AI] Allowed symbols: {len(top_symbols)} from OpportunityRanker")
                logger.debug(f"  Symbols: {top_symbols[:10]}{'...' if len(top_symbols) > 10 else ''}")
                return top_symbols
            else:
                logger.warning("[MSC AI] No symbols above 0.65 threshold, allowing all")
                return None
                
        except Exception as e:
            logger.error(f"[MSC AI] Error getting opportunity symbols: {e}")
            return None
    
    def _log_strategy_scores(self, scores: list[StrategyScore]) -> None:
        """Log strategy scoring results"""
        logger.info(f"\n[STRATEGY SCORES] Evaluated {len(scores)} strategies:")
        
        for i, score in enumerate(scores[:10], 1):  # Top 10
            logger.info(
                f"  {i:2}. {score.strategy_id:30} | "
                f"Score: {score.score:.3f} | "
                f"Trades: {score.total_trades:3} | "
                f"PF: {score.profit_factor:.3f} | "
                f"WR: {score.winrate:.3f}"
            )
