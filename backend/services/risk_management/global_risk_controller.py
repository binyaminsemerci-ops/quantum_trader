"""Global Risk Controller - Portfolio-level risk management."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from backend.config.risk_management import GlobalRiskConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    action: str  # "LONG" or "SHORT"


@dataclass
class PositionInfo:
    """Information about an open position."""
    symbol: str
    action: str
    entry_price: float
    quantity: float
    notional_usd: float
    unrealized_pnl_usd: float


@dataclass
class RiskStatus:
    """Current risk status of the portfolio."""
    equity_usd: float
    
    # Drawdown tracking
    peak_equity: float
    daily_drawdown_pct: float
    weekly_drawdown_pct: float
    
    # Position tracking
    open_positions: int
    total_exposure_pct: float
    
    # Streak tracking
    recent_trades: List[TradeRecord]
    consecutive_losses: int
    
    # Breaker status
    circuit_breaker_active: bool
    circuit_breaker_until: Optional[datetime] = None
    recovery_mode_active: bool = False


@dataclass
class RiskCheckResult:
    """Result of global risk check."""
    approved: bool
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    risk_multiplier: float = 1.0  # Adjusted risk (0.5 = half size in recovery)


class GlobalRiskController:
    """
    Portfolio-level risk management.
    
    Enforces:
    - Daily/weekly drawdown limits
    - Max concurrent positions
    - Max portfolio exposure
    - Losing streak protection
    - Circuit breaker on extreme losses
    - Recovery mode after drawdowns
    """
    
    def __init__(self, config: GlobalRiskConfig):
        self.config = config
        
        # State tracking - Initialize to None, will be set on first equity update
        self.peak_equity_daily: Optional[float] = None
        self.peak_equity_weekly: Optional[float] = None
        self.daily_reset_time: datetime = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        self.weekly_reset_time: datetime = self._get_week_start()
        
        self.circuit_breaker_active = False
        self.circuit_breaker_until: Optional[datetime] = None
        self.recovery_mode_active = False
        
        # Trade history (last 20 trades for streak detection)
        self.recent_trades: deque[TradeRecord] = deque(maxlen=20)
        
        logger.info("[OK] GlobalRiskController initialized")
        logger.info(f"   Max daily DD: {config.max_daily_drawdown_pct:.1%}")
        logger.info(f"   Max concurrent trades: {config.max_concurrent_trades}")
        logger.info(f"   Max exposure: {config.max_exposure_pct:.0%}")
        logger.info(f"   Losing streak protection: {config.losing_streak_threshold} trades")
    
    def _get_week_start(self) -> datetime:
        """Get start of current week (Monday 00:00 UTC)."""
        now = datetime.now(timezone.utc)
        days_since_monday = now.weekday()
        week_start = now - timedelta(days=days_since_monday)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def update_equity_peaks(self, current_equity: float):
        """Update peak equity tracking for drawdown calculation."""
        now = datetime.now(timezone.utc)
        
        # Initialize peaks on first call
        if self.peak_equity_daily is None:
            self.peak_equity_daily = current_equity
            logger.info(f"[INIT] Peak equity daily initialized: ${current_equity:.2f}")
        if self.peak_equity_weekly is None:
            self.peak_equity_weekly = current_equity
            logger.info(f"[INIT] Peak equity weekly initialized: ${current_equity:.2f}")
        
        # Reset daily peak at midnight UTC
        if now.date() > self.daily_reset_time.date():
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
            self.peak_equity_daily = current_equity
            logger.info(f"[RESET] Daily reset: Peak equity = ${current_equity:.2f}")
        
        # Reset weekly peak on Monday
        week_start = self._get_week_start()
        if week_start > self.weekly_reset_time:
            self.weekly_reset_time = week_start
            self.peak_equity_weekly = current_equity
            logger.info(f"[RESET] Weekly reset: Peak equity = ${current_equity:.2f}")
        
        # Update peaks if new high
        if current_equity > self.peak_equity_daily:
            self.peak_equity_daily = current_equity
        if current_equity > self.peak_equity_weekly:
            self.peak_equity_weekly = current_equity
    
    def check_new_trade(
        self,
        symbol: str,
        action: str,
        proposed_notional_usd: float,
        current_equity: float,
        open_positions: List[PositionInfo],
    ) -> RiskCheckResult:
        """
        Check if a new trade is approved under global risk rules.
        
        Args:
            symbol: Trading pair
            action: "LONG" or "SHORT"
            proposed_notional_usd: Proposed position size
            current_equity: Current account equity
            open_positions: List of currently open positions
        
        Returns:
            RiskCheckResult with approval and risk adjustments
        """
        warnings = []
        risk_multiplier = 1.0
        
        # Update equity peaks
        self.update_equity_peaks(current_equity)
        
        # Check 1: Circuit Breaker
        if self.circuit_breaker_active:
            if datetime.now(timezone.utc) < self.circuit_breaker_until:
                time_left = (self.circuit_breaker_until - datetime.now(timezone.utc)).total_seconds() / 3600
                logger.warning(
                    f"[ALERT] Circuit Breaker Active: Trading paused for {time_left:.1f}h more"
                )
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Circuit breaker active (cooling down for {time_left:.1f}h)",
                )
            else:
                # Cooldown expired, reset
                self.circuit_breaker_active = False
                self.circuit_breaker_until = None
                logger.info("[OK] Circuit breaker reset - trading resumed")
        
        # Check 2: Daily Drawdown
        daily_dd = (self.peak_equity_daily - current_equity) / self.peak_equity_daily if self.peak_equity_daily > 0 else 0
        if daily_dd > self.config.max_daily_drawdown_pct:
            logger.error(
                f"[BLOCKED] Daily drawdown limit exceeded: {daily_dd:.2%} > {self.config.max_daily_drawdown_pct:.2%}"
            )
            
            # Trigger circuit breaker if extreme
            if self.config.enable_circuit_breaker and daily_dd > self.config.circuit_breaker_loss_pct:
                self._trigger_circuit_breaker()
            
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Daily drawdown {daily_dd:.2%} exceeds limit {self.config.max_daily_drawdown_pct:.2%}",
            )
        
        # Check 3: Weekly Drawdown
        weekly_dd = (self.peak_equity_weekly - current_equity) / self.peak_equity_weekly if self.peak_equity_weekly > 0 else 0
        if weekly_dd > self.config.max_weekly_drawdown_pct:
            logger.error(
                f"[BLOCKED] Weekly drawdown limit exceeded: {weekly_dd:.2%} > {self.config.max_weekly_drawdown_pct:.2%}"
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Weekly drawdown {weekly_dd:.2%} exceeds limit {self.config.max_weekly_drawdown_pct:.2%}",
            )
        
        # Check 4: Recovery Mode
        if self.config.enable_recovery_mode and daily_dd > self.config.recovery_threshold_pct:
            if not self.recovery_mode_active:
                self.recovery_mode_active = True
                logger.warning(
                    f"[WARNING]  Recovery mode activated: DD {daily_dd:.2%} > {self.config.recovery_threshold_pct:.2%}"
                )
            
            risk_multiplier *= self.config.recovery_risk_multiplier
            warnings.append(
                f"Recovery mode: Risk reduced to {risk_multiplier:.0%} "
                f"(DD: {daily_dd:.2%})"
            )
        else:
            if self.recovery_mode_active:
                self.recovery_mode_active = False
                logger.info("[OK] Recovery mode deactivated - back to normal risk")
        
        # Check 5: Max Concurrent Positions
        if len(open_positions) >= self.config.max_concurrent_trades:
            logger.warning(
                f"[BLOCKED] Max concurrent trades reached: {len(open_positions)} / {self.config.max_concurrent_trades}"
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Max concurrent trades ({self.config.max_concurrent_trades}) reached",
                warnings=warnings,
            )
        
        # Check 6: Max Exposure
        current_exposure = sum(pos.notional_usd for pos in open_positions)
        proposed_total_exposure = current_exposure + proposed_notional_usd
        max_exposure = current_equity * self.config.max_exposure_pct
        
        if proposed_total_exposure > max_exposure:
            logger.warning(
                f"[BLOCKED] Max exposure exceeded: ${proposed_total_exposure:.2f} > ${max_exposure:.2f} "
                f"({self.config.max_exposure_pct:.0%} of equity)"
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Max exposure exceeded: ${proposed_total_exposure:.2f} > ${max_exposure:.2f}",
                warnings=warnings,
            )
        
        # Check 7: Losing Streak Protection
        consecutive_losses = self._count_consecutive_losses()
        if (
            self.config.enable_streak_protection
            and consecutive_losses >= self.config.losing_streak_threshold
        ):
            risk_multiplier *= self.config.streak_risk_reduction
            warnings.append(
                f"Losing streak: {consecutive_losses} losses → "
                f"Risk reduced to {risk_multiplier:.0%}"
            )
            logger.warning(
                f"[WARNING]  Losing streak detected ({consecutive_losses} trades) - "
                f"reducing risk to {risk_multiplier:.0%}"
            )
        
        # Check 8: Correlation (simplified - same symbol check)
        for pos in open_positions:
            if pos.symbol == symbol:
                logger.warning(f"[WARNING]  Already have open position in {symbol}")
                warnings.append(f"Duplicate symbol: already trading {symbol}")
                # Allow but warn - could reject if needed
        
        # All checks passed
        logger.info(
            f"[OK] Global risk check passed for {symbol} {action}:\n"
            f"   Daily DD: {daily_dd:.2%} / {self.config.max_daily_drawdown_pct:.2%}\n"
            f"   Open positions: {len(open_positions)} / {self.config.max_concurrent_trades}\n"
            f"   Exposure: ${proposed_total_exposure:.2f} / ${max_exposure:.2f}\n"
            + (f"   Risk multiplier: {risk_multiplier:.0%}\n" if risk_multiplier != 1.0 else "")
        )
        
        return RiskCheckResult(
            approved=True,
            warnings=warnings,
            risk_multiplier=risk_multiplier,
        )
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades from most recent."""
        count = 0
        for trade in reversed(self.recent_trades):
            if trade.pnl_usd < 0:
                count += 1
            else:
                break
        return count
    
    def _trigger_circuit_breaker(self):
        """Activate circuit breaker and pause trading."""
        self.circuit_breaker_active = True
        self.circuit_breaker_until = datetime.now(timezone.utc) + timedelta(
            hours=self.config.circuit_breaker_cooldown_hours
        )
        logger.critical(
            f"[ALERT][ALERT][ALERT] CIRCUIT BREAKER TRIGGERED [ALERT][ALERT][ALERT]\n"
            f"   Trading paused for {self.config.circuit_breaker_cooldown_hours} hours\n"
            f"   Will resume at {self.circuit_breaker_until.strftime('%Y-%m-%d %H:%M UTC')}"
        )
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade for streak tracking."""
        self.recent_trades.append(trade)
        
        # Log trade outcome
        outcome = "WIN" if trade.pnl_usd > 0 else "LOSS"
        consecutive_losses = self._count_consecutive_losses()
        
        logger.info(
            f"[MEMO] Trade recorded: {trade.symbol} {trade.action} = "
            f"{outcome} ${trade.pnl_usd:.2f} ({trade.r_multiple:.2f}R) | "
            f"Streak: {consecutive_losses} losses"
        )
    
    def get_risk_status(self, current_equity: float, open_positions: List[PositionInfo]) -> RiskStatus:
        """Get current risk status snapshot."""
        self.update_equity_peaks(current_equity)
        
        daily_dd = (self.peak_equity_daily - current_equity) / self.peak_equity_daily if self.peak_equity_daily > 0 else 0
        weekly_dd = (self.peak_equity_weekly - current_equity) / self.peak_equity_weekly if self.peak_equity_weekly > 0 else 0
        
        total_exposure = sum(pos.notional_usd for pos in open_positions)
        exposure_pct = total_exposure / current_equity if current_equity > 0 else 0
        
        return RiskStatus(
            equity_usd=current_equity,
            peak_equity=self.peak_equity_daily,
            daily_drawdown_pct=daily_dd,
            weekly_drawdown_pct=weekly_dd,
            open_positions=len(open_positions),
            total_exposure_pct=exposure_pct,
            recent_trades=list(self.recent_trades),
            consecutive_losses=self._count_consecutive_losses(),
            circuit_breaker_active=self.circuit_breaker_active,
            circuit_breaker_until=self.circuit_breaker_until,
            recovery_mode_active=self.recovery_mode_active,
        )
