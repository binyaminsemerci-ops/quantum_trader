"""
AI TRADING MATHEMATICIAN AGENT
Automatically calculates optimal trading parameters:
- Position size (margin)
- Leverage
- TP/SL levels
- Risk/Reward ratios

NO MANUAL ADJUSTMENTS NEEDED!
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    """Current account state."""
    balance: float
    equity: float
    margin_used: float
    open_positions: int
    max_positions: int


@dataclass
class MarketConditions:
    """Current market volatility and conditions."""
    symbol: str
    atr_pct: float  # Average True Range as percentage
    daily_volatility: float
    trend_strength: float  # 0-1
    liquidity_score: float  # 0-1


@dataclass
class PerformanceMetrics:
    """Historical performance data."""
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    sharpe_ratio: float


@dataclass
class OptimalParameters:
    """Calculated optimal trading parameters."""
    margin_usd: float
    leverage: float
    notional_usd: float
    tp_pct: float
    sl_pct: float
    partial_tp_pct: float
    expected_profit_usd: float
    max_loss_usd: float
    risk_reward_ratio: float
    confidence_score: float


class TradingMathematician:
    """
    AI agent that calculates optimal trading parameters
    based on account state, market conditions, and performance.
    
    NO MORE MANUAL ADJUSTMENTS!
    """
    
    def __init__(
        self,
        risk_per_trade_pct: float = 0.80,  # 80% of balance per trade (AGGRESSIVE!)
        target_profit_pct: float = 0.20,    # 20% daily profit target
        min_risk_reward: float = 2.0,       # Minimum 2:1 R:R
        safety_cap: float = 75.0,           # Safety cap (bug protection, not optimization)
        conservative_mode: bool = False,
    ):
        self.risk_per_trade_pct = risk_per_trade_pct
        self.target_profit_pct = target_profit_pct
        self.min_risk_reward = min_risk_reward
        self.safety_cap = safety_cap
        self.conservative_mode = conservative_mode
        
        # Binance Futures leverage limits per symbol
        self.binance_max_leverage = {
            "BTCUSDT": 125, "ETHUSDT": 100, "BNBUSDT": 75,
            "SOLUSDT": 75, "XRPUSDT": 75, "ADAUSDT": 75,
            "DOGEUSDT": 75, "DOTUSDT": 75, "MATICUSDT": 75,
            "LINKUSDT": 75, "AVAXUSDT": 75, "ATOMUSDT": 75,
            "UNIUSDT": 50, "LTCUSDT": 75, "ETCUSDT": 75,
            "INJUSDT": 50, "SHIBUSDT": 50, "SUIUSDT": 50,
        }
        
        logger.info(f"🧮 Trading Mathematician initialized:")
        logger.info(f"   Risk per trade: {risk_per_trade_pct*100}%")
        logger.info(f"   Target profit: {target_profit_pct*100}%")
        logger.info(f"   Min R:R: {min_risk_reward}:1")
        logger.info(f"   Kelly safety cap: {safety_cap}x")
    
    def calculate_optimal_parameters(
        self,
        account: AccountState,
        market: MarketConditions,
        performance: PerformanceMetrics,
        signal_confidence: float = 0.70,  # AI ensemble confidence
    ) -> OptimalParameters:
        """
        Calculate optimal trading parameters using AI-driven math.
        
        This is the CORE intelligence - replaces manual adjustments!
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🧮 CALCULATING OPTIMAL PARAMETERS FOR {market.symbol}")
        logger.info(f"{'='*80}")
        
        # Step 1: Calculate base margin allocation
        available_capital = account.balance - account.margin_used
        margin_target = self._calculate_risk_amount(account, available_capital)
        logger.info(f"💰 Margin target: ${margin_target:.2f} ({self.risk_per_trade_pct*100}% of ${account.balance:.2f})")
        
        # Step 2: Calculate optimal SL based on ATR
        sl_pct = self._calculate_optimal_sl(market, performance)
        logger.info(f"🛡️  Optimal SL: {sl_pct*100:.2f}% (based on ATR={market.atr_pct*100:.2f}%)")
        
        # Step 3: Calculate optimal TP based on win rate and R:R
        tp_pct = self._calculate_optimal_tp(sl_pct, performance, market)
        logger.info(f"🎯 Optimal TP: {tp_pct*100:.2f}% (R:R={tp_pct/sl_pct:.2f}:1)")
        
        # Step 4: Calculate optimal leverage WITH confidence adjustment
        leverage = self._calculate_optimal_leverage(market, performance, account, signal_confidence)
        logger.info(f"⚡ Optimal Leverage: {leverage:.1f}x")
        
        # Step 5: Use margin target directly (no complex formula)
        margin = margin_target  # Simple: Use 80% of balance as margin
        logger.info(f"📊 Position Size: ${margin:.2f} margin")
        
        # Step 6: Calculate notional and expected outcomes
        notional = margin * leverage
        expected_profit = notional * tp_pct
        max_loss = notional * sl_pct
        
        logger.info(f"💵 Notional: ${notional:.2f}")
        logger.info(f"✅ Expected Profit: ${expected_profit:.2f}")
        logger.info(f"❌ Max Loss: ${max_loss:.2f}")
        
        # Step 7: Partial TP calculation
        partial_tp_pct = tp_pct * 0.5  # Take 50% profit at halfway
        
        # Step 8: Confidence score (how confident are we in these params?)
        confidence = self._calculate_confidence(
            performance, market, account, leverage, tp_pct, sl_pct
        )
        logger.info(f"🎲 Confidence: {confidence*100:.1f}%")
        
        # Step 9: Safety checks - reduce if too risky
        if max_loss > margin_target * 0.5:  # Max loss shouldn't exceed 50% of margin
            logger.warning(f"⚠️  Max loss ${max_loss:.2f} exceeds 50% of margin ${margin_target:.2f}")
            logger.warning(f"   Reducing leverage for safety...")
            leverage = leverage * (margin_target * 0.5 / max_loss)
            margin = margin_target  # Keep same margin target
            notional = margin * leverage
            expected_profit = notional * tp_pct
            max_loss = notional * sl_pct
            logger.info(f"   Adjusted: {leverage:.1f}x leverage, ${margin:.2f} margin")
        
        logger.info(f"{'='*80}\n")
        
        return OptimalParameters(
            margin_usd=margin,
            leverage=leverage,
            notional_usd=notional,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            partial_tp_pct=partial_tp_pct,
            expected_profit_usd=expected_profit,
            max_loss_usd=max_loss,
            risk_reward_ratio=tp_pct / sl_pct,
            confidence_score=confidence,
        )
    
    def _calculate_risk_amount(self, account: AccountState, available: float) -> float:
        """Calculate how much $ MARGIN to use on this trade."""
        # Base margin allocation - use configured risk percentage
        base_margin = account.balance * self.risk_per_trade_pct
        
        logger.info(f"💰 Using {self.risk_per_trade_pct*100:.1f}% of balance AS MARGIN: ${base_margin:.2f} (of ${account.balance:.2f})")
        
        # Check if we have enough available
        if available < base_margin:
            logger.warning(f"⚠️  Available ${available:.2f} < target ${base_margin:.2f}, using what's available")
            return available * 0.95  # Use 95% of available
        
        # Return MARGIN allocation (not risk amount)
        return base_margin
    
    def _calculate_optimal_sl(
        self, market: MarketConditions, performance: PerformanceMetrics
    ) -> float:
        """
        Calculate optimal stop loss based on ATR and market conditions.
        
        NEW STRATEGY: Roomy initial SL (2.5-3%) to avoid whipsaw.
        SL will be tightened dynamically as trade moves into profit.
        
        Key principle: Start wide to give trade room, tighten as profit increases.
        """
        # Base SL on ATR (2.0x ATR for more room)
        base_sl = market.atr_pct * 2.0
        
        # Adjust for trend strength (less aggressive adjustment)
        if market.trend_strength > 0.7:
            # Strong trend, slightly tighter SL
            base_sl *= 0.9
            logger.debug(f"   Strong trend detected, tightening SL by 10%")
        elif market.trend_strength < 0.3:
            # Choppy market, wider SL to avoid whipsaw
            base_sl *= 1.15
            logger.debug(f"   Choppy market, widening SL by 15%")
        
        # Adjust based on historical losses (less aggressive)
        if performance.total_trades > 10:
            avg_loss = performance.avg_loss_pct
            if avg_loss > base_sl * 1.3:
                # Historical losses larger, adjust upward
                base_sl = avg_loss * 0.95  # 95% of avg loss
                logger.debug(f"   Historical losses larger, adjusting to {base_sl*100:.2f}%")
        
        # Conservative mode: even wider SL
        if self.conservative_mode:
            base_sl *= 1.2
            logger.debug(f"   Conservative mode: widening SL by 20%")
        
        # NEW BOUNDS: 2.5-3% range (roomy initial SL)
        min_sl = 0.025  # 2.5% minimum (avoid whipsaw)
        max_sl = 0.030  # 3.0% maximum (protect capital)
        
        return max(min_sl, min(max_sl, base_sl))
    
    def _calculate_optimal_tp(
        self, sl_pct: float, performance: PerformanceMetrics, market: MarketConditions
    ) -> float:
        """
        Calculate optimal take profit for FIRST partial TP (TP2 target).
        
        NEW STRATEGY: Tighter TP targets for frequent profit taking.
        - TP1 (50%): 1.5-2.0% (handled by partial system)
        - TP2 (30%): 3.0-4.0% (this calculation)
        - TP3 (20%): Trailing from +5% (handled by trailing system)
        
        Key principle: Take profit frequently with smaller targets.
        """
        # NEW: Target 3-4% for TP2 (main target)
        # This is independent of SL size (not R:R based anymore)
        base_tp = 0.035  # 3.5% base target
        
        # Adjust based on volatility
        if market.daily_volatility > 0.05:  # >5% daily volatility
            # High volatility, can aim slightly higher
            base_tp = 0.040  # 4.0% target
            logger.debug(f"   High volatility ({market.daily_volatility*100:.1f}%), TP target: 4.0%")
        elif market.daily_volatility < 0.02:  # <2% daily volatility
            # Low volatility, aim for smaller target
            base_tp = 0.030  # 3.0% target
            logger.debug(f"   Low volatility, TP target: 3.0%")
        
        # Adjust for trend (modest adjustment)
        if market.trend_strength > 0.7:
            # Strong trend, can aim slightly higher
            base_tp *= 1.1  # +10%
            logger.debug(f"   Strong trend, increasing TP by 10%")
        
        # Adjust based on historical wins (if available)
        if performance.total_trades > 10 and performance.avg_win_pct > 0:
            # If historical wins are smaller, use 90% of avg
            if performance.avg_win_pct < base_tp:
                historical_tp = performance.avg_win_pct * 0.9
                base_tp = historical_tp
                logger.debug(f"   Historical avg win: {performance.avg_win_pct*100:.2f}%, adjusting to {base_tp*100:.2f}%")
        
        # NEW BOUNDS: 3.0-4.5% range for TP2
        min_tp = 0.030  # 3.0% minimum for TP2
        max_tp = 0.045  # 4.5% maximum for TP2
        
        return max(min_tp, min(max_tp, base_tp))
    
    def _calculate_optimal_leverage(
        self,
        market: MarketConditions,
        performance: PerformanceMetrics,
        account: AccountState,
        signal_confidence: float = 0.70,  # Signal confidence from AI ensemble
    ) -> float:
        """
        Calculate optimal leverage using Kelly Criterion.
        
        Formula: Optimal Leverage = Edge / Variance
        - Edge = (Win_Rate × Avg_Win) - (Loss_Rate × Avg_Loss)
        - Variance = Win_Rate × (Avg_Win)² + Loss_Rate × (Avg_Loss)²
        
        This maximizes long-term growth rate while accounting for risk.
        """
        # Minimum trades for Kelly leverage
        if performance.total_trades < 5:
            # Not enough history, use conservative default
            default_lev = 10.0
            binance_max = self.binance_max_leverage.get(market.symbol, 50)
            safe_lev = min(default_lev, binance_max, self.safety_cap)
            logger.info(f"   📊 Limited history ({performance.total_trades} trades), using conservative {safe_lev:.1f}x")
            return safe_lev
        
        # Calculate edge and variance
        win_rate = performance.win_rate
        loss_rate = 1 - win_rate
        avg_win = performance.avg_win_pct
        avg_loss = abs(performance.avg_loss_pct)  # Make positive
        
        if avg_win == 0 or avg_loss == 0:
            logger.warning("   ⚠️  Invalid avg_win/loss, using default 10x")
            return 10.0
        
        # Kelly Criterion for leverage
        edge = (win_rate * avg_win) - (loss_rate * avg_loss)
        variance = (win_rate * (avg_win ** 2)) + (loss_rate * (avg_loss ** 2))
        
        if variance == 0 or edge <= 0:
            logger.warning(f"   ⚠️  No edge detected (edge={edge:.4f}), using minimum 5x")
            return 5.0
        
        # Optimal Kelly leverage
        kelly_leverage = edge / variance
        
        # Use fractional Kelly (50%) for safety
        fractional_kelly = kelly_leverage * 0.5
        
        logger.info(
            f"   🎲 KELLY LEVERAGE: Edge={edge*100:.2f}%, Variance={variance*100:.2f}% → "
            f"Full Kelly={kelly_leverage:.1f}x, Fractional (50%)={fractional_kelly:.1f}x"
        )
        
        # Adjust for market conditions
        adjusted_lev = fractional_kelly
        
        if market.daily_volatility > 0.08:  # >8% daily volatility
            adjusted_lev *= 0.7
            logger.debug(f"   High volatility ({market.daily_volatility*100:.1f}%), reducing by 30%")
        
        if market.liquidity_score < 0.5:
            adjusted_lev *= 0.8
            logger.debug(f"   Low liquidity, reducing by 20%")
        
        # Portfolio utilization adjustment
        utilization = account.open_positions / account.max_positions
        if utilization > 0.7:
            adjusted_lev *= 0.85
            logger.debug(f"   Portfolio {utilization*100:.0f}% full, reducing leverage")
        
        # Conservative mode
        if self.conservative_mode:
            adjusted_lev *= 0.6
            logger.debug(f"   Conservative mode: reducing by 40%")
        
        # 🆕 CONFIDENCE ADJUSTMENT: Scale leverage by signal confidence
        # High confidence (>80%) → Full leverage
        # Medium confidence (60-80%) → 70-100% leverage  
        # Low confidence (<60%) → 50-70% leverage
        if signal_confidence < 0.80:
            confidence_mult = 0.5 + (signal_confidence * 0.625)  # 0.5 at 0%, 1.0 at 80%
            adjusted_lev *= confidence_mult
            logger.info(
                f"   🎯 CONFIDENCE ADJUSTMENT: {signal_confidence*100:.0f}% confidence "
                f"→ {confidence_mult*100:.0f}% of Kelly leverage"
            )
        
        # Apply limits: Binance max, safety cap, minimum 5x
        binance_max = self.binance_max_leverage.get(market.symbol, 50)
        final_lev = max(5.0, min(adjusted_lev, binance_max, self.safety_cap))
        
        logger.info(
            f"   ✅ FINAL LEVERAGE: {final_lev:.1f}x (Binance max={binance_max}x, Safety cap={self.safety_cap:.0f}x)"
        )
        
        return round(final_lev, 1)
    
    def _calculate_position_size(
        self, risk_amount: float, sl_pct: float, leverage: float
    ) -> float:
        """
        Calculate position size (margin) based on risk amount and SL.
        
        Formula: Margin = Risk_Amount / (SL% × Leverage)
        
        This ensures that if SL hits, we lose exactly risk_amount.
        """
        # Calculate required margin
        # If SL is 1.5% and leverage is 10x, then:
        # Notional loss = Margin × Leverage × SL%
        # We want: Notional loss = Risk_Amount
        # So: Margin = Risk_Amount / (Leverage × SL%)
        
        margin = risk_amount / (leverage * sl_pct)
        
        # Round to 2 decimals
        return round(margin, 2)
    
    def _calculate_confidence(
        self,
        performance: PerformanceMetrics,
        market: MarketConditions,
        account: AccountState,
        leverage: float,
        tp_pct: float,
        sl_pct: float,
    ) -> float:
        """
        Calculate confidence score in these parameters (0-1).
        
        Higher confidence = more reliable parameters.
        """
        confidence = 1.0
        
        # Reduce confidence if limited trading history
        if performance.total_trades < 10:
            confidence *= 0.6
        elif performance.total_trades < 50:
            confidence *= 0.8
        elif performance.total_trades < 100:
            confidence *= 0.9
        
        # Reduce confidence for extreme leverage
        if leverage > 15:
            confidence *= 0.7
        elif leverage > 10:
            confidence *= 0.85
        
        # Reduce confidence in choppy markets
        if market.trend_strength < 0.3:
            confidence *= 0.85
        
        # Reduce confidence if portfolio is very full
        utilization = account.open_positions / account.max_positions
        if utilization > 0.8:
            confidence *= 0.9
        
        # Reduce confidence if R:R is too tight
        rr = tp_pct / sl_pct
        if rr < 2.0:
            confidence *= 0.8
        
        return round(confidence, 3)
    
    def adjust_for_kelly_criterion(
        self, params: OptimalParameters, performance: PerformanceMetrics
    ) -> OptimalParameters:
        """
        Apply Kelly Criterion for optimal position sizing.
        
        Kelly% = (Win_Rate × Avg_Win - Loss_Rate × Avg_Loss) / Avg_Win
        
        This maximizes long-term growth rate.
        """
        if performance.total_trades < 20:
            # Not enough data for Kelly
            logger.info("🎲 Skipping Kelly (need >20 trades)")
            return params
        
        win_rate = performance.win_rate
        loss_rate = 1 - win_rate
        avg_win = performance.avg_win_pct
        avg_loss = performance.avg_loss_pct
        
        if avg_win == 0:
            return params
        
        # Kelly formula
        kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # Use fractional Kelly (0.5 × Kelly) for safety
        fractional_kelly = kelly_pct * 0.5
        
        logger.info(f"🎲 Kelly Criterion: {kelly_pct*100:.2f}% → Using {fractional_kelly*100:.2f}% (fractional)")
        
        # Adjust margin based on Kelly
        if fractional_kelly > 0:
            kelly_multiplier = min(2.0, fractional_kelly / self.risk_per_trade_pct)
            adjusted_margin = params.margin_usd * kelly_multiplier
            adjusted_notional = adjusted_margin * params.leverage
            adjusted_profit = adjusted_notional * params.tp_pct
            adjusted_loss = adjusted_notional * params.sl_pct
            
            logger.info(f"   Adjusted margin: ${params.margin_usd:.2f} → ${adjusted_margin:.2f}")
            
            return OptimalParameters(
                margin_usd=adjusted_margin,
                leverage=params.leverage,
                notional_usd=adjusted_notional,
                tp_pct=params.tp_pct,
                sl_pct=params.sl_pct,
                partial_tp_pct=params.partial_tp_pct,
                expected_profit_usd=adjusted_profit,
                max_loss_usd=adjusted_loss,
                risk_reward_ratio=params.risk_reward_ratio,
                confidence_score=params.confidence_score * 0.95,  # Slightly reduce confidence
            )
        else:
            logger.warning("⚠️  Kelly Criterion suggests NO POSITION (negative edge)")
            return params


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize mathematician
    math_ai = TradingMathematician(
        risk_per_trade_pct=0.02,  # 2% risk per trade
        target_profit_pct=0.05,    # 5% daily target
        min_risk_reward=2.0,
        max_leverage=20.0,
    )
    
    # Mock data
    account = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin_used=0.0,
        open_positions=0,
        max_positions=15,
    )
    
    market = MarketConditions(
        symbol="BTCUSDT",
        atr_pct=0.015,  # 1.5% ATR
        daily_volatility=0.04,  # 4% daily volatility
        trend_strength=0.75,  # Strong trend
        liquidity_score=0.9,  # High liquidity
    )
    
    performance = PerformanceMetrics(
        total_trades=85,
        win_rate=0.55,  # 55% win rate
        avg_win_pct=0.035,  # 3.5% average win
        avg_loss_pct=0.018,  # 1.8% average loss
        profit_factor=1.6,
        sharpe_ratio=1.8,
    )
    
    # Calculate optimal parameters
    optimal = math_ai.calculate_optimal_parameters(account, market, performance)
    
    print(f"\n{'='*80}")
    print(f"✅ OPTIMAL PARAMETERS CALCULATED")
    print(f"{'='*80}")
    print(f"Margin: ${optimal.margin_usd:.2f}")
    print(f"Leverage: {optimal.leverage:.1f}x")
    print(f"Notional: ${optimal.notional_usd:.2f}")
    print(f"TP: {optimal.tp_pct*100:.2f}%")
    print(f"SL: {optimal.sl_pct*100:.2f}%")
    print(f"Expected Profit: ${optimal.expected_profit_usd:.2f}")
    print(f"Max Loss: ${optimal.max_loss_usd:.2f}")
    print(f"R:R: {optimal.risk_reward_ratio:.2f}:1")
    print(f"Confidence: {optimal.confidence_score*100:.1f}%")
    print(f"{'='*80}\n")
    
    # Apply Kelly Criterion
    kelly_optimal = math_ai.adjust_for_kelly_criterion(optimal, performance)
