"""
Integration layer for Trading Mathematician AI
Replaces manual parameter adjustments with intelligent calculations
"""
import os
import logging
from typing import Dict, Optional
from backend.services.ai.trading_mathematician import (
    TradingMathematician,
    AccountState,
    MarketConditions,
    PerformanceMetrics,
    OptimalParameters,
)

logger = logging.getLogger(__name__)


class MathAIIntegration:
    """
    Integrates Trading Mathematician into the trading system.
    
    This replaces:
    - Manual margin adjustments
    - Manual leverage adjustments  
    - Manual TP/SL adjustments
    - Manual risk calculations
    
    With AI-driven mathematical optimization!
    """
    
    def __init__(self, db_session, config):
        self.db = db_session
        self.config = config
        
        # READ FROM ENVIRONMENT! 🔧
        max_leverage = float(os.getenv("RM_MAX_LEVERAGE", "30.0"))
        risk_per_trade = float(os.getenv("RM_RISK_PER_TRADE_PCT", "0.02"))
        
        logger.info(f"🧮 Math AI Config: MAX_LEVERAGE={max_leverage}x, RISK={risk_per_trade*100}%")
        
        # Initialize the mathematician - USE ENVIRONMENT SETTINGS!
        self.math_ai = TradingMathematician(
            risk_per_trade_pct=risk_per_trade,
            target_profit_pct=0.20,        # 20% daily target
            min_risk_reward=2.0,
            max_leverage=max_leverage,     # FROM ENVIRONMENT! 🎯
            conservative_mode=False,
        )
        
        logger.info("🧮 Math AI Integration initialized")
    
    def get_optimal_parameters(
        self,
        symbol: str,
        account_balance: float,
        open_positions: int,
        max_positions: int,
    ) -> OptimalParameters:
        """
        Get optimal trading parameters for a symbol.
        
        This is called BEFORE opening any position.
        Returns margin, leverage, TP, SL automatically calculated!
        """
        logger.info(f"\n🧮 Getting optimal parameters for {symbol}...")
        
        # Get account state
        account = self._get_account_state(
            account_balance, open_positions, max_positions
        )
        
        # Get market conditions
        market = self._get_market_conditions(symbol)
        
        # Get performance metrics
        performance = self._get_performance_metrics()
        
        # Calculate optimal parameters
        optimal = self.math_ai.calculate_optimal_parameters(
            account, market, performance
        )
        
        # Apply Kelly Criterion if enough history
        if performance.total_trades >= 20:
            optimal = self.math_ai.adjust_for_kelly_criterion(optimal, performance)
        
        # Log results
        logger.info(f"✅ Optimal parameters calculated:")
        logger.info(f"   Margin: ${optimal.margin_usd:.2f}")
        logger.info(f"   Leverage: {optimal.leverage:.1f}x")
        logger.info(f"   TP: {optimal.tp_pct*100:.2f}%")
        logger.info(f"   SL: {optimal.sl_pct*100:.2f}%")
        logger.info(f"   Expected profit: ${optimal.expected_profit_usd:.2f}")
        logger.info(f"   Risk: ${optimal.max_loss_usd:.2f}")
        logger.info(f"   R:R: {optimal.risk_reward_ratio:.2f}:1")
        
        return optimal
    
    def _get_account_state(
        self, balance: float, open_positions: int, max_positions: int
    ) -> AccountState:
        """Get current account state from database."""
        # TODO: Get actual margin used from open positions
        margin_used = 0.0  # Calculate from db
        
        return AccountState(
            balance=balance,
            equity=balance,  # TODO: Calculate with unrealized PnL
            margin_used=margin_used,
            open_positions=open_positions,
            max_positions=max_positions,
        )
    
    def _get_market_conditions(self, symbol: str) -> MarketConditions:
        """Calculate current market conditions for symbol."""
        # TODO: Calculate from recent price data
        
        # For now, use reasonable defaults
        # These should be calculated from actual market data!
        return MarketConditions(
            symbol=symbol,
            atr_pct=0.015,           # 1.5% ATR (calculate from data!)
            daily_volatility=0.04,   # 4% daily vol (calculate!)
            trend_strength=0.7,      # Strong trend (calculate!)
            liquidity_score=0.9,     # High liquidity (calculate!)
        )
    
    def _get_performance_metrics(self) -> PerformanceMetrics:
        """Get historical performance from database."""
        # TODO: Query TradeLog table for actual metrics
        
        # For now, use defaults based on 85 closed positions we know about
        # These should be calculated from actual trade history!
        return PerformanceMetrics(
            total_trades=85,
            win_rate=0.55,           # TODO: Calculate from db
            avg_win_pct=0.035,       # TODO: Calculate from db
            avg_loss_pct=0.018,      # TODO: Calculate from db
            profit_factor=1.6,       # TODO: Calculate from db
            sharpe_ratio=1.8,        # TODO: Calculate from db
        )


def integrate_math_ai_into_rl_agent(rl_agent, math_ai_integration):
    """
    Monkey-patch the RL agent to use Math AI for parameter selection.
    
    This makes the system FULLY AUTONOMOUS!
    """
    original_get_action = rl_agent.get_action
    
    def math_ai_enhanced_get_action(state_key: str, symbol: str, confidence: float):
        """Enhanced action selection using Math AI."""
        
        # Get optimal parameters from Math AI
        optimal = math_ai_integration.get_optimal_parameters(
            symbol=symbol,
            account_balance=10000.0,  # TODO: Get from account
            open_positions=3,          # TODO: Get from position manager
            max_positions=15,
        )
        
        # Convert to RL action format
        # Find closest size multiplier
        size_mult = optimal.margin_usd / rl_agent.max_position_usd
        size_mult = max(0.1, min(1.0, size_mult))  # Clamp to valid range
        
        # Use calculated leverage
        leverage = optimal.leverage
        
        # Override RL's TP/SL with Math AI's calculations
        action = {
            'size_multiplier': size_mult,
            'leverage': leverage,
            'tp_pct': optimal.tp_pct,
            'sl_pct': optimal.sl_pct,
            'partial_tp_pct': optimal.partial_tp_pct,
        }
        
        logger.info(f"🧮 Math AI overriding RL action:")
        logger.info(f"   Size: {size_mult*100:.0f}% (${optimal.margin_usd:.2f})")
        logger.info(f"   Leverage: {leverage:.1f}x")
        logger.info(f"   TP/SL: {optimal.tp_pct*100:.2f}% / {optimal.sl_pct*100:.2f}%")
        
        return action
    
    # Replace RL agent's get_action with Math AI version
    rl_agent.get_action = math_ai_enhanced_get_action
    
    logger.info("✅ Math AI integrated into RL Agent - FULLY AUTONOMOUS!")
    
    return rl_agent
