"""Trade Lifecycle Manager - Complete trade orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from backend.config.risk_management import RiskManagementConfig
from backend.services.risk_management.trade_opportunity_filter import (
    FilterResult,
    MarketConditions,
    SignalQuality,
    TradeOpportunityFilter,
)
from backend.services.risk_management.risk_manager import PositionSize, RiskManager
from backend.services.risk_management.exit_policy_engine import (
    ExitDecision,
    ExitLevels,
    ExitPolicyEngine,
    ExitSignal,
)
from backend.services.risk_management.global_risk_controller import (
    GlobalRiskController,
    PositionInfo,
    RiskCheckResult,
    TradeRecord,
)

# [TARGET] ORCHESTRATOR INTEGRATION
try:
    from backend.services.governance.orchestrator_policy import TradingPolicy
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    TradingPolicy = None
    
logger = logging.getLogger(__name__)


class TradeState(str, Enum):
    """States in trade lifecycle."""
    NEW = "NEW"                      # Signal received, not yet opened
    APPROVED = "APPROVED"            # Passed all filters, ready to execute
    OPEN = "OPEN"                    # Position opened, monitoring
    PARTIAL_TP = "PARTIAL_TP"        # Partial profit taken
    BREAKEVEN = "BREAKEVEN"          # Stop moved to breakeven
    TRAILING = "TRAILING"            # Trailing stop active
    CLOSED_TP = "CLOSED_TP"          # Closed at take profit
    CLOSED_SL = "CLOSED_SL"          # Closed at stop loss
    CLOSED_PARTIAL = "CLOSED_PARTIAL"  # Closed remaining after partial TP
    CLOSED_TIME = "CLOSED_TIME"      # Closed due to time exit
    REJECTED = "REJECTED"            # Rejected by filters/risk


@dataclass
class ManagedTrade:
    """Complete trade with all tracking data."""
    # Identification
    trade_id: str
    symbol: str
    action: str  # "LONG" or "SHORT"
    state: TradeState
    
    # Entry details
    signal_quality: SignalQuality
    market_conditions: MarketConditions
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    
    # Position sizing
    position_size: Optional[PositionSize] = None
    current_quantity: Optional[float] = None  # Tracks partial exits
    
    # Exit management
    exit_levels: Optional[ExitLevels] = None
    
    # Performance tracking
    highest_price: float = 0.0  # MFE (Maximum Favorable Excursion)
    lowest_price: float = float('inf')  # MAE (Maximum Adverse Excursion)
    highest_r: float = 0.0      # Peak R-multiple
    lowest_r: float = 0.0       # Worst R-multiple
    has_partial_exit: bool = False
    
    # Exit details
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl_usd: float = 0.0
    realized_pnl_pct: float = 0.0
    final_r_multiple: float = 0.0
    
    # Rejection details (if not approved)
    rejection_reason: Optional[str] = None


@dataclass
class TradeDecision:
    """Decision about opening a new trade."""
    approved: bool
    symbol: str
    action: str
    quantity: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Decision trail
    filter_result: Optional[FilterResult] = None
    position_size: Optional[PositionSize] = None
    risk_check: Optional[RiskCheckResult] = None
    rejection_reason: Optional[str] = None


class TradeLifecycleManager:
    """
    Complete trade orchestration from signal to close.
    
    Integrates all risk management components:
    1. TradeOpportunityFilter - Quality filtering
    2. GlobalRiskController - Portfolio-level checks
    3. RiskManager - Position sizing
    4. ExitPolicyEngine - Exit management
    
    State machine:
    NEW → [filters] → APPROVED → [execution] → OPEN → [monitoring] →
    → PARTIAL_TP → BREAKEVEN → TRAILING → CLOSED_*
    """
    
    def __init__(self, config: RiskManagementConfig, ai_engine=None):
        self.config = config
        
        # Initialize components
        self.trade_filter = TradeOpportunityFilter(config.trade_filter)
        self.risk_manager = RiskManager(config.position_sizing)
        self.exit_engine = ExitPolicyEngine(config.exit_policy)
        self.global_risk = GlobalRiskController(config.global_risk)
        
        # AI engine for continuous learning callbacks
        self.ai_engine = ai_engine
        
        # Active trades tracking
        self.active_trades: Dict[str, ManagedTrade] = {}
        
        # [TARGET] ORCHESTRATOR: Track current policy for risk scaling
        self.current_policy: Optional[TradingPolicy] = None
        
        # [FIX] Trade state persistence for Trailing Stop Manager
        self.trade_state_path = Path("/app/backend/data/trade_state.json")
        
        # 🔥 PHASE 3C: Inject components if AI Engine is available
        if ai_engine and hasattr(ai_engine, 'confidence_calibrator'):
            logger.info("[PHASE 3C] 🎯 Attempting to inject Phase 3C components...")
            try:
                # Inject into Risk Manager
                if hasattr(self.risk_manager, 'set_phase3c_components'):
                    self.risk_manager.set_phase3c_components(
                        confidence_calibrator=getattr(ai_engine, 'confidence_calibrator', None),
                        performance_benchmarker=getattr(ai_engine, 'performance_benchmarker', None)
                    )
                    logger.info("[PHASE 3C] ✅ Phase 3C injected into Risk Manager")
                
                # Inject into Trade Filter
                if hasattr(self.trade_filter, 'set_health_monitor'):
                    self.trade_filter.set_health_monitor(
                        health_monitor=getattr(ai_engine, 'health_monitor', None)
                    )
                    logger.info("[PHASE 3C] ✅ Phase 3C injected into Trade Filter")
                    
            except Exception as e:
                logger.warning(f"[PHASE 3C] ⚠️ Failed to inject Phase 3C components: {e}")
        else:
            logger.info("[INFO] Phase 3C components not available in AI Engine")
        
        logger.info("[OK] TradeLifecycleManager initialized with all components")
    
    def set_policy(self, policy: Optional[TradingPolicy]) -> None:
        """Update current trading policy for risk management."""
        self.current_policy = policy
        # Pass policy to RiskManager for position sizing
        if ORCHESTRATOR_AVAILABLE:
            self.risk_manager.set_policy(policy)
    
    def _save_trade_to_state(self, trade: ManagedTrade) -> None:
        """Save trade to state file for Trailing Stop Manager."""
        try:
            # Load current state
            state = {}
            if self.trade_state_path.exists():
                state = json.loads(self.trade_state_path.read_text(encoding="utf-8"))
            
            # Calculate percentages from exit levels
            if trade.exit_levels and trade.entry_price:
                # Calculate TP/SL percentages based on action
                if trade.action == "LONG":
                    tp_pct = (trade.exit_levels.take_profit - trade.entry_price) / trade.entry_price
                    sl_pct = (trade.entry_price - trade.exit_levels.stop_loss) / trade.entry_price
                else:  # SHORT
                    tp_pct = (trade.entry_price - trade.exit_levels.take_profit) / trade.entry_price
                    sl_pct = (trade.exit_levels.stop_loss - trade.entry_price) / trade.entry_price
                
                # NEW: Tighter trail percentage (0.5-1% under peak)
                trail_pct = 0.005  # 0.5% trailing distance
                
                # NEW PARTIAL TP STRUCTURE:
                # TP1 (50%): 1.5-2.0% - Quick profit capture
                # TP2 (30%): 3.0-4.0% - Main target (tp_pct)
                # TP3 (20%): Trailing from +5% - Let winners run
                partial_tp_1_pct = 0.0175  # 1.75% for TP1 (50% position)
                partial_tp_2_pct = tp_pct   # 3-4% for TP2 (30% position)
                
                # Update state for this symbol
                state[trade.symbol] = {
                    "side": trade.action,
                    "qty": trade.current_quantity if trade.action == "LONG" else -trade.current_quantity,
                    "avg_entry": trade.entry_price,
                    "ai_trail_pct": trail_pct,
                    "ai_tp_pct": tp_pct,
                    "ai_sl_pct": sl_pct,
                    "ai_partial_tp": 0.5,  # 50% partial
                    "partial_tp_1_pct": partial_tp_1_pct,
                    "partial_tp_2_pct": partial_tp_2_pct,
                    "partial_tp_1_hit": False,
                    "partial_tp_2_hit": False,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }
                
                # Save state
                self.trade_state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                
                logger.info(
                    f"💾 Saved {trade.symbol} to trade_state.json:\n"
                    f"   Trail: {trail_pct*100:.2f}% | TP: {tp_pct*100:.2f}% | SL: {sl_pct*100:.2f}%"
                )
        except Exception as e:
            logger.error(f"Failed to save trade state for {trade.symbol}: {e}")
    
    def _remove_trade_from_state(self, symbol: str) -> None:
        """Remove closed trade from state file."""
        try:
            # Load current state
            if not self.trade_state_path.exists():
                return
            
            state = json.loads(self.trade_state_path.read_text(encoding="utf-8"))
            
            # Remove symbol if exists
            if symbol in state:
                del state[symbol]
                
                # Save updated state
                self.trade_state_path.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                
                logger.info(f"🗑️ Removed {symbol} from trade_state.json")
        except Exception as e:
            logger.error(f"Failed to remove {symbol} from trade state: {e}")
    
    def evaluate_new_signal(
        self,
        symbol: str,
        action: str,
        signal_quality: SignalQuality,
        market_conditions: MarketConditions,
        current_equity: float,
    ) -> TradeDecision:
        """
        Evaluate a new trading signal through complete risk management pipeline.
        
        Args:
            symbol: Trading pair
            action: "LONG" or "SHORT"
            signal_quality: AI consensus and confidence
            market_conditions: Current market data (price, ATR, EMA, volume)
            current_equity: Current account equity
        
        Returns:
            TradeDecision with approval and execution details
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[CLIPBOARD] Evaluating NEW signal: {symbol} {action}")
        logger.info(f"{'='*60}")
        
        # Step 1: Trade Opportunity Filter
        filter_result = self.trade_filter.evaluate_signal(
            symbol=symbol,
            signal_quality=signal_quality,
            market_conditions=market_conditions,
            action=action,
        )
        
        if not filter_result.passed:
            logger.warning(f"❌ Trade REJECTED by filter: {filter_result.rejection_reason}")
            
            if self.config.logging.log_filter_rejections:
                self._log_rejection(symbol, action, signal_quality, market_conditions, filter_result.rejection_reason)
            
            return TradeDecision(
                approved=False,
                symbol=symbol,
                action=action,
                filter_result=filter_result,
                rejection_reason=f"Filter: {filter_result.rejection_reason}",
            )
        
        # Step 2: Position Sizing
        position_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            current_price=market_conditions.price,
            atr=market_conditions.atr,
            equity_usd=current_equity,
            signal_confidence=signal_quality.confidence,
            action=action,
        )
        
        # Validate position size
        is_valid, validation_error = self.risk_manager.validate_position_size(
            position_size=position_size,
            equity_usd=current_equity,
        )
        
        if not is_valid:
            logger.warning(f"❌ Trade REJECTED by position sizing: {validation_error}")
            return TradeDecision(
                approved=False,
                symbol=symbol,
                action=action,
                filter_result=filter_result,
                position_size=position_size,
                rejection_reason=f"Position sizing: {validation_error}",
            )
        
        # Step 3: Global Risk Check
        open_positions = self._get_open_positions_info()
        risk_check = self.global_risk.check_new_trade(
            symbol=symbol,
            action=action,
            proposed_notional_usd=position_size.notional_usd,
            current_equity=current_equity,
            open_positions=open_positions,
        )
        
        if not risk_check.approved:
            logger.warning(f"❌ Trade REJECTED by global risk: {risk_check.rejection_reason}")
            return TradeDecision(
                approved=False,
                symbol=symbol,
                action=action,
                filter_result=filter_result,
                position_size=position_size,
                risk_check=risk_check,
                rejection_reason=f"Global risk: {risk_check.rejection_reason}",
            )
        
        # Apply risk multiplier if in recovery/streak mode
        if risk_check.risk_multiplier != 1.0:
            logger.info(f"[WARNING]  Adjusting position size by {risk_check.risk_multiplier:.0%}")
            position_size.quantity *= risk_check.risk_multiplier
            position_size.notional_usd *= risk_check.risk_multiplier
            position_size.risk_usd *= risk_check.risk_multiplier
        
        # Step 4: Calculate Exit Levels
        # [TARGET] ORCHESTRATOR: Pass exit mode from policy if available
        exit_mode = None
        regime_tag = None
        if self.current_policy and hasattr(self.current_policy, 'exit_mode'):
            exit_mode = self.current_policy.exit_mode
            if hasattr(self.current_policy, 'note'):
                # Extract regime from note like "BULL market, normal vol"
                note_parts = self.current_policy.note.split(',')
                if note_parts:
                    regime_tag = note_parts[0].strip().split()[0].upper()  # "BULL" from "BULL market"
        
        exit_levels = self.exit_engine.calculate_initial_exit_levels(
            symbol=symbol,
            entry_price=market_conditions.price,
            atr=market_conditions.atr,
            action=action,
            exit_mode=exit_mode,
            regime_tag=regime_tag,
        )
        
        # ALL CHECKS PASSED
        logger.info(
            f"\n{'='*60}\n"
            f"[OK][OK][OK] TRADE APPROVED: {symbol} {action} [OK][OK][OK]\n"
            f"   Quantity: {position_size.quantity:.4f} (${position_size.notional_usd:.2f})\n"
            f"   Entry: ${market_conditions.price:.4f}\n"
            f"   SL: ${exit_levels.stop_loss:.4f} (-{exit_levels.sl_distance_pct:.2%})\n"
            f"   TP: ${exit_levels.take_profit:.4f} (+{exit_levels.tp_distance_pct:.2%})\n"
            f"   Risk: ${position_size.risk_usd:.2f} ({position_size.risk_pct:.2%})\n"
            f"   R:R = {exit_levels.r_multiple:.2f}\n"
            f"{'='*60}\n"
        )
        
        if self.config.logging.log_trade_decisions:
            self._log_approval(symbol, action, signal_quality, market_conditions, position_size, exit_levels)
        
        return TradeDecision(
            approved=True,
            symbol=symbol,
            action=action,
            quantity=position_size.quantity,
            entry_price=market_conditions.price,
            stop_loss=exit_levels.stop_loss,
            take_profit=exit_levels.take_profit,
            filter_result=filter_result,
            position_size=position_size,
            risk_check=risk_check,
        )
    
    def open_trade(
        self,
        trade_id: str,
        decision: TradeDecision,
        signal_quality: SignalQuality,
        market_conditions: MarketConditions,
        actual_entry_price: float,
    ) -> ManagedTrade:
        """
        Record a trade as opened and begin monitoring.
        
        Args:
            trade_id: Unique trade identifier
            decision: Approved trade decision
            signal_quality: Original signal quality
            market_conditions: Market conditions at entry
            actual_entry_price: Actual fill price
        
        Returns:
            ManagedTrade in OPEN state
        """
        # Recalculate exit levels with actual entry price
        # [TARGET] ORCHESTRATOR: Pass exit mode from policy if available
        exit_mode = None
        regime_tag = None
        if self.current_policy and hasattr(self.current_policy, 'exit_mode'):
            exit_mode = self.current_policy.exit_mode
            if hasattr(self.current_policy, 'note'):
                note_parts = self.current_policy.note.split(',')
                if note_parts:
                    regime_tag = note_parts[0].strip().split()[0].upper()
        
        exit_levels = self.exit_engine.calculate_initial_exit_levels(
            symbol=decision.symbol,
            entry_price=actual_entry_price,
            atr=market_conditions.atr,
            action=decision.action,
            exit_mode=exit_mode,
            regime_tag=regime_tag,
        )
        
        trade = ManagedTrade(
            trade_id=trade_id,
            symbol=decision.symbol,
            action=decision.action,
            state=TradeState.OPEN,
            signal_quality=signal_quality,
            market_conditions=market_conditions,
            entry_price=actual_entry_price,
            entry_time=datetime.now(timezone.utc),
            position_size=decision.position_size,
            current_quantity=decision.quantity,
            exit_levels=exit_levels,
            highest_price=actual_entry_price,
            lowest_price=actual_entry_price,
        )
        
        self.active_trades[trade_id] = trade
        
        logger.info(
            f"[ROCKET] Trade OPENED: {trade_id}\n"
            f"   {decision.symbol} {decision.action}\n"
            f"   Entry: ${actual_entry_price:.4f}\n"
            f"   Quantity: {decision.quantity:.4f}\n"
            f"   SL: ${exit_levels.stop_loss:.4f}\n"
            f"   TP: ${exit_levels.take_profit:.4f}"
        )
        
        # [FIX] Save to trade_state.json for Trailing Stop Manager
        self._save_trade_to_state(trade)
        
        return trade
    
    def update_trade(
        self,
        trade_id: str,
        current_price: float,
    ) -> Optional[ExitDecision]:
        """
        Update trade with current price and evaluate exit.
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
        
        Returns:
            ExitDecision if action needed, None otherwise
        """
        if trade_id not in self.active_trades:
            logger.error(f"❌ Trade {trade_id} not found in active trades")
            return None
        
        trade = self.active_trades[trade_id]
        
        # Update MFE/MAE
        if trade.action == "LONG":
            trade.highest_price = max(trade.highest_price, current_price)
            trade.lowest_price = min(trade.lowest_price, current_price)
        else:  # SHORT
            trade.lowest_price = min(trade.lowest_price, current_price)
            trade.highest_price = max(trade.highest_price, current_price)
        
        # Evaluate exit
        exit_decision = self.exit_engine.evaluate_exit(
            symbol=trade.symbol,
            action=trade.action,
            current_price=current_price,
            exit_levels=trade.exit_levels,
            quantity=trade.current_quantity,
            entry_time=trade.entry_time,
            highest_price=trade.highest_price,
            lowest_price=trade.lowest_price,
            has_partial_exit=trade.has_partial_exit,
        )
        
        # Handle exit decision
        if exit_decision.move_to_breakeven and exit_decision.new_sl:
            trade.exit_levels.current_sl = exit_decision.new_sl
            trade.state = TradeState.BREAKEVEN
            logger.info(f"🔒 {trade_id} moved to BREAKEVEN")
        
        elif exit_decision.new_sl and not exit_decision.should_exit:
            # Trailing stop update
            trade.exit_levels.current_sl = exit_decision.new_sl
            if trade.state != TradeState.TRAILING:
                trade.state = TradeState.TRAILING
                logger.info(f"[CHART_UP] {trade_id} now TRAILING")
        
        return exit_decision if exit_decision.should_exit else None
    
    def close_trade(
        self,
        trade_id: str,
        exit_decision: ExitDecision,
        actual_exit_price: float,
    ):
        """
        Close a trade and record final results.
        
        Args:
            trade_id: Trade identifier
            exit_decision: Exit decision with reason
            actual_exit_price: Actual fill price on exit
        """
        if trade_id not in self.active_trades:
            logger.error(f"❌ Trade {trade_id} not found")
            return
        
        trade = self.active_trades[trade_id]
        
        # Handle partial exit
        if exit_decision.exit_quantity and exit_decision.exit_quantity < trade.current_quantity:
            # Partial close
            closed_qty = exit_decision.exit_quantity
            trade.current_quantity -= closed_qty
            trade.has_partial_exit = True
            trade.state = TradeState.PARTIAL_TP
            
            # Calculate partial PnL
            if trade.action == "LONG":
                partial_pnl = (actual_exit_price - trade.entry_price) * closed_qty
            else:
                partial_pnl = (trade.entry_price - actual_exit_price) * closed_qty
            
            trade.realized_pnl_usd += partial_pnl
            
            logger.info(
                f"[MONEY] {trade_id} PARTIAL CLOSE:\n"
                f"   Closed {closed_qty:.4f} @ ${actual_exit_price:.4f}\n"
                f"   Partial PnL: ${partial_pnl:.2f}\n"
                f"   Remaining: {trade.current_quantity:.4f}"
            )
            
            return  # Keep trade open with remaining quantity
        
        # Full close
        exit_qty = trade.current_quantity
        
        if trade.action == "LONG":
            final_pnl = (actual_exit_price - trade.entry_price) * exit_qty
        else:
            final_pnl = (trade.entry_price - actual_exit_price) * exit_qty
        
        trade.realized_pnl_usd += final_pnl
        trade.realized_pnl_pct = trade.realized_pnl_usd / (trade.entry_price * trade.position_size.quantity)
        
        # Calculate final R-multiple
        risk_amount = abs(trade.entry_price - trade.exit_levels.stop_loss) * trade.position_size.quantity
        trade.final_r_multiple = trade.realized_pnl_usd / risk_amount if risk_amount > 0 else 0
        
        trade.exit_price = actual_exit_price
        trade.exit_time = datetime.now(timezone.utc)
        trade.exit_reason = exit_decision.reason
        
        # Update state based on exit signal
        if exit_decision.exit_signal == ExitSignal.STOP_LOSS:
            trade.state = TradeState.CLOSED_SL
        elif exit_decision.exit_signal == ExitSignal.TAKE_PROFIT:
            trade.state = TradeState.CLOSED_TP
        elif exit_decision.exit_signal == ExitSignal.TIME_EXIT:
            trade.state = TradeState.CLOSED_TIME
        else:
            trade.state = TradeState.CLOSED_PARTIAL
        
        # Record in global risk controller
        trade_record = TradeRecord(
            symbol=trade.symbol,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            pnl_usd=trade.realized_pnl_usd,
            pnl_pct=trade.realized_pnl_pct,
            r_multiple=trade.final_r_multiple,
            action=trade.action,
        )
        self.global_risk.record_trade(trade_record)
        
        # Record for continuous learning (AI feedback loop)
        if self.ai_engine:
            try:
                outcome = 1.0 if trade.realized_pnl_usd > 0 else 0.0
                
                # Reconstruct features from trade's signal quality and market conditions
                features = {
                    'confidence': trade.signal_quality.confidence,
                    'consensus_count': trade.signal_quality.consensus_count,
                    'price': trade.entry_price,
                    'atr': trade.market_conditions.atr,
                    'rsi': trade.market_conditions.rsi,
                    'volatility': trade.market_conditions.volatility,
                    'volume_ratio': trade.market_conditions.volume_ratio,
                }
                
                # Call continuous learning manager with correct signature
                cl_result = self.ai_engine.agent.record_trade_outcome_for_cl(
                    symbol=trade.symbol,
                    outcome=outcome,
                    features=features,
                    pnl=trade.realized_pnl_usd
                )
                
                if cl_result.get('status') == 'success':
                    logger.info(f"📊 [CL] Trade outcome recorded for {trade.symbol}")
                    
                    if cl_result.get('retraining_triggered'):
                        logger.warning(
                            f"🔄 [CONTINUOUS LEARNING] Retraining triggered!\n"
                            f"   Symbol: {trade.symbol}\n"
                            f"   Trigger: {cl_result.get('trigger_type')}\n"
                            f"   Urgency: {cl_result.get('urgency_score', 0):.2f}"
                        )
            except Exception as e:
                logger.debug(f"[CL] Failed to record trade outcome: {e}")
        
        # Remove closed trade from active_trades to prevent memory leak
        del self.active_trades[trade_id]
        logger.info(f"[CLEANUP] Removed closed trade {trade_id} from active_trades (total: {len(self.active_trades)})")
        
        # [FIX] Remove from trade_state.json
        self._remove_trade_from_state(trade.symbol)
        
        outcome_emoji = "🎉" if trade.realized_pnl_usd > 0 else "💔"
        logger.info(
            f"{outcome_emoji} Trade CLOSED: {trade_id}\n"
            f"   {trade.symbol} {trade.action}\n"
            f"   Entry: ${trade.entry_price:.4f} @ {trade.entry_time.strftime('%H:%M:%S')}\n"
            f"   Exit: ${actual_exit_price:.4f} @ {trade.exit_time.strftime('%H:%M:%S')}\n"
            f"   PnL: ${trade.realized_pnl_usd:.2f} ({trade.realized_pnl_pct:.2%})\n"
            f"   R-multiple: {trade.final_r_multiple:.2f}R\n"
            f"   Reason: {exit_decision.reason}"
        )
        
        if self.config.logging.log_exit_decisions:
            self._log_trade_close(trade)
    
    def _get_open_positions_info(self) -> list[PositionInfo]:
        """Get list of open positions for global risk checks."""
        # States that count as "open" positions
        open_states = {
            TradeState.OPEN,
            TradeState.PARTIAL_TP,
            TradeState.BREAKEVEN,
            TradeState.TRAILING
        }
        
        # DEBUG: Log active_trades state
        total_trades = len(self.active_trades)
        state_counts = {}
        for trade in self.active_trades.values():
            state = trade.state.value if hasattr(trade.state, 'value') else str(trade.state)
            state_counts[state] = state_counts.get(state, 0) + 1
        
        logger.info(f"[DEBUG] active_trades: total={total_trades}, states={state_counts}")
        
        positions = []
        for trade in self.active_trades.values():
            # Only count trades that are actively open with remaining quantity
            if trade.state in open_states and trade.current_quantity and trade.current_quantity > 0:
                positions.append(
                    PositionInfo(
                        symbol=trade.symbol,
                        action=trade.action,
                        entry_price=trade.entry_price,
                        quantity=trade.current_quantity,
                        notional_usd=trade.position_size.notional_usd,
                        unrealized_pnl_usd=0.0,  # Would need current price to calculate
                    )
                )
        
        logger.info(f"[DEBUG] _get_open_positions_info returning {len(positions)} positions from {total_trades} active_trades")
        return positions
    
    def _log_rejection(self, symbol: str, action: str, signal_quality: SignalQuality, market_conditions: MarketConditions, reason: str):
        """Log trade rejection for auto-training."""
        logger.info(
            f"[MEMO] REJECTION LOG:\n"
            f"   Symbol: {symbol} {action}\n"
            f"   Reason: {reason}\n"
            f"   Consensus: {signal_quality.consensus_type.value}\n"
            f"   Confidence: {signal_quality.confidence:.1%}\n"
            f"   Price: ${market_conditions.price:.4f}\n"
            f"   ATR: ${market_conditions.atr:.4f} ({market_conditions.atr/market_conditions.price:.2%})\n"
            f"   EMA200: ${market_conditions.ema_200:.4f}"
        )
    
    def _log_approval(self, symbol: str, action: str, signal_quality: SignalQuality, market_conditions: MarketConditions, position_size: PositionSize, exit_levels: ExitLevels):
        """Log trade approval for auto-training."""
        logger.info(
            f"[MEMO] APPROVAL LOG:\n"
            f"   Symbol: {symbol} {action}\n"
            f"   Consensus: {signal_quality.consensus_type.value}\n"
            f"   Confidence: {signal_quality.confidence:.1%}\n"
            f"   Size: {position_size.quantity:.4f} (${position_size.notional_usd:.2f})\n"
            f"   Risk: ${position_size.risk_usd:.2f} ({position_size.risk_pct:.2%})\n"
            f"   Entry: ${market_conditions.price:.4f}\n"
            f"   SL: ${exit_levels.stop_loss:.4f}\n"
            f"   TP: ${exit_levels.take_profit:.4f}\n"
            f"   R:R: {exit_levels.r_multiple:.2f}"
        )
    
    def _log_trade_close(self, trade: ManagedTrade):
        """Log trade close for auto-training and database."""
        logger.info(
            f"[MEMO] CLOSE LOG:\n"
            f"   ID: {trade.trade_id}\n"
            f"   Symbol: {trade.symbol} {trade.action}\n"
            f"   Duration: {(trade.exit_time - trade.entry_time).total_seconds() / 3600:.1f}h\n"
            f"   PnL: ${trade.realized_pnl_usd:.2f} ({trade.realized_pnl_pct:.2%})\n"
            f"   R-multiple: {trade.final_r_multiple:.2f}R\n"
            f"   MFE: ${trade.highest_price - trade.entry_price:.4f}\n"
            f"   MAE: ${trade.entry_price - trade.lowest_price:.4f}\n"
            f"   Exit Reason: {trade.exit_reason}"
        )
        
        # Save to database for Analytics
        try:
            from backend.database import get_db, TradeLog
            db = next(get_db())
            try:
                trade_log = TradeLog(
                    symbol=trade.symbol,
                    side=trade.action.upper(),
                    qty=trade.position_size,
                    price=trade.exit_price or trade.entry_price,
                    status="CLOSED",
                    reason=trade.exit_reason or "UNKNOWN",
                    timestamp=trade.exit_time,
                    realized_pnl=trade.realized_pnl_usd,
                    realized_pnl_pct=trade.realized_pnl_pct,
                    equity_after=0.0,  # TODO: Add equity tracking
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price or trade.entry_price,
                    strategy_id=trade.strategy_id or "default"
                )
                db.add(trade_log)
                db.commit()
                logger.info(f"[DB] Saved trade close to database: {trade.symbol}")
            except Exception as e:
                logger.error(f"[DB] Failed to save trade close: {e}")
                db.rollback()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"[DB] Failed to import/connect database: {e}")
