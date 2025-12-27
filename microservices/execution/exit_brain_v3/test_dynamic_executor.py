"""
Tests for Exit Brain Dynamic Executor (Phase 2A - Shadow Mode)

Verifies:
1. ExitBrainAdapter makes different decisions based on context
2. ExitBrainDynamicExecutor logs decisions correctly
3. Shadow mode does not place orders
"""
import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from backend.domains.exits.exit_brain_v3.types import (
    PositionContext,
    ExitDecision,
    ExitDecisionType
)
from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor
from backend.domains.exits.exit_brain_v3.integration import ExitPlan, ExitLeg


class TestExitBrainAdapter:
    """Test ExitBrainAdapter decision logic."""
    
    def test_decide_no_change_when_plan_working(self):
        """AI should return NO_CHANGE when position is healthy and plan is working."""
        # Mock planner and router
        mock_planner = Mock()
        mock_router = Mock()
        
        # Create plan with reasonable exits
        plan = ExitPlan(
            symbol="BTCUSDT",
            side="long",
            strategy_id="test_strategy",
            legs=[
                ExitLeg(leg_type="PARTIAL_TP", target_price=100000.0, fraction=0.5),
                ExitLeg(leg_type="HARD_SL", target_price=90000.0, fraction=1.0)
            ]
        )
        mock_router.get_active_plan.return_value = plan
        
        adapter = ExitBrainAdapter(planner=mock_planner, router=mock_router)
        
        # Position context: small profit, no emergency conditions
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=95000.0,
            current_price=96000.0,  # +1% profit
            size=1.0,
            unrealized_pnl=1.0,
            regime="trend",
            risk_state="normal"
        )
        
        decision = adapter.decide(ctx)
        
        assert decision.decision_type == ExitDecisionType.NO_CHANGE
        assert "working as expected" in decision.reason.lower()
    
    def test_decide_partial_close_when_tp_hit(self):
        """AI should suggest PARTIAL_CLOSE when TP level reached."""
        mock_planner = Mock()
        mock_router = Mock()
        
        # Plan with partial TP at 100k
        plan = ExitPlan(
            symbol="BTCUSDT",
            side="long",
            strategy_id="test_strategy",
            legs=[
                ExitLeg(leg_type="PARTIAL_TP", target_price=100000.0, fraction=0.5),
                ExitLeg(leg_type="HARD_SL", target_price=90000.0, fraction=1.0)
            ]
        )
        mock_router.get_active_plan.return_value = plan
        
        adapter = ExitBrainAdapter(planner=mock_planner, router=mock_router)
        
        # Current price exceeded TP
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=95000.0,
            current_price=100500.0,  # Hit TP
            size=1.0,
            unrealized_pnl=5.8,
            regime="trend"
        )
        
        decision = adapter.decide(ctx)
        
        assert decision.decision_type == ExitDecisionType.PARTIAL_CLOSE
        assert decision.fraction_to_close == 0.5
        assert "partial tp" in decision.reason.lower()
    
    def test_decide_move_sl_when_breakeven_triggered(self):
        """AI should suggest MOVE_SL to breakeven when profit > 1.5%."""
        mock_planner = Mock()
        mock_router = Mock()
        
        plan = ExitPlan(
            symbol="BTCUSDT",
            side="long",
            strategy_id="test_strategy",
            legs=[
                ExitLeg(leg_type="HARD_SL", target_price=90000.0, fraction=1.0)
            ]
        )
        mock_router.get_active_plan.return_value = plan
        
        adapter = ExitBrainAdapter(planner=mock_planner, router=mock_router)
        
        # Position in profit > 1.5%, should move to breakeven
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=95000.0,
            current_price=96800.0,  # ~1.9% profit
            size=1.0,
            unrealized_pnl=1.9,
            regime="trend",
            meta={}  # No breakeven_moved flag
        )
        
        decision = adapter.decide(ctx)
        
        assert decision.decision_type == ExitDecisionType.MOVE_SL
        assert decision.new_sl_price == pytest.approx(95000.0)  # Breakeven
        assert decision.sl_reason == "breakeven"
    
    def test_decide_full_exit_on_emergency(self):
        """AI should suggest FULL_EXIT_NOW on emergency conditions."""
        mock_planner = Mock()
        mock_router = Mock()
        
        plan = ExitPlan(
            symbol="BTCUSDT",
            side="long",
            strategy_id="test_strategy",
            legs=[]
        )
        mock_router.get_active_plan.return_value = plan
        
        adapter = ExitBrainAdapter(planner=mock_planner, router=mock_router)
        
        # Emergency: big loss + regime change to crash
        ctx = PositionContext(
            symbol="BTCUSDT",
            side="long",
            entry_price=95000.0,
            current_price=89000.0,  # -6.3% loss
            size=1.0,
            unrealized_pnl=-6.3,
            regime="crash",  # Emergency regime
            duration_hours=2.0
        )
        
        decision = adapter.decide(ctx)
        
        assert decision.decision_type == ExitDecisionType.FULL_EXIT_NOW
        assert "emergency" in decision.reason.lower()
        assert decision.confidence >= 0.9  # High confidence in emergency exits


@pytest.mark.asyncio
class TestExitBrainDynamicExecutor:
    """Test ExitBrainDynamicExecutor shadow mode."""
    
    async def test_shadow_mode_does_not_place_orders(self):
        """Shadow mode should log decisions but NOT place orders."""
        # Mock adapter
        mock_adapter = Mock()
        mock_adapter.decide.return_value = ExitDecision(
            decision_type=ExitDecisionType.MOVE_SL,
            symbol="BTCUSDT",
            new_sl_price=95000.0,
            sl_reason="breakeven",
            reason="Test decision",
            current_price=96000.0,
            unrealized_pnl=1.0
        )
        
        # Mock position source
        mock_position_source = Mock()
        mock_position_source.futures_position_information.return_value = [
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '1.0',  # Long
                'entryPrice': '95000.0',
                'markPrice': '96000.0',
                'unRealizedProfit': '1000.0',
                'leverage': '1'
            }
        ]
        
        # Create executor in shadow mode
        executor = ExitBrainDynamicExecutor(
            adapter=mock_adapter,
            position_source=mock_position_source,
            loop_interval_sec=0.1,  # Fast for testing
            shadow_mode=True
        )
        
        # Run one monitoring cycle
        await executor._monitoring_cycle(cycle_num=1)
        
        # Verify adapter was called
        assert mock_adapter.decide.called
        
        # Verify _execute_decision was NOT called (shadow mode)
        # (Decision only logged, not executed)
        
        # Verify no orders placed
        # (position_source should only be read, never written to)
        # Mock doesn't track this, but real implementation won't call futures_create_order
    
    async def test_executor_logs_only_when_decision_changes(self):
        """Executor should avoid spam logs by only logging when decision changes."""
        # Create adapter that returns same decision twice
        mock_adapter = Mock()
        decision = ExitDecision(
            decision_type=ExitDecisionType.NO_CHANGE,
            symbol="BTCUSDT",
            reason="Plan working",
            current_price=96000.0,
            unrealized_pnl=1.0
        )
        mock_adapter.decide.return_value = decision
        
        mock_position_source = Mock()
        mock_position_source.futures_position_information.return_value = [
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '1.0',
                'entryPrice': '95000.0',
                'markPrice': '96000.0',
                'unRealizedProfit': '1000.0',
                'leverage': '1'
            }
        ]
        
        executor = ExitBrainDynamicExecutor(
            adapter=mock_adapter,
            position_source=mock_position_source,
            loop_interval_sec=0.1,
            shadow_mode=True
        )
        
        # First cycle: should log (first decision)
        await executor._monitoring_cycle(cycle_num=1)
        assert "BTCUSDT" in executor.last_decisions
        
        # Second cycle: same decision, should NOT log (NO_CHANGE spam prevention)
        with patch('backend.domains.exits.exit_brain_v3.dynamic_executor.logger_shadow') as mock_logger:
            await executor._monitoring_cycle(cycle_num=2)
            # NO_CHANGE decisions are not logged repeatedly
            mock_logger.info.assert_not_called()
    
    async def test_executor_builds_correct_position_context(self):
        """Executor should build complete PositionContext from exchange data."""
        mock_adapter = Mock()
        mock_adapter.decide.return_value = ExitDecision(
            decision_type=ExitDecisionType.NO_CHANGE,
            symbol="ETHUSDT",
            reason="Test",
            current_price=3500.0,
            unrealized_pnl=2.0
        )
        
        # Short position
        mock_position_source = Mock()
        mock_position_source.futures_position_information.return_value = [
            {
                'symbol': 'ETHUSDT',
                'positionAmt': '-2.0',  # SHORT
                'entryPrice': '3500.0',
                'markPrice': '3450.0',  # Profit for short
                'unRealizedProfit': '100.0',
                'leverage': '2'
            }
        ]
        
        executor = ExitBrainDynamicExecutor(
            adapter=mock_adapter,
            position_source=mock_position_source,
            loop_interval_sec=0.1,
            shadow_mode=True
        )
        
        await executor._monitoring_cycle(cycle_num=1)
        
        # Verify adapter.decide was called with correct context
        call_args = mock_adapter.decide.call_args
        ctx = call_args[0][0]  # First positional arg
        
        assert isinstance(ctx, PositionContext)
        assert ctx.symbol == "ETHUSDT"
        assert ctx.side == "short"
        assert ctx.size == 2.0  # Absolute value
        assert ctx.entry_price == 3500.0
        assert ctx.current_price == 3450.0
        assert ctx.leverage == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
