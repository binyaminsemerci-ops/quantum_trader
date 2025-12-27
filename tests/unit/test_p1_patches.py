"""
Unit Tests for P1 Patches
==========================

Tests for all P1 patches to ensure production readiness.

P1 patches:
1. Event schemas with Pydantic validation
2. Signal quality filter (HIGH_VOL noise reduction)
3. Consolidated RiskManager (DEFERRED)
4. AI-HFOS instant regime reaction
5. ESS auto-recovery mode
"""

import pytest
import asyncio
from datetime import datetime
from typing import List


# ==============================================================================
# P1-01: Event Schemas Tests
# ==============================================================================

class TestEventSchemas:
    """Unit tests for Pydantic event schemas (P1-01)."""
    
    def test_market_regime_changed_event_valid(self):
        """Test MarketRegimeChangedEvent with valid data."""
        from backend.events.schemas import MarketRegimeChangedEvent
        
        event = MarketRegimeChangedEvent(
            symbol="BTCUSDT",
            old_regime="CONSOLIDATION",
            new_regime="TRENDING",
            regime_confidence=0.87
        )
        
        assert event.symbol == "BTCUSDT"
        assert event.old_regime == "CONSOLIDATION"
        assert event.new_regime == "TRENDING"
        assert event.regime_confidence == 0.87
    
    def test_market_regime_changed_event_invalid_confidence(self):
        """Test MarketRegimeChangedEvent rejects invalid confidence."""
        from backend.events.schemas import MarketRegimeChangedEvent
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            MarketRegimeChangedEvent(
                symbol="BTCUSDT",
                old_regime="CONSOLIDATION",
                new_regime="TRENDING",
                regime_confidence=1.5  # Invalid: >1.0
            )
    
    def test_system_emergency_triggered_event(self):
        """Test SystemEmergencyTriggeredEvent validation."""
        from backend.events.schemas import SystemEmergencyTriggeredEvent
        
        event = SystemEmergencyTriggeredEvent(
            severity="CRITICAL",
            trigger_reason="Drawdown exceeded -10%",
            trigger_source="DrawdownEvaluator",
            positions_closed=3,
            trading_blocked=True
        )
        
        assert event.severity == "CRITICAL"
        assert event.positions_closed == 3
        assert event.trading_blocked is True
    
    def test_rl_reward_received_event(self):
        """Test RLRewardReceivedEvent validation."""
        from backend.events.schemas import RLRewardReceivedEvent
        
        event = RLRewardReceivedEvent(
            reward=150.50,
            strategy_id="long_momentum_v1",
            final_pnl=150.50,
            hold_duration_sec=3600,
            outcome="WIN"
        )
        
        assert event.reward == 150.50
        assert event.outcome == "WIN"


# ==============================================================================
# P1-02: Signal Quality Filter Tests
# ==============================================================================

class TestSignalQualityFilter:
    """Unit tests for SignalQualityFilter (P1-02)."""
    
    @pytest.fixture
    def signal_filter(self):
        from backend.services.risk.signal_quality_filter import SignalQualityFilter
        return SignalQualityFilter(
            min_model_agreement=0.75,
            min_confidence_normal=0.45,
            min_confidence_high_vol=0.65
        )
    
    def test_signal_rejected_low_agreement(self, signal_filter):
        """Test signal rejected when model agreement < 75%."""
        from backend.services.risk.signal_quality_filter import ModelPrediction
        
        predictions = [
            ModelPrediction("model1", "BUY", 0.80),
            ModelPrediction("model2", "BUY", 0.70),
            ModelPrediction("model3", "SELL", 0.60),
            ModelPrediction("model4", "HOLD", 0.50)
        ]
        
        result = signal_filter.filter_signal("BTCUSDT", predictions, atr_pct=0.02)
        
        assert result.passed is False
        assert "agreement" in result.reason.lower()
    
    def test_signal_passed_high_agreement(self, signal_filter):
        """Test signal passes with 3/4 model agreement."""
        from backend.services.risk.signal_quality_filter import ModelPrediction
        
        predictions = [
            ModelPrediction("model1", "BUY", 0.80),
            ModelPrediction("model2", "BUY", 0.75),
            ModelPrediction("model3", "BUY", 0.70),
            ModelPrediction("model4", "HOLD", 0.50)
        ]
        
        result = signal_filter.filter_signal("BTCUSDT", predictions, atr_pct=0.02)
        
        assert result.passed is True
        assert result.recommended_action == "EXECUTE"
    
    def test_signal_rejected_high_vol_low_confidence(self, signal_filter):
        """Test signal rejected in HIGH_VOL with confidence < 65%."""
        from backend.services.risk.signal_quality_filter import ModelPrediction
        
        predictions = [
            ModelPrediction("model1", "BUY", 0.60),
            ModelPrediction("model2", "BUY", 0.62),
            ModelPrediction("model3", "BUY", 0.58),
            ModelPrediction("model4", "HOLD", 0.50)
        ]
        
        # HIGH_VOL market (4% ATR)
        result = signal_filter.filter_signal("BTCUSDT", predictions, atr_pct=0.04)
        
        assert result.passed is False
        assert "confidence" in result.reason.lower()
    
    def test_signal_passed_high_vol_high_confidence(self, signal_filter):
        """Test signal passes in HIGH_VOL with confidence >= 65%."""
        from backend.services.risk.signal_quality_filter import ModelPrediction
        
        predictions = [
            ModelPrediction("model1", "BUY", 0.75),
            ModelPrediction("model2", "BUY", 0.72),
            ModelPrediction("model3", "BUY", 0.68),
            ModelPrediction("model4", "HOLD", 0.50)
        ]
        
        # HIGH_VOL market (4% ATR)
        result = signal_filter.filter_signal("BTCUSDT", predictions, atr_pct=0.04)
        
        assert result.passed is True
        assert result.recommended_action == "EXECUTE"


# ==============================================================================
# P1-04: AI-HFOS Instant Regime Reaction Tests
# ==============================================================================

class TestAIHFOSRegimeReaction:
    """Unit tests for AI-HFOS instant regime reaction (P1-04)."""
    
    @pytest.mark.asyncio
    async def test_regime_change_event_subscription(self):
        """Test AI-HFOS subscribes to regime change events."""
        # Mock EventBus
        class MockEventBus:
            def __init__(self):
                self.subscriptions = {}
            
            def subscribe(self, event_type, handler):
                self.subscriptions[event_type] = handler
        
        # Create AI-HFOS with event bus
        from backend.services.ai.ai_hedgefund_os import AIHedgeFundOS
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            event_bus = MockEventBus()
            
            ai_hfos = AIHedgeFundOS(
                data_dir=Path(tmpdir),
                config_path=None,
                event_bus=event_bus
            )
            
            # Verify subscription
            assert "market.regime.changed" in event_bus.subscriptions
    
    @pytest.mark.asyncio
    async def test_regime_change_triggers_coordination(self):
        """Test regime change triggers immediate coordination."""
        # TODO: Full integration test
        pass


# ==============================================================================
# P1-05: ESS Auto-Recovery Tests
# ==============================================================================

class TestESSAutoRecovery:
    """Unit tests for ESS auto-recovery (P1-05)."""
    
    @pytest.fixture
    def ess_controller(self):
        from backend.services.risk.emergency_stop_system import (
            EmergencyStopController,
            RecoveryMode
        )
        
        # Mock dependencies
        class MockPolicyStore:
            def get(self, key): return {}
            def set(self, key, value): pass
        
        class MockExchange:
            async def close_all_positions(self): return 0
            async def cancel_all_orders(self): return 0
        
        class MockEventBus:
            def __init__(self):
                self.published_events = []
            async def publish(self, event):
                self.published_events.append(event)
        
        event_bus = MockEventBus()
        controller = EmergencyStopController(
            policy_store=MockPolicyStore(),
            exchange=MockExchange(),
            event_bus=event_bus
        )
        controller.event_bus = event_bus  # Store for assertions
        return controller
    
    @pytest.mark.asyncio
    async def test_recovery_emergency_to_protective(self, ess_controller):
        """Test transition from EMERGENCY to PROTECTIVE mode."""
        from backend.services.risk.emergency_stop_system import RecoveryMode
        
        # Activate ESS (EMERGENCY mode)
        await ess_controller.activate("Test drawdown")
        assert ess_controller.state.recovery_mode == RecoveryMode.EMERGENCY
        
        # Simulate DD improving to -8%
        await ess_controller.check_recovery(-8.0)
        
        assert ess_controller.state.recovery_mode == RecoveryMode.PROTECTIVE
        assert ess_controller.is_active is False  # Trading unlocked
    
    @pytest.mark.asyncio
    async def test_recovery_protective_to_cautious(self, ess_controller):
        """Test transition from PROTECTIVE to CAUTIOUS mode."""
        from backend.services.risk.emergency_stop_system import RecoveryMode
        
        # Start in PROTECTIVE mode
        await ess_controller.activate("Test")
        await ess_controller.check_recovery(-8.0)
        assert ess_controller.state.recovery_mode == RecoveryMode.PROTECTIVE
        
        # DD improves to -3%
        await ess_controller.check_recovery(-3.0)
        
        assert ess_controller.state.recovery_mode == RecoveryMode.CAUTIOUS
    
    @pytest.mark.asyncio
    async def test_recovery_cautious_to_normal(self, ess_controller):
        """Test transition from CAUTIOUS to NORMAL mode."""
        from backend.services.risk.emergency_stop_system import RecoveryMode
        
        # Start in CAUTIOUS mode
        await ess_controller.activate("Test")
        await ess_controller.check_recovery(-3.0)
        assert ess_controller.state.recovery_mode == RecoveryMode.CAUTIOUS
        
        # DD improves to -1%
        await ess_controller.check_recovery(-1.0)
        
        assert ess_controller.state.recovery_mode == RecoveryMode.NORMAL
    
    @pytest.mark.asyncio
    async def test_recovery_publishes_events(self, ess_controller):
        """Test recovery state transitions publish events."""
        # Activate and recover
        await ess_controller.activate("Test")
        await ess_controller.check_recovery(-8.0)
        
        # Check recovery event was published
        recovery_events = [
            e for e in ess_controller.event_bus.published_events
            if hasattr(e, 'type') and e.type == "emergency.recovery"
        ]
        
        assert len(recovery_events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
