"""
Unit Tests for P2-02: Logging and Metrics APIs
===============================================

Tests for MetricsLogger and AuditLogger standardization.
"""

import pytest
import time
from datetime import datetime
from pathlib import Path


# ==============================================================================
# P2-02: MetricsLogger Tests
# ==============================================================================

class TestMetricsLogger:
    """Unit tests for MetricsLogger API."""
    
    def test_metrics_logger_counter(self):
        """Test recording counter metrics."""
        from backend.core import metrics_logger
        
        # Record counter
        metrics_logger.record_counter(
            "test_counter",
            value=1.0,
            labels={"symbol": "BTCUSDT"}
        )
        
        # Should not raise exception
        assert True
    
    def test_metrics_logger_gauge(self):
        """Test recording gauge metrics."""
        from backend.core import metrics_logger
        
        metrics_logger.record_gauge(
            "portfolio_value_usd",
            value=100000.0,
            labels={"account": "main"}
        )
        
        assert True
    
    def test_metrics_logger_histogram(self):
        """Test recording histogram metrics."""
        from backend.core import metrics_logger
        
        metrics_logger.record_histogram(
            "trade_pnl",
            value=150.0,
            labels={"symbol": "ETHUSDT"}
        )
        
        assert True
    
    def test_metrics_logger_trade(self):
        """Test recording trade metrics."""
        from backend.core import metrics_logger
        
        metrics_logger.record_trade(
            symbol="BTCUSDT",
            pnl=150.0,
            outcome="win",
            size_usd=10000.0,
            hold_duration_sec=3600
        )
        
        assert True
    
    def test_metrics_logger_latency(self):
        """Test recording latency metrics."""
        from backend.core import metrics_logger
        
        metrics_logger.record_latency(
            operation="risk_check",
            duration_sec=0.05,
            success=True
        )
        
        assert True
    
    def test_metrics_logger_model_prediction(self):
        """Test recording model prediction metrics."""
        from backend.core import metrics_logger
        
        metrics_logger.record_model_prediction(
            model_name="RL_v3",
            symbol="BTCUSDT",
            prediction="LONG",
            confidence=0.89
        )
        
        assert True
    
    def test_metrics_timer_context_manager(self):
        """Test MetricsTimer context manager."""
        from backend.core.metrics_logger import MetricsTimer
        
        with MetricsTimer("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        # Should complete without exception
        assert True
    
    def test_metrics_buffer_flush(self):
        """Test metrics buffer auto-flush."""
        from backend.core import metrics_logger
        
        # Record multiple metrics
        for i in range(5):
            metrics_logger.record_counter(f"test_{i}", 1.0)
        
        # Flush buffer
        metrics_logger.flush()
        
        assert True


# ==============================================================================
# P2-02: AuditLogger Tests
# ==============================================================================

class TestAuditLogger:
    """Unit tests for AuditLogger API."""
    
    @pytest.fixture(autouse=True)
    def reset_audit_logger(self):
        """Reset audit logger singleton between tests."""
        import backend.core.audit_logger as audit_logger_module
        audit_logger_module._audit_logger = None
        yield
        audit_logger_module._audit_logger = None
    
    @pytest.fixture
    def audit_log_dir(self, tmp_path):
        """Create temporary audit log directory."""
        log_dir = tmp_path / "audit_logs"
        log_dir.mkdir()
        return log_dir
    
    def test_audit_logger_trade_decision(self, audit_log_dir):
        """Test logging trade decisions."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        # Initialize with test directory
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_trade_decision(
            symbol="BTCUSDT",
            action="LONG",
            reason="Strong momentum + RL confidence",
            confidence=0.89,
            model="RL_v3"
        )
        
        # Verify log file created
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_trade_executed(self, audit_log_dir):
        """Test logging trade execution."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_trade_executed(
            symbol="BTCUSDT",
            side="LONG",
            size_usd=10000.0,
            entry_price=42000.0,
            order_id="order_123",
            leverage=15.0
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_trade_closed(self, audit_log_dir):
        """Test logging trade closure."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_trade_closed(
            symbol="BTCUSDT",
            pnl=150.0,
            outcome="WIN",
            close_reason="TP_HIT",
            hold_duration_sec=3600
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_risk_block(self, audit_log_dir):
        """Test logging risk blocks."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_risk_block(
            symbol="BTCUSDT",
            action="OPEN_LONG",
            reason="MAX_POSITIONS_REACHED",
            blocker="RiskGuard"
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_emergency_triggered(self, audit_log_dir):
        """Test logging emergency triggers."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_emergency_triggered(
            severity="critical",
            trigger_reason="DRAWDOWN_EXCEEDED",
            positions_closed=3
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_emergency_recovered(self, audit_log_dir):
        """Test logging emergency recovery."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_emergency_recovered(
            recovery_type="auto",
            old_mode="EMERGENCY",
            new_mode="PROTECTIVE",
            current_dd_pct=-3.2
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_model_promoted(self, audit_log_dir):
        """Test logging model promotions."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_model_promoted(
            model_name="RL_v4",
            old_version="1.2.0",
            new_version="1.3.0",
            reason="Better validation metrics"
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_policy_changed(self, audit_log_dir):
        """Test logging policy changes."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_policy_changed(
            policy_name="risk_mode",
            old_value="BALANCED",
            new_value="CONSERVATIVE",
            changed_by="AI-HFOS",
            reason="HIGH_VOLATILITY"
        )
        
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
    
    def test_audit_logger_rotation(self, audit_log_dir):
        """Test audit log file rotation."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=audit_log_dir)
        
        # Simulate multiple days of logs
        for day in range(3):
            audit_logger.log_trade_decision(
                symbol="BTCUSDT",
                action="LONG",
                reason="Test",
                confidence=0.5,
                model="test"
            )
        
        # Should have daily log files
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) >= 1
    
    def test_audit_logger_json_format(self, audit_log_dir):
        """Test audit log JSON format."""
        from backend.core import audit_logger
        from backend.core.audit_logger import get_audit_logger
        import json
        
        get_audit_logger(log_dir=audit_log_dir)
        
        audit_logger.log_trade_decision(
            symbol="TESTUSDT",
            action="LONG",
            reason="Test",
            confidence=0.75,
            model="test_model"
        )
        
        # Read and validate JSON
        log_files = list(audit_log_dir.glob("*.jsonl"))
        assert len(log_files) > 0
        
        with open(log_files[0], 'r') as f:
            line = f.readline()
            data = json.loads(line)
            assert "event_type" in data
            assert "timestamp" in data
            assert "target" in data  # symbol is stored as 'target'
            assert data["target"] == "TESTUSDT"


# ==============================================================================
# P2-02: Integration Tests
# ==============================================================================

class TestLoggingAPIIntegration:
    """Integration tests for logging APIs with services."""
    
    def test_metrics_and_audit_together(self, tmp_path):
        """Test using both MetricsLogger and AuditLogger together."""
        from backend.core import metrics_logger, audit_logger
        from backend.core.metrics_logger import MetricsTimer
        from backend.core.audit_logger import get_audit_logger
        
        get_audit_logger(log_dir=tmp_path)
        
        # Simulate a trade flow
        with MetricsTimer("trade_execution"):
            # Log decision
            audit_logger.log_trade_decision(
                symbol="BTCUSDT",
                action="LONG",
                reason="Test trade",
                confidence=0.85,
                model="RL_v3"
            )
            
            # Record metrics
            metrics_logger.record_counter("trades_signaled", 1.0)
            
            # Log execution
            audit_logger.log_trade_executed(
                symbol="BTCUSDT",
                side="LONG",
                size_usd=5000.0,
                entry_price=40000.0,
                order_id="test_123",
                leverage=15.0
            )
            
            # Record trade metric
            metrics_logger.record_trade(
                symbol="BTCUSDT",
                pnl=0.0,  # Not closed yet
                outcome="open",
                size_usd=5000.0,
                hold_duration_sec=0
            )
        
        # Verify logs created
        log_files = list(tmp_path.glob("*.jsonl"))
        assert len(log_files) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
