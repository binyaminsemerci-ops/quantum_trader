"""
Tests for performance monitoring functionality.

These tests verify that performance metrics are correctly collected
and that monitoring endpoints work as expected.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timezone

from performance_monitor import (
    collector, 
    PerformanceCollector,
    RequestMetrics,
    DatabaseMetrics,
    performance_timer
)


class TestPerformanceMonitoring:
    """Test suite for performance monitoring."""
    
    def test_performance_collector_creation(self):
        """Test that performance collector can be created."""
        test_collector = PerformanceCollector(max_metrics=100)
        assert test_collector.max_metrics == 100
        assert len(test_collector.request_metrics) == 0
        assert len(test_collector.db_metrics) == 0
    
    def test_request_metrics_creation(self):
        """Test RequestMetrics dataclass functionality."""
        metric = RequestMetrics(
            method="GET",
            path="/api/test",
            status_code=200,
            duration_ms=150.5,
            timestamp="2025-10-04T13:54:00Z"
        )
        
        assert metric.method == "GET"
        assert metric.path == "/api/test" 
        assert metric.status_code == 200
        assert metric.duration_ms == 150.5
        
        # Test dict conversion
        metric_dict = metric.to_dict()
        assert metric_dict["method"] == "GET"
        assert metric_dict["duration_ms"] == 150.5
    
    def test_database_metrics_creation(self):
        """Test DatabaseMetrics dataclass functionality."""
        metric = DatabaseMetrics(
            query="SELECT * FROM trades WHERE symbol = ?",
            duration_ms=25.3,
            timestamp="2025-10-04T13:54:00Z"
        )
        
        assert metric.query.startswith("SELECT")
        assert metric.duration_ms == 25.3
        
        # Test dict conversion
        metric_dict = metric.to_dict()
        assert "query" in metric_dict
        assert metric_dict["duration_ms"] == 25.3
    
    def test_request_metrics_summary_empty(self):
        """Test request summary when no metrics exist."""
        test_collector = PerformanceCollector()
        summary = test_collector.get_request_summary()
        
        assert "message" in summary
        assert summary["message"] == "No request metrics available"
    
    def test_db_metrics_summary_empty(self):
        """Test database summary when no metrics exist."""
        test_collector = PerformanceCollector()
        summary = test_collector.get_db_summary()
        
        assert "message" in summary
        assert summary["message"] == "No database metrics available"
    
    def test_performance_collector_summary_methods(self):
        """Test the performance collector summary methods."""
        # Test when no metrics are available
        collector.request_metrics.clear()
        collector.db_metrics.clear()
        
        request_summary = collector.get_request_summary()
        assert "message" in request_summary
        
        db_summary = collector.get_db_summary()
        assert "message" in db_summary
        
        # Test reset functionality
        collector.reset_request_counters()
        assert collector.current_request_db_queries == 0
        assert collector.current_request_db_time == 0.0
    
    def test_slow_request_detection(self):
        """Test that slow requests are properly detected in metrics."""
        from backend.performance_monitor import RequestMetrics
        from datetime import datetime, timezone
        
        # Add a slow request metric
        slow_metric = RequestMetrics(
            method="GET",
            path="/api/slow-endpoint",
            status_code=200,
            duration_ms=1500.0,  # 1.5 seconds - considered slow
            timestamp=datetime.now(timezone.utc).isoformat(),
            db_queries=5,
            db_time_ms=800.0
        )
        
        collector.request_metrics.clear()
        collector.add_request_metric(slow_metric)
        
        summary = collector.get_request_summary()
        assert summary["slow_requests"] == 1
        assert summary["max_duration_ms"] == 1500.0
    
    def test_endpoint_performance_grouping(self):
        """Test that endpoint performance is correctly grouped."""
        from backend.performance_monitor import RequestMetrics
        from datetime import datetime, timezone
        
        collector.request_metrics.clear()
        
        # Add multiple requests to the same endpoint
        for i in range(3):
            metric = RequestMetrics(
                method="GET",
                path="/api/trades",
                status_code=200,
                duration_ms=100.0 + i * 10,  # 100, 110, 120ms
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            collector.add_request_metric(metric)
        
        summary = collector.get_request_summary()
        endpoint_perf = summary["endpoint_performance"]
        
        assert "GET /api/trades" in endpoint_perf
        assert endpoint_perf["GET /api/trades"]["count"] == 3
        assert endpoint_perf["GET /api/trades"]["avg_ms"] == 110.0  # (100+110+120)/3