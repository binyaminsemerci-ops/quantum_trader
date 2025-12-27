"""
[P2-02] Common Metrics Logger API
==================================

Standardized metrics collection and logging to avoid duplication
across modules. Provides consistent interface for:
- Performance metrics
- Trading metrics
- System health metrics
- Custom metrics

Usage:
    from backend.core.metrics_logger import get_metrics_logger
    
    metrics = get_metrics_logger()
    metrics.record_trade(symbol="BTCUSDT", pnl=150.50, outcome="WIN")
    metrics.record_latency("execution.order_place", 0.045)
    metrics.record_counter("signals.generated", labels={"model": "ensemble_v2"})
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric type classification."""
    COUNTER = "counter"        # Incrementing value (e.g., trade count)
    GAUGE = "gauge"            # Current value (e.g., open positions)
    HISTOGRAM = "histogram"    # Distribution (e.g., latency)
    SUMMARY = "summary"        # Aggregated stats (e.g., PnL)


@dataclass
class Metric:
    """Standard metric structure."""
    name: str
    value: float
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsLogger:
    """
    [P2-02] Standardized metrics collection and logging.
    
    Features:
    - Consistent metric naming (namespace.category.metric)
    - Automatic timestamping
    - Label support for filtering/grouping
    - Buffering for batch writes
    - Export to multiple backends (Prometheus, InfluxDB, logs)
    """
    
    def __init__(
        self,
        namespace: str = "quantum_trader",
        buffer_size: int = 100,
        flush_interval_sec: int = 10,
    ):
        """
        Initialize metrics logger.
        
        Args:
            namespace: Metric namespace prefix
            buffer_size: Max metrics to buffer before flush
            flush_interval_sec: Auto-flush interval
        """
        self.namespace = namespace
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval_sec
        
        self._buffer: List[Metric] = []
        self._last_flush = time.time()
        
        logger.info(f"[P2-02] MetricsLogger initialized (namespace={namespace})")
    
    def _format_metric_name(self, name: str) -> str:
        """Format metric name with namespace."""
        if name.startswith(self.namespace):
            return name
        return f"{self.namespace}.{name}"
    
    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        **metadata
    ) -> None:
        """
        Record a counter metric (incrementing value).
        
        Examples:
            metrics.record_counter("trades.executed", labels={"symbol": "BTCUSDT"})
            metrics.record_counter("signals.generated", labels={"model": "rl_v3"})
        """
        metric = Metric(
            name=self._format_metric_name(name),
            value=value,
            type=MetricType.COUNTER,
            labels=labels or {},
            metadata=metadata
        )
        self._add_metric(metric)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        **metadata
    ) -> None:
        """
        Record a gauge metric (current value snapshot).
        
        Examples:
            metrics.record_gauge("positions.open", 5)
            metrics.record_gauge("balance.usdt", 10000.50)
        """
        metric = Metric(
            name=self._format_metric_name(name),
            value=value,
            type=MetricType.GAUGE,
            labels=labels or {},
            metadata=metadata
        )
        self._add_metric(metric)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        **metadata
    ) -> None:
        """
        Record a histogram metric (distribution).
        
        Examples:
            metrics.record_histogram("latency.execution", 0.045)
            metrics.record_histogram("pnl.per_trade", 150.50)
        """
        metric = Metric(
            name=self._format_metric_name(name),
            value=value,
            type=MetricType.HISTOGRAM,
            labels=labels or {},
            metadata=metadata
        )
        self._add_metric(metric)
    
    def record_trade(
        self,
        symbol: str,
        pnl: float,
        outcome: str,
        size_usd: Optional[float] = None,
        hold_duration_sec: Optional[int] = None,
        **metadata
    ) -> None:
        """
        Record trade metrics (convenience method).
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            pnl: Profit/loss in USD
            outcome: "WIN", "LOSS", or "BREAKEVEN"
            size_usd: Position size in USD
            hold_duration_sec: How long position was held
        """
        labels = {"symbol": symbol, "outcome": outcome}
        
        self.record_histogram("trade.pnl", pnl, labels=labels)
        
        if size_usd:
            self.record_histogram("trade.size_usd", size_usd, labels=labels)
        
        if hold_duration_sec:
            self.record_histogram("trade.duration_sec", hold_duration_sec, labels=labels)
        
        self.record_counter("trade.total", labels=labels, **metadata)
    
    def record_latency(
        self,
        operation: str,
        duration_sec: float,
        success: bool = True,
        **metadata
    ) -> None:
        """
        Record operation latency.
        
        Args:
            operation: Operation name (e.g., "execution.order_place")
            duration_sec: Duration in seconds
            success: Whether operation succeeded
        """
        labels = {"operation": operation, "success": str(success)}
        self.record_histogram("latency", duration_sec, labels=labels, **metadata)
    
    def record_model_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: str,
        confidence: float,
        **metadata
    ) -> None:
        """
        Record AI model prediction metrics.
        
        Args:
            model_name: Model identifier (e.g., "ensemble_v2")
            symbol: Trading pair
            prediction: "BUY", "SELL", or "HOLD"
            confidence: Prediction confidence (0-1)
        """
        labels = {
            "model": model_name,
            "symbol": symbol,
            "prediction": prediction
        }
        
        self.record_counter("model.predictions", labels=labels)
        self.record_histogram("model.confidence", confidence, labels=labels, **metadata)
    
    def _add_metric(self, metric: Metric) -> None:
        """Add metric to buffer and flush if needed."""
        self._buffer.append(metric)
        
        # Auto-flush if buffer full or time elapsed
        if (len(self._buffer) >= self.buffer_size or 
            time.time() - self._last_flush >= self.flush_interval):
            self.flush()
    
    def flush(self) -> None:
        """Flush buffered metrics to output."""
        if not self._buffer:
            return
        
        try:
            # Write to log file (structured JSON)
            for metric in self._buffer:
                logger.info(
                    f"[METRIC] {metric.name}={metric.value} "
                    f"type={metric.type.value} "
                    f"labels={metric.labels} "
                    f"ts={metric.timestamp.isoformat()}"
                )
            
            # TODO: Export to Prometheus/InfluxDB/TimescaleDB
            # TODO: Update metrics dashboard
            
            self._buffer.clear()
            self._last_flush = time.time()
        
        except Exception as e:
            logger.error(f"[P2-02] Failed to flush metrics: {e}")


# Singleton instance
_metrics_logger: Optional[MetricsLogger] = None


def get_metrics_logger(namespace: str = "quantum_trader") -> MetricsLogger:
    """Get or create singleton metrics logger."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger(namespace=namespace)
    return _metrics_logger


# Module-level convenience functions for static-like usage (no recursion)
def record_counter(name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None, **metadata) -> None:
    """Record counter metric."""
    get_metrics_logger().record_counter(name, value, labels, **metadata)


def record_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None, **metadata) -> None:
    """Record gauge metric."""
    get_metrics_logger().record_gauge(name, value, labels, **metadata)


def record_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None, **metadata) -> None:
    """Record histogram metric."""
    get_metrics_logger().record_histogram(name, value, labels, **metadata)


def record_trade(symbol: str, pnl: float, outcome: str, size_usd: float, hold_duration_sec: float = 0) -> None:
    """Record trade metrics."""
    get_metrics_logger().record_trade(symbol, pnl, outcome, size_usd, hold_duration_sec)


def record_latency(operation: str, duration_sec: float, success: bool = True) -> None:
    """Record latency metric."""
    get_metrics_logger().record_latency(operation, duration_sec, success)


def record_model_prediction(model_name: str, symbol: str, prediction: str, confidence: float) -> None:
    """Record model prediction."""
    get_metrics_logger().record_model_prediction(model_name, symbol, prediction, confidence)


def flush() -> None:
    """Flush metrics buffer."""
    get_metrics_logger().flush()


# Context manager for timing operations
class MetricsTimer:
    """
    Context manager for timing operations.
    
    Usage:
        with MetricsTimer("execution.order_place"):
            # ... execute operation ...
    """
    
    def __init__(self, operation: str, metrics: Optional[MetricsLogger] = None):
        self.operation = operation
        self.metrics = metrics or get_metrics_logger()
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        self.metrics.record_latency(self.operation, duration, success=success)


# Export singleton instance for direct import
metrics_logger = get_metrics_logger()
