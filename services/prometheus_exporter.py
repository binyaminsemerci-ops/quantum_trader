"""
Prometheus Metrics Exporter for Quantum Trader Core Loop

Subscribes to EventBus topics and exports metrics at :9090/metrics
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ai_engine.services.eventbus_bridge import (
    EventBusClient,
    ExecutionResult,
    PositionUpdate,
    RiskApprovedSignal,
    TradeSignal,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

# Counters (monotonically increasing)
signals_total = Counter(
    "quantum_signals_total", "Total signals received from AI Engine", ["symbol", "action"]
)

approvals_total = Counter(
    "quantum_approvals_total",
    "Total signals approved by Risk Safety",
    ["symbol", "action"],
)

rejections_total = Counter(
    "quantum_rejections_total", "Total signals rejected by Risk Safety", ["symbol"]
)

executions_total = Counter(
    "quantum_executions_total", "Total orders executed", ["symbol", "action"]
)

position_updates_total = Counter(
    "quantum_position_updates_total",
    "Total position updates published",
    ["symbol", "side"],
)

# Gauges (can go up or down)
approval_rate = Gauge(
    "quantum_approval_rate", "Signal approval rate (approved/total)"
)

fill_rate = Gauge("quantum_fill_rate", "Order fill rate (filled/approved)")

confidence_avg = Gauge(
    "quantum_confidence_avg", "Average signal confidence (recent 100)"
)

pnl_unrealized = Gauge(
    "quantum_pnl_unrealized", "Total unrealized PnL across all positions"
)

pnl_realized = Gauge("quantum_pnl_realized", "Total realized PnL from closed trades")

positions_open = Gauge("quantum_positions_open", "Number of open positions")

exposure_usd = Gauge(
    "quantum_exposure_usd", "Total exposure in USD across all positions"
)

fees_total = Gauge("quantum_fees_total", "Total fees paid (execution + funding)")

balance_current = Gauge("quantum_balance_current", "Current account balance")

# Histograms (distributions)
execution_latency = Histogram(
    "quantum_execution_latency_seconds",
    "Time from signal to execution",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

position_duration = Histogram(
    "quantum_position_duration_minutes",
    "Duration of closed positions",
    buckets=[5, 15, 30, 60, 120, 360, 720, 1440],
)

# =============================================================================
# TRACKING STATE
# =============================================================================


class MetricsTracker:
    """Track metrics from EventBus topics"""

    def __init__(self):
        self.signals: Dict[str, TradeSignal] = {}
        self.approvals: Dict[str, RiskApprovedSignal] = {}
        self.executions: Dict[str, ExecutionResult] = {}
        self.positions: Dict[str, PositionUpdate] = {}

        # Running tallies
        self.total_signals = 0
        self.total_approvals = 0
        self.total_executions = 0
        self.confidence_history = []

    async def signal_consumer(self):
        """Track all AI signals"""
        logger.info("📡 Starting signal consumer (trade.signal.v5)...")
        async with EventBusClient() as bus:
            async for signal_data in bus.subscribe("trade.signal.v5"):
                try:
                    # Remove EventBus metadata
                    signal_data = {
                        k: v for k, v in signal_data.items() if not k.startswith("_")
                    }
                    signal = TradeSignal(**signal_data)

                    # Track signal
                    self.signals[signal.timestamp] = signal
                    self.total_signals += 1

                    # Update metrics
                    signals_total.labels(
                        symbol=signal.symbol, action=signal.action
                    ).inc()

                    # Track confidence
                    self.confidence_history.append(signal.confidence)
                    if len(self.confidence_history) > 100:
                        self.confidence_history.pop(0)
                    confidence_avg.set(sum(self.confidence_history) / len(self.confidence_history))

                    logger.info(
                        f"📥 Signal: {signal.symbol} {signal.action} (conf={signal.confidence:.3f})"
                    )

                except Exception as e:
                    logger.error(f"Error processing signal: {e}")

    async def approval_consumer(self):
        """Track approved signals"""
        logger.info("✅ Starting approval consumer (trade.signal.safe)...")
        async with EventBusClient() as bus:
            async for approval_data in bus.subscribe("trade.signal.safe"):
                try:
                    # Remove EventBus metadata
                    approval_data = {
                        k: v
                        for k, v in approval_data.items()
                        if not k.startswith("_")
                    }
                    approval = RiskApprovedSignal(**approval_data)

                    # Track approval
                    self.approvals[approval.timestamp] = approval
                    self.total_approvals += 1

                    # Update metrics
                    approvals_total.labels(
                        symbol=approval.symbol, action=approval.action
                    ).inc()

                    # Calculate approval rate
                    if self.total_signals > 0:
                        approval_rate.set(self.total_approvals / self.total_signals)

                    logger.info(
                        f"✅ Approved: {approval.symbol} {approval.action} (size=${approval.position_size_usd:.2f})"
                    )

                except Exception as e:
                    logger.error(f"Error processing approval: {e}")

    async def execution_consumer(self):
        """Track executed orders"""
        logger.info("💼 Starting execution consumer (trade.execution.res)...")
        async with EventBusClient() as bus:
            async for execution_data in bus.subscribe("trade.execution.res"):
                try:
                    # Remove EventBus metadata
                    execution_data = {
                        k: v
                        for k, v in execution_data.items()
                        if not k.startswith("_")
                    }
                    execution = ExecutionResult(**execution_data)

                    # Track execution
                    self.executions[execution.order_id] = execution
                    self.total_executions += 1

                    # Update metrics
                    executions_total.labels(
                        symbol=execution.symbol, action=execution.action
                    ).inc()

                    # Calculate fill rate
                    if self.total_approvals > 0:
                        fill_rate.set(self.total_executions / self.total_approvals)

                    # Calculate latency (signal → execution)
                    signal = self.signals.get(execution.timestamp.split(".")[0] + "Z")
                    if signal:
                        signal_time = datetime.fromisoformat(
                            signal.timestamp.replace("Z", "+00:00")
                        )
                        exec_time = datetime.fromisoformat(
                            execution.timestamp.replace("Z", "+00:00")
                        )
                        latency = (exec_time - signal_time).total_seconds()
                        execution_latency.observe(latency)

                    logger.info(
                        f"💼 Executed: {execution.order_id} {execution.symbol} @ ${execution.entry_price:.2f}"
                    )

                except Exception as e:
                    logger.error(f"Error processing execution: {e}")

    async def position_consumer(self):
        """Track position updates"""
        logger.info("📊 Starting position consumer (trade.position.update)...")
        async with EventBusClient() as bus:
            async for position_data in bus.subscribe("trade.position.update"):
                try:
                    # Remove EventBus metadata
                    position_data = {
                        k: v
                        for k, v in position_data.items()
                        if not k.startswith("_")
                    }
                    position = PositionUpdate(**position_data)

                    # Track position
                    self.positions[position.symbol] = position

                    # Update metrics
                    position_updates_total.labels(
                        symbol=position.symbol, side=position.side
                    ).inc()

                    # Calculate aggregate metrics
                    total_unrealized = sum(
                        p.unrealized_pnl for p in self.positions.values()
                    )
                    total_realized = sum(
                        p.realized_pnl for p in self.positions.values()
                    )
                    total_exposure = sum(
                        p.current_price * p.quantity for p in self.positions.values()
                    )
                    total_fees = sum(p.fees_paid for p in self.positions.values())
                    open_count = len(
                        [p for p in self.positions.values() if p.side != "CLOSED"]
                    )

                    pnl_unrealized.set(total_unrealized)
                    pnl_realized.set(total_realized)
                    exposure_usd.set(total_exposure)
                    fees_total.set(total_fees)
                    positions_open.set(open_count)

                    # Balance = starting balance + realized PnL + unrealized PnL - fees
                    current_balance = 10000 + total_realized + total_unrealized - total_fees
                    balance_current.set(current_balance)

                    logger.info(
                        f"📊 Position: {position.symbol} {position.side} | "
                        f"PnL=${position.unrealized_pnl:+.2f} | "
                        f"Open={open_count} | "
                        f"Total PnL=${total_unrealized+total_realized:+.2f}"
                    )

                except Exception as e:
                    logger.error(f"Error processing position: {e}")


# =============================================================================
# MAIN
# =============================================================================


async def main():
    """Run Prometheus metrics exporter"""
    logger.info("🚀 Starting Prometheus Metrics Exporter...")

    # Start HTTP server for Prometheus scraping
    port = 9090
    start_http_server(port)
    logger.info(f"📊 Metrics server started at http://localhost:{port}/metrics")

    # Create tracker
    tracker = MetricsTracker()

    # Run all consumers in parallel
    await asyncio.gather(
        tracker.signal_consumer(),
        tracker.approval_consumer(),
        tracker.execution_consumer(),
        tracker.position_consumer(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down metrics exporter...")
