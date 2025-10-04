#!/usr/bin/env python3
"""
Demo script to test performance monitoring functionality.

This script demonstrates the performance monitoring capabilities
by simulating various API operations and showing the collected metrics.
"""

import asyncio
import time
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from performance_monitor import (
    PerformanceCollector,
    RequestMetrics,
    DatabaseMetrics,
    performance_timer,
    get_performance_summary
)
from datetime import datetime, timezone


async def simulate_api_requests():
    """Simulate various API requests with different performance characteristics."""

    collector = PerformanceCollector()

    print("🚀 Starting performance monitoring demo...\n")

    # Simulate fast requests
    for i in range(5):
        start_time = time.perf_counter()
        await asyncio.sleep(0.01)  # Simulate 10ms request
        duration_ms = (time.perf_counter() - start_time) * 1000

        metric = RequestMetrics(
            method="GET",
            path="/api/health",
            status_code=200,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            db_queries=0,
            memory_mb=45.2
        )
        collector.add_request_metric(metric)

    # Simulate slower requests with DB queries
    for i in range(3):
        start_time = time.perf_counter()
        await asyncio.sleep(0.05)  # Simulate 50ms request
        duration_ms = (time.perf_counter() - start_time) * 1000

        metric = RequestMetrics(
            method="GET",
            path="/api/trades",
            status_code=200,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            db_queries=2,
            db_time_ms=15.5,
            memory_mb=47.1
        )
        collector.add_request_metric(metric)

        # Add corresponding database metrics
        db_metric = DatabaseMetrics(
            query="SELECT * FROM trade_logs ORDER BY timestamp DESC LIMIT 100",
            duration_ms=12.3,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        collector.add_db_metric(db_metric)

    # Simulate a slow request
    start_time = time.perf_counter()
    await asyncio.sleep(0.15)  # Simulate 150ms slow request
    duration_ms = (time.perf_counter() - start_time) * 1000

    slow_metric = RequestMetrics(
        method="POST",
        path="/api/trades",
        status_code=201,
        duration_ms=duration_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
        db_queries=3,
        db_time_ms=45.2,
        memory_mb=49.8
    )
    collector.add_request_metric(slow_metric)

    # Print collected metrics
    print("📊 Performance Summary:")
    print("=" * 50)

    request_summary = collector.get_request_summary()
    print(f"Total Requests: {request_summary['total_requests']}")
    print(f"Average Duration: {request_summary['avg_duration_ms']:.2f}ms")
    print(f"Max Duration: {request_summary['max_duration_ms']:.2f}ms")
    print(f"Slow Requests: {request_summary['slow_requests']}")
    print(f"Average DB Queries: {request_summary['avg_db_queries']:.1f}")

    print("\n🎯 Endpoint Performance:")
    for endpoint, stats in request_summary['endpoint_performance'].items():
        print(f"  {endpoint}: {stats['count']} requests, avg {stats['avg_ms']:.2f}ms")

    db_summary = collector.get_db_summary()
    if "total_queries" in db_summary:
        print(f"\n🗃️ Database Performance:")
        print(f"Total Queries: {db_summary['total_queries']}")
        print(f"Average Duration: {db_summary['avg_duration_ms']:.2f}ms")
        print(f"Slow Queries: {db_summary['slow_queries']}")


async def demo_performance_timer():
    """Demonstrate the performance_timer context manager."""
    print("\n⏱️ Performance Timer Demo:")
    print("-" * 30)

    async with performance_timer("Data processing operation"):
        await asyncio.sleep(0.03)  # Simulate work
        print("  Processed 1000 records")

    async with performance_timer("Model inference"):
        await asyncio.sleep(0.02)  # Simulate ML inference
        print("  Generated trading signal")

    async with performance_timer("Database batch insert"):
        await asyncio.sleep(0.01)  # Simulate DB operation
        print("  Inserted 50 trade records")


def demo_system_metrics():
    """Demonstrate system metrics collection."""
    print("\n💻 System Metrics Demo:")
    print("-" * 25)

    try:
        import psutil
        process = psutil.Process()

        print(f"CPU Usage: {process.cpu_percent():.1f}%")
        print(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        print(f"Memory Percent: {process.memory_percent():.1f}%")
        print(f"Open Files: {len(process.open_files())}")
        print(f"Threads: {process.num_threads()}")

    except ImportError:
        print("psutil not available for system metrics")


async def main():
    """Run the complete performance monitoring demo."""
    print("🔍 Quantum Trader Performance Monitoring Demo")
    print("=" * 50)

    await simulate_api_requests()
    await demo_performance_timer()
    demo_system_metrics()

    print("\n✅ Demo completed! Performance monitoring is working correctly.")
    print("\n💡 In production, these metrics would be:")
    print("   • Logged to structured log files")
    print("   • Exposed via /api/metrics/* endpoints")
    print("   • Used for alerting on slow requests")
    print("   • Tracked over time for performance trends")


if __name__ == "__main__":
    asyncio.run(main())
