"""
Failure Simulation Test Suite
Sprint 3 Part 3 - Hardening Tests

This package contains controlled failure simulations to test:
- Flash crashes (extreme volatility)
- Redis downtime (EventBus degradation)
- Binance API failures (rate limiting, connection issues)
- Signal floods (30+ signals in short time)
- ESS triggering and recovery

Purpose: Verify system robustness, safety mechanisms, and recovery capabilities.
"""

__version__ = "1.0.0"
