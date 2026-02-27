#!/usr/bin/env python3
"""
Minimal test for Binance Futures dry-run order execution
"""
import os
import sys
import asyncio

os.environ["QT_EXECUTION_EXCHANGE"] = "binance-futures"
os.environ["QT_MARKET_TYPE"] = "usdm_perp"
os.environ["QT_MARGIN_MODE"] = "cross"
os.environ["QT_DEFAULT_LEVERAGE"] = "5"
os.environ["QT_LIQUIDITY_STABLE_QUOTES"] = "USDT,USDC"
os.environ["STAGING_MODE"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from backend.config import load_execution_config, load_liquidity_config
from backend.services.execution.execution import build_execution_adapter

async def main():
    exec_cfg = load_execution_config()
    liq_cfg = load_liquidity_config()
    adapter = build_execution_adapter(exec_cfg)
    print(f"Adapter type: {type(adapter).__name__}")
    print(f"Ready: {getattr(adapter, 'ready', False)}")
    # Simulate a dry-run order
    symbol = "BTCUSDT"
    side = "BUY"
    qty = 0.001
    price = 35000.0
    print(f"Submitting dry-run futures order: {side} {symbol} qty={qty} price={price}")
    order_id = await adapter.submit_order(symbol, side, qty, price)
    print(f"Order result: {order_id}")

if __name__ == "__main__":
    asyncio.run(main())
