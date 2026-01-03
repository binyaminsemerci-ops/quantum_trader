#!/usr/bin/env python3
"""
Exit Brain V3.5 - Dynamic Executor Service Entry Point

This is a standalone service that monitors all open positions
and executes TP/SL dynamically based on live price data.

Architecture:
- Runs as separate microservice (own container)
- Monitors positions every N seconds
- Executes MARKET orders when levels hit
- Independent from auto_executor (entry logic)

Flow:
1. Initialize Exit Brain components (planner + adapter + executor)
2. Start async monitoring loop
3. Every 10s: Check all positions vs TP/SL levels
4. When triggered: Execute MARKET close orders

HYBRID STOP-LOSS MODEL:
- Internal AI-driven SL (checked every cycle)
- Hard SL Binance order (last-resort safety net)
"""
import asyncio
import logging
import os
import sys
import signal
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"ts": "%(asctime)s", "level": "%(levelname)s", "service": "exit_brain_executor", "msg": "%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def main():
    """Initialize and start Exit Brain Dynamic Executor service"""
    
    logger.info("=" * 80)
    logger.info("🚀 Exit Brain V3.5 Dynamic Executor - Service Starting...")
    logger.info("=" * 80)
    
    # ============================================================================
    # STEP 1: Verify Configuration
    # ============================================================================
    from backend.config.exit_mode import (
        get_exit_mode,
        get_exit_executor_mode,
        is_exit_brain_live_fully_enabled,
        is_challenge_100_profile
    )
    
    exit_mode = get_exit_mode()
    executor_mode = get_exit_executor_mode()
    is_live = is_exit_brain_live_fully_enabled()
    is_challenge = is_challenge_100_profile()
    
    logger.info(f"📋 Configuration Check:")
    logger.info(f"   EXIT_MODE: {exit_mode}")
    logger.info(f"   EXIT_EXECUTOR_MODE: {executor_mode}")
    logger.info(f"   LIVE_ROLLOUT_ENABLED: {is_live}")
    logger.info(f"   CHALLENGE_100_PROFILE: {is_challenge}")
    
    if exit_mode != "EXIT_BRAIN_V3":
        logger.error(f"❌ EXIT_MODE={exit_mode}, expected EXIT_BRAIN_V3")
        logger.error("   Set EXIT_MODE=EXIT_BRAIN_V3 in .env to enable")
        return 1
    
    if not is_live and executor_mode != "SHADOW":
        logger.warning("⚠️  EXIT_BRAIN_V3_LIVE_ROLLOUT not enabled - running in SHADOW mode")
        logger.warning("   Enable with EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED for live trading")
    
    # ============================================================================
    # STEP 2: Initialize Binance Client
    # ============================================================================
    logger.info("🔗 Initializing Binance client...")
    
    from binance.client import Client as BinanceClient
    from backend.utils.binance_helpers import safe_futures_call
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    testnet = os.getenv("TESTNET", "false").lower() == "true"
    
    if not api_key or not api_secret:
        logger.error("❌ Missing BINANCE_API_KEY or BINANCE_SECRET_KEY in environment")
        return 1
    
    client = BinanceClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )
    
    network = "TESTNET" if testnet else "MAINNET"
    logger.info(f"✅ Binance client initialized ({network})")
    
    # ============================================================================
    # STEP 3: Initialize Exit Brain Components
    # ============================================================================
    logger.info("🧠 Initializing Exit Brain V3.5 components...")
    
    # Import Exit Brain components
    from backend.domains.exits.exit_brain_v3.planner import ExitBrainV3
    from backend.domains.exits.exit_brain_v3.adapter import ExitBrainAdapter
    from backend.domains.exits.exit_brain_v3.dynamic_executor import ExitBrainDynamicExecutor
    from backend.domains.exits.exit_order_gateway import ExitOrderGateway
    
    # Initialize AI planner
    planner = ExitBrainV3()
    logger.info("   ✅ ExitBrainV3 planner initialized")
    
    # Initialize adapter (translates AI plans to execution decisions)
    adapter = ExitBrainAdapter(planner=planner)
    logger.info("   ✅ ExitBrainAdapter initialized")
    
    # Initialize order gateway (executes orders on Binance)
    order_gateway = ExitOrderGateway(
        binance_client=client,
        mode="LIVE" if is_live else "SHADOW"
    )
    logger.info(f"   ✅ ExitOrderGateway initialized (mode={order_gateway.mode})")
    
    # Initialize dynamic executor
    loop_interval = float(os.getenv("EXIT_BRAIN_CHECK_INTERVAL_SEC", "10.0"))
    
    executor = ExitBrainDynamicExecutor(
        adapter=adapter,
        exit_order_gateway=order_gateway,
        position_source=client,
        loop_interval_sec=loop_interval,
        shadow_mode=(not is_live)
    )
    logger.info(f"   ✅ ExitBrainDynamicExecutor initialized (interval={loop_interval}s)")
    
    # ============================================================================
    # STEP 4: Setup Graceful Shutdown
    # ============================================================================
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.warning(f"⚠️  Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ============================================================================
    # STEP 5: Start Monitoring Loop
    # ============================================================================
    logger.info("=" * 80)
    logger.info("🚀 Exit Brain V3.5 Dynamic Executor - STARTED")
    logger.info("=" * 80)
    logger.info(f"   Mode: {executor.effective_mode}")
    logger.info(f"   Monitoring Interval: {loop_interval} seconds")
    logger.info(f"   Network: {network}")
    logger.info("=" * 80)
    
    await executor.start()
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # ============================================================================
    # STEP 6: Graceful Shutdown
    # ============================================================================
    logger.info("🛑 Shutting down Exit Brain Dynamic Executor...")
    await executor.stop()
    logger.info("✅ Exit Brain Dynamic Executor stopped cleanly")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("⚠️  Interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)
