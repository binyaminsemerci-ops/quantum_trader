"""
Learning Cadence Monitor - Periodic readiness checker.

Runs continuously in background, checking if learning conditions are met.
In logging-only mode: just reports status, no automatic training.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

from .cadence_policy import LearningCadencePolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/qt/quantum_trader/logs/learning_cadence.log')
    ]
)

logger = logging.getLogger(__name__)


async def check_readiness_once(policy: LearningCadencePolicy):
    """Single readiness check with detailed logging"""
    try:
        result = policy.evaluate_learning_readiness()
        
        # Build status message
        status_emoji = "🟢" if result["ready"] else "⏸️"
        
        logger.info(f"{status_emoji} LEARNING READINESS CHECK")
        logger.info(f"  Gate: {'✅ PASSED' if result['gate_passed'] else '❌ FAILED'} - {result['gate_reason']}")
        logger.info(f"  Trigger: {'🔥 FIRED' if result['trigger_fired'] else '⏳ WAITING'} - {result['trigger_reason']} (type={result['trigger_type']})")
        logger.info(f"  Authorization: {result['allowed_actions'] if result['allowed_actions'] else 'None'}")
        logger.info(f"  Stats: {result['stats']['total_trades']} trades ({result['stats']['new_trades']} new), "
                   f"{result['stats']['time_span_days']:.1f} days span, "
                   f"WR={result['stats']['win_rate']:.1%}, LR={result['stats']['loss_rate']:.1%}")
        logger.info(f"  Last training: {result['stats']['last_training']}, Total trainings: {result['stats']['total_trainings']}")
        
        if result["ready"]:
            logger.warning(f"🚀 LEARNING READY! Allowed actions: {result['allowed_actions']}")
            logger.warning(f"   Manual intervention required - logging only mode active")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Readiness check failed: {e}", exc_info=True)
        return None


async def monitor_loop(check_interval_seconds: int = 300):
    """
    Main monitoring loop.
    
    Args:
        check_interval_seconds: Time between checks (default 5 minutes)
    """
    policy = LearningCadencePolicy()
    
    logger.info("=" * 80)
    logger.info("🎓 LEARNING CADENCE MONITOR STARTED")
    logger.info(f"   Mode: LOGGING ONLY (no automatic training)")
    logger.info(f"   Check interval: {check_interval_seconds}s ({check_interval_seconds/60:.1f} minutes)")
    logger.info(f"   CLM storage: {policy.clm_path}")
    logger.info(f"   State file: {policy.state_path}")
    logger.info("=" * 80)
    
    iteration = 0
    
    while True:
        iteration += 1
        logger.info(f"--- Check #{iteration} at {datetime.now(timezone.utc).isoformat()} ---")
        
        result = await check_readiness_once(policy)
        
        # In logging-only mode, we just report status
        # Future enhancement: trigger actual training when ready
        
        logger.info(f"⏰ Next check in {check_interval_seconds}s")
        await asyncio.sleep(check_interval_seconds)


def main():
    """Entry point for monitor service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Learning Cadence Monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(monitor_loop(check_interval_seconds=args.interval))
    except KeyboardInterrupt:
        logger.info("🛑 Monitor stopped by user")
    except Exception as e:
        logger.error(f"💥 Monitor crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
