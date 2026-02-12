#!/usr/bin/env python3
"""
PATH 2.4A — Calibration Workflow Orchestrator

Runs complete calibration pipeline:
1. Collect signals from shadow mode (quantum:stream:signal.score)
2. Match with outcomes from apply.result + execution streams
3. Fit calibrator (isotonic regression)
4. Generate reliability diagram
5. Save calibrator for production use
6. Generate confidence semantics document

Usage:
    python run_calibration_workflow.py --days 3 --min-samples 1000

Requirements:
- Ensemble predictor has been running in shadow mode for >=24h
- Sufficient signal-outcome pairs available
- sklearn installed (pip install scikit-learn)
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_engine.calibration.replay_harness import (
    OutcomeCollector,
    ConfidenceCalibrator,
    main_replay_harness
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_prerequisites():
    """Check if prerequisites are met."""
    logger.info("\n🔍 Checking prerequisites...")
    
    # Check sklearn
    try:
        import sklearn
        logger.info(f"✅ sklearn version {sklearn.__version__}")
    except ImportError:
        logger.error("❌ sklearn not installed")
        logger.error("   Run: pip install scikit-learn")
        return False
    
    # Check Redis connection
    try:
        import redis.asyncio as aioredis
        r = await aioredis.from_url("redis://localhost:6379")
        await r.ping()
        await r.close()
        logger.info("✅ Redis connection OK")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False
    
    # Check signal.score stream exists
    try:
        import redis.asyncio as aioredis
        r = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
        length = await r.xlen("quantum:stream:signal.score")
        await r.close()
        
        if length == 0:
            logger.error("❌ quantum:stream:signal.score is empty")
            logger.error("   Ensemble predictor may not be running")
            return False
        
        logger.info(f"✅ signal.score stream has {length} entries")
        
    except Exception as e:
        logger.error(f"❌ Stream check failed: {e}")
        return False
    
    return True


async def run_workflow(args):
    """Run complete calibration workflow."""
    logger.info("\n" + "="*80)
    logger.info("PATH 2.4A — CONFIDENCE CALIBRATION WORKFLOW")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Days of data: {args.days}")
    logger.info(f"  Min samples required: {args.min_samples}")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info("")
    
    # Check prerequisites
    if not await check_prerequisites():
        logger.error("\n❌ Prerequisites not met. Aborting.")
        return 1
    
    if args.dry_run:
        logger.info("\n✅ Dry run successful. Prerequisites OK.")
        logger.info("Run without --dry-run to execute calibration.")
        return 0
    
    # Run main harness
    try:
        await main_replay_harness()
        
        logger.info("\n" + "="*80)
        logger.info("✅ CALIBRATION WORKFLOW COMPLETE")
        logger.info("="*80)
        logger.info("\n📋 Next steps:")
        logger.info("1. Review calibrator_v1.pkl.json for calibration statistics")
        logger.info("2. If Expected Calibration Error (ECE) < 0.1: Deploy to production")
        logger.info("3. Restart ensemble predictor to load new calibrator")
        logger.info("4. Monitor calibrated confidence values")
        logger.info("5. Create CONFIDENCE_SEMANTICS_V1.md document")
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Calibration workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="PATH 2.4A — Confidence Calibration Workflow"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days of historical data to use (default: 3)"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Minimum number of signal-outcome pairs required (default: 1000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check prerequisites without running calibration"
    )
    
    args = parser.parse_args()
    
    exit_code = asyncio.run(run_workflow(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
