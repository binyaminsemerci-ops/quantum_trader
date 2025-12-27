"""
Simple Retraining Trigger - Calls existing CLM infrastructure

This script triggers a scheduled retraining check through the CLM orchestrator.
"""

import asyncio
import logging
import sys

sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from backend.core.database import get_db_session
from backend.domains.learning.orchestrator import ContinuousLearningOrchestrator
from backend.domains.learning.retraining import RetrainingOrchestrator
from backend.domains.learning.data_pipeline import HistoricalDataFetcher, FeatureEngineer
from backend.infrastructure.event_bus import get_event_bus
from backend.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def trigger_retraining():
    """Trigger manual retraining by calling CLM orchestrator."""
    
    logger.info("Starting manual retraining trigger...")
    logger.info("This will check retraining criteria and start training if needed")
    
    try:
        async with get_db_session() as session:
            # Initialize components
            event_bus = get_event_bus()
            data_fetcher = HistoricalDataFetcher(session)
            feature_engineer = FeatureEngineer()
            
            retraining_orchestrator = RetrainingOrchestrator(
                db=session,
                event_bus=event_bus,
                data_fetcher=data_fetcher,
                feature_engineer=feature_engineer,
                config=settings.clm
            )
            
            clm_orchestrator = ContinuousLearningOrchestrator(
                db=session,
                event_bus=event_bus,
                retraining_orchestrator=retraining_orchestrator,
                config=settings.clm
            )
            
            # Run scheduled check
            logger.info("Running scheduled retraining check...")
            await clm_orchestrator._run_scheduled_retraining()
            
            logger.info("MANUAL TRIGGER COMPLETED")
            logger.info("Check logs above for retraining status and results")
            
            return True
            
    except Exception as e:
        logger.error(f"Manual trigger failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(trigger_retraining())
    sys.exit(0 if success else 1)
