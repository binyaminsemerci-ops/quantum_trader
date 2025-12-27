"""
Direct Model Retraining - Simple approach without complex dependencies
"""
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("DIRECT MODEL RETRAINING")
    logger.info("="*60)
    
    # Check if CLM is running
    logger.info("\nChecking CLM status from logs...")
    os.system("docker logs quantum_backend --tail 100 2>&1 | grep -i 'CLM\\|Continuous Learning' | tail -5")
    
    # Trigger via CLM internal method if available
    logger.info("\nAttempting to trigger retraining via CLM...")
    
    cmd = """
python -c "
import asyncio
import sys
sys.path.insert(0, '/app')

async def trigger():
    try:
        from backend.services.ai.continuous_learning_manager import get_clm_instance
        clm = get_clm_instance()
        if clm:
            print('CLM instance found, triggering retraining...')
            await clm.trigger_full_retrain(reason='manual_50plus_trades')
            print('Retraining triggered successfully')
        else:
            print('CLM not initialized')
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(trigger())
"
    """
    
    os.system(f"docker exec quantum_backend {cmd}")
    
    logger.info("\n" + "="*60)
    logger.info("Check logs for retraining progress:")
    logger.info("docker logs -f quantum_backend | grep -i 'retrain\\|xgboost\\|lightgbm'")
    logger.info("="*60)

if __name__ == "__main__":
    main()
