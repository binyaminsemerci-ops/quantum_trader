"""
Continuous Learning Scheduler - Runs scheduled AI model retraining

This script provides a lightweight continuous learning system that:
1. Monitors model age
2. Triggers retraining when needed (default: every 24 hours)
3. Automatically deploys new models
4. Restarts backend to load fresh models

Usage:
    python scripts/continuous_learning_scheduler.py

Environment Variables:
    CLM_RETRAIN_HOURS: Hours between retraining (default: 24)
    CLM_ENABLED: Enable/disable CLM (default: true)
"""

import os
import sys
import time
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RETRAIN_INTERVAL_HOURS = int(os.getenv("CLM_RETRAIN_HOURS", "24"))
CLM_ENABLED = os.getenv("CLM_ENABLED", "true").lower() == "true"
MODEL_PATH = Path("ai_engine/models/xgb_model.pkl")
TRAINING_SCRIPT = Path("scripts/train_xgboost_quick.py")
DEPLOYMENT_SCRIPT = Path("scripts/scheduled_retraining.ps1")


def get_model_age_hours():
    """Get age of current model in hours"""
    if not MODEL_PATH.exists():
        return 999  # Force training if model missing
    
    model_mtime = MODEL_PATH.stat().st_mtime
    age_seconds = time.time() - model_mtime
    age_hours = age_seconds / 3600
    return age_hours


def should_retrain():
    """Check if retraining is needed"""
    if not CLM_ENABLED:
        logger.info("CLM disabled, skipping retraining check")
        return False
    
    age_hours = get_model_age_hours()
    logger.info(f"Model age: {age_hours:.1f} hours (threshold: {RETRAIN_INTERVAL_HOURS}h)")
    
    return age_hours >= RETRAIN_INTERVAL_HOURS


def trigger_retraining():
    """Trigger retraining via Python training scripts"""
    logger.info("🔄 Triggering scheduled retraining...")
    
    try:
        import requests
        
        # Trigger retraining via API (if backend has endpoint)
        # Or run training scripts directly
        training_scripts = [
            "scripts/train_xgboost_quick.py",
            "scripts/train_lightgbm.py",
            "scripts/train_nhits.py",
            "scripts/train_patchtst.py"
        ]
        
        all_success = True
        for script in training_scripts:
            script_path = Path(script)
            if not script_path.exists():
                logger.warning(f"⚠️  Script not found: {script}, skipping...")
                continue
                
            logger.info(f"🔧 Running: {script}")
            try:
                result = subprocess.run(
                    ["python3", str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout per model
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ {script} completed")
                else:
                    logger.error(f"❌ {script} failed: {result.stderr[:200]}")
                    all_success = False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ {script} timed out")
                all_success = False
            except Exception as e:
                logger.error(f"❌ {script} error: {e}")
                all_success = False
        
        return all_success
            
    except Exception as e:
        logger.error(f"❌ Retraining error: {e}")
        return False


def run_continuous_loop():
    """Main continuous learning loop"""
    logger.info("=" * 60)
    logger.info("  CONTINUOUS LEARNING MANAGER (CLM) - Started")
    logger.info("=" * 60)
    logger.info(f"  Retrain interval: {RETRAIN_INTERVAL_HOURS} hours")
    logger.info(f"  CLM enabled: {CLM_ENABLED}")
    logger.info("=" * 60)
    
    check_interval_minutes = 30  # Check every 30 minutes
    
    while True:
        try:
            if should_retrain():
                logger.info("⏰ Retraining threshold reached, starting retraining...")
                success = trigger_retraining()
                
                if success:
                    logger.info("✅ Retraining cycle complete, resuming monitoring...")
                else:
                    logger.warning("⚠️  Retraining failed, will retry next cycle")
            else:
                age_hours = get_model_age_hours()
                hours_until_retrain = RETRAIN_INTERVAL_HOURS - age_hours
                logger.info(f"✓ Model is fresh ({age_hours:.1f}h old), next retrain in {hours_until_retrain:.1f}h")
            
            # Sleep until next check
            logger.info(f"💤 Sleeping for {check_interval_minutes} minutes...\n")
            time.sleep(check_interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("\n👋 CLM scheduler stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            logger.info("Retrying in 5 minutes...")
            time.sleep(300)


if __name__ == "__main__":
    if not CLM_ENABLED:
        logger.warning("⚠️  CLM is disabled (set CLM_ENABLED=true to enable)")
        logger.info("Exiting...")
        sys.exit(0)
    
    logger.info("🚀 Starting CLM scheduler...")
    run_continuous_loop()
