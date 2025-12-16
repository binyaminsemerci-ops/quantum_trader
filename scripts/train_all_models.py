"""
UNIFIED 4-MODEL TRAINING SCRIPT
Trains all ensemble models in optimal order:
1. XGBoost (fastest, 2-3 min)
2. LightGBM (fast, 2-3 min)
3. N-HiTS (medium, 10-15 min)
4. PatchTST (slower, 15-20 min)

Total time: ~30-40 minutes
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import subprocess

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training_script(script_name: str, model_name: str) -> bool:
    """Run a training script and return success status."""
    logger.info("=" * 60)
    logger.info(f"[ROCKET] TRAINING {model_name.upper()}")
    logger.info("=" * 60)
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        logger.error(f"❌ Script not found: {script_path}")
        return False
    
    start_time = time.time()
    
    try:
        # Run training script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info("=" * 60)
            logger.info(f"[OK] {model_name.upper()} TRAINING COMPLETE!")
            logger.info(f"⏱️  Time: {elapsed/60:.1f} minutes")
            logger.info("=" * 60)
            return True
        else:
            logger.error(f"❌ {model_name.upper()} training failed!")
            logger.error(f"   Exit code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running {model_name} training: {e}")
        return False


def main():
    """Train all models in sequence."""
    logger.info("\n" + "=" * 60)
    logger.info("[ROCKET] 4-MODEL ENSEMBLE TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Models to train:")
    logger.info("  1. XGBoost      (25% weight, tree-based)")
    logger.info("  2. LightGBM     (25% weight, fast trees)")
    logger.info("  3. N-HiTS       (30% weight, multi-rate temporal)")
    logger.info("  4. PatchTST     (20% weight, transformer)")
    logger.info("")
    logger.info("Estimated time: 30-40 minutes")
    logger.info("=" * 60)
    logger.info("")
    
    overall_start = time.time()
    results = {}
    
    # Training sequence
    training_jobs = [
        ("train_binance_only.py", "XGBoost"),
        ("train_lightgbm.py", "LightGBM"),
        ("train_nhits.py", "N-HiTS"),
        ("train_patchtst.py", "PatchTST")
    ]
    
    for i, (script, model_name) in enumerate(training_jobs, 1):
        logger.info(f"\n[CHART] Progress: {i}/4 models")
        success = run_training_script(script, model_name)
        results[model_name] = success
        
        if not success:
            logger.warning(f"[WARNING]  {model_name} training failed, continuing with next model...")
        
        # Brief pause between models
        if i < len(training_jobs):
            logger.info("\n⏸️  Pausing 5 seconds before next model...\n")
            time.sleep(5)
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 ENSEMBLE TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total time: {overall_elapsed/60:.1f} minutes")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("Training Results:")
    
    success_count = 0
    for model_name, success in results.items():
        emoji = "[OK]" if success else "❌"
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {emoji} {model_name}: {status}")
        if success:
            success_count += 1
    
    logger.info("")
    logger.info(f"[CHART] Summary: {success_count}/4 models trained successfully")
    
    if success_count == 4:
        logger.info("=" * 60)
        logger.info("🎉 ALL MODELS TRAINED! FULL ENSEMBLE READY!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Restart backend: docker-compose restart backend")
        logger.info("  2. Monitor ensemble: docker logs quantum_backend --tail 100 -f")
        logger.info("  3. Check model files: ls ai_engine/models/")
        logger.info("")
        logger.info("Ensemble will use:")
        logger.info("  - XGBoost (25%)")
        logger.info("  - LightGBM (25%)")
        logger.info("  - N-HiTS (30%)")
        logger.info("  - PatchTST (20%)")
        logger.info("")
        logger.info("Smart consensus requires 3/4 models to agree!")
        logger.info("=" * 60)
    
    elif success_count >= 2:
        logger.warning("=" * 60)
        logger.warning("[WARNING]  PARTIAL ENSEMBLE READY")
        logger.warning("=" * 60)
        logger.warning(f"{success_count}/4 models trained.")
        logger.warning("System will work with reduced models.")
        logger.warning("Consider re-running failed models individually.")
    
    else:
        logger.error("=" * 60)
        logger.error("❌ TRAINING FAILED")
        logger.error("=" * 60)
        logger.error(f"Only {success_count}/4 models trained successfully.")
        logger.error("System may not work properly.")
        logger.error("Check logs above for error details.")
    
    logger.info("=" * 60)
    logger.info("")


if __name__ == "__main__":
    main()
