"""
Train all 4 models on Futures-specific data
Includes: OHLCV, funding rates, open interest, long/short ratios

This is optimized for leverage trading with cross margin.
"""
import subprocess
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_training(script: str, name: str) -> tuple:
    """Run a training script and return success status + duration."""
    start = time.time()
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"[ROCKET] Starting {name} training...")
        logger.info(f"{'='*60}")
        
        result = subprocess.run(
            ['python', f'scripts/{script}'],
            capture_output=True,
            text=True,
            timeout=7200  # 2h timeout
        )
        
        duration = (time.time() - start) / 60
        
        if result.returncode == 0:
            logger.info(f"\n[OK] {name} succeeded after {duration:.1f} minutes")
            return True, duration
        else:
            logger.error(f"\n❌ {name} failed after {duration:.1f} minutes")
            logger.error(f"Error: {result.stderr[-500:]}")  # Last 500 chars
            return False, duration
    
    except subprocess.TimeoutExpired:
        duration = (time.time() - start) / 60
        logger.error(f"\n⏱️ {name} timed out after {duration:.1f} minutes")
        return False, duration
    
    except Exception as e:
        duration = (time.time() - start) / 60
        logger.error(f"\n❌ {name} crashed: {e}")
        return False, duration


def main():
    """Train all models on futures data."""
    start_total = time.time()
    
    logger.info("=" * 60)
    logger.info("[TARGET] TRAINING ALL 4 MODELS ON FUTURES DATA")
    
    # Check if futures data exists
    data_path = Path("data/binance_futures_training_data.csv")
    if not data_path.exists():
        logger.error("❌ Futures training data not found!")
        logger.error(f"Expected: {data_path}")
        logger.error("Run: python scripts/fetch_futures_data.py")
        return False
    
    # Get data size
    import pandas as pd
    df = pd.read_csv(data_path)
    logger.info(f"   Dataset: {len(df):,} rows ({df['symbol'].nunique()} symbols)")
    logger.info("=" * 60)
    
    # Training order: fastest to slowest
    models = [
        ("train_futures_xgboost.py", "XGBoost"),
        # We'll skip LightGBM for now (uses XGB as placeholder)
        ("train_futures_nhits.py", "N-HiTS"),
        ("train_futures_patchtst.py", "PatchTST"),
    ]
    
    results = {}
    
    for script, name in models:
        success, duration = run_training(script, name)
        results[name] = {
            'success': success,
            'duration': duration
        }
    
    # Summary
    total_duration = (time.time() - start_total) / 60
    
    logger.info("\n" + "="*60)
    logger.info("[CHART] TRAINING SUMMARY")
    logger.info("="*60)
    
    for name, result in results.items():
        status = "[OK] SUCCESS" if result['success'] else "❌ FAILED"
        logger.info(f"  {name:12} - {status:12} ({result['duration']:.1f} min)")
    
    logger.info(f"\n⏱️  Total time: {total_duration:.1f} minutes")
    
    # Check if all succeeded
    all_success = all(r['success'] for r in results.values())
    
    if all_success:
        logger.info("\n" + "="*60)
        logger.info("[OK] ALL MODELS TRAINED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Setup testnet keys in config.yaml")
        logger.info("2. Run testnet trading: python scripts/testnet_trading.py")
        logger.info("3. Monitor: tail -f logs/testnet_trading.log")
        return True
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ SOME MODELS FAILED")
        logger.error("="*60)
        logger.error("Check logs above for details")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
