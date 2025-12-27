"""
Manual Retraining Trigger - Train XGBoost/LightGBM with historical testnet data

This script directly triggers model retraining using data from the database,
bypassing the CLM API endpoints.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add backend to path
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/backend')

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Configure logging - NO EMOJIS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def trigger_retraining():
    """Trigger manual retraining of XGBoost and LightGBM models."""
    
    # Database connection - use internal postgres hostname
    db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://quantum_user:quantum_pass@postgres:5432/quantum_trader")
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    logger.info("="*60)
    logger.info("MANUAL RETRAINING TRIGGER - XGBoost + LightGBM")
    logger.info("="*60)
    logger.info("Using testnet historical data from database")
    
    try:
        async with async_session() as session:
            # Import training components
            from backend.domains.learning.model_training import (
                train_xgboost,
                train_lightgbm,
                TrainingConfig
            )
            from backend.domains.learning.data_pipeline import (
                HistoricalDataFetcher,
                FeatureEngineer
            )
            
            # Initialize data pipeline
            logger.info("\n[1/5] Initializing data pipeline...")
            data_fetcher = HistoricalDataFetcher(session)
            feature_engineer = FeatureEngineer()
            
            # Fetch historical data (90 days)
            logger.info("\n[2/5] Fetching historical trade data (90 days)...")
            
            query = """
            SELECT 
                symbol,
                entry_price,
                exit_price,
                pnl,
                pnl_percent,
                side,
                leverage,
                size_usdt,
                open_time,
                close_time,
                holding_time_seconds,
                exit_reason
            FROM positions
            WHERE close_time IS NOT NULL
            AND close_time >= NOW() - INTERVAL '90 days'
            ORDER BY close_time ASC
            """
            
            result = await session.fetch_all(query)
            if not result:
                logger.error("No historical data found. Cannot train models.")
                return False
            
            raw_data = pd.DataFrame([dict(row) for row in result])
            logger.info(f"Fetched {len(raw_data)} closed positions from database")
            
            if len(raw_data) < 50:
                logger.error(f"Insufficient data: {len(raw_data)} samples (need 50+)")
                return False
            
            logger.info(f"Fetched {len(raw_data)} historical positions")
            
            # Engineer features
            logger.info("\n[3/5] Engineering features...")
            X_train, y_train = feature_engineer.engineer_features(raw_data)
            
            if X_train is None or len(X_train) < 50:
                logger.error(f"Feature engineering failed or insufficient features: {len(X_train) if X_train is not None else 0}")
                return False
            
            logger.info(f"Features engineered: {X_train.shape[1]} features, {len(X_train)} samples")
            
            # Split into train/val
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            logger.info(f"Train samples: {len(X_train_split)}, Validation samples: {len(X_val)}")
            
            # Training configuration
            config = TrainingConfig(
                task="regression",
                cv_folds=5,
                xgb_n_estimators=100,
                lgb_n_estimators=100,
                random_state=42
            )
            
            # Train XGBoost
            logger.info("\n[4/5] Training XGBoost model...")
            xgb_model, xgb_metrics = train_xgboost(
                X_train_split, y_train_split, X_val, y_val, config
            )
            
            logger.info("XGBoost training completed:")
            logger.info(f"  Train R2: {xgb_metrics.get('train_r2', 0):.4f}")
            logger.info(f"  Test R2: {xgb_metrics.get('test_r2', 0):.4f}")
            logger.info(f"  Test MAE: {xgb_metrics.get('test_mae', 0):.4f}")
            
            # Train LightGBM
            logger.info("\n[5/5] Training LightGBM model...")
            lgb_model, lgb_metrics = train_lightgbm(
                X_train_split, y_train_split, X_val, y_val, config
            )
            
            logger.info("LightGBM training completed:")
            logger.info(f"  Train R2: {lgb_metrics.get('train_r2', 0):.4f}")
            logger.info(f"  Test R2: {lgb_metrics.get('test_r2', 0):.4f}")
            logger.info(f"  Test MAE: {lgb_metrics.get('test_mae', 0):.4f}")
            
            # Save models
            logger.info("\nSaving trained models...")
            
            import joblib
            os.makedirs("models/xgboost", exist_ok=True)
            os.makedirs("models/lightgbm", exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            xgb_path = f"models/xgboost/xgb_model_{timestamp}.pkl"
            lgb_path = f"models/lightgbm/lgb_model_{timestamp}.pkl"
            
            joblib.dump(xgb_model, xgb_path)
            joblib.dump(lgb_model, lgb_path)
            
            logger.info(f"XGBoost saved: {xgb_path}")
            logger.info(f"LightGBM saved: {lgb_path}")
            
            logger.info("\n" + "="*60)
            logger.info("RETRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Models trained on {len(X_train)} samples")
            logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
            logger.info("="*60)
            
            return True
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure backend.domains.learning modules are available")
        return False
        
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    logger.info("Starting manual retraining script...")
    logger.info("This will train XGBoost and LightGBM with historical testnet data\n")
    
    success = asyncio.run(trigger_retraining())
    
    if success:
        logger.info("\nRetraining successful!")
        sys.exit(0)
    else:
        logger.error("\nRetraining failed!")
        sys.exit(1)
