"""
PHASE 4G: Model Validation Layer
Automatic quality control that validates retrained models before production deployment.

Functionality:
- Evaluates new vs old models on validation dataset (12h data)
- Measures MAPE, PnL, and Sharpe ratio
- Only promotes models with 3%+ MAPE improvement AND better Sharpe
- Logs all decisions for audit trail
- Automatically rolls back poor models
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelValidationLayer:
    """
    Validates retrained models before promoting them to production.
    
    Validation Criteria:
    - New MAPE must be < 97% of old MAPE (3%+ improvement)
    - New Sharpe ratio must be > old Sharpe ratio
    - Both conditions must be met for promotion
    
    Actions:
    - ACCEPT: Replace production model with new adaptive model
    - REJECT: Discard adaptive model, keep current production model
    """
    
    def __init__(self, model_paths, val_data_api):
        """
        Initialize Model Validation Layer.
        
        Args:
            model_paths: Dict mapping model names to production model paths
            val_data_api: Data API instance for fetching validation data
        """
        self.model_paths = model_paths
        self.data_api = val_data_api
        self.val_log = "/app/logs/model_validation.log"
        os.makedirs("/app/logs", exist_ok=True)
        logger.info("[PHASE 4G] Model Validation Layer initialized")

    def evaluate_model(self, model, X, y):
        """
        Evaluate a model on validation data.
        
        Args:
            model: PyTorch model to evaluate
            X: Input features (N, window_size, n_features)
            y: Target values (N,)
            
        Returns:
            (mape, pnl, sharpe): Tuple of evaluation metrics
        """
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
        
        # Calculate MAPE
        mape = np.mean(np.abs((y - preds) / (y + 1e-8)))
        
        # Calculate PnL (profit from predicting direction correctly)
        pnl = np.sum(np.diff(preds) * np.sign(np.diff(y)))
        
        # Calculate Sharpe ratio (risk-adjusted returns)
        returns = np.diff(preds)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        
        return mape, pnl, sharpe

    def validate(self, name, cls):
        """
        Validate a specific model's adaptive version against production.
        
        Args:
            name: Model name (e.g., "patchtst", "nhits")
            cls: Model class constructor
            
        Returns:
            bool: True if new model was promoted, False otherwise
        """
        try:
            # Fetch validation data (12 hours)
            logger.info(f"[Validator] Fetching validation data for {name}...")
            df = self.data_api.get_recent_data(symbol="BTCUSDT", hours=12)
            
            if df is None or len(df) < 200:
                logger.warning(f"[Validator] Insufficient validation data for {name}")
                return False
            
            # Prepare validation dataset
            features = df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
            X, y = [], []
            window = 128
            
            for i in range(len(features) - window):
                X.append(features[i:i+window])
                y.append(features[i+window, 3])  # Predict next close price
            
            X, y = np.array(X), np.array(y)
            logger.info(f"[Validator] Prepared {len(X)} validation samples for {name}")
            
            # Load production model
            old_model = cls()
            old_model_path = self.model_paths[name]
            if not os.path.exists(old_model_path):
                logger.warning(f"[Validator] No production model found at {old_model_path}")
                return False
            old_model.load_state_dict(torch.load(old_model_path))
            
            # Load adaptive (retrained) model
            new_path = old_model_path.replace(".pth", "_adaptive.pth")
            if not os.path.exists(new_path):
                logger.warning(f"[Validator] No adaptive model found for {name} at {new_path}")
                return False
            new_model = cls()
            new_model.load_state_dict(torch.load(new_path))
            
            # Evaluate both models
            logger.info(f"[Validator] Evaluating production {name} model...")
            old_mape, old_pnl, old_sharpe = self.evaluate_model(old_model, X, y)
            
            logger.info(f"[Validator] Evaluating adaptive {name} model...")
            new_mape, new_pnl, new_sharpe = self.evaluate_model(new_model, X, y)
            
            # Decision: New model must have 3%+ MAPE improvement AND better Sharpe
            mape_improvement = (old_mape - new_mape) / old_mape
            decision = (new_mape < old_mape * 0.97) and (new_sharpe > old_sharpe)
            
            # Log decision
            msg = (
                f"[Validator] {name}: "
                f"old(MAPE={old_mape:.4f}, PnL={old_pnl:.2f}, Sharpe={old_sharpe:.2f}) → "
                f"new(MAPE={new_mape:.4f}, PnL={new_pnl:.2f}, Sharpe={new_sharpe:.2f}) → "
                f"MAPE_improvement={mape_improvement*100:.1f}% → "
                f"{'✅ ACCEPT' if decision else '❌ REJECT'}"
            )
            logger.info(msg)
            
            # Write to audit log
            timestamp = datetime.utcnow().isoformat()
            with open(self.val_log, "a") as f:
                f.write(f"{timestamp} {msg}\n")
            
            # Execute decision
            if decision:
                # Promote adaptive model to production
                os.replace(new_path, old_model_path)
                logger.info(f"[Validator] ✅ Promoted new {name} model to production")
                return True
            else:
                # Discard adaptive model
                os.remove(new_path)
                logger.info(f"[Validator] ❌ Discarded {name} adaptive model (insufficient improvement)")
                return False
                
        except Exception as e:
            logger.error(f"[Validator] Validation error for {name}: {e}", exc_info=True)
            return False

    def run_validation_cycle(self):
        """
        Run validation for all retrained models.
        Called after adaptive retrainer completes a training cycle.
        """
        try:
            logger.info("[Validator] Starting validation cycle...")
            
            # Import model classes (deferred to avoid circular imports)
            from backend.microservices.ai_engine.models.patchtst import PatchTSTModel
            from backend.microservices.ai_engine.models.nhits import NHiTSModel
            
            # Validate PatchTST
            patchtst_promoted = self.validate("patchtst", PatchTSTModel)
            
            # Validate N-HiTS
            nhits_promoted = self.validate("nhits", NHiTSModel)
            
            # Summary
            logger.info(
                f"[Validator] Validation cycle complete: "
                f"PatchTST={'promoted' if patchtst_promoted else 'rejected'}, "
                f"NHiTS={'promoted' if nhits_promoted else 'rejected'}"
            )
            
            return {
                "patchtst": patchtst_promoted,
                "nhits": nhits_promoted
            }
            
        except Exception as e:
            logger.error(f"[Validator] Validation cycle error: {e}", exc_info=True)
            return {}

    def get_status(self):
        """
        Get validation layer status for health endpoint.
        
        Returns:
            dict: Status information
        """
        # Read recent validation log entries
        recent_validations = []
        if os.path.exists(self.val_log):
            try:
                with open(self.val_log, "r") as f:
                    lines = f.readlines()
                    recent_validations = [line.strip() for line in lines[-10:]]
            except Exception as e:
                logger.error(f"[Validator] Error reading validation log: {e}")
        
        return {
            "enabled": True,
            "validation_log_path": self.val_log,
            "recent_validations": recent_validations,
            "criteria": {
                "mape_improvement_required": "3%",
                "sharpe_improvement_required": True
            }
        }
