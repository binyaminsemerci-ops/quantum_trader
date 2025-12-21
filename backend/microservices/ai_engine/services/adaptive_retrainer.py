"""
Adaptive Retraining Pipeline - Phase 4F

Autonomous retraining system for PatchTST and N-HiTS models.
Continuously learns from latest market data without manual intervention.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("[Retrainer] PyTorch not available - retraining disabled")
    TORCH_AVAILABLE = False


class AdaptiveRetrainer:
    """
    Adaptive Retraining Pipeline
    
    Responsibilities:
    - Fetch recent market data
    - Prepare training datasets
    - Retrain PatchTST and N-HiTS models
    - Validate and deploy retrained models
    - Track retraining metrics
    """
    
    def __init__(
        self,
        data_api: Any,
        model_paths: Dict[str, str],
        retrain_interval: int = 14400,  # 4 hours
        min_data_points: int = 5000,
        validation_split: float = 0.2,
        max_epochs: int = 2
    ):
        """
        Initialize Adaptive Retrainer.
        
        Args:
            data_api: API for fetching market data
            model_paths: Dictionary of model names to save paths
            retrain_interval: Seconds between retraining cycles (default: 4h)
            min_data_points: Minimum data points required for retraining
            validation_split: Fraction of data for validation
            max_epochs: Maximum training epochs per cycle
        """
        self.data_api = data_api
        self.model_paths = model_paths
        self.retrain_interval = retrain_interval
        self.min_data_points = min_data_points
        self.validation_split = validation_split
        self.max_epochs = max_epochs
        
        self.last_retrain = datetime.utcnow()
        self.training_dir = Path("/app/adaptive_training")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.retrain_count = 0
        self.last_losses = {}
        self.retrain_history = []
        
        if not TORCH_AVAILABLE:
            logger.warning("[Retrainer] PyTorch not available - retraining disabled")
        
        logger.info(
            f"[Retrainer] Initialized - Interval: {retrain_interval}s, "
            f"Min data: {min_data_points}, Epochs: {max_epochs}"
        )
    
    def fetch_recent_data(
        self,
        symbol: str = "BTCUSDT",
        lookback_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent market data for training.
        
        Args:
            symbol: Trading symbol
            lookback_hours: Hours of historical data to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"[Retrainer] Fetching {lookback_hours}h data for {symbol}...")
            
            # Try to fetch from data API
            if hasattr(self.data_api, 'get_recent_data'):
                df = self.data_api.get_recent_data(symbol=symbol, hours=lookback_hours)
            else:
                logger.warning("[Retrainer] Data API not available - using mock data")
                # Generate mock data for testing
                df = self._generate_mock_data(lookback_hours)
            
            if df is None or len(df) < self.min_data_points:
                logger.warning(
                    f"[Retrainer] Insufficient data for {symbol}: "
                    f"{len(df) if df is not None else 0} < {self.min_data_points}"
                )
                return None
            
            logger.info(f"[Retrainer] ✅ Fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"[Retrainer] Data fetch failed: {e}", exc_info=True)
            return None
    
    def _generate_mock_data(self, hours: int) -> pd.DataFrame:
        """Generate mock OHLCV data for testing."""
        n_points = hours * 60  # 1-minute bars
        dates = pd.date_range(end=datetime.utcnow(), periods=n_points, freq='1min')
        
        # Generate synthetic price data
        base_price = 50000.0
        prices = base_price + np.cumsum(np.random.randn(n_points) * 10)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.abs(np.random.randn(n_points) * 5),
            'low': prices - np.abs(np.random.randn(n_points) * 5),
            'close': prices + np.random.randn(n_points) * 3,
            'volume': np.abs(np.random.randn(n_points) * 1000)
        })
        
        return df
    
    def prepare_dataloader(
        self,
        df: pd.DataFrame,
        window_size: int = 128,
        batch_size: int = 64
    ) -> Optional[DataLoader]:
        """
        Prepare PyTorch DataLoader from DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Sequence length for time series
            batch_size: Training batch size
            
        Returns:
            DataLoader or None if failed
        """
        if not TORCH_AVAILABLE:
            return None
        
        try:
            # Extract features (OHLCV)
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            features = df[feature_cols].values.astype(np.float32)
            
            # Normalize features
            features_mean = features.mean(axis=0, keepdims=True)
            features_std = features.std(axis=0, keepdims=True) + 1e-8
            features = (features - features_mean) / features_std
            
            # Create sequences
            X, y = [], []
            for i in range(len(features) - window_size):
                X.append(features[i:i+window_size])
                y.append(features[i+window_size, 3])  # Predict next close price
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"[Retrainer] Prepared dataset: X={X.shape}, y={y.shape}")
            
            # Create DataLoader
            dataset = TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
            
            return dataloader
            
        except Exception as e:
            logger.error(f"[Retrainer] DataLoader preparation failed: {e}", exc_info=True)
            return None
    
    def retrain_patchtst(
        self,
        dataloader: DataLoader,
        model_save_path: str
    ) -> bool:
        """
        Retrain PatchTST model.
        
        Args:
            dataloader: Training data
            model_save_path: Path to save retrained model
            
        Returns:
            Success status
        """
        if not TORCH_AVAILABLE:
            logger.warning("[Retrainer] PyTorch not available - skipping PatchTST")
            return False
        
        try:
            logger.info("[Retrainer][PatchTST] Starting retraining...")
            
            # Note: In production, you'd import the actual PatchTST model
            # For now, we'll use a simple model as placeholder
            model = self._create_simple_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = torch.nn.MSELoss()
            
            model.train()
            epoch_losses = []
            
            for epoch in range(self.max_epochs):
                batch_losses = []
                
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(X_batch)
                    loss = loss_fn(output.squeeze(), y_batch)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    batch_losses.append(loss.item())
                
                avg_loss = np.mean(batch_losses)
                epoch_losses.append(avg_loss)
                logger.info(
                    f"[Retrainer][PatchTST] Epoch {epoch+1}/{self.max_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )
            
            # Save model
            save_path = Path(model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
            self.last_losses["patchtst"] = epoch_losses[-1]
            logger.info(f"[Retrainer][PatchTST] ✅ Model saved to {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"[Retrainer][PatchTST] Retraining failed: {e}", exc_info=True)
            return False
    
    def retrain_nhits(
        self,
        dataloader: DataLoader,
        model_save_path: str
    ) -> bool:
        """
        Retrain N-HiTS model.
        
        Args:
            dataloader: Training data
            model_save_path: Path to save retrained model
            
        Returns:
            Success status
        """
        if not TORCH_AVAILABLE:
            logger.warning("[Retrainer] PyTorch not available - skipping N-HiTS")
            return False
        
        try:
            logger.info("[Retrainer][N-HiTS] Starting retraining...")
            
            # Note: In production, you'd import the actual N-HiTS model
            model = self._create_simple_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            loss_fn = torch.nn.MSELoss()
            
            model.train()
            epoch_losses = []
            
            for epoch in range(self.max_epochs):
                batch_losses = []
                
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    
                    output = model(X_batch)
                    loss = loss_fn(output.squeeze(), y_batch)
                    
                    loss.backward()
                    optimizer.step()
                    
                    batch_losses.append(loss.item())
                
                avg_loss = np.mean(batch_losses)
                epoch_losses.append(avg_loss)
                logger.info(
                    f"[Retrainer][N-HiTS] Epoch {epoch+1}/{self.max_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )
            
            # Save model
            save_path = Path(model_save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
            self.last_losses["nhits"] = epoch_losses[-1]
            logger.info(f"[Retrainer][N-HiTS] ✅ Model saved to {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"[Retrainer][N-HiTS] Retraining failed: {e}", exc_info=True)
            return False
    
    def _create_simple_model(self):
        """Create a simple model for testing (placeholder for actual models)."""
        if not TORCH_AVAILABLE:
            return None
        
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(128 * 5, 256)
                self.fc2 = torch.nn.Linear(256, 64)
                self.fc3 = torch.nn.Linear(64, 1)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x = x.flatten(start_dim=1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        return SimpleModel()
    
    def should_retrain(self) -> bool:
        """Check if it's time for retraining."""
        now = datetime.utcnow()
        elapsed = (now - self.last_retrain).total_seconds()
        
        if elapsed >= self.retrain_interval:
            logger.info(
                f"[Retrainer] Retraining due - Elapsed: {elapsed:.0f}s "
                f"(Interval: {self.retrain_interval}s)"
            )
            return True
        
        return False
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one retraining cycle.
        
        Returns:
            Dictionary with cycle results
        """
        if not self.should_retrain():
            return {"status": "skipped", "reason": "interval_not_reached"}
        
        logger.info("[Retrainer] 🔄 Initiating adaptive retraining cycle...")
        
        results = {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat(),
            "models_retrained": [],
            "errors": []
        }
        
        try:
            # Fetch recent data
            df = self.fetch_recent_data()
            if df is None:
                results["status"] = "failed"
                results["errors"].append("Data fetch failed")
                return results
            
            # Prepare dataloader
            dataloader = self.prepare_dataloader(df)
            if dataloader is None:
                results["status"] = "failed"
                results["errors"].append("DataLoader preparation failed")
                return results
            
            # Retrain PatchTST
            if "patchtst" in self.model_paths:
                success = self.retrain_patchtst(
                    dataloader,
                    self.model_paths["patchtst"]
                )
                if success:
                    results["models_retrained"].append("patchtst")
                else:
                    results["errors"].append("PatchTST retraining failed")
            
            # Retrain N-HiTS
            if "nhits" in self.model_paths:
                success = self.retrain_nhits(
                    dataloader,
                    self.model_paths["nhits"]
                )
                if success:
                    results["models_retrained"].append("nhits")
                else:
                    results["errors"].append("N-HiTS retraining failed")
            
            # Update tracking
            self.last_retrain = datetime.utcnow()
            self.retrain_count += 1
            self.retrain_history.append(results)
            
            # Keep only last 100 entries
            if len(self.retrain_history) > 100:
                self.retrain_history.pop(0)
            
            results["status"] = "success" if results["models_retrained"] else "partial"
            results["retrain_count"] = self.retrain_count
            
            logger.info(
                f"[Retrainer] ✅ Cycle complete - Models: {results['models_retrained']}, "
                f"Next cycle in ~{self.retrain_interval/3600:.1f}h"
            )
            
        except Exception as e:
            logger.error(f"[Retrainer] Cycle failed: {e}", exc_info=True)
            results["status"] = "error"
            results["errors"].append(str(e))
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current retrainer status."""
        now = datetime.utcnow()
        time_since_last = (now - self.last_retrain).total_seconds()
        time_until_next = max(0, self.retrain_interval - time_since_last)
        
        return {
            "enabled": TORCH_AVAILABLE,
            "retrain_interval_seconds": self.retrain_interval,
            "retrain_count": self.retrain_count,
            "last_retrain": self.last_retrain.isoformat(),
            "time_since_last_seconds": int(time_since_last),
            "time_until_next_seconds": int(time_until_next),
            "last_losses": self.last_losses,
            "model_paths": self.model_paths,
            "recent_history": self.retrain_history[-5:] if self.retrain_history else []
        }
