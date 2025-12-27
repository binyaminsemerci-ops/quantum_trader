"""
Model Training - Functions for training XGBoost, LightGBM, N-HiTS, PatchTST models.

Provides:
- XGBoost/LightGBM training with cross-validation
- N-HiTS/PatchTST time series training
- Comprehensive evaluation metrics
- Hyperparameter configuration via PolicyStore
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - N-HiTS and PatchTST training will fail")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for all model types."""
    
    # General
    task: str = "regression"  # "regression" or "classification"
    random_state: int = 42
    n_jobs: int = -1
    
    # Cross-validation
    cv_folds: int = 5
    
    # XGBoost
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    
    # LightGBM
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.1
    lgb_n_estimators: int = 100
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.1
    lgb_reg_lambda: float = 1.0
    
    # Deep Learning (N-HiTS, PatchTST)
    dl_epochs: int = 50
    dl_batch_size: int = 64
    dl_learning_rate: float = 0.001
    dl_hidden_dim: int = 128
    dl_dropout: float = 0.1
    dl_patience: int = 10  # Early stopping
    
    # N-HiTS specific
    nhits_n_blocks: int = 3
    nhits_mlp_units: List[int] = None
    
    # PatchTST specific
    patchtst_patch_len: int = 16
    patchtst_stride: int = 8
    patchtst_n_heads: int = 4
    patchtst_n_layers: int = 3
    
    def __post_init__(self):
        if self.nhits_mlp_units is None:
            self.nhits_mlp_units = [512, 512]


# ============================================================================
# XGBoost Training
# ============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainingConfig,
    cv: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train XGBoost model with optional cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration
        cv: Whether to use cross-validation
        
    Returns:
        (model, metrics_dict)
    """
    logger.info(f"Training XGBoost ({config.task}) with {len(X_train)} samples, {X_train.shape[1]} features")
    
    # XGBoost parameters
    params = {
        "max_depth": config.xgb_max_depth,
        "learning_rate": config.xgb_learning_rate,
        "n_estimators": config.xgb_n_estimators,
        "subsample": config.xgb_subsample,
        "colsample_bytree": config.xgb_colsample_bytree,
        "reg_alpha": config.xgb_reg_alpha,
        "reg_lambda": config.xgb_reg_lambda,
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "tree_method": "hist",
    }
    
    if config.task == "classification":
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        model = xgb.XGBClassifier(**params)
    else:
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"
        model = xgb.XGBRegressor(**params)
    
    # Cross-validation
    cv_scores = {}
    if cv and len(X_train) > 1000:
        logger.info(f"Running {config.cv_folds}-fold cross-validation...")
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        cv_train_scores = []
        cv_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_model = model.__class__(**params)
            fold_model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )
            
            train_pred = fold_model.predict(X_fold_train)
            val_pred = fold_model.predict(X_fold_val)
            
            if config.task == "regression":
                train_rmse = np.sqrt(mean_squared_error(y_fold_train, train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
                cv_train_scores.append(train_rmse)
                cv_val_scores.append(val_rmse)
            else:
                train_acc = accuracy_score(y_fold_train, train_pred)
                val_acc = accuracy_score(y_fold_val, val_pred)
                cv_train_scores.append(train_acc)
                cv_val_scores.append(val_acc)
        
        cv_scores = {
            "cv_train_mean": np.mean(cv_train_scores),
            "cv_train_std": np.std(cv_train_scores),
            "cv_val_mean": np.mean(cv_val_scores),
            "cv_val_std": np.std(cv_val_scores),
        }
        logger.info(f"CV results: train={cv_scores['cv_train_mean']:.4f}±{cv_scores['cv_train_std']:.4f}, "
                   f"val={cv_scores['cv_val_mean']:.4f}±{cv_scores['cv_val_std']:.4f}")
    
    # Train final model on full training set
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, config.task)
    metrics.update(cv_scores)
    
    logger.info(f"XGBoost training complete: val_metric={metrics.get('val_rmse' if config.task == 'regression' else 'val_accuracy', 0):.4f}")
    
    return model, metrics


# ============================================================================
# LightGBM Training
# ============================================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainingConfig,
    cv: bool = True,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train LightGBM model with optional cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration
        cv: Whether to use cross-validation
        
    Returns:
        (model, metrics_dict)
    """
    logger.info(f"Training LightGBM ({config.task}) with {len(X_train)} samples, {X_train.shape[1]} features")
    
    # LightGBM parameters
    params = {
        "num_leaves": config.lgb_num_leaves,
        "learning_rate": config.lgb_learning_rate,
        "n_estimators": config.lgb_n_estimators,
        "subsample": config.lgb_subsample,
        "colsample_bytree": config.lgb_colsample_bytree,
        "reg_alpha": config.lgb_reg_alpha,
        "reg_lambda": config.lgb_reg_lambda,
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "verbose": -1,
    }
    
    if config.task == "classification":
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
        model = lgb.LGBMClassifier(**params)
    else:
        params["objective"] = "regression"
        params["metric"] = "rmse"
        model = lgb.LGBMRegressor(**params)
    
    # Cross-validation
    cv_scores = {}
    if cv and len(X_train) > 1000:
        logger.info(f"Running {config.cv_folds}-fold cross-validation...")
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        cv_train_scores = []
        cv_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_model = model.__class__(**params)
            fold_model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
            )
            
            train_pred = fold_model.predict(X_fold_train)
            val_pred = fold_model.predict(X_fold_val)
            
            if config.task == "regression":
                train_rmse = np.sqrt(mean_squared_error(y_fold_train, train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_fold_val, val_pred))
                cv_train_scores.append(train_rmse)
                cv_val_scores.append(val_rmse)
            else:
                train_acc = accuracy_score(y_fold_train, train_pred)
                val_acc = accuracy_score(y_fold_val, val_pred)
                cv_train_scores.append(train_acc)
                cv_val_scores.append(val_acc)
        
        cv_scores = {
            "cv_train_mean": np.mean(cv_train_scores),
            "cv_train_std": np.std(cv_train_scores),
            "cv_val_mean": np.mean(cv_val_scores),
            "cv_val_std": np.std(cv_val_scores),
        }
        logger.info(f"CV results: train={cv_scores['cv_train_mean']:.4f}±{cv_scores['cv_train_std']:.4f}, "
                   f"val={cv_scores['cv_val_mean']:.4f}±{cv_scores['cv_val_std']:.4f}")
    
    # Train final model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val, config.task)
    metrics.update(cv_scores)
    
    logger.info(f"LightGBM training complete: val_metric={metrics.get('val_rmse' if config.task == 'regression' else 'val_accuracy', 0):.4f}")
    
    return model, metrics


# ============================================================================
# N-HiTS Training (Neural Hierarchical Interpolation for Time Series)
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 64,
        horizon: int = 12,
    ):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.horizon = horizon
    
    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y_seq = self.y[idx + self.seq_len:idx + self.seq_len + self.horizon]
        
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)


class NHiTSModel(nn.Module):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.
    
    Simplified implementation focusing on multi-rate processing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_blocks: int = 3,
        mlp_units: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if mlp_units is None:
            mlp_units = [512, 512]
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_blocks = n_blocks
        
        # Multi-rate blocks
        self.blocks = nn.ModuleList([
            self._create_block(input_dim, hidden_dim, output_dim, mlp_units, dropout)
            for _ in range(n_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(output_dim * n_blocks, output_dim)
    
    def _create_block(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mlp_units: List[int],
        dropout: float,
    ) -> nn.Sequential:
        """Create a single N-HiTS block."""
        layers = []
        
        layers.append(nn.Linear(input_dim, mlp_units[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for i in range(len(mlp_units) - 1):
            layers.append(nn.Linear(mlp_units[i], mlp_units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(mlp_units[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        
        block_outputs = []
        for block in self.blocks:
            block_out = block(x)
            block_outputs.append(block_out)
        
        # Concatenate all block outputs
        concatenated = torch.cat(block_outputs, dim=-1)
        output = self.output_layer(concatenated)
        
        return output


def train_nhits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainingConfig,
    seq_len: int = 64,
    horizon: int = 12,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train N-HiTS model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration
        seq_len: Input sequence length
        horizon: Forecast horizon
        
    Returns:
        (model, metrics_dict)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for N-HiTS training")
    
    logger.info(f"Training N-HiTS with {len(X_train)} samples, seq_len={seq_len}, horizon={horizon}")
    
    # Prepare datasets
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_val_np = X_val.values
    y_val_np = y_val.values
    
    train_dataset = TimeSeriesDataset(X_train_np, y_train_np, seq_len, horizon)
    val_dataset = TimeSeriesDataset(X_val_np, y_val_np, seq_len, horizon)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dl_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dl_batch_size,
        shuffle=False,
    )
    
    # Initialize model
    input_dim = seq_len * X_train.shape[1]
    model = NHiTSModel(
        input_dim=input_dim,
        hidden_dim=config.dl_hidden_dim,
        output_dim=horizon,
        n_blocks=config.nhits_n_blocks,
        mlp_units=config.nhits_mlp_units,
        dropout=config.dl_dropout,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.dl_learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(config.dl_epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config.dl_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.dl_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    
    logger.info(f"N-HiTS training complete: val_loss={val_loss:.4f}")
    
    return model, metrics


# ============================================================================
# PatchTST Training (Patch Time Series Transformer)
# ============================================================================

class PatchTSTModel(nn.Module):
    """
    PatchTST: Patch Time Series Transformer.
    
    Simplified implementation using patching and transformer encoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        patch_len: int = 16,
        stride: int = 8,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.patch_len = patch_len
        self.stride = stride
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_len * input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))  # Max 100 patches
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.size()
        
        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]
            patch = patch.reshape(batch_size, -1)  # Flatten patch
            patches.append(patch)
        
        patches = torch.stack(patches, dim=1)  # (batch, n_patches, patch_len * features)
        
        # Embed patches
        embedded = self.patch_embedding(patches)  # (batch, n_patches, hidden_dim)
        
        # Add positional encoding
        n_patches = embedded.size(1)
        embedded = embedded + self.pos_encoding[:, :n_patches, :]
        
        # Transformer
        transformed = self.transformer(embedded)  # (batch, n_patches, hidden_dim)
        
        # Global average pooling
        pooled = transformed.mean(dim=1)  # (batch, hidden_dim)
        
        # Output projection
        output = self.output_proj(pooled)  # (batch, output_dim)
        
        return output


def train_patchtst(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: TrainingConfig,
    seq_len: int = 64,
    horizon: int = 12,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train PatchTST model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Training configuration
        seq_len: Input sequence length
        horizon: Forecast horizon
        
    Returns:
        (model, metrics_dict)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PatchTST training")
    
    logger.info(f"Training PatchTST with {len(X_train)} samples, seq_len={seq_len}, horizon={horizon}")
    
    # Prepare datasets
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_val_np = X_val.values
    y_val_np = y_val.values
    
    train_dataset = TimeSeriesDataset(X_train_np, y_train_np, seq_len, horizon)
    val_dataset = TimeSeriesDataset(X_val_np, y_val_np, seq_len, horizon)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dl_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dl_batch_size,
        shuffle=False,
    )
    
    # Initialize model
    model = PatchTSTModel(
        input_dim=X_train.shape[1],
        hidden_dim=config.dl_hidden_dim,
        output_dim=horizon,
        patch_len=config.patchtst_patch_len,
        stride=config.patchtst_stride,
        n_heads=config.patchtst_n_heads,
        n_layers=config.patchtst_n_layers,
        dropout=config.dl_dropout,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.dl_learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(config.dl_epochs):
        model.train()
        train_losses = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config.dl_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.dl_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    
    logger.info(f"PatchTST training complete: val_loss={val_loss:.4f}")
    
    return model, metrics


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str = "regression",
) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Metrics:
    - Regression: RMSE, MAE, R², directional accuracy, Sharpe ratio
    - Classification: Accuracy, Precision, Recall, F1
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        task: "regression" or "classification"
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    if task == "regression":
        # RMSE
        metrics["train_rmse"] = np.sqrt(mean_squared_error(y_train, y_train_pred))
        metrics["val_rmse"] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # MAE
        metrics["train_mae"] = mean_absolute_error(y_train, y_train_pred)
        metrics["val_mae"] = mean_absolute_error(y_val, y_val_pred)
        
        # R²
        metrics["train_r2"] = r2_score(y_train, y_train_pred)
        metrics["val_r2"] = r2_score(y_val, y_val_pred)
        
        # Directional accuracy
        train_direction = np.sign(y_train) == np.sign(y_train_pred)
        val_direction = np.sign(y_val) == np.sign(y_val_pred)
        metrics["train_directional_accuracy"] = train_direction.mean()
        metrics["val_directional_accuracy"] = val_direction.mean()
        
        # Sharpe ratio (assuming returns)
        if np.std(y_val_pred) > 0:
            sharpe = np.mean(y_val_pred) / np.std(y_val_pred)
            metrics["val_sharpe"] = sharpe * np.sqrt(252)  # Annualized
        else:
            metrics["val_sharpe"] = 0.0
    
    else:  # classification
        # Accuracy
        metrics["train_accuracy"] = accuracy_score(y_train, y_train_pred)
        metrics["val_accuracy"] = accuracy_score(y_val, y_val_pred)
        
        # Precision, Recall, F1
        metrics["train_precision"] = precision_score(y_train, y_train_pred, zero_division=0)
        metrics["val_precision"] = precision_score(y_val, y_val_pred, zero_division=0)
        
        metrics["train_recall"] = recall_score(y_train, y_train_pred, zero_division=0)
        metrics["val_recall"] = recall_score(y_val, y_val_pred, zero_division=0)
        
        metrics["train_f1"] = f1_score(y_train, y_train_pred, zero_division=0)
        metrics["val_f1"] = f1_score(y_val, y_val_pred, zero_division=0)
    
    return metrics


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: XGBoost or LightGBM model
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances.tolist()))
    else:
        logger.warning("Model does not have feature_importances_")
        return {}
