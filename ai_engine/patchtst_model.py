"""
PatchTST (Patch Time Series Transformer) - COMPLETE IMPLEMENTATION
State-of-the-art from 2023 (NeurIPS)

Key innovations:
1. Patch-based attention (like Vision Transformer for time series)
2. Channel independence (each feature processed separately)
3. Instance normalization (removes distribution shift)
4. Positional encoding (learnable + sinusoidal)
5. Efficient for long sequences (O(L/P)^2 vs O(L^2))

Performance:
- SOTA on 8/9 long-term forecasting benchmarks
- 80% faster training than Autoformer/FEDformer
- Better accuracy with fewer parameters

Paper: https://arxiv.org/abs/2211.14730
GitHub: https://github.com/yuqinie98/PatchTST
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    
    Removes distribution shift by normalizing each instance (batch sample).
    Critical for time series with non-stationary distributions (like crypto!)
    
    Steps:
    1. Normalize: (x - mean) / std
    2. Process with model
    3. Denormalize: x * std + mean
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, num_features]
            mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            # Normalize
            self.mean = x.mean(dim=1, keepdim=True)
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - self.mean) / self.stdev
            
            if self.affine:
                x = x * self.affine_weight.unsqueeze(0).unsqueeze(0)
                x = x + self.affine_bias.unsqueeze(0).unsqueeze(0)
        
        elif mode == 'denorm':
            # Denormalize
            if self.affine:
                x = x - self.affine_bias.unsqueeze(0).unsqueeze(0)
                x = x / (self.affine_weight.unsqueeze(0).unsqueeze(0) + self.eps * self.eps)
            
            x = x * self.stdev
            x = x + self.mean
        
        return x


class PatchEmbedding(nn.Module):
    """
    Convert time series into patches and embed them.
    
    Key insight: Local patterns matter more than individual timesteps.
    Patches capture local context, attention captures global context.
    
    Example: 120 timesteps with patch_len=12 -> 10 patches
    Each patch represents 12 hours of data (if 1h candles)
    
    Variants:
    - Linear projection (default, fastest)
    - Conv1d (better for overlapping patterns)
    """
    
    def __init__(
        self,
        num_features: int,
        patch_len: int,
        stride: int,
        d_model: int,
        dropout: float = 0.1,
        use_conv: bool = False
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_conv = use_conv
        
        if use_conv:
            # Convolutional patching (overlapping patches)
            self.patch_proj = nn.Conv1d(
                in_channels=num_features,
                out_channels=d_model,
                kernel_size=patch_len,
                stride=stride,
                padding=0,
                bias=True
            )
        else:
            # Linear patching (non-overlapping, faster)
            self.patch_proj = nn.Linear(patch_len * num_features, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, num_features]
        
        Returns:
            patches: [batch, num_patches, d_model]
        """
        batch_size, seq_len, num_features = x.shape
        
        if self.use_conv:
            # Conv expects [batch, features, seq_len]
            x = x.permute(0, 2, 1)
            x = self.patch_proj(x)  # [batch, d_model, num_patches]
            x = x.permute(0, 2, 1)  # [batch, num_patches, d_model]
        else:
            # Unfold into patches
            num_patches = (seq_len - self.patch_len) // self.stride + 1
            patches = []
            
            for i in range(num_patches):
                start_idx = i * self.stride
                end_idx = start_idx + self.patch_len
                patch = x[:, start_idx:end_idx, :]  # [batch, patch_len, num_features]
                patch = patch.reshape(batch_size, -1)  # [batch, patch_len * num_features]
                patches.append(patch)
            
            patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len * num_features]
            x = self.patch_proj(patches)  # [batch, num_patches, d_model]
        
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for patches.
    
    Combines:
    1. Sinusoidal encoding (Vaswani et al., 2017)
    2. Learnable encoding (Gehring et al., 2017)
    
    Both shown effective for time series transformers.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        learnable: bool = True
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable
        
        if learnable:
            # Learnable positional embedding
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        else:
            # Sinusoidal positional encoding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, d_model]
        """
        if self.learnable:
            x = x + self.pos_embedding[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)


class FlattenHead(nn.Module):
    """
    Flatten multivariate output and project to target dimension.
    
    For classification: projects to num_classes
    For forecasting: projects to forecast_horizon
    """
    
    def __init__(
        self,
        n_vars: int,
        nf: int,
        target_dim: int,
        head_dropout: float = 0.1
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf * n_vars, target_dim)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_vars, d_model] or [batch, num_patches, d_model]
        
        Returns:
            [batch, target_dim]
        """
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words
    
    Full implementation with all optimizations:
    - Reversible Instance Normalization (RevIN)
    - Channel independence (CI)
    - Patch-based attention
    - Learnable positional encoding
    - Multi-head self-attention
    - Feed-forward networks with GELU
    - Dropout regularization
    
    Architecture:
    1. RevIN normalization
    2. Patch embedding (per channel)
    3. Positional encoding
    4. Transformer encoder (N layers)
    5. Flatten head
    6. Classification/regression output
    7. RevIN denormalization (optional)
    """
    
    def __init__(
        self,
        # Input parameters
        input_size: int = 120,
        num_features: int = 14,
        
        # Patch parameters
        patch_len: int = 12,
        stride: int = 12,  # Non-overlapping by default
        
        # Model architecture
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        
        # Task parameters
        num_classes: int = 3,  # BUY, HOLD, SELL
        
        # Optimization parameters
        use_revin: bool = True,
        use_instance_norm: bool = True,
        channel_independent: bool = True,
        learnable_pe: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_classes = num_classes
        self.channel_independent = channel_independent
        self.use_revin = use_revin
        
        # Calculate number of patches
        self.num_patches = (input_size - patch_len) // stride + 1
        
        logger.info("[PatchTST] Building COMPLETE PatchTST Architecture:")
        logger.info(f"   Input: {input_size} timesteps x {num_features} features")
        logger.info(f"   Patch: length={patch_len}, stride={stride} -> {self.num_patches} patches")
        logger.info(f"   Model: d_model={d_model}, heads={nhead}, layers={num_layers}")
        logger.info(f"   Channel Independent: {channel_independent}")
        logger.info(f"   RevIN: {use_revin}")
        
        # RevIN normalization (removes distribution shift)
        if use_revin:
            self.revin = RevIN(num_features)
        else:
            self.revin = None
        
        # Patch embedding (per channel if channel_independent)
        if channel_independent:
            # Process each feature independently
            self.patch_embed = nn.ModuleList([
                PatchEmbedding(
                    num_features=1,
                    patch_len=patch_len,
                    stride=stride,
                    d_model=d_model,
                    dropout=dropout
                ) for _ in range(num_features)
            ])
        else:
            # Process all features together
            self.patch_embed = PatchEmbedding(
                num_features=num_features,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                dropout=dropout
            )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=self.num_patches,
            dropout=dropout,
            learnable=learnable_pe
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN (better for deep networks)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Classification head
        if channel_independent:
            # Aggregate across channels: [batch, num_features, d_model] -> [batch, num_classes]
            self.head = FlattenHead(
                n_vars=num_features,
                nf=d_model,  # Changed from num_patches - output is d_model per channel
                target_dim=num_classes,
                head_dropout=dropout
            )
        else:
            # Direct classification
            self.head = nn.Sequential(
                nn.Linear(d_model * self.num_patches, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            )
        
        # Instance normalization (optional, for stable training)
        self.instance_norm = nn.InstanceNorm1d(num_features) if use_instance_norm else None
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"[OK] PatchTST complete! Parameters: {num_params:,}")
    
    def _init_weights(self, module):
        """Initialize weights (Xavier for Linear, constant for norms)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with full PatchTST pipeline.
        
        Args:
            x: [batch, seq_len, num_features]
        
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.size(0)
        
        # 1. RevIN normalization (remove distribution shift)
        if self.revin is not None:
            x = self.revin(x, mode='norm')
        
        # 2. Instance normalization (optional)
        if self.instance_norm is not None:
            x = x.permute(0, 2, 1)  # [batch, features, seq_len]
            x = self.instance_norm(x)
            x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        
        # 3. Patch embedding
        if self.channel_independent:
            # Process each channel independently
            channel_embeddings = []
            for i in range(self.num_features):
                channel_x = x[:, :, i:i+1]  # [batch, seq_len, 1]
                channel_emb = self.patch_embed[i](channel_x)  # [batch, num_patches, d_model]
                channel_embeddings.append(channel_emb)
            
            # Stack: [batch, num_features, num_patches, d_model]
            x = torch.stack(channel_embeddings, dim=1)
            
            # Process each channel through transformer
            channel_outputs = []
            for i in range(self.num_features):
                channel_x = x[:, i, :, :]  # [batch, num_patches, d_model]
                
                # Add positional encoding
                channel_x = self.pos_encoder(channel_x)
                
                # Transformer encoding
                channel_x = self.transformer_encoder(channel_x)  # [batch, num_patches, d_model]
                
                # Global average pooling
                channel_x = channel_x.mean(dim=1)  # [batch, d_model]
                
                channel_outputs.append(channel_x)
            
            # Concatenate all channels
            x = torch.stack(channel_outputs, dim=1)  # [batch, num_features, d_model]
            
            # Flatten and classify
            logits = self.head(x)
        
        else:
            # Process all channels together
            x = self.patch_embed(x)  # [batch, num_patches, d_model]
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer encoding
            x = self.transformer_encoder(x)  # [batch, num_patches, d_model]
            
            # Flatten and classify
            x = x.reshape(batch_size, -1)  # [batch, num_patches * d_model]
            logits = self.head(x)
        
        return logits


def save_model(
    model: PatchTST,
    path: str,
    feature_mean: Optional[np.ndarray] = None,
    feature_std: Optional[np.ndarray] = None
):
    """Save PatchTST model with normalization stats."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'patch_len': model.patch_len,
        'num_features': model.num_features,
        'feature_mean': feature_mean,
        'feature_std': feature_std
    }
    torch.save(checkpoint, path)
    logger.info(f"[OK] Model saved to {path}")


def load_model(path: str, device: str = 'cpu') -> Tuple[PatchTST, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load PatchTST model with normalization stats."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model = PatchTST(
        input_size=checkpoint['input_size'],
        patch_len=checkpoint['patch_len'],
        num_features=checkpoint['num_features']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    feature_mean = checkpoint.get('feature_mean')
    feature_std = checkpoint.get('feature_std')
    
    logger.info(f"[OK] Model loaded from {path}")
    
    return model, feature_mean, feature_std


class PatchTSTTrainer:
    """Trainer for PatchTST model."""
    
    def __init__(self, model: PatchTST, device: str = 'cpu', learning_rate: float = 0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("[TARGET] PatchTST Trainer initialized")
        logger.info(f"   Device: {device}")
        logger.info(f"   Learning rate: {learning_rate}")
    
    def train_epoch(self, train_loader) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(sequences)
            loss = self.criterion(logits, targets.long())
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                logger.info(f"   Batch {batch_idx}/{num_batches}: Loss={loss.item():.4f}")
        
        return total_loss / num_batches
    
    def evaluate(self, val_loader) -> dict:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(sequences)
                loss = self.criterion(logits, targets.long())
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
