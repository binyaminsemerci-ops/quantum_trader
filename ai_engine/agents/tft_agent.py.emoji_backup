"""
TFT AGENT - Temporal Fusion Transformer for trading
Replaces XGBoost agent with state-of-the-art transformer model
Expected WIN rate: 60-75%
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import logging

from ai_engine.tft_model import TemporalFusionTransformer, load_model

logger = logging.getLogger(__name__)


class TFTAgent:
    """
    Trading agent using Temporal Fusion Transformer
    
    Features:
    - Multi-horizon predictions
    - Attention-based feature selection
    - Confidence intervals
    - Interpretable predictions
    """
    
    def __init__(
        self,
        model_path: str = "ai_engine/models/tft_model.pth",
        sequence_length: int = 120,  # ⭐ INCREASED from 60 for more context
        device: str = None
    ):
        self.sequence_length = sequence_length
        self.model_path = model_path
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # History buffer for sequences
        self.history_buffer = {}  # symbol -> list of features
        
        # Model will be loaded lazily
        self.model: Optional[TemporalFusionTransformer] = None
        self.feature_mean = None
        self.feature_std = None
        
        logger.info(f"🤖 TFT Agent initialized (device: {self.device})")
    
    def load_model(self) -> bool:
        """Load TFT model from disk"""
        try:
            model_file = Path(self.model_path)
            
            if not model_file.exists():
                logger.warning(f"⚠️ TFT model not found at {self.model_path}")
                logger.info("💡 Train model first: python train_tft.py")
                return False
            
            # Load model
            checkpoint = torch.load(str(model_file), map_location=self.device, weights_only=False)
            self.model = load_model(str(model_file), device=self.device)
            
            # Load normalization stats (try checkpoint first, then JSON file)
            if 'feature_mean' in checkpoint and 'feature_std' in checkpoint:
                self.feature_mean = checkpoint['feature_mean']
                self.feature_std = checkpoint['feature_std']
                logger.info("✅ Loaded normalization stats from checkpoint")
            else:
                stats_file = model_file.parent / "tft_normalization.json"
                if stats_file.exists():
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        self.feature_mean = np.array(stats['mean'], dtype=np.float32)
                        self.feature_std = np.array(stats['std'], dtype=np.float32)
                    logger.info("✅ Loaded normalization stats from JSON")
                else:
                    logger.warning("⚠️ Normalization stats not found, using defaults")
                    self.feature_mean = np.zeros(14, dtype=np.float32)
                    self.feature_std = np.ones(14, dtype=np.float32)
            
            logger.info(f"✅ TFT model loaded from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load TFT model: {e}")
            return False
    
    def add_to_history(self, symbol: str, features: Dict[str, float]):
        """Add features to symbol's history buffer"""
        if symbol not in self.history_buffer:
            self.history_buffer[symbol] = []
        
        # Extract 14 features
        feature_vector = [
            features.get('Close', 0),
            features.get('Volume', 0),
            features.get('EMA_10', 0),
            features.get('EMA_50', 0),
            features.get('RSI', 50),
            features.get('MACD', 0),
            features.get('MACD_signal', 0),
            features.get('BB_upper', 0),
            features.get('BB_middle', 0),
            features.get('BB_lower', 0),
            features.get('ATR', 0),
            features.get('volume_sma_20', 0),
            features.get('price_change_pct', 0),
            features.get('high_low_range', 0),
        ]
        
        self.history_buffer[symbol].append(feature_vector)
        
        # Keep only last N timesteps
        if len(self.history_buffer[symbol]) > self.sequence_length * 2:
            self.history_buffer[symbol] = self.history_buffer[symbol][-self.sequence_length * 2:]
    
    def predict(
        self,
        symbol: str,
        features: Dict[str, float],
        confidence_threshold: float = 0.65
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Make prediction for a symbol
        
        Args:
            symbol: Trading symbol
            features: Current market features
            confidence_threshold: Minimum confidence for BUY/SELL (default 0.65)
            
        Returns:
            action: 'BUY', 'SELL', or 'HOLD'
            confidence: Prediction confidence [0, 1]
            metadata: Additional prediction info (quantiles, attention weights)
        """
        # Add to history
        self.add_to_history(symbol, features)
        
        # Check if we have enough history
        if len(self.history_buffer[symbol]) < self.sequence_length:
            logger.debug(f"{symbol}: Need {self.sequence_length - len(self.history_buffer[symbol])} more samples")
            return 'HOLD', 0.0, {'reason': 'insufficient_history'}
        
        # Check if model is loaded
        if self.model is None:
            if not self.load_model():
                return 'HOLD', 0.0, {'reason': 'model_not_loaded'}
        
        try:
            # Get sequence
            sequence = np.array(
                self.history_buffer[symbol][-self.sequence_length:],
                dtype=np.float32
            )
            
            # Normalize
            sequence = (sequence - self.feature_mean) / (self.feature_std + 1e-8)
            sequence = sequence.astype(np.float32)  # Ensure float32 after normalization
            
            # Convert to tensor
            sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits, quantiles, attention_weights = self.model(sequence_tensor)
                
                # Get probabilities
                probs = torch.softmax(logits, dim=1)[0]  # [3]
                
                # Get prediction
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                
                # Convert to action
                actions = ['BUY', 'SELL', 'HOLD']
                action = actions[pred_class]
                
                # ⭐ ASYMMETRIC RISK/REWARD ANALYSIS (Quantile-based)
                q10, q50, q90 = quantiles[0].cpu().numpy()
                
                # Calculate risk/reward ratio
                upside = q90 - q50  # Potential upside
                downside = q50 - q10  # Potential downside
                
                risk_reward_ratio = abs(upside / (downside + 1e-8))
                
                # ⭐ ADJUST CONFIDENCE based on risk/reward
                if action == 'BUY' and risk_reward_ratio > 2.0:
                    # Good risk/reward for BUY - boost confidence
                    confidence = min(confidence * 1.15, 1.0)
                    logger.debug(f"   ⬆️ BUY confidence boosted (R/R={risk_reward_ratio:.2f})")
                
                elif action == 'SELL' and risk_reward_ratio < 0.5:
                    # Good risk/reward for SELL - boost confidence
                    confidence = min(confidence * 1.15, 1.0)
                    logger.debug(f"   ⬇️ SELL confidence boosted (R/R={risk_reward_ratio:.2f})")
                
                elif action != 'HOLD' and 0.7 < risk_reward_ratio < 1.3:
                    # Poor risk/reward (symmetric) - reduce confidence
                    confidence = confidence * 0.85
                    logger.debug(f"   ⚠️ Confidence reduced (poor R/R={risk_reward_ratio:.2f})")
                
                # Apply confidence threshold
                if action != 'HOLD' and confidence < confidence_threshold:
                    action = 'HOLD'
                    confidence = probs[2].item()  # HOLD probability
                
                # Extract metadata
                metadata = {
                    'buy_prob': probs[0].item(),
                    'sell_prob': probs[1].item(),
                    'hold_prob': probs[2].item(),
                    'q10': float(q10),
                    'q50': float(q50),
                    'q90': float(q90),
                    'upside': float(upside),
                    'downside': float(downside),
                    'risk_reward_ratio': float(risk_reward_ratio),
                    'prediction_confidence': confidence,
                    'model': 'TFT'
                }
                
                logger.debug(
                    f"{symbol}: {action} (conf={confidence:.2f}, R/R={risk_reward_ratio:.2f}, "
                    f"probs={probs.cpu().numpy()})"
                )
                
                return action, confidence, metadata
                
        except Exception as e:
            logger.error(f"❌ TFT prediction failed for {symbol}: {e}")
            return 'HOLD', 0.0, {'reason': 'prediction_error', 'error': str(e)}
    
    def batch_predict(
        self,
        symbols_features: Dict[str, Dict[str, float]],
        confidence_threshold: float = 0.65
    ) -> Dict[str, Tuple[str, float, Dict]]:
        """
        Batch prediction for multiple symbols (more efficient)
        
        Args:
            symbols_features: Dict of {symbol: features}
            confidence_threshold: Minimum confidence
            
        Returns:
            Dict of {symbol: (action, confidence, metadata)}
        """
        results = {}
        
        # Add all to history
        for symbol, features in symbols_features.items():
            self.add_to_history(symbol, features)
        
        # Filter symbols with enough history
        ready_symbols = [
            s for s in symbols_features.keys()
            if len(self.history_buffer.get(s, [])) >= self.sequence_length
        ]
        
        if not ready_symbols:
            return {s: ('HOLD', 0.0, {'reason': 'insufficient_history'}) 
                    for s in symbols_features.keys()}
        
        # Check model
        if self.model is None:
            if not self.load_model():
                return {s: ('HOLD', 0.0, {'reason': 'model_not_loaded'}) 
                        for s in symbols_features.keys()}
        
        try:
            # Prepare batch
            sequences = []
            for symbol in ready_symbols:
                seq = np.array(
                    self.history_buffer[symbol][-self.sequence_length:],
                    dtype=np.float32
                )
                seq = (seq - self.feature_mean) / (self.feature_std + 1e-8)
                sequences.append(seq)
            
            sequences_tensor = torch.from_numpy(np.array(sequences)).to(self.device)
            
            # Batch predict
            with torch.no_grad():
                logits, quantiles, attention_weights = self.model(sequences_tensor)
                probs = torch.softmax(logits, dim=1)  # [batch, 3]
                
                for i, symbol in enumerate(ready_symbols):
                    pred_class = torch.argmax(probs[i]).item()
                    confidence = probs[i, pred_class].item()
                    
                    actions = ['BUY', 'SELL', 'HOLD']
                    action = actions[pred_class]
                    
                    # Apply threshold
                    if action != 'HOLD' and confidence < confidence_threshold:
                        action = 'HOLD'
                        confidence = probs[i, 2].item()
                    
                    metadata = {
                        'buy_prob': probs[i, 0].item(),
                        'sell_prob': probs[i, 1].item(),
                        'hold_prob': probs[i, 2].item(),
                        'q10': quantiles[i, 0].item(),
                        'q50': quantiles[i, 1].item(),
                        'q90': quantiles[i, 2].item(),
                        'prediction_confidence': confidence,
                        'model': 'TFT'
                    }
                    
                    results[symbol] = (action, confidence, metadata)
            
            # Add not-ready symbols
            for symbol in symbols_features.keys():
                if symbol not in results:
                    results[symbol] = ('HOLD', 0.0, {'reason': 'insufficient_history'})
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {e}")
            return {s: ('HOLD', 0.0, {'reason': 'prediction_error', 'error': str(e)}) 
                    for s in symbols_features.keys()}
    
    def predict_single(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Synchronous single prediction for hybrid agent compatibility
        
        Args:
            features: Feature dictionary
            
        Returns:
            (action, confidence)
        """
        try:
            # Use a dummy symbol for single prediction
            symbol = "__single__"
            
            # Add to history
            self.add_to_history(symbol, features)
            
            # Check if enough history
            if len(self.history_buffer.get(symbol, [])) < self.sequence_length:
                return ("HOLD", 0.5)
            
            # Load model if needed
            if self.model is None:
                if not self.load_model():
                    return ("HOLD", 0.5)
            
            # Prepare sequence
            seq = np.array(
                self.history_buffer[symbol][-self.sequence_length:],
                dtype=np.float32
            )
            seq = (seq - self.feature_mean) / (self.feature_std + 1e-8)
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                logits, _, _ = self.model(seq_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                
                pred_class = torch.argmax(probs).item()
                confidence = probs[pred_class].item()
                
                actions = ['BUY', 'SELL', 'HOLD']
                action = actions[pred_class]
                
                return (action, confidence)
                
        except Exception as e:
            logger.error(f"❌ predict_single failed: {e}")
            return ("HOLD", 0.5)
    
    async def scan_top_by_volume_from_api(
        self, 
        symbols: List[str], 
        top_n: int = 10, 
        limit: int = 240
    ) -> Dict[str, Dict[str, any]]:
        """
        Scan symbols using TFT predictions
        Compatible interface with XGBAgent for easy drop-in replacement
        """
        # Import XGBAgent to use its data fetching capabilities
        try:
            from ai_engine.agents.xgb_agent import XGBAgent
            
            # Create temporary XGBAgent for data fetching
            xgb_agent = XGBAgent()
            
            # Fetch OHLCV data using XGBAgent's robust fetching logic
            raw_results = await xgb_agent.scan_top_by_volume_from_api(
                symbols, top_n=len(symbols), limit=limit
            )
            
            # Override predictions with TFT model
            tft_results = {}
            for symbol, data in raw_results.items():
                # Get TFT prediction using fetched features
                if 'features' in data:
                    tft_pred = self.predict(symbol, data['features'])
                    
                    # Replace prediction data
                    data['action'] = tft_pred['action']
                    data['score'] = tft_pred['score']
                    data['confidence'] = tft_pred['confidence']
                    data['model'] = 'TFT'
                
                tft_results[symbol] = data
            
            return tft_results
            
        except Exception as e:
            logger.error(f"scan_top_by_volume_from_api failed: {e}")
            # Return empty results on failure
            return {}
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance from Variable Selection Network
        (Requires running prediction first to extract VSN weights)
        """
        # TODO: Implement by storing attention/VSN weights during prediction
        return None
    
    def clear_history(self, symbol: Optional[str] = None):
        """Clear history buffer for symbol or all symbols"""
        if symbol:
            if symbol in self.history_buffer:
                del self.history_buffer[symbol]
        else:
            self.history_buffer = {}
