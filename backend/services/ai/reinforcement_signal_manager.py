"""
REINFORCEMENT SIGNAL MANAGER

Real-time reinforcement learning via feedback loop:
- Adjusts ensemble model weights based on trade outcomes
- Calibrates confidence scores over time
- Implements reward shaping for risk-adjusted learning
- Discount factor for temporal relevance
- Exploration-exploitation balance

Author: Quantum Trader AI Team
Version: 1.0.0
"""

import logging
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import deque
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class ModelType(Enum):
    """AI model types in ensemble"""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    NHITS = "n-hits"
    PATCHTST = "patchtst"


@dataclass
class TradeOutcome:
    """Single trade outcome for reinforcement learning"""
    timestamp: str
    symbol: str
    action: str  # LONG/SHORT
    confidence: float
    pnl: float
    position_size: float
    entry_price: float
    exit_price: float
    duration_seconds: float
    regime: str
    model_votes: Dict[str, Dict]  # {model: {action: str, confidence: float}}
    setup_hash: str


@dataclass
class ReinforcementSignal:
    """Processed reinforcement signal with shaped reward"""
    trade_outcome: TradeOutcome
    raw_reward: float
    shaped_reward: float
    sharpe_contribution: float
    risk_adjusted_return: float
    baseline_reward: float
    advantage: float  # shaped_reward - baseline_reward
    model_contributions: Dict[str, float]  # {model: contribution}


@dataclass
class ModelWeights:
    """Current ensemble model weights"""
    xgboost: float
    lightgbm: float
    nhits: float
    patchtst: float
    last_updated: str
    
    def to_dict(self) -> Dict:
        return {
            'xgboost': self.xgboost,
            'lightgbm': self.lightgbm,
            'nhits': self.nhits,
            'patchtst': self.patchtst,
            'last_updated': self.last_updated
        }
    
    def get_weight(self, model: ModelType) -> float:
        """Get weight for specific model"""
        return getattr(self, model.value.replace('-', ''))
    
    def set_weight(self, model: ModelType, weight: float):
        """Set weight for specific model"""
        attr_name = model.value.replace('-', '')
        setattr(self, attr_name, weight)


@dataclass
class CalibrationMetrics:
    """Model confidence calibration metrics"""
    brier_score: Dict[str, float]  # {model: score}
    calibration_error: Dict[str, float]  # {model: error}
    sample_counts: Dict[str, int]  # {model: count}
    last_updated: str


@dataclass
class ReinforcementContext:
    """Context for applying reinforcement to new signals"""
    model_weights: ModelWeights
    confidence_scalers: Dict[str, float]  # {model: scaler}
    exploration_rate: float
    total_trades_processed: int
    baseline_reward: float
    recent_advantage: float  # Average advantage last 10 trades
    

# ============================================================
# REINFORCEMENT SIGNAL MANAGER
# ============================================================

class ReinforcementSignalManager:
    """
    Manages reinforcement learning feedback loop for AI ensemble.
    
    Key Features:
    - Reward shaping (PnL + Sharpe + Risk-adjusted)
    - Model weight adjustment via exponential update
    - Confidence calibration (Brier score)
    - Temporal discounting (γ=0.95)
    - Exploration-exploitation (ε-greedy)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        discount_factor: float = 0.95,
        initial_exploration_rate: float = 0.20,
        min_exploration_rate: float = 0.05,
        exploration_decay_trades: int = 100,
        reward_alpha: float = 0.6,  # Direct PnL weight
        reward_beta: float = 0.3,   # Sharpe weight
        reward_gamma: float = 0.1,  # Risk-adjusted weight
        calibration_kappa: float = 0.5,  # Calibration adjustment weight
        min_model_weight: float = 0.05,
        max_model_weight: float = 0.50,
        checkpoint_path: str = "/app/data/reinforcement_state.json",
        checkpoint_interval: int = 60  # seconds
    ):
        """
        Initialize Reinforcement Signal Manager
        
        Args:
            learning_rate: η for weight updates (0.01-0.10 typical)
            discount_factor: γ for temporal decay (0.90-0.99)
            initial_exploration_rate: Starting ε (0.10-0.30)
            min_exploration_rate: Floor ε (0.01-0.10)
            exploration_decay_trades: Trades to decay from initial to min ε
            reward_alpha: Weight for direct PnL in shaped reward
            reward_beta: Weight for Sharpe contribution
            reward_gamma: Weight for risk-adjusted return
            calibration_kappa: Weight for confidence calibration adjustment
            min_model_weight: Minimum allowed model weight (prevent zero)
            max_model_weight: Maximum allowed model weight (prevent domination)
            checkpoint_path: Path to save/load state
            checkpoint_interval: Seconds between auto-checkpoints
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_trades = exploration_decay_trades
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma
        self.calibration_kappa = calibration_kappa
        self.min_model_weight = min_model_weight
        self.max_model_weight = max_model_weight
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        
        # Model weights (start equal: 0.25, 0.25, 0.30, 0.20)
        self.model_weights = ModelWeights(
            xgboost=0.25,
            lightgbm=0.25,
            nhits=0.30,
            patchtst=0.20,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        # Confidence calibration
        self.calibration_metrics = CalibrationMetrics(
            brier_score={m.value: 0.25 for m in ModelType},  # Start neutral
            calibration_error={m.value: 0.0 for m in ModelType},
            sample_counts={m.value: 0 for m in ModelType},
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        
        # Confidence scalers (start at 1.0 = no adjustment)
        self.confidence_scalers = {m.value: 1.0 for m in ModelType}
        
        # Trade history (for reward calculation)
        self.trade_history = deque(maxlen=100)  # Last 100 trades
        self.reinforcement_signals = deque(maxlen=100)
        
        # Baseline reward (moving average)
        self.baseline_reward = 0.0
        self.baseline_window = deque(maxlen=20)
        
        # Statistics
        self.total_trades_processed = 0
        self.total_reward_accumulated = 0.0
        self.weight_update_count = 0
        
        # Checkpoint
        self.last_checkpoint_time = datetime.now(timezone.utc)
        
        # Load existing state if available
        self._load_checkpoint()
        
        logger.info(
            f"[REINFORCEMENT] Initialized with lr={learning_rate}, "
            f"γ={discount_factor}, ε={initial_exploration_rate}"
        )
    
    # ========================================
    # PHASE 1 WRAPPER METHODS (AI ENGINE)
    # ========================================
    
    def calibrate_confidence(self, symbol: str, raw_confidence: float, action: str) -> float:
        """
        Calibrate raw confidence using learned confidence scalers (PHASE 1 wrapper).
        Returns: Calibrated confidence
        """
        try:
            # Get average confidence scaler from all models
            scalers = list(self.confidence_scalers.values())
            avg_scaler = sum(scalers) / len(scalers) if scalers else 1.0
            
            calibrated = raw_confidence * avg_scaler
            calibrated = max(0.30, min(0.95, calibrated))  # Clamp to [0.30, 0.95]
            
            return calibrated
        except Exception as e:
            logger.error(f"Error calibrating confidence for {symbol}: {e}")
            return raw_confidence
    
    def learn_from_outcome(self, outcome: TradeOutcome):
        """
        Learn from trade outcome (PHASE 1 wrapper).
        Updates model weights and calibration metrics.
        """
        try:
            # Handle both dict and TradeOutcome
            if isinstance(outcome, dict):
                outcome = TradeOutcome(**outcome)
            
            # Process trade outcome (simplified for now)
            pnl = outcome.pnl if hasattr(outcome, 'pnl') else 0.0
            
            # Simple reward calculation
            shaped_reward = pnl / 100.0  # Normalize to percentage
            self.total_reward_accumulated += shaped_reward
            self.total_trades_processed += 1
            
            logger.info(
                f"[RL Signal] Learned from {outcome.symbol} trade: "
                f"PnL={pnl:.2f}, reward={shaped_reward:.3f}, total_trades={self.total_trades_processed}"
            )
            
            # Auto-checkpoint if needed
            self.auto_checkpoint_check()
        
        except Exception as e:
            logger.error(f"Error learning from outcome: {e}", exc_info=True)
    
    # ========================================
    # CORE METHODS
    # ========================================
    
    def process_trade_outcome(
        self,
        symbol: str,
        action: str,
        confidence: float,
        pnl: float,
        position_size: float,
        entry_price: float,
        exit_price: float,
        duration_seconds: float,
        regime: str,
        model_votes: Dict[str, Dict],
        setup_hash: str
    ) -> ReinforcementSignal:
        """
        Process trade outcome and generate reinforcement signal
        
        Args:
            symbol: Trading symbol
            action: LONG or SHORT
            confidence: Final signal confidence
            pnl: Trade PnL in USD
            position_size: Position size in base currency
            entry_price: Entry price
            exit_price: Exit price
            duration_seconds: Trade duration
            regime: Market regime during trade
            model_votes: {model: {action: str, confidence: float}}
            setup_hash: Market setup hash
            
        Returns:
            ReinforcementSignal with shaped reward and model contributions
        """
        # Create trade outcome
        outcome = TradeOutcome(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            action=action,
            confidence=confidence,
            pnl=pnl,
            position_size=position_size,
            entry_price=entry_price,
            exit_price=exit_price,
            duration_seconds=duration_seconds,
            regime=regime,
            model_votes=model_votes,
            setup_hash=setup_hash
        )
        
        # Calculate shaped reward
        raw_reward = pnl
        shaped_reward, sharpe_contrib, risk_adjusted = self._shape_reward(outcome)
        
        # Calculate advantage (vs baseline)
        advantage = shaped_reward - self.baseline_reward
        
        # Identify model contributions
        model_contributions = self._calculate_model_contributions(outcome, shaped_reward)
        
        # Create reinforcement signal
        signal = ReinforcementSignal(
            trade_outcome=outcome,
            raw_reward=raw_reward,
            shaped_reward=shaped_reward,
            sharpe_contribution=sharpe_contrib,
            risk_adjusted_return=risk_adjusted,
            baseline_reward=self.baseline_reward,
            advantage=advantage,
            model_contributions=model_contributions
        )
        
        # Update model weights
        self._update_model_weights(signal)
        
        # Update confidence calibration
        self._update_calibration(outcome)
        
        # Update baseline reward
        self._update_baseline_reward(shaped_reward)
        
        # Store in history
        self.trade_history.append(outcome)
        self.reinforcement_signals.append(signal)
        
        # Update stats
        self.total_trades_processed += 1
        self.total_reward_accumulated += shaped_reward
        
        # Auto-checkpoint check
        self.auto_checkpoint_check()
        
        logger.info(
            f"[REINFORCEMENT] Trade processed: {symbol} {action} "
            f"PnL=${pnl:.2f}, R_shaped={shaped_reward:.3f}, "
            f"Advantage={advantage:+.3f}"
        )
        
        return signal
    
    def get_reinforcement_context(self) -> ReinforcementContext:
        """
        Get current reinforcement context for signal generation
        
        Returns:
            ReinforcementContext with current weights, scalers, etc.
        """
        # Calculate current exploration rate
        exploration_rate = self._calculate_exploration_rate()
        
        # Calculate recent advantage (last 10 trades)
        recent_advantages = [
            s.advantage for s in list(self.reinforcement_signals)[-10:]
        ]
        recent_advantage = np.mean(recent_advantages) if recent_advantages else 0.0
        
        return ReinforcementContext(
            model_weights=self.model_weights,
            confidence_scalers=self.confidence_scalers.copy(),
            exploration_rate=exploration_rate,
            total_trades_processed=self.total_trades_processed,
            baseline_reward=self.baseline_reward,
            recent_advantage=recent_advantage
        )
    
    def apply_reinforcement_to_signal(
        self,
        model_predictions: Dict[str, Dict],
        use_exploration: bool = True
    ) -> Tuple[Dict[str, float], bool]:
        """
        Apply reinforcement learning to model predictions
        
        Args:
            model_predictions: {model: {action: str, confidence: float}}
            use_exploration: Whether to use exploration (ε-greedy)
            
        Returns:
            Tuple of (adjusted_weights, is_exploring)
        """
        exploration_rate = self._calculate_exploration_rate()
        
        # Decide: explore or exploit?
        is_exploring = use_exploration and np.random.random() < exploration_rate
        
        if is_exploring:
            # Uniform weights (explore)
            weights = {
                ModelType.XGBOOST.value: 0.25,
                ModelType.LIGHTGBM.value: 0.25,
                ModelType.NHITS.value: 0.25,
                ModelType.PATCHTST.value: 0.25
            }
            logger.debug(f"[REINFORCEMENT] Exploring (ε={exploration_rate:.3f})")
        else:
            # Use learned weights (exploit)
            weights = self.model_weights.to_dict()
            del weights['last_updated']
            logger.debug(f"[REINFORCEMENT] Exploiting learned weights")
        
        return weights, is_exploring
    
    def get_confidence_scaler(self, model: str) -> float:
        """
        Get confidence scaler for a model based on calibration
        
        Args:
            model: Model name (e.g., 'xgboost')
            
        Returns:
            Scaler value (0.5 to 1.5 typically)
        """
        return self.confidence_scalers.get(model, 1.0)
    
    # ========================================
    # REWARD SHAPING
    # ========================================
    
    def _shape_reward(self, outcome: TradeOutcome) -> Tuple[float, float, float]:
        """
        Shape reward using PnL, Sharpe, and risk-adjusted components
        
        Returns:
            Tuple of (shaped_reward, sharpe_contribution, risk_adjusted_return)
        """
        # Component 1: Normalized PnL (z-score)
        if len(self.trade_history) >= 10:
            recent_pnls = [t.pnl for t in self.trade_history]
            pnl_mean = np.mean(recent_pnls)
            pnl_std = np.std(recent_pnls)
            if pnl_std > 0:
                pnl_normalized = (outcome.pnl - pnl_mean) / pnl_std
            else:
                pnl_normalized = 0.0
        else:
            # Not enough history, use raw PnL scaled
            pnl_normalized = outcome.pnl / 50.0  # Assume $50 avg trade
        
        # Component 2: Sharpe contribution
        # Sharpe = (Return - RiskFreeRate) / Volatility
        # Simplified: PnL / trade_volatility_estimate
        risk_free_rate = 0.0  # Assume 0 for simplicity
        trade_volatility = self._estimate_trade_volatility(outcome)
        if trade_volatility > 0:
            sharpe_contribution = (outcome.pnl - risk_free_rate) / trade_volatility
        else:
            sharpe_contribution = 0.0
        
        # Component 3: Risk-adjusted return
        # PnL relative to maximum potential loss (position size * stop loss %)
        max_potential_loss = outcome.position_size * outcome.entry_price * 0.05  # Assume 5% SL
        if max_potential_loss > 0:
            risk_adjusted_return = outcome.pnl / max_potential_loss
        else:
            risk_adjusted_return = 0.0
        
        # Combine with weights
        shaped_reward = (
            self.reward_alpha * pnl_normalized +
            self.reward_beta * sharpe_contribution +
            self.reward_gamma * risk_adjusted_return
        )
        
        return shaped_reward, sharpe_contribution, risk_adjusted_return
    
    def _estimate_trade_volatility(self, outcome: TradeOutcome) -> float:
        """
        Estimate volatility of this trade type
        
        Uses recent trades with similar characteristics (symbol, regime)
        """
        similar_trades = [
            t for t in self.trade_history
            if t.symbol == outcome.symbol and t.regime == outcome.regime
        ]
        
        if len(similar_trades) >= 5:
            pnls = [t.pnl for t in similar_trades]
            return np.std(pnls)
        else:
            # Default estimate based on position size
            return outcome.position_size * outcome.entry_price * 0.02  # 2% volatility
    
    # ========================================
    # MODEL WEIGHT UPDATES
    # ========================================
    
    def _update_model_weights(self, signal: ReinforcementSignal):
        """
        Update model weights using exponential update rule
        
        w_i(t+1) = w_i(t) * exp(η * R_shaped * I_i)
        """
        outcome = signal.trade_outcome
        shaped_reward = signal.shaped_reward
        
        # Identify which models voted for the winning action
        model_indicators = {}
        for model_str, vote in outcome.model_votes.items():
            # Did this model vote for the action that was taken?
            voted_for_action = (vote['action'] == outcome.action)
            model_indicators[model_str] = 1.0 if voted_for_action else 0.0
        
        # Update weights
        new_weights = {}
        for model in ModelType:
            model_str = model.value
            current_weight = self.model_weights.get_weight(model)
            indicator = model_indicators.get(model_str, 0.0)
            
            # Exponential update
            delta = self.learning_rate * shaped_reward * indicator
            new_weight = current_weight * np.exp(delta)
            
            # Clip to bounds
            new_weight = max(self.min_model_weight, min(self.max_model_weight, new_weight))
            
            new_weights[model] = new_weight
        
        # Normalize to sum to 1.0
        total_weight = sum(new_weights.values())
        for model in ModelType:
            normalized_weight = new_weights[model] / total_weight
            self.model_weights.set_weight(model, normalized_weight)
        
        self.model_weights.last_updated = datetime.now(timezone.utc).isoformat()
        self.weight_update_count += 1
        
        logger.debug(
            f"[REINFORCEMENT] Weights updated: "
            f"XGB={self.model_weights.xgboost:.3f}, "
            f"LGB={self.model_weights.lightgbm:.3f}, "
            f"N-HiTS={self.model_weights.nhits:.3f}, "
            f"PatchTST={self.model_weights.patchtst:.3f}"
        )
    
    def _calculate_model_contributions(
        self,
        outcome: TradeOutcome,
        shaped_reward: float
    ) -> Dict[str, float]:
        """
        Calculate each model's contribution to this outcome
        
        Returns:
            {model: contribution_value}
        """
        contributions = {}
        
        for model_str, vote in outcome.model_votes.items():
            # Contribution = (alignment * confidence * shaped_reward)
            alignment = 1.0 if vote['action'] == outcome.action else -1.0
            confidence = vote['confidence']
            contribution = alignment * confidence * shaped_reward
            contributions[model_str] = contribution
        
        return contributions
    
    # ========================================
    # CONFIDENCE CALIBRATION
    # ========================================
    
    def _update_calibration(self, outcome: TradeOutcome):
        """
        Update confidence calibration metrics (Brier score)
        """
        was_win = outcome.pnl > 0
        actual_outcome = 1.0 if was_win else 0.0
        
        for model_str, vote in outcome.model_votes.items():
            predicted_prob = vote['confidence']
            
            # Brier score component
            squared_error = (predicted_prob - actual_outcome) ** 2
            
            # Update running Brier score (exponential moving average)
            current_brier = self.calibration_metrics.brier_score.get(model_str, 0.25)
            alpha = 0.1  # EMA weight
            new_brier = alpha * squared_error + (1 - alpha) * current_brier
            
            self.calibration_metrics.brier_score[model_str] = new_brier
            
            # Calculate calibration error (abs difference)
            calibration_error = abs(predicted_prob - actual_outcome)
            current_cal_error = self.calibration_metrics.calibration_error.get(model_str, 0.0)
            new_cal_error = alpha * calibration_error + (1 - alpha) * current_cal_error
            
            self.calibration_metrics.calibration_error[model_str] = new_cal_error
            
            # Update sample count
            self.calibration_metrics.sample_counts[model_str] = \
                self.calibration_metrics.sample_counts.get(model_str, 0) + 1
        
        # Update confidence scalers based on calibration
        for model_str in self.calibration_metrics.brier_score.keys():
            brier = self.calibration_metrics.brier_score[model_str]
            
            # Scaler = 1 - κ * Brier
            # Good calibration (Brier~0) → scaler~1.0
            # Poor calibration (Brier~0.25) → scaler~0.875
            scaler = 1.0 - self.calibration_kappa * brier
            scaler = max(0.5, min(1.5, scaler))  # Clip to [0.5, 1.5]
            
            self.confidence_scalers[model_str] = scaler
        
        self.calibration_metrics.last_updated = datetime.now(timezone.utc).isoformat()
        
        logger.debug(f"[REINFORCEMENT] Calibration updated, Brier scores: {self.calibration_metrics.brier_score}")
    
    # ========================================
    # BASELINE REWARD
    # ========================================
    
    def _update_baseline_reward(self, shaped_reward: float):
        """
        Update baseline reward (moving average for advantage calculation)
        """
        self.baseline_window.append(shaped_reward)
        self.baseline_reward = np.mean(self.baseline_window)
    
    # ========================================
    # EXPLORATION
    # ========================================
    
    def _calculate_exploration_rate(self) -> float:
        """
        Calculate current exploration rate (ε) with exponential decay
        
        ε(t) = max(ε_min, ε_initial * exp(-t / decay_trades))
        """
        if self.total_trades_processed == 0:
            return self.initial_exploration_rate
        
        decay_rate = -np.log(self.min_exploration_rate / self.initial_exploration_rate) / self.exploration_decay_trades
        epsilon = self.initial_exploration_rate * np.exp(-decay_rate * self.total_trades_processed)
        epsilon = max(self.min_exploration_rate, epsilon)
        
        return epsilon
    
    # ========================================
    # DIAGNOSTICS
    # ========================================
    
    def get_diagnostics(self) -> Dict:
        """
        Get comprehensive diagnostics for monitoring
        """
        recent_signals = list(self.reinforcement_signals)[-20:]
        
        return {
            "total_trades_processed": self.total_trades_processed,
            "total_reward_accumulated": self.total_reward_accumulated,
            "weight_update_count": self.weight_update_count,
            "model_weights": self.model_weights.to_dict(),
            "confidence_scalers": self.confidence_scalers.copy(),
            "calibration_metrics": {
                "brier_scores": self.calibration_metrics.brier_score.copy(),
                "calibration_errors": self.calibration_metrics.calibration_error.copy(),
                "sample_counts": self.calibration_metrics.sample_counts.copy()
            },
            "baseline_reward": self.baseline_reward,
            "exploration_rate": self._calculate_exploration_rate(),
            "recent_performance": {
                "avg_shaped_reward": np.mean([s.shaped_reward for s in recent_signals]) if recent_signals else 0.0,
                "avg_advantage": np.mean([s.advantage for s in recent_signals]) if recent_signals else 0.0,
                "avg_raw_pnl": np.mean([s.raw_reward for s in recent_signals]) if recent_signals else 0.0
            },
            "learning_params": {
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "reward_alpha": self.reward_alpha,
                "reward_beta": self.reward_beta,
                "reward_gamma": self.reward_gamma
            }
        }
    
    # ========================================
    # PERSISTENCE
    # ========================================
    
    def checkpoint(self):
        """Save current state to disk"""
        try:
            state = {
                "model_weights": self.model_weights.to_dict(),
                "confidence_scalers": self.confidence_scalers,
                "calibration_metrics": {
                    "brier_score": self.calibration_metrics.brier_score,
                    "calibration_error": self.calibration_metrics.calibration_error,
                    "sample_counts": self.calibration_metrics.sample_counts,
                    "last_updated": self.calibration_metrics.last_updated
                },
                "baseline_reward": self.baseline_reward,
                "baseline_window": list(self.baseline_window),
                "total_trades_processed": self.total_trades_processed,
                "total_reward_accumulated": self.total_reward_accumulated,
                "weight_update_count": self.weight_update_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Atomic write
            temp_path = f"{self.checkpoint_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            os.replace(temp_path, self.checkpoint_path)
            
            self.last_checkpoint_time = datetime.now(timezone.utc)
            logger.info(f"[REINFORCEMENT] Checkpoint saved: {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"[REINFORCEMENT] Checkpoint failed: {e}")
    
    def _load_checkpoint(self):
        """Load state from disk if available"""
        if not os.path.exists(self.checkpoint_path):
            logger.info("[REINFORCEMENT] No checkpoint found, starting fresh")
            return
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            # Restore model weights
            weights_data = state['model_weights']
            self.model_weights = ModelWeights(
                xgboost=weights_data['xgboost'],
                lightgbm=weights_data['lightgbm'],
                nhits=weights_data['nhits'],
                patchtst=weights_data['patchtst'],
                last_updated=weights_data['last_updated']
            )
            
            # Restore confidence scalers
            self.confidence_scalers = state['confidence_scalers']
            
            # Restore calibration metrics
            cal_data = state['calibration_metrics']
            self.calibration_metrics = CalibrationMetrics(
                brier_score=cal_data['brier_score'],
                calibration_error=cal_data['calibration_error'],
                sample_counts=cal_data['sample_counts'],
                last_updated=cal_data['last_updated']
            )
            
            # Restore baseline
            self.baseline_reward = state['baseline_reward']
            self.baseline_window = deque(state['baseline_window'], maxlen=20)
            
            # Restore stats
            self.total_trades_processed = state['total_trades_processed']
            self.total_reward_accumulated = state['total_reward_accumulated']
            self.weight_update_count = state['weight_update_count']
            
            logger.info(
                f"[REINFORCEMENT] Checkpoint loaded: {self.total_trades_processed} trades, "
                f"baseline_reward={self.baseline_reward:.3f}"
            )
        
        except Exception as e:
            logger.error(f"[REINFORCEMENT] Failed to load checkpoint: {e}")
            logger.warning("[REINFORCEMENT] Starting with default state")
    
    def auto_checkpoint_check(self):
        """Check if it's time to auto-checkpoint"""
        elapsed = (datetime.now(timezone.utc) - self.last_checkpoint_time).total_seconds()
        if elapsed >= self.checkpoint_interval:
            self.checkpoint()
    
    # ========================================
    # MANUAL CONTROLS
    # ========================================
    
    def reset_weights(self):
        """Reset model weights to initial values"""
        self.model_weights = ModelWeights(
            xgboost=0.25,
            lightgbm=0.25,
            nhits=0.30,
            patchtst=0.20,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        logger.warning("[REINFORCEMENT] Model weights reset to initial values")
    
    def reset_calibration(self):
        """Reset confidence calibration"""
        self.calibration_metrics = CalibrationMetrics(
            brier_score={m.value: 0.25 for m in ModelType},
            calibration_error={m.value: 0.0 for m in ModelType},
            sample_counts={m.value: 0 for m in ModelType},
            last_updated=datetime.now(timezone.utc).isoformat()
        )
        self.confidence_scalers = {m.value: 1.0 for m in ModelType}
        logger.warning("[REINFORCEMENT] Calibration metrics reset")
