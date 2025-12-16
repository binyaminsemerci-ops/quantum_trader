"""
Gym trading environment for RL v3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from backend.domains.learning.rl_v3.config_v3 import RLv3Config
from backend.domains.learning.rl_v3.features_v3 import build_feature_vector
from backend.domains.learning.rl_v3.reward_v3 import compute_reward
from backend.domains.learning.rl_v3.market_data_provider import (
    MarketDataProvider,
    SyntheticMarketDataProvider
)


class TradingEnvV3(gym.Env):
    """
    Gym environment for trading with 6 discrete actions.
    
    ⚠️ NOTE: Uses SYNTHETIC prices by default. Pass market_data_provider for real data.
    
    Actions:
        0: HOLD
        1: LONG
        2: SHORT
        3: REDUCE
        4: CLOSE
        5: EMERGENCY_FLATTEN
    
    Usage:
        # Testing with synthetic prices
        env = TradingEnvV3(config)
        
        # Production with real prices
        from backend.domains.learning.rl_v3.market_data_provider import RealMarketDataProvider
        provider = RealMarketDataProvider(symbol="BTC/USDT", timeframe="1h")
        env = TradingEnvV3(config, market_data_provider=provider)
    """
    
    def __init__(
        self,
        config: RLv3Config,
        market_data_provider: Optional[MarketDataProvider] = None,
        tp_reward_weight: float = 1.0
    ):
        """
        Initialize environment.
        
        Args:
            config: RL v3 configuration
            market_data_provider: Optional provider for real market data.
                                 If None, uses synthetic random walk.
            tp_reward_weight: Weight for TP accuracy reward component (default 1.0).
                            Set by CLM v3 based on production TP performance.
        """
        super().__init__()
        
        self.config = config
        self.tp_reward_weight = tp_reward_weight  # [CLM v3 TP feedback]
        
        # Market data provider (synthetic by default)
        if market_data_provider is None:
            self.market_data_provider = SyntheticMarketDataProvider(
                initial_price=100.0,
                volatility=0.02
            )
        else:
            self.market_data_provider = market_data_provider
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(config.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.state_dim,),
            dtype=np.float32
        )
        
        # State variables
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.position_size = 0.0
        self.position_side = 0  # 0=none, 1=long, -1=short
        self.entry_price = 0.0
        
        # [TP v3] TP zone tracking for reward calculation
        self.tp_target = 0.0
        self.tp_zone_width = 0.06  # 6% default TP zone
        self.tp_hit_count = 0
        self.tp_miss_count = 0
        self.tp_zone_accuracy = 0.0
        
        self.current_step = 0
        self.max_steps = config.max_steps_per_episode
        
        # Price series from provider
        self.prices = self.market_data_provider.get_price_series(self.max_steps)
        self.current_price = self.prices[0]
    
    def _build_observation(self) -> np.ndarray:
        """Build observation from current state."""
        # Create observation dict
        obs_dict = {
            'price_change_1m': (self.current_price - self.prices[max(0, self.current_step-1)]) / self.prices[max(0, self.current_step-1)],
            'price_change_5m': (self.current_price - self.prices[max(0, self.current_step-5)]) / self.prices[max(0, self.current_step-5)] if self.current_step >= 5 else 0.0,
            'price_change_15m': (self.current_price - self.prices[max(0, self.current_step-15)]) / self.prices[max(0, self.current_step-15)] if self.current_step >= 15 else 0.0,
            'volatility': np.std(self.prices[max(0, self.current_step-20):self.current_step+1]) / self.current_price if self.current_step > 0 else 0.01,
            'rsi': 50.0,  # Placeholder
            'macd': 0.0,  # Placeholder
            'position_size': abs(self.position_size),
            'position_side': float(self.position_side),
            'balance': self.balance,
            'equity': self.equity,
            'regime': 'TREND',  # Placeholder
            'trend_strength': 0.5,
            'volume_ratio': 1.0,
            'spread': 0.001,
            'time_of_day': (self.current_step % 1440) / 1440.0  # Assume 1 step = 1 min
        }
        
        return build_feature_vector(obs_dict)
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.position_size = 0.0
        self.position_side = 0
        self.entry_price = 0.0
        
        # [TP v3] Reset TP tracking
        self.tp_target = 0.0
        self.tp_hit_count = 0
        self.tp_miss_count = 0
        self.tp_zone_accuracy = 0.0
        
        self.current_step = 0
        
        # Get fresh price series from provider
        self.prices = self.market_data_provider.get_price_series(self.max_steps)
        self.current_price = self.prices[0]
        
        return self._build_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        prev_equity = self.equity
        
        # Execute action
        if action == 0:  # HOLD
            pass
        elif action == 1:  # LONG
            if self.position_size == 0:
                self.position_size = self.equity * 0.1  # 10% of equity
                self.position_side = 1
                self.entry_price = self.current_price
                # [TP v3] Set TP target zone
                self.tp_target = self.current_price * (1 + self.tp_zone_width)
        elif action == 2:  # SHORT
            if self.position_size == 0:
                self.position_size = self.equity * 0.1
                self.position_side = -1
                self.entry_price = self.current_price
                # [TP v3] Set TP target zone
                self.tp_target = self.current_price * (1 - self.tp_zone_width)
        elif action == 3:  # REDUCE
            if self.position_size > 0:
                self.position_size *= 0.5
        elif action == 4:  # CLOSE
            if self.position_size > 0:
                # [TP v3] Check if closing near TP target
                if self.tp_target > 0:
                    if self.position_side == 1:  # LONG
                        tp_hit = self.current_price >= self.tp_target * 0.95  # Within 5% of TP
                    else:  # SHORT
                        tp_hit = self.current_price <= self.tp_target * 1.05
                    
                    if tp_hit:
                        self.tp_hit_count += 1
                    else:
                        self.tp_miss_count += 1
                    
                    # Update accuracy
                    total = self.tp_hit_count + self.tp_miss_count
                    self.tp_zone_accuracy = self.tp_hit_count / total if total > 0 else 0.0
                
                self.position_size = 0.0
                self.position_side = 0
                self.entry_price = 0.0
                self.tp_target = 0.0
        elif action == 5:  # EMERGENCY_FLATTEN
            # [TP v3] Count as TP miss (emergency exit)
            if self.tp_target > 0:
                self.tp_miss_count += 1
                total = self.tp_hit_count + self.tp_miss_count
                self.tp_zone_accuracy = self.tp_hit_count / total if total > 0 else 0.0
            
            self.position_size = 0.0
            self.position_side = 0
            self.entry_price = 0.0
            self.tp_target = 0.0
        
        # Move to next step
        self.current_step += 1
        self.current_price = self.prices[self.current_step]
        
        # Calculate PnL
        if self.position_size > 0 and self.entry_price > 0:
            price_change = (self.current_price - self.entry_price) / self.entry_price
            position_pnl = self.position_size * price_change * self.position_side
            self.equity = prev_equity + position_pnl
        else:
            self.equity = prev_equity
        
        # Calculate reward
        pnl_delta = self.equity - prev_equity
        drawdown = max(0, 1.0 - self.equity / self.balance)
        regime_alignment = 1.0 if action in [0, 1, 2] else 0.0  # Placeholder
        volatility = 0.02  # Placeholder
        
        # [TP v3] Include TP zone accuracy in reward with configurable weight
        reward = compute_reward(
            pnl_delta / self.balance,
            drawdown,
            abs(self.position_size) / self.equity if self.equity > 0 else 0.0,
            regime_alignment,
            volatility,
            tp_zone_accuracy=self.tp_zone_accuracy,
            tp_reward_weight=self.tp_reward_weight  # [CLM v3 TP feedback]
        )
        
        # Check if done
        done = bool(self.current_step >= self.max_steps - 1 or self.equity < self.balance * 0.5)
        
        info = {
            'equity': self.equity,
            'balance': self.balance,
            'position_size': self.position_size,
            'pnl': self.equity - self.balance,
            'tp_hit_count': self.tp_hit_count,
            'tp_miss_count': self.tp_miss_count,
            'tp_zone_accuracy': self.tp_zone_accuracy
        }
        
        return self._build_observation(), reward, done, info
