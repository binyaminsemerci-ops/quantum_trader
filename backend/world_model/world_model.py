"""World Model - Probabilistic Market State Projections.

The World Model approximates possible future market states using:
- Historical volatility patterns
- Regime transition probabilities
- Sentiment and trend strength
- Statistical price projections

This is a heuristic + statistical model (not ML-based in v1).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Market regime classifications."""
    
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"
    VOLATILE_UP = "VOLATILE_UP"
    VOLATILE_DOWN = "VOLATILE_DOWN"


@dataclass
class MarketState:
    """
    Current market state snapshot.
    
    Used as input to World Model for generating scenarios.
    """
    
    current_price: float
    current_regime: MarketRegime
    volatility: float  # Recent historical volatility
    trend_strength: float  # 0.0 to 1.0
    volume_ratio: float  # Current volume / average volume
    
    # Optional sentiment indicators
    sentiment_score: Optional[float] = None  # -1.0 to 1.0
    fear_greed_index: Optional[float] = None  # 0 to 100
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_price": self.current_price,
            "current_regime": self.current_regime.value,
            "volatility": self.volatility,
            "trend_strength": self.trend_strength,
            "volume_ratio": self.volume_ratio,
            "sentiment_score": self.sentiment_score,
            "fear_greed_index": self.fear_greed_index,
        }


@dataclass
class Scenario:
    """
    Single scenario projection.
    
    Represents one possible future market trajectory with probability.
    """
    
    scenario_id: str
    probability: float  # Probability of this scenario
    
    # Projected state
    regime: MarketRegime
    price_paths: list[float]  # Price trajectory (e.g., 24 hourly points)
    volatility_projection: float
    
    # Risk metrics for this scenario
    max_drawdown_pct: float
    max_gain_pct: float
    
    # Tags
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "probability": self.probability,
            "regime": self.regime.value,
            "price_paths": self.price_paths,
            "volatility_projection": self.volatility_projection,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_gain_pct": self.max_gain_pct,
            "tags": self.tags,
        }


class WorldModel:
    """
    World Model for market state projection.
    
    Features:
    - Generate multiple scenarios with probabilities
    - Project price paths using statistical methods
    - Estimate regime transitions
    - Calculate scenario risk metrics
    
    Usage:
        model = WorldModel()
        
        # Define current market state
        state = MarketState(
            current_price=43000,
            current_regime=MarketRegime.TRENDING_UP,
            volatility=0.35,
            trend_strength=0.75,
            volume_ratio=1.2,
        )
        
        # Generate scenarios
        scenarios = model.generate_scenarios(state, num_scenarios=5)
        
        for scenario in scenarios:
            print(f"Scenario {scenario.scenario_id}: {scenario.probability:.2%}")
            print(f"  Regime: {scenario.regime.value}")
            print(f"  Max DD: {scenario.max_drawdown_pct:.2%}")
    """
    
    # Regime transition matrix (simplified)
    REGIME_TRANSITIONS = {
        MarketRegime.TRENDING_UP: {
            MarketRegime.TRENDING_UP: 0.60,
            MarketRegime.RANGING: 0.20,
            MarketRegime.CHOPPY: 0.10,
            MarketRegime.TRENDING_DOWN: 0.05,
            MarketRegime.VOLATILE_UP: 0.05,
        },
        MarketRegime.TRENDING_DOWN: {
            MarketRegime.TRENDING_DOWN: 0.60,
            MarketRegime.RANGING: 0.20,
            MarketRegime.CHOPPY: 0.10,
            MarketRegime.TRENDING_UP: 0.05,
            MarketRegime.VOLATILE_DOWN: 0.05,
        },
        MarketRegime.RANGING: {
            MarketRegime.RANGING: 0.50,
            MarketRegime.CHOPPY: 0.20,
            MarketRegime.TRENDING_UP: 0.15,
            MarketRegime.TRENDING_DOWN: 0.15,
        },
        MarketRegime.CHOPPY: {
            MarketRegime.CHOPPY: 0.40,
            MarketRegime.RANGING: 0.30,
            MarketRegime.TRENDING_UP: 0.15,
            MarketRegime.TRENDING_DOWN: 0.15,
        },
        MarketRegime.VOLATILE_UP: {
            MarketRegime.RANGING: 0.40,
            MarketRegime.TRENDING_UP: 0.30,
            MarketRegime.CHOPPY: 0.20,
            MarketRegime.VOLATILE_UP: 0.10,
        },
        MarketRegime.VOLATILE_DOWN: {
            MarketRegime.RANGING: 0.40,
            MarketRegime.TRENDING_DOWN: 0.30,
            MarketRegime.CHOPPY: 0.20,
            MarketRegime.VOLATILE_DOWN: 0.10,
        },
    }
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize World Model.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        logger.info("WorldModel initialized")
    
    def generate_scenarios(
        self,
        current_state: MarketState,
        num_scenarios: int = 5,
        time_horizon_hours: int = 24,
    ) -> list[Scenario]:
        """
        Generate scenarios from current market state.
        
        Args:
            current_state: Current market state
            num_scenarios: Number of scenarios to generate
            time_horizon_hours: Projection time horizon
        
        Returns:
            List of scenarios with probabilities
        """
        scenarios: list[Scenario] = []
        
        # Generate most likely scenario
        scenarios.append(
            self._generate_base_scenario(
                current_state,
                time_horizon_hours,
                scenario_id="BASE",
            )
        )
        
        # Generate optimistic scenario
        scenarios.append(
            self._generate_optimistic_scenario(
                current_state,
                time_horizon_hours,
                scenario_id="OPTIMISTIC",
            )
        )
        
        # Generate pessimistic scenario
        scenarios.append(
            self._generate_pessimistic_scenario(
                current_state,
                time_horizon_hours,
                scenario_id="PESSIMISTIC",
            )
        )
        
        # Generate additional scenarios if requested
        for i in range(num_scenarios - 3):
            scenarios.append(
                self._generate_random_scenario(
                    current_state,
                    time_horizon_hours,
                    scenario_id=f"RANDOM_{i+1}",
                )
            )
        
        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        for scenario in scenarios:
            scenario.probability /= total_prob
        
        logger.debug(f"Generated {len(scenarios)} scenarios")
        
        return scenarios
    
    def estimate_regime_transition(
        self,
        current_regime: MarketRegime,
    ) -> MarketRegime:
        """
        Estimate next regime based on transition probabilities.
        
        Args:
            current_regime: Current market regime
        
        Returns:
            Projected next regime
        """
        transitions = self.REGIME_TRANSITIONS.get(
            current_regime,
            {current_regime: 1.0},
        )
        
        # Sample from transition probabilities
        regimes = list(transitions.keys())
        probs = list(transitions.values())
        
        return random.choices(regimes, weights=probs)[0]
    
    def _generate_base_scenario(
        self,
        state: MarketState,
        horizon: int,
        scenario_id: str,
    ) -> Scenario:
        """Generate most likely scenario (continuation of current trend)."""
        # Project regime
        projected_regime = self.estimate_regime_transition(state.current_regime)
        
        # Generate price path
        price_path = self._generate_price_path(
            start_price=state.current_price,
            regime=projected_regime,
            volatility=state.volatility,
            trend_strength=state.trend_strength,
            horizon=horizon,
            drift_multiplier=1.0,
        )
        
        # Calculate metrics
        max_dd = self._calculate_max_drawdown(price_path)
        max_gain = self._calculate_max_gain(price_path)
        
        return Scenario(
            scenario_id=scenario_id,
            probability=0.50,  # Base scenario has highest weight
            regime=projected_regime,
            price_paths=price_path,
            volatility_projection=state.volatility,
            max_drawdown_pct=max_dd,
            max_gain_pct=max_gain,
            tags=["base", "likely"],
        )
    
    def _generate_optimistic_scenario(
        self,
        state: MarketState,
        horizon: int,
        scenario_id: str,
    ) -> Scenario:
        """Generate optimistic scenario (strong upward movement)."""
        # Force upward regime
        projected_regime = MarketRegime.TRENDING_UP
        
        # Generate bullish price path
        price_path = self._generate_price_path(
            start_price=state.current_price,
            regime=projected_regime,
            volatility=state.volatility * 0.8,  # Lower volatility
            trend_strength=0.85,  # Strong trend
            horizon=horizon,
            drift_multiplier=1.5,  # Strong upward drift
        )
        
        max_dd = self._calculate_max_drawdown(price_path)
        max_gain = self._calculate_max_gain(price_path)
        
        return Scenario(
            scenario_id=scenario_id,
            probability=0.25,
            regime=projected_regime,
            price_paths=price_path,
            volatility_projection=state.volatility * 0.8,
            max_drawdown_pct=max_dd,
            max_gain_pct=max_gain,
            tags=["optimistic", "bullish"],
        )
    
    def _generate_pessimistic_scenario(
        self,
        state: MarketState,
        horizon: int,
        scenario_id: str,
    ) -> Scenario:
        """Generate pessimistic scenario (downward movement / high volatility)."""
        # Force downward or volatile regime
        projected_regime = MarketRegime.VOLATILE_DOWN
        
        # Generate bearish price path
        price_path = self._generate_price_path(
            start_price=state.current_price,
            regime=projected_regime,
            volatility=state.volatility * 1.5,  # Higher volatility
            trend_strength=0.70,
            horizon=horizon,
            drift_multiplier=-0.8,  # Downward drift
        )
        
        max_dd = self._calculate_max_drawdown(price_path)
        max_gain = self._calculate_max_gain(price_path)
        
        return Scenario(
            scenario_id=scenario_id,
            probability=0.20,
            regime=projected_regime,
            price_paths=price_path,
            volatility_projection=state.volatility * 1.5,
            max_drawdown_pct=max_dd,
            max_gain_pct=max_gain,
            tags=["pessimistic", "bearish", "volatile"],
        )
    
    def _generate_random_scenario(
        self,
        state: MarketState,
        horizon: int,
        scenario_id: str,
    ) -> Scenario:
        """Generate random scenario."""
        projected_regime = self.estimate_regime_transition(state.current_regime)
        
        drift_mult = np.random.uniform(-1.0, 1.0)
        vol_mult = np.random.uniform(0.7, 1.5)
        
        price_path = self._generate_price_path(
            start_price=state.current_price,
            regime=projected_regime,
            volatility=state.volatility * vol_mult,
            trend_strength=state.trend_strength,
            horizon=horizon,
            drift_multiplier=drift_mult,
        )
        
        max_dd = self._calculate_max_drawdown(price_path)
        max_gain = self._calculate_max_gain(price_path)
        
        return Scenario(
            scenario_id=scenario_id,
            probability=0.05,
            regime=projected_regime,
            price_paths=price_path,
            volatility_projection=state.volatility * vol_mult,
            max_drawdown_pct=max_dd,
            max_gain_pct=max_gain,
            tags=["random"],
        )
    
    def _generate_price_path(
        self,
        start_price: float,
        regime: MarketRegime,
        volatility: float,
        trend_strength: float,
        horizon: int,
        drift_multiplier: float = 1.0,
    ) -> list[float]:
        """
        Generate price path using geometric Brownian motion.
        
        Args:
            start_price: Starting price
            regime: Market regime
            volatility: Volatility (annualized)
            trend_strength: Trend strength
            horizon: Number of time steps
            drift_multiplier: Drift adjustment
        
        Returns:
            List of prices
        """
        # Adjust drift based on regime
        if regime == MarketRegime.TRENDING_UP:
            drift = volatility * 0.5 * trend_strength
        elif regime == MarketRegime.TRENDING_DOWN:
            drift = -volatility * 0.5 * trend_strength
        else:
            drift = 0.0
        
        drift *= drift_multiplier
        
        # Adjust volatility for hourly steps
        dt = 1.0 / (365 * 24)  # Hourly time step
        hourly_vol = volatility * np.sqrt(dt)
        
        # Generate price path
        prices = [start_price]
        
        for _ in range(horizon - 1):
            # Geometric Brownian motion
            dW = np.random.normal(0, 1)
            price_change = drift * dt + hourly_vol * dW
            
            next_price = prices[-1] * (1 + price_change)
            prices.append(next_price)
        
        return prices
    
    def _calculate_max_drawdown(self, prices: list[float]) -> float:
        """Calculate maximum drawdown percentage."""
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_max_gain(self, prices: list[float]) -> float:
        """Calculate maximum gain percentage from start."""
        start_price = prices[0]
        max_price = max(prices)
        
        return (max_price - start_price) / start_price
