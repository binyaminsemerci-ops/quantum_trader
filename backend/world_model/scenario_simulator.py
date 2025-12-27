"""Scenario Simulator - Policy Simulation and Evaluation.

Simulates trading system behavior under different scenarios and policies,
allowing AI agents to evaluate "what-if" scenarios before making decisions.

Features:
- Simulate multiple policy configurations
- Run Monte Carlo simulations across scenarios
- Calculate expected outcomes (PnL, drawdown, risk metrics)
- Identify worst-case scenarios
- Compare policy performance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from backend.world_model.world_model import MarketState, Scenario, WorldModel

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """
    Configuration for a policy to simulate.
    
    Represents a specific policy configuration to test
    in scenario simulations.
    """
    
    policy_id: str
    
    # Policy parameters
    global_mode: str
    leverage: float
    max_positions: int
    position_size_pct: float
    risk_per_trade_pct: float
    
    # Strategy settings
    enable_rl: bool = True
    enable_pba: bool = True
    enable_pal: bool = True
    enable_pil: bool = True
    
    # Risk limits
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.05
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "global_mode": self.global_mode,
            "leverage": self.leverage,
            "max_positions": self.max_positions,
            "position_size_pct": self.position_size_pct,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "enable_rl": self.enable_rl,
            "enable_pba": self.enable_pba,
            "enable_pal": self.enable_pal,
            "enable_pil": self.enable_pil,
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
        }


@dataclass
class SimulationResult:
    """
    Result of policy simulation.
    
    Contains aggregated metrics across all scenarios and paths.
    """
    
    policy_id: str
    
    # Expected value metrics
    expected_pnl: float
    expected_return_pct: float
    expected_drawdown_pct: float
    
    # Distribution metrics
    pnl_std: float
    pnl_median: float
    pnl_p5: float  # 5th percentile (downside)
    pnl_p95: float  # 95th percentile (upside)
    
    # Risk metrics
    worst_case_pnl: float
    worst_case_drawdown_pct: float
    probability_of_loss: float
    probability_of_large_loss: float  # Loss > 5%
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    
    # Scenario breakdown
    scenario_results: list[dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "expected_pnl": self.expected_pnl,
            "expected_return_pct": self.expected_return_pct,
            "expected_drawdown_pct": self.expected_drawdown_pct,
            "pnl_std": self.pnl_std,
            "pnl_median": self.pnl_median,
            "pnl_p5": self.pnl_p5,
            "pnl_p95": self.pnl_p95,
            "worst_case_pnl": self.worst_case_pnl,
            "worst_case_drawdown_pct": self.worst_case_drawdown_pct,
            "probability_of_loss": self.probability_of_loss,
            "probability_of_large_loss": self.probability_of_large_loss,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "scenario_results": self.scenario_results,
        }


class ScenarioSimulator:
    """
    Scenario simulator for policy evaluation.
    
    Features:
    - Simulate policy behavior under different market scenarios
    - Run Monte Carlo simulations (multiple paths per scenario)
    - Calculate expected outcomes and risk metrics
    - Compare multiple candidate policies
    - Identify worst-case scenarios
    
    Usage:
        simulator = ScenarioSimulator()
        
        # Define current state
        state = MarketState(
            current_price=43000,
            current_regime=MarketRegime.TRENDING_UP,
            volatility=0.35,
            trend_strength=0.75,
            volume_ratio=1.2,
        )
        
        # Define candidate policies
        current_policy = SimulationConfig(
            policy_id="CURRENT",
            global_mode="GROWTH",
            leverage=3.0,
            max_positions=5,
            position_size_pct=0.02,
            risk_per_trade_pct=0.015,
        )
        
        aggressive_policy = SimulationConfig(
            policy_id="AGGRESSIVE",
            global_mode="EXPANSION",
            leverage=5.0,
            max_positions=8,
            position_size_pct=0.03,
            risk_per_trade_pct=0.02,
        )
        
        # Run simulations
        results = simulator.run_scenarios(
            current_state=state,
            candidate_policies=[current_policy, aggressive_policy],
            num_paths=1000,
        )
        
        # Compare results
        for result in results:
            print(f"{result.policy_id}: Expected PnL = {result.expected_pnl:.2f}")
            print(f"  Worst case: {result.worst_case_pnl:.2f}")
            print(f"  P(loss): {result.probability_of_loss:.2%}")
    """
    
    DEFAULT_NUM_PATHS = 1000
    DEFAULT_NUM_SCENARIOS = 5
    LARGE_LOSS_THRESHOLD = 0.05  # 5%
    
    def __init__(
        self,
        world_model: Optional[WorldModel] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Scenario Simulator.
        
        Args:
            world_model: Optional WorldModel instance
            seed: Random seed for reproducibility
        """
        self.world_model = world_model or WorldModel(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info("ScenarioSimulator initialized")
    
    def run_scenarios(
        self,
        current_state: MarketState,
        candidate_policies: list[SimulationConfig],
        num_paths: int = DEFAULT_NUM_PATHS,
        num_scenarios: Optional[int] = None,
    ) -> list[SimulationResult]:
        """
        Run scenario simulations for multiple policies.
        
        Args:
            current_state: Current market state
            candidate_policies: List of policies to simulate
            num_paths: Number of Monte Carlo paths per scenario
            num_scenarios: Number of scenarios (default: 5)
        
        Returns:
            List of SimulationResults, one per policy
        """
        # Generate scenarios
        num_scenarios = num_scenarios or self.DEFAULT_NUM_SCENARIOS
        scenarios = self.world_model.generate_scenarios(
            current_state,
            num_scenarios=num_scenarios,
        )
        
        logger.info(
            f"Running simulations: {len(candidate_policies)} policies, "
            f"{len(scenarios)} scenarios, {num_paths} paths"
        )
        
        # Simulate each policy
        results: list[SimulationResult] = []
        
        for policy in candidate_policies:
            result = self._simulate_policy(
                policy=policy,
                scenarios=scenarios,
                num_paths=num_paths,
                current_price=current_state.current_price,
            )
            results.append(result)
            
            logger.debug(
                f"Policy {policy.policy_id}: "
                f"E[PnL]={result.expected_pnl:.2f}, "
                f"Worst={result.worst_case_pnl:.2f}"
            )
        
        return results
    
    def _simulate_policy(
        self,
        policy: SimulationConfig,
        scenarios: list[Scenario],
        num_paths: int,
        current_price: float,
    ) -> SimulationResult:
        """Simulate a single policy across all scenarios."""
        all_pnls: list[float] = []
        all_drawdowns: list[float] = []
        scenario_results: list[dict[str, Any]] = []
        
        # Simulate each scenario
        for scenario in scenarios:
            # Run multiple paths for this scenario
            scenario_pnls: list[float] = []
            scenario_dds: list[float] = []
            
            for _ in range(int(num_paths * scenario.probability)):
                pnl, dd = self._simulate_single_path(
                    policy=policy,
                    scenario=scenario,
                    current_price=current_price,
                )
                scenario_pnls.append(pnl)
                scenario_dds.append(dd)
                all_pnls.append(pnl)
                all_drawdowns.append(dd)
            
            # Store scenario summary
            if scenario_pnls:
                scenario_results.append({
                    "scenario_id": scenario.scenario_id,
                    "probability": scenario.probability,
                    "mean_pnl": float(np.mean(scenario_pnls)),
                    "mean_drawdown": float(np.mean(scenario_dds)),
                    "worst_pnl": float(min(scenario_pnls)),
                })
        
        # Calculate aggregate metrics
        pnls_array = np.array(all_pnls)
        dds_array = np.array(all_drawdowns)
        
        expected_pnl = float(np.mean(pnls_array))
        expected_return_pct = expected_pnl / current_price
        expected_dd = float(np.mean(dds_array))
        
        pnl_std = float(np.std(pnls_array))
        pnl_median = float(np.median(pnls_array))
        pnl_p5 = float(np.percentile(pnls_array, 5))
        pnl_p95 = float(np.percentile(pnls_array, 95))
        
        worst_case_pnl = float(np.min(pnls_array))
        worst_case_dd = float(np.max(dds_array))
        
        prob_loss = float(np.sum(pnls_array < 0) / len(pnls_array))
        large_loss_threshold = -current_price * self.LARGE_LOSS_THRESHOLD
        prob_large_loss = float(np.sum(pnls_array < large_loss_threshold) / len(pnls_array))
        
        # Risk-adjusted metrics
        sharpe = expected_pnl / pnl_std if pnl_std > 0 else 0.0
        
        # Sortino (downside deviation)
        downside_returns = pnls_array[pnls_array < 0]
        downside_std = float(np.std(downside_returns)) if len(downside_returns) > 0 else 1e-6
        sortino = expected_pnl / downside_std if downside_std > 0 else 0.0
        
        return SimulationResult(
            policy_id=policy.policy_id,
            expected_pnl=expected_pnl,
            expected_return_pct=expected_return_pct,
            expected_drawdown_pct=expected_dd,
            pnl_std=pnl_std,
            pnl_median=pnl_median,
            pnl_p5=pnl_p5,
            pnl_p95=pnl_p95,
            worst_case_pnl=worst_case_pnl,
            worst_case_drawdown_pct=worst_case_dd,
            probability_of_loss=prob_loss,
            probability_of_large_loss=prob_large_loss,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            scenario_results=scenario_results,
        )
    
    def _simulate_single_path(
        self,
        policy: SimulationConfig,
        scenario: Scenario,
        current_price: float,
    ) -> tuple[float, float]:
        """
        Simulate a single path for policy + scenario.
        
        Returns:
            (pnl, max_drawdown_pct)
        """
        # Simplified simulation (heuristic-based)
        # In production, this would use actual strategy logic
        
        # Estimate win rate based on regime alignment
        base_win_rate = 0.55
        
        # Adjust win rate based on policy aggressiveness
        leverage_factor = policy.leverage / 3.0  # Normalize to 3x
        position_size_factor = policy.position_size_pct / 0.02  # Normalize to 2%
        
        # Higher leverage/position size increases both upside and downside
        win_rate = base_win_rate * (1.0 + 0.1 * (1.0 - leverage_factor))
        
        # Estimate trade count
        trade_count = policy.max_positions * 3  # Assume 3 trades per position slot
        
        # Simulate trades
        wins = int(trade_count * win_rate)
        losses = trade_count - wins
        
        # Average win/loss size (based on position size and scenario)
        avg_win = current_price * policy.position_size_pct * policy.leverage * 0.015
        avg_loss = current_price * policy.position_size_pct * policy.leverage * 0.010
        
        # Adjust based on scenario
        if "bullish" in scenario.tags or "optimistic" in scenario.tags:
            avg_win *= 1.3
            avg_loss *= 0.8
        elif "bearish" in scenario.tags or "pessimistic" in scenario.tags:
            avg_win *= 0.7
            avg_loss *= 1.4
        
        # Calculate PnL
        total_wins = wins * avg_win
        total_losses = losses * avg_loss
        pnl = total_wins - total_losses
        
        # Estimate drawdown (simplified)
        # Drawdown increases with volatility and leverage
        base_dd = scenario.max_drawdown_pct
        leverage_multiplier = policy.leverage / 3.0
        drawdown = base_dd * leverage_multiplier
        
        # Add some randomness
        pnl *= np.random.uniform(0.8, 1.2)
        drawdown *= np.random.uniform(0.9, 1.1)
        
        # Clamp drawdown to policy limits
        drawdown = min(drawdown, policy.max_drawdown_pct)
        
        return pnl, drawdown
    
    def compare_policies(
        self,
        results: list[SimulationResult],
        risk_tolerance: str = "MODERATE",
    ) -> dict[str, Any]:
        """
        Compare simulation results and recommend best policy.
        
        Args:
            results: List of simulation results
            risk_tolerance: Risk tolerance level (CONSERVATIVE, MODERATE, AGGRESSIVE)
        
        Returns:
            Comparison dictionary with recommendation
        """
        if not results:
            return {
                "recommendation": None,
                "reason": "No simulation results provided",
            }
        
        # Score policies based on risk tolerance
        scored: list[tuple[float, SimulationResult]] = []
        
        for result in results:
            if risk_tolerance == "CONSERVATIVE":
                # Prioritize: low drawdown, low loss probability
                score = (
                    result.expected_pnl * 0.30
                    - result.worst_case_drawdown_pct * 1000 * 0.40
                    - result.probability_of_loss * 500 * 0.30
                )
            elif risk_tolerance == "AGGRESSIVE":
                # Prioritize: high expected return, high upside
                score = (
                    result.expected_pnl * 0.50
                    + result.pnl_p95 * 0.30
                    + result.sharpe_ratio * 100 * 0.20
                )
            else:  # MODERATE
                # Balance risk and return
                score = (
                    result.expected_pnl * 0.40
                    + result.sharpe_ratio * 100 * 0.30
                    - result.worst_case_drawdown_pct * 500 * 0.30
                )
            
            scored.append((score, result))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        best_score, best_result = scored[0]
        
        # Generate recommendation
        comparison = {
            "recommendation": best_result.policy_id,
            "score": best_score,
            "risk_tolerance": risk_tolerance,
            "reason": self._generate_recommendation_reason(best_result, risk_tolerance),
            "all_scores": [
                {
                    "policy_id": r.policy_id,
                    "score": s,
                    "expected_pnl": r.expected_pnl,
                    "worst_case_dd": r.worst_case_drawdown_pct,
                    "sharpe": r.sharpe_ratio,
                }
                for s, r in scored
            ],
        }
        
        return comparison
    
    def _generate_recommendation_reason(
        self,
        result: SimulationResult,
        risk_tolerance: str,
    ) -> str:
        """Generate human-readable recommendation reason."""
        reasons = []
        
        if result.expected_pnl > 0:
            reasons.append(f"positive expected PnL of {result.expected_pnl:.2f}")
        
        if result.sharpe_ratio > 1.0:
            reasons.append(f"strong Sharpe ratio of {result.sharpe_ratio:.2f}")
        
        if result.probability_of_loss < 0.40:
            reasons.append(f"low loss probability of {result.probability_of_loss:.1%}")
        
        if result.worst_case_drawdown_pct < 0.10:
            reasons.append(f"controlled worst-case drawdown of {result.worst_case_drawdown_pct:.1%}")
        
        if not reasons:
            return f"Best balance of risk/return for {risk_tolerance} tolerance"
        
        return f"Policy {result.policy_id} recommended: {', '.join(reasons)}"
