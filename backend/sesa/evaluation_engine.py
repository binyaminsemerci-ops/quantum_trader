"""Evaluation Engine - Strategy Performance Assessment.

Evaluates mutated strategies using:
- Replay Engine v3 (historical backtesting)
- Scenario Simulator (forward-looking simulation)
- Risk metrics from AI Risk Officer models
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from backend.sesa.mutation_operators import MutatedStrategy, StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Strategy performance metrics.
    
    Comprehensive metrics calculated from backtest or simulation.
    """
    
    # Return metrics
    total_pnl: float
    total_return_pct: float
    avg_trade_pnl: float
    
    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    volatility: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Risk/Reward
    avg_win: float
    avg_loss: float
    profit_factor: float  # Total wins / Total losses
    expectancy: float  # Average PnL per trade
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # Return / Max DD
    
    # Tail risk
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR (95%)
    worst_trade_pnl: float
    best_trade_pnl: float
    
    # Consistency
    profit_factor_by_month: list[float] = field(default_factory=list)
    monthly_returns: list[float] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "avg_trade_pnl": self.avg_trade_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "volatility": self.volatility,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "worst_trade_pnl": self.worst_trade_pnl,
            "best_trade_pnl": self.best_trade_pnl,
            "monthly_returns": self.monthly_returns,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result for a strategy.
    """
    
    strategy_id: str
    mutation_id: Optional[str]
    
    # Performance metrics
    metrics: PerformanceMetrics
    
    # Evaluation metadata
    evaluation_type: str  # "backtest" or "simulation"
    evaluation_start: datetime
    evaluation_end: datetime
    evaluation_duration_seconds: float
    
    # Data quality
    data_points: int
    data_completeness: float
    
    # Risk assessment
    risk_score: float  # 0-100 from risk models
    risk_warnings: list[str] = field(default_factory=list)
    
    # Overall score (composite)
    composite_score: float = 0.0
    
    # Pass/Fail flags
    passed_minimum_requirements: bool = False
    reason_if_failed: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "mutation_id": self.mutation_id,
            "metrics": self.metrics.to_dict(),
            "evaluation_type": self.evaluation_type,
            "evaluation_start": self.evaluation_start.isoformat(),
            "evaluation_end": self.evaluation_end.isoformat(),
            "evaluation_duration_seconds": self.evaluation_duration_seconds,
            "data_points": self.data_points,
            "data_completeness": self.data_completeness,
            "risk_score": self.risk_score,
            "risk_warnings": self.risk_warnings,
            "composite_score": self.composite_score,
            "passed_minimum_requirements": self.passed_minimum_requirements,
            "reason_if_failed": self.reason_if_failed,
        }


class EvaluationEngine:
    """
    Evaluation Engine for strategy assessment.
    
    Features:
    - Backtest strategies using Replay Engine v3
    - Simulate strategies using Scenario Simulator
    - Calculate comprehensive performance metrics
    - Apply risk models for risk scoring
    - Compute composite scores for ranking
    
    Usage:
        engine = EvaluationEngine(
            replay_engine=replay_engine,
            scenario_simulator=simulator,
        )
        
        result = await engine.evaluate_strategy(
            strategy=mutated_strategy,
            evaluation_type="backtest",
            lookback_days=30,
        )
        
        if result.passed_minimum_requirements:
            print(f"Strategy passed: Score = {result.composite_score:.2f}")
        else:
            print(f"Strategy failed: {result.reason_if_failed}")
    """
    
    # Minimum requirements for a strategy to pass
    MIN_WIN_RATE = 0.45
    MIN_PROFIT_FACTOR = 1.0
    MIN_SHARPE = 0.0
    MAX_DRAWDOWN = 0.25
    MIN_TRADES = 10
    
    def __init__(
        self,
        replay_engine: Optional[Any] = None,
        scenario_simulator: Optional[Any] = None,
        risk_models: Optional[Any] = None,
    ):
        """
        Initialize Evaluation Engine.
        
        Args:
            replay_engine: ReplayEngine instance for backtesting
            scenario_simulator: ScenarioSimulator for forward testing
            risk_models: RiskModels instance for risk scoring
        """
        self.replay_engine = replay_engine
        self.scenario_simulator = scenario_simulator
        self.risk_models = risk_models
        
        logger.info("EvaluationEngine initialized")
    
    async def evaluate_strategy(
        self,
        strategy: MutatedStrategy | StrategyConfig,
        evaluation_type: str = "backtest",
        lookback_days: int = 30,
    ) -> EvaluationResult:
        """
        Evaluate a strategy.
        
        Args:
            strategy: Strategy to evaluate (MutatedStrategy or StrategyConfig)
            evaluation_type: "backtest" or "simulation"
            lookback_days: Days of historical data to use
        
        Returns:
            EvaluationResult
        """
        start_time = datetime.now()
        
        # Extract config
        if isinstance(strategy, MutatedStrategy):
            config = strategy.config
            mutation_id = strategy.mutation_id
            strategy_id = config.strategy_id
        else:
            config = strategy
            mutation_id = None
            strategy_id = config.strategy_id
        
        logger.info(
            f"Evaluating strategy: {strategy_id}, "
            f"type={evaluation_type}, lookback={lookback_days}d"
        )
        
        # Run evaluation
        if evaluation_type == "backtest":
            trades = await self._run_backtest(config, lookback_days)
        elif evaluation_type == "simulation":
            trades = await self._run_simulation(config, lookback_days)
        else:
            raise ValueError(f"Unknown evaluation_type: {evaluation_type}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades)
        
        # Calculate risk score
        risk_score, risk_warnings = self._calculate_risk_score(metrics, config)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(metrics, risk_score)
        
        # Check minimum requirements
        passed, reason = self._check_requirements(metrics)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = EvaluationResult(
            strategy_id=strategy_id,
            mutation_id=mutation_id,
            metrics=metrics,
            evaluation_type=evaluation_type,
            evaluation_start=start_time,
            evaluation_end=end_time,
            evaluation_duration_seconds=duration,
            data_points=len(trades),
            data_completeness=1.0 if len(trades) > 0 else 0.0,
            risk_score=risk_score,
            risk_warnings=risk_warnings,
            composite_score=composite_score,
            passed_minimum_requirements=passed,
            reason_if_failed=reason,
        )
        
        logger.info(
            f"Evaluation complete: {strategy_id}, "
            f"score={composite_score:.2f}, passed={passed}"
        )
        
        return result
    
    async def evaluate_batch(
        self,
        strategies: list[MutatedStrategy],
        evaluation_type: str = "backtest",
        lookback_days: int = 30,
        max_concurrent: int = 5,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple strategies concurrently.
        
        Args:
            strategies: List of strategies to evaluate
            evaluation_type: "backtest" or "simulation"
            lookback_days: Days of historical data
            max_concurrent: Maximum concurrent evaluations
        
        Returns:
            List of EvaluationResults
        """
        logger.info(f"Evaluating {len(strategies)} strategies in batch")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(strat):
            async with semaphore:
                return await self.evaluate_strategy(
                    strat,
                    evaluation_type,
                    lookback_days,
                )
        
        # Evaluate concurrently
        results = await asyncio.gather(
            *[evaluate_with_semaphore(s) for s in strategies],
            return_exceptions=True,
        )
        
        # Filter out exceptions
        valid_results = [
            r for r in results
            if isinstance(r, EvaluationResult)
        ]
        
        logger.info(
            f"Batch evaluation complete: {len(valid_results)}/{len(strategies)} succeeded"
        )
        
        return valid_results
    
    async def _run_backtest(
        self,
        config: StrategyConfig,
        lookback_days: int,
    ) -> list[dict]:
        """
        Run backtest using Replay Engine.
        
        Returns:
            List of simulated trades
        """
        if not self.replay_engine:
            # Fallback: Generate synthetic trades
            return self._generate_synthetic_trades(config, lookback_days)
        
        # TODO: Integration with actual Replay Engine v3
        # replay_result = await self.replay_engine.replay_strategy(
        #     strategy_config=config,
        #     start_date=datetime.now() - timedelta(days=lookback_days),
        #     end_date=datetime.now(),
        # )
        # return replay_result.trades
        
        return self._generate_synthetic_trades(config, lookback_days)
    
    async def _run_simulation(
        self,
        config: StrategyConfig,
        lookback_days: int,
    ) -> list[dict]:
        """
        Run forward simulation using Scenario Simulator.
        
        Returns:
            List of simulated trades
        """
        if not self.scenario_simulator:
            return self._generate_synthetic_trades(config, lookback_days)
        
        # TODO: Integration with actual Scenario Simulator
        # simulation_result = await self.scenario_simulator.simulate_strategy(
        #     strategy_config=config,
        #     num_scenarios=5,
        #     time_horizon_days=lookback_days,
        # )
        # return simulation_result.trades
        
        return self._generate_synthetic_trades(config, lookback_days)
    
    def _generate_synthetic_trades(
        self,
        config: StrategyConfig,
        lookback_days: int,
    ) -> list[dict]:
        """
        Generate synthetic trades for evaluation (fallback).
        
        This is a simplified simulation for demonstration.
        Real implementation uses Replay Engine or Scenario Simulator.
        """
        trades: list[dict] = []
        
        # Estimate trade count based on config
        trades_per_day = config.max_concurrent_positions * 1.5
        num_trades = int(trades_per_day * lookback_days)
        
        # Base win rate influenced by strategy parameters
        base_win_rate = 0.55
        
        # Adjust based on entry threshold (stricter = higher win rate)
        threshold_factor = (config.entry_confidence_threshold - 0.50) / 0.40
        win_rate = base_win_rate + (threshold_factor * 0.10)
        win_rate = np.clip(win_rate, 0.40, 0.70)
        
        # Generate trades
        for i in range(num_trades):
            is_win = np.random.random() < win_rate
            
            # Position size
            position_value = 10000 * config.position_size_pct
            
            if is_win:
                # Win: Use R:R ratio
                pnl = position_value * config.base_sl_pct * config.risk_reward_ratio
            else:
                # Loss: Base SL
                pnl = -position_value * config.base_sl_pct
            
            # Add some noise
            pnl *= np.random.uniform(0.8, 1.2)
            
            trades.append({
                "trade_id": f"trade_{i+1}",
                "pnl": pnl,
                "return_pct": pnl / position_value,
                "duration_minutes": np.random.randint(5, 120),
                "is_win": is_win,
            })
        
        return trades
    
    def _calculate_metrics(self, trades: list[dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from trades."""
        if not trades:
            return PerformanceMetrics(
                total_pnl=0,
                total_return_pct=0,
                avg_trade_pnl=0,
                max_drawdown_pct=0,
                max_drawdown_duration_days=0,
                volatility=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                expectancy=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                var_95=0,
                cvar_95=0,
                worst_trade_pnl=0,
                best_trade_pnl=0,
            )
        
        # Extract PnLs
        pnls = np.array([t["pnl"] for t in trades])
        returns = np.array([t.get("return_pct", 0) for t in trades])
        
        # Basic stats
        total_pnl = float(np.sum(pnls))
        avg_trade_pnl = float(np.mean(pnls))
        total_return_pct = float(np.sum(returns))
        
        # Win/Loss
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        total_trades = len(trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0
        
        total_wins = float(np.sum(wins)) if len(wins) > 0 else 0
        total_losses = float(abs(np.sum(losses))) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        expectancy = avg_trade_pnl
        
        # Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown_pct = float(np.max(drawdown) / 10000) if len(drawdown) > 0 else 0
        
        # Volatility
        volatility = float(np.std(returns)) if len(returns) > 1 else 0
        
        # Risk-adjusted returns
        sharpe = (avg_trade_pnl / np.std(pnls)) if np.std(pnls) > 0 else 0
        sharpe = float(sharpe) * np.sqrt(252)  # Annualize
        
        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-6
        sortino = float(avg_trade_pnl / (downside_std * np.std(pnls))) * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar
        calmar = float(total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0
        
        # Tail risk
        var_95 = float(np.percentile(pnls, 5))
        cvar_95 = float(np.mean(pnls[pnls <= var_95]))
        worst_trade_pnl = float(np.min(pnls))
        best_trade_pnl = float(np.max(pnls))
        
        return PerformanceMetrics(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_days=0,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            var_95=var_95,
            cvar_95=cvar_95,
            worst_trade_pnl=worst_trade_pnl,
            best_trade_pnl=best_trade_pnl,
        )
    
    def _calculate_risk_score(
        self,
        metrics: PerformanceMetrics,
        config: StrategyConfig,
    ) -> tuple[float, list[str]]:
        """
        Calculate risk score (0-100) and identify warnings.
        
        Returns:
            (risk_score, warnings)
        """
        risk_score = 0.0
        warnings: list[str] = []
        
        # Drawdown risk (30 points)
        if metrics.max_drawdown_pct > 0.20:
            risk_score += 30
            warnings.append(f"High drawdown: {metrics.max_drawdown_pct:.1%}")
        elif metrics.max_drawdown_pct > 0.15:
            risk_score += 20
        elif metrics.max_drawdown_pct > 0.10:
            risk_score += 10
        
        # Tail risk (25 points)
        if abs(metrics.worst_trade_pnl) > 500:
            risk_score += 25
            warnings.append(f"Large loss risk: ${metrics.worst_trade_pnl:.2f}")
        elif abs(metrics.worst_trade_pnl) > 300:
            risk_score += 15
        
        # Volatility risk (20 points)
        if metrics.volatility > 0.05:
            risk_score += 20
            warnings.append(f"High volatility: {metrics.volatility:.2%}")
        elif metrics.volatility > 0.03:
            risk_score += 10
        
        # Low win rate (15 points)
        if metrics.win_rate < 0.40:
            risk_score += 15
            warnings.append(f"Low win rate: {metrics.win_rate:.1%}")
        elif metrics.win_rate < 0.45:
            risk_score += 8
        
        # Poor profit factor (10 points)
        if metrics.profit_factor < 1.0:
            risk_score += 10
            warnings.append(f"Negative edge: PF={metrics.profit_factor:.2f}")
        elif metrics.profit_factor < 1.2:
            risk_score += 5
        
        return risk_score, warnings
    
    def _calculate_composite_score(
        self,
        metrics: PerformanceMetrics,
        risk_score: float,
    ) -> float:
        """
        Calculate composite score for ranking strategies.
        
        Score components:
        - Total PnL: 25%
        - Sharpe Ratio: 20%
        - Win Rate: 15%
        - Profit Factor: 15%
        - Drawdown (inverted): 15%
        - Risk Score (inverted): 10%
        """
        # Normalize components to 0-100 scale
        pnl_score = np.clip(metrics.total_pnl / 100, 0, 100)
        sharpe_score = np.clip(metrics.sharpe_ratio * 20, 0, 100)
        winrate_score = metrics.win_rate * 100
        pf_score = np.clip((metrics.profit_factor - 1.0) * 50, 0, 100)
        dd_score = np.clip((1 - metrics.max_drawdown_pct * 5) * 100, 0, 100)
        risk_score_inverted = 100 - risk_score
        
        # Weighted composite
        composite = (
            pnl_score * 0.25
            + sharpe_score * 0.20
            + winrate_score * 0.15
            + pf_score * 0.15
            + dd_score * 0.15
            + risk_score_inverted * 0.10
        )
        
        return float(composite)
    
    def _check_requirements(
        self,
        metrics: PerformanceMetrics,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if strategy meets minimum requirements.
        
        Returns:
            (passed, reason_if_failed)
        """
        if metrics.total_trades < self.MIN_TRADES:
            return False, f"Insufficient trades: {metrics.total_trades} < {self.MIN_TRADES}"
        
        if metrics.win_rate < self.MIN_WIN_RATE:
            return False, f"Low win rate: {metrics.win_rate:.1%} < {self.MIN_WIN_RATE:.1%}"
        
        if metrics.profit_factor < self.MIN_PROFIT_FACTOR:
            return False, f"Negative edge: PF={metrics.profit_factor:.2f} < {self.MIN_PROFIT_FACTOR}"
        
        if metrics.sharpe_ratio < self.MIN_SHARPE:
            return False, f"Low Sharpe: {metrics.sharpe_ratio:.2f} < {self.MIN_SHARPE}"
        
        if metrics.max_drawdown_pct > self.MAX_DRAWDOWN:
            return False, f"Excessive drawdown: {metrics.max_drawdown_pct:.1%} > {self.MAX_DRAWDOWN:.1%}"
        
        return True, None
