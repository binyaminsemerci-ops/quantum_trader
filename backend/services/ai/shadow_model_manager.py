"""
Shadow Model Manager - Parallel Challenger Testing with Automatic Promotion

Enables zero-risk testing of challenger models against production champion through:
- Pure shadow mode (0% allocation) with performance tracking
- Statistical hypothesis testing (t-test, bootstrap, Sharpe comparison)
- Automatic promotion based on rigorous criteria (p<0.05, risk-adjusted performance)
- Thompson sampling for adaptive allocation (optional exploratory mode)
- Rollback protection (monitor first 100 trades post-promotion)

Architecture:
- ShadowModelManager: Main orchestrator
- PerformanceTracker: Track PnL, WR, Sharpe, MDD for each model
- StatisticalTester: T-test, bootstrap CI, Sharpe comparison
- PromotionEngine: Criteria checking, scoring, auto-promote
- ThompsonSampling: Multi-armed bandit allocation (optional)

Integration:
- Called by AITradingEngine after each prediction
- Stores shadow predictions alongside champion predictions
- Triggers statistical tests every 100 trades
- Auto-promotes when criteria met

File: backend/services/ai/shadow_model_manager.py
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from scipy import stats
from scipy.stats import norm
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ModelRole(Enum):
    """Role of model in shadow testing system"""
    CHAMPION = "champion"  # Production model (100% allocation)
    CHALLENGER = "challenger"  # Testing model (0% or ε% allocation)
    ARCHIVE = "archive"  # Previous champion (available for rollback)


class PromotionStatus(Enum):
    """Status of promotion decision"""
    PENDING = "pending"  # Insufficient data for decision
    REJECTED = "rejected"  # Challenger failed criteria
    APPROVED = "approved"  # Challenger passed, ready for promotion
    PROMOTED = "promoted"  # Challenger promoted to champion
    ROLLED_BACK = "rolled_back"  # Promotion failed, rolled back


@dataclass
class ModelMetadata:
    """Metadata for a model in the shadow system"""
    model_name: str
    model_type: str  # 'xgboost', 'lightgbm', 'catboost', 'neural_network'
    version: str
    role: ModelRole
    deployed_at: datetime
    allocation: float  # 0.0-1.0 (percentage of trades)
    description: str = ""
    
    def to_dict(self):
        data = asdict(self)
        data['role'] = self.role.value
        data['deployed_at'] = self.deployed_at.isoformat()
        return data


@dataclass
class TradeResult:
    """Single trade result for performance tracking"""
    timestamp: datetime
    model_name: str
    prediction: int  # 1 (long), -1 (short), 0 (neutral)
    actual_outcome: int  # 1 (win), 0 (loss)
    pnl: float
    confidence: float
    executed: bool  # True if actually executed (champion), False if shadow
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
@dataclass
class PerformanceMetrics:
    """Performance metrics for a model"""
    model_name: str
    n_trades: int
    win_rate: float
    mean_pnl: float
    std_pnl: float
    total_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    rolling_wr_std: float  # Consistency metric
    last_updated: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class StatisticalTestResults:
    """Results of statistical hypothesis tests"""
    timestamp: datetime
    champion_name: str
    challenger_name: str
    
    # T-test
    t_statistic: float
    t_p_value: float
    t_test_passed: bool
    
    # Bootstrap CI
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_passed: bool
    
    # Sharpe comparison
    sharpe_z_statistic: float
    sharpe_p_value: float
    sharpe_test_passed: bool
    
    # Win rate comparison
    wr_z_statistic: float
    wr_p_value: float
    wr_test_passed: bool
    
    # Overall
    statistical_significance: bool
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PromotionDecision:
    """Promotion decision with detailed reasoning"""
    timestamp: datetime
    champion_name: str
    challenger_name: str
    
    # Criteria results
    statistical_significance: bool
    sharpe_criterion: bool
    sample_size_criterion: bool
    mdd_criterion: bool
    win_rate_criterion: bool
    
    # Secondary criteria
    consistency_criterion: bool
    diversity_criterion: bool
    sortino_criterion: bool
    
    # Scoring
    promotion_score: float  # 0-100
    
    # Decision
    status: PromotionStatus
    reason: str
    
    # Metrics comparison
    champion_metrics: Dict
    challenger_metrics: Dict
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class PromotionEvent:
    """Record of a promotion event"""
    timestamp: datetime
    old_champion: str
    new_champion: str
    promotion_score: float
    reason: str
    performance_improvement: Dict  # {'wr': +0.02, 'sharpe': +0.3, ...}
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Track performance metrics for all models"""
    
    def __init__(self):
        self.trade_history: Dict[str, List[TradeResult]] = {}  # model_name → trades
        self.performance_cache: Dict[str, PerformanceMetrics] = {}  # model_name → metrics
    
    def record_trade(self, trade: TradeResult):
        """Record a trade result"""
        if trade.model_name not in self.trade_history:
            self.trade_history[trade.model_name] = []
        
        self.trade_history[trade.model_name].append(trade)
        
        # Invalidate cache
        if trade.model_name in self.performance_cache:
            del self.performance_cache[trade.model_name]
    
    def compute_metrics(self, model_name: str) -> Optional[PerformanceMetrics]:
        """Compute performance metrics for a model"""
        if model_name not in self.trade_history:
            return None
        
        # Check cache
        if model_name in self.performance_cache:
            return self.performance_cache[model_name]
        
        trades = self.trade_history[model_name]
        if len(trades) == 0:
            return None
        
        # Extract data
        pnls = np.array([t.pnl for t in trades])
        outcomes = np.array([t.actual_outcome for t in trades])
        
        # Win rate
        win_rate = outcomes.mean()
        
        # PnL statistics
        mean_pnl = pnls.mean()
        std_pnl = pnls.std()
        total_pnl = pnls.sum()
        
        # Sharpe ratio (assuming risk-free rate ≈ 0)
        sharpe_ratio = mean_pnl / std_pnl if std_pnl > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_pnls = pnls[pnls < 0]
        downside_std = downside_pnls.std() if len(downside_pnls) > 0 else std_pnl
        sortino_ratio = mean_pnl / downside_std if downside_std > 0 else 0
        
        # Maximum drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = drawdowns.max()
        
        # Rolling win rate consistency
        if len(outcomes) >= 100:
            rolling_wrs = []
            for i in range(len(outcomes) - 100 + 1):
                window_wr = outcomes[i:i+100].mean()
                rolling_wrs.append(window_wr)
            rolling_wr_std = np.std(rolling_wrs)
        else:
            rolling_wr_std = 0.0
        
        metrics = PerformanceMetrics(
            model_name=model_name,
            n_trades=len(trades),
            win_rate=win_rate,
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            rolling_wr_std=rolling_wr_std,
            last_updated=datetime.utcnow()
        )
        
        # Cache
        self.performance_cache[model_name] = metrics
        
        return metrics
    
    def get_pnls(self, model_name: str) -> np.ndarray:
        """Get PnL array for statistical testing"""
        if model_name not in self.trade_history:
            return np.array([])
        return np.array([t.pnl for t in self.trade_history[model_name]])
    
    def get_trade_count(self, model_name: str) -> int:
        """Get number of trades for a model"""
        if model_name not in self.trade_history:
            return 0
        return len(self.trade_history[model_name])


# ============================================================================
# STATISTICAL TESTER
# ============================================================================

class StatisticalTester:
    """Perform statistical hypothesis tests"""
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
    
    def t_test(self, champion_pnls: np.ndarray, challenger_pnls: np.ndarray) -> Tuple[float, float, bool]:
        """
        Two-sample t-test for mean PnL comparison
        
        Returns:
            (t_statistic, p_value, passed)
        """
        if len(champion_pnls) == 0 or len(challenger_pnls) == 0:
            return 0.0, 1.0, False
        
        t_stat, p_value = stats.ttest_ind(challenger_pnls, champion_pnls, equal_var=False)
        
        # One-tailed test (challenger > champion)
        p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        
        passed = p_value_one_tailed < self.alpha
        
        return t_stat, p_value_one_tailed, passed
    
    def bootstrap_ci(self, champion_pnls: np.ndarray, challenger_pnls: np.ndarray) -> Tuple[float, float, bool]:
        """
        Bootstrap confidence interval for mean difference
        
        Returns:
            (ci_lower, ci_upper, passed)
        """
        if len(champion_pnls) == 0 or len(challenger_pnls) == 0:
            return 0.0, 0.0, False
        
        diffs = []
        
        n_champion = len(champion_pnls)
        n_challenger = len(challenger_pnls)
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            sample_champion = np.random.choice(champion_pnls, size=n_champion, replace=True)
            sample_challenger = np.random.choice(challenger_pnls, size=n_challenger, replace=True)
            
            # Compute difference
            diff = sample_challenger.mean() - sample_champion.mean()
            diffs.append(diff)
        
        # Compute 95% CI
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        
        # Passed if 0 not in CI (challenger significantly better)
        passed = ci_lower > 0
        
        return ci_lower, ci_upper, passed
    
    def sharpe_test(self, champion_metrics: PerformanceMetrics, challenger_metrics: PerformanceMetrics) -> Tuple[float, float, bool]:
        """
        Jobson-Korkie test for Sharpe ratio comparison
        
        Returns:
            (z_statistic, p_value, passed)
        """
        sharpe_champ = champion_metrics.sharpe_ratio
        sharpe_chal = challenger_metrics.sharpe_ratio
        
        n = min(champion_metrics.n_trades, challenger_metrics.n_trades)
        
        if n < 30:
            return 0.0, 1.0, False
        
        # Assume ρ ≈ 0 (uncorrelated, shadow testing)
        rho = 0.0
        
        se_diff = np.sqrt((1 / n) * (2 - rho**2))
        
        z_stat = (sharpe_chal - sharpe_champ) / se_diff
        
        # One-tailed test
        p_value = 1 - norm.cdf(z_stat)
        
        passed = p_value < self.alpha
        
        return z_stat, p_value, passed
    
    def win_rate_test(self, champion_metrics: PerformanceMetrics, challenger_metrics: PerformanceMetrics) -> Tuple[float, float, bool]:
        """
        Z-test for proportions (win rate comparison)
        
        Returns:
            (z_statistic, p_value, passed)
        """
        wr_champ = champion_metrics.win_rate
        wr_chal = challenger_metrics.win_rate
        
        n_champ = champion_metrics.n_trades
        n_chal = challenger_metrics.n_trades
        
        if n_champ < 30 or n_chal < 30:
            return 0.0, 1.0, False
        
        # Pooled proportion
        k_champ = int(wr_champ * n_champ)
        k_chal = int(wr_chal * n_chal)
        
        wr_pooled = (k_champ + k_chal) / (n_champ + n_chal)
        
        se_diff = np.sqrt(wr_pooled * (1 - wr_pooled) * (1/n_champ + 1/n_chal))
        
        if se_diff == 0:
            return 0.0, 1.0, False
        
        z_stat = (wr_chal - wr_champ) / se_diff
        
        # One-tailed test
        p_value = 1 - norm.cdf(z_stat)
        
        passed = p_value < self.alpha
        
        return z_stat, p_value, passed
    
    def run_all_tests(self, champion_name: str, challenger_name: str,
                      champion_metrics: PerformanceMetrics, challenger_metrics: PerformanceMetrics,
                      champion_pnls: np.ndarray, challenger_pnls: np.ndarray) -> StatisticalTestResults:
        """Run all statistical tests"""
        
        # T-test
        t_stat, t_p_value, t_passed = self.t_test(champion_pnls, challenger_pnls)
        
        # Bootstrap CI
        ci_lower, ci_upper, bootstrap_passed = self.bootstrap_ci(champion_pnls, challenger_pnls)
        
        # Sharpe test
        sharpe_z, sharpe_p, sharpe_passed = self.sharpe_test(champion_metrics, challenger_metrics)
        
        # Win rate test
        wr_z, wr_p, wr_passed = self.win_rate_test(champion_metrics, challenger_metrics)
        
        # Overall statistical significance (at least one test passed)
        statistical_significance = t_passed or bootstrap_passed or sharpe_passed
        
        return StatisticalTestResults(
            timestamp=datetime.utcnow(),
            champion_name=champion_name,
            challenger_name=challenger_name,
            t_statistic=t_stat,
            t_p_value=t_p_value,
            t_test_passed=t_passed,
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper,
            bootstrap_passed=bootstrap_passed,
            sharpe_z_statistic=sharpe_z,
            sharpe_p_value=sharpe_p,
            sharpe_test_passed=sharpe_passed,
            wr_z_statistic=wr_z,
            wr_p_value=wr_p,
            wr_test_passed=wr_passed,
            statistical_significance=statistical_significance
        )


# ============================================================================
# PROMOTION ENGINE
# ============================================================================

class PromotionEngine:
    """Check promotion criteria and make promotion decisions"""
    
    def __init__(self, min_trades: int = 500, mdd_tolerance: float = 1.20):
        self.min_trades = min_trades
        self.mdd_tolerance = mdd_tolerance
    
    def check_criteria(self, champion_metrics: PerformanceMetrics, challenger_metrics: PerformanceMetrics,
                       test_results: StatisticalTestResults) -> PromotionDecision:
        """
        Check all promotion criteria
        
        Primary criteria (all must pass):
        1. Statistical significance (p < 0.05)
        2. Sharpe ratio ≥ champion
        3. Minimum sample size (500 trades)
        4. Max drawdown ≤ champion * 1.20
        5. Win rate ≥ 0.50
        
        Secondary criteria (nice-to-have):
        6. Consistency (rolling WR std < 0.05)
        7. Sortino ratio ≥ champion
        """
        
        # Primary criteria
        stat_sig = test_results.statistical_significance
        sharpe_ok = challenger_metrics.sharpe_ratio >= champion_metrics.sharpe_ratio
        sample_ok = challenger_metrics.n_trades >= self.min_trades
        mdd_ok = challenger_metrics.max_drawdown <= champion_metrics.max_drawdown * self.mdd_tolerance
        wr_ok = challenger_metrics.win_rate >= 0.50
        
        # Secondary criteria
        consistency_ok = challenger_metrics.rolling_wr_std < 0.05
        sortino_ok = challenger_metrics.sortino_ratio >= champion_metrics.sortino_ratio
        diversity_ok = True  # Placeholder (requires correlation computation)
        
        # Compute promotion score
        score = self._compute_score(champion_metrics, challenger_metrics, test_results)
        
        # Determine status
        primary_passed = stat_sig and sharpe_ok and sample_ok and mdd_ok and wr_ok
        
        if not primary_passed:
            status = PromotionStatus.REJECTED
            reason = self._generate_rejection_reason(stat_sig, sharpe_ok, sample_ok, mdd_ok, wr_ok)
        elif score >= 70:
            status = PromotionStatus.APPROVED
            reason = f"All criteria passed with score {score:.1f}/100. Ready for auto-promotion."
        elif score >= 50:
            status = PromotionStatus.PENDING
            reason = f"Moderate evidence (score {score:.1f}/100). Manual review recommended."
        else:
            status = PromotionStatus.REJECTED
            reason = f"Insufficient evidence (score {score:.1f}/100). Need stronger performance."
        
        return PromotionDecision(
            timestamp=datetime.utcnow(),
            champion_name=champion_metrics.model_name,
            challenger_name=challenger_metrics.model_name,
            statistical_significance=stat_sig,
            sharpe_criterion=sharpe_ok,
            sample_size_criterion=sample_ok,
            mdd_criterion=mdd_ok,
            win_rate_criterion=wr_ok,
            consistency_criterion=consistency_ok,
            diversity_criterion=diversity_ok,
            sortino_criterion=sortino_ok,
            promotion_score=score,
            status=status,
            reason=reason,
            champion_metrics={
                'win_rate': champion_metrics.win_rate,
                'sharpe_ratio': champion_metrics.sharpe_ratio,
                'mean_pnl': champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown
            },
            challenger_metrics={
                'win_rate': challenger_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl,
                'max_drawdown': challenger_metrics.max_drawdown
            }
        )
    
    def _compute_score(self, champion_metrics: PerformanceMetrics, challenger_metrics: PerformanceMetrics,
                       test_results: StatisticalTestResults) -> float:
        """Compute 0-100 promotion score"""
        score = 0.0
        
        # Statistical significance (30 points)
        if test_results.t_p_value < 0.01:
            score += 30
        elif test_results.t_p_value < 0.05:
            score += 20
        elif test_results.bootstrap_passed:
            score += 15
        
        # Sharpe ratio improvement (25 points)
        if champion_metrics.sharpe_ratio > 0:
            sharpe_improvement = (challenger_metrics.sharpe_ratio - champion_metrics.sharpe_ratio) / champion_metrics.sharpe_ratio
            score += min(25, max(0, sharpe_improvement * 100))
        
        # Win rate improvement (20 points)
        wr_improvement = (challenger_metrics.win_rate - champion_metrics.win_rate)
        score += min(20, max(0, wr_improvement * 400))  # +5pp = 20 points
        
        # MDD improvement (15 points)
        if champion_metrics.max_drawdown > 0:
            mdd_improvement = (champion_metrics.max_drawdown - challenger_metrics.max_drawdown) / champion_metrics.max_drawdown
            score += min(15, max(0, mdd_improvement * 30))
        
        # Consistency (10 points)
        if challenger_metrics.rolling_wr_std < 0.05:
            score += 10
        elif challenger_metrics.rolling_wr_std < 0.07:
            score += 5
        
        return score
    
    def _generate_rejection_reason(self, stat_sig: bool, sharpe_ok: bool, sample_ok: bool, 
                                   mdd_ok: bool, wr_ok: bool) -> str:
        """Generate human-readable rejection reason"""
        reasons = []
        
        if not stat_sig:
            reasons.append("Statistical significance not achieved (p ≥ 0.05)")
        if not sharpe_ok:
            reasons.append("Sharpe ratio lower than champion")
        if not sample_ok:
            reasons.append(f"Insufficient sample size (need {self.min_trades} trades)")
        if not mdd_ok:
            reasons.append(f"Max drawdown exceeds tolerance ({self.mdd_tolerance}x champion)")
        if not wr_ok:
            reasons.append("Win rate below 50% (unprofitable)")
        
        return "; ".join(reasons)


# ============================================================================
# THOMPSON SAMPLING (OPTIONAL EXPLORATORY ALLOCATION)
# ============================================================================

class ThompsonSampling:
    """Multi-armed bandit for adaptive allocation"""
    
    def __init__(self):
        # Bayesian belief: μ ~ N(m, s²)
        self.means: Dict[str, float] = {}  # m_i (posterior mean)
        self.stds: Dict[str, float] = {}   # s_i (posterior std)
        self.observation_noise: float = 100.0  # σ² (PnL noise)
    
    def initialize_model(self, model_name: str, prior_mean: float = 50.0, prior_std: float = 50.0):
        """Initialize belief for a new model"""
        self.means[model_name] = prior_mean
        self.stds[model_name] = prior_std
    
    def update(self, model_name: str, observed_pnl: float):
        """Bayesian update after observing PnL"""
        if model_name not in self.means:
            self.initialize_model(model_name)
        
        # Prior
        m_prior = self.means[model_name]
        s_prior = self.stds[model_name]
        
        # Likelihood: x ~ N(μ, σ²)
        sigma_sq = self.observation_noise
        
        # Posterior (Bayesian update)
        s_post_sq = 1 / (1/(s_prior**2) + 1/sigma_sq)
        m_post = s_post_sq * (m_prior/(s_prior**2) + observed_pnl/sigma_sq)
        
        self.means[model_name] = m_post
        self.stds[model_name] = np.sqrt(s_post_sq)
    
    def sample_allocation(self, model_names: List[str]) -> str:
        """Sample allocation using Thompson sampling"""
        if len(model_names) == 0:
            return None
        
        # Sample from each posterior
        samples = {}
        for name in model_names:
            if name not in self.means:
                self.initialize_model(name)
            
            sample = np.random.normal(self.means[name], self.stds[name])
            samples[name] = sample
        
        # Select model with highest sample
        selected = max(samples, key=samples.get)
        
        return selected


# ============================================================================
# SHADOW MODEL MANAGER (MAIN CLASS)
# ============================================================================

class ShadowModelManager:
    """
    Main orchestrator for shadow model testing and promotion
    
    Workflow:
    1. Deploy challenger as shadow (0% allocation)
    2. Track performance in parallel with champion
    3. Run statistical tests every 100 trades
    4. Auto-promote when criteria met (score ≥ 70)
    5. Monitor post-promotion (first 100 trades)
    6. Rollback if new champion degrades
    """
    
    def __init__(self,
                 min_trades_for_promotion: int = 500,
                 mdd_tolerance: float = 1.20,
                 alpha: float = 0.05,
                 n_bootstrap: int = 10000,
                 checkpoint_path: str = 'data/shadow_models_checkpoint.json'):
        
        self.min_trades = min_trades_for_promotion
        self.checkpoint_path = checkpoint_path
        
        # Components
        self.performance_tracker = PerformanceTracker()
        self.statistical_tester = StatisticalTester(alpha=alpha, n_bootstrap=n_bootstrap)
        self.promotion_engine = PromotionEngine(min_trades=min_trades_for_promotion, mdd_tolerance=mdd_tolerance)
        self.thompson_sampling = ThompsonSampling()
        
        # State
        self.models: Dict[str, ModelMetadata] = {}  # model_name → metadata
        self.test_results_history: List[StatisticalTestResults] = []
        self.promotion_history: List[PromotionEvent] = []
        self.pending_decisions: Dict[str, PromotionDecision] = {}  # challenger_name → decision
        
        # Champion tracking
        self.current_champion: Optional[str] = None
        self.trades_since_promotion: int = 0
        self.promotion_baseline_wr: float = 0.0
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        logger.info(f"ShadowModelManager initialized: min_trades={min_trades_for_promotion}, alpha={alpha}")
    
    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================
    
    def register_model(self, model_name: str, model_type: str, version: str, role: ModelRole, description: str = ""):
        """Register a model in the shadow system"""
        allocation = 1.0 if role == ModelRole.CHAMPION else 0.0
        
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version=version,
            role=role,
            deployed_at=datetime.utcnow(),
            allocation=allocation,
            description=description
        )
        
        self.models[model_name] = metadata
        
        # Set as champion if first model
        if role == ModelRole.CHAMPION:
            self.current_champion = model_name
            self.trades_since_promotion = 0
        
        # Initialize Thompson sampling
        self.thompson_sampling.initialize_model(model_name)
        
        logger.info(f"Registered model: {model_name} ({role.value}, allocation={allocation:.2%})")
    
    def get_champion(self) -> Optional[str]:
        """Get current champion model name"""
        return self.current_champion
    
    def get_challengers(self) -> List[str]:
        """Get list of active challenger model names"""
        return [name for name, meta in self.models.items() if meta.role == ModelRole.CHALLENGER]
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a model"""
        return self.models.get(model_name)
    
    # ========================================================================
    # PERFORMANCE TRACKING
    # ========================================================================
    
    def record_prediction(self, model_name: str, prediction: int, actual_outcome: int, 
                         pnl: float, confidence: float, executed: bool = False):
        """Record a prediction and outcome"""
        trade = TradeResult(
            timestamp=datetime.utcnow(),
            model_name=model_name,
            prediction=prediction,
            actual_outcome=actual_outcome,
            pnl=pnl,
            confidence=confidence,
            executed=executed
        )
        
        self.performance_tracker.record_trade(trade)
        
        # Update Thompson sampling
        self.thompson_sampling.update(model_name, pnl)
        
        # If champion, increment trades since promotion
        if model_name == self.current_champion:
            self.trades_since_promotion += 1
            
            # Check for post-promotion degradation
            if self.trades_since_promotion <= 100:
                self._check_post_promotion_health()
    
    def get_metrics(self, model_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a model"""
        return self.performance_tracker.compute_metrics(model_name)
    
    def get_trade_count(self, model_name: str) -> int:
        """Get number of trades for a model"""
        return self.performance_tracker.get_trade_count(model_name)
    
    # ========================================================================
    # STATISTICAL TESTING
    # ========================================================================
    
    def run_statistical_tests(self, challenger_name: str) -> Optional[StatisticalTestResults]:
        """
        Run statistical tests comparing challenger to champion
        
        Should be called every 100 trades or when trade count ≥ min_trades
        """
        if self.current_champion is None:
            logger.warning("No champion set, cannot run tests")
            return None
        
        champion_name = self.current_champion
        
        # Get metrics
        champion_metrics = self.get_metrics(champion_name)
        challenger_metrics = self.get_metrics(challenger_name)
        
        if champion_metrics is None or challenger_metrics is None:
            logger.warning(f"Insufficient data for testing: {champion_name}, {challenger_name}")
            return None
        
        # Get PnL arrays
        champion_pnls = self.performance_tracker.get_pnls(champion_name)
        challenger_pnls = self.performance_tracker.get_pnls(challenger_name)
        
        # Run tests
        test_results = self.statistical_tester.run_all_tests(
            champion_name=champion_name,
            challenger_name=challenger_name,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            champion_pnls=champion_pnls,
            challenger_pnls=challenger_pnls
        )
        
        # Store results
        self.test_results_history.append(test_results)
        
        logger.info(
            f"Statistical tests completed: {challenger_name} vs {champion_name} | "
            f"t_test: {test_results.t_test_passed}, bootstrap: {test_results.bootstrap_passed}, "
            f"sharpe: {test_results.sharpe_test_passed}, overall: {test_results.statistical_significance}"
        )
        
        return test_results
    
    # ========================================================================
    # PROMOTION LOGIC
    # ========================================================================
    
    def check_promotion_criteria(self, challenger_name: str) -> Optional[PromotionDecision]:
        """
        Check if challenger meets promotion criteria
        
        Returns PromotionDecision with status:
        - APPROVED: Ready for auto-promotion
        - PENDING: Manual review needed
        - REJECTED: Failed criteria
        """
        if self.current_champion is None:
            logger.warning("No champion set, cannot check promotion")
            return None
        
        # Run statistical tests
        test_results = self.run_statistical_tests(challenger_name)
        if test_results is None:
            return None
        
        # Get metrics
        champion_metrics = self.get_metrics(self.current_champion)
        challenger_metrics = self.get_metrics(challenger_name)
        
        # Check criteria
        decision = self.promotion_engine.check_criteria(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            test_results=test_results
        )
        
        # Store decision
        self.pending_decisions[challenger_name] = decision
        
        logger.info(
            f"Promotion check: {challenger_name} | Status: {decision.status.value} | "
            f"Score: {decision.promotion_score:.1f}/100 | Reason: {decision.reason}"
        )
        
        return decision
    
    def promote_challenger(self, challenger_name: str, force: bool = False) -> bool:
        """
        Promote challenger to champion
        
        Args:
            challenger_name: Name of challenger to promote
            force: Skip criteria check (manual promotion)
        
        Returns:
            True if promotion successful, False otherwise
        """
        if challenger_name not in self.models:
            logger.error(f"Model {challenger_name} not registered")
            return False
        
        # Check criteria unless forced
        if not force:
            decision = self.check_promotion_criteria(challenger_name)
            if decision is None or decision.status != PromotionStatus.APPROVED:
                logger.warning(f"Challenger {challenger_name} not ready for promotion")
                return False
        
        old_champion = self.current_champion
        
        # Archive old champion
        if old_champion:
            self.models[old_champion].role = ModelRole.ARCHIVE
            self.models[old_champion].allocation = 0.0
        
        # Promote challenger
        self.models[challenger_name].role = ModelRole.CHAMPION
        self.models[challenger_name].allocation = 1.0
        
        # Update state
        self.current_champion = challenger_name
        self.trades_since_promotion = 0
        
        # Get metrics for promotion event
        challenger_metrics = self.get_metrics(challenger_name)
        champion_metrics = self.get_metrics(old_champion) if old_champion else None
        
        if challenger_metrics and champion_metrics:
            self.promotion_baseline_wr = challenger_metrics.win_rate
            
            performance_improvement = {
                'wr': challenger_metrics.win_rate - champion_metrics.win_rate,
                'sharpe': challenger_metrics.sharpe_ratio - champion_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl - champion_metrics.mean_pnl,
                'mdd': champion_metrics.max_drawdown - challenger_metrics.max_drawdown
            }
        else:
            self.promotion_baseline_wr = 0.56  # Default
            performance_improvement = {}
        
        # Record promotion event
        decision = self.pending_decisions.get(challenger_name)
        score = decision.promotion_score if decision else 0.0
        reason = decision.reason if decision else "Manual promotion"
        
        event = PromotionEvent(
            timestamp=datetime.utcnow(),
            old_champion=old_champion,
            new_champion=challenger_name,
            promotion_score=score,
            reason=reason,
            performance_improvement=performance_improvement
        )
        
        self.promotion_history.append(event)
        
        # Update decision status
        if challenger_name in self.pending_decisions:
            self.pending_decisions[challenger_name].status = PromotionStatus.PROMOTED
        
        # Checkpoint
        self.checkpoint()
        
        logger.info(
            f"🎉 PROMOTION: {challenger_name} promoted to champion (replacing {old_champion}) | "
            f"Score: {score:.1f}/100 | Improvement: WR +{performance_improvement.get('wr', 0):.2%}, "
            f"Sharpe +{performance_improvement.get('sharpe', 0):.2f}"
        )
        
        return True
    
    def _check_post_promotion_health(self):
        """Check if new champion is degrading post-promotion"""
        if self.trades_since_promotion > 100:
            return  # Past monitoring window
        
        if self.current_champion is None:
            return
        
        metrics = self.get_metrics(self.current_champion)
        if metrics is None:
            return
        
        # Check if WR dropped >3pp from promotion baseline
        wr_drop = self.promotion_baseline_wr - metrics.win_rate
        
        if wr_drop > 0.03:
            logger.critical(
                f"⚠️ POST-PROMOTION DEGRADATION DETECTED: {self.current_champion} | "
                f"WR dropped {wr_drop:.2%} from baseline {self.promotion_baseline_wr:.2%} → {metrics.win_rate:.2%} | "
                f"Trades since promotion: {self.trades_since_promotion}/100"
            )
            
            # Trigger rollback if severe (>5pp drop)
            if wr_drop > 0.05:
                logger.critical(f"Triggering automatic rollback due to severe degradation")
                self.rollback_to_previous_champion(reason="Post-promotion WR drop >5pp")
    
    def rollback_to_previous_champion(self, reason: str = "Manual rollback"):
        """Rollback to previous champion"""
        if len(self.promotion_history) == 0:
            logger.error("No promotion history, cannot rollback")
            return False
        
        last_promotion = self.promotion_history[-1]
        previous_champion = last_promotion.old_champion
        
        if previous_champion not in self.models:
            logger.error(f"Previous champion {previous_champion} not found in archives")
            return False
        
        current_champion = self.current_champion
        
        # Restore previous champion
        self.models[previous_champion].role = ModelRole.CHAMPION
        self.models[previous_champion].allocation = 1.0
        
        # Demote current champion to challenger
        if current_champion:
            self.models[current_champion].role = ModelRole.CHALLENGER
            self.models[current_champion].allocation = 0.0
        
        # Update state
        self.current_champion = previous_champion
        self.trades_since_promotion = 0
        
        # Update promotion status
        if current_champion in self.pending_decisions:
            self.pending_decisions[current_champion].status = PromotionStatus.ROLLED_BACK
        
        # Checkpoint
        self.checkpoint()
        
        logger.critical(
            f"🔄 ROLLBACK: Restored {previous_champion} as champion (rolled back {current_champion}) | "
            f"Reason: {reason}"
        )
        
        return True
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_promotion_history(self, n: int = 10) -> List[PromotionEvent]:
        """Get last N promotion events"""
        return self.promotion_history[-n:]
    
    def get_test_results_history(self, challenger_name: str, n: int = 10) -> List[StatisticalTestResults]:
        """Get last N test results for a challenger"""
        results = [r for r in self.test_results_history if r.challenger_name == challenger_name]
        return results[-n:]
    
    def get_pending_decision(self, challenger_name: str) -> Optional[PromotionDecision]:
        """Get pending promotion decision for a challenger"""
        return self.pending_decisions.get(challenger_name)
    
    def checkpoint(self):
        """Save state to disk"""
        try:
            state = {
                'models': {name: meta.to_dict() for name, meta in self.models.items()},
                'current_champion': self.current_champion,
                'trades_since_promotion': self.trades_since_promotion,
                'promotion_baseline_wr': self.promotion_baseline_wr,
                'promotion_history': [event.to_dict() for event in self.promotion_history],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"Checkpoint saved: {self.checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load state from disk"""
        try:
            if not Path(self.checkpoint_path).exists():
                logger.info("No checkpoint found, starting fresh")
                return
            
            with open(self.checkpoint_path, 'r') as f:
                state = json.load(f)
            
            # Restore models
            for name, meta_dict in state['models'].items():
                meta_dict['role'] = ModelRole(meta_dict['role'])
                meta_dict['deployed_at'] = datetime.fromisoformat(meta_dict['deployed_at'])
                self.models[name] = ModelMetadata(**meta_dict)
            
            # Restore champion tracking
            self.current_champion = state['current_champion']
            self.trades_since_promotion = state['trades_since_promotion']
            self.promotion_baseline_wr = state['promotion_baseline_wr']
            
            # Restore promotion history
            for event_dict in state['promotion_history']:
                event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])
                self.promotion_history.append(PromotionEvent(**event_dict))
            
            logger.info(f"Checkpoint loaded: {len(self.models)} models, champion={self.current_champion}")
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Initialize manager
    manager = ShadowModelManager(
        min_trades_for_promotion=500,
        mdd_tolerance=1.20,
        alpha=0.05
    )
    
    # Register champion
    manager.register_model(
        model_name='xgboost_v1',
        model_type='xgboost',
        version='1.0',
        role=ModelRole.CHAMPION,
        description='Current production model'
    )
    
    # Register challenger
    manager.register_model(
        model_name='lightgbm_v1',
        model_type='lightgbm',
        version='1.0',
        role=ModelRole.CHALLENGER,
        description='New model with better features'
    )
    
    # Simulate 500 trades
    np.random.seed(42)
    
    for i in range(500):
        # Champion predictions (56% WR, mean $50)
        outcome_champ = 1 if np.random.rand() < 0.56 else 0
        pnl_champ = np.random.normal(50, 120)
        
        manager.record_prediction(
            model_name='xgboost_v1',
            prediction=1,
            actual_outcome=outcome_champ,
            pnl=pnl_champ,
            confidence=0.68,
            executed=True
        )
        
        # Challenger predictions (58% WR, mean $60)
        outcome_chal = 1 if np.random.rand() < 0.58 else 0
        pnl_chal = np.random.normal(60, 125)
        
        manager.record_prediction(
            model_name='lightgbm_v1',
            prediction=1,
            actual_outcome=outcome_chal,
            pnl=pnl_chal,
            confidence=0.70,
            executed=False  # Shadow
        )
    
    # Check promotion criteria
    decision = manager.check_promotion_criteria('lightgbm_v1')
    
    print(f"\n{'='*60}")
    print(f"PROMOTION DECISION")
    print(f"{'='*60}")
    print(f"Challenger: lightgbm_v1")
    print(f"Status: {decision.status.value}")
    print(f"Score: {decision.promotion_score:.1f}/100")
    print(f"Reason: {decision.reason}")
    print(f"\nChampion Metrics:")
    print(f"  WR: {decision.champion_metrics['win_rate']:.2%}")
    print(f"  Sharpe: {decision.champion_metrics['sharpe_ratio']:.2f}")
    print(f"  Mean PnL: ${decision.champion_metrics['mean_pnl']:.2f}")
    print(f"\nChallenger Metrics:")
    print(f"  WR: {decision.challenger_metrics['win_rate']:.2%}")
    print(f"  Sharpe: {decision.challenger_metrics['sharpe_ratio']:.2f}")
    print(f"  Mean PnL: ${decision.challenger_metrics['mean_pnl']:.2f}")
    print(f"{'='*60}\n")
    
    # Promote if approved
    if decision.status == PromotionStatus.APPROVED:
        manager.promote_challenger('lightgbm_v1')
        print("✅ Challenger promoted to champion!")
    else:
        print("❌ Challenger not ready for promotion")
