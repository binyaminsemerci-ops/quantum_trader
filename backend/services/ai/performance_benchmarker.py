"""
Phase 3C-2: Performance Benchmarker

Real-time performance tracking and benchmarking for all AI modules.
Tracks accuracy, latency, throughput, and generates comparative performance reports.

Features:
- Module performance benchmarking (Phase 2B, 2D, 3A, 3B, Ensemble)
- Signal generation latency tracking with percentiles
- Prediction accuracy tracking
- Comparative performance analysis
- A/B testing framework
- Performance regression detection
- Automated performance reports

Author: AI Agent
Date: December 24, 2025
"""

import asyncio
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics
import structlog

logger = structlog.get_logger()


class PerformanceMetric(str, Enum):
    """Performance metric types."""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"


class BenchmarkStatus(str, Enum):
    """Benchmark status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class LatencyStats:
    """Latency statistics."""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    sample_count: int
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AccuracyStats:
    """Accuracy statistics."""
    correct_predictions: int
    total_predictions: int
    accuracy_pct: float
    precision: float
    recall: float
    f1_score: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThroughputStats:
    """Throughput statistics."""
    operations_per_second: float
    operations_per_minute: float
    operations_per_hour: float
    total_operations: int
    measurement_duration_sec: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModulePerformance:
    """Performance metrics for a single module."""
    module_name: str
    module_type: str  # "phase_2b", "phase_2d", etc.
    timestamp: datetime
    latency_stats: Optional[LatencyStats]
    accuracy_stats: Optional[AccuracyStats]
    throughput_stats: Optional[ThroughputStats]
    error_rate: float
    memory_usage_mb: float
    status: BenchmarkStatus
    performance_score: float  # 0-100
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        data = {
            'module_name': self.module_name,
            'module_type': self.module_type,
            'timestamp': self.timestamp.isoformat(),
            'error_rate': self.error_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'status': self.status.value,
            'performance_score': self.performance_score,
            'issues': self.issues
        }
        if self.latency_stats:
            data['latency_stats'] = self.latency_stats.to_dict()
        if self.accuracy_stats:
            data['accuracy_stats'] = self.accuracy_stats.to_dict()
        if self.throughput_stats:
            data['throughput_stats'] = self.throughput_stats.to_dict()
        return data


@dataclass
class PerformanceComparison:
    """Comparison of module performances."""
    timestamp: datetime
    fastest_module: str
    slowest_module: str
    most_accurate_module: str
    highest_throughput_module: str
    performance_rankings: Dict[str, int]
    latency_comparison: Dict[str, float]
    accuracy_comparison: Dict[str, float]
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'fastest_module': self.fastest_module,
            'slowest_module': self.slowest_module,
            'most_accurate_module': self.most_accurate_module,
            'highest_throughput_module': self.highest_throughput_module,
            'performance_rankings': self.performance_rankings,
            'latency_comparison': self.latency_comparison,
            'accuracy_comparison': self.accuracy_comparison
        }


@dataclass
class PerformanceReport:
    """Comprehensive performance report."""
    report_id: str
    generated_at: datetime
    time_window_hours: int
    overall_status: BenchmarkStatus
    overall_performance_score: float
    module_performances: Dict[str, ModulePerformance]
    comparison: PerformanceComparison
    regressions_detected: List[str]
    improvements_detected: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> dict:
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'time_window_hours': self.time_window_hours,
            'overall_status': self.overall_status.value,
            'overall_performance_score': self.overall_performance_score,
            'module_performances': {k: v.to_dict() for k, v in self.module_performances.items()},
            'comparison': self.comparison.to_dict(),
            'regressions_detected': self.regressions_detected,
            'improvements_detected': self.improvements_detected,
            'recommendations': self.recommendations
        }


class PerformanceBenchmarker:
    """
    Performance benchmarker for AI modules.
    
    Tracks:
    - Signal generation latency (with percentiles)
    - Prediction accuracy
    - Throughput (operations/sec)
    - Error rates
    - Memory usage
    
    Features:
    - Real-time performance tracking
    - Historical performance analysis
    - Module comparison
    - Regression detection
    - A/B testing support
    """
    
    def __init__(
        self,
        benchmark_interval_sec: int = 300,  # 5 minutes
        history_retention_hours: int = 168,  # 7 days
        latency_sample_size: int = 1000,
        regression_threshold_pct: float = 20.0  # 20% performance drop = regression
    ):
        """
        Initialize performance benchmarker.
        
        Args:
            benchmark_interval_sec: How often to calculate benchmarks
            history_retention_hours: How long to keep historical data
            latency_sample_size: Number of latency samples to keep per module
            regression_threshold_pct: Performance drop % to flag as regression
        """
        self.benchmark_interval = benchmark_interval_sec
        self.history_retention = timedelta(hours=history_retention_hours)
        self.latency_sample_size = latency_sample_size
        self.regression_threshold = regression_threshold_pct / 100.0
        
        # Module references
        self.orderbook_module = None
        self.volatility_engine = None
        self.risk_mode_predictor = None
        self.strategy_selector = None
        self.ensemble_manager = None
        
        # Performance tracking
        self.module_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=latency_sample_size))
        self.module_predictions: Dict[str, Dict[str, int]] = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.module_operations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.module_errors: Dict[str, int] = defaultdict(int)
        
        # Benchmarking
        self.current_benchmarks: Dict[str, ModulePerformance] = {}
        self.benchmark_history: deque = deque(maxlen=1000)
        self.baseline_performance: Dict[str, ModulePerformance] = {}
        
        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        # Control
        self.is_benchmarking = False
        self._benchmark_start_time = None
        
        logger.info("[PHASE 3C-2] 📊 Performance Benchmarker initialized")
    
    def set_modules(
        self,
        orderbook_module=None,
        volatility_engine=None,
        risk_mode_predictor=None,
        strategy_selector=None,
        ensemble_manager=None
    ):
        """Link AI modules for benchmarking."""
        self.orderbook_module = orderbook_module
        self.volatility_engine = volatility_engine
        self.risk_mode_predictor = risk_mode_predictor
        self.strategy_selector = strategy_selector
        self.ensemble_manager = ensemble_manager
        
        logger.info("[PHASE 3C-2] PB: Modules linked for benchmarking")
    
    async def start_benchmarking(self):
        """Start continuous performance benchmarking."""
        self.is_benchmarking = True
        self._benchmark_start_time = datetime.utcnow()
        logger.info(f"[PHASE 3C-2] 📊 Performance benchmarking started (interval: {self.benchmark_interval}s)")
        
        while self.is_benchmarking:
            try:
                await self.run_benchmark()
                await asyncio.sleep(self.benchmark_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[PHASE 3C-2] ❌ Benchmark error: {e}", exc_info=True)
                await asyncio.sleep(self.benchmark_interval)
    
    def stop_benchmarking(self):
        """Stop benchmarking."""
        self.is_benchmarking = False
        logger.info("[PHASE 3C-2] Performance benchmarking stopped")
    
    # ========================================================================
    # RECORDING METHODS
    # ========================================================================
    
    def record_latency(self, module_type: str, latency_ms: float):
        """Record module operation latency."""
        self.module_latencies[module_type].append(latency_ms)
        self.module_operations[module_type].append(datetime.utcnow())
    
    def record_prediction(self, module_type: str, correct: bool):
        """Record prediction accuracy."""
        self.module_predictions[module_type]['total'] += 1
        if correct:
            self.module_predictions[module_type]['correct'] += 1
    
    def record_error(self, module_type: str):
        """Record module error."""
        self.module_errors[module_type] += 1
    
    # ========================================================================
    # BENCHMARKING
    # ========================================================================
    
    async def run_benchmark(self) -> Dict[str, ModulePerformance]:
        """Run comprehensive performance benchmark."""
        logger.info("[PHASE 3C-2] Running performance benchmark...")
        
        benchmarks = {}
        
        # Benchmark each module
        if self.orderbook_module:
            benchmarks['phase_2b'] = await self._benchmark_module('phase_2b', 'Orderbook Imbalance')
        
        if self.volatility_engine:
            benchmarks['phase_2d'] = await self._benchmark_module('phase_2d', 'Volatility Structure Engine')
        
        if self.risk_mode_predictor:
            benchmarks['phase_3a'] = await self._benchmark_module('phase_3a', 'Risk Mode Predictor')
        
        if self.strategy_selector:
            benchmarks['phase_3b'] = await self._benchmark_module('phase_3b', 'Strategy Selector')
        
        if self.ensemble_manager:
            benchmarks['ensemble'] = await self._benchmark_module('ensemble', 'Ensemble Manager')
        
        # Update current benchmarks
        self.current_benchmarks = benchmarks
        
        # Add to history
        self.benchmark_history.append({
            'timestamp': datetime.utcnow(),
            'benchmarks': benchmarks
        })
        
        # Clean old history
        self._cleanup_old_history()
        
        # Detect regressions
        await self._detect_regressions(benchmarks)
        
        logger.info(f"[PHASE 3C-2] ✅ Benchmark complete ({len(benchmarks)} modules)")
        
        return benchmarks
    
    async def _benchmark_module(self, module_type: str, module_name: str) -> ModulePerformance:
        """Benchmark a single module."""
        latencies = list(self.module_latencies[module_type])
        predictions = self.module_predictions[module_type]
        operations = list(self.module_operations[module_type])
        errors = self.module_errors[module_type]
        
        # Calculate latency stats
        latency_stats = None
        if latencies:
            latency_stats = LatencyStats(
                min_ms=min(latencies),
                max_ms=max(latencies),
                mean_ms=statistics.mean(latencies),
                median_ms=statistics.median(latencies),
                p50_ms=self._percentile(latencies, 50),
                p95_ms=self._percentile(latencies, 95),
                p99_ms=self._percentile(latencies, 99),
                std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                sample_count=len(latencies)
            )
        
        # Calculate accuracy stats
        accuracy_stats = None
        if predictions['total'] > 0:
            accuracy_pct = (predictions['correct'] / predictions['total']) * 100
            accuracy_stats = AccuracyStats(
                correct_predictions=predictions['correct'],
                total_predictions=predictions['total'],
                accuracy_pct=accuracy_pct,
                precision=accuracy_pct / 100.0,  # Simplified
                recall=accuracy_pct / 100.0,
                f1_score=accuracy_pct / 100.0
            )
        
        # Calculate throughput stats
        throughput_stats = None
        if operations:
            duration_sec = (datetime.utcnow() - operations[0]).total_seconds()
            if duration_sec > 0:
                ops_per_sec = len(operations) / duration_sec
                throughput_stats = ThroughputStats(
                    operations_per_second=ops_per_sec,
                    operations_per_minute=ops_per_sec * 60,
                    operations_per_hour=ops_per_sec * 3600,
                    total_operations=len(operations),
                    measurement_duration_sec=duration_sec
                )
        
        # Calculate error rate
        total_ops = predictions['total'] + errors
        error_rate = (errors / total_ops * 100) if total_ops > 0 else 0.0
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            latency_stats, accuracy_stats, error_rate
        )
        
        # Determine status
        status = self._determine_benchmark_status(performance_score)
        
        # Identify issues
        issues = []
        if latency_stats and latency_stats.p95_ms > 200:
            issues.append(f"High P95 latency: {latency_stats.p95_ms:.1f}ms")
        if accuracy_stats and accuracy_stats.accuracy_pct < 70:
            issues.append(f"Low accuracy: {accuracy_stats.accuracy_pct:.1f}%")
        if error_rate > 5:
            issues.append(f"High error rate: {error_rate:.1f}%")
        
        return ModulePerformance(
            module_name=module_name,
            module_type=module_type,
            timestamp=datetime.utcnow(),
            latency_stats=latency_stats,
            accuracy_stats=accuracy_stats,
            throughput_stats=throughput_stats,
            error_rate=error_rate,
            memory_usage_mb=0.0,  # TODO: Implement memory tracking
            status=status,
            performance_score=performance_score,
            issues=issues
        )
    
    def _calculate_performance_score(
        self,
        latency_stats: Optional[LatencyStats],
        accuracy_stats: Optional[AccuracyStats],
        error_rate: float
    ) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # Latency penalty (max 30 points)
        if latency_stats:
            if latency_stats.p95_ms > 500:
                score -= 30
            elif latency_stats.p95_ms > 200:
                score -= 20
            elif latency_stats.p95_ms > 100:
                score -= 10
        
        # Accuracy penalty (max 40 points)
        if accuracy_stats:
            if accuracy_stats.accuracy_pct < 50:
                score -= 40
            elif accuracy_stats.accuracy_pct < 70:
                score -= 30
            elif accuracy_stats.accuracy_pct < 80:
                score -= 20
            elif accuracy_stats.accuracy_pct < 90:
                score -= 10
        
        # Error rate penalty (max 30 points)
        if error_rate > 20:
            score -= 30
        elif error_rate > 10:
            score -= 20
        elif error_rate > 5:
            score -= 10
        
        return max(0.0, min(100.0, score))
    
    def _determine_benchmark_status(self, performance_score: float) -> BenchmarkStatus:
        """Determine benchmark status from score."""
        if performance_score >= 90:
            return BenchmarkStatus.EXCELLENT
        elif performance_score >= 75:
            return BenchmarkStatus.GOOD
        elif performance_score >= 60:
            return BenchmarkStatus.ACCEPTABLE
        elif performance_score >= 40:
            return BenchmarkStatus.POOR
        else:
            return BenchmarkStatus.CRITICAL
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    # ========================================================================
    # COMPARISON & ANALYSIS
    # ========================================================================
    
    def compare_modules(self) -> PerformanceComparison:
        """Compare performance across modules."""
        if not self.current_benchmarks:
            raise ValueError("No benchmark data available")
        
        # Extract latencies and accuracies
        latencies = {}
        accuracies = {}
        throughputs = {}
        
        for module_type, perf in self.current_benchmarks.items():
            if perf.latency_stats:
                latencies[module_type] = perf.latency_stats.p95_ms
            if perf.accuracy_stats:
                accuracies[module_type] = perf.accuracy_stats.accuracy_pct
            if perf.throughput_stats:
                throughputs[module_type] = perf.throughput_stats.operations_per_second
        
        # Find best performers
        fastest_module = min(latencies, key=latencies.get) if latencies else "none"
        slowest_module = max(latencies, key=latencies.get) if latencies else "none"
        most_accurate = max(accuracies, key=accuracies.get) if accuracies else "none"
        highest_throughput = max(throughputs, key=throughputs.get) if throughputs else "none"
        
        # Performance rankings
        rankings = {}
        for module_type, perf in self.current_benchmarks.items():
            rankings[module_type] = int(perf.performance_score)
        
        return PerformanceComparison(
            timestamp=datetime.utcnow(),
            fastest_module=fastest_module,
            slowest_module=slowest_module,
            most_accurate_module=most_accurate,
            highest_throughput_module=highest_throughput,
            performance_rankings=rankings,
            latency_comparison=latencies,
            accuracy_comparison=accuracies
        )
    
    async def generate_performance_report(self, time_window_hours: int = 24) -> PerformanceReport:
        """Generate comprehensive performance report."""
        report_id = f"perf_report_{int(time.time())}"
        
        # Get current benchmarks
        if not self.current_benchmarks:
            await self.run_benchmark()
        
        # Calculate overall metrics
        overall_score = statistics.mean([
            perf.performance_score 
            for perf in self.current_benchmarks.values()
        ]) if self.current_benchmarks else 0.0
        
        overall_status = self._determine_benchmark_status(overall_score)
        
        # Compare modules
        comparison = self.compare_modules()
        
        # Detect regressions and improvements
        regressions, improvements = await self._analyze_trends(time_window_hours)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            self.current_benchmarks, regressions
        )
        
        return PerformanceReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            time_window_hours=time_window_hours,
            overall_status=overall_status,
            overall_performance_score=overall_score,
            module_performances=self.current_benchmarks,
            comparison=comparison,
            regressions_detected=regressions,
            improvements_detected=improvements,
            recommendations=recommendations
        )
    
    async def _detect_regressions(self, current_benchmarks: Dict[str, ModulePerformance]):
        """Detect performance regressions."""
        if not self.baseline_performance:
            # Set baseline on first run
            self.baseline_performance = current_benchmarks.copy()
            return
        
        for module_type, current_perf in current_benchmarks.items():
            if module_type not in self.baseline_performance:
                continue
            
            baseline_perf = self.baseline_performance[module_type]
            
            # Check for latency regression
            if current_perf.latency_stats and baseline_perf.latency_stats:
                current_latency = current_perf.latency_stats.p95_ms
                baseline_latency = baseline_perf.latency_stats.p95_ms
                
                if current_latency > baseline_latency * (1 + self.regression_threshold):
                    logger.warning(
                        f"[PHASE 3C-2] ⚠️ Latency regression detected in {module_type}: "
                        f"{baseline_latency:.1f}ms → {current_latency:.1f}ms "
                        f"({((current_latency/baseline_latency - 1) * 100):.1f}% increase)"
                    )
            
            # Check for accuracy regression
            if current_perf.accuracy_stats and baseline_perf.accuracy_stats:
                current_acc = current_perf.accuracy_stats.accuracy_pct
                baseline_acc = baseline_perf.accuracy_stats.accuracy_pct
                
                if current_acc < baseline_acc * (1 - self.regression_threshold):
                    logger.warning(
                        f"[PHASE 3C-2] ⚠️ Accuracy regression detected in {module_type}: "
                        f"{baseline_acc:.1f}% → {current_acc:.1f}% "
                        f"({((baseline_acc - current_acc) / baseline_acc * 100):.1f}% decrease)"
                    )
    
    async def _analyze_trends(self, time_window_hours: int) -> Tuple[List[str], List[str]]:
        """Analyze performance trends to detect regressions and improvements."""
        regressions = []
        improvements = []
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Analyze historical benchmarks
        historical = [
            entry for entry in self.benchmark_history
            if entry['timestamp'] > cutoff_time
        ]
        
        if len(historical) < 2:
            return regressions, improvements
        
        # Compare first and last benchmarks in window
        first_benchmarks = historical[0]['benchmarks']
        last_benchmarks = historical[-1]['benchmarks']
        
        for module_type in first_benchmarks:
            if module_type not in last_benchmarks:
                continue
            
            first_perf = first_benchmarks[module_type]
            last_perf = last_benchmarks[module_type]
            
            score_change = last_perf.performance_score - first_perf.performance_score
            
            if score_change < -10:
                regressions.append(
                    f"{module_type}: Performance dropped {abs(score_change):.1f} points"
                )
            elif score_change > 10:
                improvements.append(
                    f"{module_type}: Performance improved {score_change:.1f} points"
                )
        
        return regressions, improvements
    
    def _generate_recommendations(
        self,
        benchmarks: Dict[str, ModulePerformance],
        regressions: List[str]
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        for module_type, perf in benchmarks.items():
            # Latency recommendations
            if perf.latency_stats and perf.latency_stats.p95_ms > 200:
                recommendations.append(
                    f"{module_type}: Optimize latency (P95: {perf.latency_stats.p95_ms:.1f}ms) - "
                    f"Consider caching, parallel processing, or algorithm optimization"
                )
            
            # Accuracy recommendations
            if perf.accuracy_stats and perf.accuracy_stats.accuracy_pct < 70:
                recommendations.append(
                    f"{module_type}: Improve accuracy ({perf.accuracy_stats.accuracy_pct:.1f}%) - "
                    f"Consider retraining models or feature engineering"
                )
            
            # Error rate recommendations
            if perf.error_rate > 5:
                recommendations.append(
                    f"{module_type}: Reduce error rate ({perf.error_rate:.1f}%) - "
                    f"Review error logs and add error handling"
                )
        
        # Regression recommendations
        if regressions:
            recommendations.append(
                f"Performance regressions detected in {len(regressions)} modules - "
                f"Review recent code changes and consider rollback"
            )
        
        return recommendations
    
    # ========================================================================
    # A/B TESTING
    # ========================================================================
    
    def start_ab_test(
        self,
        test_name: str,
        module_type: str,
        variant_a: str,
        variant_b: str,
        duration_hours: int = 24
    ):
        """Start A/B test for a module."""
        self.ab_tests[test_name] = {
            'module_type': module_type,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'start_time': datetime.utcnow(),
            'end_time': datetime.utcnow() + timedelta(hours=duration_hours),
            'results_a': {'latencies': [], 'predictions': {'correct': 0, 'total': 0}},
            'results_b': {'latencies': [], 'predictions': {'correct': 0, 'total': 0}}
        }
        
        logger.info(
            f"[PHASE 3C-2] A/B test started: {test_name} "
            f"({variant_a} vs {variant_b}, {duration_hours}h)"
        )
    
    def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get A/B test results."""
        if test_name not in self.ab_tests:
            raise ValueError(f"A/B test not found: {test_name}")
        
        test = self.ab_tests[test_name]
        
        # Calculate statistics for each variant
        results = {}
        for variant in ['a', 'b']:
            variant_data = test[f'results_{variant}']
            latencies = variant_data['latencies']
            predictions = variant_data['predictions']
            
            results[f'variant_{variant}'] = {
                'name': test[f'variant_{variant}'],
                'mean_latency_ms': statistics.mean(latencies) if latencies else 0,
                'p95_latency_ms': self._percentile(latencies, 95) if latencies else 0,
                'accuracy_pct': (predictions['correct'] / predictions['total'] * 100) 
                                if predictions['total'] > 0 else 0,
                'sample_count': len(latencies)
            }
        
        # Determine winner
        winner = None
        if results['variant_a']['mean_latency_ms'] < results['variant_b']['mean_latency_ms']:
            winner = 'variant_a'
        else:
            winner = 'variant_b'
        
        return {
            'test_name': test_name,
            'module_type': test['module_type'],
            'start_time': test['start_time'].isoformat(),
            'end_time': test['end_time'].isoformat(),
            'is_complete': datetime.utcnow() >= test['end_time'],
            'results': results,
            'winner': winner
        }
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def _cleanup_old_history(self):
        """Remove old benchmark history."""
        cutoff_time = datetime.utcnow() - self.history_retention
        
        # Clean benchmark history
        while self.benchmark_history and self.benchmark_history[0]['timestamp'] < cutoff_time:
            self.benchmark_history.popleft()
    
    def get_current_benchmarks(self) -> Dict[str, ModulePerformance]:
        """Get current module benchmarks."""
        return self.current_benchmarks
    
    def get_module_benchmark(self, module_type: str) -> Optional[ModulePerformance]:
        """Get benchmark for specific module."""
        return self.current_benchmarks.get(module_type)
    
    def reset_baseline(self):
        """Reset performance baseline."""
        self.baseline_performance = self.current_benchmarks.copy()
        logger.info("[PHASE 3C-2] Performance baseline reset")
