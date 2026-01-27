#!/usr/bin/env python3
"""
P3.9 Harvest Optimizer - READ-ONLY Analytics & Recommendation Engine

Consumes metrics from MetricPack Builder + Exit Intelligence to generate
advanced analytics and regime-aware harvest recommendations.

AUDIT-SAFE GUARANTEES:
- No writes to trading streams (apply.plan, trade.intent, ai.decision)
- No modification of trading logic
- Only reads from Prometheus metrics endpoints
- Only writes to quantum:obs:* namespace and analytics tables
- Generates recommendations (NOT automatic changes)
"""

import os
import sys
import time
import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum

import requests
from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, REGISTRY
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger("harvest_optimizer")

METRICPACK_URL = os.getenv("METRICPACK_URL", "http://localhost:8051/metrics")
EXIT_INTEL_URL = os.getenv("EXIT_INTEL_URL", "http://localhost:9109/metrics")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9091")

SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8052"))
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL_SECONDS", "60"))

ROLLING_WINDOW_SIZE = int(os.getenv("ROLLING_WINDOW_SIZE", "200"))
SMOOTHING_ALPHA = float(os.getenv("SMOOTHING_ALPHA", "0.3"))

SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,TRXUSDT").split(",")

# ============================================================================
# Data Models
# ============================================================================

class Regime(str, Enum):
    TREND = "trend"
    CHOP = "chop"
    UNKNOWN = "unknown"

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics per (symbol, regime, exit_type)"""
    symbol: str
    regime: Regime
    exit_type: str
    
    # Core metrics
    total_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Advanced metrics
    expectancy: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    exit_efficiency: float = 0.0
    
    # Time efficiency
    avg_time_in_trade: float = 0.0
    expectancy_per_hour: float = 0.0
    
    # Risk metrics
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    adverse_recovery_rate: float = 0.0
    
    # Confidence
    sample_size: int = 0
    last_updated: float = 0.0

@dataclass
class Recommendation:
    """Harvest optimization recommendation"""
    symbol: str
    regime: Regime
    rule_name: str
    
    action: str  # "widen", "tighten", "delay", "accelerate", "maintain"
    confidence: float  # 0.0 - 1.0
    reason: str
    suggested_value: Optional[str] = None
    
    current_metrics: Dict[str, float] = None
    timestamp: float = 0.0

@dataclass
class AnalyticsReport:
    """Complete analytics report"""
    generated_at: str
    analysis_window: str
    
    performance_by_regime: Dict[str, Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    
    regime_stability: Dict[str, float]
    top_performers: List[Dict[str, Any]]
    improvement_areas: List[Dict[str, Any]]
    
    summary: str

# ============================================================================
# Prometheus Metrics
# ============================================================================

# Build info
build_info = Info("quantum_harvest_optimizer_build", "Build information")
build_info.info({"version": "1.0.0", "service": "harvest_optimizer"})

# Core metrics
ho_recommendation_score = Gauge(
    "quantum_ho_recommendation_score",
    "Recommendation confidence score",
    ["symbol", "regime", "rule"]
)

ho_expectancy = Gauge(
    "quantum_ho_expectancy",
    "Expected value per trade (USDT)",
    ["symbol", "regime", "exit_type"]
)

ho_profit_factor = Gauge(
    "quantum_ho_profit_factor",
    "Gross profit / gross loss ratio",
    ["symbol", "regime"]
)

ho_payoff_ratio = Gauge(
    "quantum_ho_payoff_ratio",
    "Average win / average loss ratio",
    ["symbol", "regime"]
)

ho_exit_efficiency = Gauge(
    "quantum_ho_exit_efficiency",
    "Exit price / MFE price ratio",
    ["symbol", "regime"]
)

ho_time_efficiency = Gauge(
    "quantum_ho_time_efficiency",
    "Expectancy per hour (USDT/hour)",
    ["symbol", "regime"]
)

ho_regime_stability = Gauge(
    "quantum_ho_regime_stability",
    "Regime persistence score (0-1)",
    ["symbol", "regime"]
)

ho_adverse_recovery = Gauge(
    "quantum_ho_adverse_recovery_rate",
    "Recovery from MAE to exit (0-1)",
    ["symbol", "regime"]
)

ho_win_rate = Gauge(
    "quantum_ho_win_rate",
    "Win rate by regime and exit type",
    ["symbol", "regime", "exit_type"]
)

ho_avg_time = Gauge(
    "quantum_ho_avg_time_seconds",
    "Average time in trade (seconds)",
    ["symbol", "regime"]
)

ho_reports_generated = Counter(
    "quantum_ho_reports_generated_total",
    "Total reports generated"
)

ho_metrics_updated = Counter(
    "quantum_ho_metrics_updated_total",
    "Total metric updates"
)

ho_last_update = Gauge(
    "quantum_ho_last_update_timestamp",
    "Last successful update timestamp"
)

# ============================================================================
# Metrics Parser
# ============================================================================

class PrometheusMetricsParser:
    """Parse Prometheus text format metrics"""
    
    @staticmethod
    def parse_metrics(text: str) -> Dict[str, List[Tuple[Dict[str, str], float]]]:
        """Parse Prometheus metrics text into structured data"""
        metrics = defaultdict(list)
        
        current_metric = None
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Parse metric line
            if "{" in line:
                metric_name, rest = line.split("{", 1)
                labels_str, value_str = rest.rsplit("}", 1)
                value = float(value_str.strip())
                
                # Parse labels
                labels = {}
                for label_pair in labels_str.split(","):
                    if "=" in label_pair:
                        key, val = label_pair.split("=", 1)
                        labels[key.strip()] = val.strip('"')
                
                metrics[metric_name].append((labels, value))
            else:
                # Simple metric without labels
                parts = line.split()
                if len(parts) == 2:
                    metric_name, value_str = parts
                    try:
                        value = float(value_str)
                        metrics[metric_name].append(({}, value))
                    except ValueError:
                        pass
        
        return dict(metrics)

# ============================================================================
# Analytics Engine
# ============================================================================

class HarvestAnalyzer:
    """Analyze harvest/exit performance and generate recommendations"""
    
    def __init__(self):
        self.performance_cache: Dict[Tuple[str, Regime], PerformanceMetrics] = {}
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW_SIZE))
        self.regime_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.parser = PrometheusMetricsParser()
    
    async def update_from_sources(self):
        """Fetch and update from MetricPack + Exit Intelligence"""
        try:
            # Fetch MetricPack metrics
            metricpack_data = await self._fetch_metrics(METRICPACK_URL)
            if metricpack_data:
                self._process_metricpack(metricpack_data)
            
            # Fetch Exit Intelligence metrics (optional)
            exit_intel_data = await self._fetch_metrics(EXIT_INTEL_URL)
            if exit_intel_data:
                self._process_exit_intel(exit_intel_data)
            
            # Calculate advanced metrics
            self._calculate_advanced_metrics()
            
            # Update Prometheus
            self._export_metrics()
            
            ho_metrics_updated.inc()
            ho_last_update.set(time.time())
            
            logger.info("Metrics updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}", exc_info=True)
    
    async def _fetch_metrics(self, url: str) -> Optional[Dict]:
        """Fetch metrics from endpoint"""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return self.parser.parse_metrics(response.text)
            else:
                logger.warning(f"Failed to fetch {url}: {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Error fetching {url}: {e}")
            return None
    
    def _process_metricpack(self, metrics: Dict):
        """Process MetricPack Builder metrics"""
        # Extract trade counts by regime
        if "quantum_harvest_trades_closed_total" in metrics:
            for labels, value in metrics["quantum_harvest_trades_closed_total"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                
                key = (symbol, regime)
                if key not in self.performance_cache:
                    self.performance_cache[key] = PerformanceMetrics(
                        symbol=symbol,
                        regime=regime,
                        exit_type="all"
                    )
                
                self.performance_cache[key].total_trades = int(value)
        
        # Extract win rate
        if "quantum_harvest_winrate" in metrics:
            for labels, value in metrics["quantum_harvest_winrate"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                key = (symbol, regime)
                
                if key in self.performance_cache:
                    self.performance_cache[key].win_rate = value
        
        # Extract expectancy
        if "quantum_harvest_expectancy" in metrics:
            for labels, value in metrics["quantum_harvest_expectancy"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                key = (symbol, regime)
                
                if key in self.performance_cache:
                    self.performance_cache[key].expectancy = value
    
    def _process_exit_intel(self, metrics: Dict):
        """Process Exit Intelligence metrics"""
        # Extract MFE/MAE data
        if "quantum_exit_mfe_usdt" in metrics:
            for labels, value in metrics["quantum_exit_mfe_usdt"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                key = (symbol, regime)
                
                if key in self.performance_cache:
                    self.performance_cache[key].avg_mfe = value
        
        if "quantum_exit_mae_usdt" in metrics:
            for labels, value in metrics["quantum_exit_mae_usdt"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                key = (symbol, regime)
                
                if key in self.performance_cache:
                    self.performance_cache[key].avg_mae = value
        
        # Extract exit efficiency
        if "quantum_exit_efficiency" in metrics:
            for labels, value in metrics["quantum_exit_efficiency"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                key = (symbol, regime)
                
                if key in self.performance_cache:
                    self.performance_cache[key].exit_efficiency = value
        
        # Extract time in trade
        if "quantum_exit_time_seconds" in metrics:
            for labels, value in metrics["quantum_exit_time_seconds"]:
                symbol = labels.get("symbol", "UNKNOWN")
                regime = Regime(labels.get("regime", "unknown"))
                key = (symbol, regime)
                
                if key in self.performance_cache:
                    self.performance_cache[key].avg_time_in_trade = value
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced analytics from cached data"""
        for key, perf in self.performance_cache.items():
            symbol, regime = key
            
            # Profit factor (from win_rate + expectancy)
            if perf.win_rate > 0 and perf.expectancy != 0:
                # Proxy: if expectancy > 0, profit factor = 1 + expectancy/avg_loss
                # For simplicity, use win_rate as proxy
                loss_rate = 1.0 - perf.win_rate
                if loss_rate > 0:
                    perf.profit_factor = perf.win_rate / loss_rate
                else:
                    perf.profit_factor = 999.0  # All wins
            
            # Payoff ratio (from MFE/MAE if available)
            if perf.avg_mfe > 0 and perf.avg_mae > 0:
                perf.payoff_ratio = perf.avg_mfe / abs(perf.avg_mae)
            
            # Time efficiency (expectancy per hour)
            if perf.avg_time_in_trade > 0:
                hours = perf.avg_time_in_trade / 3600.0
                perf.expectancy_per_hour = perf.expectancy / hours if hours > 0 else 0.0
            
            # Adverse recovery rate (how much recovered from MAE to exit)
            if perf.avg_mae < 0 and perf.avg_mfe > 0:
                # If final PnL > MAE, we recovered
                # Proxy: exit_efficiency indicates how much of MFE we captured
                # Recovery = (exit - MAE) / (MFE - MAE)
                # Simplified: use exit_efficiency as proxy
                perf.adverse_recovery_rate = perf.exit_efficiency
            
            perf.last_updated = time.time()
    
    def _export_metrics(self):
        """Export to Prometheus gauges"""
        for key, perf in self.performance_cache.items():
            symbol, regime = key
            regime_str = regime.value
            
            # Export all metrics
            ho_expectancy.labels(
                symbol=symbol,
                regime=regime_str,
                exit_type="all"
            ).set(perf.expectancy)
            
            ho_profit_factor.labels(
                symbol=symbol,
                regime=regime_str
            ).set(perf.profit_factor)
            
            ho_payoff_ratio.labels(
                symbol=symbol,
                regime=regime_str
            ).set(perf.payoff_ratio)
            
            ho_exit_efficiency.labels(
                symbol=symbol,
                regime=regime_str
            ).set(perf.exit_efficiency)
            
            ho_time_efficiency.labels(
                symbol=symbol,
                regime=regime_str
            ).set(perf.expectancy_per_hour)
            
            ho_adverse_recovery.labels(
                symbol=symbol,
                regime=regime_str
            ).set(perf.adverse_recovery_rate)
            
            ho_win_rate.labels(
                symbol=symbol,
                regime=regime_str,
                exit_type="all"
            ).set(perf.win_rate)
            
            ho_avg_time.labels(
                symbol=symbol,
                regime=regime_str
            ).set(perf.avg_time_in_trade)
    
    def generate_recommendations(self) -> List[Recommendation]:
        """Generate harvest optimization recommendations"""
        recommendations = []
        
        for key, perf in self.performance_cache.items():
            symbol, regime = key
            
            if perf.total_trades < 10:
                # Insufficient data
                continue
            
            # Confidence based on sample size
            confidence = min(1.0, perf.total_trades / 50.0)
            
            # Regime-specific recommendations
            if regime == Regime.TREND:
                recommendations.extend(self._trend_recommendations(symbol, perf, confidence))
            elif regime == Regime.CHOP:
                recommendations.extend(self._chop_recommendations(symbol, perf, confidence))
        
        # Update Prometheus recommendation scores
        for rec in recommendations:
            ho_recommendation_score.labels(
                symbol=rec.symbol,
                regime=rec.regime.value,
                rule=rec.rule_name
            ).set(rec.confidence)
        
        return recommendations
    
    def _trend_recommendations(self, symbol: str, perf: PerformanceMetrics, confidence: float) -> List[Recommendation]:
        """Generate recommendations for trending markets"""
        recommendations = []
        
        # High exit efficiency in trend = let winners run
        if perf.exit_efficiency > 0.75 and perf.expectancy > 0:
            recommendations.append(Recommendation(
                symbol=symbol,
                regime=Regime.TREND,
                rule_name="partial_ladder",
                action="widen",
                confidence=confidence,
                reason=f"High exit efficiency ({perf.exit_efficiency:.2f}) suggests trends persist. Widen partial ladder.",
                current_metrics={"exit_efficiency": perf.exit_efficiency, "expectancy": perf.expectancy}
            ))
        
        # Low time efficiency = exits too slow
        if perf.expectancy_per_hour > 0 and perf.avg_time_in_trade > 7200:
            recommendations.append(Recommendation(
                symbol=symbol,
                regime=Regime.TREND,
                rule_name="time_stop",
                action="tighten",
                confidence=confidence * 0.8,
                reason=f"Avg time {perf.avg_time_in_trade/3600:.1f}h reduces capital efficiency. Consider earlier time-stops.",
                current_metrics={"time_hours": perf.avg_time_in_trade/3600, "expectancy_per_hour": perf.expectancy_per_hour}
            ))
        
        # Poor adverse recovery = stop too loose
        if perf.adverse_recovery_rate < 0.5 and perf.avg_mae < -10:
            recommendations.append(Recommendation(
                symbol=symbol,
                regime=Regime.TREND,
                rule_name="stop_loss",
                action="tighten",
                confidence=confidence,
                reason=f"Low recovery from MAE ({perf.adverse_recovery_rate:.2f}). Consider tighter stops.",
                current_metrics={"recovery_rate": perf.adverse_recovery_rate, "avg_mae": perf.avg_mae}
            ))
        
        return recommendations
    
    def _chop_recommendations(self, symbol: str, perf: PerformanceMetrics, confidence: float) -> List[Recommendation]:
        """Generate recommendations for choppy markets"""
        recommendations = []
        
        # Low exit efficiency in chop = take profits earlier
        if perf.exit_efficiency < 0.5 and perf.avg_mfe > 5:
            recommendations.append(Recommendation(
                symbol=symbol,
                regime=Regime.CHOP,
                rule_name="partial_25",
                action="accelerate",
                confidence=confidence,
                reason=f"Low exit efficiency ({perf.exit_efficiency:.2f}) in chop. Take partial_25 earlier.",
                current_metrics={"exit_efficiency": perf.exit_efficiency, "avg_mfe": perf.avg_mfe}
            ))
        
        # High payoff ratio but low win rate = targets too far
        if perf.payoff_ratio > 2.0 and perf.win_rate < 0.5:
            recommendations.append(Recommendation(
                symbol=symbol,
                regime=Regime.CHOP,
                rule_name="take_profit",
                action="tighten",
                confidence=confidence * 0.9,
                reason=f"Payoff ratio {perf.payoff_ratio:.2f} but win rate {perf.win_rate:.2%}. Tighten TPs in chop.",
                current_metrics={"payoff_ratio": perf.payoff_ratio, "win_rate": perf.win_rate}
            ))
        
        # Long trades in chop = bad
        if perf.avg_time_in_trade > 3600 and perf.expectancy < 5:
            recommendations.append(Recommendation(
                symbol=symbol,
                regime=Regime.CHOP,
                rule_name="time_stop",
                action="tighten",
                confidence=confidence,
                reason=f"Avg time {perf.avg_time_in_trade/60:.0f}min too long for chop. Earlier time-stops recommended.",
                current_metrics={"time_minutes": perf.avg_time_in_trade/60, "expectancy": perf.expectancy}
            ))
        
        return recommendations
    
    def generate_report(self) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        now = datetime.utcnow()
        
        # Performance by regime
        performance_by_regime = {}
        for key, perf in self.performance_cache.items():
            symbol, regime = key
            regime_key = f"{symbol}_{regime.value}"
            performance_by_regime[regime_key] = asdict(perf)
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Top performers
        top_performers = sorted(
            [perf for perf in self.performance_cache.values() if perf.total_trades >= 10],
            key=lambda p: p.expectancy,
            reverse=True
        )[:5]
        
        # Improvement areas (negative expectancy)
        improvement_areas = sorted(
            [perf for perf in self.performance_cache.values() if perf.expectancy < 0],
            key=lambda p: p.expectancy
        )[:5]
        
        # Regime stability (mock for now, needs time-series analysis)
        regime_stability = {}
        for symbol in SYMBOLS:
            regime_stability[symbol] = 0.65  # Placeholder
        
        # Summary
        total_expectancy = sum(p.expectancy for p in self.performance_cache.values())
        avg_exit_efficiency = sum(p.exit_efficiency for p in self.performance_cache.values()) / max(len(self.performance_cache), 1)
        
        top_rec = ""
        if recommendations:
            rec = recommendations[0]
            top_rec = f"{rec.action.upper()} {rec.rule_name} for {rec.symbol} in {rec.regime.value} (confidence: {rec.confidence:.0%})"
        else:
            top_rec = "No recommendations yet"
        
        summary = f"""
Harvest Optimizer Report - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC

Total Tracked Configurations: {len(self.performance_cache)}
Recommendations Generated: {len(recommendations)}
Average Exit Efficiency: {avg_exit_efficiency:.2%}
Combined Expectancy: {total_expectancy:.2f} USDT

Top Recommendation: {top_rec}
"""
        
        return AnalyticsReport(
            generated_at=now.isoformat(),
            analysis_window=f"Last {ROLLING_WINDOW_SIZE} trades",
            performance_by_regime=performance_by_regime,
            recommendations=[asdict(r) for r in recommendations],
            regime_stability=regime_stability,
            top_performers=[asdict(p) for p in top_performers],
            improvement_areas=[asdict(p) for p in improvement_areas],
            summary=summary.strip()
        )

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Harvest Optimizer", version="1.0.0")
analyzer = HarvestAnalyzer()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "harvest_optimizer",
        "tracked_configs": len(analyzer.performance_cache),
        "last_update": ho_last_update._value._value if hasattr(ho_last_update._value, '_value') else 0
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(REGISTRY), media_type="text/plain")

@app.get("/report")
async def report():
    """Generate analytics report"""
    try:
        report_data = analyzer.generate_report()
        ho_reports_generated.inc()
        return JSONResponse(content=asdict(report_data))
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/recommendations")
async def recommendations():
    """Get current recommendations"""
    try:
        recs = analyzer.generate_recommendations()
        return JSONResponse(content={
            "recommendations": [asdict(r) for r in recs],
            "count": len(recs),
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# ============================================================================
# Background Update Loop
# ============================================================================

async def update_loop():
    """Background task to update metrics periodically"""
    logger.info("Starting update loop")
    
    while True:
        try:
            await analyzer.update_from_sources()
            await asyncio.sleep(UPDATE_INTERVAL)
        except Exception as e:
            logger.error(f"Error in update loop: {e}", exc_info=True)
            await asyncio.sleep(10)

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    logger.info("Harvest Optimizer starting...")
    logger.info(f"MetricPack URL: {METRICPACK_URL}")
    logger.info(f"Exit Intel URL: {EXIT_INTEL_URL}")
    logger.info(f"Update interval: {UPDATE_INTERVAL}s")
    
    # Initial update
    await analyzer.update_from_sources()
    
    # Start background loop
    asyncio.create_task(update_loop())
    
    logger.info(f"Harvest Optimizer ready on port {SERVICE_PORT}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_config=None
    )
