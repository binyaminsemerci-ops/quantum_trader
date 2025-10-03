#!/usr/bin/env python3
"""
Production Performance Monitoring System

This system provides real-time monitoring of the AI trading system including:
- Live performance tracking
- Risk monitoring and alerts
- Model performance degradation detection
- Trading signal analysis
- Portfolio health monitoring
- Real-time dashboards and logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading

import numpy as np

from config.config import settings
from production_risk_manager import RiskManager, PortfolioState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics tracking."""
    timestamp: datetime
    total_equity: float
    daily_pnl: float
    daily_pnl_pct: float
    trades_today: int
    win_rate_today: float
    active_positions: int
    portfolio_risk: float
    max_drawdown: float
    sharpe_ratio: float
    model_accuracy: float
    signal_count: int
    avg_signal_strength: float

@dataclass  
class AlertConfig:
    """Alert configuration for monitoring."""
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 15.0
    min_model_accuracy: float = 0.55
    max_portfolio_risk: float = 20.0
    min_signal_strength: float = 0.4
    alert_cooldown_minutes: int = 30

@dataclass
class Alert:
    """Alert message."""
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'CRITICAL'
    category: str  # 'PERFORMANCE', 'RISK', 'MODEL', 'SYSTEM'
    message: str
    data: Dict[str, Any]

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(
        self, 
        risk_manager: Optional[RiskManager] = None,
        alert_config: Optional[AlertConfig] = None,
        output_dir: Path = None
    ):
        self.risk_manager = risk_manager
        self.alert_config = alert_config or AlertConfig()
        self.output_dir = output_dir or Path("logs/monitoring")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_thread = None
        self.performance_history: List[PerformanceMetrics] = []
        self.alerts: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Performance tracking
        self.daily_trades = []
        self.model_predictions = []
        self.signal_history = []
        
        # Files for logging
        self.performance_log = self.output_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.alerts_log = self.output_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down monitoring...")
        self.stop_monitoring()
        sys.exit(0)
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start the performance monitoring loop."""
        if self.is_running:
            logger.warning("Monitoring is already running")
            return
        
        logger.info(f"Starting performance monitoring (interval: {interval_seconds}s)")
        self.is_running = True
        
        # Start monitoring in separate thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop the performance monitoring."""
        logger.info("Stopping performance monitoring...")
        self.is_running = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Save final state
        self._save_monitoring_state()
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                if metrics:
                    self.performance_history.append(metrics)
                    
                    # Log metrics
                    self._log_performance_metrics(metrics)
                    
                    # Check for alerts
                    self._check_alerts(metrics)
                    
                    # Cleanup old data (keep last 24 hours)
                    self._cleanup_old_data()
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect current performance metrics."""
        try:
            # Get current portfolio state
            if self.risk_manager:
                portfolio = self.risk_manager.get_portfolio_summary()
            else:
                # Fallback: simulate portfolio data
                portfolio = {
                    "total_equity": settings.starting_equity,
                    "daily_pnl": 0.0,
                    "positions_count": 0,
                    "daily_pnl_pct": 0.0,
                    "max_drawdown": 0.0
                }
            
            # Calculate today's metrics
            today_trades = self._get_today_trades()
            win_rate = self._calculate_win_rate(today_trades)
            
            # Model performance metrics
            model_accuracy = self._calculate_model_accuracy()
            avg_signal_strength = self._calculate_avg_signal_strength()
            
            # Risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(timezone.utc),
                total_equity=portfolio.get("total_equity", 0),
                daily_pnl=portfolio.get("daily_pnl", 0),
                daily_pnl_pct=portfolio.get("daily_pnl_pct", 0),
                trades_today=len(today_trades),
                win_rate_today=win_rate,
                active_positions=portfolio.get("positions_count", 0),
                portfolio_risk=self._calculate_portfolio_risk(),
                max_drawdown=portfolio.get("max_drawdown", 0),
                sharpe_ratio=sharpe_ratio,
                model_accuracy=model_accuracy,
                signal_count=len(self.signal_history),
                avg_signal_strength=avg_signal_strength
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            return None
    
    def _get_today_trades(self) -> List[Dict]:
        """Get trades executed today."""
        today = datetime.now().date()
        return [
            trade for trade in self.daily_trades
            if datetime.fromisoformat(trade.get("timestamp", "")).date() == today
        ]
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
        return winning_trades / len(trades)
    
    def _calculate_model_accuracy(self) -> float:
        """Calculate recent model prediction accuracy."""
        if len(self.model_predictions) < 10:
            return 0.0
        
        # Use last 100 predictions
        recent_predictions = self.model_predictions[-100:]
        correct_predictions = sum(
            1 for pred in recent_predictions 
            if pred.get("correct", False)
        )
        
        return correct_predictions / len(recent_predictions)
    
    def _calculate_avg_signal_strength(self) -> float:
        """Calculate average signal strength."""
        if not self.signal_history:
            return 0.0
        
        # Use signals from last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_signals = [
            sig for sig in self.signal_history
            if datetime.fromisoformat(sig["timestamp"]) > one_hour_ago
        ]
        
        if not recent_signals:
            return 0.0
        
        return np.mean([sig.get("strength", 0) for sig in recent_signals])
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk percentage."""
        if self.risk_manager:
            portfolio = self.risk_manager.get_portfolio_summary()
            positions = portfolio.get("positions", [])
            
            total_risk = sum(pos.get("risk_amount", 0) for pos in positions)
            total_equity = portfolio.get("total_equity", 1)
            
            return (total_risk / total_equity) * 100
        
        return 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from recent performance."""
        if len(self.performance_history) < 30:  # Need at least 30 data points
            return 0.0
        
        # Get daily returns from last 30 days
        recent_metrics = self.performance_history[-30:]
        daily_returns = []
        
        for i in range(1, len(recent_metrics)):
            prev_equity = recent_metrics[i-1].total_equity
            curr_equity = recent_metrics[i].total_equity
            
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
        
        if len(daily_returns) < 2:
            return 0.0
        
        returns_array = np.array(daily_returns)
        excess_returns = returns_array - (0.02 / 365)  # Risk-free rate
        
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0.0
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions."""
        alerts_to_send = []
        
        # Daily loss alert
        if metrics.daily_pnl_pct < -self.alert_config.max_daily_loss_pct:
            alerts_to_send.append(Alert(
                timestamp=metrics.timestamp,
                level="CRITICAL",
                category="PERFORMANCE",
                message=f"Daily loss exceeded limit: {metrics.daily_pnl_pct:.2f}%",
                data={"daily_pnl_pct": metrics.daily_pnl_pct, "limit": -self.alert_config.max_daily_loss_pct}
            ))
        
        # Drawdown alert
        if metrics.max_drawdown * 100 > self.alert_config.max_drawdown_pct:
            alerts_to_send.append(Alert(
                timestamp=metrics.timestamp,
                level="CRITICAL",
                category="RISK",
                message=f"Maximum drawdown exceeded: {metrics.max_drawdown*100:.2f}%",
                data={"max_drawdown_pct": metrics.max_drawdown*100, "limit": self.alert_config.max_drawdown_pct}
            ))
        
        # Model accuracy alert
        if metrics.model_accuracy > 0 and metrics.model_accuracy < self.alert_config.min_model_accuracy:
            alerts_to_send.append(Alert(
                timestamp=metrics.timestamp,
                level="WARNING",
                category="MODEL",
                message=f"Model accuracy below threshold: {metrics.model_accuracy:.1%}",
                data={"model_accuracy": metrics.model_accuracy, "threshold": self.alert_config.min_model_accuracy}
            ))
        
        # Portfolio risk alert
        if metrics.portfolio_risk > self.alert_config.max_portfolio_risk:
            alerts_to_send.append(Alert(
                timestamp=metrics.timestamp,
                level="WARNING",
                category="RISK",
                message=f"Portfolio risk too high: {metrics.portfolio_risk:.1f}%",
                data={"portfolio_risk": metrics.portfolio_risk, "limit": self.alert_config.max_portfolio_risk}
            ))
        
        # Signal strength alert
        if metrics.avg_signal_strength > 0 and metrics.avg_signal_strength < self.alert_config.min_signal_strength:
            alerts_to_send.append(Alert(
                timestamp=metrics.timestamp,
                level="INFO",
                category="MODEL",
                message=f"Low average signal strength: {metrics.avg_signal_strength:.2f}",
                data={"avg_signal_strength": metrics.avg_signal_strength, "threshold": self.alert_config.min_signal_strength}
            ))
        
        # Send alerts (with cooldown)
        for alert in alerts_to_send:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """Send an alert (with cooldown logic)."""
        alert_key = f"{alert.category}_{alert.level}"
        now = datetime.now()
        
        # Check cooldown
        if alert_key in self.last_alert_times:
            time_since_last = now - self.last_alert_times[alert_key]
            if time_since_last.total_seconds() < (self.alert_config.alert_cooldown_minutes * 60):
                return  # Skip due to cooldown
        
        # Log alert
        self.alerts.append(alert)
        self.last_alert_times[alert_key] = now
        
        # Write to alerts log
        with open(self.alerts_log, 'a') as f:
            f.write(json.dumps(asdict(alert), default=str) + '\n')
        
        # Log to console
        logger.warning(f"ALERT [{alert.level}] {alert.category}: {alert.message}")
    
    def _log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics to file."""
        with open(self.performance_log, 'a') as f:
            f.write(json.dumps(asdict(metrics), default=str) + '\n')
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues."""
        # Keep last 24 hours of performance data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history = [
            m for m in self.performance_history
            if m.timestamp > cutoff_time
        ]
        
        # Keep last 1000 trades
        if len(self.daily_trades) > 1000:
            self.daily_trades = self.daily_trades[-1000:]
        
        # Keep last 1000 model predictions
        if len(self.model_predictions) > 1000:
            self.model_predictions = self.model_predictions[-1000:]
        
        # Keep last 1000 signals
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def _save_monitoring_state(self):
        """Save current monitoring state."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "performance_history_count": len(self.performance_history),
            "alerts_count": len(self.alerts),
            "trades_count": len(self.daily_trades),
            "model_predictions_count": len(self.model_predictions),
            "signals_count": len(self.signal_history)
        }
        
        state_file = self.output_dir / f"monitoring_state_{int(time.time())}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def add_trade(self, trade_data: Dict):
        """Add a completed trade to tracking."""
        trade_data["timestamp"] = datetime.now().isoformat()
        self.daily_trades.append(trade_data)
    
    def add_model_prediction(self, prediction_data: Dict):
        """Add a model prediction for accuracy tracking."""
        prediction_data["timestamp"] = datetime.now().isoformat()
        self.model_predictions.append(prediction_data)
    
    def add_signal(self, signal_data: Dict):
        """Add a trading signal for analysis."""
        signal_data["timestamp"] = datetime.now().isoformat()
        self.signal_history.append(signal_data)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        if not self.performance_history:
            return {"status": "No data available"}
        
        latest = self.performance_history[-1]
        recent_alerts = [a for a in self.alerts if (datetime.now() - a.timestamp).total_seconds() < 3600]
        
        return {
            "monitoring_active": self.is_running,
            "latest_update": latest.timestamp.isoformat(),
            "current_equity": latest.total_equity,
            "daily_pnl": f"{latest.daily_pnl_pct:.2f}%",
            "active_positions": latest.active_positions,
            "model_accuracy": f"{latest.model_accuracy:.1%}",
            "recent_alerts": len(recent_alerts),
            "total_metrics_collected": len(self.performance_history)
        }
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily performance report."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Get today's data
        today = datetime.now().date()
        today_metrics = [
            m for m in self.performance_history
            if m.timestamp.date() == today
        ]
        
        if not today_metrics:
            return {"error": "No data for today"}
        
        # Calculate daily statistics
        start_equity = today_metrics[0].total_equity
        end_equity = today_metrics[-1].total_equity
        daily_return = ((end_equity - start_equity) / start_equity) * 100
        
        max_equity = max(m.total_equity for m in today_metrics)
        min_equity = min(m.total_equity for m in today_metrics)
        intraday_drawdown = ((max_equity - min_equity) / max_equity) * 100
        
        total_trades = sum(m.trades_today for m in today_metrics)
        avg_accuracy = np.mean([m.model_accuracy for m in today_metrics if m.model_accuracy > 0])
        
        today_alerts = [a for a in self.alerts if a.timestamp.date() == today]
        
        report = {
            "date": today.isoformat(),
            "performance": {
                "daily_return_pct": daily_return,
                "start_equity": start_equity,
                "end_equity": end_equity,
                "max_equity": max_equity,
                "min_equity": min_equity,
                "intraday_drawdown_pct": intraday_drawdown
            },
            "trading": {
                "total_trades": total_trades,
                "avg_model_accuracy": avg_accuracy if not np.isnan(avg_accuracy) else 0,
                "signals_generated": len([s for s in self.signal_history if datetime.fromisoformat(s["timestamp"]).date() == today])
            },
            "alerts": {
                "total_alerts": len(today_alerts),
                "critical_alerts": len([a for a in today_alerts if a.level == "CRITICAL"]),
                "warning_alerts": len([a for a in today_alerts if a.level == "WARNING"])
            }
        }
        
        # Save daily report
        report_file = self.output_dir / f"daily_report_{today.strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report


def main():
    """Main monitoring execution."""
    parser = argparse.ArgumentParser(description="Production Performance Monitoring")
    parser.add_argument("--start", action="store_true", help="Start monitoring")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--report", action="store_true", help="Generate daily report")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    if args.start:
        print("🚀 Starting production monitoring...")
        monitor.start_monitoring(args.interval)
        
        try:
            # Keep main thread alive
            while monitor.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n📊 Stopping monitoring...")
            monitor.stop_monitoring()
    
    elif args.status:
        status = monitor.get_current_status()
        print("📊 Current Monitoring Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    elif args.report:
        report = monitor.generate_daily_report()
        print("📈 Daily Performance Report:")
        print(json.dumps(report, indent=2, default=str))
    
    else:
        print("Use --start to begin monitoring, --status for current status, or --report for daily report")


if __name__ == "__main__":
    import argparse
    main()