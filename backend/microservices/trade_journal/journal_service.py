"""
PHASE 7: TRADE JOURNAL & PERFORMANCE ANALYTICS
================================================
Autonomous trade logging, PnL analysis, and performance reporting system.

Features:
- Reads trade_log from Redis
- Calculates Sharpe, Sortino, Drawdown, WinRate
- Generates daily JSON reports
- Publishes to Governance Dashboard
- Supports weekly email alerts

Author: Quantum Trader AI System
Date: 2025-12-20
"""

import os
import json
import redis
import time
import statistics
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

# Redis connection
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Report storage directory
REPORT_DIR = "/app/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Configuration
UPDATE_INTERVAL = int(os.getenv("REPORT_INTERVAL_HOURS", 6)) * 3600  # Default: 6 hours
MAX_TRADES_TO_ANALYZE = int(os.getenv("MAX_TRADES", 1000))  # Analyze last 1000 trades
STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", 100000))  # Default: $100,000


def calc_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: List of equity values over time
    
    Returns:
        Maximum drawdown as percentage
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
    
    max_equity = equity_curve[0]
    max_dd = 0.0
    
    for equity in equity_curve:
        if equity > max_equity:
            max_equity = equity
        
        if max_equity > 0:
            dd = (max_equity - equity) / max_equity
            max_dd = max(max_dd, dd)
    
    return round(max_dd * 100, 2)


def calc_sharpe(pnls: List[float]) -> float:
    """
    Calculate Sharpe ratio (annualized).
    
    Args:
        pnls: List of PnL percentages
    
    Returns:
        Sharpe ratio
    """
    if len(pnls) < 2:
        return 0.0
    
    try:
        mean = statistics.mean(pnls)
        stdev = statistics.stdev(pnls)
        
        if stdev == 0:
            return 0.0
        
        # Annualize: sqrt(252 trading days)
        sharpe = (mean / stdev) * (252 ** 0.5)
        return round(sharpe, 2)
    except Exception as e:
        logging.error(f"Error calculating Sharpe: {e}")
        return 0.0


def calc_sortino(pnls: List[float]) -> float:
    """
    Calculate Sortino ratio (annualized).
    Uses only downside deviation (negative returns).
    
    Args:
        pnls: List of PnL percentages
    
    Returns:
        Sortino ratio
    """
    if len(pnls) < 2:
        return 0.0
    
    try:
        negative_pnls = [p for p in pnls if p < 0]
        
        if not negative_pnls:
            return 0.0  # No downside risk
        
        mean = statistics.mean(pnls)
        downside_dev = (sum(p**2 for p in negative_pnls) / len(negative_pnls)) ** 0.5
        
        if downside_dev == 0:
            return 0.0
        
        # Annualize
        sortino = (mean / downside_dev) * (252 ** 0.5)
        return round(sortino, 2)
    except Exception as e:
        logging.error(f"Error calculating Sortino: {e}")
        return 0.0


def calc_win_rate(pnls: List[float]) -> float:
    """Calculate win rate percentage."""
    if not pnls:
        return 0.0
    
    wins = sum(1 for p in pnls if p > 0)
    return round((wins / len(pnls)) * 100, 2)


def calc_profit_factor(pnls: List[float]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Returns:
        Profit factor (>1 is profitable)
    """
    if not pnls:
        return 0.0
    
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    
    if gross_loss == 0:
        return 0.0 if gross_profit == 0 else 999.99  # Infinite profit factor
    
    return round(gross_profit / gross_loss, 2)


def get_trades_from_redis() -> List[Dict]:
    """
    Fetch trade logs from Redis.
    
    Returns:
        List of trade dictionaries
    """
    try:
        logs = r.lrange("trade_log", 0, MAX_TRADES_TO_ANALYZE)
        trades = []
        
        for log in logs:
            try:
                trade = json.loads(log)
                trades.append(trade)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse trade log: {e}")
                continue
        
        return trades
    except Exception as e:
        logging.error(f"Error fetching trades from Redis: {e}")
        return []


def generate_report() -> Optional[Dict]:
    """
    Generate comprehensive performance report.
    
    Returns:
        Performance report dictionary
    """
    logging.info("📊 Generating performance report...")
    
    # Fetch trades
    trades = get_trades_from_redis()
    
    if not trades:
        logging.warning("⚠️  No trades found in trade_log")
        return None
    
    # Reverse to chronological order (oldest first)
    trades.reverse()
    
    # Extract PnLs
    pnls = [trade.get("pnl", 0) for trade in trades]
    
    # Calculate equity curve
    equity_curve = [STARTING_EQUITY]
    for pnl in pnls:
        # PnL is in percentage
        new_equity = equity_curve[-1] * (1 + pnl / 100)
        equity_curve.append(new_equity)
    
    # Calculate metrics
    total_trades = len(pnls)
    win_rate = calc_win_rate(pnls)
    total_pnl_pct = round(sum(pnls), 2)
    sharpe = calc_sharpe(pnls)
    sortino = calc_sortino(pnls)
    max_drawdown = calc_drawdown(equity_curve)
    profit_factor = calc_profit_factor(pnls)
    latest_equity = round(equity_curve[-1], 2)
    
    # Calculate additional statistics
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    avg_win = round(statistics.mean(wins), 2) if wins else 0.0
    avg_loss = round(statistics.mean(losses), 2) if losses else 0.0
    largest_win = round(max(pnls), 2) if pnls else 0.0
    largest_loss = round(min(pnls), 2) if pnls else 0.0
    
    # Build report
    report = {
        "date": datetime.utcnow().isoformat(),
        "timestamp": int(time.time()),
        
        # Trading statistics
        "total_trades": total_trades,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_%": win_rate,
        
        # Performance metrics
        "total_pnl_%": total_pnl_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "max_drawdown_%": max_drawdown,
        
        # Equity
        "starting_equity": STARTING_EQUITY,
        "current_equity": latest_equity,
        "equity_change_%": round(((latest_equity - STARTING_EQUITY) / STARTING_EQUITY) * 100, 2),
        
        # Win/Loss analysis
        "avg_win_%": avg_win,
        "avg_loss_%": avg_loss,
        "largest_win_%": largest_win,
        "largest_loss_%": largest_loss,
        "avg_trade_%": round(statistics.mean(pnls), 2) if pnls else 0.0,
        
        # Latest trade info
        "latest_trade": trades[-1] if trades else None
    }
    
    # Save to disk
    report_filename = f"daily_report_{datetime.utcnow().strftime('%Y-%m-%d')}.json"
    report_path = os.path.join(REPORT_DIR, report_filename)
    
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"✅ Report saved → {report_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save report: {e}")
    
    # Store latest report in Redis
    try:
        r.set("latest_report", json.dumps(report))
        r.set("journal_last_update", datetime.utcnow().isoformat())
        logging.info("✅ Report published to Redis")
    except Exception as e:
        logging.error(f"❌ Failed to publish to Redis: {e}")
    
    # Log summary
    logging.info(f"""
    📊 PERFORMANCE SUMMARY
    =====================
    Total Trades: {total_trades}
    Win Rate: {win_rate}%
    Total PnL: {total_pnl_pct}%
    Sharpe: {sharpe}
    Sortino: {sortino}
    Max Drawdown: {max_drawdown}%
    Profit Factor: {profit_factor}
    Current Equity: ${latest_equity:,.2f}
    """)
    
    return report


def check_alert_conditions(report: Dict) -> None:
    """
    Check if alert conditions are met and publish to alerts system.
    
    Args:
        report: Performance report dictionary
    """
    alerts = []
    
    # Alert on high drawdown
    if report["max_drawdown_%"] > 10:
        alerts.append({
            "type": "HIGH_DRAWDOWN",
            "severity": "WARNING",
            "message": f"⚠️  Max drawdown exceeded 10%: {report['max_drawdown_%']}%",
            "metric": report["max_drawdown_%"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Alert on low win rate
    if report["total_trades"] > 20 and report["win_rate_%"] < 50:
        alerts.append({
            "type": "LOW_WIN_RATE",
            "severity": "WARNING",
            "message": f"⚠️  Win rate below 50%: {report['win_rate_%']}%",
            "metric": report["win_rate_%"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Alert on negative Sharpe
    if report["sharpe_ratio"] < 0:
        alerts.append({
            "type": "NEGATIVE_SHARPE",
            "severity": "CRITICAL",
            "message": f"🚨 Negative Sharpe ratio: {report['sharpe_ratio']}",
            "metric": report["sharpe_ratio"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Alert on equity loss > 5%
    if report["equity_change_%"] < -5:
        alerts.append({
            "type": "EQUITY_LOSS",
            "severity": "CRITICAL",
            "message": f"🚨 Equity down {report['equity_change_%']}%",
            "metric": report["equity_change_%"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Publish alerts
    if alerts:
        for alert in alerts:
            try:
                r.lpush("journal_alerts", json.dumps(alert))
                logging.warning(f"🚨 ALERT: {alert['message']}")
            except Exception as e:
                logging.error(f"Failed to publish alert: {e}")
    else:
        logging.info("✅ All performance metrics within acceptable range")


def run_loop():
    """Main execution loop."""
    logging.info("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  PHASE 7: TRADE JOURNAL & PERFORMANCE ANALYTICS          ║
    ║  Status: ACTIVE                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    logging.info(f"📊 Report interval: Every {UPDATE_INTERVAL // 3600} hours")
    logging.info(f"📊 Starting equity: ${STARTING_EQUITY:,.2f}")
    logging.info(f"📊 Max trades to analyze: {MAX_TRADES_TO_ANALYZE}")
    logging.info(f"📊 Report directory: {REPORT_DIR}")
    logging.info("🚀 Starting performance analytics loop...")
    
    # Generate initial report immediately
    try:
        report = generate_report()
        if report:
            check_alert_conditions(report)
    except Exception as e:
        logging.error(f"Error in initial report generation: {e}")
    
    # Main loop
    while True:
        try:
            time.sleep(UPDATE_INTERVAL)
            
            logging.info(f"⏰ Generating scheduled report (interval: {UPDATE_INTERVAL // 3600}h)...")
            report = generate_report()
            
            if report:
                check_alert_conditions(report)
            
        except KeyboardInterrupt:
            logging.info("🛑 Shutting down Trade Journal service...")
            break
        except Exception as e:
            logging.error(f"❌ Error in main loop: {e}")
            time.sleep(300)  # Wait 5 minutes on error


if __name__ == "__main__":
    run_loop()
