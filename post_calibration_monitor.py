#!/usr/bin/env python3
"""
Post-Calibration Continuous Monitor
Collects metrics silently in background - does NOT trigger actions
Logs to: /root/logs/post_calibration_monitor.log

Usage:
    python3 post_calibration_monitor.py &
    # Let it run for 24-72 hours
    # Review log file to analyze behavioral changes
"""

import redis
import json
import time
import datetime
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional
import logging

# Configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
LOG_FILE = "/root/logs/post_calibration_monitor.log"
SAMPLE_INTERVAL = 30  # seconds
METRICS_WINDOW = 100  # Keep last N decisions for rolling stats

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

@dataclass
class Snapshot:
    """Single point-in-time measurement"""
    timestamp: str
    phase: str  # A, B, C, D
    
    # Core stability
    ai_engine_alive: bool
    calibration_loaded: bool
    exceptions_detected: bool
    
    # Equity & Risk
    equity: Optional[float]
    equity_age_sec: Optional[float]
    peak_equity: Optional[float]
    drawdown_pct: Optional[float]
    
    # Decision behavior
    decision_action: Optional[str]
    decision_confidence: Optional[float]
    
    # Risk Guards
    risk_guards_triggered: list
    
    # Meta stats (rolling window)
    avg_confidence_recent: Optional[float]
    buy_rate: Optional[float]
    sell_rate: Optional[float]
    hold_rate: Optional[float]


class PostCalibrationMonitor:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.decisions = deque(maxlen=METRICS_WINDOW)
        self.calibration_load_count = 0
        self.start_time = datetime.datetime.now()
        
        logging.info("=" * 80)
        logging.info("POST-CALIBRATION MONITOR STARTED")
        logging.info(f"Start time: {self.start_time.isoformat()}")
        logging.info("=" * 80)
    
    def get_phase(self) -> str:
        """Determine current monitoring phase based on elapsed time"""
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        if elapsed < 1800:  # 30 min
            return "A"
        elif elapsed < 21600:  # 6 hours
            return "B"
        elif elapsed < 86400:  # 24 hours
            return "C"
        else:
            return "D"
    
    def check_ai_engine(self) -> bool:
        """Check if AI Engine is responding"""
        try:
            health = self.r.get("quantum:health:ai_engine")
            if not health:
                return False
            data = json.loads(health)
            return data.get("status") == "healthy"
        except:
            return False
    
    def check_calibration_loaded(self) -> bool:
        """Check if calibration data is loaded"""
        try:
            cal_status = self.r.get("quantum:calibration:status")
            if cal_status:
                data = json.loads(cal_status)
                # Count loads (detect reloads)
                load_count = data.get("load_count", 0)
                if load_count > self.calibration_load_count:
                    logging.warning(f"🚨 CalibrationLoader RELOAD detected! Count: {load_count}")
                    self.calibration_load_count = load_count
                return data.get("loaded", False)
            return False
        except:
            return False
    
    def check_exceptions(self) -> bool:
        """Check for recent exceptions"""
        try:
            errors = self.r.lrange("quantum:errors:recent", 0, 9)
            # Look for calibration-related errors in last 10
            for error in errors:
                if "calibration" in error.lower():
                    logging.error(f"🚨 Calibration exception detected: {error}")
                    return True
            return False
        except:
            return False
    
    def get_equity_data(self) -> tuple:
        """Get current equity, age, peak, and drawdown"""
        try:
            equity_str = self.r.get("quantum:equity:current")
            if not equity_str:
                return None, None, None, None
            
            equity_data = json.loads(equity_str)
            equity = float(equity_data.get("equity", 0))
            timestamp = equity_data.get("timestamp", 0)
            peak = float(equity_data.get("peak_equity", equity))
            
            age = time.time() - timestamp
            drawdown = ((peak - equity) / peak * 100) if peak > 0 else 0
            
            return equity, age, peak, drawdown
        except:
            return None, None, None, None
    
    def get_latest_decision(self) -> tuple:
        """Get most recent decision from Meta-Agent"""
        try:
            decision_str = self.r.get("quantum:decision:latest")
            if not decision_str:
                return None, None
            
            decision = json.loads(decision_str)
            action = decision.get("action", "HOLD")
            confidence = decision.get("confidence", 0.0)
            
            # Store for rolling stats
            self.decisions.append({"action": action, "confidence": confidence})
            
            return action, confidence
        except:
            return None, None
    
    def get_risk_guards(self) -> list:
        """Get currently triggered risk guards"""
        try:
            guards_str = self.r.get("quantum:risk:active_guards")
            if not guards_str:
                return []
            
            guards_data = json.loads(guards_str)
            return guards_data.get("triggered", [])
        except:
            return []
    
    def calculate_rolling_stats(self) -> tuple:
        """Calculate rolling window statistics"""
        if not self.decisions:
            return None, None, None, None
        
        confidences = [d["confidence"] for d in self.decisions]
        avg_conf = sum(confidences) / len(confidences)
        
        buy_count = sum(1 for d in self.decisions if d["action"] == "BUY")
        sell_count = sum(1 for d in self.decisions if d["action"] == "SELL")
        hold_count = sum(1 for d in self.decisions if d["action"] == "HOLD")
        total = len(self.decisions)
        
        return (
            avg_conf,
            buy_count / total,
            sell_count / total,
            hold_count / total
        )
    
    def take_snapshot(self) -> Snapshot:
        """Collect all metrics at this moment"""
        phase = self.get_phase()
        equity, eq_age, peak, dd = self.get_equity_data()
        action, conf = self.get_latest_decision()
        guards = self.get_risk_guards()
        avg_conf, buy_rate, sell_rate, hold_rate = self.calculate_rolling_stats()
        
        return Snapshot(
            timestamp=datetime.datetime.now().isoformat(),
            phase=phase,
            ai_engine_alive=self.check_ai_engine(),
            calibration_loaded=self.check_calibration_loaded(),
            exceptions_detected=self.check_exceptions(),
            equity=equity,
            equity_age_sec=eq_age,
            peak_equity=peak,
            drawdown_pct=dd,
            decision_action=action,
            decision_confidence=conf,
            risk_guards_triggered=guards,
            avg_confidence_recent=avg_conf,
            buy_rate=buy_rate,
            sell_rate=sell_rate,
            hold_rate=hold_rate
        )
    
    def evaluate_snapshot(self, snap: Snapshot):
        """Log warnings for concerning patterns (informational only)"""
        
        # FASE A checks (0-30 min)
        if snap.phase == "A":
            if not snap.ai_engine_alive:
                logging.error("🚨 FASE A CRITICAL: AI Engine not responding!")
            if snap.exceptions_detected:
                logging.error("🚨 FASE A CRITICAL: Calibration exceptions detected!")
            if snap.decision_confidence is not None and (snap.decision_confidence < 0.01 or snap.decision_confidence > 0.99):
                logging.warning(f"⚠️  FASE A WARNING: Extreme confidence detected: {snap.decision_confidence:.3f}")
        
        # FASE B checks (0-6 hours)
        elif snap.phase == "B":
            if snap.hold_rate and snap.hold_rate > 0.85:
                logging.warning(f"⚠️  FASE B WARNING: High HOLD rate: {snap.hold_rate:.1%}")
            if snap.buy_rate and snap.buy_rate > 0.85:
                logging.warning(f"⚠️  FASE B WARNING: High BUY rate: {snap.buy_rate:.1%}")
            if snap.sell_rate and snap.sell_rate > 0.85:
                logging.warning(f"⚠️  FASE B WARNING: High SELL rate: {snap.sell_rate:.1%}")
        
        # FASE C checks (6-24 hours)
        elif snap.phase == "C":
            if snap.drawdown_pct and snap.drawdown_pct > 15:  # Adjust threshold based on normal
                logging.warning(f"⚠️  FASE C WARNING: Elevated drawdown: {snap.drawdown_pct:.1f}%")
            if snap.risk_guards_triggered:
                logging.info(f"ℹ️  FASE C INFO: Risk guards active: {snap.risk_guards_triggered}")
        
        # FASE D checks (24+ hours)
        elif snap.phase == "D":
            if snap.avg_confidence_recent:
                logging.info(f"ℹ️  FASE D BASELINE: Avg confidence: {snap.avg_confidence_recent:.3f}")
    
    def run(self):
        """Main monitoring loop"""
        logging.info("Monitoring loop started. Taking snapshots every 30 seconds.")
        
        try:
            while True:
                snap = self.take_snapshot()
                
                # Log snapshot
                logging.info(f"FASE {snap.phase} | "
                           f"AI:{snap.ai_engine_alive} | "
                           f"CAL:{snap.calibration_loaded} | "
                           f"EQ:${snap.equity:.2f if snap.equity else 0} | "
                           f"DD:{snap.drawdown_pct:.1f if snap.drawdown_pct else 0}% | "
                           f"ACT:{snap.decision_action or 'NONE'} | "
                           f"CONF:{snap.decision_confidence:.3f if snap.decision_confidence else 0} | "
                           f"GUARDS:{len(snap.risk_guards_triggered)}")
                
                # Evaluate for warnings
                self.evaluate_snapshot(snap)
                
                # Sleep until next sample
                time.sleep(SAMPLE_INTERVAL)
                
        except KeyboardInterrupt:
            logging.info("Monitor stopped by user (Ctrl+C)")
        except Exception as e:
            logging.error(f"Monitor crashed: {e}")
            raise


if __name__ == "__main__":
    monitor = PostCalibrationMonitor()
    monitor.run()
