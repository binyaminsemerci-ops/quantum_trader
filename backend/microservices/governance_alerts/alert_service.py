#!/usr/bin/env python3
"""
Phase 4I: Governance Alert System
Real-time monitoring and alerting for AI Governance components
"""
import os
import smtplib
import requests
import redis
import time
import json
import psutil
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis connection
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"), 
    port=int(os.getenv("REDIS_PORT", 6379)), 
    decode_responses=True
)


class GovernanceAlertService:
    """
    24/7 Real-time alerting system for AI Governance
    Monitors: CPU, Memory, MAPE drift, Model validation failures
    Notifies via: Email, Telegram
    """
    
    def __init__(self):
        # Email configuration
        self.email_to = os.getenv("ALERT_EMAIL")
        self.email_user = os.getenv("EMAIL_USER")
        self.email_pass = os.getenv("EMAIL_PASS")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        
        # Telegram configuration
        self.tg_token = os.getenv("TELEGRAM_TOKEN")
        self.tg_chat = os.getenv("TELEGRAM_CHAT_ID")
        
        # Alert thresholds
        self.cpu_threshold = float(os.getenv("CPU_THRESHOLD", 85))
        self.mem_threshold = float(os.getenv("MEM_THRESHOLD", 80))
        self.mape_threshold = float(os.getenv("MAPE_THRESHOLD", 0.06))
        self.sharpe_threshold = float(os.getenv("SHARPE_THRESHOLD", 0.8))
        
        # Alert cooldown (prevent spam)
        self.last_alerts = {}
        self.cooldown_seconds = 300  # 5 minutes
        
        logger.info("=" * 60)
        logger.info("Phase 4I: Governance Alert System Initialized")
        logger.info("=" * 60)
        logger.info(f"Email alerts: {'✅ Enabled' if self.email_to else '❌ Disabled'}")
        logger.info(f"Telegram alerts: {'✅ Enabled' if self.tg_token else '❌ Disabled'}")
        logger.info(f"Thresholds: CPU={self.cpu_threshold}%, MEM={self.mem_threshold}%, MAPE={self.mape_threshold}")
        logger.info("=" * 60)

    def should_alert(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert to prevent spam"""
        now = time.time()
        if alert_key in self.last_alerts:
            if now - self.last_alerts[alert_key] < self.cooldown_seconds:
                return False
        self.last_alerts[alert_key] = now
        return True

    def send_email(self, subject: str, body: str):
        """Send email alert via SMTP"""
        if not self.email_to or not self.email_user or not self.email_pass:
            logger.debug("Email not configured, skipping email alert")
            return
        
        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"🚨 Quantum Trader Alert: {subject}"
            msg["From"] = f"AI Governance <{self.email_user}>"
            msg["To"] = self.email_to
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                server.send_message(msg)
            
            logger.info(f"[✅ Email Alert] Sent to {self.email_to}: {subject}")
        except Exception as e:
            logger.error(f"[❌ Email Error] {e}")

    def send_telegram(self, text: str):
        """Send Telegram alert via Bot API"""
        if not self.tg_token or not self.tg_chat:
            logger.debug("Telegram not configured, skipping Telegram alert")
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            payload = {
                "chat_id": self.tg_chat,
                "text": f"🚨 *Quantum Trader Alert*\n\n{text}",
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"[✅ Telegram Alert] Sent: {text[:50]}...")
            else:
                logger.error(f"[❌ Telegram Error] Status {response.status_code}")
        except Exception as e:
            logger.error(f"[❌ Telegram Error] {e}")

    def check_system_metrics(self):
        """Monitor CPU and Memory usage"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory().percent
            
            if cpu > self.cpu_threshold and self.should_alert("high_cpu"):
                self.notify(
                    "High CPU Usage",
                    f"⚠️ CPU usage at {cpu:.1f}% (threshold: {self.cpu_threshold}%)\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Check container resource limits"
                )
            
            if mem > self.mem_threshold and self.should_alert("high_memory"):
                self.notify(
                    "High Memory Usage",
                    f"⚠️ Memory usage at {mem:.1f}% (threshold: {self.mem_threshold}%)\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Consider scaling resources"
                )
                
            # Log metrics periodically
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                logger.info(f"[System Metrics] CPU: {cpu:.1f}%, Memory: {mem:.1f}%")
                
        except Exception as e:
            logger.error(f"[Error] System metrics check failed: {e}")

    def check_model_drift(self):
        """Monitor MAPE and model performance metrics"""
        try:
            # Check latest metrics from Redis
            metrics_json = r.get("latest_metrics")
            if not metrics_json:
                return
            
            data = json.loads(metrics_json)
            mape = data.get("mape", 0)
            sharpe = data.get("sharpe_ratio", 999)
            
            if mape > self.mape_threshold and self.should_alert("high_mape"):
                self.notify(
                    "Model Drift Detected",
                    f"⚠️ MAPE={mape:.4f} exceeded threshold ({self.mape_threshold})\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Model retraining may be required\n"
                    f"Status: Phase 4F Retrainer should trigger automatically"
                )
            
            if sharpe < self.sharpe_threshold and self.should_alert("low_sharpe"):
                self.notify(
                    "Low Sharpe Ratio",
                    f"⚠️ Sharpe Ratio={sharpe:.3f} below threshold ({self.sharpe_threshold})\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Review model predictions and governance weights"
                )
                
        except json.JSONDecodeError:
            logger.debug("Invalid JSON in latest_metrics")
        except Exception as e:
            logger.error(f"[Error] Model drift check failed: {e}")

    def check_governance_state(self):
        """Monitor governance weights and configuration"""
        try:
            # Check if governance is active
            governance_active = r.get("governance_active")
            if governance_active == "false" and self.should_alert("governance_inactive"):
                self.notify(
                    "Governance System Inactive",
                    f"⚠️ Predictive Governance (Phase 4E) is not active\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Check AI Engine health and restart if needed"
                )
            
            # Check model weights
            weights = r.hgetall("governance_weights")
            if not weights and self.should_alert("no_weights"):
                self.notify(
                    "No Model Weights",
                    f"⚠️ Governance weights not found in Redis\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Verify Phase 4E Predictive Governance is running"
                )
                
        except Exception as e:
            logger.error(f"[Error] Governance state check failed: {e}")

    def check_validation_failures(self):
        """Monitor model validation log for REJECT events"""
        try:
            log_path = "/app/logs/model_validation.log"
            if not os.path.exists(log_path):
                logger.debug(f"Validation log not found: {log_path}")
                return
            
            # Read last 10 lines
            with open(log_path, 'r') as f:
                lines = f.readlines()[-10:]
            
            for line in lines:
                if "REJECT" in line and self.should_alert(f"validation_reject_{line[:50]}"):
                    self.notify(
                        "Model Validation Rejected",
                        f"⚠️ Model failed validation check\n"
                        f"Details: {line.strip()}\n"
                        f"Timestamp: {datetime.utcnow().isoformat()}\n"
                        f"Action: Review Phase 4G validation criteria"
                    )
                    
        except Exception as e:
            logger.error(f"[Error] Validation log check failed: {e}")

    def check_retrainer_status(self):
        """Monitor retraining pipeline health"""
        try:
            retrainer_enabled = r.get("retrainer_enabled")
            if retrainer_enabled == "false" and self.should_alert("retrainer_disabled"):
                self.notify(
                    "Retrainer Disabled",
                    f"⚠️ Phase 4F Adaptive Retraining is disabled\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}\n"
                    f"Action: Models will not automatically retrain on drift"
                )
                
        except Exception as e:
            logger.error(f"[Error] Retrainer status check failed: {e}")

    def notify(self, title: str, message: str):
        """Send notification via all configured channels"""
        timestamp = datetime.utcnow().isoformat()
        full_message = f"[{timestamp}] {title}\n\n{message}"
        
        # Console log
        logger.warning(f"[🚨 ALERT] {title}")
        logger.warning(message)
        
        # Email notification
        self.send_email(title, full_message)
        
        # Telegram notification
        self.send_telegram(full_message)
        
        # Store in Redis for dashboard
        try:
            alert_data = {
                "timestamp": timestamp,
                "title": title,
                "message": message,
                "severity": "warning"
            }
            r.lpush("governance_alerts", json.dumps(alert_data))
            r.ltrim("governance_alerts", 0, 99)  # Keep last 100 alerts
        except Exception as e:
            logger.error(f"[Error] Failed to store alert in Redis: {e}")

    def run_monitor_loop(self):
        """Main monitoring loop - runs 24/7"""
        logger.info("🚀 Starting 24/7 monitoring loop...")
        logger.info("Checks run every 2 minutes")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"[Cycle {cycle}] Running health checks...")
                
                # Run all checks
                self.check_system_metrics()
                self.check_model_drift()
                self.check_governance_state()
                self.check_validation_failures()
                self.check_retrainer_status()
                
                logger.info(f"[Cycle {cycle}] All checks complete ✓")
                
                # Sleep for 2 minutes
                time.sleep(120)
                
            except KeyboardInterrupt:
                logger.info("Shutting down alert service...")
                break
            except Exception as e:
                logger.error(f"[Error] Monitor loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retry


def main():
    """Entry point"""
    try:
        service = GovernanceAlertService()
        service.run_monitor_loop()
    except Exception as e:
        logger.critical(f"[FATAL] Alert service failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
