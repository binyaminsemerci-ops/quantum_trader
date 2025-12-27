"""
ESS Alert System - Slack, SMS, and Email Notifications

Subscribes to EmergencyStopEvent and EmergencyResetEvent from EventBus
and sends notifications to configured channels.

Author: Quantum Trader Team
Date: 2024-11-30
"""

import asyncio
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Protocol

import aiohttp

logger = logging.getLogger(__name__)


# ============================================================================
# Event Types (must match emergency_stop_system.py)
# ============================================================================

@dataclass
class EmergencyStopEvent:
    """Event published when ESS activates."""
    reason: str
    timestamp: datetime
    triggered_by: str  # Which evaluator triggered
    details: dict[str, Any]


@dataclass
class EmergencyResetEvent:
    """Event published when ESS is reset."""
    reset_by: str
    timestamp: datetime
    duration_seconds: float
    previous_reason: str


# ============================================================================
# EventBus Protocol
# ============================================================================

class EventBus(Protocol):
    """EventBus interface for alert subscription."""
    async def subscribe(self, event_type: type, handler) -> None:
        """Subscribe to event type."""
        ...


# ============================================================================
# Base Alerter
# ============================================================================

class BaseAlerter(ABC):
    """Abstract base class for alert channels."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def send_emergency_stop_alert(self, event: EmergencyStopEvent) -> bool:
        """Send emergency stop alert. Returns True if successful."""
        pass
    
    @abstractmethod
    async def send_reset_alert(self, event: EmergencyResetEvent) -> bool:
        """Send reset alert. Returns True if successful."""
        pass
    
    async def handle_emergency_stop_event(self, event: EmergencyStopEvent) -> None:
        """Event handler for emergency stop events."""
        if not self.enabled:
            self.logger.debug("Alerter disabled, skipping emergency stop alert")
            return
        
        try:
            success = await self.send_emergency_stop_alert(event)
            if success:
                self.logger.info(f"✅ Emergency stop alert sent via {self.__class__.__name__}")
            else:
                self.logger.warning(f"❌ Failed to send emergency stop alert via {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"❌ Error sending emergency stop alert: {e}", exc_info=True)
    
    async def handle_reset_event(self, event: EmergencyResetEvent) -> None:
        """Event handler for reset events."""
        if not self.enabled:
            self.logger.debug("Alerter disabled, skipping reset alert")
            return
        
        try:
            success = await self.send_reset_alert(event)
            if success:
                self.logger.info(f"✅ Reset alert sent via {self.__class__.__name__}")
            else:
                self.logger.warning(f"❌ Failed to send reset alert via {self.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"❌ Error sending reset alert: {e}", exc_info=True)


# ============================================================================
# Slack Alerter
# ============================================================================

class SlackAlerter(BaseAlerter):
    """Send ESS alerts to Slack via webhook."""
    
    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "ESS Bot",
        icon_emoji: str = ":rotating_light:",
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji
    
    async def send_emergency_stop_alert(self, event: EmergencyStopEvent) -> bool:
        """Send emergency stop alert to Slack."""
        message = self._format_emergency_stop_message(event)
        return await self._send_slack_message(message, color="danger")
    
    async def send_reset_alert(self, event: EmergencyResetEvent) -> bool:
        """Send reset alert to Slack."""
        message = self._format_reset_message(event)
        return await self._send_slack_message(message, color="good")
    
    def _format_emergency_stop_message(self, event: EmergencyStopEvent) -> dict:
        """Format emergency stop message for Slack."""
        fields = [
            {"title": "Reason", "value": event.reason, "short": False},
            {"title": "Triggered By", "value": event.triggered_by, "short": True},
            {"title": "Timestamp", "value": event.timestamp.isoformat(), "short": True},
        ]
        
        # Add relevant details
        if "daily_loss_pct" in event.details:
            fields.append({
                "title": "Daily Loss",
                "value": f"{event.details['daily_loss_pct']:.2f}%",
                "short": True
            })
        
        if "equity_dd_pct" in event.details:
            fields.append({
                "title": "Equity Drawdown",
                "value": f"{event.details['equity_dd_pct']:.2f}%",
                "short": True
            })
        
        return {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "channel": self.channel,
            "attachments": [{
                "color": "danger",
                "title": "🚨 EMERGENCY STOP ACTIVATED",
                "text": "The Emergency Stop System has been triggered. All trading halted.",
                "fields": fields,
                "footer": "Quantum Trader ESS",
                "ts": int(event.timestamp.timestamp())
            }]
        }
    
    def _format_reset_message(self, event: EmergencyResetEvent) -> dict:
        """Format reset message for Slack."""
        duration_minutes = event.duration_seconds / 60
        
        return {
            "username": self.username,
            "icon_emoji": ":white_check_mark:",
            "channel": self.channel,
            "attachments": [{
                "color": "good",
                "title": "✅ Emergency Stop Reset",
                "text": "The Emergency Stop System has been reset. Trading can resume.",
                "fields": [
                    {"title": "Reset By", "value": event.reset_by, "short": True},
                    {"title": "Duration", "value": f"{duration_minutes:.1f} minutes", "short": True},
                    {"title": "Previous Reason", "value": event.previous_reason, "short": False},
                    {"title": "Timestamp", "value": event.timestamp.isoformat(), "short": True},
                ],
                "footer": "Quantum Trader ESS",
                "ts": int(event.timestamp.timestamp())
            }]
        }
    
    async def _send_slack_message(self, payload: dict, color: str = "danger") -> bool:
        """Send message to Slack webhook."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        return True
                    else:
                        text = await resp.text()
                        self.logger.error(f"Slack webhook failed: {resp.status} - {text}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack message: {e}")
            return False


# ============================================================================
# SMS Alerter (Twilio)
# ============================================================================

class SMSAlerter(BaseAlerter):
    """Send ESS alerts via SMS using Twilio."""
    
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        to_numbers: list[str],
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.to_numbers = to_numbers
        self.twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    
    async def send_emergency_stop_alert(self, event: EmergencyStopEvent) -> bool:
        """Send emergency stop alert via SMS."""
        message = self._format_emergency_stop_sms(event)
        return await self._send_sms_to_all(message)
    
    async def send_reset_alert(self, event: EmergencyResetEvent) -> bool:
        """Send reset alert via SMS."""
        message = self._format_reset_sms(event)
        return await self._send_sms_to_all(message)
    
    def _format_emergency_stop_sms(self, event: EmergencyStopEvent) -> str:
        """Format emergency stop SMS (max 160 chars)."""
        details_str = ""
        if "daily_loss_pct" in event.details:
            details_str = f" Loss: {event.details['daily_loss_pct']:.1f}%"
        
        return (
            f"🚨 ESS ACTIVATED: {event.reason}. "
            f"Triggered by {event.triggered_by}.{details_str} "
            f"All trading HALTED."
        )[:160]
    
    def _format_reset_sms(self, event: EmergencyResetEvent) -> str:
        """Format reset SMS (max 160 chars)."""
        duration_min = event.duration_seconds / 60
        return (
            f"✅ ESS RESET by {event.reset_by}. "
            f"Down for {duration_min:.0f}m. Trading can resume."
        )[:160]
    
    async def _send_sms_to_all(self, message: str) -> bool:
        """Send SMS to all configured numbers."""
        success_count = 0
        
        for to_number in self.to_numbers:
            try:
                if await self._send_single_sms(to_number, message):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"Failed to send SMS to {to_number}: {e}")
        
        return success_count > 0
    
    async def _send_single_sms(self, to_number: str, message: str) -> bool:
        """Send SMS to a single number via Twilio."""
        try:
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            data = {
                "From": self.from_number,
                "To": to_number,
                "Body": message
            }
            
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.post(self.twilio_url, data=data, timeout=15) as resp:
                    if resp.status in (200, 201):
                        return True
                    else:
                        text = await resp.text()
                        self.logger.error(f"Twilio API failed: {resp.status} - {text}")
                        return False
        except Exception as e:
            self.logger.error(f"Failed to send SMS: {e}")
            return False


# ============================================================================
# Email Alerter (SMTP)
# ============================================================================

class EmailAlerter(BaseAlerter):
    """Send ESS alerts via email using SMTP."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
        enabled: bool = True
    ):
        super().__init__(enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    async def send_emergency_stop_alert(self, event: EmergencyStopEvent) -> bool:
        """Send emergency stop alert via email."""
        subject = "🚨 QUANTUM TRADER: Emergency Stop Activated"
        body = self._format_emergency_stop_email(event)
        return await self._send_email(subject, body)
    
    async def send_reset_alert(self, event: EmergencyResetEvent) -> bool:
        """Send reset alert via email."""
        subject = "✅ QUANTUM TRADER: Emergency Stop Reset"
        body = self._format_reset_email(event)
        return await self._send_email(subject, body)
    
    def _format_emergency_stop_email(self, event: EmergencyStopEvent) -> str:
        """Format emergency stop email body."""
        details_lines = []
        for key, value in event.details.items():
            details_lines.append(f"  - {key}: {value}")
        
        details_str = "\n".join(details_lines) if details_lines else "  (none)"
        
        return f"""
EMERGENCY STOP SYSTEM ACTIVATED

The Emergency Stop System has been triggered and all trading operations have been halted.

ACTIVATION DETAILS:
-------------------
Reason:       {event.reason}
Triggered By: {event.triggered_by}
Timestamp:    {event.timestamp.isoformat()}

ADDITIONAL DETAILS:
{details_str}

IMMEDIATE ACTIONS TAKEN:
- All open positions have been closed
- All pending orders have been canceled
- Trading is now DISABLED system-wide

REQUIRED ACTIONS:
1. Investigate the root cause
2. Verify all positions are closed
3. Review system logs
4. Manually reset ESS when safe to resume trading

DO NOT reset the system until the underlying issue is resolved.

---
Quantum Trader Emergency Stop System
This is an automated alert - do not reply
"""
    
    def _format_reset_email(self, event: EmergencyResetEvent) -> str:
        """Format reset email body."""
        duration_minutes = event.duration_seconds / 60
        
        return f"""
EMERGENCY STOP SYSTEM RESET

The Emergency Stop System has been reset and trading operations can resume.

RESET DETAILS:
--------------
Reset By:        {event.reset_by}
Timestamp:       {event.timestamp.isoformat()}
Duration:        {duration_minutes:.1f} minutes
Previous Reason: {event.previous_reason}

SYSTEM STATUS:
- ESS is now INACTIVE
- Trading is ENABLED
- All systems operational

NEXT STEPS:
1. Monitor system closely for any issues
2. Verify trading operations resume normally
3. Review activation logs for patterns
4. Update procedures if necessary

---
Quantum Trader Emergency Stop System
This is an automated alert - do not reply
"""
    
    async def _send_email(self, subject: str, body: str) -> bool:
        """Send email via SMTP (blocking, runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._send_email_sync, subject, body)
    
    def _send_email_sync(self, subject: str, body: str) -> bool:
        """Send email synchronously via SMTP."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False


# ============================================================================
# Alert Manager
# ============================================================================

class ESSAlertManager:
    """
    Manages all ESS alert channels and subscribes to EventBus.
    
    Usage:
        manager = ESSAlertManager.from_env()
        await manager.subscribe_to_eventbus(event_bus)
    """
    
    def __init__(self, alerters: list[BaseAlerter]):
        self.alerters = alerters
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_env(cls) -> "ESSAlertManager":
        """Create alert manager from environment variables."""
        alerters: list[BaseAlerter] = []
        
        # Slack
        slack_enabled = os.getenv("QT_ESS_SLACK_ENABLED", "false").lower() == "true"
        if slack_enabled:
            webhook_url = os.getenv("QT_ESS_SLACK_WEBHOOK_URL")
            if webhook_url:
                alerters.append(SlackAlerter(
                    webhook_url=webhook_url,
                    channel=os.getenv("QT_ESS_SLACK_CHANNEL"),
                    username=os.getenv("QT_ESS_SLACK_USERNAME", "ESS Bot"),
                    icon_emoji=os.getenv("QT_ESS_SLACK_ICON_EMOJI", ":rotating_light:"),
                    enabled=True
                ))
                logger.info("✅ Slack alerter configured")
            else:
                logger.warning("❌ Slack enabled but QT_ESS_SLACK_WEBHOOK_URL not set")
        
        # SMS (Twilio)
        sms_enabled = os.getenv("QT_ESS_SMS_ENABLED", "false").lower() == "true"
        if sms_enabled:
            account_sid = os.getenv("QT_ESS_TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("QT_ESS_TWILIO_AUTH_TOKEN")
            from_number = os.getenv("QT_ESS_TWILIO_FROM_NUMBER")
            to_numbers_str = os.getenv("QT_ESS_TWILIO_TO_NUMBERS", "")
            
            if all([account_sid, auth_token, from_number, to_numbers_str]):
                to_numbers = [n.strip() for n in to_numbers_str.split(",") if n.strip()]
                alerters.append(SMSAlerter(
                    account_sid=account_sid,
                    auth_token=auth_token,
                    from_number=from_number,
                    to_numbers=to_numbers,
                    enabled=True
                ))
                logger.info(f"✅ SMS alerter configured ({len(to_numbers)} recipients)")
            else:
                logger.warning("❌ SMS enabled but Twilio credentials incomplete")
        
        # Email (SMTP)
        email_enabled = os.getenv("QT_ESS_EMAIL_ENABLED", "false").lower() == "true"
        if email_enabled:
            smtp_host = os.getenv("QT_ESS_SMTP_HOST")
            smtp_port_str = os.getenv("QT_ESS_SMTP_PORT", "587")
            username = os.getenv("QT_ESS_SMTP_USERNAME")
            password = os.getenv("QT_ESS_SMTP_PASSWORD")
            from_email = os.getenv("QT_ESS_EMAIL_FROM")
            to_emails_str = os.getenv("QT_ESS_EMAIL_TO", "")
            
            if all([smtp_host, username, password, from_email, to_emails_str]):
                to_emails = [e.strip() for e in to_emails_str.split(",") if e.strip()]
                alerters.append(EmailAlerter(
                    smtp_host=smtp_host,
                    smtp_port=int(smtp_port_str),
                    username=username,
                    password=password,
                    from_email=from_email,
                    to_emails=to_emails,
                    enabled=True
                ))
                logger.info(f"✅ Email alerter configured ({len(to_emails)} recipients)")
            else:
                logger.warning("❌ Email enabled but SMTP credentials incomplete")
        
        if not alerters:
            logger.warning("[WARNING] No alert channels configured - ESS will log only")
        
        return cls(alerters)
    
    async def subscribe_to_eventbus(self, event_bus: EventBus) -> None:
        """Subscribe all alerters to EventBus."""
        for alerter in self.alerters:
            await event_bus.subscribe(EmergencyStopEvent, alerter.handle_emergency_stop_event)
            await event_bus.subscribe(EmergencyResetEvent, alerter.handle_reset_event)
        
        self.logger.info(f"[OK] {len(self.alerters)} alert channel(s) subscribed to EventBus")
