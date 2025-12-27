#!/usr/bin/env python3
"""
Quantum Trader V3 – Audit Logger
Collects entries from realtime monitor + weekly jobs and sends daily summary.
"""
import pathlib
import datetime
import smtplib
import os
import sys
from email.message import EmailMessage
from typing import List, Optional

# Configuration
# Try /mnt/status first, fallback to ~/quantum_trader/status
try:
    LOG_DIR = pathlib.Path("/mnt/status")
    LOG_DIR.mkdir(exist_ok=True, parents=True)
except PermissionError:
    LOG_DIR = pathlib.Path.home() / "quantum_trader" / "status"
    LOG_DIR.mkdir(exist_ok=True, parents=True)

AUDIT_FILE = LOG_DIR / "AUTO_REPAIR_AUDIT.log"
WEBHOOK = os.getenv("QT_ALERT_WEBHOOK", "")
EMAIL_TO = os.getenv("QT_AUDIT_EMAIL", "")
SMTP_SERVER = os.getenv("QT_SMTP_SERVER", "localhost")
SMTP_PORT = int(os.getenv("QT_SMTP_PORT", "25"))


def log_action(action: str) -> None:
    """
    Log an auto-repair action to the audit file
    
    Args:
        action: Description of the action performed
    """
    ts = datetime.datetime.now(datetime.UTC).isoformat().replace('+00:00', 'Z')
    
    try:
        AUDIT_FILE.parent.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        pass  # Directory already exists or will fail on write
    
    log_entry = f"[{ts}] {action}\n"
    
    try:
        with open(AUDIT_FILE, "a") as f:
            f.write(log_entry)
        print(f"✅ Audit logged: {action}")
    except Exception as e:
        print(f"⚠️ Failed to log audit entry: {e}")


def get_today_entries() -> List[str]:
    """Get all audit entries from today"""
    if not AUDIT_FILE.exists():
        return []
    
    today = datetime.date.today().isoformat()
    
    try:
        with open(AUDIT_FILE, "r") as f:
            lines = [line.strip() for line in f.readlines() if today in line]
        return lines
    except Exception as e:
        print(f"⚠️ Failed to read audit file: {e}")
        return []


def send_webhook(message: str) -> bool:
    """Send notification via Discord/Slack webhook"""
    if not WEBHOOK:
        return False
    
    try:
        import requests
        
        # Truncate message to fit webhook limits (Discord: 2000 chars)
        content = message[:1900] if len(message) > 1900 else message
        
        response = requests.post(
            WEBHOOK,
            json={"content": content},
            timeout=10
        )
        
        if response.status_code in [200, 204]:
            print("✅ Webhook notification sent")
            return True
        else:
            print(f"⚠️ Webhook failed: HTTP {response.status_code}")
            return False
            
    except ImportError:
        print("⚠️ requests library not available for webhook")
        return False
    except Exception as e:
        print(f"⚠️ Webhook error: {e}")
        return False


def send_email(subject: str, body: str) -> bool:
    """Send notification via email"""
    if not EMAIL_TO:
        return False
    
    try:
        msg = EmailMessage()
        msg["To"] = EMAIL_TO
        msg["From"] = "quantum@vps"
        msg["Subject"] = subject
        msg.set_content(body)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.send_message(msg)
        
        print(f"✅ Email sent to {EMAIL_TO}")
        return True
        
    except Exception as e:
        print(f"⚠️ Email error: {e}")
        return False


def daily_summary() -> str:
    """
    Generate and send daily audit summary
    
    Returns:
        Summary text
    """
    today = datetime.date.today()
    entries = get_today_entries()
    
    if not entries:
        summary = f"Quantum Trader V3 – Daily Audit Summary ({today})\n\nNo audit events recorded today."
        print("📊 No audit events today")
    else:
        # Get last 50 entries (or all if fewer)
        recent_entries = entries[-50:]
        entry_count = len(entries)
        
        summary_lines = [
            f"Quantum Trader V3 – Daily Audit Summary ({today})",
            f"",
            f"📊 Total Actions: {entry_count}",
            f"",
            "Recent Actions:",
            "=" * 60,
        ]
        
        for entry in recent_entries:
            summary_lines.append(entry)
        
        summary_lines.extend([
            "=" * 60,
            "",
            f"✅ {entry_count} auto-repair actions executed"
        ])
        
        summary = "\n".join(summary_lines)
        print(f"📊 Generated summary with {entry_count} entries")
    
    # Send notifications
    webhook_sent = send_webhook(summary)
    email_sent = send_email(f"Quantum Trader Audit {today}", summary)
    
    if webhook_sent or email_sent:
        print("✅ Daily summary notifications sent")
    elif WEBHOOK or EMAIL_TO:
        print("⚠️ Failed to send daily summary notifications")
    else:
        print("ℹ️ No notification channels configured (set QT_ALERT_WEBHOOK or QT_AUDIT_EMAIL)")
    
    return summary


def get_audit_stats() -> dict:
    """Get statistics from audit log"""
    if not AUDIT_FILE.exists():
        return {
            "total_entries": 0,
            "today_entries": 0,
            "last_action": None
        }
    
    try:
        with open(AUDIT_FILE, "r") as f:
            all_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        today = datetime.date.today().isoformat()
        today_lines = [line for line in all_lines if today in line]
        
        last_action = all_lines[-1] if all_lines else None
        
        return {
            "total_entries": len(all_lines),
            "today_entries": len(today_lines),
            "last_action": last_action
        }
    except Exception as e:
        print(f"⚠️ Failed to get audit stats: {e}")
        return {
            "total_entries": 0,
            "today_entries": 0,
            "last_action": None
        }


def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "daily_summary":
            print("=" * 70)
            print("📊 QUANTUM TRADER V3 - DAILY AUDIT SUMMARY")
            print("=" * 70)
            summary = daily_summary()
            print("\n" + summary)
            
        elif command == "stats":
            stats = get_audit_stats()
            print("=" * 70)
            print("📊 QUANTUM TRADER V3 - AUDIT STATISTICS")
            print("=" * 70)
            print(f"Total Entries: {stats['total_entries']}")
            print(f"Today's Entries: {stats['today_entries']}")
            if stats['last_action']:
                print(f"Last Action: {stats['last_action']}")
            
        elif command == "test":
            print("🧪 Testing audit logger...")
            log_action("Test audit entry - system check")
            print("✅ Test complete")
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: audit_logger.py [daily_summary|stats|test]")
            sys.exit(1)
    else:
        print("Quantum Trader V3 - Audit Logger")
        print("Usage: audit_logger.py [daily_summary|stats|test]")
        print("")
        print("Commands:")
        print("  daily_summary  - Generate and send daily audit report")
        print("  stats          - Show audit log statistics")
        print("  test           - Test logging functionality")


if __name__ == "__main__":
    main()
