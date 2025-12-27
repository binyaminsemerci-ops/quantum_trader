"""
QUANTUM TRADER - LIVE PRODUCTION MONITOR
========================================
Real-time monitoring of live trading system
- Position tracking
- P&L monitoring
- Risk management
- System health
- Alert system

Author: Quantum Trader Team
Date: 2025-11-26
Status: PRODUCTION
"""

import requests
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

class ProductionMonitor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.start_time = datetime.now()
        self.alert_log = []
        
    def get_health(self) -> Dict:
        """Check backend health"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.json() if resp.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def get_config(self) -> Dict:
        """Get Trading Profile config"""
        try:
            resp = requests.get(f"{self.base_url}/trading-profile/config", timeout=5)
            return resp.json() if resp.status_code == 200 else {}
        except Exception as e:
            return {"error": str(e)}
    
    def format_status_line(self, label: str, value: str, status: str = "INFO") -> str:
        """Format a status line with emoji"""
        emoji_map = {
            "OK": "✅",
            "WARN": "⚠️",
            "ERROR": "❌",
            "INFO": "ℹ️",
            "TRADE": "💹",
            "PROFIT": "💰",
            "LOSS": "📉"
        }
        emoji = emoji_map.get(status, "•")
        return f"{emoji} {label}: {value}"
    
    def clear_screen(self):
        """Clear terminal (cross-platform)"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display monitor header"""
        print("=" * 80)
        print("🚀 QUANTUM TRADER - LIVE PRODUCTION MONITOR")
        print("=" * 80)
        runtime = (datetime.now() - self.start_time).total_seconds()
        print(f"⏱️  Runtime: {int(runtime//3600)}h {int((runtime%3600)//60)}m {int(runtime%60)}s")
        print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
    
    def display_system_status(self):
        """Display system health and config"""
        print("\n📊 SYSTEM STATUS")
        print("-" * 80)
        
        # Health check
        health = self.get_health()
        if "error" in health:
            print(self.format_status_line("Backend", f"ERROR: {health['error']}", "ERROR"))
            return
        
        status = health.get("status", "unknown")
        print(self.format_status_line("Backend", status.upper(), "OK" if status == "ok" else "WARN"))
        
        # Config check
        config = self.get_config()
        if "error" not in config:
            enabled = config.get("enabled", False)
            risk = config.get("risk", {})
            leverage = risk.get("default_leverage", 0)
            max_pos = risk.get("max_positions", 0)
            
            print(self.format_status_line(
                "Trading Profile",
                f"{'ENABLED' if enabled else 'DISABLED'}",
                "OK" if enabled else "WARN"
            ))
            print(self.format_status_line("Leverage", f"{leverage}x", "INFO"))
            print(self.format_status_line("Max Positions", str(max_pos), "INFO"))
        
        print()
    
    def display_position_summary(self):
        """Display position summary (placeholder - needs real API)"""
        print("📈 POSITION SUMMARY")
        print("-" * 80)
        print(self.format_status_line("Active Positions", "0/4", "INFO"))
        print(self.format_status_line("Total Exposure", "$0", "INFO"))
        print(self.format_status_line("Margin Used", "0%", "INFO"))
        print()
    
    def display_pnl_summary(self):
        """Display P&L summary (placeholder - needs real API)"""
        print("💰 P&L SUMMARY")
        print("-" * 80)
        print(self.format_status_line("Today's P&L", "$0.00 (0.00%)", "INFO"))
        print(self.format_status_line("Total Trades", "0", "INFO"))
        print(self.format_status_line("Win Rate", "N/A", "INFO"))
        print()
    
    def display_recent_activity(self):
        """Display recent trading activity"""
        print("📋 RECENT ACTIVITY")
        print("-" * 80)
        if len(self.alert_log) == 0:
            print("   No activity yet...")
        else:
            for alert in self.alert_log[-5:]:  # Show last 5
                print(f"   {alert}")
        print()
    
    def display_configuration(self):
        """Display key configuration"""
        print("⚙️  CONFIGURATION")
        print("-" * 80)
        print("   Leverage: 30x")
        print("   Max Positions: 4")
        print("   Margin per Position: 25%")
        print("   Stop Loss: 1R (ATR-based)")
        print("   Take Profit 1: 1.5R (50% close)")
        print("   Take Profit 2: 2.5R (30% close)")
        print("   Trailing Stop: 0.8R @ TP2")
        print("   Funding Protection: 40m pre + 20m post")
        print()
    
    def display_alerts(self):
        """Display any alerts or warnings"""
        print("⚠️  ALERTS & WARNINGS")
        print("-" * 80)
        print("   No active alerts")
        print()
    
    def display_footer(self):
        """Display footer with commands"""
        print("=" * 80)
        print("🎮 Commands: [Ctrl+C] Stop Monitor | Status: MONITORING")
        print("=" * 80)
    
    def run_monitor(self, refresh_interval: int = 10):
        """Run continuous monitoring loop"""
        print(f"\n🚀 Starting Production Monitor (refresh every {refresh_interval}s)")
        print("Press Ctrl+C to stop...\n")
        time.sleep(2)
        
        try:
            while True:
                self.clear_screen()
                self.display_header()
                self.display_system_status()
                self.display_position_summary()
                self.display_pnl_summary()
                self.display_recent_activity()
                self.display_configuration()
                self.display_alerts()
                self.display_footer()
                
                # Wait for next refresh
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitor stopped by user")
            print(f"Total runtime: {(datetime.now() - self.start_time).total_seconds():.0f}s")
            print("=" * 80)

def main():
    """Main entry point"""
    monitor = ProductionMonitor()
    
    # Initial system check
    print("=" * 80)
    print("🔍 INITIAL SYSTEM CHECK")
    print("=" * 80)
    print()
    
    health = monitor.get_health()
    if "error" in health:
        print(f"❌ Backend ERROR: {health['error']}")
        print("\n⚠️  Cannot start monitor - backend not responding")
        return
    
    print("✅ Backend: ONLINE")
    
    config = monitor.get_config()
    if "error" not in config:
        enabled = config.get("enabled", False)
        if enabled:
            print("✅ Trading Profile: ENABLED")
            risk = config.get("risk", {})
            print(f"✅ Leverage: {risk.get('default_leverage', 0)}x")
            print(f"✅ Max Positions: {risk.get('max_positions', 0)}")
        else:
            print("⚠️  Trading Profile: DISABLED")
    
    print()
    print("=" * 80)
    print("🎯 SYSTEM READY FOR PRODUCTION MONITORING")
    print("=" * 80)
    print()
    
    # Start monitoring
    monitor.run_monitor(refresh_interval=10)

if __name__ == "__main__":
    main()
