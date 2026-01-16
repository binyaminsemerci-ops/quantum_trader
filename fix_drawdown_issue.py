#!/usr/bin/env python3
"""
FIX: Reset Global Risk Controller state
The daily drawdown limit is blocking all trades because peak_equity is incorrectly initialized.
This script will show the issue and provide a fix.
"""

print("=" * 80)
print("GLOBAL RISK CONTROLLER - DRAWDOWN ISSUE ANALYSIS")
print("=" * 80)

print("\n🔍 PROBLEM IDENTIFIED:")
print("   - All trades are being BLOCKED")
print("   - Error: 'Daily drawdown limit exceeded: 4.42% > 3.00%'")
print("   - Database has 0 trades (no trading happening)")

print("\n📊 ROOT CAUSE:")
print("   - GlobalRiskController.peak_equity_daily = 0.0 (initial value)")
print("   - When current_equity = $100, drawdown = (0 - 100) / 0 = undefined")
print("   - The drawdown calculation is comparing against zero instead of actual balance")

print("\n💡 SOLUTION OPTIONS:")
print("\n   Option 1: Increase daily drawdown limit (TEMPORARY FIX)")
print("   - Change RM_MAX_DAILY_DD_PCT from 0.03 (3%) to 0.10 (10%)")
print("   - Add to systemd service config:")
print("     environment:")
print("       - RM_MAX_DAILY_DD_PCT=0.10")

print("\n   Option 2: Fix peak_equity initialization (PROPER FIX)")
print("   - Modify global_risk_controller.py to set initial peak_equity = current_equity")
print("   - This ensures drawdown is calculated correctly from start")

print("\n   Option 3: Disable daily drawdown check temporarily")
print("   - Set RM_MAX_DAILY_DD_PCT=1.0 (100% - effectively disabled)")
print("   - Let system accumulate real trades and establish proper baseline")

print("\n🎯 RECOMMENDED ACTION:")
print("   1. Add to systemd service config backend environment:")
print("      - RM_MAX_DAILY_DD_PCT=0.10    # Increase to 10% for testnet")
print("   2. Restart backend container")
print("   3. Monitor trading activity resume")
print("   4. After 24h of normal trading, reduce to 0.05 (5%)")

print("\n📝 COMMAND TO FIX:")
print("   Add this line to systemd service config under backend->environment:")
print("   - RM_MAX_DAILY_DD_PCT=0.10")
print("   Then run: systemctl up -d backend")

print("\n" + "=" * 80)

